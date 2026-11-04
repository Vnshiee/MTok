#!/usr/bin/env python
"""
Mirage ablation runner (one condition per invocation).

Condition = (benchmark, model_mode, quota K, intervention). Writes one JSON line per question with telemetry, plus a
.meta.json describing the run. Resumable: questions already present in the output .jsonl are skipped.

Decoding protocol (batch size 1, KV cache):
  * The assistant turn is opened exactly as in src/test.py L57 ("...<|im_start|>assistant"). With
    --entry forced (default for target runs) the runner appends "\n<|latent_start|>", which is the token sequence the
    Stage-1/2 training text contains at this point (chat template "<|im_start|>assistant\n" followed by the helper-image
    placeholder rewritten to <|latent_start|> by src/utils.py L116-L123).
  * Latent phase: at each latent step the vector fed at the <|latent_pad|> position is chosen by LatentIntervention;
    under `none` it is the last-layer post-norm hidden state of the previous position, as in the fork
    (generation/utils.py L3417-L3419). A <|latent_pad|> id is appended to input_ids for every latent step
    (generation/utils.py L3387, L3398) so repetition-penalty bookkeeping matches the fork.
  * --quota auto reproduces the fork's exit rule: leave latent mode when the token chosen at a latent step is
    <|latent_end|> (L3380-L3384) or when more than 8 latents would be produced (L3393-L3396).
    --quota K (integer) feeds exactly K latents, ignores early <|latent_end|> predictions (logged), then appends
    <|latent_end|>. K=0 emits "<|latent_start|><|latent_end|>" with no latent vector.
  * Text phase: greedy (default) or test.py sampling (temperature 0.7, top-p 0.9); repetition penalty 1.05 as in the
    checkpoint's generation_config.json; stops at <|im_end|>/<|endoftext|> or --max_new_tokens.

Interventions (model_loader.LatentIntervention): none | zero | noise | donor | pause | mean, plus `suppress`
(bans <|latent_start|> in the text phase and uses --entry natural: answer with no latent segment at all).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRAGE_DIR = HERE.parent
sys.path.insert(0, str(MIRAGE_DIR))
import model_loader as ML  # noqa: E402

MC_INSTRUCTION = "\nAnswer with the option's letter from the given choices. The final answer MUST BE put in \\boxed{}."
VSTAR_ORIG_INSTRUCTION = "Answer with the option's letter from the given choices directly."


# ----------------------------------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, choices=["vstar", "mmvp", "blink", "vsp_planning"])
    p.add_argument("--quota", default="4", help="number of latent steps K (integer >= 0) or 'auto' (fork exit rule)")
    p.add_argument("--intervention", default="none", choices=list(ML.LatentIntervention.MODES) + ["suppress"])
    p.add_argument("--model_mode", default="target", choices=["target", "base"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", default=None, help="defaults: target=%s base=%s" % (ML.TARGET_MODEL_ID, ML.BASE_MODEL_ID))
    p.add_argument("--repo_dir", default=str(ML.DEFAULT_REPO_DIR))
    p.add_argument("--entry", default=None, choices=["forced", "natural"],
                   help="forced: prefill '\\n<|latent_start|>'; natural: let the model decide (test.py). "
                        "Default: forced for target runs, natural for base/suppress/auto")
    p.add_argument("--decoding", default="greedy", choices=["greedy", "test_py"])
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_latent_auto", type=int, default=ML.FORK_MAX_LATENT_LEN)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--blink_subtasks", default="Jigsaw", help="comma-separated BLINK configs (val split)")
    p.add_argument("--vsp_data_root", default=str(MIRAGE_DIR / "repo" / "data" / "vsp_spatial_planning"),
                   help="directory containing test_direct.jsonl and imgs_test/ (extract vsp_spatial_planning_test.tar.gz)")
    p.add_argument("--hf_cache", default=None)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--donor_dir", default=None,
                   help="dir with natural latents (*.pt) for donor/mean; default <output_dir>/latents/<benchmark>__target__K<K>__none")
    p.add_argument("--save_latents", action="store_true", help="always save fed latents (on by default for intervention=none)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default=None)
    p.add_argument("--no_fork_required", action="store_true", help="allow stock transformers (testing only)")
    return p.parse_args(argv)


def normalize_args(a):
    a.quota = "auto" if str(a.quota).lower() == "auto" else int(a.quota)
    if a.quota != "auto" and a.quota < 0:
        raise SystemExit("--quota must be >= 0 or 'auto'")
    if a.model_path is None:
        a.model_path = ML.TARGET_MODEL_ID if a.model_mode == "target" else ML.BASE_MODEL_ID
    if a.model_mode == "base":
        if a.intervention != "none" or a.quota not in (0, "auto"):
            raise SystemExit("base model has no latent tokens: use --intervention none --quota 0")
        a.quota = 0
    if a.entry is None:
        a.entry = "natural" if (a.model_mode == "base" or a.intervention == "suppress" or a.quota == "auto") else "forced"
    if a.intervention == "suppress" and a.entry != "natural":
        raise SystemExit("--intervention suppress requires --entry natural")
    if a.quota == 0 and a.intervention not in ("none", "suppress"):
        raise SystemExit("--quota 0 feeds no latent vectors, so --intervention %s would be identical to none" % a.intervention)
    if a.run_name is None:
        a.run_name = f"{a.benchmark}__{a.model_mode}__K{a.quota}__{a.intervention}__{a.entry}"
    return a


# ----------------------------------------------------------------------------------------------------------------------
# Benchmarks. Every item: question_id, category, text (with <image> placeholders), images (PIL), ground_truth, gt_key,
# letters (MC) or map_desc (VSP).
# ----------------------------------------------------------------------------------------------------------------------
def _mc_block(options):
    return "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))


def load_vsp(args):
    from PIL import Image
    root = Path(args.vsp_data_root)
    jf = root / "test_direct.jsonl"
    if not jf.exists():
        raise SystemExit(f"{jf} not found: extract repo/data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz into {root}")
    items = []
    for i, line in enumerate(open(jf)):
        r = json.loads(line)
        rel = r["image_input"].replace("./data/vsp_spatial_planning/", "", 1)
        items.append(dict(question_id=f"vsp_{i:04d}_L{len(r['map_desc'])}_{r['map_id']}",
                          category=f"level{len(r['map_desc'])}", text=r["text_input"],
                          image_paths=[str(root / rel)], ground_truth=r["map_desc"], gt_key=json.dumps(r["map_desc"]),
                          map_desc=r["map_desc"], task="vsp"))
    return items


def load_blink(args):
    from datasets import load_dataset
    items = []
    for sub in [s.strip() for s in args.blink_subtasks.split(",") if s.strip()]:
        ds = load_dataset("BLINK-Benchmark/BLINK", sub, split="val", cache_dir=args.hf_cache)
        for r in ds:
            imgs = [r[k] for k in ("image_1", "image_2", "image_3", "image_4") if r.get(k) is not None]
            letters = [chr(65 + i) for i in range(len(r["choices"]))]
            gt = r["answer"].strip("()").strip().upper()
            text = "<image>" * len(imgs) + "\n" + r["prompt"].rstrip() + MC_INSTRUCTION
            items.append(dict(question_id=r["idx"], category=sub, text=text, images=imgs, ground_truth=gt, gt_key=gt,
                              letters=letters, options=r["choices"], task="mc"))
    return items


def load_vstar(args):
    from huggingface_hub import hf_hub_download
    qf = hf_hub_download("craigwu/vstar_bench", "test_questions.jsonl", repo_type="dataset", cache_dir=args.hf_cache)
    items = []
    for line in open(qf):
        r = json.loads(line)
        body = r["text"].replace(VSTAR_ORIG_INSTRUCTION, "").rstrip()
        letters = re.findall(r"^\(([A-Z])\)", body, flags=re.M)
        items.append(dict(question_id=f"vstar_{r['question_id']}", category=r["category"],
                          text="<image>\n" + body + MC_INSTRUCTION, hf_image=r["image"],
                          ground_truth=r["label"].strip().upper(), gt_key=r["label"].strip().upper(),
                          letters=letters, task="mc"))
    return items


def load_mmvp(args):
    from huggingface_hub import hf_hub_download
    qf = hf_hub_download("MMVP/MMVP", "Questions.csv", repo_type="dataset", cache_dir=args.hf_cache)
    items = []
    with open(qf, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx = int(r["Index"])
            opts = [o.strip() for o in re.split(r"\([a-z]\)", r["Options"]) if o.strip()]
            gt = r["Correct Answer"].strip().strip("()").upper()
            text = "<image>\n" + r["Question"].strip() + "\n" + _mc_block(opts) + MC_INSTRUCTION
            items.append(dict(question_id=f"mmvp_{idx:03d}", category=f"pair{(idx - 1) // 2:03d}", text=text,
                              hf_image=f"MMVP Images/{idx}.jpg", ground_truth=gt, gt_key=gt,
                              letters=[chr(65 + i) for i in range(len(opts))], options=opts, task="mc", pair_id=(idx - 1) // 2))
    return items


LOADERS = {"vsp_planning": load_vsp, "blink": load_blink, "vstar": load_vstar, "mmvp": load_mmvp}


def materialize_images(item, args):
    from PIL import Image
    if "images" in item:
        return [im.convert("RGB") for im in item["images"]]
    if "image_paths" in item:
        return [Image.open(p).convert("RGB") for p in item["image_paths"]]
    from huggingface_hub import hf_hub_download
    repo = "craigwu/vstar_bench" if item["question_id"].startswith("vstar_") else "MMVP/MMVP"
    path = hf_hub_download(repo, item["hf_image"], repo_type="dataset", cache_dir=args.hf_cache)
    return [Image.open(path).convert("RGB")]


# ----------------------------------------------------------------------------------------------------------------------
# Answer parsing
# ----------------------------------------------------------------------------------------------------------------------
def extract_boxed(text):
    """Last \\boxed{...} with brace matching; returns 'None' when absent (mathruler.grader.extract_boxed_content contract)."""
    try:
        from mathruler.grader import extract_boxed_content
        return extract_boxed_content(text)
    except ImportError:
        start = text.rfind("\\boxed{")
        if start < 0:
            return "None"
        depth, i = 0, start + len("\\boxed{") - 1
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j]
        return "None"


def parse_mc(text, letters, options=None):
    """Returns (letter or None, method). Ordered from most to least explicit evidence."""
    L = "".join(letters)
    clean = re.sub(r"<\|[^|>]+\|>", " ", text)
    boxed = extract_boxed(clean)
    if boxed != "None":
        m = re.search(rf"\(?\b([{L}])\b\)?", boxed.strip())
        if m:
            return m.group(1), "boxed"
        if options:
            hits = [chr(65 + i) for i, o in enumerate(options) if o.strip().lower() == boxed.strip().lower()]
            if len(hits) == 1:
                return hits[0], "boxed_option_text"
    m = re.search(rf"<answer>\s*\(?([{L}])\b", clean)
    if m:
        return m.group(1), "answer_tag"
    m = re.search(rf"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([{L}])\)?(?:\b|$)", clean, flags=re.I)
    if m:
        return m.group(1).upper(), "answer_phrase"
    # "(A) ...", "A. ...", "A: ...", "A) ..." or a bare "A"; a leading "A " followed by words is the English article
    m = re.match(rf"\s*(?:\(([{L}])\)|([{L}])(?:[.:)]|\s*$))", clean)
    if m:
        return m.group(1) or m.group(2), "leading_letter"
    if options:
        hits = [chr(65 + i) for i, o in enumerate(options) if o.strip() and o.strip().lower() in clean.lower()]
        if len(hits) == 1:
            return hits[0], "option_text"
    cands = re.findall(rf"\b([{L}])\b", clean)
    if cands:
        return cands[-1], "fallback_last_letter"
    return None, "unparsed"


def score_item(item, gen_text):
    if item["task"] == "vsp":
        import task as mirage_task  # repo/src/task.py (simulate_vsp, parse_action_sequence)
        path_str = extract_boxed(gen_text)  # src/test.py L74 uses mathruler's extract_boxed_content
        res = mirage_task.simulate_vsp(item["map_desc"], path_str)
        return dict(parsed_option=path_str, parse_method="boxed" if path_str != "None" else "unparsed",
                    correct=bool(res["success"]), vsp_status=res["status"], vsp_invalid=bool(res["invalid"]))
    letter, method = parse_mc(gen_text, item["letters"], item.get("options"))
    return dict(parsed_option=letter, parse_method=method, correct=(letter == item["ground_truth"]))


# ----------------------------------------------------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------------------------------------------------
def apply_repetition_penalty(scores, token_ids, penalty):
    """Same math as transformers RepetitionPenaltyLogitsProcessor over all previous ids (prompt included)."""
    import torch
    if penalty == 1.0:
        return scores
    idx = torch.tensor(sorted(set(token_ids)), device=scores.device)
    s = scores[idx]
    scores[idx] = torch.where(s < 0, s * penalty, s / penalty)
    return scores


def choose_token(logits, token_ids, args, gen, banned=()):
    import torch
    scores = logits.float().clone()
    scores = apply_repetition_penalty(scores, token_ids, args.repetition_penalty)
    for b in banned:
        scores[b] = -float("inf")
    if args.decoding == "greedy":
        return int(torch.argmax(scores))
    probs = torch.softmax(scores / 0.7, dim=-1)                 # src/test.py L65: temperature=0.7
    sp, si = torch.sort(probs, descending=True)                 # src/test.py L66: top_p=0.9
    keep = (torch.cumsum(sp, 0) - sp) < 0.9
    sp = sp * keep
    return int(si[torch.multinomial((sp / sp.sum()).to(gen.device), 1, generator=gen).to(si.device)])


def cosine(a, b):
    import torch
    a, b = a.float().flatten(), b.float().flatten()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


def build_prompt(lm, item, args):
    """Chat template + repo place_input_image exactly as src/test.py L53-L57, plus optional forced latent entry."""
    import utils as mirage_utils  # repo/src/utils.py
    from qwen_vl_utils import process_vision_info
    images = materialize_images(item, args)
    pix = {k: getattr(args, k) for k in ("min_pixels", "max_pixels") if getattr(args, k) is not None}
    content = [{"type": "image", "image": im, **pix} for im in images] + [{"type": "text", "text": item["text"]}]
    conversations = [{"role": "user", "content": content}]
    text = lm.processor.apply_chat_template(conversations, tokenize=False)
    text = mirage_utils.place_input_image(text, sep_token=None)
    n_pad = text.count("<|image_pad|>")
    if n_pad != len(images):
        raise ValueError(f"{item['question_id']}: {n_pad} image placeholders for {len(images)} images")
    image_inputs, _ = process_vision_info(conversations)
    prompt = text + "<|im_start|>assistant"
    if args.entry == "forced":
        prompt += "\n<|latent_start|>"
    return prompt, image_inputs


@__import__("torch").no_grad()
def run_item(lm, item, args, intervention, gen, logits_hook=None, null_img=None):
    """Decode one question. `logits_hook(step, logits) -> logits` exists only for equivalence tests.
    `null_img`: image-token embeddings of a different question (H3 null for the latent-image cosine)."""
    import torch
    prompt, image_inputs = build_prompt(lm, item, args)
    inputs = lm.processor(text=[prompt], images=image_inputs, return_tensors="pt").to(lm.model.device)
    ids = inputs["input_ids"][0].tolist()
    attn = inputs["attention_mask"]
    n_prompt = len(ids)
    n_image_tokens = int((inputs["input_ids"] == lm.model.config.image_token_id).sum())
    lat = lm.latent_ids
    eos = set(lat.eos) if lat else {lm.tokenizer.convert_tokens_to_ids("<|im_end|>"), lm.tokenizer.convert_tokens_to_ids("<|endoftext|>")}

    t0 = time.time()
    out = lm.forward(**inputs, use_cache=True, cache_position=torch.arange(n_prompt, device=lm.model.device))
    cache = out.past_key_values
    logits = out.logits[0, -1]
    hidden = lm.last_hidden()[0, -1]
    img_emb = lm.visual_out.value.float() if (lm.visual_out.value is not None and n_image_tokens) else None
    img_centroid = img_emb.mean(0) if img_emb is not None else None
    pos = n_prompt

    def step(token_id=None, embed=None):
        nonlocal cache, logits, hidden, attn, pos
        attn = torch.cat([attn, attn.new_ones((1, 1))], dim=1)
        kw = dict(past_key_values=cache, attention_mask=attn, use_cache=True,
                  cache_position=torch.tensor([pos], device=lm.model.device))
        if embed is not None:
            kw["inputs_embeds"] = embed.view(1, 1, -1).to(lm.dtype)
        else:
            kw["input_ids"] = torch.tensor([[token_id]], device=lm.model.device)
        o = lm.forward(**kw)
        cache, logits, hidden = o.past_key_values, o.logits[0, -1], lm.last_hidden()[0, -1]
        pos += 1

    phase = "latent" if (args.entry == "forced" and lat is not None) else "text"
    segments, latent_log, fed_store = [], [], []
    seg = None
    exit_reason = None
    n_new = 0
    banned_text = (lat.start,) if (args.intervention == "suppress" and lat) else ()
    if phase == "latent":
        seg = dict(start_pos=pos - 1, steps=0, would_exit_steps=[], entry="forced")
    while n_new < args.max_new_tokens:
        if phase == "latent":
            k = seg["steps"]
            lg = logits_hook(len(ids), logits) if logits_hook else logits
            would = choose_token(lg, ids, args, gen)
            if args.quota == "auto":
                if would == lat.end:
                    exit_now, reason = True, "model_latent_end"
                elif k + 1 > args.max_latent_auto:
                    exit_now, reason = True, "max_latent_cap"
                else:
                    exit_now, reason = False, None
            else:
                if would == lat.end and k < args.quota:
                    seg["would_exit_steps"].append(k)
                exit_now, reason = (k >= args.quota), "quota_reached"
            if exit_now:
                seg["exit_reason"] = reason
                segments.append(seg)
                ids.append(lat.end)
                step(token_id=lat.end)
                n_new += 1
                phase, seg = "text", None
                continue
            natural = hidden
            fed, meta = intervention.apply(item["question_id"], k, natural)
            rec = dict(segment=len(segments), step=k, natural_norm=float(natural.float().norm()),
                       fed_norm=float(fed.float().norm()), cos_fed_natural=cosine(fed, natural),
                       would_be_token=would)
            if latent_log and latent_log[-1]["segment"] == len(segments):
                rec["cos_natural_prev_natural"] = cosine(natural, prev_natural)
                rec["cos_fed_prev_fed"] = cosine(fed, prev_fed)
            if img_emb is not None:
                sims = torch.nn.functional.cosine_similarity(img_emb, natural.float().view(1, -1), dim=-1)
                rec["max_cos_natural_image_token"] = float(sims.max())
                rec["argmax_image_token"] = int(sims.argmax())
                rec["cos_natural_image_centroid"] = cosine(natural, img_centroid)
            if null_img is not None:
                sims0 = torch.nn.functional.cosine_similarity(null_img.float(), natural.float().view(1, -1), dim=-1)
                rec["max_cos_natural_null_image_token"] = float(sims0.max())
            rec.update(meta)
            latent_log.append(rec)
            prev_natural, prev_fed = natural.clone(), fed.clone()
            fed_store.append(fed.detach().float().cpu())
            ids.append(lat.pad)
            step(embed=fed)
            seg["steps"] += 1
            n_new += 1
            continue
        # text phase
        lg = logits_hook(len(ids), logits) if logits_hook else logits
        tok = choose_token(lg, ids, args, gen, banned=banned_text)
        ids.append(tok)
        n_new += 1
        if tok in eos:
            exit_reason = "eos"
            break
        step(token_id=tok)
        if lat is not None and tok == lat.start and args.intervention != "suppress":
            phase = "latent"
            seg = dict(start_pos=pos - 1, steps=0, would_exit_steps=[], entry="model")
    else:
        exit_reason = "max_new_tokens"
    if seg is not None:
        seg["exit_reason"] = "max_new_tokens"
        segments.append(seg)

    gen_ids = ids[n_prompt:]
    gen_text = lm.tokenizer.decode(gen_ids, skip_special_tokens=False)
    answer_text = gen_text.split("<|latent_end|>")[-1] if lat is not None else gen_text
    res = score_item(item, answer_text)
    record = dict(
        question_id=item["question_id"], category=item["category"], ground_truth=item["ground_truth"],
        **res, generated_text=gen_text, answer_text=answer_text, generated_token_ids=gen_ids, n_prompt_tokens=n_prompt,
        n_image_tokens=n_image_tokens, n_generated_tokens=len(gen_ids), stop_reason=exit_reason,
        n_latent_segments=len(segments), latent_steps_total=sum(s["steps"] for s in segments),
        latent_segments=segments, latent_steps=latent_log,
        latent_norms=[r["fed_norm"] for r in latent_log],
        natural_latent_norms=[r["natural_norm"] for r in latent_log],
        step_cosine=[r.get("cos_natural_prev_natural") for r in latent_log],
        seconds=round(time.time() - t0, 3),
    )
    if "pair_id" in item:
        record["pair_id"] = item["pair_id"]
    did = intervention.donor_for(item["question_id"]) if intervention.mode == "donor" else None
    if did:
        record["donor_id"] = did
    return record, (torch.stack(fed_store) if fed_store else None)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def load_bank(d):
    import torch
    bank = {}
    if d and Path(d).is_dir():
        for f in Path(d).glob("*.pt"):
            obj = torch.load(f, map_location="cpu", weights_only=True)
            bank[obj["question_id"]] = obj["latents"]
    return bank


def main(argv=None):
    args = normalize_args(parse_args(argv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.run_name}.jsonl"
    done = set()
    if out_file.exists():
        for line in open(out_file):
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                pass

    lm = ML.load_model(args.model_path, args.model_mode, device=args.device, dtype=args.dtype,
                       attn_implementation=args.attn_implementation, min_pixels=args.min_pixels,
                       max_pixels=args.max_pixels, repo_dir=Path(args.repo_dir),
                       require_fork=not args.no_fork_required, cache_dir=args.hf_cache)
    import torch
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    items = LOADERS[args.benchmark](args)
    if args.limit:
        items = items[: args.limit]

    quota_tag = args.quota
    donor_dir = args.donor_dir or str(out_dir / "latents" / f"{args.benchmark}__target__K{quota_tag}__none__forced")
    bank, donor_map = {}, {}
    if args.intervention in ("donor", "mean"):
        bank = load_bank(donor_dir)
        if not bank:
            raise SystemExit(f"no natural latents in {donor_dir}; run --intervention none --quota {quota_tag} first")
        pool = [it for it in items if it["question_id"] in bank]
        donor_map = ML.build_donor_map(pool + [it for it in items if it["question_id"] not in bank])
        donor_map = {q: d if d in bank else next(iter(bank)) for q, d in donor_map.items()}
    intervention = ML.LatentIntervention(args.intervention if args.intervention != "suppress" else "none", lm,
                                         donor_bank=bank, donor_map=donor_map, noise_salt=args.seed)
    save_latents = (args.intervention == "none" and args.model_mode == "target") or args.save_latents
    lat_dir = out_dir / "latents" / args.run_name
    if save_latents:
        lat_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(args=vars(args), model_info=lm.info, n_items=len(items), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                torch=torch.__version__, cuda=torch.version.cuda, latent_ids=vars(lm.latent_ids) if lm.latent_ids else None,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                attn_implementation=getattr(lm.model.config, "_attn_implementation", None))
    with open(out_dir / f"{args.run_name}.meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)

    n_ok = n_done = 0
    prev_img = None   # image-token embeddings of the previous question: matched null for H3 telemetry
    with open(out_file, "a") as f:
        for i, item in enumerate(items):
            if item["question_id"] in done:
                continue
            rec, fed = run_item(lm, item, args, intervention, gen, null_img=prev_img)
            if rec["n_image_tokens"] and lm.visual_out.value is not None:
                prev_img = lm.visual_out.value.detach().clone()
            rec.update(benchmark=args.benchmark, model_mode=args.model_mode, quota=args.quota,
                       intervention=args.intervention, entry=args.entry, decoding=args.decoding)
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            if save_latents and fed is not None:
                torch.save({"question_id": item["question_id"], "latents": fed.half()},
                           lat_dir / f"{safe_name(item['question_id'])}.pt")
            n_done += 1
            n_ok += int(rec["correct"])
            if n_done % 10 == 0 or i == len(items) - 1:
                print(f"[{args.run_name}] {i + 1}/{len(items)} acc(new)={n_ok}/{n_done}", flush=True)
    with open(out_dir / f"{args.run_name}.COMPLETE", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"[{args.run_name}] done -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
