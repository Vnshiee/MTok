#!/usr/bin/env python
"""
E2: teacher-forced option-letter scoring for the bias-vs-discrimination study.

See PREREGISTRATION_E2.md (committed before any run) for the full design.

For every (condition, variant, item) we rebuild the *exact* generation input used
in E1 (same chat template, images, max_pixels, prompt mode and assistant prefill),
append the tokens of "<answer" ([27, 9217]) at the id level, run ONE forward pass,
and read the next-token logits for the option-letter tokens (">A", ">B", ">C", ">D").

Hard gates (abort on failure):
  G1 prompt fidelity   : reconstructed user text == logged E1 `input` (byte-exact)
  G2 tokenization      : "<answer>X</answer>" -> [27, 9217, id_X, 522, 9217, 29];
                         in-context prefill token count == E1 count
  G3 equivalence       : on V0 (real) inputs, TF argmax == E1 greedy letter on >=99%
                         of items (format-locked conditions only); median candidate
                         probability mass >= 0.95
  G4 integrity         : exact N per (condition, variant, task), no duplicate keys

Variants:
  V0 real | V1 blind (grey canvases, same count/size) | V2 swap | V3 relabel |
  V4 swap+relabel (V2-V4 on the four 2-way splits only) | CF content-free probes
"""
import argparse
import copy
import glob
import hashlib
import importlib
import json
import os
import random
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1_DIR = os.path.join(PROJECT_DIR, "eval_outputs", "e1_matrix")
MODELS_DIR = os.path.join(PROJECT_DIR, "mull", "models")
LMMS_DIR = os.path.join(PROJECT_DIR, "mull", "lmms-eval")
for p in (MODELS_DIR, LMMS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ----------------------------------------------------------------------------- constants
MAX_PIXELS = 4194304          # E1 model_args max_pixels
MIN_PIXELS = 256 * 28 * 28    # wrapper default (not overridden in E1)
ANSWER_PREFIX_IDS = [27, 9217]  # "<", "answer"
SEED = 20261028

TASKS = {
    # task name     : (HF config,              letters,  two_way)
    "blink_sprel":    ("Spatial_Relation",     "AB",   True),
    "blink_reldepth": ("Relative_Depth",       "AB",   True),
    "blink_jigsaw":   ("Jigsaw",               "AB",   True),
    "blink_mv":       ("Multi-view_Reasoning", "AB",   True),
    "blink_iqtest":   ("IQ_Test",              "ABCD", False),
}
EXPECTED_N = {"blink_sprel": 143, "blink_reldepth": 124, "blink_jigsaw": 150, "blink_mv": 133, "blink_iqtest": 150}

MODELS = {
    "mull": "array/Qwen2.5-VL-Mull",
    "grpo": "array/Qwen2.5-VL-MullGRPO",
    "sat":  "array/Qwen2.5-VL-SAT",
}


def latents(k):
    return "<think>" + "<|latent_pad|>" * k + "</think>\n"


# cond_id: model, prompt_mode, prefill text, expected in-context prefill tokens, E1 row dir, G3 applies, tier
CONDITIONS = {
    "T1":  dict(model="mull", prompt="latents", prefill=latents(20), plen=26, e1="row1_mull_ref", gate=True, tier="A"),
    "T3":  dict(model="mull", prompt="latents", prefill="<think></think>\n", plen=5, e1="row3_mull_empty_think", gate=True, tier="A"),
    "T8":  dict(model="sat",  prompt="sft",     prefill="", plen=0, e1="row8_sat_sft_empty", gate=True, tier="A"),
    "T9":  dict(model="sat",  prompt="latents", prefill="", plen=0, e1="row9_sat_latents_empty", gate=True, tier="B"),
    "T10": dict(model="grpo", prompt="latents", prefill=latents(20), plen=26, e1="row10_mull_grpo_k20", gate=True, tier="B"),
    "T4":  dict(model="mull", prompt="latents", prefill="<think>" + " ." * 20 + "</think>\n", plen=26, e1="row4_mull_dot_pause", gate=True, tier="B"),
    "T5":  dict(model="mull", prompt="latents", prefill="<think>" + "<|latent_end|>" * 20 + "</think>\n", plen=26, e1="row5_mull_latent_end", gate=True, tier="B"),
    "T7":  dict(model="mull", prompt="sft",     prefill=latents(20), plen=26, e1="row7_mull_sft_latents", gate=True, tier="B"),
    "T11": dict(model="mull", prompt="latents", prefill=latents(1), plen=7, e1="row11_mull_k1", gate=True, tier="C"),
    "T12": dict(model="mull", prompt="latents", prefill=latents(5), plen=11, e1="row12_mull_k5", gate=True, tier="C"),
    "T13": dict(model="mull", prompt="latents", prefill=latents(10), plen=16, e1="row13_mull_k10", gate=True, tier="C"),
    "T14": dict(model="mull", prompt="latents", prefill=latents(40), plen=46, e1="row14_mull_k40", gate=True, tier="C"),
    # NEW conditions: empty prefill, answer forced immediately (NOT comparable to free-running rows 2/6)
    "T2F": dict(model="mull", prompt="latents", prefill="", plen=0, e1="row2_mull_empty", gate=False, tier="C"),
    "T6F": dict(model="mull", prompt="sft",     prefill="", plen=0, e1="row6_mull_sft_empty", gate=False, tier="C"),
}

VARIANTS_ALL = ["V0", "V1", "V2", "V3", "V4", "CF"]
CF_FILLERS = ["N/A", "", "[MASK]"]
CF_COLORS = {"grey": (128, 128, 128), "black": (0, 0, 0), "white": (255, 255, 255)}
OPT_RE = re.compile(r"^\(([A-D])\) (.*)$")
SCAFFOLD_RE = re.compile(r"Select from the following (choices|options)\.")


class GateError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- prompt helpers
def doc_to_text(doc, prompt_mode):
    """Calls the *actual* lmms-eval BLINK doc_to_text so templates cannot drift."""
    os.environ["BLINK_PROMPT_MODE"] = prompt_mode
    mod = importlib.import_module("lmms_eval.tasks.blink_sprel.utils")
    return mod.blink_doc_to_text(doc)


def doc_images(doc):
    return [doc[f"image_{i}"].convert("RGB") for i in range(1, 5) if doc[f"image_{i}"] is not None]


def parse_options(prompt):
    """Return list of (line_index, letter, content) for option lines, in order."""
    lines = prompt.split("\n")
    opts = []
    for i, ln in enumerate(lines):
        m = OPT_RE.match(ln)
        if m:
            opts.append((i, m.group(1), m.group(2)))
    return lines, opts


def build_two_way_variant(prompt, variant):
    """V2 swap / V3 relabel / V4 swap+relabel. Returns (new_prompt, letter_of_content{1,2})."""
    lines, opts = parse_options(prompt)
    assert len(opts) == 2 and opts[0][1] == "A" and opts[1][1] == "B", f"unexpected options: {opts}"
    (i1, _, c1), (i2, _, c2) = opts
    if variant == "V0":
        layout = [("A", c1), ("B", c2)]; loc = {1: "A", 2: "B"}
    elif variant == "V2":
        layout = [("A", c2), ("B", c1)]; loc = {1: "B", 2: "A"}
    elif variant == "V3":
        layout = [("B", c1), ("A", c2)]; loc = {1: "B", 2: "A"}
    elif variant == "V4":
        layout = [("B", c2), ("A", c1)]; loc = {1: "A", 2: "B"}
    else:
        raise ValueError(variant)
    new = list(lines)
    new[i1] = f"({layout[0][0]}) {layout[0][1]}"
    new[i2] = f"({layout[1][0]}) {layout[1][1]}"
    return "\n".join(new), loc


def build_content_free_prompt(prompt, filler):
    """Replace question stem and every option content with `filler`, keep task scaffold."""
    m = SCAFFOLD_RE.search(prompt)
    assert m, "scaffold sentence not found"
    stem = prompt[:m.start()]
    stripped = stem.rstrip()
    trail = stem[len(stripped):]
    rest = prompt[m.start():]
    lines = rest.split("\n")
    for i, ln in enumerate(lines):
        om = OPT_RE.match(ln)
        if om:
            lines[i] = f"({om.group(1)}) {filler}".rstrip()
    return filler + trail + "\n".join(lines)


# ----------------------------------------------------------------------------- model
class Scorer:
    def __init__(self, model_key):
        self.model_key = model_key
        pretrained = MODELS[model_key]
        Q = importlib.import_module("mmlatentdiscrete_qwen_vl").Qwen2_5_VLForConditionalGeneration
        self.model = Q.from_pretrained(pretrained, torch_dtype="bfloat16", device_map="auto",
                                       attn_implementation="flash_attention_2").eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
        # mirror Qwen2_5_VL_MMLatent_simple.__init__ token handling exactly
        tk = self.processor.tokenizer
        n_added = 0
        for t in ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>", "<|latent_image|>"]:
            n_added += tk.add_tokens(t, special_tokens=True)
        cfg = self.model.config
        cfg.latent_token_id = int(tk("<|latent_pad|>")["input_ids"][0])
        cfg.latent_start_id = int(tk("<|latent_start|>")["input_ids"][0])
        cfg.latent_end_id = int(tk("<|latent_end|>")["input_ids"][0])
        cfg.imagelatent_token_id = int(tk("<|latent_image|>")["input_ids"][0])
        if n_added > 0:
            self.model.resize_token_embeddings(len(tk))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.letter_ids = {}
        self.id_to_letter = {}
        # G2: tokenization of the full answer string
        for L in "ABCD":
            ids = self.tokenizer.encode(f"<answer>{L}</answer>", add_special_tokens=False)
            if len(ids) != 6 or ids[:2] != ANSWER_PREFIX_IDS or ids[3:] != [522, 9217, 29]:
                raise GateError(f"G2 failed: '<answer>{L}</answer>' -> {ids}")
            self.letter_ids[L] = ids[2]
            self.id_to_letter[ids[2]] = L
        assert len(set(self.letter_ids.values())) == 4
        self.assistant_hdr = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.n_added = n_added

    @torch.no_grad()
    def score(self, images, user_text, prefill, letters, expected_plen):
        messages = [
            {"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if text.endswith("<|im_end|>\n"):
            text = text[:-len("<|im_end|>\n")]
        elif text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
        if prefill:
            assert text.endswith(prefill), f"prefill tail mismatch: {text[-80:]!r}"
        else:
            assert text.endswith("<|im_start|>assistant\n"), f"empty-prefill tail mismatch: {text[-40:]!r}"
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        # G2 (in context): count tokens after the last assistant header
        h = len(self.assistant_hdr)
        pos = max(i for i in range(len(ids) - h + 1) if ids[i:i + h] == self.assistant_hdr)
        plen = len(ids) - (pos + h)
        if plen != expected_plen:
            raise GateError(f"G2 failed: in-context prefill length {plen} != expected {expected_plen}")
        # append "<answer" at id level (avoids BPE merges with the preceding "\n")
        extra = torch.tensor([ANSWER_PREFIX_IDS], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(extra)], dim=1)
        inputs = inputs.to("cuda")
        self.model.rope_deltas = None
        out = self.model(**inputs, use_cache=False, return_dict=True)
        logits = out.logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        cand = {L: float(logp[self.letter_ids[L]]) for L in letters}
        cand_logits = {L: float(logits[self.letter_ids[L]]) for L in letters}
        mass = float(torch.logsumexp(torch.tensor(list(cand.values())), 0).exp())
        top1 = int(torch.argmax(logits))
        top1_letter = self.id_to_letter.get(top1, None)
        cand_argmax = max(cand, key=cand.get)
        tf_pred = top1_letter if (top1_letter is not None and top1_letter in letters) else cand_argmax
        del out, logits, logp, inputs
        return dict(logp=cand, logit=cand_logits, cand_mass=mass, top1_id=top1,
                    top1_str=self.tokenizer.decode([top1]), tf_pred=tf_pred, n_tokens=len(ids) + 2, prefill_len=plen,
                    text_sha1=hashlib.sha1(text.encode()).hexdigest()[:12])


# ----------------------------------------------------------------------------- E1 references
def load_e1(row_dir, task):
    files = glob.glob(os.path.join(E1_DIR, row_dir, "array__*", f"*samples_{task}.jsonl"))
    if len(files) != 1:
        raise GateError(f"G1: expected 1 E1 sample file for {row_dir}/{task}, found {len(files)}")
    ref = {}
    for ln in open(files[0]):
        r = json.loads(ln)
        if r["doc_id"] in ref:
            raise GateError(f"G1: duplicate doc_id {r['doc_id']} in {files[0]}")
        ref[r["doc_id"]] = dict(input=r["input"], pred=r["blink_score_overall"]["pred"], target=r["target"])
    if len(ref) != EXPECTED_N[task]:
        raise GateError(f"G1: {row_dir}/{task} has {len(ref)} docs, expected {EXPECTED_N[task]}")
    return ref


# ----------------------------------------------------------------------------- IO
class Writer:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.done = set()
        if os.path.exists(path):
            for ln in open(path):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue  # truncated last line from an interrupted run; it is rewritten
                self.done.add(r["key"])
            # rewrite file without any truncated line
            good = [ln for ln in open(path) if ln.strip().endswith("}")]
            with open(path, "w") as f:
                f.writelines(good)
        self.f = open(path, "a")

    def write(self, rec):
        if rec["key"] in self.done:
            raise GateError(f"G4: duplicate key {rec['key']}")
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.done.add(rec["key"])


def check_g3(path, cond):
    recs = [json.loads(l) for l in open(path)]
    n = len(recs)
    agree = sum(r["tf_pred"] == r["e1_pred"] for r in recs)
    masses = sorted(r["cand_mass"] for r in recs)
    med = masses[n // 2]
    mism = [(r["task"], r["doc_id"], r["tf_pred"], r["e1_pred"], round(max(r["logp"].values()) - sorted(r["logp"].values())[-2], 4)) for r in recs if r["tf_pred"] != r["e1_pred"]]
    rep = dict(cond=cond, n=n, agree=agree, agree_rate=agree / n, median_mass=med, mismatches=mism)
    return rep


# ----------------------------------------------------------------------------- main loop
def run_condition(cid, scorer, out_root, variants, limit, task_filter, log):
    c = CONDITIONS[cid]
    cdir = os.path.join(out_root, cid)
    os.makedirs(cdir, exist_ok=True)
    json.dump(dict(cond=cid, **c, model_path=MODELS[c["model"]], max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
                   letter_ids=scorer.letter_ids, tokens_added=scorer.n_added, seed=SEED),
              open(os.path.join(cdir, "config.json"), "w"), indent=2)
    writers = {v: Writer(os.path.join(cdir, f"{v}.jsonl")) for v in variants}
    rng = random.Random(SEED)
    tasks = [t for t in TASKS if (not task_filter or t in task_filter)]

    # ---- pass 1: V0 for all tasks, then gate G3 before anything else
    order = (["V0"] if "V0" in variants else []) + [v for v in variants if v != "V0"]
    for phase_variants in ([["V0"]] if "V0" in variants else []) + [[v for v in order if v != "V0"]]:
        if not phase_variants:
            continue
        for task in tasks:
            cfg_name, letters, two_way = TASKS[task]
            ds = load_dataset("BLINK-Benchmark/BLINK", cfg_name, split="val")
            assert len(ds) == EXPECTED_N[task]
            ref = load_e1(c["e1"], task)
            n_docs = len(ds) if limit is None else min(limit, len(ds))
            t0 = time.time()
            for doc_id in range(n_docs):
                doc = ds[doc_id]
                gold = doc["answer"].replace("(", "").replace(")", "")
                base_text = doc_to_text(doc, c["prompt"])
                # G1: byte-exact prompt fidelity against the logged E1 input
                if base_text != ref[doc_id]["input"]:
                    raise GateError(f"G1 failed: {cid}/{task}/doc {doc_id} prompt differs from E1 log")
                if gold != ref[doc_id]["target"]:
                    raise GateError(f"G1 failed: gold mismatch {cid}/{task}/doc {doc_id}")
                imgs = None
                for v in phase_variants:
                    if v in ("V2", "V3", "V4", "CF") and v != "CF" and not two_way:
                        continue
                    if v == "CF":
                        continue  # CF handled per task below
                    key = f"{task}|{doc_id}"
                    if key in writers[v].done:
                        continue
                    if imgs is None:
                        imgs = doc_images(doc)
                    rec = dict(key=key, cond=cid, variant=v, task=task, doc_id=doc_id, gold=gold, letters=letters)
                    if v == "V0":
                        r = scorer.score(imgs, base_text, c["prefill"], letters, c["plen"])
                        rec["e1_pred"] = ref[doc_id]["pred"]
                    elif v == "V1":
                        grey = [Image.new("RGB", im.size, CF_COLORS["grey"]) for im in imgs]
                        r = scorer.score(grey, base_text, c["prefill"], letters, c["plen"])
                    else:
                        new_prompt, loc = build_two_way_variant(doc["prompt"], v)
                        d2 = copy.copy(doc); d2 = dict(d2); d2["prompt"] = new_prompt
                        vtext = doc_to_text(d2, c["prompt"])
                        r = scorer.score(imgs, vtext, c["prefill"], letters, c["plen"])
                        gold_content = 1 if gold == "A" else 2
                        rec["letter_of_content"] = loc
                        rec["gold_content"] = gold_content
                        rec["variant_gold"] = loc[gold_content]
                        rec["prompt_variant"] = new_prompt
                    rec.update(r)
                    writers[v].write(rec)
            # ---- CF probes for this task
            if "CF" in phase_variants:
                pick = random.Random(f"{SEED}-{task}").sample(range(len(ds)), 3)
                for item in pick:
                    doc = ds[item]
                    sizes = [im.size for im in doc_images(doc)]
                    for filler in CF_FILLERS:
                        cf_prompt = build_content_free_prompt(doc["prompt"], filler)
                        d2 = dict(doc); d2["prompt"] = cf_prompt
                        cf_text = doc_to_text(d2, c["prompt"])
                        for cname, col in CF_COLORS.items():
                            key = f"{task}|{item}|{filler or '<empty>'}|{cname}"
                            if key in writers["CF"].done:
                                continue
                            canv = [Image.new("RGB", s, col) for s in sizes]
                            r = scorer.score(canv, cf_text, c["prefill"], letters, c["plen"])
                            rec = dict(key=key, cond=cid, variant="CF", task=task, doc_id=item, filler=filler,
                                       color=cname, letters=letters, prompt_variant=cf_prompt, **r)
                            writers["CF"].write(rec)
            log(f"[{cid}] {task} {phase_variants} done in {time.time()-t0:.0f}s")

        if phase_variants == ["V0"]:
            rep = check_g3(writers["V0"].path, cid)
            json.dump(rep, open(os.path.join(cdir, "G3_report.json"), "w"), indent=2)
            log(f"[{cid}] G3: agree {rep['agree']}/{rep['n']} ({rep['agree_rate']*100:.2f}%), median mass {rep['median_mass']:.4f}")
            min_agree = 0.95 if limit is None else 0.85
            min_mass = 0.95 if limit is None else 0.90
            if c["gate"] and (rep["agree_rate"] < min_agree or rep["median_mass"] < min_mass):
                raise GateError(f"G3 failed for {cid}: {rep['agree_rate']:.4f} agree, median mass {rep['median_mass']:.4f}")

    # ---- G4 counts (full runs only)
    if limit is None and not task_filter:
        for v in variants:
            recs = [json.loads(l) for l in open(writers[v].path)]
            keys = [r["key"] for r in recs]
            if len(keys) != len(set(keys)):
                raise GateError(f"G4: duplicates in {cid}/{v}")
            for task in TASKS:
                n = sum(r["task"] == task for r in recs)
                two = TASKS[task][2]
                exp = (EXPECTED_N[task] if v in ("V0", "V1") else (EXPECTED_N[task] if two else 0)) if v != "CF" else 27
                if n != exp:
                    raise GateError(f"G4: {cid}/{v}/{task} has {n}, expected {exp}")
        log(f"[{cid}] G4 passed for {variants}")
        open(os.path.join(cdir, "COMPLETE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=VARIANTS_ALL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--out", default=os.path.join(PROJECT_DIR, "e2_bias", "outputs"))
    a = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if a.limit is not None and os.path.abspath(a.out) == os.path.abspath(os.path.join(PROJECT_DIR, "e2_bias", "outputs")):
        raise SystemExit("Refusing to write a --limit (smoke) run into the main results directory; pass --out <scratch dir>.")
    logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_run.log" if a.limit is None else "e2_smoke.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    # group conditions by checkpoint so each model loads once, preserving requested order
    by_model = {}
    for cid in a.conds:
        by_model.setdefault(CONDITIONS[cid]["model"], []).append(cid)
    for mk, cids in by_model.items():
        log(f"Loading {MODELS[mk]} for {cids}")
        scorer = Scorer(mk)
        log(f"letter ids {scorer.letter_ids}, tokens added {scorer.n_added}")
        for cid in cids:
            if a.limit is None and os.path.exists(os.path.join(a.out, cid, "COMPLETE")):
                log(f"[{cid}] already COMPLETE, skipping"); continue
            run_condition(cid, scorer, a.out, a.variants, a.limit, a.tasks, log)
        del scorer
        torch.cuda.empty_cache()
    log("ALL DONE")


if __name__ == "__main__":
    main()

