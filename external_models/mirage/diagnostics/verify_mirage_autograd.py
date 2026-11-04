#!/usr/bin/env python
"""
CPU diagnostic for Mirage (repo UMass-Embodied-AGI/Mirage @ 53f26de, checkpoint Miiche/vsp_spatial_planning_direct_sft).

Runs the repository's own code, not re-implementations:
  * the transformers fork in repo/transformers/src (model forward, Stage-2 multi-pass, generate/_sample latent loop),
  * collate_fn_stage1 / collate_fn_stage2, executed from the source of repo/src/main.py,
  * CustomTrainerStage1 / CustomTrainerStage2 from repo/src/trainer.py,
  * the VSP preprocessing in repo/src/task.py,
on a tiny randomly initialised Qwen2.5-VL that keeps the released config fields, tokenizer, chat template and image
processor. Each check prints PASS/FAIL; the process exits non-zero if any check fails.

Usage (from external_models/mirage):
    CUDA_VISIBLE_DEVICES="" python diagnostics/verify_mirage_autograd.py [--meta_dir DIR]
--meta_dir: a directory with the released checkpoint's config/tokenizer/processor files. If omitted they are fetched
from the Hugging Face Hub (small files only; no weights).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import textwrap
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).resolve().parent
MIRAGE = HERE.parent
sys.path.insert(0, str(MIRAGE))
sys.path.insert(0, str(MIRAGE / "ablation_study"))
import model_loader as ML  # noqa: E402

META_FILES = ["config.json", "generation_config.json", "tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json",
              "added_tokens.json", "special_tokens_map.json", "preprocessor_config.json", "chat_template.json"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def get_meta(meta_dir):
    if meta_dir:
        return Path(meta_dir)
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(ML.TARGET_MODEL_ID, f) for f in META_FILES]
    return Path(paths[0]).parent


def exec_function_from(path: Path, names, namespace):
    """Execute top-level function definitions verbatim from a source file into `namespace`."""
    src = path.read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    missing = [n for n in names if n not in namespace]
    assert not missing, missing


def tiny_model(meta: Path, stage: str, latent_size: int, seed=0):
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    cfg = json.loads((meta / "config.json").read_text())
    cfg.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
               max_window_layers=2, rope_scaling={"mrope_section": [2, 3, 3], "rope_type": "default", "type": "default"})
    cfg["vision_config"].update(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64,
                                fullatt_block_indexes=[1], torch_dtype="float32")
    cfg["torch_dtype"] = "float32"
    config = Qwen2_5_VLConfig.from_dict(cfg)
    # src/main.py L50-L52
    config.compress_strategy = "average"
    config.latent_size = latent_size
    config.stage = stage
    torch.manual_seed(seed)
    model = Qwen2_5_VLForConditionalGeneration(config)
    # src/main.py L68-L69: the vision tower is frozen in both stages
    for param in model.visual.parameters():
        param.requires_grad = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default=None)
    ap.add_argument("--repo_dir", default=str(ML.DEFAULT_REPO_DIR))
    ap.add_argument("--latent_size", type=int, default=4)
    a = ap.parse_args()

    transformers, is_fork = ML.activate_mirage_fork(Path(a.repo_dir))
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList
    from qwen_vl_utils import process_vision_info
    import utils as mutils          # repo/src/utils.py
    import task as mtask            # repo/src/task.py
    import trainer as mtrainer      # repo/src/trainer.py
    from trl import SFTConfig

    repo = Path(a.repo_dir)
    K = a.latent_size
    torch.set_grad_enabled(True)
    print(f"transformers {transformers.__version__} from {transformers.__file__} (fork={is_fork}); torch {torch.__version__}")
    check("fork active (transformers 4.50.0.dev0 from repo/transformers/src)", is_fork and transformers.__version__ == "4.50.0.dev0")

    meta = get_meta(a.meta_dir)
    processor = AutoProcessor.from_pretrained(meta)
    tok = processor.tokenizer
    # src/main.py L36-L38
    for t in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"):
        tok.add_tokens(t, special_tokens=True)
    released_cfg = json.loads((meta / "config.json").read_text())

    # ------------------------------------------------------------------ T1 token ids
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>")}
    gen_src = (repo / "transformers/src/transformers/generation/utils.py").read_text().splitlines()
    hard = {m.group(1): int(m.group(2)) for line in gen_src[3369:3372]
            for m in [re.match(r"\s*(latent_\w+_idx) = (\d+)", line)] if m}
    check("T1a tokenizer ids == released config ids",
          (ids["<|latent_pad|>"], ids["<|latent_start|>"], ids["<|latent_end|>"]) ==
          (released_cfg["latent_token_id"], released_cfg["latent_start_id"], released_cfg["latent_end_id"]), str(ids))
    check("T1b tokenizer ids == ids hard-coded in fork generation/utils.py L3370-L3372",
          hard == {"latent_start_idx": ids["<|latent_start|>"], "latent_end_idx": ids["<|latent_end|>"],
                   "latent_pad_idx": ids["<|latent_pad|>"]}, str(hard))
    check("T1c released vocab_size covers latent ids (151668 = 151665 + 3)", released_cfg["vocab_size"] == ids["<|latent_end|>"] + 1)
    max_line = gen_src[3294]
    check("T1d fork MAX_LATENT_LEN = 8 (generation/utils.py L3295)", max_line.strip() == "MAX_LATENT_LEN = 8", max_line.strip())

    # ------------------------------------------------------------------ synthetic VSP-format sample, repo preprocessing
    tmp = Path(tempfile.mkdtemp(prefix="mirage_diag_"))
    rng = torch.Generator().manual_seed(0)
    def rand_img(path, w, h):
        arr = (torch.rand(h, w, 3, generator=rng) * 255).to(torch.uint8).numpy()
        Image.fromarray(arr).save(path)
        return str(path)
    sample = {"text_input": "Here is the map: <image>\nPlease provide your action plan.\n The final answer MUST BE put in \\boxed{}.",
              "text_output": "\\boxed{DOWN}", "image_input": rand_img(tmp / "in.png", 112, 112),
              "image_output": rand_img(tmp / "helper.png", 168, 112), "map_desc": [[0, 1], [0, 2]]}
    conv = mtask.task_preporcess_config["vsp-spatial-planning"](sample)

    ns = dict(vars(mutils))
    ns.update(processor=processor, process_vision_info=process_vision_info, args=types.SimpleNamespace(latent_size=K))
    exec_function_from(repo / "src/main.py", ["collate_fn_stage1", "collate_fn_stage2"], ns)

    # ------------------------------------------------------------------ T2 Stage-1 batch
    b1 = ns["collate_fn_stage1"]([conv])
    seq = b1["input_ids"][0]
    LS, LE, PAD = ids["<|latent_start|>"], ids["<|latent_end|>"], ids["<|latent_pad|>"]
    pads = (seq == PAD).nonzero().flatten().tolist()
    ls_pos = (seq == LS).nonzero().flatten().tolist()
    le_pos = (seq == LE).nonzero().flatten().tolist()
    check("T2a Stage-1 sequence: one <|latent_start|>, K pads, one <|latent_end|>, contiguous",
          len(ls_pos) == 1 and len(le_pos) == 1 and pads == list(range(ls_pos[0] + 1, ls_pos[0] + 1 + K)) and le_pos[0] == ls_pos[0] + K + 1,
          f"start={ls_pos} pads={pads} end={le_pos}")
    labels = b1["labels"][0]
    asst = mutils.find_subsequence(seq, tok("<|im_start|>assistant", return_tensors="pt")["input_ids"][0])
    check("T2b labels: prompt masked, pads masked, <|latent_start|> and <|latent_end|> supervised",
          bool((labels[: asst + 2] == -100).all()) and bool((labels[pads] == -100).all())
          and int(labels[ls_pos[0]]) == LS and int(labels[le_pos[0]]) == LE)
    check("T2c image_out_mask marks exactly the pad positions", b1["image_out_mask"][0].nonzero().flatten().tolist() == pads)
    n_user_img_tokens = int((seq == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    check("T2d user image tokens match user pixel grid; helper image kept in pixel_values_latent",
          n_user_img_tokens == int(b1["image_grid_thw"].prod(-1).sum()) // 4 and "pixel_values_latent" in b1,
          f"user tokens={n_user_img_tokens}")

    # ------------------------------------------------------------------ T3 Stage-1 forward + loss (CustomTrainerStage1)
    m1 = tiny_model(meta, "stage1", K)
    m1.config.latent_token_id, m1.config.latent_start_id, m1.config.latent_end_id = PAD, LS, LE  # src/main.py L64-L66
    m1.train()
    cfg = SFTConfig(output_dir=str(tmp / "out"), per_device_train_batch_size=1, remove_unused_columns=False,
                    dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True}, report_to=[], use_cpu=True,
                    bf16=False, gradient_checkpointing=False)
    tr1 = mtrainer.CustomTrainerStage1(model=m1, args=cfg, train_dataset=[conv], data_collator=ns["collate_fn_stage1"],
                                       tokenizer=processor.tokenizer)
    loss1, out1 = tr1.compute_loss(m1, dict(b1), return_outputs=True)
    with torch.no_grad():
        feats = m1.visual(b1["pixel_values_latent"].type(m1.visual.dtype), grid_thw=b1["image_grid_thw_latent"])
        feats = feats.view(1, -1, feats.shape[-1])
        res = feats.shape[1] % K
        if res:
            feats = feats[:, :-res]
        target = torch.stack([c.mean(1) for c in torch.split(feats, feats.shape[1] // K, dim=1)], 1)[0]
    check("T3a pad inputs = average of K contiguous raster-order chunks of helper-image features (modeling L2103-L2122)",
          torch.allclose(out1.inputs_embeds[0, pads].detach(), target, atol=1e-6),
          f"helper tokens={feats.shape[1] + res}, chunk={feats.shape[1] // K}, dropped remainder={res}")
    h = out1.hidden_states[0]
    sim = torch.nn.functional.cosine_similarity(out1.inputs_embeds[0, pads], h[[p - 1 for p in pads]]).mean()
    ce = out1.loss
    check("T3b loss1 == 0.1*CE + (1 - mean cos(h[p-1], e[p])) over pads (trainer.py L13-L25)",
          torch.allclose(loss1, 0.1 * ce + (1 - sim), atol=1e-6), f"loss={loss1.item():.6f} ce={ce.item():.4f} cos={sim.item():.4f}")
    loss1.backward()
    vis_grads = [p.grad for p in m1.visual.parameters()]
    llm_grad = sum(float(p.grad.abs().sum()) for p in m1.model.layers.parameters() if p.grad is not None)
    check("T3c gradient reaches LLM layers; frozen vision tower receives none", llm_grad > 0 and all(g is None for g in vis_grads))
    pad_row_grad = m1.model.embed_tokens.weight.grad[PAD].abs().sum().item()
    check("T3d <|latent_pad|> embedding row gets zero gradient in Stage 1 (every pad input is overwritten by masked_scatter)",
          pad_row_grad == 0.0, f"|grad|={pad_row_grad}")
    tgt_has_path = out1.inputs_embeds.grad_fn is not None
    check("T3e sim-loss targets are not detached in code but carry no trainable path (vision frozen)", tgt_has_path and all(g is None for g in vis_grads))

    # ------------------------------------------------------------------ T4 Stage-2 batch + multi-pass forward (CustomTrainerStage2)
    b2 = ns["collate_fn_stage2"]([conv])
    check("T4a Stage-2 batch has no helper-image pixels (latent supervision dropped)", "pixel_values_latent" not in b2)
    check("T4b Stage-2 token sequence identical to Stage-1", torch.equal(b2["input_ids"], b1["input_ids"]) and torch.equal(b2["labels"], b1["labels"]))
    m2 = tiny_model(meta, "stage2", K, seed=1)
    m2.config.latent_token_id, m2.config.latent_start_id, m2.config.latent_end_id = PAD, LS, LE
    m2.train()
    tr2 = mtrainer.CustomTrainerStage2(model=m2, args=cfg, train_dataset=[conv], data_collator=ns["collate_fn_stage2"],
                                       tokenizer=processor.tokenizer)
    pass_outputs = []   # output hidden states of each language-model pass; the last row of pass i is latent i
    def keep_pass(mod, i, o):
        hs = o[0]
        if hs.requires_grad:
            hs.retain_grad()
        pass_outputs.append(hs)
    hk_pass = m2.model.register_forward_hook(keep_pass)
    loss2, out2 = tr2.compute_loss(m2, dict(b2), return_outputs=True)
    hk_pass.remove()
    T = seq.shape[0]
    check("T4c multi-pass logits cover the full sequence", out2.logits.shape[1] == T, f"{tuple(out2.logits.shape)} vs T={T}")
    emb2, hid2 = out2.inputs_embeds[0], out2.hidden_states[0]
    fill_ok = all(torch.equal(emb2[p], hid2[p - 1]) for p in pads)
    check("T4d latent i input == last-layer post-norm hidden state at position (pad_i - 1) (modeling L2257-L2290)", fill_ok)
    check("T4e loss2 == CE only (trainer.py L34-L38)", torch.allclose(loss2, out2.loss))
    with torch.no_grad():
        single = m2(input_ids=b2["input_ids"], attention_mask=b2["attention_mask"], inputs_embeds=emb2.detach()[None],
                    image_grid_thw=b2["image_grid_thw"], generate_mode=True, return_dict=True)
    diff = (single.logits[0] - out2.logits[0].detach()).abs().max().item()
    check("T4f multi-pass logits == one causal pass over the filled sequence (positions and cache consistent)", diff < 1e-4, f"max|diff|={diff:.2e}")
    loss2.backward()
    # passes 0..K-1 each end at the position whose hidden state becomes the next latent; pass K is the final pass
    producer_grads = [float(pass_outputs[i].grad[0, -1].abs().sum()) for i in range(K)]
    check("T4g text CE gradient reaches the producer of every latent (pass i last row), i.e. no detach in the latent chain",
          len(pass_outputs) == K + 1 and all(gp > 0 for gp in producer_grads) and all(emb2[p].grad_fn is not None for p in pads),
          f"passes={len(pass_outputs)} |grad| at producers={[f'{x:.2e}' for x in producer_grads]}")
    check("T4h CE does not supervise pad tokens; <|latent_end|> is predicted from the last latent position",
          bool((b2["labels"][0][pads] == -100).all()) and int(b2["labels"][0][pads[-1] + 1]) == LE)

    # ------------------------------------------------------------------ T5 fork generate(): latent feedback, cap, flags
    m2.eval()
    item = dict(question_id="diag", category="diag", text=sample["text_input"], image_paths=[sample["image_input"]],
                ground_truth=sample["map_desc"], gt_key="m", map_desc=sample["map_desc"], task="vsp")
    calls = []
    def pre_hook(mod, args_, kwargs):
        calls.append({"embeds": kwargs.get("inputs_embeds"), "ids": kwargs.get("input_ids")})
    def norm_hook(mod, i, o):
        calls[-1]["last_hidden"] = o[:, -1].detach().clone() if calls else None
    hk1 = m2.register_forward_pre_hook(pre_hook, with_kwargs=True)
    hk2 = m2.model.norm.register_forward_hook(norm_hook)

    class Rule(LogitsProcessor):
        def __init__(self, L, plan):
            self.L, self.plan = L, plan
        def __call__(self, input_ids, scores):
            k = input_ids.shape[1] - self.L
            s = scores.clone()
            for b in range(s.shape[0]):
                want = self.plan(b, k)
                if want is not None:
                    s[b] = -1e9
                    s[b, want] = 0
                else:
                    s[b, [LS, LE]] = -1e9
            return s

    import run_mirage_ablation as RUN
    rargs = RUN.normalize_args(RUN.parse_args(["--benchmark", "vsp_planning", "--quota", "auto", "--output_dir", str(tmp)]))
    prompt, image_inputs = RUN.build_prompt(types.SimpleNamespace(processor=processor), item, rargs)
    inputs = processor(text=[prompt], images=image_inputs, return_tensors="pt")
    L = inputs["input_ids"].shape[1]
    with torch.no_grad():
        o = m2.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=1.05, tokenizer=tok,
                        logits_processor=LogitsProcessorList([Rule(L, lambda b, k: LS if k == 0 else None)]))
    gen = o[0, L:].tolist()
    n_pad = gen.count(PAD)
    check("T5a cap: with no <|latent_end|> predicted, exactly 8 latent pads then a forced <|latent_end|>",
          gen[0] == LS and gen[1:9] == [PAD] * 8 and gen[9] == LE, f"pads={n_pad}")
    feeds = [i for i, c in enumerate(calls) if c["embeds"] is not None]
    fed_ok = len(feeds) == 8 and all(torch.equal(calls[i]["embeds"][0, -1], calls[i - 1]["last_hidden"][0]) for i in feeds)
    check("T5b every latent input equals the previous step's last-layer hidden state (generation/utils.py L3417-L3419)", fed_ok,
          f"embedding-fed forwards={len(feeds)}")
    calls.clear()
    # second segment: LS at k=0, LE at latent step 2 (k=3), LS again at k=6, LE again at k=10
    plan2 = {0: LS, 3: LE, 6: LS, 10: LE}
    with torch.no_grad():
        o = m2.generate(**inputs, max_new_tokens=24, do_sample=False, repetition_penalty=1.05, tokenizer=tok,
                        logits_processor=LogitsProcessorList([Rule(L, lambda b, k: plan2.get(k))]))
    gen2 = o[0, L:].tolist()
    after = [i for i, c in enumerate(calls) if i > 7]
    embed_after_second = sum(calls[i]["embeds"] is not None for i in after)
    check("T5c FINDING reproduced: after a second <|latent_start|>, the loop keeps feeding hidden states and the second "
          "<|latent_end|> does not end latent mode (latent_end/latent_end_num never reset, L3380-L3390)",
          gen2[:4] == [LS, PAD, PAD, LE] and gen2[6] == LS and gen2[10] == LE and embed_after_second == len(after) and len(after) > 0,
          f"tokens={gen2[:14]}... embedding-fed forwards after 2nd start: {embed_after_second}/{len(after)}")
    calls.clear()
    # batch of 2: only row 0 asked to start latent mode
    inputs2 = processor(text=[prompt, prompt], images=image_inputs + image_inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        o = m2.generate(**inputs2, max_new_tokens=4, do_sample=False, tokenizer=tok,
                        logits_processor=LogitsProcessorList([Rule(L, lambda b, k: LS if (k == 0 and b == 0) else None)]))
    check("T5d FINDING reproduced: in_latent_mode is batch-global, so a row that never emitted <|latent_start|> is forced to "
          "<|latent_pad|> (L3375-L3387)", o[1, L + 1].item() == PAD and o[1, L].item() != LS, f"row1 tokens={o[1, L:].tolist()}")
    hk1.remove(); hk2.remove()

    # ------------------------------------------------------------------ T6 runner == fork generate (token-exact)
    lm = types.SimpleNamespace(model=m2, processor=processor, tokenizer=tok, latent_ids=ML.resolve_latent_ids(tok, m2.config),
                               dtype=torch.float32, forward=lambda **kw: m2(**kw, generate_mode=True, return_dict=True))
    cap = ML.ForwardCapture(m2.model.norm); vis = ML.ForwardCapture(m2.visual)
    lm.last_hidden = lambda: cap.value
    lm.visual_out = vis
    for name, plan in (("cap8", lambda k: LS if k == 0 else None), ("end@3", lambda k: {0: LS, 4: LE}.get(k))):
        def hook(cur_len, logits, plan=plan):
            want = plan(cur_len - L)
            s = logits.clone()
            if want is not None:
                s[:] = -1e9
                s[want] = 0
            else:
                s[[LS, LE]] = -1e9
            return s
        with torch.no_grad():
            ref = m2.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=1.05, tokenizer=tok,
                              logits_processor=LogitsProcessorList([Rule(L, lambda b, k, plan=plan: plan(k))]))[0, L:].tolist()
        rargs.max_new_tokens = 20
        rec, _ = RUN.run_item(lm, item, rargs, ML.LatentIntervention("none", lm), torch.Generator().manual_seed(0), logits_hook=hook)
        check(f"T6 runner --quota auto reproduces fork generate() token-for-token [{name}]", rec["generated_token_ids"] == ref,
              f"latent steps={rec['latent_steps_total']} exit={rec['latent_segments'][0]['exit_reason'] if rec['latent_segments'] else None}")
    cap.remove(); vis.remove()

    # ------------------------------------------------------------------ T7 intervention operators
    nat = torch.randn(64)
    lm_e = types.SimpleNamespace(tokenizer=tok, model=m2, input_embeddings=m2.get_input_embeddings)
    z, _ = ML.LatentIntervention("zero").apply("q", 0, nat)
    n1, _ = ML.LatentIntervention("noise", noise_salt=42).apply("q", 1, nat)
    n2, _ = ML.LatentIntervention("noise", noise_salt=42).apply("q", 1, nat)
    bank = {"q": torch.randn(4, 64), "d": torch.randn(4, 64)}
    d, meta_d = ML.LatentIntervention("donor", donor_bank=bank, donor_map={"q": "d"}).apply("q", 2, nat)
    p, _ = ML.LatentIntervention("pause", lm_e).apply("q", 0, nat)
    dot = tok.encode(".", add_special_tokens=False)[0]
    check("T7 interventions: zero, norm-matched seeded noise, exact donor, pause = token embedding",
          bool((z == 0).all()) and abs(n1.norm() - nat.norm()) < 1e-4 and torch.equal(n1, n2)
          and torch.equal(d, bank["d"][2]) and meta_d["donor_id"] == "d" and torch.equal(p, m2.get_input_embeddings().weight[dot]))

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
