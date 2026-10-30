#!/usr/bin/env python3
"""
WSDM Experiments W1, W3 & W5:
1. W1: Full 6-Checkpoint Training Lineage Audit on BLINK (N=700, 2x2 Factorial V0-V4 + CF):
   - Qwen/Qwen2.5-VL-7B-Instruct (Base Backbone: Direct Answer & <think></think>)
   - array/Qwen2.5-VL-SATBase (Paper's exact Table 1 Row b DirAns: SAT + Video-R1 + Zebra-CoT)
   - array/Qwen2.5-VL-Latent-Custom (Stage-1 / Intermediate Latent Checkpoint: 20 latents & <think></think>)
   - (Integrated with existing T1 Mull, T8 SAT, T10 Mull-GRPO from E2)

2. W3: Cross-Benchmark Generalization (SAT-Real N=300, ERQA N=400, MMSI-Bench N=1000):
   - Evaluates Mull (20 latents, <think></think>, 20x" .", T2F forced <answer),
     Mull-GRPO (20 latents, <think></think>), SAT, SATBase, and Base Qwen2.5-VL-7B.
   - Computes Raw Accuracy, Content-Free Calibrated Accuracy, Continuous Logit Margins,
     Paired McNemar, Wilcoxon Signed-Rank, and TOST Equivalence across all 1,700 external items.

3. W5: Constructive Inference Protocol (Zero-Latent Format Lock + Prior Calibration).
"""

import argparse
import copy
import gc
import glob
import io
import importlib
import json
import math
import os
import random
import re
import sys
import time
import numpy as np
import scipy.stats as stats
import torch
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

# Allow downloading array/Qwen2.5-VL-SATBase and array/Qwen2.5-VL-Latent-Custom from HF Hub if needed
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "mull", "models")
LMMS_DIR = os.path.join(PROJECT_DIR, "mull", "lmms-eval")
E2_DIR = os.path.join(PROJECT_DIR, "e2_bias")
for p in (MODELS_DIR, LMMS_DIR, E2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from score_letters import (
    MAX_PIXELS, MIN_PIXELS, ANSWER_PREFIX_IDS, SEED,
    TASKS as BLINK_TASKS, EXPECTED_N as BLINK_EXPECTED_N,
    CF_FILLERS, CF_COLORS, doc_to_text as blink_doc_to_text,
    doc_images as blink_doc_images, build_two_way_variant,
    build_content_free_prompt, latents
)

from lmms_eval.tasks.sat_real.utils import process_docs as sat_process_docs, sat_doc_to_text, sat_doc_to_visual, sat_doc_to_target
from lmms_eval.tasks.erqa.utils import erqa_doc_to_text, erqa_doc_to_visual, erqa_doc_to_target
from lmms_eval.tasks.mmsi_bench.utils import mmsi_doc_to_text, mmsi_doc_to_visual

OUT_DIR = os.path.join(PROJECT_DIR, "wsdm_experiments", "outputs_w1_w3")

MODEL_PATHS = {
    "base": "Qwen/Qwen2.5-VL-7B-Instruct",
    "sat": "array/Qwen2.5-VL-SAT",
    "satbase": "array/Qwen2.5-VL-SATBase",
    "latcustom": "array/Qwen2.5-VL-Latent-Custom",
    "mull": "array/Qwen2.5-VL-Mull",
    "grpo": "array/Qwen2.5-VL-MullGRPO",
}


class UniversalScorer:
    def __init__(self, model_key):
        self.model_key = model_key
        pretrained = MODEL_PATHS[model_key]
        print(f"\n[UniversalScorer] Loading {model_key} -> {pretrained} ...")
        Q = importlib.import_module("mmlatentdiscrete_qwen_vl").Qwen2_5_VLForConditionalGeneration
        self.model = Q.from_pretrained(
            pretrained,
            torch_dtype="bfloat16",
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
        tk = self.processor.tokenizer
        n_added = 0
        for t in ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>", "<|latent_image|>"]:
            n_added += tk.add_tokens(t, special_tokens=True)
        cfg = self.model.config
        cfg.latent_token_id = int(tk("<|latent_pad|>")["input_ids"][0])
        cfg.latent_start_id = int(tk("<|latent_start|>")["input_ids"][0])
        cfg.latent_end_id = int(tk("<|latent_end|>")["input_ids"][0])
        cfg.imagelatent_token_id = int(tk("<|latent_image|>")["input_ids"][0])
        if n_added > 0 and self.model.get_input_embeddings().weight.shape[0] < len(tk):
            self.model.resize_token_embeddings(len(tk))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.letter_ids = {}
        self.id_to_letter = {}
        for L in "ABCDEF":
            ids = self.tokenizer.encode(f"<answer>{L}</answer>", add_special_tokens=False)
            assert len(ids) == 6 and ids[:2] == ANSWER_PREFIX_IDS
            self.letter_ids[L] = ids[2]
            self.id_to_letter[ids[2]] = L

        # 1-entry cache for same-image multi-condition / multi-variant evaluation
        self._cache_key = None
        self._cached_imgs_ref = None
        self._cached_vision_info = None
        self._cached_img_proc = None
        self._cached_vis_embeds = None

        self._orig_img_proc = self.processor.image_processor.__call__
        orig_img_proc = self._orig_img_proc
        def cached_img_proc(*args, **kwargs):
            if self._cache_key is not None and self._cached_img_proc is not None:
                return BatchFeature(data=dict(self._cached_img_proc))
            res = orig_img_proc(*args, **kwargs)
            if self._cache_key is not None:
                self._cached_img_proc = dict(res)
            return BatchFeature(data=dict(res))
        self.processor.image_processor.__call__ = cached_img_proc

        self._orig_vis_forward = self.model.visual.forward
        orig_vis_forward = self._orig_vis_forward
        def cached_vis_forward(*args, **kwargs):
            if self._cache_key is not None and self._cached_vis_embeds is not None:
                return self._cached_vis_embeds
            res = orig_vis_forward(*args, **kwargs)
            if self._cache_key is not None:
                self._cached_vis_embeds = res
            return res
        self.model.visual.forward = cached_vis_forward

    def cleanup(self):
        self._cache_key = None
        self._cached_imgs_ref = None
        self._cached_vision_info = None
        self._cached_img_proc = None
        self._cached_vis_embeds = None
        if hasattr(self, "processor") and self.processor is not None:
            self.processor.image_processor.__call__ = self._orig_img_proc
        if hasattr(self, "model") and self.model is not None:
            self.model.visual.forward = self._orig_vis_forward
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def score(self, images, user_text, prefill, letters):
        img_key = tuple(id(im) for im in images)
        if img_key != self._cache_key:
            self._cache_key = img_key
            self._cached_imgs_ref = images
            self._cached_vision_info = None
            self._cached_img_proc = None
            self._cached_vis_embeds = None

        messages = [
            {"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if text.endswith("<|im_end|>\n"):
            text = text[:-len("<|im_end|>\n")]
        elif text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
        if self._cached_vision_info is None:
            self._cached_vision_info = process_vision_info(messages)
        image_inputs, video_inputs = self._cached_vision_info
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        extra = torch.tensor([ANSWER_PREFIX_IDS], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(extra)], dim=1)
        inputs = inputs.to("cuda:0")
        self.model.rope_deltas = None
        out = self.model(**inputs, use_cache=False, return_dict=True)
        logits = out.logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        cand = {L: float(logp[self.letter_ids[L]]) for L in letters}
        cand_logits = {L: float(logits[self.letter_ids[L]]) for L in letters}
        top1 = int(torch.argmax(logits))
        top1_letter = self.id_to_letter.get(top1, None)
        cand_argmax = max(cand, key=cand.get)
        tf_pred = top1_letter if (top1_letter is not None and top1_letter in letters) else cand_argmax
        del out, logits, logp, inputs
        return dict(logp=cand, logit=cand_logits, tf_pred=tf_pred)


# ----------------------------------------------------------------------------- W1: Lineage on BLINK
W1_CONDITIONS = {
    "W1_Base_Dir":      dict(model="base",      prompt="sft",     prefill=""),
    "W1_Base_Think":    dict(model="base",      prompt="latents", prefill="<think></think>\n"),
    "W1_SATBase_Dir":   dict(model="satbase",   prompt="sft",     prefill=""),
    "W1_SATBase_Lat":   dict(model="satbase",   prompt="latents", prefill=""),
    "W1_LatCust_K20":   dict(model="latcustom", prompt="latents", prefill=latents(20)),
    "W1_LatCust_Think": dict(model="latcustom", prompt="latents", prefill="<think></think>\n"),
}


def run_w1_blink_lineage_group(scorer, cids, out_dir, limit=None):
    active_cids = []
    for cid in cids:
        cdir = os.path.join(out_dir, "w1_blink", cid)
        os.makedirs(cdir, exist_ok=True)
        done_marker = os.path.join(cdir, "COMPLETE")
        if limit is None and os.path.exists(done_marker):
            print(f"  [W1] {cid} already COMPLETE, skipping.")
        else:
            active_cids.append(cid)
    if not active_cids:
        return

    variants = ["V0", "V2", "V3", "V4", "CF"]
    files = {
        cid: {v: open(os.path.join(out_dir, "w1_blink", cid, f"{v}.jsonl"), "w") for v in variants}
        for cid in active_cids
    }

    t0 = time.time()
    for task, (cfg_name, letters, two_way) in BLINK_TASKS.items():
        t_task = time.time()
        ds = load_dataset("BLINK-Benchmark/BLINK", cfg_name, split="val")
        n_docs = len(ds) if limit is None else min(limit, len(ds))
        for doc_id in range(n_docs):
            doc = ds[doc_id]
            gold = doc["answer"].replace("(", "").replace(")", "")
            imgs = blink_doc_images(doc)
            key = f"{task}|{doc_id}"

            for cid in active_cids:
                c = W1_CONDITIONS[cid]
                base_text = blink_doc_to_text(doc, c["prompt"])
                # V0
                r0 = scorer.score(imgs, base_text, c["prefill"], letters)
                files[cid]["V0"].write(json.dumps(dict(key=key, cond=cid, variant="V0", task=task, doc_id=doc_id, gold=gold, letters=letters, **r0)) + "\n")

                # V2, V3, V4 on 2-way splits
                if two_way:
                    for v in ("V2", "V3", "V4"):
                        new_prompt, loc = build_two_way_variant(doc["prompt"], v)
                        d2 = dict(doc); d2["prompt"] = new_prompt
                        vtext = blink_doc_to_text(d2, c["prompt"])
                        rv = scorer.score(imgs, vtext, c["prefill"], letters)
                        files[cid][v].write(json.dumps(dict(key=key, cond=cid, variant=v, task=task, doc_id=doc_id, gold=gold, letters=letters, **rv)) + "\n")

        # CF probes (27 per task)
        n_cf_items = min(3, len(ds))
        pick = random.Random(f"{SEED}-{task}").sample(range(len(ds)), n_cf_items)
        for item in pick:
            doc = ds[item]
            sizes = [im.size for im in blink_doc_images(doc)]
            for cname, col in CF_COLORS.items():
                canv = [Image.new("RGB", s, col) for s in sizes]
                for filler in CF_FILLERS:
                    cf_prompt = build_content_free_prompt(doc["prompt"], filler)
                    d2 = dict(doc); d2["prompt"] = cf_prompt
                    cf_key = f"{task}|{item}|{filler or '<empty>'}|{cname}"
                    for cid in active_cids:
                        c = W1_CONDITIONS[cid]
                        cf_text = blink_doc_to_text(d2, c["prompt"])
                        rcf = scorer.score(canv, cf_text, c["prefill"], letters)
                        files[cid]["CF"].write(json.dumps(dict(key=cf_key, cond=cid, variant="CF", task=task, doc_id=item, letters=letters, **rcf)) + "\n")
        print(f"    [W1] {active_cids} finished {task} (n={n_docs}) in {time.time() - t_task:.1f}s")

    for cid in active_cids:
        for f in files[cid].values():
            f.close()
        if limit is None:
            done_marker = os.path.join(out_dir, "w1_blink", cid, "COMPLETE")
            open(done_marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"  [W1] Completed {active_cids} in {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------------- W3: External Benchmarks (SAT-Real, ERQA, MMSI-Bench)
W3_CONDITIONS = {
    "W3_Mull_K20":      dict(model="mull",      prompt="latents", prefill=latents(20)),
    "W3_Mull_Think":    dict(model="mull",      prompt="latents", prefill="<think></think>\n"),
    "W3_Mull_Dots":     dict(model="mull",      prompt="latents", prefill="<think>" + " ." * 20 + "</think>\n"),
    "W3_Mull_T2F":      dict(model="mull",      prompt="latents", prefill=""),
    "W3_GRPO_K20":      dict(model="grpo",      prompt="latents", prefill=latents(20)),
    "W3_GRPO_Think":    dict(model="grpo",      prompt="latents", prefill="<think></think>\n"),
    "W3_SAT_Dir":       dict(model="sat",       prompt="sft",     prefill=""),
    "W3_SATBase_Dir":   dict(model="satbase",   prompt="sft",     prefill=""),
    "W3_LatCust_K20":   dict(model="latcustom", prompt="latents", prefill=latents(20)),
    "W3_Base_Dir":      dict(model="base",      prompt="sft",     prefill=""),
}


def load_external_benchmark(bench_name):
    if bench_name == "sat_real":
        raw_ds = load_dataset("array/SAT-v2", split="test")
        ds = sat_process_docs(raw_ds)  # 150 original + 150 reversed = 300 items
        items = []
        for idx in range(len(ds)):
            doc = ds[idx]
            gold = sat_doc_to_target(doc).strip().upper()
            n_opts = len(doc["answers"])
            letters = "".join(["ABCDEF"[i] for i in range(n_opts)])
            items.append({
                "doc_id": idx,
                "doc": doc,
                "gold": gold,
                "letters": letters,
                "subtask": doc.get("question_type", "sat_real"),
            })
        return items
    elif bench_name == "erqa":
        ds = load_dataset("RunsenXu/ERQA", split="test", verification_mode="no_checks")
        items = []
        for idx in range(len(ds)):
            doc = ds[idx]
            gold = erqa_doc_to_target(doc).replace("(", "").replace(")", "").strip().upper()
            # Detect option letters present in question text (default ABCD)
            found = re.findall(r"\b([A-F])\.", doc["question"])
            if not found:
                found = re.findall(r"\(([A-F])\)", doc["question"])
            letters = "".join(sorted(list(set(found)))) if len(set(found)) >= 2 else "ABCD"
            if gold not in letters:
                letters = "".join(sorted(list(set(letters + gold))))
            items.append({
                "doc_id": idx,
                "doc": doc,
                "gold": gold,
                "letters": letters,
                "subtask": doc.get("question_type", "erqa"),
            })
        return items
    elif bench_name == "mmsi_bench":
        ds = load_dataset("RunsenXu/MMSI-Bench", split="test")
        items = []
        for idx in range(len(ds)):
            doc = ds[idx]
            gold = doc["answer"].strip().upper()
            letters = "ABCD"
            items.append({
                "doc_id": idx,
                "doc": doc,
                "gold": gold,
                "letters": letters,
                "subtask": doc.get("question_type", "mmsi"),
            })
        return items
    else:
        raise ValueError(bench_name)


def get_bench_text(bench_name, doc, prompt_mode):
    if bench_name == "sat_real":
        return sat_doc_to_text(doc, {"prompt_mode": prompt_mode})
    elif bench_name == "erqa":
        return erqa_doc_to_text(doc, {"prompt_mode": prompt_mode})
    elif bench_name == "mmsi_bench":
        return mmsi_doc_to_text(doc, {"prompt_mode": prompt_mode})
    else:
        raise ValueError(bench_name)


def get_bench_imgs(bench_name, doc):
    if bench_name == "sat_real":
        return sat_doc_to_visual(doc)
    elif bench_name == "erqa":
        return erqa_doc_to_visual(doc)
    elif bench_name == "mmsi_bench":
        return mmsi_doc_to_visual(doc)
    else:
        raise ValueError(bench_name)


def run_w3_external_benchmarks_group(scorer, cids, benchmarks, out_dir, limit=None):
    for bname in benchmarks:
        active_cids = []
        for cid in cids:
            cdir = os.path.join(out_dir, "w3_benchmarks", cid)
            os.makedirs(cdir, exist_ok=True)
            done_marker = os.path.join(cdir, f"{bname}.COMPLETE")
            if limit is None and os.path.exists(done_marker):
                print(f"  [W3] {cid}/{bname} already COMPLETE, skipping.")
            else:
                active_cids.append(cid)
        if not active_cids:
            continue

        items = load_external_benchmark(bname)
        n_docs = len(items) if limit is None else min(limit, len(items))
        t0 = time.time()

        fps = {cid: open(os.path.join(out_dir, "w3_benchmarks", cid, f"{bname}.jsonl"), "w") for cid in active_cids}
        fps_cf = {cid: open(os.path.join(out_dir, "w3_benchmarks", cid, f"{bname}_CF.jsonl"), "w") for cid in active_cids}

        for i in range(n_docs):
            it = items[i]
            imgs = get_bench_imgs(bname, it["doc"])
            for cid in active_cids:
                c = W3_CONDITIONS[cid]
                txt = get_bench_text(bname, it["doc"], c["prompt"])
                res = scorer.score(imgs, txt, c["prefill"], it["letters"])
                rec = {
                    "key": f"{bname}|{it['doc_id']}",
                    "bench": bname,
                    "cond": cid,
                    "doc_id": it["doc_id"],
                    "subtask": it["subtask"],
                    "gold": it["gold"],
                    "letters": it["letters"],
                    **res,
                }
                fps[cid].write(json.dumps(rec) + "\n")

        # Run 9 content-free calibration probes per benchmark
        sample_it = items[0]
        sample_imgs = get_bench_imgs(bname, sample_it["doc"])
        sizes = [im.size for im in sample_imgs]
        letters_all = "ABCD" if bname != "sat_real" else "AB"
        for cname, col in CF_COLORS.items():
            canv = [Image.new("RGB", s, col) for s in sizes]
            for filler in CF_FILLERS:
                opts_str = "\n".join([f"({L}) {filler}" for L in letters_all])
                raw_q = f"{filler}\nChoose from the following options:\n{opts_str}\nProvide your final answer between the <answer> </answer> tags. Please provide only the single option letter within the <answer> </answer> tags."
                for cid in active_cids:
                    c = W3_CONDITIONS[cid]
                    rcf = scorer.score(canv, raw_q, c["prefill"], letters_all)
                    fps_cf[cid].write(json.dumps({"bench": bname, "cond": cid, "filler": filler, "color": cname, "letters": letters_all, **rcf}) + "\n")

        for cid in active_cids:
            fps[cid].close()
            fps_cf[cid].close()
            if limit is None:
                done_marker = os.path.join(out_dir, "w3_benchmarks", cid, f"{bname}.COMPLETE")
                open(done_marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"  [W3] Completed {active_cids}/{bname} (n={n_docs}) in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["mull", "grpo", "sat", "satbase", "latcustom", "base"])
    ap.add_argument("--benchmarks", nargs="+", default=["sat_real", "erqa", "mmsi_bench"])
    ap.add_argument("--skip_w1", action="store_true")
    ap.add_argument("--skip_w3", action="store_true")
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out if args.limit is None else os.path.join(PROJECT_DIR, "wsdm_experiments", "scratch_w1_w3_smoke")
    os.makedirs(out_dir, exist_ok=True)

    for mk in args.models:
        w1_cids = [k for k, v in W1_CONDITIONS.items() if v["model"] == mk] if not args.skip_w1 else []
        w3_cids = [k for k, v in W3_CONDITIONS.items() if v["model"] == mk] if not args.skip_w3 else []
        if not w1_cids and not w3_cids:
            continue

        scorer = UniversalScorer(mk)
        if w1_cids:
            run_w1_blink_lineage_group(scorer, w1_cids, out_dir, limit=args.limit)
        if w3_cids:
            run_w3_external_benchmarks_group(scorer, w3_cids, args.benchmarks, out_dir, limit=args.limit)
        scorer.cleanup()
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n>>> All requested W1 & W3 runs finished!")


if __name__ == "__main__":
    main()
