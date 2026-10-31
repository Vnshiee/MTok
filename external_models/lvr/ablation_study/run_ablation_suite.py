"""
Master Resumable Evaluation & Ablation Runner for the LVR Ablation Study.

Supports:
- Axis 1A: Inference-Time Causal Interventions (`A1A-BASE`, `A1A-MASK-ALL-IMG`,
  `A1A-MASK-ANS-TO-LVR`, `A1A-MASK-ANS-TO-IMG`, `A1A-ORACLE-ROI`, `A1A-ORACLE-BKG`, `A1A-2D-MROPE`)
  + Teacher-Forcing Exposure Bias Rollout Drift (`delta_k`).
- Axis 2 (Baseline Recovery): `A2-BASE-3B` (`Qwen/Qwen2.5-VL-3B-Instruct`) and `BASE-7B`
  (`Qwen/Qwen2.5-VL-7B-Instruct`).
- Axis 3A & 3B: Step Quota Scaling (`K in {0, 1, 2, 4, 8, 16, 32, 64}`) and Decoding
  Termination Strategies (`A3B-STEPS-8`, `A3B-FORCED-END`, `A3B-VANILLA-LOGIT`, `A3B-LATENT-END`).
- Axis 4: 9-Condition Latent Representation Substrate & Format-Locking Matrix
  (`C1-CONT-DEFAULT` .. `C9-DIRECT-ANS`).
- Axis 5: 2 x 2 x 2 Evaluation Harness Audit (Official First-Char Parser vs. Robust Regex Parser,
  Official Mutated Prompt vs. Clean/BBox Prompt, Canonical vs. CircularEval Option Permutation).

Every item is checkpointed immediately to JSONL so runs inside `tmux` are 100% resumable.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT_DIR = "/home/devraj/MTok/external_models/lvr"
REPO_DIR = os.path.join(ROOT_DIR, "repo")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from ablation_study.data_loader import (
    LVR_SFT_BBOX_INSTRUCTION,
    load_blink_bench,
    load_mmvp_bench,
    load_vstar_bench,
)
from ablation_study.lvr_interventions import create_qwen_messages, run_lvr_intervention_single
from ablation_study.parsers_and_stats import (
    accuracy_reward_official,
    accuracy_reward_robust,
    build_circular_shifted_item,
    compute_paired_statistics,
    holm_bonferroni_correction,
)


_ORIG_QWEN25_VL_FORWARD = Qwen2_5_VLForConditionalGeneration.forward


def load_lvr_model(model_id: str = "vincentleebang/LVR-7B", attn_impl: str = "sdpa"):
    from src.model.qwen_lvr_model import QwenWithLVR
    from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

    config = AutoConfig.from_pretrained(model_id)
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True, lvr_head=getattr(config, "lvr_head", False), rl=False
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = QwenWithLVR.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()
    return model, processor


def load_base_qwen_model(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", attn_impl: str = "sdpa"):
    # Restore original Qwen2_5_VLForConditionalGeneration.forward if monkey-patched by LVR
    Qwen2_5_VLForConditionalGeneration.forward = _ORIG_QWEN25_VL_FORWARD
    config = AutoConfig.from_pretrained(model_id)
    if isinstance(getattr(config, "text_config", None), dict):
        delattr(config, "text_config")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()
    return model, processor


@torch.no_grad()
def run_base_qwen_single(
    model: torch.nn.Module,
    processor: Any,
    img_input: Any,
    prompt_text: str,
    max_new_tokens: int = 128,
) -> str:
    device = next(model.parameters()).device
    messages = create_qwen_messages(img_input, prompt_text)
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    trimmed = gen_ids[0, inputs.input_ids.shape[1] :]
    return processor.decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def evaluate_condition_on_dataset(
    model: torch.nn.Module,
    processor: Any,
    dataset: List[Dict[str, Any]],
    benchmark_name: str,
    condition: str,
    lvr_steps: int,
    out_dir: str,
    prompt_mode: str = "official",
    collect_attention_telemetry: bool = False,
    max_samples: Optional[int] = None,
    is_base_model: bool = False,
    donor_states_cache: Optional[Dict[str, torch.Tensor]] = None,
    base_states_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Runs or resumes evaluation of a single `(benchmark_name, condition, lvr_steps, prompt_mode)`
    configuration, saving per-item JSONL records and a summary JSON.
    """
    bench_dir = os.path.join(out_dir, benchmark_name)
    os.makedirs(bench_dir, exist_ok=True)
    run_tag = f"{condition}_steps{lvr_steps:02d}_{prompt_mode}"
    jsonl_path = os.path.join(bench_dir, f"{run_tag}.jsonl")
    summary_path = os.path.join(bench_dir, f"{run_tag}_summary.json")

    eval_items = dataset[:max_samples] if max_samples is not None else dataset

    # Load existing completed items for seamless resume
    completed_records: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed_records[str(rec["question_id"])] = rec
                except Exception:
                    continue

    records_list: List[Dict[str, Any]] = []
    with open(jsonl_path, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(eval_items, desc=f"[{benchmark_name}] {run_tag}")):
            qid = str(item["question_id"])
            if qid in completed_records:
                records_list.append(completed_records[qid])
                continue

            if prompt_mode == "official":
                prompt_text = item["text_official_eval"]
            elif prompt_mode == "clean":
                prompt_text = item["text_clean"]
            elif prompt_mode == "bbox_aligned":
                prompt_text = item["text_clean"] + LVR_SFT_BBOX_INSTRUCTION
            else:
                raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

            t0 = time.time()
            if is_base_model:
                output_text = run_base_qwen_single(
                    model=model,
                    processor=processor,
                    img_input=item["image"],
                    prompt_text=prompt_text,
                    max_new_tokens=128,
                )
                telemetry = {}
            else:
                # For C8-SHUFFLE-BATCH, get donor states from previous/next item in dataset
                donor_states = None
                if condition == "C8-SHUFFLE-BATCH":
                    donor_item = eval_items[(idx + 1) % len(eval_items)]
                    donor_qid = str(donor_item["question_id"])
                    if donor_states_cache and donor_qid in donor_states_cache:
                        donor_states = donor_states_cache[donor_qid]
                    else:
                        donor_res = run_lvr_intervention_single(
                            model=model,
                            processor=processor,
                            img_input=donor_item["image"],
                            prompt_text=donor_item["text_official_eval"],
                            condition="A1A-BASE",
                            lvr_steps=lvr_steps,
                            max_new_tokens=32,
                            bboxes_norm=donor_item.get("bboxes_norm", []),
                            collect_attention_telemetry=False,
                        )
                        donor_states = donor_res["raw_latent_states"]
                        if donor_states_cache is not None:
                            donor_states_cache[donor_qid] = donor_states

                res = run_lvr_intervention_single(
                    model=model,
                    processor=processor,
                    img_input=item["image"],
                    prompt_text=prompt_text,
                    condition=condition,
                    lvr_steps=lvr_steps,
                    max_new_tokens=128,
                    bboxes_norm=item.get("bboxes_norm", []),
                    donor_latent_states=donor_states,
                    collect_attention_telemetry=collect_attention_telemetry,
                )
                output_text = res["output_text"]
                telemetry = res["telemetry"]
                raw_states = res["raw_latent_states"]

                if base_states_cache is not None and condition in ("A1A-BASE", "C1-CONT-DEFAULT") and lvr_steps == 8:
                    base_states_cache[qid] = raw_states
                if donor_states_cache is not None and condition in ("A1A-BASE", "C1-CONT-DEFAULT") and lvr_steps == 8:
                    donor_states_cache[qid] = raw_states

                # If running A1A-ORACLE-ROI and base_states_cache has free-running states for qid,
                # compute step-wise Teacher-Forcing Exposure Bias Rollout Drift delta_k (H_1b)
                if (
                    condition == "A1A-ORACLE-ROI"
                    and base_states_cache is not None
                    and qid in base_states_cache
                ):
                    free_h = base_states_cache[qid]
                    tf_h = raw_states
                    k_min = min(free_h.shape[0], tf_h.shape[0])
                    if k_min > 0:
                        l2_drift = torch.norm(free_h[:k_min] - tf_h[:k_min], p=2, dim=-1).tolist()
                        cos_drift = F.cosine_similarity(free_h[:k_min], tf_h[:k_min], dim=-1).tolist()
                        telemetry["rollout_l2_drift_per_step"] = [round(float(x), 4) for x in l2_drift]
                        telemetry["rollout_l2_drift_mean"] = round(float(np.mean(l2_drift)), 4)
                        telemetry["rollout_cos_per_step"] = [round(float(x), 4) for x in cos_drift]
                        telemetry["rollout_cos_mean"] = round(float(np.mean(cos_drift)), 4)

            elapsed = time.time() - t0
            corr_official = accuracy_reward_official(output_text, item["label"])
            corr_robust = accuracy_reward_robust(output_text, item["label"], item.get("options_dict"))

            rec = {
                "question_id": qid,
                "category": item["category"],
                "label": item["label"],
                "condition": condition,
                "lvr_steps": lvr_steps,
                "prompt_mode": prompt_mode,
                "output_text": output_text,
                "correct_official": corr_official,
                "correct_robust": corr_robust,
                "elapsed_sec": round(elapsed, 3),
                "telemetry": telemetry,
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            records_list.append(rec)

    # Compute aggregate summary metrics
    n_total = len(records_list)
    acc_off = float(np.mean([r["correct_official"] for r in records_list])) * 100.0 if n_total else 0.0
    acc_rob = float(np.mean([r["correct_robust"] for r in records_list])) * 100.0 if n_total else 0.0

    by_category: Dict[str, Dict[str, Any]] = {}
    for r in records_list:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"n": 0, "off": 0, "rob": 0}
        by_category[cat]["n"] += 1
        by_category[cat]["off"] += int(r["correct_official"])
        by_category[cat]["rob"] += int(r["correct_robust"])

    cat_summary = {}
    for cat, counts in by_category.items():
        cat_summary[cat] = {
            "n": counts["n"],
            "acc_official_pct": round(counts["off"] / counts["n"] * 100.0, 2),
            "acc_robust_pct": round(counts["rob"] / counts["n"] * 100.0, 2),
        }

    # For BLINK, paper reports unweighted macro average across the 5 subtasks
    if benchmark_name == "blink" and len(cat_summary) > 0:
        macro_off = float(np.mean([v["acc_official_pct"] for v in cat_summary.values()]))
        macro_rob = float(np.mean([v["acc_robust_pct"] for v in cat_summary.values()]))
    else:
        macro_off = acc_off
        macro_rob = acc_rob

    # Aggregate mechanistic telemetry means across dataset
    tel_keys = [
        "actual_latent_steps",
        "cs_max_mean",
        "cs_roi_mean",
        "cs_bkg_mean",
        "roi_recall_at_1_mean",
        "roi_recall_at_5_mean",
        "latent_norm_mean",
        "effective_rank",
        "adj_cosine_mean",
        "latent_attn_to_vis_mean",
        "latent_attn_to_roi_mean",
        "latent_attn_to_bkg_mean",
        "ans_attn_to_lvr_mean",
        "ans_attn_to_vis_mean",
        "rollout_l2_drift_mean",
        "rollout_cos_mean",
    ]
    agg_telemetry = {}
    for tk in tel_keys:
        vals = [
            r["telemetry"].get(tk)
            for r in records_list
            if r.get("telemetry") and r["telemetry"].get(tk) is not None
        ]
        if vals:
            agg_telemetry[tk] = round(float(np.mean(vals)), 4)

    summary = {
        "benchmark": benchmark_name,
        "condition": condition,
        "lvr_steps": lvr_steps,
        "prompt_mode": prompt_mode,
        "n_total": n_total,
        "acc_official_micro_pct": round(acc_off, 2),
        "acc_robust_micro_pct": round(acc_rob, 2),
        "acc_official_macro_pct": round(macro_off, 2),
        "acc_robust_macro_pct": round(macro_rob, 2),
        "by_category": cat_summary if benchmark_name != "mmvp" else {},
        "telemetry_means": agg_telemetry,
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print(
        f"==> [{benchmark_name}] {run_tag}: "
        f"Official={summary['acc_official_micro_pct']:.2f}% (Macro={summary['acc_official_macro_pct']:.2f}%) | "
        f"Robust={summary['acc_robust_micro_pct']:.2f}% (Macro={summary['acc_robust_macro_pct']:.2f}%)"
    )
    return jsonl_path, summary


def evaluate_circular_eval_on_dataset(
    model: torch.nn.Module,
    processor: Any,
    dataset: List[Dict[str, Any]],
    benchmark_name: str,
    condition: str,
    lvr_steps: int,
    out_dir: str,
    max_samples: Optional[int] = None,
    is_base_model: bool = False,
) -> Dict[str, Any]:
    """
    Executes Axis 5 Factor C (`CircularEval`): for each item with M choices, tests all M
    cyclic option permutations. An item passes `CircularEval` iff answered correctly under
    ALL M cyclic shifts.
    """
    bench_dir = os.path.join(out_dir, benchmark_name)
    os.makedirs(bench_dir, exist_ok=True)
    run_tag = f"{condition}_steps{lvr_steps:02d}_circular_eval"
    jsonl_path = os.path.join(bench_dir, f"{run_tag}.jsonl")
    summary_path = os.path.join(bench_dir, f"{run_tag}_summary.json")

    eval_items = dataset[:max_samples] if max_samples is not None else dataset
    completed: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed[str(rec["question_id"])] = rec

    records_list = []
    with open(jsonl_path, "a", encoding="utf-8") as out_f:
        for item in tqdm(eval_items, desc=f"[{benchmark_name}] {run_tag}"):
            qid = str(item["question_id"])
            if qid in completed:
                records_list.append(completed[qid])
                continue

            n_opts = max(1, len(item.get("options_dict", {})))
            shift_off_list = []
            shift_rob_list = []
            for s_idx in range(n_opts):
                shifted = build_circular_shifted_item(item, s_idx)
                if is_base_model:
                    out_txt = run_base_qwen_single(
                        model, processor, shifted["image"], shifted["text_official_eval"], max_new_tokens=128
                    )
                else:
                    res = run_lvr_intervention_single(
                        model=model,
                        processor=processor,
                        img_input=shifted["image"],
                        prompt_text=shifted["text_official_eval"],
                        condition=condition,
                        lvr_steps=lvr_steps,
                        max_new_tokens=128,
                    )
                    out_txt = res["output_text"]
                shift_off_list.append(accuracy_reward_official(out_txt, shifted["label"]))
                shift_rob_list.append(accuracy_reward_robust(out_txt, shifted["label"], shifted.get("options_dict")))

            rec = {
                "question_id": qid,
                "category": item["category"],
                "condition": condition,
                "lvr_steps": lvr_steps,
                "prompt_mode": "circular_eval",
                "shift_correct_official": shift_off_list,
                "shift_correct_robust": shift_rob_list,
                "correct_official": all(shift_off_list),
                "correct_robust": all(shift_rob_list),
                "telemetry": {},
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            records_list.append(rec)

    n_tot = len(records_list)
    acc_off = float(np.mean([r["correct_official"] for r in records_list])) * 100.0 if n_tot else 0.0
    acc_rob = float(np.mean([r["correct_robust"] for r in records_list])) * 100.0 if n_tot else 0.0
    summary = {
        "benchmark": benchmark_name,
        "condition": condition,
        "lvr_steps": lvr_steps,
        "prompt_mode": "circular_eval",
        "n_total": n_tot,
        "acc_official_micro_pct": round(acc_off, 2),
        "acc_robust_micro_pct": round(acc_rob, 2),
        "acc_official_macro_pct": round(acc_off, 2),
        "acc_robust_macro_pct": round(acc_rob, 2),
        "by_category": {},
        "telemetry_means": {},
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    print(f"==> [{benchmark_name}] {run_tag}: CircularEval Official={acc_off:.2f}% | Robust={acc_rob:.2f}%")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run LVR Ablation Study Matrix")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["vstar", "mmvp", "blink"],
        choices=["vstar", "mmvp", "blink", "blink_ext"],
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["all"],
        choices=["all", "axis1a", "axis2_base", "axis3a", "axis3b", "axis4", "axis5"],
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--collect_attention_telemetry", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "ablation_outputs"),
    )
    args = parser.parse_args()

    axes_to_run = set(args.axes)
    if "all" in axes_to_run:
        axes_to_run = {"axis1a", "axis2_base", "axis3a", "axis3b", "axis4", "axis5"}

    datasets_map: Dict[str, List[Dict[str, Any]]] = {}
    for b in args.benchmarks:
        if b == "vstar":
            datasets_map[b] = load_vstar_bench()
        elif b == "mmvp":
            datasets_map[b] = load_mmvp_bench()
        elif b == "blink":
            datasets_map[b] = load_blink_bench(include_extended=False)
        elif b == "blink_ext":
            datasets_map[b] = load_blink_bench(include_extended=True)

    lvr_axes = axes_to_run & {"axis1a", "axis3a", "axis3b", "axis4", "axis5"}
    if lvr_axes:
        model, processor = load_lvr_model("vincentleebang/LVR-7B", attn_impl="sdpa")

        for b_name, ds in datasets_map.items():
            donor_cache: Dict[str, torch.Tensor] = {}
            base_cache: Dict[str, torch.Tensor] = {}

            # Always run A1A-BASE (K=8) first so base_cache and donor_cache are populated
            evaluate_condition_on_dataset(
                model=model,
                processor=processor,
                dataset=ds,
                benchmark_name=b_name,
                condition="A1A-BASE",
                lvr_steps=8,
                out_dir=args.out_dir,
                prompt_mode="official",
                collect_attention_telemetry=args.collect_attention_telemetry,
                max_samples=args.max_samples,
                donor_states_cache=donor_cache,
                base_states_cache=base_cache,
            )

            if "axis1a" in axes_to_run:
                for cond in [
                    "A1A-MASK-ALL-IMG",
                    "A1A-MASK-ANS-TO-LVR",
                    "A1A-MASK-ANS-TO-IMG",
                    "A1A-ORACLE-ROI",
                    "A1A-ORACLE-BKG",
                    "A1A-2D-MROPE",
                ]:
                    if cond in ("A1A-ORACLE-ROI", "A1A-ORACLE-BKG") and b_name != "vstar":
                        continue
                    evaluate_condition_on_dataset(
                        model=model,
                        processor=processor,
                        dataset=ds,
                        benchmark_name=b_name,
                        condition=cond,
                        lvr_steps=8,
                        out_dir=args.out_dir,
                        prompt_mode="official",
                        collect_attention_telemetry=args.collect_attention_telemetry,
                        max_samples=args.max_samples,
                        donor_states_cache=donor_cache,
                        base_states_cache=base_cache,
                    )

            if "axis3a" in axes_to_run:
                for k_val in [0, 1, 2, 4, 8, 16, 32, 64]:
                    evaluate_condition_on_dataset(
                        model=model,
                        processor=processor,
                        dataset=ds,
                        benchmark_name=b_name,
                        condition=f"A3A-STEPS-{k_val}",
                        lvr_steps=k_val,
                        out_dir=args.out_dir,
                        prompt_mode="official",
                        collect_attention_telemetry=False,
                        max_samples=args.max_samples,
                    )

            if "axis3b" in axes_to_run:
                for cond in ["A3B-FORCED-END", "A3B-VANILLA-LOGIT", "A3B-LATENT-END"]:
                    evaluate_condition_on_dataset(
                        model=model,
                        processor=processor,
                        dataset=ds,
                        benchmark_name=b_name,
                        condition=cond,
                        lvr_steps=8,
                        out_dir=args.out_dir,
                        prompt_mode="official",
                        collect_attention_telemetry=False,
                        max_samples=args.max_samples,
                    )

            if "axis4" in axes_to_run:
                for cond, k_val in [
                    ("C2-EMPTY-TAG", 0),
                    ("C3-DISCRETE-LVR", 8),
                    ("C4-MEAN-IMG-POOL", 8),
                    ("C5-NN-QUANT-IMG", 8),
                    ("C6-ZERO-VEC", 8),
                    ("C7-GAUSS-NOISE", 8),
                    ("C8-SHUFFLE-BATCH", 8),
                    ("C9-DIRECT-ANS", 0),
                ]:
                    evaluate_condition_on_dataset(
                        model=model,
                        processor=processor,
                        dataset=ds,
                        benchmark_name=b_name,
                        condition=cond,
                        lvr_steps=k_val,
                        out_dir=args.out_dir,
                        prompt_mode="official",
                        collect_attention_telemetry=False,
                        max_samples=args.max_samples,
                        donor_states_cache=donor_cache,
                        base_states_cache=base_cache,
                    )

            if "axis5" in axes_to_run:
                for p_mode in ["clean", "bbox_aligned"]:
                    evaluate_condition_on_dataset(
                        model=model,
                        processor=processor,
                        dataset=ds,
                        benchmark_name=b_name,
                        condition="A1A-BASE",
                        lvr_steps=8,
                        out_dir=args.out_dir,
                        prompt_mode=p_mode,
                        collect_attention_telemetry=False,
                        max_samples=args.max_samples,
                    )
                if b_name in ("vstar", "mmvp"):
                    evaluate_circular_eval_on_dataset(
                        model=model,
                        processor=processor,
                        dataset=ds,
                        benchmark_name=b_name,
                        condition="A1A-BASE",
                        lvr_steps=8,
                        out_dir=args.out_dir,
                        max_samples=args.max_samples,
                        is_base_model=False,
                    )

        del model
        torch.cuda.empty_cache()

    if "axis2_base" in axes_to_run or "axis5" in axes_to_run:
        # Evaluate Base Qwen2.5-VL-7B-Instruct
        base7b, proc7b = load_base_qwen_model("Qwen/Qwen2.5-VL-7B-Instruct", attn_impl="sdpa")
        for b_name, ds in datasets_map.items():
            evaluate_condition_on_dataset(
                model=base7b,
                processor=proc7b,
                dataset=ds,
                benchmark_name=b_name,
                condition="BASE-QWEN25-VL-7B",
                lvr_steps=0,
                out_dir=args.out_dir,
                prompt_mode="official",
                max_samples=args.max_samples,
                is_base_model=True,
            )
            if "axis5" in axes_to_run:
                evaluate_condition_on_dataset(
                    model=base7b,
                    processor=proc7b,
                    dataset=ds,
                    benchmark_name=b_name,
                    condition="BASE-QWEN25-VL-7B",
                    lvr_steps=0,
                    out_dir=args.out_dir,
                    prompt_mode="clean",
                    max_samples=args.max_samples,
                    is_base_model=True,
                )
        del base7b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

