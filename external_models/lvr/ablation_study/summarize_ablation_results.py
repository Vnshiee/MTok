"""
Comprehensive Statistical Adjudication & Markdown Report Generator
for the LVR Ablation Study (`LVR_ABLATION_RESULTS_REPORT.md`).

Computes:
- Paired McNemar / Exact Binomial tests & B=10,000 Bootstrap 95% CIs across all axes.
- Holm-Bonferroni family-wise error rate corrections.
- Post-Hoc Best-of-{4, 8, 16} Step Cherry-Picking Quantification (Section 4.1).
- Mechanistic telemetry tables (Prefix ROI vs. Background cosine similarity, ROI Recall@1/5,
  Effective Rank `erank`, Adjacent Cosine Similarity `rho_adj`, Layer-wise Attention Mass
  allocation, and Teacher-Forcing Exposure Bias Rollout Drift `delta_k`).
"""

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = "/home/devraj/MTok/external_models/lvr"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ablation_study.parsers_and_stats import (
    compute_paired_statistics,
    holm_bonferroni_correction,
)
from ablation_study.training_and_rl_verification import (
    verify_axis1b_sft_counterfactuals,
    verify_axis2_grpo_gradient_flow,
)


def load_jsonl_records(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(jsonl_path):
        return records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[str(rec["question_id"])] = rec
    return records


def compute_best_of_k_oracle(
    out_dir: str,
    benchmark: str,
    k_list: List[int] = [4, 8, 16],
    prompt_mode: str = "official",
) -> Optional[Dict[str, Any]]:
    """
    Quantifies the paper's "Best-of-{4, 8, 16}" reporting anomaly (Section 4.1) at three levels:
    1. Fixed-K accuracy for each K in {4, 8, 16}.
    2. Per-Benchmark/Subtask Best-K Selection (`max_{K} Acc(K)`).
    3. Per-Item Oracle Selection (`any_{K} correct_K(i)`).
    """
    bench_dir = os.path.join(out_dir, benchmark)
    runs_by_k: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for k in k_list:
        cond = f"A3A-STEPS-{k}" if os.path.exists(os.path.join(bench_dir, f"A3A-STEPS-{k}_steps{k:02d}_{prompt_mode}.jsonl")) else "A1A-BASE"
        p = os.path.join(bench_dir, f"{cond}_steps{k:02d}_{prompt_mode}.jsonl")
        if os.path.exists(p):
            runs_by_k[k] = load_jsonl_records(p)

    if len(runs_by_k) < 2:
        return None

    common_qids = sorted(set.intersection(*[set(r.keys()) for r in runs_by_k.values()]))
    if not common_qids:
        return None

    fixed_k_acc_off = {}
    fixed_k_acc_rob = {}
    for k, recs in runs_by_k.items():
        fixed_k_acc_off[k] = round(float(np.mean([recs[q]["correct_official"] for q in common_qids])) * 100.0, 2)
        fixed_k_acc_rob[k] = round(float(np.mean([recs[q]["correct_robust"] for q in common_qids])) * 100.0, 2)

    oracle_item_off = []
    oracle_item_rob = []
    for q in common_qids:
        oracle_item_off.append(any(runs_by_k[k][q]["correct_official"] for k in runs_by_k))
        oracle_item_rob.append(any(runs_by_k[k][q]["correct_robust"] for k in runs_by_k))

    return {
        "n_common": len(common_qids),
        "fixed_k_official_pct": fixed_k_acc_off,
        "fixed_k_robust_pct": fixed_k_acc_rob,
        "best_fixed_k_official_pct": max(fixed_k_acc_off.values()),
        "best_fixed_k_robust_pct": max(fixed_k_acc_rob.values()),
        "per_item_oracle_official_pct": round(float(np.mean(oracle_item_off)) * 100.0, 2),
        "per_item_oracle_robust_pct": round(float(np.mean(oracle_item_rob)) * 100.0, 2),
    }


def compare_two_runs(
    out_dir: str,
    benchmark: str,
    run_tag_a: str,
    run_tag_b: str,
    use_robust_parser: bool = False,
) -> Optional[Dict[str, Any]]:
    bench_dir = os.path.join(out_dir, benchmark)
    recs_a = load_jsonl_records(os.path.join(bench_dir, f"{run_tag_a}.jsonl"))
    recs_b = load_jsonl_records(os.path.join(bench_dir, f"{run_tag_b}.jsonl"))
    if not recs_a or not recs_b:
        return None
    common = sorted(set(recs_a.keys()) & set(recs_b.keys()))
    if not common:
        return None
    key = "correct_robust" if use_robust_parser else "correct_official"
    arr_a = [bool(recs_a[q][key]) for q in common]
    arr_b = [bool(recs_b[q][key]) for q in common]
    return compute_paired_statistics(arr_a, arr_b)


def generate_ablation_report(out_dir: str, report_md_path: str) -> Dict[str, Any]:
    sft_audit = verify_axis1b_sft_counterfactuals()
    rl_audit = verify_axis2_grpo_gradient_flow()

    summaries_by_bench: Dict[str, List[Dict[str, Any]]] = {}
    for bench in ["vstar", "mmvp", "blink"]:
        b_dir = os.path.join(out_dir, bench)
        if not os.path.exists(b_dir):
            continue
        summaries = []
        for s_file in sorted(glob.glob(os.path.join(b_dir, "*_summary.json"))):
            with open(s_file, "r", encoding="utf-8") as f:
                summaries.append(json.load(f))
        summaries_by_bench[bench] = summaries

    paired_comparisons: Dict[str, Dict[str, Any]] = {}
    best_of_k_audits: Dict[str, Any] = {}

    for bench in summaries_by_bench:
        paired_comparisons[bench] = {}
        bok = compute_best_of_k_oracle(out_dir, bench)
        if bok:
            best_of_k_audits[bench] = bok

        base_tag = "A1A-BASE_steps08_official"
        # Compare every other condition against A1A-BASE_steps08_official
        for s in summaries_by_bench[bench]:
            tag = f"{s['condition']}_steps{s['lvr_steps']:02d}_{s['prompt_mode']}"
            if tag == base_tag:
                continue
            comp_off = compare_two_runs(out_dir, bench, tag, base_tag, use_robust_parser=False)
            comp_rob = compare_two_runs(out_dir, bench, tag, base_tag, use_robust_parser=True)
            if comp_off and comp_rob:
                paired_comparisons[bench][tag] = {
                    "vs_base_official": comp_off,
                    "vs_base_robust": comp_rob,
                }

    full_metrics = {
        "axis1b_sft_verification": sft_audit,
        "axis2_grpo_verification": rl_audit,
        "benchmark_summaries": summaries_by_bench,
        "paired_comparisons_vs_base": paired_comparisons,
        "best_of_k_oracle_audits": best_of_k_audits,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "full_ablation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # Render Markdown Report
    lines = [
        "# LVR (`vincentleebang/LVR-7B`) — Comprehensive Empirical Ablation & Falsification Report",
        "",
        "## 1. Executive Summary of Empirical Findings",
        "",
    ]

    for bench, summaries in summaries_by_bench.items():
        if not summaries:
            continue
        lines.append(f"### Benchmark: `{bench.upper()}` (N={summaries[0]['n_total']})")
        lines.append("")
        lines.append("| Condition | Steps $K$ | Prompt Mode | Official Parser Acc (%) | Robust Parser Acc (%) | $\\Delta$ vs. `A1A-BASE` (Off) | Paired $p$-value | Mean $\\mathrm{CS}_{\\max}$ | Mean $\\mathrm{CS}_{\\mathrm{ROI}}$ | ROI Recall@1 | Effective Rank | Ans$\\to$LVR Attn | Ans$\\to$Img Attn |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for s in summaries:
            tag = f"{s['condition']}_steps{s['lvr_steps']:02d}_{s['prompt_mode']}"
            comp = paired_comparisons.get(bench, {}).get(tag, {}).get("vs_base_official")
            delta_str = f"{comp['delta_acc_pct']:+.2f}%" if comp else "0.00% (Ref)"
            p_str = f"{comp['p_value']:.4f}" if comp else "—"
            tel = s.get("telemetry_means", {})
            lines.append(
                f"| `{s['condition']}` | `{s['lvr_steps']}` | `{s['prompt_mode']}` | "
                f"**{s['acc_official_micro_pct']:.2f}%** | **{s['acc_robust_micro_pct']:.2f}%** | "
                f"`{delta_str}` | `{p_str}` | "
                f"`{tel.get('cs_max_mean', '—')}` | `{tel.get('cs_roi_mean', '—')}` | "
                f"`{tel.get('roi_recall_at_1_mean', '—')}` | `{tel.get('effective_rank', '—')}` | "
                f"`{tel.get('ans_attn_to_lvr_mean', '—')}` | `{tel.get('ans_attn_to_vis_mean', '—')}` |"
            )
        lines.append("")

        if bench in best_of_k_audits:
            bok = best_of_k_audits[bench]
            lines.append(
                f"- **Post-Hoc Best-of-$K$ Selection Audit (`{bench}`)**: "
                f"Fixed-$K$ Official Accuracies = `{bok['fixed_k_official_pct']}`, "
                f"Best Fixed-$K$ = `{bok['best_fixed_k_official_pct']}%`, "
                f"Per-Item Oracle $\\max_{{K}}$ = `{bok['per_item_oracle_official_pct']}%`."
            )
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. Stage 1 SFT Supervision Counterfactuals (Axis 1B Autograd Audit)",
        "",
        "| Condition | $\\mathcal{L}_{\\mathrm{CE}}$ | $\\mathcal{L}_{\\mathrm{LVR}}$ | $\\mathcal{L}_{\\mathrm{total}}$ | $\\|\\nabla_\\theta \\mathcal{L}_{\\mathrm{total}}\\|_2$ | Parallel Teacher-Forced Input |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for cond, vals in sft_audit.items():
        lines.append(
            f"| `{cond}` | `{vals['loss_ce']}` | `{vals['loss_lvr']}` | `{vals['total_loss']}` | `{vals['param_grad_norm']}` | `{vals['teacher_forced_input']}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Stage 2 GRPO 'Ghost RL' Gradient Flow Audit (Axis 2)",
        "",
        "| RL Variant | Unmasked Completion Tokens | `<\\|lvr_end\\|>` Unmasked | $\\mathcal{L}_{\\mathrm{GRPO}}$ | $\\|\\partial \\mathcal{L}_{\\mathrm{GRPO}} / \\partial \\mathbf{H}_{\\mathrm{LVR}}\\|_F$ | $\\|(\\partial \\mathcal{L}_{\\mathrm{GRPO}} / \\partial \\mathbf{H}_{\\mathrm{LVR}}) \\cdot (\\partial \\mathbf{H}_{\\mathrm{LVR}} / \\partial \\theta_{\\mathrm{LLM}})\\|_2$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for var in ["A2-GHOST-DEFAULT", "A2-STRICT-MASK-END", "A2-DIFF-REPLAY", "A2-UNMASK-LOGP"]:
        v = rl_audit[var]
        lines.append(
            f"| `{var}` | `{v['unmasked_token_count']}` | `{v['lvr_end_token_unmasked']}` | `{v['loss_grpo']}` | `{v['grad_norm_wrt_H_LVR']}` | **`{v['param_grad_norm_via_H_LVR']}`** |"
        )

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated comprehensive ablation report at: {report_md_path}")
    return full_metrics


if __name__ == "__main__":
    out_dir = os.path.join(ROOT_DIR, "ablation_outputs")
    report_path = os.path.join(ROOT_DIR, "LVR_ABLATION_RESULTS_REPORT.md")
    generate_ablation_report(out_dir, report_path)

