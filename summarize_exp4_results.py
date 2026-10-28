#!/usr/bin/env python3
"""
Experiment 4 Results Aggregator: Thinking Token Budget Scaling (K in {0, 1, 5, 10, 20, 40}).
"""

import os
import glob
import json
import sys


from eval_metric_utils import extract_blink_metrics, BLINK_EXPECTED_N


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(root_dir, "eval_outputs/exp4_length_scaling")
    output_summary_file = os.path.join(base_dir, "EXPERIMENT4_RESULTS_SUMMARY.md")

    # K=0 from Exp 1
    k0_dir = os.path.join(root_dir, "eval_outputs/exp1_causal_ablation/cond_zero_latents_K0")
    # K=20 from baseline
    k20_dir = os.path.join(root_dir, "eval_outputs/core_benchmarks_20261005_155255")

    k_dirs = {
        0: k0_dir,
        1: os.path.join(base_dir, "K_1"),
        5: os.path.join(base_dir, "K_5"),
        10: os.path.join(base_dir, "K_10"),
        20: k20_dir,
        40: os.path.join(base_dir, "K_40"),
    }

    scores_by_k = {}
    for k, d in k_dirs.items():
        scores_by_k[k] = extract_blink_metrics(d, require_all=True)

    display_tasks = [
        ("blink_sprel", "BLINK Spatial Relation (143)"),
        ("blink_reldepth", "BLINK Relative Depth (124)"),
        ("blink_jigsaw", "BLINK Jigsaw Puzzles (150)"),
        ("blink_mv", "BLINK Multi-View (133)"),
        ("blink_iqtest", "BLINK IQ Test (150)"),
    ]

    lines = []
    lines.append("# Experiment 4: Thinking Token Budget Scaling Curve")
    lines.append("")
    lines.append("**Goal**: Evaluate whether reasoning capability scales dynamically with latent token count ($K \\in \\{0, 1, 5, 10, 20, 40\\}$), or whether performance degrades away from the hardcoded $K=20$ training setting, exposing static prefix overfitting.")
    lines.append("")
    lines.append("| Benchmark Task | K=0 (Direct) | K=1 | K=5 | K=10 | K=20 (Default) | K=40 (Deep) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    k_list = [0, 1, 5, 10, 20, 40]
    avg_by_k = {k: [] for k in k_list}

    for task_key, task_label in display_tasks:
        row_cells = []
        for k in k_list:
            val = scores_by_k[k].get(task_key, None)
            if val is not None:
                row_cells.append(f"**{val:.2f}%**")
                avg_by_k[k].append(val)
            else:
                row_cells.append("Pending...")
        lines.append(f"| **{task_label}** | " + " | ".join(row_cells) + " |")

    # Averages
    avg_cells = []
    for k in k_list:
        vals = avg_by_k[k]
        if len(vals) == len(display_tasks):
            avg_cells.append(f"**{sum(vals)/len(vals):.2f}%**")
        elif len(vals) > 0:
            avg_cells.append(f"{sum(vals)/len(vals):.2f}%*")
        else:
            avg_cells.append("Pending...")
    lines.append(f"| **BLINK Average (5 Tasks)** | " + " | ".join(avg_cells) + " |")

    lines.append("")
    summary_text = "\n".join(lines)
    os.makedirs(base_dir, exist_ok=True)
    with open(output_summary_file, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nSaved Exp 4 report to: {output_summary_file}")


if __name__ == "__main__":
    main()
