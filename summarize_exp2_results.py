#!/usr/bin/env python3
"""
Experiment 2 Results Aggregator: Stage 2 SFT vs. Stage 3 GRPO Head-to-Head.
"""

import os
import glob
import json
import sys


from eval_metric_utils import extract_blink_metrics, BLINK_EXPECTED_N


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(root_dir, "eval_outputs/exp2_stage3_grpo")
    output_summary_file = os.path.join(base_dir, "EXPERIMENT2_RESULTS_SUMMARY.md")

    # Baseline Stage 2 SFT run
    baseline_dir = os.path.join(root_dir, "eval_outputs/core_benchmarks_20261005_155255")
    grpo_dir = base_dir

    baseline_scores = extract_blink_metrics(baseline_dir, require_all=True)
    grpo_scores = extract_blink_metrics(grpo_dir, require_all=True)

    display_tasks = [
        ("blink_sprel", "BLINK Spatial Relation (143)"),
        ("blink_reldepth", "BLINK Relative Depth (124)"),
        ("blink_jigsaw", "BLINK Jigsaw Puzzles (150)"),
        ("blink_mv", "BLINK Multi-View (133)"),
        ("blink_iqtest", "BLINK IQ Test (150)"),
    ]

    lines = []
    lines.append("# Experiment 2: Stage 2 SFT vs. Stage 3 GRPO Head-to-Head Results")
    lines.append("")
    lines.append("**Goal**: Evaluate whether 50,000 steps of Reinforcement Learning via GRPO on 8x H100 GPUs actually yielded real reasoning improvements over Stage 2 SFT, or if group size $G=2$ led to spurious variance and dead updates.")
    lines.append("")
    lines.append("| Benchmark Task | Stage 2 SFT (`array/Qwen2.5-VL-Mull`) | Stage 3 GRPO (`array/Qwen2.5-VL-MullGRPO`) | Delta (GRPO vs. SFT) |")
    lines.append("| :--- | :---: | :---: | :---: |")

    b_vals, g_vals = [], []

    for task_key, task_label in display_tasks:
        b_val = baseline_scores.get(task_key, None)
        g_val = grpo_scores.get(task_key, None)

        b_str = f"**{b_val:.2f}%**" if b_val is not None else "N/A"
        if g_val is None:
            g_str = "Pending..."
            d_str = "—"
        else:
            diff = g_val - b_val if b_val is not None else 0
            sign = "+" if diff >= 0 else ""
            g_str = f"**{g_val:.2f}%**"
            d_str = f"{sign}{diff:.2f}%"

        lines.append(f"| **{task_label}** | {b_str} | {g_str} | {d_str} |")
        if b_val is not None: b_vals.append(b_val)
        if g_val is not None: g_vals.append(g_val)

    if b_vals:
        avg_b = sum(b_vals) / len(b_vals)
        if len(g_vals) == len(b_vals):
            avg_g = sum(g_vals) / len(g_vals)
            diff = avg_g - avg_b
            sign = "+" if diff >= 0 else ""
            lines.append(f"| **BLINK Average (5 Tasks)** | **{avg_b:.2f}%** | **{avg_g:.2f}%** | **{sign}{diff:.2f}%** |")
        else:
            lines.append(f"| **BLINK Average (5 Tasks)** | **{avg_b:.2f}%** | Pending... | — |")

    lines.append("")
    summary_text = "\n".join(lines)
    os.makedirs(base_dir, exist_ok=True)
    with open(output_summary_file, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nSaved Exp 2 report to: {output_summary_file}")


if __name__ == "__main__":
    main()
