#!/usr/bin/env python3
"""
Experiment 3 Results Aggregator: Mull-Tokens vs. Direct SAT SFT Baseline.
"""

import os
import glob
import json
import sys


def extract_all_metrics_from_dir(dir_path):
    if not os.path.isdir(dir_path):
        return {}
    files = glob.glob(os.path.join(dir_path, "**/results_*.json"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(dir_path, "**/*result*.json"), recursive=True)
    
    all_scores = {}
    for f_path in files:
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
            results = data.get("results", {})
            for task_name, task_metrics in results.items():
                for k, v in task_metrics.items():
                    if isinstance(v, (int, float)) and ("score" in k.lower() or "acc" in k.lower()) and "stderr" not in k.lower():
                        all_scores[task_name] = round(v * 100.0, 2)
                        break
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
    return all_scores


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(root_dir, "eval_outputs/exp3_sat_baseline")
    output_summary_file = os.path.join(base_dir, "EXPERIMENT3_RESULTS_SUMMARY.md")

    # Baseline Mull-Tokens run
    mull_dir = os.path.join(root_dir, "eval_outputs/core_benchmarks_20261005_155255")
    sat_dir = base_dir

    mull_scores = extract_all_metrics_from_dir(mull_dir)
    sat_scores = extract_all_metrics_from_dir(sat_dir)

    display_tasks = [
        ("blink_sprel", "BLINK Spatial Relation (143)"),
        ("blink_reldepth", "BLINK Relative Depth (124)"),
        ("blink_jigsaw", "BLINK Jigsaw Puzzles (150)"),
        ("blink_mv", "BLINK Multi-View (133)"),
        ("blink_iqtest", "BLINK IQ Test (150)"),
    ]

    lines = []
    lines.append("# Experiment 3: Confounding Data Leakage Test (Direct SAT SFT vs. Mull-Tokens)")
    lines.append("")
    lines.append("**Goal**: Disentangle the effect of the Mull latent mechanism from the effect of heavy domain post-training on the SAT dataset. Does `array/Qwen2.5-VL-Mull` outperform standard direct fine-tuning (`array/Qwen2.5-VL-SAT`) on the exact same data distribution?")
    lines.append("")
    lines.append("| Benchmark Task | Mull-Tokens (`array/Qwen2.5-VL-Mull`) | Direct SFT Baseline (`array/Qwen2.5-VL-SAT`) | Difference (Mull Advantage) |")
    lines.append("| :--- | :---: | :---: | :---: |")

    m_vals, s_vals = [], []

    for task_key, task_label in display_tasks:
        m_val = mull_scores.get(task_key, None)
        s_val = sat_scores.get(task_key, None)

        m_str = f"**{m_val:.2f}%**" if m_val is not None else "N/A"
        if s_val is None:
            s_str = "Pending..."
            d_str = "—"
        else:
            diff = m_val - s_val if m_val is not None else 0
            sign = "+" if diff >= 0 else ""
            s_str = f"**{s_val:.2f}%**"
            d_str = f"{sign}{diff:.2f}%"

        lines.append(f"| **{task_label}** | {m_str} | {s_str} | {d_str} |")
        if m_val is not None: m_vals.append(m_val)
        if s_val is not None: s_vals.append(s_val)

    if m_vals:
        avg_m = sum(m_vals) / len(m_vals)
        if len(s_vals) == len(m_vals):
            avg_s = sum(s_vals) / len(s_vals)
            diff = avg_m - avg_s
            sign = "+" if diff >= 0 else ""
            lines.append(f"| **BLINK Average (5 Tasks)** | **{avg_m:.2f}%** | **{avg_s:.2f}%** | **{sign}{diff:.2f}%** |")
        else:
            lines.append(f"| **BLINK Average (5 Tasks)** | **{avg_m:.2f}%** | Pending... | — |")

    lines.append("")
    summary_text = "\n".join(lines)
    os.makedirs(base_dir, exist_ok=True)
    with open(output_summary_file, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nSaved Exp 3 report to: {output_summary_file}")


if __name__ == "__main__":
    main()
