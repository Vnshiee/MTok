#!/usr/bin/env python3
"""
Experiment 1 Results Aggregator and Formatter.

Parses outputs from all Experiment 1 ablation conditions, computes delta comparisons
relative to the intact Mull-Tokens baseline, and outputs a formatted Markdown summary.
"""

import os
import glob
import json
import sys


from eval_metric_utils import extract_blink_metrics, BLINK_EXPECTED_N


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(root_dir, "eval_outputs/exp1_causal_ablation")
    output_summary_file = os.path.join(base_dir, "EXPERIMENT1_RESULTS_SUMMARY.md")

    # Baseline replicated run
    baseline_dir = os.path.join(root_dir, "eval_outputs/core_benchmarks_20261005_155255")
    cond_k0_dir = os.path.join(base_dir, "cond_zero_latents_K0")
    cond_zeros_dir = os.path.join(base_dir, "cond_zeroed_embeddings")
    cond_noise_dir = os.path.join(base_dir, "cond_random_noise_embeddings")
    attn_json = os.path.join(base_dir, "exp1_attention_distribution.json")

    baseline_scores = extract_blink_metrics(baseline_dir, require_all=True)
    k0_scores = extract_blink_metrics(cond_k0_dir, require_all=True)
    zeros_scores = extract_blink_metrics(cond_zeros_dir, require_all=True)
    noise_scores = extract_blink_metrics(cond_noise_dir, require_all=True)

    print(f"Loaded scores:")
    print(f"  Baseline: {baseline_scores}")
    print(f"  K=0:      {k0_scores}")
    print(f"  Zeros:    {zeros_scores}")
    print(f"  Noise:    {noise_scores}")

    display_tasks = [
        ("blink_sprel", "BLINK Spatial Relation (143)"),
        ("blink_reldepth", "BLINK Relative Depth (124)"),
        ("blink_jigsaw", "BLINK Jigsaw Puzzles (150)"),
        ("blink_mv", "BLINK Multi-View (133)"),
        ("blink_iqtest", "BLINK IQ Test (150)"),
    ]

    lines = []
    lines.append("# Experiment 1: Causal Necessity and Latent Ablation Results")
    lines.append("")
    lines.append("**Goal**: Rigorously test whether the 20 `<|latent_pad|>` tokens perform causal reasoning or function as an ungrounded shortcut / placebo.")
    lines.append("")
    lines.append("### Evaluated Conditions:")
    lines.append("1. **Baseline**: Intact Mull-Tokens ($K=20$) with full learned latent embeddings.")
    lines.append("2. **Zero Latents ($K=0$)**: Complete removal of latent thinking tokens (direct answer without `<think>` tags).")
    lines.append("3. **Zeroed Embeddings ($\mathbf{0}$)**: 20 latent token positions retained, but input embedding vector zeroed out.")
    lines.append("4. **Random Noise Embeddings ($\mathcal{N}$)**: 20 latent token positions retained, but filled with Gaussian noise.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### Benchmark Comparison Table (BLINK Spatial Suite)")
    lines.append("")
    lines.append("| Benchmark Task | Baseline ($K=20$ Intact) | Zero Latents ($K=0$) | Zeroed Embeddings ($\mathbf{0}$) | Random Noise ($\mathcal{N}$) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    b_vals, k0_vals, zeros_vals, noise_vals = [], [], [], []

    for task_key, task_label in display_tasks:
        b_val = baseline_scores.get(task_key, None)
        k0_val = k0_scores.get(task_key, None)
        z_val = zeros_scores.get(task_key, None)
        n_val = noise_scores.get(task_key, None)

        def fmt_cell(val, ref):
            if val is None:
                return "Pending..."
            if ref is None:
                return f"{val:.2f}%"
            diff = val - ref
            sign = "+" if diff >= 0 else ""
            return f"**{val:.2f}%** ({sign}{diff:.2f}%)"

        b_str = f"**{b_val:.2f}%**" if b_val is not None else "N/A"
        k0_str = fmt_cell(k0_val, b_val)
        z_str = fmt_cell(z_val, b_val)
        n_str = fmt_cell(n_val, b_val)

        lines.append(f"| **{task_label}** | {b_str} | {k0_str} | {z_str} | {n_str} |")

        if b_val is not None: b_vals.append(b_val)
        if k0_val is not None: k0_vals.append(k0_val)
        if z_val is not None: zeros_vals.append(z_val)
        if n_val is not None: noise_vals.append(n_val)

    if b_vals:
        avg_b = sum(b_vals) / len(b_vals)
        avg_k0_str = f"**{sum(k0_vals)/len(k0_vals):.2f}%** ({('+' if sum(k0_vals)/len(k0_vals)-avg_b>=0 else '')}{sum(k0_vals)/len(k0_vals)-avg_b:.2f}%)" if len(k0_vals) == len(b_vals) else "Pending..."
        avg_z_str = f"**{sum(zeros_vals)/len(zeros_vals):.2f}%** ({('+' if sum(zeros_vals)/len(zeros_vals)-avg_b>=0 else '')}{sum(zeros_vals)/len(zeros_vals)-avg_b:.2f}%)" if len(zeros_vals) == len(b_vals) else "Pending..."
        avg_n_str = f"**{sum(noise_vals)/len(noise_vals):.2f}%** ({('+' if sum(noise_vals)/len(noise_vals)-avg_b>=0 else '')}{sum(noise_vals)/len(noise_vals)-avg_b:.2f}%)" if len(noise_vals) == len(b_vals) else "Pending..."
        lines.append(f"| **BLINK Average (5 Tasks)** | **{avg_b:.2f}%** | {avg_k0_str} | {avg_z_str} | {avg_n_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Load Attention Allocation if present
    if os.path.exists(attn_json):
        lines.append("### Mechanistic Interpretability: Attention Allocation Analysis")
        lines.append("")
        with open(attn_json, "r") as f:
            attn_data = json.load(f)
        lines.append("Attention percentage allocated from the generated `<answer>` token back to preceding token groups:")
        lines.append("")
        lines.append(r"| Sample Task | Visual Patches ($\alpha_{\text{vis}}$) | Question Text ($\alpha_{\text{txt}}$) | Mull Latents ($\alpha_{\text{mull}}$) | Early Layers (1-9) | Mid Layers (10-19) | Late Decision Layers (20-28) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in attn_data:
            s_id = s.get("sample_id", "")
            ov = s.get("overall_attention", {})
            by_d = s.get("mull_attention_by_depth", {})
            lines.append(f"| `{s_id}` | **{ov.get('image_tokens_pct', 0)}%** | **{ov.get('text_tokens_pct', 0)}%** | **{ov.get('mull_tokens_pct', 0)}%** | {by_d.get('early_layers_1_to_9', 0)}% | {by_d.get('mid_layers_10_to_19', 0)}% | **{by_d.get('late_decision_layers_20_to_28', 0)}%** |")
        lines.append("")

    summary_text = "\n".join(lines)
    os.makedirs(base_dir, exist_ok=True)
    with open(output_summary_file, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"\nSaved summary report to: {output_summary_file}")


if __name__ == "__main__":
    main()
