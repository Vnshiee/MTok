#!/usr/bin/env python3
"""
Master Report Aggregator: Experiments 1 to 5 Consolidated Analysis.
"""

import os
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "eval_outputs")
    master_file = os.path.join(base_dir, "ALL_EXPERIMENTS_MASTER_SUMMARY.md")

    exp_files = [
        ("Experiment 1: Causal Necessity & Latent Ablation", os.path.join(base_dir, "exp1_causal_ablation/EXPERIMENT1_RESULTS_SUMMARY.md")),
        ("Experiment 2: Stage 2 SFT vs. Stage 3 GRPO", os.path.join(base_dir, "exp2_stage3_grpo/EXPERIMENT2_RESULTS_SUMMARY.md")),
        ("Experiment 3: Confounder Test (SAT Baseline vs. Mull)", os.path.join(base_dir, "exp3_sat_baseline/EXPERIMENT3_RESULTS_SUMMARY.md")),
        ("Experiment 4: Thinking Token Budget Scaling Curve", os.path.join(base_dir, "exp4_length_scaling/EXPERIMENT4_RESULTS_SUMMARY.md")),
        ("Experiment 5: Latent Dynamics & Representational Drift", os.path.join(base_dir, "exp5_latent_dynamics/EXPERIMENT5_LATENT_DYNAMICS_SUMMARY.md")),
    ]

    lines = []
    lines.append("# Master Research Summary: Empirical Evaluation of Mull-Tokens (Ray et al., CVPR 2026)")
    lines.append("")
    lines.append("This document aggregates all findings from Experiments 1 through 5, systematically evaluating the core architectural, optimization, and empirical claims of the Mull-Tokens paper against the critical gaps identified in `FinalGaps.md`.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for title, fpath in exp_files:
        lines.append(f"## {title}")
        lines.append("")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                content = f.read().strip()
            lines.append(content)
        else:
            lines.append(f"*Results pending execution at `{fpath}`.*")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(master_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Master summary updated at: {master_file}")

if __name__ == "__main__":
    main()
