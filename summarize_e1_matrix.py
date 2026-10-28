#!/usr/bin/env python3
"""
summarize_e1_matrix.py: Compile results for the 9-condition E1 Experimental Matrix.

Computes:
1. Exact BLINK scores per task and overall (N=700).
2. Paired McNemar's tests against Row 1 (Reference baseline).
3. 10,000-iteration bootstrap 95% confidence intervals.
4. Generated length statistics and K=0 forensic analysis.
"""

import os
import glob
import json
import numpy as np
from scipy import stats
from eval_metric_utils import BLINK_EXPECTED_N, BLINK_METRIC_KEY, extract_blink_metrics

ROWS_CONFIG = [
    {"row": 1, "dir": "row1_mull_ref", "model": "Mull", "prompt": "latents", "prefill": "think + 20 latents", "notes": "Clean reference baseline"},
    {"row": 2, "dir": "row2_mull_empty", "model": "Mull", "prompt": "latents", "prefill": "empty", "notes": "K=0 unconstrained reasoning"},
    {"row": 3, "dir": "row3_mull_empty_think", "model": "Mull", "prompt": "latents", "prefill": "<think></think>\\n", "notes": "Empty think tags"},
    {"row": 4, "dir": "row4_mull_dot_pause", "model": "Mull", "prompt": "latents", "prefill": "think + 20x' .'", "notes": "Generic pause tokens"},
    {"row": 5, "dir": "row5_mull_latent_end", "model": "Mull", "prompt": "latents", "prefill": "think + 20x<|latent_end|>", "notes": "Special token control"},
    {"row": 6, "dir": "row6_mull_sft_empty", "model": "Mull", "prompt": "sft", "prefill": "empty", "notes": "Prompt sensitivity (Mull un-prefilled)"},
    {"row": 7, "dir": "row7_mull_sft_latents", "model": "Mull", "prompt": "sft", "prefill": "think + 20 latents", "notes": "Mull with SFT prompt"},
    {"row": 8, "dir": "row8_sat_sft_empty", "model": "SAT", "prompt": "sft", "prefill": "empty", "notes": "Direct SAT SFT baseline (DirAns)"},
    {"row": 9, "dir": "row9_sat_latents_empty", "model": "SAT", "prompt": "latents", "prefill": "empty", "notes": "SAT with latent prompt"},
]

TASKS = ["blink_sprel", "blink_reldepth", "blink_jigsaw", "blink_mv", "blink_iqtest"]
REASONING_TASKS = ["blink_reldepth", "blink_mv", "blink_iqtest"]

def load_row_results(base_dir, row_info):
    row_dir = os.path.join(base_dir, row_info["dir"])
    if not os.path.exists(row_dir):
        return None

    try:
        scores = extract_blink_metrics(row_dir, require_all=False)
    except Exception:
        return None

    if not scores:
        return None

    total_correct = sum(round((scores[t] / 100.0) * BLINK_EXPECTED_N[t]) for t in scores)
    total_samples = sum(BLINK_EXPECTED_N[t] for t in scores)

    overall_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    reasoning_tasks_present = [t for t in REASONING_TASKS if t in scores]
    reasoning_acc = np.mean([scores[t] for t in reasoning_tasks_present]) if reasoning_tasks_present else 0.0

    return {
        **row_info,
        "scores": scores,
        "overall": overall_acc,
        "reasoning": reasoning_acc,
        "total_samples": total_samples,
    }

def load_sample_details(base_dir, row_info):
    row_dir = os.path.join(base_dir, row_info["dir"])
    log_dir = os.path.join(row_dir, "per_sample_logs")
    if not os.path.exists(log_dir):
        log_dir = row_dir

    sample_dict = {}
    for task in TASKS:
        files = glob.glob(os.path.join(log_dir, f"per_sample_details_{task}*.jsonl"))
        if not files:
            continue
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = (item["task"], str(item["doc_id"]))
                    sample_dict[key] = item
    return sample_dict

def load_lmms_samples(base_dir, row_info):
    row_dir = os.path.join(base_dir, row_info["dir"])
    sample_dict = {}
    for task in TASKS:
        files = glob.glob(os.path.join(row_dir, "**", f"*samples_{task}*.jsonl"), recursive=True)
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    doc_id = item.get("doc_id", item.get("id"))
                    key = (task, str(doc_id))
                    target = item.get("target", "")
                    resps = item.get("filtered_resps")
                    if resps and isinstance(resps, list):
                        pred = resps[0]
                    else:
                        pred = item.get("filtered_res", "")
                    item["is_correct"] = int(pred == target)
                    sample_dict[key] = item
    return sample_dict

def compute_mcnemar(correct_ref, correct_target):
    # b: ref=1, target=0
    # c: ref=0, target=1
    b = np.sum((correct_ref == 1) & (correct_target == 0))
    c = np.sum((correct_ref == 0) & (correct_target == 1))
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_val": 1.0, "significant": False}

    if n_discordant < 25:
        # Exact two-sided binomial test
        res = stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
        p_val = res.pvalue
        chi2 = ((abs(b - c) - 1.0) ** 2) / n_discordant
    else:
        # Continuity-corrected chi-squared
        chi2 = ((abs(b - c) - 1.0) ** 2) / n_discordant
        p_val = 1.0 - stats.chi2.cdf(chi2, df=1)

    return {
        "b": int(b),
        "c": int(c),
        "chi2": float(chi2),
        "p_val": float(p_val),
        "significant": bool(p_val < 0.05)
    }

def compute_bootstrap_ci(correct_arr, n_bootstraps=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(correct_arr)
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(correct_arr[indices], axis=1) * 100.0
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    return float(ci_lower), float(ci_upper)

def compute_delta_bootstrap_ci(correct_ref, correct_target, n_bootstraps=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(correct_ref)
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    deltas = (np.mean(correct_target[indices], axis=1) - np.mean(correct_ref[indices], axis=1)) * 100.0
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    return float(ci_lower), float(ci_upper)

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_outputs", "e1_matrix")
    results = []

    print(f"Scanning E1 matrix results in: {base_dir}")
    for row in ROWS_CONFIG:
        res = load_row_results(base_dir, row)
        if res is not None:
            results.append(res)
            print(f"  [Found] Row {row['row']}: {row['dir']} -> Overall: {res['overall']:.2f}%")
        else:
            print(f"  [Pending] Row {row['row']}: {row['dir']}")

    if not results:
        print("\nNo complete E1 matrix runs found yet. Run `bash run_e1_matrix.sh` to generate data.")
        return

    # Generate Markdown Summary
    out_md = os.path.join(base_dir, "E1_MATRIX_SUMMARY.md")
    lines = [
        "# Experiment E1: Mechanistic Matrix Results",
        f"**Standardized Image Resolution:** 4.19 MP (max_pixels=4194304)  ",
        f"**Expected Benchmark Total:** N=700 across 5 BLINK tasks  \n",
        "## 1. Overall Performance Matrix\n",
        "| # | Model | Prompt | Assistant Prefill | SpRel | RelDepth | Jigsaw | Multi-View | IQ-Test | Reasoning Avg | BLINK Overall | Delta vs Row 1 |",
        "|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    ref_row = next((r for r in results if r["row"] == 1), None)

    for r in sorted(results, key=lambda x: x["row"]):
        delta_str = "-"
        if ref_row and r["row"] != 1:
            diff = r["overall"] - ref_row["overall"]
            delta_str = f"{diff:+.2f}%"
        elif r["row"] == 1:
            delta_str = "Reference"

        lines.append(
            f"| **{r['row']}** | {r['model']} | `{r['prompt']}` | `{r['prefill']}` | "
            f"{r['scores'].get('blink_sprel', 0):.2f}% | "
            f"{r['scores'].get('blink_reldepth', 0):.2f}% | "
            f"{r['scores'].get('blink_jigsaw', 0):.2f}% | "
            f"{r['scores'].get('blink_mv', 0):.2f}% | "
            f"{r['scores'].get('blink_iqtest', 0):.2f}% | "
            f"**{r['reasoning']:.2f}%** | "
            f"**{r['overall']:.2f}%** | "
            f"{delta_str} |"
        )

    lines.append("\n## 2. Statistical Significance vs. Row 1 Reference (N=700)\n")
    lines.append("| Target Row | Condition | Overall Acc | Delta (95% CI) | McNemar b/c | p-value | Significant (p<0.05) |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|")

    # Load samples if available for statistical tests
    if ref_row:
        ref_samples = load_lmms_samples(base_dir, ref_row)
        for r in sorted(results, key=lambda x: x["row"]):
            if r["row"] == 1:
                continue
            target_samples = load_lmms_samples(base_dir, r)
            common_keys = sorted(list(set(ref_samples.keys()) & set(target_samples.keys())))
            if len(common_keys) == 700:
                ref_corr = np.array([ref_samples[k]["is_correct"] for k in common_keys], dtype=int)
                tgt_corr = np.array([target_samples[k]["is_correct"] for k in common_keys], dtype=int)

                mcn = compute_mcnemar(ref_corr, tgt_corr)
                ci_low, ci_high = compute_delta_bootstrap_ci(ref_corr, tgt_corr)
                sig_str = "**YES (Reject H0)**" if mcn["significant"] else "No (p >= 0.05)"

                lines.append(
                    f"| Row {r['row']} | {r['model']} / `{r['prompt']}` / `{r['prefill']}` | "
                    f"{r['overall']:.2f}% | "
                    f"{r['overall'] - ref_row['overall']:+.2f}% [{ci_low:+.2f}%, {ci_high:+.2f}%] | "
                    f"b={mcn['b']}, c={mcn['c']} | "
                    f"p={mcn['p_val']:.4f} | "
                    f"{sig_str} |"
                )
            else:
                lines.append(f"| Row {r['row']} | {r['model']} / `{r['prompt']}` / `{r['prefill']}` | {r['overall']:.2f}% | {r['overall'] - ref_row['overall']:+.2f}% | Samples logged: {len(common_keys)}/700 | - | Pending full samples |")

    # Length Forensics Section
    lines.append("\n## 3. Output Forensics & Generation Length Analysis\n")
    lines.append("| Row | Condition | Mean Gen Tokens | Median Gen Tokens | P95 Gen Tokens | Mean Latency (s/sample) |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|")

    for r in sorted(results, key=lambda x: x["row"]):
        detail_samples = load_sample_details(base_dir, r)
        if detail_samples:
            lengths = [s["gen_tokens_count"] for s in detail_samples.values()]
            latencies = [s["latency_sec"] for s in detail_samples.values()]
            mean_len = np.mean(lengths)
            med_len = np.median(lengths)
            p95_len = np.percentile(lengths, 95)
            mean_lat = np.mean(latencies)
            lines.append(f"| Row {r['row']} | {r['notes']} | {mean_len:.1f} | {med_len:.1f} | {p95_len:.1f} | {mean_lat:.2f}s |")
        else:
            lines.append(f"| Row {r['row']} | {r['notes']} | - | - | - | - |")

    content = "\n".join(lines) + "\n"
    with open(out_md, "w", encoding="utf-8") as fp:
        fp.write(content)

    print(f"\nGenerated E1 Matrix Summary: {out_md}")
    print(content)

if __name__ == "__main__":
    main()
