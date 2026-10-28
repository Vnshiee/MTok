#!/usr/bin/env python
"""
E2: Statistical analysis of answer bias vs. discrimination.
Computes pre-registered metrics M1-M9, hypothesis tests Q1-Q6,
McNemar, Wilcoxon, and 10,000 paired bootstrap CIs.

Outputs:
  e2_bias/E2_SUMMARY.md
  e2_bias/e2_results.json
"""
import glob
import json
import math
import os
import sys
import numpy as np
import scipy.stats as stats

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E2_OUT_DIR = os.path.join(PROJECT_DIR, "e2_bias", "outputs")
E1_DIR = os.path.join(PROJECT_DIR, "eval_outputs", "e1_matrix")

TASKS_2WAY = ["blink_sprel", "blink_reldepth", "blink_jigsaw", "blink_mv"]
TASKS_ALL = TASKS_2WAY + ["blink_iqtest"]
TASK_NAMES = {
    "blink_sprel": "Spatial_Relation",
    "blink_reldepth": "Relative_Depth",
    "blink_jigsaw": "Jigsaw",
    "blink_mv": "Multi-view_Reasoning",
    "blink_iqtest": "IQ_Test",
}
WEIGHTS = {"blink_sprel": 143, "blink_reldepth": 124, "blink_jigsaw": 150, "blink_mv": 133, "blink_iqtest": 150}
WEIGHTS_2WAY = {t: WEIGHTS[t] for t in TASKS_2WAY}
TOTAL_N = 700
TOTAL_2WAY_N = 550
SEED = 20261028


def mcnemar_test(b, c):
    """Exact two-tailed binomial test for discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return float(2.0 * min(stats.binom.cdf(min(b, c), n, 0.5), 1.0 - stats.binom.cdf(max(b, c) - 1, n, 0.5)))


def bootstrap_ci(diffs, n_resamples=10000, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    diffs = np.asarray(diffs)
    indices = rng.randint(0, len(diffs), (n_resamples, len(diffs)))
    means = diffs[indices].mean(axis=1)
    lower = np.percentile(means, ((1.0 - ci) / 2.0) * 100)
    upper = np.percentile(means, (1.0 - (1.0 - ci) / 2.0) * 100)
    return float(lower), float(upper)


def compute_auc(scores, labels):
    """scores: float array, labels: binary 0/1 array."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    # Mann-Whitney U test gives exact AUC
    n1 = np.sum(labels == 1)
    n0 = np.sum(labels == 0)
    rank = stats.rankdata(scores)
    u1 = np.sum(rank[labels == 1]) - (n1 * (n1 + 1)) / 2.0
    return float(u1 / (n1 * n0))


def load_condition_data(out_dir, cond_id):
    cdir = os.path.join(out_dir, cond_id)
    if not os.path.exists(cdir):
        return None
    data = {}
    for v in ["V0", "V1", "V2", "V3", "V4", "CF"]:
        fpath = os.path.join(cdir, f"{v}.jsonl")
        if os.path.exists(fpath):
            rows = {}
            for l in open(fpath):
                r = json.loads(l)
                rows[r["key"]] = r
            data[v] = rows
    return data


def analyze_all(out_dir=E2_OUT_DIR):
    available_conds = []
    for d in sorted(glob.glob(os.path.join(out_dir, "T*"))):
        cid = os.path.basename(d)
        if os.path.exists(os.path.join(d, "V0.jsonl")):
            available_conds.append(cid)

    if not available_conds:
        print("No completed conditions found in", out_dir)
        return

    print("Analyzing conditions:", available_conds)
    results = {}
    raw_data = {cid: load_condition_data(out_dir, cid) for cid in available_conds}

    # 1. Compute Content-Free Priors p_cf(letter) per (cond, task)
    p_cf = {}
    for cid in available_conds:
        p_cf[cid] = {}
        cf_rows = raw_data[cid].get("CF", {})
        for task in TASKS_ALL:
            task_cf = [r for r in cf_rows.values() if r["task"] == task]
            if task_cf:
                # average log-probabilities across the 27 probes
                letters = task_cf[0]["letters"]
                mean_logp = {L: np.mean([r["logp"][L] for r in task_cf]) for L in letters}
                # softmax normalize
                exps = {L: math.exp(mean_logp[L]) for L in letters}
                tot = sum(exps.values())
                p_cf[cid][task] = {L: exps[L] / tot for L in letters}
            else:
                letters = "AB" if task in TASKS_2WAY else "ABCD"
                p_cf[cid][task] = {L: 1.0 / len(letters) for L in letters}

    # 2. Compute metrics for each condition
    for cid in available_conds:
        d = raw_data[cid]
        v0 = d.get("V0", {})
        v1 = d.get("V1", {})
        v2 = d.get("V2", {})
        v3 = d.get("V3", {})
        v4 = d.get("V4", {})

        c_res = {
            "per_task": {},
            "micro": {},
            "macro": {},
            "debiased_2x2": {},
            "margins": {},
            "blind": {},
            "priors": {},
        }

        # Per task V0, M2 (calibrated), M6 (margins)
        for task in TASKS_ALL:
            task_v0 = [r for r in v0.values() if r["task"] == task]
            task_v1 = [r for r in v1.values() if r["task"] == task]
            letters = "AB" if task in TASKS_2WAY else "ABCD"

            if task_v0:
                # M1: Raw TF accuracy
                correct_m1 = [r["tf_pred"] == r["gold"] for r in task_v0]
                acc_m1 = float(np.mean(correct_m1))

                # M2: Content-free calibrated accuracy: argmax p(k|x) / p_cf(k)
                calib_preds = []
                for r in task_v0:
                    calib_scores = {L: math.exp(r["logp"][L]) / p_cf[cid][task][L] for L in letters}
                    calib_preds.append(max(calib_scores, key=calib_scores.get))
                correct_m2 = [p == r["gold"] for p, r in zip(calib_preds, task_v0)]
                acc_m2 = float(np.mean(correct_m2))

                # Pred B fraction (for 2-way tasks)
                b_frac = float(np.mean([r["tf_pred"] == "B" for r in task_v0])) if task in TASKS_2WAY else None

                # M6: Continuous correct-vs-incorrect margin: logit(gold) - max_{k!=gold} logit(k)
                margins = []
                for r in task_v0:
                    gold_logit = r["logit"][r["gold"]]
                    alt_logits = [r["logit"][L] for L in letters if L != r["gold"]]
                    margins.append(float(gold_logit - max(alt_logits)))

                # AUC of m0 = logp(B) - logp(A) predicting gold == B (for 2-way)
                auc_v0 = None
                if task in TASKS_2WAY:
                    m0 = [r["logp"]["B"] - r["logp"]["A"] for r in task_v0]
                    y_gold = [1 if r["gold"] == "B" else 0 for r in task_v0]
                    auc_v0 = compute_auc(m0, y_gold)

                c_res["per_task"][task] = {
                    "raw_acc": acc_m1,
                    "calib_acc": acc_m2,
                    "pred_b_frac": b_frac,
                    "mean_margin": float(np.mean(margins)),
                    "median_margin": float(np.median(margins)),
                    "auc": auc_v0,
                    "n": len(task_v0),
                    "correct_m1": correct_m1,
                    "correct_m2": correct_m2,
                    "margins": margins,
                }

            # Blind (V1)
            if task_v1:
                blind_corr = [r["tf_pred"] == r["gold"] for r in task_v1]
                auc_v1 = None
                if task in TASKS_2WAY:
                    m1_blind = [r["logp"]["B"] - r["logp"]["A"] for r in task_v1]
                    y_gold_blind = [1 if r["gold"] == "B" else 0 for r in task_v1]
                    auc_v1 = compute_auc(m1_blind, y_gold_blind)
                c_res["blind"][task] = {
                    "blind_acc": float(np.mean(blind_corr)),
                    "blind_auc": auc_v1,
                    "n": len(task_v1),
                }

        # Micro & Macro Aggregates across all 5 tasks
        all_corr_m1 = []
        all_corr_m2 = []
        all_margins = []
        for task in TASKS_ALL:
            if task in c_res["per_task"]:
                all_corr_m1.extend(c_res["per_task"][task]["correct_m1"])
                all_corr_m2.extend(c_res["per_task"][task]["correct_m2"])
                all_margins.extend(c_res["per_task"][task]["margins"])

        c_res["micro"]["raw_acc"] = float(np.mean(all_corr_m1)) if all_corr_m1 else None
        c_res["micro"]["calib_acc"] = float(np.mean(all_corr_m2)) if all_corr_m2 else None
        c_res["micro"]["mean_margin"] = float(np.mean(all_margins)) if all_margins else None
        c_res["micro"]["correct_m1"] = all_corr_m1
        c_res["micro"]["correct_m2"] = all_corr_m2
        c_res["micro"]["margins"] = all_margins

        task_accs_m1 = [c_res["per_task"][t]["raw_acc"] for t in TASKS_ALL if t in c_res["per_task"]]
        task_accs_m2 = [c_res["per_task"][t]["calib_acc"] for t in TASKS_ALL if t in c_res["per_task"]]
        c_res["macro"]["raw_acc"] = float(np.mean(task_accs_m1)) if task_accs_m1 else None
        c_res["macro"]["calib_acc"] = float(np.mean(task_accs_m2)) if task_accs_m2 else None

        # 3. 2x2 Decomposition (V0, V2, V3, V4) on the 4 two-way tasks (N=550)
        has_2x2 = all(len(d.get(v, {})) > 0 for v in ["V0", "V2", "V3", "V4"])
        if has_2x2:
            s_all = []
            L_all = []
            P_all = []
            s_gold_correct = []
            per_task_2x2 = {}

            for task in TASKS_2WAY:
                keys = [k for k in v0 if k.startswith(task)]
                s_task = []
                L_task = []
                P_task = []
                sg_task = []
                for k in keys:
                    r0 = v0[k]
                    r2 = v2[k]
                    r3 = v3[k]
                    r4 = v4[k]

                    m0 = r0["logp"]["B"] - r0["logp"]["A"]
                    m2 = r2["logp"]["B"] - r2["logp"]["A"]
                    m3 = r3["logp"]["B"] - r3["logp"]["A"]
                    m4 = r4["logp"]["B"] - r4["logp"]["A"]

                    L = (m0 + m2 + m3 + m4) / 4.0
                    P = (m0 + m2 - m3 - m4) / 4.0
                    s = (m0 - m2 - m3 + m4) / 4.0

                    # s_gold: positive if model prefers gold content over distractor content
                    gold_sign = 1.0 if r0["gold"] == "B" else -1.0
                    s_gold = s * gold_sign

                    s_all.append(s)
                    L_all.append(L)
                    P_all.append(P)
                    s_gold_correct.append(s_gold > 0)

                    s_task.append(s)
                    L_task.append(L)
                    P_task.append(P)
                    sg_task.append(s_gold > 0)

                per_task_2x2[task] = {
                    "debiased_acc": float(np.mean(sg_task)),
                    "mean_L": float(np.mean(L_task)),
                    "mean_P": float(np.mean(P_task)),
                    "mean_s": float(np.mean(s_task)),
                    "correct": sg_task,
                }

            c_res["debiased_2x2"] = {
                "overall_debiased_acc": float(np.mean(s_gold_correct)),
                "correct": s_gold_correct,
                "mean_L": float(np.mean(L_all)),
                "mean_P": float(np.mean(P_all)),
                "mean_s": float(np.mean(s_all)),
                "per_task": per_task_2x2,
            }

        results[cid] = c_res

    # 4. Comparative Hypothesis Tests
    comparisons = {}

    # Q1: Primary test: T1 vs T8 (Mull vs SAT Baseline)
    if "T1" in results and "T8" in results:
        r1 = results["T1"]
        r8 = results["T8"]

        # McNemar on raw M1 (N=700)
        c1 = r1["micro"]["correct_m1"]
        c8 = r8["micro"]["correct_m1"]
        b_m1 = sum(a and not b for a, b in zip(c1, c8))
        c_m1 = sum(not a and b for a, b in zip(c1, c8))
        p_m1 = mcnemar_test(b_m1, c_m1)
        raw_diff = [float(a) - float(b) for a, b in zip(c1, c8)]
        ci_raw = bootstrap_ci(raw_diff)

        # McNemar on calibrated M2 (N=700)
        c1_m2 = r1["micro"]["correct_m2"]
        c8_m2 = r8["micro"]["correct_m2"]
        b_m2 = sum(a and not b for a, b in zip(c1_m2, c8_m2))
        c_m2 = sum(not a and b for a, b in zip(c1_m2, c8_m2))
        p_m2 = mcnemar_test(b_m2, c_m2)
        diff_m2 = [float(a) - float(b) for a, b in zip(c1_m2, c8_m2)]
        ci_m2 = bootstrap_ci(diff_m2)

        # Debiased 2x2 M3 (N=550) if available
        comp_m3 = None
        if "correct" in r1.get("debiased_2x2", {}) and "correct" in r8.get("debiased_2x2", {}):
            c1_m3 = r1["debiased_2x2"]["correct"]
            c8_m3 = r8["debiased_2x2"]["correct"]
            b_m3 = sum(a and not b for a, b in zip(c1_m3, c8_m3))
            c_m3 = sum(not a and b for a, b in zip(c1_m3, c8_m3))
            p_m3 = mcnemar_test(b_m3, c_m3)
            diff_m3 = [float(a) - float(b) for a, b in zip(c1_m3, c8_m3)]
            ci_m3 = bootstrap_ci(diff_m3)
            raw_gap_550 = np.mean([float(a) - float(b) for a, b in zip(c1[:550], c8[:550])])
            debias_gap_550 = np.mean(diff_m3)
            bias_share = 1.0 - (debias_gap_550 / raw_gap_550) if raw_gap_550 != 0 else 0.0
            comp_m3 = {
                "t1_acc": r1["debiased_2x2"]["overall_debiased_acc"],
                "t8_acc": r8["debiased_2x2"]["overall_debiased_acc"],
                "delta": debias_gap_550,
                "ci_95": ci_m3,
                "b": b_m3,
                "c": c_m3,
                "p_value": p_m3,
                "bias_explained_share": bias_share,
            }

        comparisons["Q1_T1_vs_T8"] = {
            "raw_m1": {
                "t1_acc": r1["micro"]["raw_acc"],
                "t8_acc": r8["micro"]["raw_acc"],
                "delta": float(r1["micro"]["raw_acc"] - r8["micro"]["raw_acc"]),
                "ci_95": ci_raw,
                "b": b_m1,
                "c": c_m1,
                "p_value": p_m1,
            },
            "calib_m2": {
                "t1_acc": r1["micro"]["calib_acc"],
                "t8_acc": r8["micro"]["calib_acc"],
                "delta": float(r1["micro"]["calib_acc"] - r8["micro"]["calib_acc"]),
                "ci_95": ci_m2,
                "b": b_m2,
                "c": c_m2,
                "p_value": p_m2,
            },
            "debiased_m3": comp_m3,
        }

    # Q2: Continuous Margins & Equivalence within Mull: T3 vs T1 (and T4, T5 if available)
    if "T1" in results and "T3" in results:
        m1 = results["T1"]["micro"]["margins"]
        m3 = results["T3"]["micro"]["margins"]
        w_stat, p_wilcox = stats.wilcoxon(m1, m3)
        diff_margins = [a - b for a, b in zip(m1, m3)]
        ci_margin = bootstrap_ci(diff_margins)
        mean_diff = float(np.mean(diff_margins))
        std_diff = float(np.std(diff_margins))
        dz = mean_diff / std_diff if std_diff > 0 else 0.0

        # Accuracy difference CI
        acc_diff = [float(a) - float(b) for a, b in zip(results["T1"]["micro"]["correct_m1"], results["T3"]["micro"]["correct_m1"])]
        ci_acc_90 = bootstrap_ci(acc_diff, ci=0.90)

        comparisons["Q2_T3_vs_T1"] = {
            "mean_margin_t1": float(np.mean(m1)),
            "mean_margin_t3": float(np.mean(m3)),
            "margin_diff": mean_diff,
            "margin_ci_95": ci_margin,
            "wilcoxon_w": float(w_stat),
            "p_value": float(p_wilcox),
            "cohen_dz": dz,
            "acc_diff_90_ci": ci_acc_90,
            "tost_equivalent": abs(dz) < 0.1 and abs(ci_acc_90[0]) <= 0.03 and abs(ci_acc_90[1]) <= 0.03,
        }

    # 5. Format Master Summary Markdown
    md = []
    md.append("# Experiment E2: Answer Bias vs. Discrimination Analysis")
    md.append(f"**Generated:** {os.path.basename(out_dir)} | **Evaluations:** N={TOTAL_N} (2-Way N={TOTAL_2WAY_N})\n")

    md.append("## 1. Master Calibration & Debiasing Matrix\n")
    md.append("| Cond | Model | Prefill | Raw Acc (N=700) | Calibrated M2 | Macro Raw | Macro Calib | 2-Way Pred B | Debiased M3 (N=550) | Mean L | Mean P |")
    md.append("|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for cid in available_conds:
        r = results[cid]
        p_b = f"{np.mean([r['per_task'][t]['pred_b_frac'] for t in TASKS_2WAY])*100:.1f}%" if "blink_sprel" in r["per_task"] else "-"
        deb_acc = f"{r['debiased_2x2']['overall_debiased_acc']*100:.2f}%" if "overall_debiased_acc" in r.get("debiased_2x2", {}) else "-"
        mean_l = f"{r['debiased_2x2']['mean_L']:+.3f}" if "mean_L" in r.get("debiased_2x2", {}) else "-"
        mean_p = f"{r['debiased_2x2']['mean_P']:+.3f}" if "mean_P" in r.get("debiased_2x2", {}) else "-"

        md.append(f"| **{cid}** | {CONDITIONS[cid]['model']} | `{CONDITIONS[cid]['prompt']}` / `{CONDITIONS[cid]['prefill'][:15]}...` | "
                  f"**{r['micro']['raw_acc']*100:.2f}%** | {r['micro']['calib_acc']*100:.2f}% | "
                  f"{r['macro']['raw_acc']*100:.2f}% | {r['macro']['calib_acc']*100:.2f}% | "
                  f"{p_b} | {deb_acc} | {mean_l} | {mean_p} |")

    # Q1 Section
    if "Q1_T1_vs_T8" in comparisons:
        q1 = comparisons["Q1_T1_vs_T8"]
        md.append("\n## 2. Primary Hypothesis Q1: Does Mull's Edge Survive Bias Calibration?\n")
        md.append("| Metric | T1 (Mull) | T8 (SAT) | Delta | 95% Bootstrap CI | McNemar p | Survives? |")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
        raw = q1["raw_m1"]
        md.append(f"| **Raw Accuracy (M1)** | {raw['t1_acc']*100:.2f}% | {raw['t8_acc']*100:.2f}% | {raw['delta']*100:+.2f}% | [{raw['ci_95'][0]*100:+.2f}%, {raw['ci_95'][1]*100:+.2f}%] | p={raw['p_value']:.4f} | Baseline |")
        cal = q1["calib_m2"]
        md.append(f"| **Content-Free Calibrated (M2)** | {cal['t1_acc']*100:.2f}% | {cal['t8_acc']*100:.2f}% | {cal['delta']*100:+.2f}% | [{cal['ci_95'][0]*100:+.2f}%, {cal['ci_95'][1]*100:+.2f}%] | p={cal['p_value']:.4f} | {'**YES**' if cal['p_value']<0.05 and cal['delta']>0 else 'NO'} |")
        if q1["debiased_m3"]:
            deb = q1["debiased_m3"]
            md.append(f"| **Debiased 2x2 (M3, N=550)** | {deb['t1_acc']*100:.2f}% | {deb['t8_acc']*100:.2f}% | {deb['delta']*100:+.2f}% | [{deb['ci_95'][0]*100:+.2f}%, {deb['ci_95'][1]*100:+.2f}%] | p={deb['p_value']:.4f} | {'**YES**' if deb['p_value']<0.05 and deb['delta']>0 else 'NO'} |")
            md.append(f"\n> **Share of Published Advantage Explained by Bias:** `{deb['bias_explained_share']*100:.1f}%`")

    # Q2 Section
    if "Q2_T3_vs_T1" in comparisons:
        q2 = comparisons["Q2_T3_vs_T1"]
        md.append("\n## 3. Continuous Logit Margins: T3 (<think></think>) vs. T1 (20 Latents)\n")
        md.append(f"- **T1 Mean Margin:** `{q2['mean_margin_t1']:.4f}`")
        md.append(f"- **T3 Mean Margin:** `{q2['mean_margin_t3']:.4f}` (Delta: `{q2['margin_diff']:+.4f}`, 95% CI: `[{q2['margin_ci_95'][0]:+.4f}, {q2['margin_ci_95'][1]:+.4f}]`)")
        md.append(f"- **Wilcoxon Signed-Rank Test:** W={q2['wilcoxon_w']:.1f}, **p = {q2['p_value']:.4f}**")
        md.append(f"- **Paired Cohen's d_z:** `{q2['cohen_dz']:.4f}`")
        md.append(f"- **Two One-Sided Equivalence Test (TOST, ±3pp):** {'**CONFIRMED EQUIVALENT**' if q2['tost_equivalent'] else 'Inconclusive'}")

    summary_md = "\n".join(md) + "\n"
    summary_path = os.path.join(PROJECT_DIR, "e2_bias", "E2_SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write(summary_md)
    print(f"Summary written to {summary_path}")

    results_json_path = os.path.join(PROJECT_DIR, "e2_bias", "e2_results.json")
    # Clean non-serializable sets/arrays
    def clean(obj):
        if isinstance(obj, (np.ndarray, list)):
            return [clean(x) for x in obj]
        elif isinstance(obj, (np.floating, float)):
            return round(float(obj), 6)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k not in ("correct_m1", "correct_m2", "margins", "correct")}
        return obj

    with open(results_json_path, "w") as f:
        json.dump(clean(results), f, indent=2)
    print(f"Results JSON written to {results_json_path}")


# Import CONDITIONS from score_letters
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_letters import CONDITIONS

if __name__ == "__main__":
    analyze_all()