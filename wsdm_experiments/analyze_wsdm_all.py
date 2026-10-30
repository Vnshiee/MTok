#!/usr/bin/env python3
"""
Master Analyzer for WSDM Experiments (W1, W2, W3, W4, W5) + Integration with E1 & E2.
Computes:
  - W1: Full 6-Checkpoint Training Lineage Audit on BLINK (Raw M1, Calibrated M2, 2x2 Debiased M3, L, P, s)
  - W2: Causal 4D Attention Mask Knockout on BLINK (7 conditions, McNemar, Wilcoxon, TOST)
  - W3: Cross-Benchmark Generalization on SAT-Real (N=300), ERQA (N=400), MMSI-Bench (N=1000)
        including paired McNemar, Wilcoxon margin tests, TOST equivalence, and SAT-Real 150-pair debiased analysis.
  - W4: Counterfactual Activation Patching (N=550) & Pre-Norm Layer 0-27 Geometry + Linear Probing (N=143).
  - W5: Constructive Inference Protocol (Zero-Latent Format Lock + Prior Calibration vs. K=20 Latents).
"""

import glob
import json
import math
import os
import sys
import numpy as np
import scipy.stats as stats

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W2_W4_DIR = os.path.join(PROJECT_DIR, "wsdm_experiments", "outputs_w2_w4")
W1_W3_DIR = os.path.join(PROJECT_DIR, "wsdm_experiments", "outputs_w1_w3")
E2_DIR = os.path.join(PROJECT_DIR, "e2_bias", "outputs")

TASKS_2WAY = ["blink_sprel", "blink_reldepth", "blink_jigsaw", "blink_mv"]
TASKS_ALL = TASKS_2WAY + ["blink_iqtest"]
SEED = 20261028


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(2.0 * min(stats.binom.cdf(min(b, c), n, 0.5), 1.0 - stats.binom.cdf(max(b, c) - 1, n, 0.5)))


def bootstrap_ci(diffs, n_resamples=10000, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    diffs = np.asarray(diffs, dtype=float)
    indices = rng.randint(0, len(diffs), (n_resamples, len(diffs)))
    means = diffs[indices].mean(axis=1)
    lower = np.percentile(means, ((1.0 - ci) / 2.0) * 100)
    upper = np.percentile(means, (1.0 - (1.0 - ci) / 2.0) * 100)
    return float(lower), float(upper)


def analyze_blink_cond_dir(cdir):
    if not os.path.exists(os.path.join(cdir, "V0.jsonl")):
        return None
    data = {}
    for v in ["V0", "V2", "V3", "V4", "CF"]:
        p = os.path.join(cdir, f"{v}.jsonl")
        if os.path.exists(p):
            data[v] = {json.loads(l)["key"]: json.loads(l) for l in open(p) if l.strip()}

    v0 = data.get("V0", {})
    cf = data.get("CF", {})
    p_cf = {}
    for t in TASKS_ALL:
        t_cf = [r for r in cf.values() if r["task"] == t]
        letters = "AB" if t in TASKS_2WAY else "ABCD"
        if t_cf:
            mean_lp = {L: np.mean([r["logp"][L] for r in t_cf]) for L in letters}
            exps = {L: math.exp(mean_lp[L]) for L in letters}
            tot = sum(exps.values())
            p_cf[t] = {L: exps[L] / tot for L in letters}
        else:
            p_cf[t] = {L: 1.0 / len(letters) for L in letters}

    corr_m1, corr_m2, margins, pred_b_2way = [], [], [], []
    per_task = {}
    for t in TASKS_ALL:
        t_v0 = [v0[k] for k in sorted(v0.keys()) if v0[k]["task"] == t]
        letters = "AB" if t in TASKS_2WAY else "ABCD"
        if not t_v0:
            continue
        c1 = [r["tf_pred"] == r["gold"] for r in t_v0]
        c2 = []
        m_list = []
        for r in t_v0:
            sc = {L: math.exp(r["logp"][L]) / p_cf[t][L] for L in letters}
            c2.append(max(sc, key=sc.get) == r["gold"])
            g = r["gold"]
            alt = max(r["logit"][L] for L in letters if L != g)
            m_list.append(r["logit"][g] - alt)
            if t in TASKS_2WAY:
                pred_b_2way.append(r["tf_pred"] == "B")
        corr_m1.extend(c1)
        corr_m2.extend(c2)
        margins.extend(m_list)
        per_task[t] = {
            "raw_acc": round(float(np.mean(c1)) * 100.0, 2),
            "calib_acc": round(float(np.mean(c2)) * 100.0, 2),
        }

    # 2x2 Debiased
    deb_res = {}
    if all(v in data for v in ["V0", "V2", "V3", "V4"]):
        L_all, P_all, S_abs_all, sg_corr = [], [], [], []
        for t in TASKS_2WAY:
            keys = sorted([k for k in v0 if k.startswith(t)])
            for k in keys:
                if k not in data["V2"] or k not in data["V3"] or k not in data["V4"]:
                    continue
                m0 = v0[k]["logp"]["B"] - v0[k]["logp"]["A"]
                m2 = data["V2"][k]["logp"]["B"] - data["V2"][k]["logp"]["A"]
                m3 = data["V3"][k]["logp"]["B"] - data["V3"][k]["logp"]["A"]
                m4 = data["V4"][k]["logp"]["B"] - data["V4"][k]["logp"]["A"]
                L = (m0 + m2 + m3 + m4) / 4.0
                P = (m0 + m2 - m3 - m4) / 4.0
                s = (m0 - m2 - m3 + m4) / 4.0
                gsign = 1.0 if v0[k]["gold"] == "B" else -1.0
                L_all.append(L)
                P_all.append(P)
                S_abs_all.append(abs(s))
                sg_corr.append((s * gsign) > 0)
        if sg_corr:
            deb_res = {
                "debiased_acc_pct": round(float(np.mean(sg_corr)) * 100.0, 2),
                "mean_L": round(float(np.mean(L_all)), 4),
                "mean_P": round(float(np.mean(P_all)), 4),
                "mean_abs_s": round(float(np.mean(S_abs_all)), 4),
                "sg_corr": sg_corr,
            }

    return {
        "n": len(corr_m1),
        "raw_micro_pct": round(float(np.mean(corr_m1)) * 100.0, 2) if corr_m1 else 0.0,
        "calib_micro_pct": round(float(np.mean(corr_m2)) * 100.0, 2) if corr_m2 else 0.0,
        "raw_macro_pct": round(float(np.mean([x["raw_acc"] for x in per_task.values()])), 2) if per_task else 0.0,
        "calib_macro_pct": round(float(np.mean([x["calib_acc"] for x in per_task.values()])), 2) if per_task else 0.0,
        "pred_b_2way_pct": round(float(np.mean(pred_b_2way)) * 100.0, 2) if pred_b_2way else 0.0,
        "mean_margin": round(float(np.mean(margins)), 4) if margins else 0.0,
        "per_task": per_task,
        "debiased": deb_res,
        "corr_m1": corr_m1,
        "corr_m2": corr_m2,
        "margins": margins,
    }


def analyze_external_benchmarks():
    w3_root = os.path.join(W1_W3_DIR, "w3_benchmarks")
    if not os.path.exists(w3_root):
        return {}
    conds = sorted([os.path.basename(d) for d in glob.glob(os.path.join(w3_root, "W3_*")) if os.path.isdir(d)])
    benchmarks = ["sat_real", "erqa", "mmsi_bench"]
    out = {b: {"conditions": {}, "comparisons": {}} for b in benchmarks}

    for b in benchmarks:
        raw_by_cond = {}
        for cid in conds:
            fpath = os.path.join(w3_root, cid, f"{b}.jsonl")
            cfpath = os.path.join(w3_root, cid, f"{b}_CF.jsonl")
            if not os.path.exists(fpath):
                continue
            rows = [json.loads(l) for l in open(fpath) if l.strip()]
            if not rows:
                continue
            cf_rows = [json.loads(l) for l in open(cfpath) if l.strip()] if os.path.exists(cfpath) else []

            # Prior p_cf
            letters_all = cf_rows[0]["letters"] if cf_rows else ("AB" if b == "sat_real" else "ABCD")
            if cf_rows:
                mlp = {L: np.mean([r["logp"][L] for r in cf_rows if L in r["logp"]]) for L in letters_all}
                exps = {L: math.exp(mlp[L]) for L in letters_all}
                tot = sum(exps.values())
                p_cf = {L: exps[L] / tot for L in letters_all}
            else:
                p_cf = {L: 1.0 / len(letters_all) for L in letters_all}

            # Also compute batch prior (label-free mean logp across dataset, Zhou et al. 2024)
            batch_mlp = {L: np.mean([r["logp"][L] for r in rows if L in r["logp"]]) for L in letters_all}

            c_raw, c_cf, c_batch, margins = [], [], [], []
            for r in rows:
                ls = r["letters"]
                g = r["gold"]
                c_raw.append(r["tf_pred"] == g)
                sc_cf = {L: math.exp(r["logp"][L]) / p_cf.get(L, 0.25) for L in ls}
                c_cf.append(max(sc_cf, key=sc_cf.get) == g)
                sc_b = {L: r["logp"][L] - batch_mlp.get(L, 0.0) for L in ls}
                c_batch.append(max(sc_b, key=sc_b.get) == g)
                alt = max(r["logit"][L] for L in ls if L != g)
                margins.append(r["logit"][g] - alt)

            pair_stats = None
            if b == "sat_real" and len(rows) == 300:
                pair_both_corr = 0
                debiased_corr = 0
                pos_bias_list = []
                for i in range(150):
                    r_orig = rows[i]
                    r_swap = rows[i + 150]
                    if (r_orig["tf_pred"] == r_orig["gold"]) and (r_swap["tf_pred"] == r_swap["gold"]):
                        pair_both_corr += 1
                    if "A" in r_orig["logp"] and "B" in r_orig["logp"] and "A" in r_swap["logp"] and "B" in r_swap["logp"]:
                        m_o = r_orig["logp"]["B"] - r_orig["logp"]["A"]
                        m_s = r_swap["logp"]["B"] - r_swap["logp"]["A"]
                        s_content = (m_o - m_s) / 2.0
                        bias_lp = (m_o + m_s) / 2.0
                        gsign = 1.0 if r_orig["gold"] == "B" else -1.0
                        if (s_content * gsign) > 0:
                            debiased_corr += 1
                        pos_bias_list.append(bias_lp)
                pair_stats = {
                    "pair_consistent_acc_pct": round((pair_both_corr / 150.0) * 100.0, 2),
                    "swap_debiased_acc_pct": round((debiased_corr / 150.0) * 100.0, 2),
                    "mean_B_bias_log_odds": round(float(np.mean(pos_bias_list)), 4),
                    "pred_B_frac_pct": round(float(np.mean([r["tf_pred"] == "B" for r in rows])) * 100.0, 2),
                }

            raw_by_cond[cid] = {"c_raw": c_raw, "margins": margins}
            out[b]["conditions"][cid] = {
                "n": len(rows),
                "raw_acc_pct": round(float(np.mean(c_raw)) * 100.0, 2),
                "cf_calib_acc_pct": round(float(np.mean(c_cf)) * 100.0, 2),
                "batch_calib_acc_pct": round(float(np.mean(c_batch)) * 100.0, 2),
                "mean_margin": round(float(np.mean(margins)), 4),
                "pair_stats": pair_stats,
            }

        # Key comparisons against W3_Mull_K20
        if "W3_Mull_K20" in raw_by_cond:
            ref = raw_by_cond["W3_Mull_K20"]
            for target_cid, tgt in raw_by_cond.items():
                if target_cid == "W3_Mull_K20" or len(tgt["c_raw"]) != len(ref["c_raw"]):
                    continue
                b_cnt = sum(x and not y for x, y in zip(ref["c_raw"], tgt["c_raw"]))
                c_cnt = sum(not x and y for x, y in zip(ref["c_raw"], tgt["c_raw"]))
                p_mcn = mcnemar_p(b_cnt, c_cnt)
                diffs = [float(y) - float(x) for x, y in zip(ref["c_raw"], tgt["c_raw"])]
                ci95 = bootstrap_ci(diffs, ci=0.95)
                ci90 = bootstrap_ci(diffs, ci=0.90)
                m_diffs = [y - x for x, y in zip(ref["margins"], tgt["margins"])]
                _, p_wilcox = stats.wilcoxon(ref["margins"], tgt["margins"]) if not np.allclose(ref["margins"], tgt["margins"]) else (0.0, 1.0)
                dz = float(np.mean(m_diffs) / (np.std(m_diffs) + 1e-12))
                out[b]["comparisons"][f"{target_cid}_vs_W3_Mull_K20"] = {
                    "delta_acc_pct": round(float(np.mean(diffs)) * 100.0, 2),
                    "ci95_pct": [round(ci95[0] * 100.0, 2), round(ci95[1] * 100.0, 2)],
                    "ci90_pct": [round(ci90[0] * 100.0, 2), round(ci90[1] * 100.0, 2)],
                    "mcnemar_b": int(b_cnt),
                    "mcnemar_c": int(c_cnt),
                    "mcnemar_p": round(p_mcn, 4),
                    "margin_delta": round(float(np.mean(m_diffs)), 4),
                    "wilcoxon_p": round(float(p_wilcox), 4),
                    "cohen_dz": round(dz, 4),
                    "tost_3pp_equiv": bool(abs(dz) < 0.1 and abs(ci90[0]) <= 0.03 and abs(ci90[1]) <= 0.03),
                }

    return out


def write_w1_w3_markdown(master):
    md = []
    md.append("# WSDM Experiments W1, W3 & W5: Lineage Audit, Cross-Benchmark Generalization & Constructive Protocol")
    md.append("")
    md.append("## 1. Experiment W1: Full 6-Checkpoint Training Lineage Audit on BLINK (N=700)")
    md.append("")
    md.append("| Condition | Checkpoint | Raw Micro (M1) | CF Calib Micro (M2) | 2x2 Debiased (M3, N=550) | 2-Way %B | Label Bias `L` | Pos Bias `P` | Signal `|s|` | Mean Margin |")
    md.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    order = [
        ("W1_Base_Dir", "Qwen2.5-VL-7B-Instruct (Dir)"),
        ("W1_Base_Think", "Qwen2.5-VL-7B-Instruct (<think></think>)"),
        ("E2_T8_SAT_Dir", "Qwen2.5-VL-SAT (SFT Dir)"),
        ("E2_T9_SAT_LatPrompt", "Qwen2.5-VL-SAT (Latent Prompt)"),
        ("W1_SATBase_Dir", "Qwen2.5-VL-SATBase (Paper DirAns, SFT Prompt)"),
        ("W1_SATBase_Lat", "Qwen2.5-VL-SATBase (Paper DirAns, Lat Prompt)"),
        ("W1_LatCust_K20", "Qwen2.5-VL-Latent-Custom (Stage-1, K=20)"),
        ("W1_LatCust_Think", "Qwen2.5-VL-Latent-Custom (Stage-1, <think></think>)"),
        ("E2_T1_Mull_K20", "Qwen2.5-VL-Mull (Stage-2 SFT, K=20)"),
        ("E2_T3_Mull_Think", "Qwen2.5-VL-Mull (Stage-2 SFT, <think></think>)"),
        ("E2_T2F_Mull_ForcedAns", "Qwen2.5-VL-Mull (Forced <answer, 0 think)"),
        ("E2_T10_GRPO_K20", "Qwen2.5-VL-MullGRPO (Stage-3 RL, K=20)"),
    ]
    for cid, label in order:
        if cid not in master["W1_lineage"]:
            continue
        r = master["W1_lineage"][cid]
        deb = r.get("debiased", {})
        d_acc = f"**{deb['debiased_acc_pct']:.2f}%**" if deb else "N/A"
        L_val = f"{deb['mean_L']:+.3f}" if deb else "N/A"
        P_val = f"{deb['mean_P']:+.3f}" if deb else "N/A"
        S_val = f"{deb.get('mean_abs_s', 0.0):.3f}" if deb else "N/A"
        md.append(
            f"| **{cid}** | {label} | **{r['raw_micro_pct']:.2f}%** | {r['calib_micro_pct']:.2f}% | "
            f"{d_acc} | {r['pred_b_2way_pct']:.1f}% | {L_val} | {P_val} | {S_val} | {r['mean_margin']:.3f} |"
        )

    md.append("")
    md.append("## 2. Experiment W3: Cross-Benchmark Generalization (SAT-Real N=300, ERQA N=400, MMSI-Bench N=1000)")
    for bname, btitle in [
        ("sat_real", "SAT-Real (N=300; 150 Original + 150 A<->B Reversed Pairs)"),
        ("erqa", "ERQA Embodied Robotics QA (N=400)"),
        ("mmsi_bench", "MMSI-Bench Multi-Image Spatial Intelligence (N=1,000)"),
    ]:
        if bname not in master["W3_external"] or not master["W3_external"][bname]["conditions"]:
            continue
        md.append("")
        md.append(f"### {btitle}")
        bdata = master["W3_external"][bname]
        if bname == "sat_real":
            md.append("| Condition | N | Raw Acc | CF Calib Acc | Batch Calib Acc | Pair-Consistent Acc | Swap-Debiased Acc | % Pred B | Mean Margin | vs Mull K=20 (McNemar p / Wilcoxon p) |")
            md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
            for cid, r in bdata["conditions"].items():
                ps = r.get("pair_stats") or {}
                comp = bdata["comparisons"].get(f"{cid}_vs_W3_Mull_K20")
                c_str = "Reference" if cid == "W3_Mull_K20" else (
                    f"{comp['delta_acc_pct']:+.2f}pp (p={comp['mcnemar_p']:.4f} / w_p={comp['wilcoxon_p']:.4f})" if comp else "N/A"
                )
                md.append(
                    f"| **{cid}** | {r['n']} | **{r['raw_acc_pct']:.2f}%** | {r['cf_calib_acc_pct']:.2f}% | "
                    f"{r['batch_calib_acc_pct']:.2f}% | {ps.get('pair_consistent_acc_pct', 0.0):.2f}% | "
                    f"**{ps.get('swap_debiased_acc_pct', 0.0):.2f}%** | {ps.get('pred_B_frac_pct', 0.0):.1f}% | "
                    f"{r['mean_margin']:.3f} | {c_str} |"
                )
        else:
            md.append("| Condition | N | Raw Acc | CF Calib Acc | Batch Calib Acc | Mean Margin | vs Mull K=20 (Δ Acc, 95% CI, McNemar p, Wilcoxon p) |")
            md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
            for cid, r in bdata["conditions"].items():
                comp = bdata["comparisons"].get(f"{cid}_vs_W3_Mull_K20")
                c_str = "Reference" if cid == "W3_Mull_K20" else (
                    f"{comp['delta_acc_pct']:+.2f}pp [{comp['ci95_pct'][0]:+.2f}, {comp['ci95_pct'][1]:+.2f}] "
                    f"(p={comp['mcnemar_p']:.4f}, w_p={comp['wilcoxon_p']:.4f})" if comp else "N/A"
                )
                md.append(
                    f"| **{cid}** | {r['n']} | **{r['raw_acc_pct']:.2f}%** | {r['cf_calib_acc_pct']:.2f}% | "
                    f"{r['batch_calib_acc_pct']:.2f}% | {r['mean_margin']:.3f} | {c_str} |"
                )

    out_md_path = os.path.join(W1_W3_DIR, "W1_W3_W5_SUMMARY.md")
    os.makedirs(W1_W3_DIR, exist_ok=True)
    with open(out_md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved Markdown summary to {out_md_path}")


def main():
    master = {"W1_lineage": {}, "W2_W4": {}, "W3_external": {}}

    # 1. Load W1 Lineage + E2 Reference Conditions (T1, T3, T8, T9, T10, T2F)
    e2_map = {
        "E2_T1_Mull_K20": os.path.join(E2_DIR, "T1"),
        "E2_T3_Mull_Think": os.path.join(E2_DIR, "T3"),
        "E2_T2F_Mull_ForcedAns": os.path.join(E2_DIR, "T2F"),
        "E2_T8_SAT_Dir": os.path.join(E2_DIR, "T8"),
        "E2_T9_SAT_LatPrompt": os.path.join(E2_DIR, "T9"),
        "E2_T10_GRPO_K20": os.path.join(E2_DIR, "T10"),
    }
    w1_root = os.path.join(W1_W3_DIR, "w1_blink")
    if os.path.exists(w1_root):
        for d in sorted(glob.glob(os.path.join(w1_root, "W1_*"))):
            e2_map[os.path.basename(d)] = d

    loaded_w1 = {}
    for name, path in e2_map.items():
        res = analyze_blink_cond_dir(path)
        if res is not None:
            loaded_w1[name] = res
            master["W1_lineage"][name] = {k: v for k, v in res.items() if k not in ("corr_m1", "corr_m2", "margins") and k != "debiased"}
            if res["debiased"]:
                master["W1_lineage"][name]["debiased"] = {k: v for k, v in res["debiased"].items() if k != "sg_corr"}

    # Compare Mull (E2_T1_Mull_K20) vs True DirAns (W1_SATBase_Dir and W1_SATBase_Lat) and Base Model
    w1_comps = {}
    if "E2_T1_Mull_K20" in loaded_w1:
        r_mull = loaded_w1["E2_T1_Mull_K20"]
        for k_other, r_other in loaded_w1.items():
            if k_other == "E2_T1_Mull_K20" or r_other["n"] != r_mull["n"]:
                continue
            b1 = sum(x and not y for x, y in zip(r_mull["corr_m1"], r_other["corr_m1"]))
            c1 = sum(not x and y for x, y in zip(r_mull["corr_m1"], r_other["corr_m1"]))
            p1 = mcnemar_p(b1, c1)
            comp_entry = {
                "raw_delta_mull_vs_other_pct": round(r_mull["raw_micro_pct"] - r_other["raw_micro_pct"], 2),
                "raw_mcnemar_b": int(b1),
                "raw_mcnemar_c": int(c1),
                "raw_mcnemar_p": round(p1, 4),
            }
            if r_mull["debiased"] and r_other["debiased"]:
                sg_m = r_mull["debiased"]["sg_corr"]
                sg_o = r_other["debiased"]["sg_corr"]
                if len(sg_m) == len(sg_o):
                    bd = sum(x and not y for x, y in zip(sg_m, sg_o))
                    cd = sum(not x and y for x, y in zip(sg_m, sg_o))
                    pd = mcnemar_p(bd, cd)
                    comp_entry["debiased_delta_mull_vs_other_pct"] = round(
                        r_mull["debiased"]["debiased_acc_pct"] - r_other["debiased"]["debiased_acc_pct"], 2
                    )
                    comp_entry["debiased_mcnemar_b"] = int(bd)
                    comp_entry["debiased_mcnemar_c"] = int(cd)
                    comp_entry["debiased_mcnemar_p"] = round(pd, 4)
            w1_comps[f"Mull_K20_vs_{k_other}"] = comp_entry
    master["W1_comparisons"] = w1_comps

    # 2. Load W2 & W4
    w2_w4_json = os.path.join(W2_W4_DIR, "w2_w4_summary.json")
    if os.path.exists(w2_w4_json):
        master["W2_W4"] = json.load(open(w2_w4_json))

    # 3. Load W3 External Benchmarks
    master["W3_external"] = analyze_external_benchmarks()

    out_json = os.path.join(PROJECT_DIR, "wsdm_experiments", "WSDM_MASTER_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump(master, f, indent=2)
    print(f"Saved master JSON to {out_json}")
    write_w1_w3_markdown(master)


if __name__ == "__main__":
    main()
