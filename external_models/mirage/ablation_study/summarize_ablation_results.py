#!/usr/bin/env python
"""
Summarize Mirage ablation outputs.

Reads every <output_dir>/*.jsonl written by run_mirage_ablation.py and emits:
  * <output_dir>/ABLATION_SUMMARY.md   publication-ready Markdown tables
  * <output_dir>/ablation_summary.json machine-readable numbers

Per condition: n, accuracy, standard error, 95% Wilson interval, per-category accuracy, MMVP pair accuracy, latent
telemetry (mean latent steps, natural-latent norm, consecutive-step cosine, max cosine to image tokens, early
<|latent_end|> predictions). Per comparison against the benchmark's reference condition (default: target, K=4, none,
forced entry; K=4 is the checkpoint's training latent size): paired items only, delta accuracy, exact McNemar b/c and
two-sided p (binomial, capped at 1), Holm-adjusted p within each benchmark, paired bootstrap 95% CI of the delta
(10,000 resamples, fixed seed), and a TOST-style equivalence flag (90% bootstrap CI inside +/- --equiv_margin).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path


def load_runs(out_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        name = Path(f).stem
        rows = {}
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["question_id"]] = r        # last write wins (resume-safe)
        if rows:
            first = next(iter(rows.values()))
            runs[name] = dict(name=name, rows=rows, benchmark=first["benchmark"], model_mode=first["model_mode"],
                              quota=str(first["quota"]), intervention=first["intervention"], entry=first["entry"],
                              decoding=first.get("decoding", "greedy"),
                              complete=os.path.exists(os.path.join(out_dir, name + ".COMPLETE")))
    return runs


def binom_two_sided(b, c):
    """Exact two-sided McNemar p: P(X <= min(b,c)) * 2 under Binomial(b+c, 0.5), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def paired_bootstrap(x, y, B=10000, seed=20261103, alpha=0.05):
    """CI of mean(y) - mean(x) for paired binary vectors."""
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    d = [yi - xi for xi, yi in zip(x, y)]
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        dv = np.asarray(d, dtype=np.float64)
        stats = []
        for start in range(0, B, 1000):                      # chunked to bound memory
            idx = rng.integers(0, n, size=(min(1000, B - start), n))
            stats.append(dv[idx].mean(axis=1))
        stats = np.sort(np.concatenate(stats)).tolist()
    except ImportError:
        rng = random.Random(seed)
        stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    q = lambda a: stats[min(B - 1, max(0, int(round(a * (B - 1)))))]
    return (q(alpha / 2), q(1 - alpha / 2)), (q(0.05), q(0.95))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def one_sample_boot(vals, B=10000, seed=20261103):
    """95% bootstrap CI of the mean of `vals`."""
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), (float("nan"), float("nan"))
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        v = np.asarray(vals, dtype=float)
        st = np.sort(v[rng.integers(0, len(v), size=(B, len(v)))].mean(1))
        return float(v.mean()), (float(st[int(0.025 * (B - 1))]), float(st[int(0.975 * (B - 1))]))
    except ImportError:
        m = sum(vals) / len(vals)
        return m, (float("nan"), float("nan"))


def h3_contrast(run, B, seed):
    """Per question: mean over latent steps of max cos(latent, own image tokens) - max cos(latent, other question's image tokens)."""
    diffs = []
    for r in run["rows"].values():
        d = [s["max_cos_natural_image_token"] - s["max_cos_natural_null_image_token"] for s in r.get("latent_steps", [])
             if s.get("max_cos_natural_image_token") is not None and s.get("max_cos_natural_null_image_token") is not None]
        if d:
            diffs.append(sum(d) / len(d))
    m, ci = one_sample_boot(diffs, B, seed)
    return dict(n=len(diffs), mean=m, ci95=ci)


def donor_adoption(run, natural):
    """Donor runs: does the answer follow the donor's natural answer (H1) or stay with the question's own (H2/H3)?"""
    n = adopt = keep = 0
    for q, r in run["rows"].items():
        d = r.get("donor_id")
        if not d or q not in natural["rows"] or d not in natural["rows"]:
            continue
        own, don = natural["rows"][q].get("parsed_option"), natural["rows"][d].get("parsed_option")
        if own in (None, "None") or don in (None, "None") or str(own) == str(don):  # VSP stores unparsed as "None"
            continue
        n += 1
        adopt += str(r.get("parsed_option")) == str(don)
        keep += str(r.get("parsed_option")) == str(own)
    return dict(n=n, adoption=adopt / n if n else float("nan"), retention=keep / n if n else float("nan"))


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def stop_gate(run, rows):
    """H2 telemetry from the first latent segment of each question.
    forced runs with K > 4: share of questions whose model would emit <|latent_end|> right after the 4th latent
    (the training latent size) and the mean step of the first such prediction (would_exit_steps holds k = latents fed so far);
    auto runs: share that left latent mode through a predicted <|latent_end|> (vs the 8-latent cap) and the latent-count histogram."""
    segs = [r["latent_segments"][0] for r in rows if r.get("latent_segments")]
    out = {}
    q = run["quota"]
    if segs and q != "auto" and int(q) > 4:
        out["end_pred_after_4"] = mean([1.0 if 4 in s.get("would_exit_steps", []) else 0.0 for s in segs])
        out["first_end_pred_step"] = mean([min(s["would_exit_steps"]) for s in segs if s.get("would_exit_steps")])
    if segs and q == "auto":
        out["exit_by_model_end"] = mean([1.0 if s.get("exit_reason") == "model_latent_end" else 0.0 for s in segs])
        hist = defaultdict(int)
        for s in segs:
            hist[int(s["steps"])] += 1
        out["latent_count_hist"] = dict(sorted(hist.items()))
    return out


def describe(run):
    rows = list(run["rows"].values())
    n = len(rows)
    k = sum(bool(r["correct"]) for r in rows)
    acc = k / n if n else float("nan")
    se = math.sqrt(acc * (1 - acc) / n) if n else float("nan")
    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(bool(r["correct"]))
    steps = [s for r in rows for s in r.get("latent_steps", [])]
    out = dict(n=n, correct=k, acc=acc, se=se, wilson95=wilson(k, n),
               per_category={c: (sum(v) / len(v), len(v)) for c, v in sorted(cats.items())},
               mean_latent_steps=mean([r.get("latent_steps_total", 0) for r in rows]),
               entered_latent_frac=mean([1.0 if r.get("n_latent_segments", 0) > 0 else 0.0 for r in rows]),
               natural_norm=mean([s.get("natural_norm") for s in steps]),
               fed_norm=mean([s.get("fed_norm") for s in steps]),
               step_cos=mean([s.get("cos_natural_prev_natural") for s in steps]),
               max_cos_image=mean([s.get("max_cos_natural_image_token") for s in steps]),
               max_cos_null_image=mean([s.get("max_cos_natural_null_image_token") for s in steps]),
               early_end_pred=mean([1.0 if r["latent_segments"] and r["latent_segments"][0].get("would_exit_steps") else 0.0
                                    for r in rows]) if rows and "latent_segments" in rows[0] else float("nan"),
               unparsed_frac=mean([1.0 if r.get("parse_method") == "unparsed" else 0.0 for r in rows]),
               **stop_gate(run, rows),
               mean_gen_tokens=mean([r.get("n_generated_tokens") for r in rows]))
    pairs = defaultdict(list)
    for r in rows:
        if "pair_id" in r:
            pairs[r["pair_id"]].append(bool(r["correct"]))
    full = [v for v in pairs.values() if len(v) == 2]
    if full:
        out["pair_acc"] = sum(all(v) for v in full) / len(full)
        out["n_pairs"] = len(full)
    return out


def compare(ref, run, B, seed, margin):
    common = sorted(set(ref["rows"]) & set(run["rows"]))
    x = [int(bool(ref["rows"][q]["correct"])) for q in common]
    y = [int(bool(run["rows"][q]["correct"])) for q in common]
    b = sum(1 for xi, yi in zip(x, y) if xi == 1 and yi == 0)   # reference right, condition wrong
    c = sum(1 for xi, yi in zip(x, y) if xi == 0 and yi == 1)   # reference wrong, condition right
    n = len(common)
    delta = (sum(y) - sum(x)) / n if n else float("nan")
    ci95, ci90 = paired_bootstrap(x, y, B=B, seed=seed)
    return dict(n_paired=n, ref_acc=sum(x) / n if n else float("nan"), acc=sum(y) / n if n else float("nan"), delta=delta,
                b=b, c=c, p_mcnemar=binom_two_sided(b, c), ci95=ci95, ci90=ci90,
                equivalent=bool(n and -margin <= ci90[0] and ci90[1] <= margin))


def fmt_pct(x, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:+.2f}" if signed else f"{x * 100:.2f}"


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


PRIMARY_DEFAULT = ("target__K4__zero__forced", "target__K4__noise__forced", "target__K4__donor__forced",
                   "target__K4__pause__forced", "target__K4__mean__forced", "target__K0__none__forced",
                   "target__Kauto__suppress__natural")


def condition_label(r):
    return f"{r['model_mode']} · K={r['quota']} · {r['intervention']} · {r['entry']}" + ("" if r["decoding"] == "greedy" else f" · {r['decoding']}")


def sort_key(r):
    q = r["quota"]
    qn = -1 if q == "auto" else int(q)
    order = {"none": 0, "zero": 1, "noise": 2, "donor": 3, "pause": 4, "mean": 5, "suppress": 6}
    return (r["model_mode"] != "base", r["intervention"] != "none", qn, order.get(r["intervention"], 9), r["entry"], r["decoding"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--reference", default="target__K4__none__forced",
                    help="<model_mode>__K<quota>__<intervention>__<entry> used as the paired reference per benchmark")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--equiv_margin", type=float, default=0.03, help="equivalence margin in accuracy (0.03 = 3 points; PREREGISTRATION_MIRAGE.md §5)")
    ap.add_argument("--md_name", default="ABLATION_SUMMARY.md")
    ap.add_argument("--primary", default=",".join(PRIMARY_DEFAULT),
                    help="comma-separated <model_mode>__K<quota>__<intervention>__<entry> conditions forming the confirmatory "
                         "family; Holm is applied within this family per benchmark (PREREGISTRATION_MIRAGE.md §5)")
    a = ap.parse_args(argv)

    runs = load_runs(a.output_dir)
    if not runs:
        raise SystemExit(f"no .jsonl outputs in {a.output_dir}")
    by_bench = defaultdict(list)
    for r in runs.values():
        r["stats"] = describe(r)
        by_bench[r["benchmark"]].append(r)

    md = ["# Mirage ablation summary", "",
          f"Source: `{a.output_dir}` · reference per benchmark: `{a.reference}` · paired bootstrap B={a.bootstrap} (seed {a.seed}) · "
          f"equivalence margin ±{a.equiv_margin * 100:.1f} pts (90% CI) · McNemar exact two-sided; `p (Holm)` = Holm over all conditions of the benchmark, "
          f"`p (Holm, family)` = Holm over the confirmatory family `{a.primary}`.", ""]
    js = {}
    for bench in sorted(by_bench):
        rs = sorted(by_bench[bench], key=sort_key)
        ref_name = f"{bench}__{a.reference}"
        ref = runs.get(ref_name)
        comps = {}
        if ref:
            for r in rs:
                if r["name"] != ref_name:
                    comps[r["name"]] = compare(ref, r, a.bootstrap, a.seed, a.equiv_margin)
            adj = holm([c["p_mcnemar"] for c in comps.values()])
            for c, p in zip(comps.values(), adj):
                c["p_holm"] = p
            prim = [f"{bench}__{x}" for x in a.primary.split(",") if f"{bench}__{x}" in comps]
            for name, p in zip(prim, holm([comps[n]["p_mcnemar"] for n in prim])):
                comps[name]["p_holm_primary"] = p
        md += [f"## {bench}", ""]
        if not ref:
            md += [f"_Reference run `{ref_name}` not found; paired statistics omitted._", ""]
        md += ["| Condition | n | Acc (%) | SE | 95% Wilson | Δ vs ref (pts) | 95% bootstrap CI | McNemar b/c | p | p (Holm) | Equiv. | "
               "Latent steps | ‖h‖ | cos(h_t,h_t-1) | max cos(h,img) | Unparsed |",
               "|---|---:|---:|---:|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|"]
        for r in rs:
            s = r["stats"]
            c = comps.get(r["name"])
            is_ref = r["name"] == ref_name
            lab = condition_label(r) + (" **(ref)**" if is_ref else "") + ("" if r["complete"] else " *(incomplete)*")
            md.append("| " + " | ".join([
                lab, str(s["n"]), fmt_pct(s["acc"]), fmt_pct(s["se"]),
                f"[{fmt_pct(s['wilson95'][0])}, {fmt_pct(s['wilson95'][1])}]",
                "—" if c is None else fmt_pct(c["delta"], True),
                "—" if c is None else f"[{fmt_pct(c['ci95'][0], True)}, {fmt_pct(c['ci95'][1], True)}]",
                "—" if c is None else f"{c['b']}/{c['c']}",
                "—" if c is None else fmt(c["p_mcnemar"], 4),
                "—" if c is None else fmt(c["p_holm"], 4),
                "—" if c is None else ("yes" if c["equivalent"] else "no"),
                fmt(s["mean_latent_steps"], 2), fmt(s["natural_norm"], 1), fmt(s["step_cos"]), fmt(s["max_cos_image"]),
                fmt_pct(s["unparsed_frac"])]) + " |")
        md.append("")
        prim = [r for r in rs if comps.get(r["name"], {}).get("p_holm_primary") is not None]
        if prim:
            md += [f"Confirmatory family ({len(prim)} conditions; Holm within this family):", "",
                   "| Condition | n | Δ vs ref (pts) | 95% bootstrap CI | 90% bootstrap CI | p | p (Holm, family) | Equiv. |",
                   "|---|---:|---:|---|---|---:|---:|:---:|"]
            for r in prim:
                c = comps[r["name"]]
                md.append("| " + " | ".join([
                    condition_label(r), str(c["n_paired"]), fmt_pct(c["delta"], True),
                    f"[{fmt_pct(c['ci95'][0], True)}, {fmt_pct(c['ci95'][1], True)}]",
                    f"[{fmt_pct(c['ci90'][0], True)}, {fmt_pct(c['ci90'][1], True)}]",
                    fmt(c["p_mcnemar"], 4), fmt(c["p_holm_primary"], 4), "yes" if c["equivalent"] else "no"]) + " |")
            md.append("")
        gate = [r for r in rs if "end_pred_after_4" in r["stats"] or "exit_by_model_end" in r["stats"]]
        if gate:
            md += ["Stop-gate telemetry (H2), first latent segment: forced K>4 runs report the share of questions whose model "
                   "would emit <|latent_end|> right after the 4th latent and the mean step of its first such prediction; auto runs "
                   "report the share leaving latent mode by a predicted <|latent_end|> and the latent-count histogram:", "",
                   "| Condition | End pred. after 4 (%) | First end-pred. step | Exit by <\\|latent_end\\|> (%) | Latent counts |",
                   "|---|---:|---:|---:|---|"]
            for r in gate:
                s = r["stats"]
                md.append("| " + " | ".join([
                    condition_label(r), fmt_pct(s.get("end_pred_after_4")), fmt(s.get("first_end_pred_step"), 2),
                    fmt_pct(s.get("exit_by_model_end")),
                    ", ".join(f"{k}:{v}" for k, v in s.get("latent_count_hist", {}).items()) or "—"]) + " |")
            md.append("")
        cats = sorted({cat for r in rs for cat in r["stats"]["per_category"]})
        if 1 < len(cats) <= 12:
            md += ["Per-category accuracy (%):", "", "| Condition | " + " | ".join(cats) + " |", "|---|" + "---:|" * len(cats)]
            for r in rs:
                pc = r["stats"]["per_category"]
                md.append(f"| {condition_label(r)} | " + " | ".join(fmt_pct(pc[c][0]) if c in pc else "—" for c in cats) + " |")
            md.append("")
        if any("pair_acc" in r["stats"] for r in rs):
            md += ["MMVP pair accuracy (both questions of a pair correct):", "", "| Condition | pairs | Pair acc (%) |", "|---|---:|---:|"]
            for r in rs:
                if "pair_acc" in r["stats"]:
                    md.append(f"| {condition_label(r)} | {r['stats']['n_pairs']} | {fmt_pct(r['stats']['pair_acc'])} |")
            md.append("")
        h3 = {r["name"]: h3_contrast(r, a.bootstrap, a.seed) for r in rs if r["stats"]["mean_latent_steps"] > 0}
        h3 = {k: v for k, v in h3.items() if v["n"]}
        if h3:
            md += ["Latent–image alignment (H3): per-question mean over latent steps of max cos(latent, own image tokens) minus "
                   "max cos(latent, image tokens of a different question); 95% bootstrap CI:", "",
                   "| Condition | n | own − other | 95% CI |", "|---|---:|---:|---|"]
            for r in rs:
                if r["name"] in h3:
                    v = h3[r["name"]]
                    md.append(f"| {condition_label(r)} | {v['n']} | {fmt(v['mean'], 4)} | [{fmt(v['ci95'][0], 4)}, {fmt(v['ci95'][1], 4)}] |")
            md.append("")
        don = {}
        for r in rs:
            if r["intervention"] == "donor":
                nat = runs.get(f"{bench}__target__K{r['quota']}__none__forced")
                if nat:
                    don[r["name"]] = donor_adoption(r, nat)
        if don:
            md += ["Donor runs (H1): among questions whose donor's natural answer differs from their own natural answer, the share "
                   "answering with the donor's answer (adoption) or their own (retention):", "",
                   "| Condition | n | Adoption (%) | Retention (%) |", "|---|---:|---:|---:|"]
            for r in rs:
                if r["name"] in don:
                    v = don[r["name"]]
                    md.append(f"| {condition_label(r)} | {v['n']} | {fmt_pct(v['adoption'])} | {fmt_pct(v['retention'])} |")
            md.append("")
        js[bench] = {r["name"]: dict(condition=condition_label(r), stats=r["stats"], vs_reference=comps.get(r["name"]),
                                     h3_contrast=h3.get(r["name"]), donor=don.get(r["name"]), complete=r["complete"]) for r in rs}

    md += ["## Notes", "",
           "- Δ, b/c and p use only questions present in both the condition and the reference (paired).",
           "- b = reference correct & condition wrong; c = reference wrong & condition correct.",
           "- ‖h‖, cos(h_t,h_t-1), max cos(h,img): means over latent steps of the natural latent (the hidden state the model "
           "would feed), its cosine to the previous step, and its maximum cosine to any image-token embedding of the prompt.",
           "- `Equiv.` = yes when the 90% paired-bootstrap CI of Δ falls within the equivalence margin (two one-sided tests).", ""]
    Path(a.output_dir, a.md_name).write_text("\n".join(md))
    Path(a.output_dir, "ablation_summary.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
