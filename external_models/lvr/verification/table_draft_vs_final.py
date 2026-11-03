"""
Compare the commented-out draft Table 1 (paper_src/tables/main_table.tex L20-L40, 2-decimal values)
with the published Table 1 (L71-L80, 1-decimal values), and check that every reported
percentage is achievable as k/N on the benchmark split sizes.

Usage: python3 table_draft_vs_final.py <path to external_models/lvr/paper_src>
"""
import re, sys

COLS = ['V*', 'DA', 'RP', 'MMVP', 'Count', 'IQ', 'Jig', 'RelRef', 'SpRel']
# split sizes: V* = 115 direct-attribute + 76 relative-position; MMVP = 300; BLINK val splits
N = {'V*': 191, 'DA': 115, 'RP': 76, 'MMVP': 300, 'Count': 120, 'IQ': 150, 'Jig': 150, 'RelRef': 134, 'SpRel': 143}

def rows(tex, commented):
    out = {}
    for line in tex.splitlines():
        s = line.strip()
        if commented != s.startswith('%'):
            continue
        s = s.lstrip('%').strip()
        cells = [c.strip() for c in s.rstrip('\\').strip().rstrip('\\').split('&')]
        if len(cells) != 10:
            continue
        name = re.sub(r'\\cellcolor\{[^}]*\}', '', cells[0]).strip()
        try:
            vals = [float(re.sub(r'[^0-9.]', '', c)) for c in cells[1:]]
        except ValueError:
            continue
        out[name] = vals
    return out

tex = open(f"{sys.argv[1]}/tables/main_table.tex").read()
seven_b = tex[tex.index('7B Open Models'):tex.index('\\caption{\\small \\textbf{Main Table}')]
draft = rows(seven_b, commented=True)
final = rows(tex[tex.index('Open Models based on Qwen2.5-VL-7B'):], commented=False)
alias = {'VISION-R1': 'Vision-R1'}
draft = {alias.get(k, k): v for k, v in draft.items()}
lvr = lambda k: k.replace('\\modelname', 'LVR')
draft = {lvr(k): v for k, v in draft.items()}
final = {lvr(k): v for k, v in final.items()}

print("Cells where final != 1-decimal rounding of the draft value:")
for m in final:
    if m not in draft:
        continue
    for c, d, f in zip(COLS, draft[m], final[m]):
        if abs(round(d + 1e-9, 1) - f) > 0.051:
            print(f"  {m:16s} {c:6s} draft={d:6.2f} final={f:5.1f}  delta={f - d:+.2f}  (= {round((f - d) / 100 * N[c], 1):+} items of {N[c]})")

def winners(t):
    return {c: sorted(k for k, v in t.items() if abs(v[i] - max(x[i] for x in t.values())) < 1e-9) for i, c in enumerate(COLS)}
wd, wf = winners({k: v for k, v in draft.items() if k in final}), winners(final)
print("\nBest open 7B model per column (draft -> final):")
for c in COLS:
    flag = "   <-- changed" if wd[c] != wf[c] else ""
    print(f"  {c:6s} {wd[c]} -> {wf[c]}{flag}")

print("\nFeasibility (value == round(k/N*100) for some integer k), LVR + baseline rows of both tables:")
for label, t in (("draft", draft), ("final", final)):
    for m, vals in t.items():
        for c, x in zip(COLS, vals):
            dec = 1 if label == "final" else 2
            ok = any(abs(round(k / N[c] * 100, dec) - x) < 1e-9 for k in range(N[c] + 1))
            if not ok:
                exact2 = [k for k in range(N[c] + 1) if abs(round(k / N[c] * 100, 2) - (x - 0.05)) < 0.051]
                print(f"  {label:5s} {m:16s} {c:6s} {x} not k/{N[c]} at {dec} dp "
                      f"(consistent with double rounding of a 2-dp draft value)" if exact2 or label == "final" else "")
