"""
Consistency checks on Monet's results tables (paper_src/sec/4_experiment.tex, 3_method.tex):
  (1) every value equals round(k/N*100, 2) for an integer k, given benchmark sizes;
  (2) 'Overall' equals the size-weighted mean of its sub-scores (V*: 115 attribute + 76 spatial; HRBench: 400 FSP + 400 FCP).
Usage: python3 table_checks.py <path to external_models/monet/paper_src/sec>
"""
import re, sys
d = sys.argv[1]
N = {'V*': 191, 'V*-Attr': 115, 'V*-Spat': 76, 'HR4K': 800, 'HR4K-FSP': 400, 'HR4K-FCP': 400,
     'HR8K': 800, 'HR8K-FSP': 400, 'HR8K-FCP': 400, 'MME-Lite': 1919, 'VP': 1168}
ok = lambda x, n: any(abs(round(k / n * 100 + 1e-12, 2) - x) < 1e-9 for k in range(n + 1))
num = lambda c: float(re.sub(r'[^0-9.]', '', c.replace('\\textbf', '')) or 'nan')
main_cols = ['V*', 'V*-Attr', 'V*-Spat', 'HR4K', 'HR4K-FSP', 'HR4K-FCP', 'HR8K', 'HR8K-FSP', 'HR8K-FCP', 'MME-Lite', None, None]
abl_cols = ['V*', 'HR4K', 'HR8K', 'MME-Lite', 'VP']
checked, bad = 0, []
for f in ('4_experiment.tex', '3_method.tex'):
    for line in open(f"{d}/{f}"):
        if line.strip().startswith('%') or '*' in line or 'Relative' in line:
            continue
        cells = [c.strip() for c in line.split('&')]
        if len(cells) == 13 and 'Overall' not in line:
            v = [num(c) for c in cells[1:]]
            for c, x in zip(main_cols, v):
                if c: checked += 1; bad += [] if ok(x, N[c]) else [(f, cells[0][:30], c, x)]
            print(f"  {cells[0][:26]:26s} V* {v[0]:.2f} vs weighted {(115*v[1]+76*v[2])/191:.2f} | "
                  f"HR4K {v[3]:.2f} vs {(v[4]+v[5])/2:.3f} | HR8K {v[6]:.2f} vs {(v[7]+v[8])/2:.3f}")
        elif len(cells) == 6 and 'Model' not in line:
            for c, x in zip(abl_cols, [num(c) for c in cells[1:]]):
                checked += 1; bad += [] if ok(x, N[c]) else [(f, cells[0][:30], c, x)]
        elif len(cells) == 7 and 'Overall' not in line:
            checked += 1; x = num(cells[1]); bad += [] if ok(x, N['VP']) else [(f, cells[0][:30], 'VP', x)]
print(f"values checked: {checked}; not achievable as k/N: {len(bad)}")
for b in bad: print("  ", b)
