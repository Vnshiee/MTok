import re, sys, os, collections
doc, repo, lvr = sys.argv[1], sys.argv[2], sys.argv[3]
L = open(doc).read().splitlines()
link = re.compile(r'external_models/lvr/((?:repo|paper_src|hf_metadata)/[^)#\s]+)#L(\d+)(?:-L?(\d+))?')
def resolve(p): return os.path.join(repo, p[5:]) if p.startswith('repo/') else os.path.join(lvr, p)
stats = collections.Counter(); bad = []
cites = 0; range_bad = []
for i, line in enumerate(L):
    for m in link.finditer(line):
        cites += 1
        fp = resolve(m.group(1)); a = int(m.group(2)); b = int(m.group(3) or a)
        if not os.path.exists(fp): range_bad.append((i+1, m.group(1), 'missing file')); continue
        n = len(open(fp, errors='ignore').read().splitlines())
        if b > n or a > b: range_bad.append((i+1, m.group(1), f'L{a}-{b} > {n} lines'))
    ms = list(link.finditer(line))
    if not ms: continue
    # code block within next 12 lines
    blk = []; j = i + 1; inside = False
    while j < min(len(L), i + 14):
        s = L[j].strip()
        if s.startswith('```'):
            if inside: break
            inside = True; j += 1; continue
        if inside: blk.append(L[j])
        elif s.startswith('- **') or s.startswith('#'): break
        j += 1
    if not blk: continue
    m = ms[-1]; fp = resolve(m.group(1))
    if not os.path.exists(fp): continue
    src = open(fp, errors='ignore').read().splitlines()
    a = int(m.group(2)); b = int(m.group(3) or a)
    rng = "\n".join(src[max(0, a-3):b+2]); full = "\n".join(src)
    for cl in blk:
        c = cl.strip()
        if not c or c.startswith(('#', '...', '//', '%')) or c in ('...', ')', ']', '}', '):'): continue
        if c in rng: stats['in_range'] += 1
        elif c in full:
            stats['in_file_wrong_lines'] += 1
            bad.append((i+1, m.group(1), f'L{a}-{b}', 'actual L' + ",".join(str(k+1) for k, x in enumerate(src) if c in x)[:40], c[:70]))
        else:
            stats['not_in_file'] += 1; bad.append((i+1, m.group(1), f'L{a}-{b}', 'ABSENT', c[:70]))
print(os.path.basename(doc), 'citations:', cites, 'bad ranges:', len(range_bad), dict(stats))
for r in range_bad[:15]: print('  RANGE', r)
for r in bad[:40]: print('  SNIP', r)
