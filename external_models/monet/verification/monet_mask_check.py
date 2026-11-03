"""Run Monet's own build_4d_attn (src/utils.py) on a synthetic SFT Stage-2 student sequence and report visibility."""
import ast, sys, textwrap, torch
REPO = sys.argv[1]
src = open(f"{REPO}/src/utils.py").read()
tree = ast.parse(src)
ns = {"torch": torch}
# load every top-level function in utils.py that does not need heavy imports (all are pure torch helpers or unused here)
for n in tree.body:
    if isinstance(n, ast.FunctionDef):
        try: exec(textwrap.dedent(ast.get_source_segment(src, n)), ns)
        except Exception: pass
tid = dict(v_start=torch.tensor([1]), v_end=torch.tensor([2]), img_pad=torch.tensor([3]), abs_start=torch.tensor([4]),
           abs_end=torch.tensor([5]), abs_pad=torch.tensor([6]), obs_start=torch.tensor([7]), obs_end=torch.tensor([8]),
           ans_start=torch.tensor([9, 10]))
lab, seq = [], []
def add(t, l, n=1):
    for _ in range(n): seq.append(t); lab.append(l)
add(30, 'sys'); add(1, 'qimg_start'); add(3, 'Q_IMG', 3); add(2, 'qimg_end'); add(31, 'question')
add(9, 'ans'); add(10, 'ans'); add(32, 'text')
add(1, 'aux_start'); add(3, 'AUX_IMG', 3); add(2, 'aux_end')
add(4, 'latent_start'); add(6, 'LATENT', 2); add(5, 'latent_end')
add(7, 'obs_start'); add(33, 'OBS', 2); add(8, 'obs_end'); add(34, 'text_after')
ids = torch.tensor([seq]); pad = torch.ones_like(ids)
out = ns['build_4d_attn'](input_ids=ids, pad_mask=pad, token_ids=tid, return_type='bool')
m = out[0] if isinstance(out, tuple) else out
while m.dim() > 2: m = m[0]
aux = [i for i, l in enumerate(lab) if l == 'AUX_IMG']; qimg = [i for i, l in enumerate(lab) if l == 'Q_IMG']
print('True = may attend')
for qi, l in enumerate(lab):
    if qi < aux[0]: continue
    print(f"{qi:2d} {l:12s} sees AUX_IMG {[int(m[qi, k]) for k in aux]}  sees Q_IMG {[int(m[qi, k]) for k in qimg]}  sees LATENT {[int(m[qi, k]) for k, x in enumerate(lab) if x == 'LATENT']}")
