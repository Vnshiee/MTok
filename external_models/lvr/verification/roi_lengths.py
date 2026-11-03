import json, math, sys, glob, ast, textwrap, os
import numpy as np
REPO, DATA = sys.argv[1], sys.argv[2]
src = open(f"{REPO}/src/dataset/lvr_sft_dataset_packed.py").read()
fn = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "bbox_to_token_idxs"][0]
ns = {"np": np}; exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
bbox_to_token_idxs = ns["bbox_to_token_idxs"]
class G:  # image_grid_thw[0].tolist() -> [t,h,w]
    def __init__(s, h, w): s.v = [1, h, w]
    def tolist(s): return s.v
def smart_resize(h, w, factor=28, min_pixels=128*28*28, max_pixels=5120*28*28):   # qwen_vl_utils formula
    hb, wb = max(factor, round(h/factor)*factor), max(factor, round(w/factor)*factor)
    if hb*wb > max_pixels:
        b = math.sqrt(h*w/max_pixels); hb, wb = math.floor(h/b/factor)*factor, math.floor(w/b/factor)*factor
    elif hb*wb < min_pixels:
        b = math.sqrt(min_pixels/(h*w)); hb, wb = math.ceil(h*b/factor)*factor, math.ceil(w*b/factor)*factor
    return hb, wb
allL, per = [], {}
for f in sorted(glob.glob(f"{DATA}/*.jsonl")):
    L = []
    for line in open(f):
        d = json.loads(line); W, H = d["width"], d["height"]
        hb, wb = smart_resize(H, W)
        grid = [G(hb // 14, wb // 14)]
        for (x0, y0, x1, y1) in d["bboxs"][:1]:     # first bbox (LVR example has one bbox per sample)
            n = len(bbox_to_token_idxs(None, [[x0/W, y0/H, x1/W, y1/H]], grid)[0])
            L.append(n)
    per[os.path.basename(f)[:-6]] = L; allL += L
a = np.array(allL)
print(f"N={len(a)}  mean={a.mean():.1f}  median={np.median(a):.0f}  p10={np.percentile(a,10):.0f}  p90={np.percentile(a,90):.0f}  max={a.max()}")
for K in (4, 8, 16): print(f"  share of training samples with T_v <= {K:2d}: {np.mean(a<=K)*100:5.1f}%   (fraction of ROI covered by K steps, median: {np.median(np.minimum(K/a,1))*100:.1f}%)")
for k, L in per.items():
    L = np.array(L); print(f"  {k:16s} n={len(L):6d} median={np.median(L):5.0f} mean={L.mean():7.1f}")
