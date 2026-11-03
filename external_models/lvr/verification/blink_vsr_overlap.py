"""
Contamination check for Gaps_LVR.md §5.1: does BLINK Spatial Relation (val, 143 items) overlap with
the VSR subset of Visual-CoT, which is part of LVR's SFT data?

Step 1: normalized exact question-string overlap.
Step 2: for question matches, compare the BLINK image against every COCO image of the matching
        Visual-CoT/VSR training item with a 256-bit difference hash (dHash). Near-duplicates are <= ~20.

Usage: python3 blink_vsr_overlap.py <dir with Visual-CoT metadata vsr.jsonl and gqa.jsonl> <out_dir>
  metadata:  https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/main/metadata  (*_cot_train.jsonl)
  BLINK:     fetched from the HF datasets-server rows API (config Spatial_Relation, split val)
Needs: numpy, pillow, network access to huggingface.co and images.cocodataset.org
"""
import io, json, os, re, sys, urllib.request
import numpy as np
from PIL import Image

meta, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()

rows = []
for off in (0, 100):
    url = ("https://datasets-server.huggingface.co/rows?dataset=BLINK-Benchmark/BLINK"
           f"&config=Spatial_Relation&split=val&offset={off}&length=100")
    rows += [r["row"] for r in json.loads(fetch(url))["rows"]]
assert len(rows) == 143, len(rows)

norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()
vsr = [json.loads(l) for l in open(f"{meta}/vsr.jsonl")]
gqa_q = {norm(json.loads(l)["question"]) for l in open(f"{meta}/gqa.jsonl")}
by_q = {}
for d in vsr:
    by_q.setdefault(norm(d["question"]), []).append(d)
matched = [r for r in rows if norm(r["question"]) in by_q]
print(f"question overlap with Visual-CoT/VSR train: {len(matched)}/143; with Visual-CoT/GQA: "
      f"{sum(norm(r['question']) in gqa_q for r in rows)}/143")

def dhash(im, n=16):
    a = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).flatten()

results = []
for r in matched:
    bh = dhash(Image.open(io.BytesIO(fetch(r["image_1"]["src"]))))
    best = None
    for d in by_q[norm(r["question"])]:
        im = None
        for split in ("train2017", "val2017"):
            try:
                im = Image.open(io.BytesIO(fetch(f"http://images.cocodataset.org/{split}/{d['image']}")))
                break
            except Exception:
                pass
        if im is not None:
            dist = int((dhash(im) != bh).sum())
            if best is None or dist < best[0]:
                best = (dist, d["image"], d["answer"])
    results.append({"blink_idx": r["idx"], "question": r["question"], "nearest_vsr_train": best})

dists = sorted(x["nearest_vsr_train"][0] for x in results if x["nearest_vsr_train"])
print(f"image near-duplicates (dHash Hamming <= 20 of 256): {sum(d <= 20 for d in dists)}/{len(results)}")
print("nearest-image Hamming distances:", dists)
json.dump(results, open(f"{out}/blink_vsr_overlap.json", "w"), indent=1)
