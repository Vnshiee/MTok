"""
Unified, Self-Verifying Benchmark Data Loader for LVR Ablation Study.

Supports the exact three evaluation benchmarks from LVR (arXiv:2509.24251):
1. V* Bench (`craigwu/vstar_bench`, N=191):
   - Direct Attributes (`direct_attributes`, N=115)
   - Relative Position (`relative_position`, N=76)
   - Includes ground-truth target object bounding boxes (`bboxes_xywh` and `bboxes_norm`)
     from each item's paired JSON annotation file (`sa_XXXX.json`).
2. MMVP (`MMVP/MMVP`, N=300):
   - 150 paired visual pattern contrast instances (300 questions).
   - Preserves both raw `(a)/(b)` query formatting and `evaluation.py`'s `A./B.` mutated formatting.
3. BLINK (`BLINK-Benchmark/BLINK`, `val` split):
   - Core 5 subtasks reported in Table 1 (`Counting`, `IQ_Test`, `Jigsaw`,
     `Relative_Reflectance`, `Spatial_Relation`, N=697).
   - Extended 6 subtasks present in `evaluation.py` L340 (`Multi-view_Reasoning`,
     `Object_Localization`, `Relative_Depth`, `Semantic_Correspondence`,
     `Visual_Correspondence`, `Visual_Similarity`, N=859).
"""

import csv
import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image

CORE_5_BLINK_CONFIGS = [
    "Counting",
    "IQ_Test",
    "Jigsaw",
    "Relative_Reflectance",
    "Spatial_Relation",
]

EXTENDED_6_BLINK_CONFIGS = [
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Semantic_Correspondence",
    "Visual_Correspondence",
    "Visual_Similarity",
]

LVR_EVAL_INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."
LVR_SFT_BBOX_INSTRUCTION = (
    " Please provide the bounding box coordinate of the region that can help you answer the question better."
)


def bbox_norm_to_token_idxs(
    bboxes_norm: List[Tuple[float, float, float, float]],
    image_grid_thw: Tuple[int, int, int],
) -> List[int]:
    """
    Exact replica of `IterableSupervisedDatasetLVR.bbox_to_token_idxs`
    (repo/src/dataset/lvr_sft_dataset_packed.py L171-L208).

    Given normalized bounding boxes `[x0, y0, x1, y1]` in [0, 1] and `image_grid_thw = (t, h, w)`
    (where `h, w` are 14x14 ViT patch counts before 2x2 spatial merging, so total merged visual
    tokens = `(h // 2) * (w // 2)`), returns the sorted unique list of merged visual token indices
    falling inside the union of the bounding boxes.
    """
    _, h, w = image_grid_thw
    h2, w2 = h // 2, w // 2
    token_set = set()
    for bbox in bboxes_norm:
        x0, y0, x1, y1 = bbox
        x0_grid = max(0, min(int(np.floor(x0 * w)), w - 1))
        x1_grid = max(0, min(int(np.ceil(x1 * w)), w))
        y0_grid = max(0, min(int(np.floor(y0 * h)), h - 1))
        y1_grid = max(0, min(int(np.ceil(y1 * h)), h))

        x0_token = x0_grid // 2
        x1_token = (x1_grid + 1) // 2
        y0_token = y0_grid // 2
        y1_token = (y1_grid + 1) // 2

        # Ensure at least 1 token if box is very small
        if x1_token <= x0_token:
            x1_token = min(w2, x0_token + 1)
        if y1_token <= y0_token:
            y1_token = min(h2, y0_token + 1)

        for yy in range(y0_token, y1_token):
            for xx in range(x0_token, x1_token):
                idx = int(yy * w2 + xx)
                if 0 <= idx < h2 * w2:
                    token_set.add(idx)
    return sorted(token_set)


def load_vstar_bench(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Downloads and loads `craigwu/vstar_bench` (N=191) along with per-sample ground-truth
    bounding box annotations from the paired `.json` files.
    """
    repo_dir = snapshot_download(
        repo_id="craigwu/vstar_bench",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    jsonl_path = os.path.join(repo_dir, "test_questions.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing test_questions.jsonl in {repo_dir}")

    items: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rel_img_path = row["image"]
            abs_img_path = os.path.join(repo_dir, rel_img_path)
            if not os.path.exists(abs_img_path):
                raise FileNotFoundError(f"Missing V* image: {abs_img_path}")

            # Load paired JSON annotation for ground-truth bbox
            base_no_ext = os.path.splitext(abs_img_path)[0]
            json_ann_path = base_no_ext + ".json"
            bboxes_xywh = []
            bboxes_norm = []
            target_objects = []
            img_size = None
            if os.path.exists(json_ann_path):
                with open(json_ann_path, "r", encoding="utf-8") as jf:
                    ann = json.load(jf)
                bboxes_xywh = ann.get("bbox", [])
                target_objects = ann.get("target_object", [])
                with Image.open(abs_img_path) as im:
                    w_img, h_img = im.size
                    img_size = (w_img, h_img)
                for bx, by, bw, bh in bboxes_xywh:
                    x0 = max(0.0, min(1.0, float(bx) / w_img))
                    y0 = max(0.0, min(1.0, float(by) / h_img))
                    x1 = max(0.0, min(1.0, float(bx + bw) / w_img))
                    y1 = max(0.0, min(1.0, float(by + bh) / h_img))
                    bboxes_norm.append((x0, y0, x1, y1))

            # Parse choices from text
            raw_text = row["text"]
            # Note: `raw_text` in test_questions.jsonl already ends with:
            # "\nAnswer with the option's letter from the given choices directly."
            # `evaluation.py` L197 does `text = dat['text'] + task_instruction` (duplicating it).
            # We store both `text_official_eval` (exact match to `evaluation.py`) and `text_clean`.
            text_official_eval = raw_text + LVR_EVAL_INSTRUCTION
            text_clean = raw_text

            # Extract options dict {letter: option_text} for robust parser & CircularEval
            options_dict = {}
            for m in re.finditer(r"^\(([A-D])\)\s*(.+)$", raw_text, flags=re.MULTILINE):
                options_dict[m.group(1)] = m.group(2).strip()

            qid = int(row["question_id"])
            category = row.get("category") or ("direct_attributes" if qid <= 114 else "relative_position")

            items.append(
                {
                    "benchmark": "vstar",
                    "question_id": str(row["question_id"]),
                    "int_id": qid,
                    "image": abs_img_path,
                    "image_size": img_size,
                    "text_official_eval": text_official_eval,
                    "text_clean": text_clean,
                    "options_dict": options_dict,
                    "label": row["label"].strip().upper(),
                    "category": category,
                    "bboxes_xywh": bboxes_xywh,
                    "bboxes_norm": bboxes_norm,
                    "target_objects": target_objects,
                }
            )
    return items


def load_mmvp_bench(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Downloads and loads `MMVP/MMVP` (N=300) from HuggingFace Hub.
    Replicates `load_mmvp_dataset` and `evaluate_mmvp` in `repo/evaluation/evaluation.py` L258-L276 & L369-L395.
    """
    repo_dir = snapshot_download(
        repo_id="MMVP/MMVP",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    csv_file = os.path.join(repo_dir, "Questions.csv")
    # Note: HuggingFace repo `MMVP/MMVP` stores images in `MMVP Images/` (with a space)
    image_dir = os.path.join(repo_dir, "MMVP Images")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(repo_dir, "MMVP_Images")

    items: List[Dict[str, Any]] = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"])
            img_path = os.path.join(image_dir, f"{idx}.jpg")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing MMVP image: {img_path}")

            raw_query = row["Question"] + "\nOptions:\n" + row["Options"]
            # `evaluation.py` L262-L264 replaces `(a)` -> `A.` and `(b)` -> `B.`
            mutated_query = raw_query.replace("(a)", "A.").replace("(b)", "B.")
            text_official_eval = mutated_query + LVR_EVAL_INSTRUCTION
            text_clean = raw_query + LVR_EVAL_INSTRUCTION

            raw_label = row["Correct Answer"].strip()
            if raw_label in ["(a)", "(b)", "(A)", "(B)"]:
                label = raw_label.upper()[1]
            else:
                label = raw_label.upper()[0]

            # Parse options (a) ... (b) ...
            options_dict = {}
            parts = re.split(r"\([abAB]\)\s*", row["Options"])
            if len(parts) >= 3:
                options_dict["A"] = parts[1].strip()
                options_dict["B"] = parts[2].strip()

            items.append(
                {
                    "benchmark": "mmvp",
                    "question_id": str(idx),
                    "int_id": idx,
                    "image": img_path,
                    "image_size": None,
                    "text_official_eval": text_official_eval,
                    "text_clean": text_clean,
                    "options_dict": options_dict,
                    "label": label,
                    "category": f"pair_{(idx - 1) // 2 + 1}",
                    "bboxes_xywh": [],
                    "bboxes_norm": [],
                    "target_objects": [],
                }
            )
    return items


def load_blink_bench(
    include_extended: bool = False,
    configs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Loads `BLINK-Benchmark/BLINK` (`val` split).
    By default loads the 5 core subtasks reported in Table 1 of the LVR paper (N=697).
    If `include_extended=True`, also loads the 6 extended subtasks from `evaluation.py` L340.
    """
    if configs is None:
        configs = list(CORE_5_BLINK_CONFIGS)
        if include_extended:
            configs.extend(EXTENDED_6_BLINK_CONFIGS)

    items: List[Dict[str, Any]] = []
    letters = string.ascii_uppercase
    for cfg in configs:
        ds = load_dataset("BLINK-Benchmark/BLINK", cfg)["val"]
        for dat in ds:
            idx = dat["idx"]
            choices = dat["choices"]
            paired = list(zip(letters, choices))
            option_string = ""
            options_dict = {}
            for letter, choice in paired:
                option_string += f"{letter}. {choice}\n"
                options_dict[letter] = str(choice).strip()

            ans_raw = dat["answer"].strip()
            if len(ans_raw) > 1 and ans_raw.startswith("("):
                ans = ans_raw[1].upper()
            else:
                ans = ans_raw[0].upper()

            images = []
            for k in ["image_1", "image_2", "image_3", "image_4"]:
                if k in dat and dat[k] is not None:
                    images.append(dat[k])

            base_query = dat["question"] + "\nOptions:\n" + option_string
            text_official_eval = base_query + LVR_EVAL_INSTRUCTION

            items.append(
                {
                    "benchmark": "blink",
                    "question_id": str(idx),
                    "int_id": len(items),
                    "image": images if len(images) > 1 else images[0],
                    "image_size": None,
                    "text_official_eval": text_official_eval,
                    "text_clean": text_official_eval,
                    "options_dict": options_dict,
                    "label": ans,
                    "category": cfg,
                    "bboxes_xywh": [],
                    "bboxes_norm": [],
                    "target_objects": [],
                }
            )
    return items


if __name__ == "__main__":
    print("Verifying LVR Benchmark Loaders...")
    vstar = load_vstar_bench()
    da = [x for x in vstar if x["category"] == "direct_attributes"]
    rp = [x for x in vstar if x["category"] == "relative_position"]
    with_bbox = [x for x in vstar if len(x["bboxes_norm"]) > 0]
    print(f"[V* Bench] Total={len(vstar)} (Direct Attributes={len(da)}, Relative Position={len(rp)}), With GT BBox={len(with_bbox)}")

    mmvp = load_mmvp_bench()
    print(f"[MMVP] Total={len(mmvp)}, Sample 0 Label={mmvp[0]['label']}, Options={mmvp[0]['options_dict']}")

    blink5 = load_blink_bench(include_extended=False)
    by_cat = {}
    for item in blink5:
        by_cat[item["category"]] = by_cat.get(item["category"], 0) + 1
    print(f"[BLINK Core 5] Total={len(blink5)}, Breakdown={by_cat}")

