import re
import os

print("=== FORENSIC AUDIT: MIRAGE (arXiv:2506.17218) ===")

src_dir = "/home/devraj/MTok/external_models/mirage/repo/src"
for f in ["trainer.py", "task.py", "main.py"]:
    fpath = os.path.join(src_dir, f)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
            code = fp.read()
        print(f"--- {f} (lines: {len(code.splitlines())}) ---")
        for term in ["stage", "visual_loss", "imagery", "detach", "relax", "mental"]:
            cnt = len(re.findall(r"\b" + term + r"\b", code, re.I))
            print(f"  '{term}': {cnt}")

