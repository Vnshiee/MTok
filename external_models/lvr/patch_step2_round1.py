import re

# -------------------------------------------------------------
# 1. Patch Gaps_LVR.md
# -------------------------------------------------------------
gaps_path = "/home/devraj/MTok/external_models/lvr/Gaps_LVR.md"
with open(gaps_path, "r", encoding="utf-8") as f:
    gaps = f.read()

# Issue 2: \t tab escapes corrupted to actual tabs (0x09)
gaps = gaps.replace("\text{bf}{67.0}", "\\textbf{67.0}")
gaps = gaps.replace("\times", "\\times")
gaps = gaps.replace("\t", "    ") # replace any remaining stray literal tabs

# Issue 1 & 11: evaluation.py instruction formatting in Gaps_LVR.md
old_eval_snippet = """```python
def get_task_instruction(bench_name):
    if bench_name in ["vstar", "mmvp", "blink"]:
        return "
Answer with the option's letter from the given choices directly."
```"""
new_eval_snippet = """```python
def get_task_instruction(bench_name):
    if bench_name == "vstar":
        return "\\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "mmvp":
        return "\\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "blink":
        return "\\nAnswer with the option's letter from the given choices directly."
```"""
gaps = gaps.replace(old_eval_snippet, new_eval_snippet)
# Fallback regex in case formatting differed slightly
gaps = re.sub(
    r'def get_task_instruction\(bench_name\):.*?"\s*\nAnswer with the option\'s letter from the given choices directly\."',
    r'def get_task_instruction(bench_name):\n    if bench_name == "vstar":\n        return "\\\\nAnswer with the option\'s letter from the given choices directly."\n    elif bench_name == "mmvp":\n        return "\\\\nAnswer with the option\'s letter from the given choices directly."\n    elif bench_name == "blink":\n        return "\\\\nAnswer with the option\'s letter from the given choices directly."',
    gaps,
    flags=re.DOTALL
)

# Issue 12: Bash export syntax wrongly attributed to python file
gaps = gaps.replace('export access_key_id = "..."', 'access_key_id = "..."')
gaps = gaps.replace('export secret_access_key = "..."', 'secret_access_key = "..."')

# Issue 13: stage2_3b.sh -> scripts/finetune_lvr_stage2_3b.sh
gaps = gaps.replace("`stage2_3b.sh`", "[`scripts/finetune_lvr_stage2_3b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh)")

# Issue 14: URL anchors
gaps = gaps.replace(
    "file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L52-L58",
    "file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24"
)
gaps = gaps.replace(
    "file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1013-L1055",
    "file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1045-L1049"
)

with open(gaps_path, "w", encoding="utf-8") as f:
    f.write(gaps)

print("Patched Gaps_LVR.md successfully.")
