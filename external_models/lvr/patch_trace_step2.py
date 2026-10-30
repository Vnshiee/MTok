trace_path = "/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md"
with open(trace_path, "r", encoding="utf-8") as f:
    text = f.read()

# Issue 1: broken \n in code blocks
text = text.replace(
    "pattern = r'\n?(?:<\|im_start\|>)?(?:<\|lvr_start\|>)?<\|lvr\|>+(?:<\|lvr_end\|>)?\\s*<answer>\\s*([A-Za-z0-9])\\s*</answer>'",
    'pattern = r"\\n?(?:<\\|im_start\\|>)?(?:<\\|lvr_start\\|>)?<\\|lvr\\|>+(?:<\\|lvr_end\\|>)?\\s*<answer>\\s*([A-Za-z0-9])\\s*</answer>"'
)
text = text.replace(
    'return "\nAnswer with the option\'s letter from the given choices directly."',
    'return "\\nAnswer with the option\'s letter from the given choices directly."'
)

# Issue 3: M-RoPE citation in config.json
text = text.replace(
    "config.json lines 24–28",
    "config.json lines 28–33"
)

# Issue 4: Projector freezing citation
text = text.replace(
    "Lines 42–44, 63–66 | `self.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`)",
    "Lines 41, 43 | Projector (`self.visual.merger`) is frozen in `src/train/train_lvr.py:43` and `src/train/train_grpo.py:41`"
)

# Issue 5: GRPO surrogate loss citation
text = text.replace(
    "Lines 1025–1035",
    "Lines 1058–1085"
)

# Issue 7: binary training_args.bin L83 -> deserialized key
text = text.replace(
    "training_args.bin L83",
    "training_args.bin (unpickled `TrainingArguments.loss_lvr_lambda`)"
)

# Issue 8: _lvr_deocding_by_steps line range
text = text.replace("855–1050", "855–1095")
text = text.replace("L855-L1050", "L855-L1095")

# Issue 13: stage2_3b.sh
text = text.replace("`stage2_3b.sh`", "`scripts/finetune_lvr_stage2_3b.sh`")

# Issue 9: Equation 6 in Section 2 table
eq6_row = """| **(6)** | `\\tilde{R}_i = \\frac{R(y_i)-\\mathrm{mean}(R(y_{1}),\\dots,R(y_{G}))}{\\mathrm{std}(R(y_{1}),\\dots,R(y_{G}))}, \\quad \\hat{A}_{i,t} = \\tilde{R}_i` | [`3_lvr.tex` L111–L116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116) | [`src/trainer/grpo_trainer.py` L882–L895](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L882-L895) | Standard Group Advantage Normalization over $G$ rollouts | Exact Match |
"""
if "| **(6)** |" not in text:
    text = text.replace(
        "| **(5)** |",
        eq6_row + "| **(5)** |"
    )

# Issue 10: Trace modules lvr_heads.py, monkey_patch_patch_emb.py, monkey_patch_dataloader.py
patch_modules_section = """
### 1.4 Auxiliary Patch & Architecture Modules

1. **`src/model/lvr_heads.py` (Lines 1–35)**:
   - Defines custom projection heads (`LVRHead`, `LVRHeadGLU`) for mapping hidden states $\mathbf{h}_t \in \mathbb{R}^{3584}$ to visual feature space.
   - In default `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states directly supervise against visual features without intermediate projection.
2. **`src/train/monkey_patch_patch_emb.py` (Lines 1–22)**:
   - Monkey patches `Qwen2_5_VisionTransformerPretrainedModel.patch_embed` to force `fp32` computation on vision patch embeddings, avoiding numerical underflow during RoPE and 3D grid spatial scaling.
3. **`src/train/monkey_patch_dataloader.py` (Lines 1–59)**:
   - Replaces standard HuggingFace DataLoader with an InternVL-compatible packed sequence data loader preserving batch-instance grouping across distributed ranks.
"""
if "### 1.4 Auxiliary Patch & Architecture Modules" not in text:
    text = text.replace(
        "## 2. Mathematical Equation Cross-Examination Matrix",
        patch_modules_section + "\n## 2. Mathematical Equation Cross-Examination Matrix"
    )

# Issue 6: Verbatim LVR span masking code in grpo_trainer.py
old_mask_block = """```python
# Pseudo-code / simplified representation
for b in range(B):
    for t in range(T):
        if completion_ids[b, t] == lvr_start_id:
            mask = 0
```"""
new_mask_block = """```python
# Verbatim from src/trainer/grpo_trainer.py L740-L757
for i in range(prompt_completion_ids.size(0)):
    is_lvr = False
    for j in range(prompt_completion_ids.size(1)):
        if prompt_completion_ids[i, j] == lvr_start_token_id:
            is_lvr = True
            final_mask[i, j] = 0
        elif prompt_completion_ids[i, j] == lvr_end_token_id:
            is_lvr = False
            final_mask[i, j] = 0
        elif is_lvr:
            final_mask[i, j] = 0
```"""
text = text.replace(old_mask_block, new_mask_block)
# Also replace any loose variant
import re
text = re.sub(
    r'```python\s*# Masking completion tokens between.*?```',
    new_mask_block,
    text,
    flags=re.DOTALL
)

with open(trace_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Patched CODE_VS_PAPER_TRACE.md successfully.")
