with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix M-RoPE citation
    line = line.replace("config.json` lines 24–28", "config.json` lines 28–33")
    line = line.replace("config.json#L24-L28", "config.json#L28-L33")

    # Fix stage2_3b.sh
    line = line.replace("`stage2_3b.sh`", "`scripts/finetune_lvr_stage2_3b.sh`")

    # Fix projector freezing citation in table row 2
    if line.startswith("| **Multimodal Projector**"):
        line = "| **Multimodal Projector** | $proj(\\mathbf{V}) = \\mathbf{V}_T$ | [`3_lvr.tex` L18](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L18) | [`src/train/train_lvr.py` L43](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) & [`train_grpo.py` L41](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_grpo.py#L41) | `model.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`) | Match (Frozen) |\n"

    # Fix binary training_args.bin L83
    line = line.replace("`training_args.bin` L83", "unpickled `TrainingArguments.loss_lvr_lambda` in `training_args.bin`")

    # Fix GRPO surrogate loss citation
    line = line.replace("`src/trainer/grpo_trainer.py` L1025–L1035", "`src/trainer/grpo_trainer.py` L1058–L1085")

    # Add Eq 6 after Eq 5
    if line.startswith("| **(5)** |"):
        eq6 = "| **(6)** | `\\tilde{R}_i = \\frac{R(y_i)-\\mathrm{mean}(R(y_{1}),\\dots,R(y_{G}))}{\\mathrm{std}(R(y_{1}),\\dots,R(y_{G}))}, \\quad \\hat{A}_{i,t} = \\tilde{R}_i` | [`3_lvr.tex` L111–L116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116) | [`src/trainer/grpo_trainer.py` L882–L895](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L882-L895) | Standard Group Advantage Normalization over $G$ rollouts | Exact Match |\n"
        new_lines.append(line)
        new_lines.append(eq6)
        continue

    # Add auxiliary patch section before Section 2
    if line.startswith("## 2. Mathematical Equation Cross-Examination Matrix"):
        patch_sec = """### 1.4 Auxiliary Patch & Architecture Modules

1. **`src/model/lvr_heads.py` (Lines 1–35)**:
   - Defines custom projection heads (`LVRHead`, `LVRHeadGLU`) for mapping hidden states $\\mathbf{h}_t \\in \\mathbb{R}^{3584}$ to visual feature space.
   - In default `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states directly supervise against visual features without intermediate projection.
2. **`src/train/monkey_patch_patch_emb.py` (Lines 1–22)**:
   - Monkey patches `Qwen2_5_VisionTransformerPretrainedModel.patch_embed` to force `fp32` computation on vision patch embeddings, avoiding numerical underflow during RoPE and 3D grid spatial scaling.
3. **`src/train/monkey_patch_dataloader.py` (Lines 1–59)**:
   - Replaces standard HuggingFace DataLoader with an InternVL-compatible packed sequence data loader preserving batch-instance grouping across distributed ranks.

"""
        new_lines.append(patch_sec)
        new_lines.append(line)
        continue

    new_lines.append(line)

with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Direct line-based patch applied to CODE_VS_PAPER_TRACE.md")
