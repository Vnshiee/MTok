trace_path = "/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md"
with open(trace_path, "r", encoding="utf-8") as f:
    text = f.read()

aux_modules = """### 1.4 Auxiliary Patch & Architecture Modules

1. **`src/model/lvr_heads.py` (Lines 1–35)**:
   - Defines custom projection heads (`LVRHead`, `LVRHeadGLU`) for mapping hidden states $\\mathbf{h}_t \\in \\mathbb{R}^{3584}$ to visual feature space.
   - In default `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states directly supervise against visual features without intermediate projection.
2. **`src/train/monkey_patch_patch_emb.py` (Lines 1–22)**:
   - Monkey patches `Qwen2_5_VisionTransformerPretrainedModel.patch_embed` to force `fp32` computation on vision patch embeddings, avoiding numerical underflow during RoPE and 3D grid spatial scaling.
3. **`src/train/monkey_patch_dataloader.py` (Lines 1–59)**:
   - Replaces standard HuggingFace DataLoader with an InternVL-compatible packed sequence data loader preserving batch-instance grouping across distributed ranks.

"""

text = text.replace(
    "## 2. Mathematical Formulation & Equation Mapping Matrix",
    aux_modules + "## 2. Mathematical Formulation & Equation Mapping Matrix"
)

with open(trace_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Inserted Section 1.4 cleanly.")
