trace_path = "/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md"
with open(trace_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add Equation 6 in Section 2
eq6_block = """
```
+-------------------------------------------------------------------------------------------------------------+
| Formal Equation (6) — paper_src/sections/3_lvr.tex L111-L116                                                |
|   \\begin{align}                                                                                             |
|   \\tilde{R}_i = \\frac{R(y_i)-\\mathrm{mean}(R(y_{1}),\\dots,R(y_{G}))}{\\mathrm{std}(R(y_{1}),\\dots,R(y_{G}))}, \\qquad \\hat{A}_{i,t} = \\tilde{R}_i \\label{eq:adv_text_only}                                                      |
|   \\end{align}                                                                                               |
|                                                                                                             |
| Code Implementation (repo/src/trainer/grpo_trainer.py L882-L895):                                           |
|   mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)                |
|   std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)                  |
|   advantages = rewards - mean_grouped_rewards                                                               |
|   if self.scale_rewards:                                                                                    |
|       advantages = advantages / (std_grouped_rewards + 1e-4)                                                |
|                                                                                                             |
| Mathematical Analysis & Verdict: EXACT MATCH.                                                               |
|   Standard group advantage normalization across G completions. Zero type-specific normalization exists.      |
+-------------------------------------------------------------------------------------------------------------+
```
"""

if "Formal Equation (6)" not in text:
    target_needle = "|   prefix rather than an actively guided policy trajectory during optimization.                              |\n+-------------------------------------------------------------------------------------------------------------+\n```"
    text = text.replace(target_needle, target_needle + "\n" + eq6_block)

# 2. Add Section 1.4 Auxiliary Patch Modules
aux_modules = """
### 1.4 Auxiliary Patch & Architecture Modules

1. **`src/model/lvr_heads.py` (Lines 1–35)**:
   - Defines custom projection heads (`LVRHead`, `LVRHeadGLU`) for mapping hidden states $\\mathbf{h}_t \\in \\mathbb{R}^{3584}$ to visual feature space.
   - In default `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states directly supervise against visual features without intermediate projection.
2. **`src/train/monkey_patch_patch_emb.py` (Lines 1–22)**:
   - Monkey patches `Qwen2_5_VisionTransformerPretrainedModel.patch_embed` to force `fp32` computation on vision patch embeddings, avoiding numerical underflow during RoPE and 3D grid spatial scaling.
3. **`src/train/monkey_patch_dataloader.py` (Lines 1–59)**:
   - Replaces standard HuggingFace DataLoader with an InternVL-compatible packed sequence data loader preserving batch-instance grouping across distributed ranks.
"""

if "### 1.4 Auxiliary Patch & Architecture Modules" not in text:
    text = text.replace(
        "## 2. Mathematical Equation Cross-Examination Matrix",
        aux_modules + "\n## 2. Mathematical Equation Cross-Examination Matrix"
    )

# 3. Update GRPO loss citation in text
text = text.replace("repo/src/trainer/grpo_trainer.py L750-L758, L1025-L1035", "repo/src/trainer/grpo_trainer.py L740-L757, L1058-L1085")

with open(trace_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Applied full patches to CODE_VS_PAPER_TRACE.md")
