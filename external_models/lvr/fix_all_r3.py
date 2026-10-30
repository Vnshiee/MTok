import re

# 1. Update Gaps_LVR.md
with open("/home/devraj/MTok/external_models/lvr/Gaps_LVR.md", "r", encoding="utf-8") as f:
    g = f.read()

# Fix 2: \textbf{67.0}
g = re.sub(r'\s*extbf\{67\.0\}', r'\\textbf{67.0}', g)

# Fix 8: get_task_instruction in Gaps_LVR.md without double escaping
clean_inst = """```python
def get_task_instruction(bench_name):
    if bench_name == "vstar":
        return "\\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "mmvp":
        return "\\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "blink":
        return "\\nAnswer with the option's letter from the given choices directly."
    else:
        raise ValueError(f"Unknown benchmark: {bench_name}")
```"""
g = re.sub(r'```python\s*def get_task_instruction\(bench_name\):.*?```', clean_inst, g, flags=re.DOTALL)

# Fix 9: Differentiate Cloud Credentials
cloud_block = """Bash environment variable exports in [`scripts/vanilla_sft_7b.sh` lines 8–12](file:///home/devraj/MTok/external_models/lvr/repo/scripts/vanilla_sft_7b.sh#L8-L12):
```bash
export ACCESS_KEY_ID="cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
export SECRET_ACCESS_KEY="bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
export ENDPOINT_URL="https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
export BUCKET_NAME="bangzhengli"
export REGION_NAME="us-ashburn-1"
```

Python variable assignments in [`evaluation/evaluation.py` lines 17–21](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L17-L21):
```python
access_key_id = "cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
secret_access_key = "bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
endpoint_url = "https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
bucket_name = "bangzhengli"
region_name = "us-ashburn-1"
```"""
g = re.sub(r'Bash environment variables in.*?```python\s*access_key_id = .*?```', cloud_block, g, flags=re.DOTALL)

# Fix 10: Fully qualify script paths
g = g.replace("`finetune_lvr_stage2_3b.sh`", "[`scripts/finetune_lvr_stage2_3b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh)")
g = g.replace("`finetune_lvr_stage2_7b.sh`", "[`scripts/finetune_lvr_stage2_7b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh)")

# Fix 11: Multi-range anchors
g = g.replace(
    "[`3_lvr.tex` lines 20–24, 52–58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24)",
    "[`3_lvr.tex` lines 20–24](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24) and [lines 52–58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L52-L58)"
)
g = g.replace(
    "[`src/model/qwen_lvr_model.py` lines 953–954, 1045–1049](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954)",
    "[`src/model/qwen_lvr_model.py` lines 953–954](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954) and [lines 1045–1049](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L1045-L1049)"
)
g = g.replace("#L1013-L1055", "#L1013-L1014")

with open("/home/devraj/MTok/external_models/lvr/Gaps_LVR.md", "w", encoding="utf-8") as f:
    f.write(g)

# 2. Update CODE_VS_PAPER_TRACE.md
with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "r", encoding="utf-8") as f:
    t = f.read()

# Fix 3: Table 1.1 line 22 7 columns
t = re.sub(
    r'\|\s*\*\*Multimodal Projector\*\*.*?\n',
    '| **Multimodal Projector** | $proj(\\\\mathbf{V}) = \\\\mathbf{V}_T$ | [`3_lvr.tex` L18](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L18) | [`src/train/train_lvr.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) | Lines 41, 43 | `model.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`) | Match (Frozen) |\n',
    t
)

# Fix 4: Table 1.1 line 30 GRPO line range
t = re.sub(
    r'\|\s*\*\*Latent GRPO Policy Loss\*\*.*?\n',
    '| **Latent GRPO Policy Loss** | $\\\\mathcal{J}_{\\\\mathrm{GRPO}}(\\\\theta)$ | [`3_lvr.tex` L85](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L85) | [`src/trainer/grpo_trainer.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L740-L757) | Lines 740–757, 1058–1085 | Masked clipped surrogate policy gradient objective | Architectural Discrepancy |\n',
    t
)

# Fix 7: Section 1.4 Links and Accurate Specs
sec14_clean = """### 1.4 Auxiliary Patch & Architecture Modules

1. [`src/model/lvr_heads.py` (Lines 1–41)](file:///home/devraj/MTok/external_models/lvr/repo/src/model/lvr_heads.py#L1-L41):
   - Defines custom projection heads (`LVRHead` lines 6–23, `LVRHeadGLU` lines 25–41) for projecting language hidden states $\\mathbf{h}_t \\in \\mathbb{R}^{3584}$ into the visual feature space.
   - In the released checkpoint `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states are supervised directly against raw patch embeddings without an auxiliary head.
2. [`src/train/monkey_patch_patch_emb.py` (Lines 1–22)](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_patch_emb.py#L1-L22):
   - Overwrites `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed.forward` with `Qwen2_5_VisionPatchEmbedFp32Forward` to force `fp32` computation on raw pixel patches, preventing numerical underflow during initial projection.
3. [`src/train/monkey_patch_dataloader.py` (Lines 1–59)](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_dataloader.py#L1-L59):
   - Replaces the standard HuggingFace DataLoader with an InternVL-compatible packed sequence loader supporting instance-grouped batching."""
t = re.sub(r'### 1\.4 Auxiliary Patch & Architecture Modules.*?### 2\.', sec14_clean + '\n\n## 2.', t, flags=re.DOTALL)

# Fix 5: Replace fabricated one-liner in Section 2 Eq 4 box with verbatim clipped surrogate
verbatim_clipped_code = """Code Implementation (repo/src/trainer/grpo_trainer.py L740-L757, L1058-L1085):
  # Lines 1058-1066: Clipped surrogate objective
  surr1 = coef_1 * advantages.unsqueeze(1)
  surr2 = torch.clamp(coef_1, 1 - self.cliprange, 1 + self.cliprange) * advantages.unsqueeze(1)
  per_token_loss = -torch.min(surr1, surr2)
  per_token_loss = per_token_loss + self.beta * per_token_kl
  loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()"""
t = re.sub(r'Code Implementation \(repo/src/trainer/grpo_trainer\.py L750-L758.*?\n\s*Mathematical Analysis', verbatim_clipped_code + '\n\nMathematical Analysis', t, flags=re.DOTALL)

# Fix 6: Replace pseudo-code in Section 5.2 lines 478-489 with verbatim code
verbatim_mask_code = """```python
# Verbatim implementation from repo/src/trainer/grpo_trainer.py lines 743-753
final_mask = torch.ones_like(prompt_completion_ids, dtype=torch.float32)
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
t = re.sub(r'```python\s*for b in range\(B\):.*?```', verbatim_mask_code, t, flags=re.DOTALL)

# Fix 8: Synchronize decoding function lines in Section 6.4 line 670
t = t.replace("lines 309–347, 855–1070", "lines 855–1095")
t = t.replace("lines 953–1050", "lines 953–1095")

# Fix 1 & 8: Verbatim get_task_instruction in Trace lines 708-710
t = re.sub(r'```python\s*def get_task_instruction\(bench_name\):.*?```', clean_inst, t, flags=re.DOTALL)

with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "w", encoding="utf-8") as f:
    f.write(t)

print("Applied Round 3 fixes successfully.")
