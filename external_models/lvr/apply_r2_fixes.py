import re

# 1. Update Gaps_LVR.md
with open("/home/devraj/MTok/external_models/lvr/Gaps_LVR.md", "r", encoding="utf-8") as f:
    g = f.read()

# Fix \textbf{67.0}
g = g.replace("extbf{67.0}", "\\textbf{67.0}")
g = g.replace("    extbf{67.0}", "\\textbf{67.0}")

# Fix get_task_instruction in Gaps_LVR.md
exact_task_inst = '''```python
def get_task_instruction(bench_name):
    if bench_name == "vstar":
        return "\\nAnswer with the option\'s letter from the given choices directly."
    elif bench_name == "mmvp":
        return "\\nAnswer with the option\'s letter from the given choices directly."
    elif bench_name == "blink":
        return "\\nAnswer with the option\'s letter from the given choices directly."
    else:
        raise ValueError(f"Unknown benchmark: {bench_name}")
```'''
g = re.sub(r'```python\s*def get_task_instruction\(bench_name\):.*?```', exact_task_inst, g, flags=re.DOTALL)

# Fix script paths
g = g.replace("`finetune_lvr_stage2_3b.sh`", "[`scripts/finetune_lvr_stage2_3b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh)")
g = g.replace("`finetune_lvr_stage2_7b.sh`", "[`scripts/finetune_lvr_stage2_7b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh)")

# Fix credentials block
cred_block = '''Bash environment variables in [`scripts/vanilla_sft_7b.sh` lines 8–12](file:///home/devraj/MTok/external_models/lvr/repo/scripts/vanilla_sft_7b.sh#L8-L12):
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
```'''
g = re.sub(r'```bash\s*export access_key_id = .*?```', cred_block, g, flags=re.DOTALL)

with open("/home/devraj/MTok/external_models/lvr/Gaps_LVR.md", "w", encoding="utf-8") as f:
    f.write(g)

# 2. Update CODE_VS_PAPER_TRACE.md
with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "r", encoding="utf-8") as f:
    t = f.read()

# Fix Table 1.1 line 22 (Multimodal Projector) columns
old_row = '| **Multimodal Projector** | $proj(\\mathbf{V}) = \\mathbf{V}_T$ | [`3_lvr.tex` L18](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L18) | [`src/train/train_lvr.py` L43](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) & [`train_grpo.py` L41](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_grpo.py#L41) | `model.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`) | Match (Frozen) |'
new_row = '| **Multimodal Projector** | $proj(\\mathbf{V}) = \\mathbf{V}_T$ | [`3_lvr.tex` L18](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L18) | [`src/train/train_lvr.py` L43](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) | Lines 41, 43 | `model.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`) | Match (Frozen) |'
t = t.replace(old_row, new_row)

# Fix Table 1.1 line 30 GRPO line range
t = t.replace("Lines 750–817, 1025–1035", "Lines 740–757, 1058–1085")

# Fix Section 1.4 lvr_heads lines and monkey_patch_patch_emb
old_s14 = '''### 1.4 Auxiliary Patch & Architecture Modules

1. **`src/model/lvr_heads.py` (Lines 1–35)**:
   - Defines custom projection heads (`LVRHead`, `LVRHeadGLU`) for mapping hidden states $\\mathbf{h}_t \\in \\mathbb{R}^{3584}$ to visual feature space.
   - In default `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states directly supervise against visual features without intermediate projection.
2. **`src/train/monkey_patch_patch_emb.py` (Lines 1–22)**:
   - Monkey patches `Qwen2_5_VisionTransformerPretrainedModel.patch_embed` to force `fp32` computation on vision patch embeddings, avoiding numerical underflow during RoPE and 3D grid spatial scaling.
3. **`src/train/monkey_patch_dataloader.py` (Lines 1–59)**:
   - Replaces standard HuggingFace DataLoader with an InternVL-compatible packed sequence data loader preserving batch-instance grouping across distributed ranks.'''

new_s14 = '''### 1.4 Auxiliary Patch & Architecture Modules

1. [`src/model/lvr_heads.py` (Lines 1–41)](file:///home/devraj/MTok/external_models/lvr/repo/src/model/lvr_heads.py#L1-L41):
   - Defines custom projection heads (`LVRHead` lines 6–23, `LVRHeadGLU` lines 25–41) for projecting language hidden states $\\mathbf{h}_t \\in \\mathbb{R}^{3584}$ into the visual feature space.
   - In the released checkpoint `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states are supervised directly against raw patch embeddings without an auxiliary head.
2. [`src/train/monkey_patch_patch_emb.py` (Lines 1–22)](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_patch_emb.py#L1-L22):
   - Overwrites `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed.forward` with `Qwen2_5_VisionPatchEmbedFp32Forward` to force `fp32` computation on raw pixel patches, preventing numerical underflow during initial projection.
3. [`src/train/monkey_patch_dataloader.py` (Lines 1–59)](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_dataloader.py#L1-L59):
   - Replaces the standard HuggingFace DataLoader with an InternVL-compatible packed sequence loader supporting instance-grouped batching.'''

t = t.replace(old_s14, new_s14)

# Fix GRPO span masking snippet in Section 5.2
old_grpo_snippet = '''```python
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
```'''
verbatim_grpo = '''```python
# Verbatim from repo/src/trainer/grpo_trainer.py lines 743-753
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
```'''
t = t.replace(old_grpo_snippet, verbatim_grpo)

with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "w", encoding="utf-8") as f:
    f.write(t)

print("Applied R2 fixes cleanly.")
