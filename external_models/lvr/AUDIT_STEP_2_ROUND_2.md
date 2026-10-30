# Adversarial Hostile Line-by-Line Audit Report (Round 2): LVR Step 2 Deliverables

**Target Documents**:
1. [`/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md`](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md) (754 lines, 61,055 bytes)
2. [`/home/devraj/MTok/external_models/lvr/Gaps_LVR.md`](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md) (401 lines, 36,781 bytes)

**Audited Assets**:
- [`/home/devraj/MTok/external_models/lvr/paper_src/`](file:///home/devraj/MTok/external_models/lvr/paper_src/) (13 LaTeX documents, 25 total assets)
- [`/home/devraj/MTok/external_models/lvr/repo/`](file:///home/devraj/MTok/external_models/lvr/repo/) (50 codebase files, scripts, and configs)
- [`/home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/) (18 checkpoint metadata artifacts)
- Round 1 Audit Baseline: [`/home/devraj/MTok/external_models/lvr/AUDIT_STEP_2_ROUND_1.md`](file:///home/devraj/MTok/external_models/lvr/AUDIT_STEP_2_ROUND_1.md)

**Auditor**: Hostile Line-by-Line Code & Paper Auditor Agent  
**Audit Date**: October 30, 2026  

---

## Executive Summary & Verdict

```
VERDICT: FAIL (11 issues found)
```

In Round 2 of the adversarial re-audit, `forensic-builder` attempted remediation of the 14 defects identified in Round 1. While progress was made on 3 items (**Issue 3: M-RoPE citation**, **Issue 7: binary checkpoint line hallucination removal**, and **Issue 9: formal Equation 6 addition**), an exhaustive line-by-line inspection of raw disk files reveals that **11 issues remain broken, incomplete, or introduced new formatting regressions**:

1. **Issue 1 (Still Broken)**: `CODE_VS_PAPER_TRACE.md` lines 622–624 still contains unescaped, split-line syntax for `replace_lvr_tokens`. Furthermore, in `Gaps_LVR.md` lines 339–343, string literals were double-escaped to `\\n` and `\'` inside markdown python code blocks.
2. **Issue 2 (Still Broken)**: The remediation script in `Gaps_LVR.md` executed a blanket replacement of `\t` with four spaces without restoring `\textbf{67.0}`. Consequently, line 185 literally reads `\modelname_{RL} $(4 | 8 | 16)$ & $65.5 |     extbf{67.0} | 66.5$ & ... \\`.
3. **Issue 4 (Table Formatting Regression)**: The replacement applied to Table 1.1 line 22 omitted a column pipe separator, reducing the row from 7 columns to 6 columns. This misaligned the entire table rendering and shifted implementation notes into the "Line Range" column.
4. **Issue 5 (Still Broken)**: Table 1.1 line 30 still cites the old range `Lines 750–817, 1025–1035`. In Section 2 Equation 4 box (line 168), the header was updated, but the code snippet still quotes the fabricated one-line loss formula `per_token_loss = - torch.exp(per_token_logps - old_per_token_logps) * advantages.unsqueeze(1)` instead of the verbatim clipped surrogate objective from lines 1059–1066.
5. **Issue 6 (Still Broken)**: Section 5.2 lines 478–489 still quotes pseudo-code (`for b in range(B):`, `for t in range(T):`, `completion_ids[b, t]`, `if ... if ...`) instead of the verbatim code from `src/trainer/grpo_trainer.py:743-753`.
6. **Issue 8 (Incomplete)**: While Table 1.1 line 34 was updated to `855–1095`, Section 6.4 line 670 still cites `lines 309–347, 855–1070` and line 671 still cites `lines 953–1050`.
7. **Issue 10 (Incomplete & Inaccurate)**: Section 1.4 lacks clickable file links, truncates `src/model/lvr_heads.py` to `Lines 1–35` (omitting lines 36–41 of `LVRHeadGLU`), and misidentifies the target class in `monkey_patch_patch_emb.py` as `Qwen2_5_VisionTransformerPretrainedModel.patch_embed` instead of `Qwen2_5_VisionPatchEmbed.forward`.
8. **Issue 11 (Incomplete)**: `Gaps_LVR.md` lines 336–344 omitted the `else: raise ValueError(...)` branch and contains double-escaped `\\n` and `\'`. `CODE_VS_PAPER_TRACE.md` lines 708–710 still collapses the function to a single line without the benchmark conditional.
9. **Issue 12 (Still Broken)**: `Gaps_LVR.md` lines 384–390 still presents bash `export` syntax and claims it exists in `evaluation/evaluation.py:17-21`. Python variable assignments were not differentiated.
10. **Issue 13 (Incomplete)**: In `Gaps_LVR.md`, lines 224 and 371 still cite `finetune_lvr_stage2_3b.sh` without the `scripts/` prefix.
11. **Issue 14 (Still Broken)**: In `Gaps_LVR.md`, line 16 omits `L52-L58`, line 273 omits `L1045-L1049`, and line 284 spans the entire 43-line block `L1013-L1055` instead of targeting lines 1013 and 1055.

---

## 14-Point Remediation Verification Matrix

| # | Round 1 Issue Description | Verification Status | Exact File & Line Location | Concrete Defect / Evidence |
| :---: | :--- | :---: | :--- | :--- |
| **1** | Python snippet newlines fixed | **FAIL** | [`CODE_VS_PAPER_TRACE.md` L622–L624](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L622-L624)<br>[`Gaps_LVR.md` L339–L343](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L339-L343) | `pattern = r'\n...` is still broken across lines 622–624 in `trace`. In `gaps`, string is double-escaped as `\\n` and `\'`. |
| **2** | LaTeX tab escapes (`0x09`) fixed | **FAIL** | [`Gaps_LVR.md` L185](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L185) | Literal `\t` was replaced with 4 spaces instead of restoring `\textbf{67.0}`, producing `    extbf{67.0}`. |
| **3** | M-RoPE citation updated to lines 28–33 | **PASS** | [`CODE_VS_PAPER_TRACE.md` L367](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L367) | Verified exact match to [`config.json` lines 28–33](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L28-L33). |
| **4** | Projector freezing cited in `train_lvr.py:43` & `train_grpo.py:41` | **FAIL** | [`CODE_VS_PAPER_TRACE.md` Table 1.1 L22](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L22) | Dropped table column separator, reducing row from 7 to 6 columns; broken table layout. |
| **5** | GRPO surrogate loss citation updated to lines 1058–1085 & verbatim quote | **FAIL** | [`CODE_VS_PAPER_TRACE.md` L30, L162–L168](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L162-L168) | Table 1.1 L30 still cites lines 1025–1035. Eq 4 box line 168 still quotes fabricated one-liner loss instead of verbatim clipped code (L1059–1066). |
| **6** | LVR span masking code quoted verbatim from `grpo_trainer.py:743-753` | **FAIL** | [`CODE_VS_PAPER_TRACE.md` L478–L489](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L478-L489) | Pseudo-code (`for b in range(B):`, `completion_ids[b, t]`, `if ... if ...`) was never replaced in Section 5.2. |
| **7** | `training_args.bin` cited as deserialized attribute without line number | **PASS** | [`CODE_VS_PAPER_TRACE.md` L139](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L139) | `L83` removed; accurately cited as deserialized `TrainingArguments.loss_lvr_lambda`. |
| **8** | `_lvr_deocding_by_steps` cited as lines 855–1095 | **FAIL** | [`CODE_VS_PAPER_TRACE.md` L670–L671](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L670-L671) | Table 1.1 L34 fixed, but Section 6.4 line 670 still cites `855–1070` and line 671 cites `953–1050`. |
| **9** | Formal Equation (6) included with exact LaTeX lines & `grpo_trainer.py:882-895` | **PASS** | [`CODE_VS_PAPER_TRACE.md` L204–L221](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L204-L221) | Verified inclusion of Equation 6 box with LaTeX source lines 111–116 and `grpo_trainer.py` advantage normalization. |
| **10** | Architecture & patch modules added to trace | **FAIL** | [`CODE_VS_PAPER_TRACE.md` L40–L49](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L40-L49) | Section 1.4 lacks clickable `file://` links; `lvr_heads.py` truncated to lines 1–35 (misses L36–41); misnames target class in `monkey_patch_patch_emb.py`. |
| **11** | Verbatim `evaluation.py` instruction conditional updated | **FAIL** | [`Gaps_LVR.md` L336–L344](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L336-L344)<br>[`CODE_VS_PAPER_TRACE.md` L708–L710](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L708-L710) | `Gaps` omits `else: raise ValueError(...)` and double-escapes backslashes. `Trace` still collapses conditional to single line. |
| **12** | AWS environment keys formatted as Python variable assignments in `evaluation.py` | **FAIL** | [`Gaps_LVR.md` L384–L390](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L384-L390) | Bash `export` syntax still attributed to `evaluation/evaluation.py:17-21`. Python variables never added. |
| **13** | Filename updated to `scripts/finetune_lvr_stage2_3b.sh` | **FAIL** | [`Gaps_LVR.md` L224, L371](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L224) | Lines 224 and 371 still cite `finetune_lvr_stage2_3b.sh` without the `scripts/` directory prefix. |
| **14** | URL anchors in `Gaps_LVR.md` checked and corrected | **FAIL** | [`Gaps_LVR.md` L16, L273, L284](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L16) | L16 omits `L52-L58`; L273 omits `L1045-L1049`; L284 spans entire 43-line block `L1013-L1055` instead of targeting lines 1013 and 1055. |

---

## Detailed Line-by-Line Adversarial Findings

### Issue 1: Split Newlines in `CODE_VS_PAPER_TRACE.md` & Double Escapes in `Gaps_LVR.md`
- **Location in Deliverables**:
  - [`CODE_VS_PAPER_TRACE.md` lines 621–624](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L621-L624)
  - [`Gaps_LVR.md` lines 338–344](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L338-L344)
- **Hostile Inspection**:
  In `CODE_VS_PAPER_TRACE.md`:
  ```python
  def replace_lvr_tokens(input_string, lvr_token_idxs_list, latent_end_token, fixed_num_of_lvr_tokens):
      pattern = r'
  ?' + re.escape(LVR_PLACEHOLDER) + r'
  ?'
  ```
  The string is still broken across lines! The builder attempted to patch a regex in `reward_funcs.py` instead of the broken multiline string in `data_utils.py:35`:
  `pattern = r'\n?' + re.escape(LVR_PLACEHOLDER) + r'\n?'`
  In `Gaps_LVR.md`:
  ```python
  def get_task_instruction(bench_name):
    if bench_name == "vstar":
        return "\\nAnswer with the option\'s letter from the given choices directly."
    elif bench_name == "mmvp":
        return "\\nAnswer with the option\'s letter from the given choices directly."
    elif bench_name == "blink":
        return "\\nAnswer with the option\'s letter from the given choices directly."
  ```
  The builder used double backslashes `\\n` and escaped apostrophe `\'` inside the python code block, producing unfaithful pseudo-escapes. In [`evaluation/evaluation.py` lines 64–69](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L64-L69), it is:
  `return "\nAnswer with the option's letter from the given choices directly."`

---

### Issue 2: Botched LaTeX Tab Replacement Leaving `    extbf{67.0}`
- **Location in Deliverable**: [`Gaps_LVR.md` line 185](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L185)
- **Hostile Inspection**:
  In Round 1, line 185 contained a literal tab byte `0x09` resulting from `\t` escaping: `\modelname_{RL} $(4 | 8 | 16)$ & $65.5 | \text{bf}{67.0} ...`.
  In `patch_step2_round1.py`, the builder executed:
  ```python
  gaps = gaps.replace("\text{bf}{67.0}", "\\textbf{67.0}")
  gaps = gaps.replace("\t", "    ")
  ```
  Because the search target `\text{bf}{67.0}` did not match (the actual string was `\t` followed by `extbf{67.0}`), line 11 failed silently. Line 13 then replaced the tab `\t` with four spaces, leaving:
  ```latex
  \modelname_{RL} $(4 | 8 | 16)$ & $65.5 |     extbf{67.0} | 66.5$ & ... \
  ```
  The backslash was never restored! The LaTeX command remains completely corrupted as four spaces and `extbf{67.0}`. In [`paper_src/tables/sft_rl.tex` line 15](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex#L15), the source is `\textbf{67.0}`.

---

### Issue 4: Markdown Table Column Break in Table 1.1
- **Location in Deliverable**: [`CODE_VS_PAPER_TRACE.md` line 22](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L22)
- **Hostile Inspection**:
  The header of Table 1.1 specifies 7 columns:
  ```markdown
  | Architectural Component | Formal Notation (Paper) | Paper LaTeX Citation | Implementation Source File | Implementation Line Range | Exact Tensor Shape / Dtype / Logic | Discrepancy Status |
  ```
  However, line 22 was edited to:
  ```markdown
  | **Multimodal Projector** | $proj(\mathbf{V}) = \mathbf{V}_T$ | [`3_lvr.tex` L18](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L18) | [`src/train/train_lvr.py` L43](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) & [`train_grpo.py` L41](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_grpo.py#L41) | `model.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`) | Match (Frozen) |
  ```
  This row has **only 6 columns**!
  The pipe between `Implementation Source File` and `Implementation Line Range` was deleted, causing `model.visual.merger...` to appear under `Implementation Line Range` and shifting `Match (Frozen)` into `Exact Tensor Shape / Dtype / Logic`, leaving `Discrepancy Status` completely empty.

---

### Issue 5: Fabricated GRPO Loss Snippet in Section 2 & Old Citation in Table 1.1
- **Locations in Deliverable**:
  - [`CODE_VS_PAPER_TRACE.md` Table 1.1 line 30](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L30)
  - [`CODE_VS_PAPER_TRACE.md` Section 2 lines 162–168](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L162-L168)
- **Hostile Inspection**:
  1. In Table 1.1 line 30, the line citation for the Latent GRPO objective still reads `Lines 750–817, 1025–1035`, completely ignoring the surrogate loss calculation block at lines 1058–1085.
  2. In Section 2 Equation 4 box (line 162), the header was changed to `repo/src/trainer/grpo_trainer.py L740-L757, L1058-L1085`, but lines 167–168 **still quote the fabricated one-line pseudo-formula**:
     ```python
     per_token_logps = selective_log_softmax(logits, input_tok_ids)
     per_token_loss = - torch.exp(per_token_logps - old_per_token_logps) * advantages.unsqueeze(1)
     ```
     This completely omits the PPO/GRPO clipping logic, dual coefficient calculation, and `torch.min`.
     The verbatim implementation in [`src/trainer/grpo_trainer.py` lines 1059–1066](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1059-L1066) is:
     ```python
     adv = advantages.to(device).unsqueeze(1)

     coef_1 = torch.exp(per_token_logps - old_logps)
     coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

     per_token_loss1 = coef_1 * adv
     per_token_loss2 = coef_2 * adv
     per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
     ```

---

### Issue 6: Pseudo-Code Still Present for LVR Span Masking
- **Location in Deliverable**: [`CODE_VS_PAPER_TRACE.md` lines 478–489](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L478-L489)
- **Hostile Inspection**:
  In Section 5.2, the code block still reads:
  ```python
  for b in range(B):
      active = False
      for t in range(T):
          tok = completion_ids[b, t]
          if tok == self.model.config.lvr_start_id:
              active = True
          if tok == self.model.config.lvr_end_id:
              active = False
          if active:
              lvr_mask[b, t] = False  # zero out inside LVR span
  ```
  The builder's replacement script (`patch_trace_step2.py`) targeted a different block and failed to execute.
  The verbatim implementation from [`src/trainer/grpo_trainer.py` lines 743–753](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L743-L753) is:
  ```python
  for b in range(prompt_completion_ids.size(0)):
      active = False
      for t in range(prompt_completion_ids.size(1)):
          tok = prompt_completion_ids[b, t].item()
          if tok == self.model.config.lvr_start_id:
              active = True
          elif tok == self.model.config.lvr_end_id:
              active = False
          if active:
              lvr_mask[b, t] = False  # zero out inside LVR span
  ```

---

### Issue 8: Inconsistent Decoding Line Range Citations
- **Locations in Deliverable**:
  - [`CODE_VS_PAPER_TRACE.md` line 670](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L670)
  - [`CODE_VS_PAPER_TRACE.md` line 671](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L671)
- **Hostile Inspection**:
  While Table 1.1 line 34 was successfully updated to `Lines 855–1095`, Section 6.4 was not updated:
  - Line 670 cites `lines 309–347, 855–1070` (cutting off before streamer shutdown and return statement).
  - Line 671 cites `lines 953–1050`.
  In [`src/model/qwen_lvr_model.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L855-L1095), `def _lvr_deocding_by_steps` spans lines 855–1095.

---

### Issue 10: Section 1.4 Lacks Links, Truncates Line Numbers, and Misidentifies Class
- **Location in Deliverable**: [`CODE_VS_PAPER_TRACE.md` lines 40–49](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L40-L49)
- **Hostile Inspection**:
  ```markdown
  ### 1.4 Auxiliary Patch & Architecture Modules

  1. **`src/model/lvr_heads.py` (Lines 1–35)**:
     - Defines custom projection heads (`LVRHead`, `LVRHeadGLU`) for mapping hidden states $\mathbf{h}_t \in \mathbb{R}^{3584}$ to visual feature space.
     - In default `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states directly supervise against visual features without intermediate projection.
  2. **`src/train/monkey_patch_patch_emb.py` (Lines 1–22)**:
     - Monkey patches `Qwen2_5_VisionTransformerPretrainedModel.patch_embed` to force `fp32` computation on vision patch embeddings, avoiding numerical underflow during RoPE and 3D grid spatial scaling.
  3. **`src/train/monkey_patch_dataloader.py` (Lines 1–59)**:
     - Replaces standard HuggingFace DataLoader with an InternVL-compatible packed sequence data loader preserving batch-instance grouping across distributed ranks.
  ```
  1. **Missing Clickable Links**: None of the three modules include clickable markdown links (`[src/model/lvr_heads.py](file:///home/devraj/MTok/external_models/lvr/repo/src/model/lvr_heads.py)`).
  2. **Truncated Line Citation**: `src/model/lvr_heads.py` is cited as `(Lines 1–35)`. In reality, `LVRHeadGLU` extends through line 41, and the file is 62 lines long.
  3. **Hallucinated Class Target**: Point 2 claims it patches `Qwen2_5_VisionTransformerPretrainedModel.patch_embed`. In [`src/train/monkey_patch_patch_emb.py` lines 10–11](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_patch_emb.py#L10-L11), the code actually patches:
     `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbedFp32Forward`

---

### Issue 11: Incomplete `evaluation.py` Conditional & Double Escaping
- **Locations in Deliverables**:
  - [`Gaps_LVR.md` lines 336–344](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L336-L344)
  - [`CODE_VS_PAPER_TRACE.md` lines 708–710](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L708-L710)
- **Hostile Inspection**:
  1. In `Gaps_LVR.md`, the code block quotes:
     ```python
     def get_task_instruction(bench_name):
       if bench_name == "vstar":
           return "\\nAnswer with the option\'s letter from the given choices directly."
       elif bench_name == "mmvp":
           return "\\nAnswer with the option\'s letter from the given choices directly."
       elif bench_name == "blink":
           return "\\nAnswer with the option\'s letter from the given choices directly."
     ```
     This block omits lines 70–71 (`else: raise ValueError(f"Unknown benchmark: {bench_name}")`) and contains double-escaped `\\n` and `\'`.
  2. In `CODE_VS_PAPER_TRACE.md` lines 708–710, the function is still collapsed:
     ```python
     def get_task_instruction(bench_name):
         return "\nAnswer with the option's letter from the given choices directly."
     ```

---

### Issue 12: Bash `export` Syntax Falsely Maintained for Python File
- **Location in Deliverable**: [`Gaps_LVR.md` lines 384–390](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L384-L390)
- **Hostile Inspection**:
  The block reads:
  ```markdown
  In multiple source files in the public repository:
  - [`scripts/vanilla_sft_7b.sh` lines 8–12](file:///home/devraj/MTok/external_models/lvr/repo/scripts/vanilla_sft_7b.sh#L8-L12)
  - [`evaluation/evaluation.py` lines 17–21](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L17-L21)
  The author's active Oracle Cloud Infrastructure (OCI) credentials are hardcoded in plaintext:
  ```bash
  export ACCESS_KEY_ID="cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
  export SECRET_ACCESS_KEY="bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
  export ENDPOINT_URL="https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
  export BUCKET_NAME="bangzhengli"
  export REGION_NAME="us-ashburn-1"
  ```
  ```
  `evaluation/evaluation.py` lines 17–21 is a Python script and does NOT contain bash `export` statements or capitalized variable names. It contains lowercase Python variable definitions:
  ```python
  access_key_id = "cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
  secret_access_key = "bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
  endpoint_url = "https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
  bucket_name = "bangzhengli"
  region_name = "us-ashburn-1"
  ```
  The deliverable must present both syntax blocks separately.

---

### Issue 13: Incomplete Script Path Qualification in `Gaps_LVR.md`
- **Locations in Deliverable**:
  - [`Gaps_LVR.md` line 224](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L224): cites `[`finetune_lvr_stage2_3b.sh` line 51]` without `scripts/`.
  - [`Gaps_LVR.md` line 371](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L371): cites ``LR=5e-6` (`finetune_lvr_stage2_3b.sh` L19)` without `scripts/`.
  - [`Gaps_LVR.md` line 217](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L217): cites `and finetune_lvr_stage2_7b.sh line 19` without `scripts/` or link.
- **Hostile Inspection**:
  The script resides in `repo/scripts/`. Citing bare filenames creates ambiguity with top-level scripts.

---

### Issue 14: URL Anchors Truncated or Over-Broad in `Gaps_LVR.md`
- **Locations in Deliverable**:
  - [`Gaps_LVR.md` line 16](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L16): Text cites `3_lvr.tex lines 20–24, 52–58`, but URL anchor is [`#L20-L24`](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24), completely omitting lines 52–58.
  - [`Gaps_LVR.md` line 273](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L273): Text cites `src/model/qwen_lvr_model.py lines 953–954, 1045–1049`, but URL anchor is [`#L953-L954`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954), completely omitting lines 1045–1049.
  - [`Gaps_LVR.md` line 284](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L284): Text cites `lines 1013, 1055`, but URL anchor spans [`#L1013-L1055`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L1013-L1055) (all 43 lines).
- **Hostile Inspection**:
  In `patch_step2_round1.py`, the builder mistakenly attempted to replace `repo/src/trainer/grpo_trainer.py#L1013-L1055` instead of `repo/src/model/qwen_lvr_model.py#L1013-L1055`. As a result, the anchor was never updated.

---

## Actionable Remediation Checklist for `forensic-builder` (Round 3)

To achieve an unassailable `VERDICT: PASS (0 MISTAKES)` in Round 3, `forensic-builder` must execute the following concrete modifications:

1. [ ] **Fix Split Regex Newline in `CODE_VS_PAPER_TRACE.md`**:
   At line 621, replace:
   ```python
   def replace_lvr_tokens(input_string, lvr_token_idxs_list, latent_end_token, fixed_num_of_lvr_tokens):
       pattern = r'
   ?' + re.escape(LVR_PLACEHOLDER) + r'
   ?'
   ```
   with the single, valid line from `src/dataset/data_utils.py:35`:
   ```python
   def replace_lvr_tokens(input_string, lvr_token_idxs_list, latent_end_token, fixed_num_of_lvr_tokens):
       pattern = r'\n?' + re.escape(LVR_PLACEHOLDER) + r'\n?'
   ```

2. [ ] **Restore `\textbf{67.0}` in `Gaps_LVR.md`**:
   At line 185, replace:
   `\modelname_{RL} $(4 | 8 | 16)$ & $65.5 |     extbf{67.0} | 66.5$ & ... \`
   with:
   `\modelname_{RL} $(4 | 8 | 16)$ & $65.5 | \textbf{67.0} | 66.5$ & ... \`

3. [ ] **Repair Table 1.1 Column Alignment in `CODE_VS_PAPER_TRACE.md`**:
   At line 22, ensure exactly 7 columns are present:
   ```markdown
   | **Multimodal Projector** | $proj(\mathbf{V}) = \mathbf{V}_T$ | [`3_lvr.tex` L18](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L18) | [`src/train/train_lvr.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) & [`train_grpo.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_grpo.py#L41) | Lines 43 (SFT), 41 (GRPO) | `model.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`) | Match (Frozen) |
   ```

4. [ ] **Update Table 1.1 Citation & Verbatim Code for GRPO Loss in `CODE_VS_PAPER_TRACE.md`**:
   - In Table 1.1 line 30, update implementation line range to `Lines 740–758, 1058–1085`.
   - In Section 2 Equation 4 box (line 167–168), replace the fabricated one-liner with verbatim lines 1059–1066 from `src/trainer/grpo_trainer.py`:
     ```python
     adv = advantages.to(device).unsqueeze(1)
     coef_1 = torch.exp(per_token_logps - old_logps)
     coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
     per_token_loss1 = coef_1 * adv
     per_token_loss2 = coef_2 * adv
     per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
     ```

5. [ ] **Restore Verbatim Masking Code in Section 5.2 of `CODE_VS_PAPER_TRACE.md`**:
   At lines 478–489, replace pseudo-code with verbatim lines 743–753 from `src/trainer/grpo_trainer.py`:
   ```python
   for b in range(prompt_completion_ids.size(0)):
       active = False
       for t in range(prompt_completion_ids.size(1)):
           tok = prompt_completion_ids[b, t].item()
           if tok == self.model.config.lvr_start_id:
               active = True
           elif tok == self.model.config.lvr_end_id:
               active = False
           if active:
               lvr_mask[b, t] = False  # zero out inside LVR span
   ```

6. [ ] **Synchronize Decoding Function Line Numbers in `CODE_VS_PAPER_TRACE.md`**:
   In Section 6.4 line 670, update `lines 309–347, 855–1070` to `lines 855–1095`.

7. [ ] **Add Clickable Links & Accurate Specs in Section 1.4 of `CODE_VS_PAPER_TRACE.md`**:
   Format Section 1.4 with clickable links, lines 1–41 for `src/model/lvr_heads.py`, and correct monkey-patched class name:
   ```markdown
   1. [`src/model/lvr_heads.py` (Lines 1–41)](file:///home/devraj/MTok/external_models/lvr/repo/src/model/lvr_heads.py#L1-L41):
      - Defines custom projection heads (`LVRHead` lines 5–20, `LVRHeadGLU` lines 24–41) for mapping hidden states $\mathbf{h}_t \in \mathbb{R}^{3584}$ to visual feature space.
      - In default `vincentleebang/LVR-7B`, `lvr_head = False`, meaning latent hidden states directly supervise against visual features without intermediate projection.
   2. [`src/train/monkey_patch_patch_emb.py` (Lines 1–22)](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_patch_emb.py#L1-L22):
      - Monkey patches `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed.forward` to force `fp32` computation on vision patch embeddings via `torch.autocast`, avoiding numerical underflow during RoPE and 3D grid spatial scaling.
   3. [`src/train/monkey_patch_dataloader.py` (Lines 1–59)](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_dataloader.py#L1-L59):
      - Replaces standard HuggingFace DataLoader with an InternVL-compatible packed sequence data loader preserving batch-instance grouping across distributed ranks.
   ```

8. [ ] **Fix Verbatim `evaluation.py` Instructions**:
   - In `Gaps_LVR.md` lines 336–344, remove double escapes (`\\n` -> `\n`, `\'` -> `'`) and include full function with `else: raise ValueError(f"Unknown benchmark: {bench_name}")`.
   - In `CODE_VS_PAPER_TRACE.md` lines 708–710, replace the single return line with the full `if ... elif ... else` conditional block.

9. [ ] **Differentiate Cloud Credentials Syntax in `Gaps_LVR.md`**:
   Replace lines 384–390 with two separate blocks:
   Bash export in [`scripts/vanilla_sft_7b.sh` lines 8–12](file:///home/devraj/MTok/external_models/lvr/repo/scripts/vanilla_sft_7b.sh#L8-L12):
   ```bash
   export ACCESS_KEY_ID="cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
   export SECRET_ACCESS_KEY="bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
   export ENDPOINT_URL="https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
   export BUCKET_NAME="bangzhengli"
   export REGION_NAME="us-ashburn-1"
   ```
   Python assignments in [`evaluation/evaluation.py` lines 17–21](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L17-L21):
   ```python
   access_key_id = "cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
   secret_access_key = "bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
   endpoint_url = "https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
   bucket_name = "bangzhengli"
   region_name = "us-ashburn-1"
   ```

10. [ ] **Fully Qualify Script Paths in `Gaps_LVR.md`**:
    At lines 217, 224, and 371, prefix script references with `scripts/` (e.g. [`scripts/finetune_lvr_stage2_3b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh) and [`scripts/finetune_lvr_stage2_7b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh)).

11. [ ] **Fix Multi-Range Anchors in `Gaps_LVR.md`**:
    - At line 16, split link or cite lines 20–24: `[`3_lvr.tex` lines 20–24](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24) and [lines 52–58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L52-L58)`.
    - At line 273, split link: `[`src/model/qwen_lvr_model.py` lines 953–954](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954), [lines 1045–1049](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L1045-L1049)`.
    - At line 284, update link from `#L1013-L1055` to target lines 1013 and 1055.
