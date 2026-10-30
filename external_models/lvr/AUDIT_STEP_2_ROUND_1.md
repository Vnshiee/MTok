# Adversarial Hostile Line-by-Line Audit Report (Round 1): LVR Step 2 Deliverables

**Target Documents**:
1. [`/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md`](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md) (726 lines, 58,077 bytes)
2. [`/home/devraj/MTok/external_models/lvr/Gaps_LVR.md`](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md) (399 lines, 36,453 bytes)

**Audited Assets**:
- [`/home/devraj/MTok/external_models/lvr/paper_src/`](file:///home/devraj/MTok/external_models/lvr/paper_src/) (13 LaTeX documents, 25 total assets)
- [`/home/devraj/MTok/external_models/lvr/repo/`](file:///home/devraj/MTok/external_models/lvr/repo/) (50 codebase files, scripts, and configs)
- [`/home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/) (18 checkpoint metadata artifacts)

**Auditor**: Hostile Line-by-Line Code & Paper Auditor Agent  
**Audit Date**: October 30, 2026  

---

## Executive Summary & Verdict

```
VERDICT: FAIL (14 issues found)
```

The Step 2 deliverables produced by `forensic-builder` represent an exceptionally deep and incisive technical forensic analysis. In particular, the identification and mathematical verification of the 5 core conceptual and empirical gaps—specifically the **ROI feature copying paradox**, the **catastrophic collapse of adaptive termination**, the **deliberate suppression of 3B base model superiority in Table 2**, the **zero-gradient severing in Latent GRPO RL**, and the **brittle evaluation parsing artifacts**—are thoroughly grounded in the raw LaTeX source and repository files.

However, an adversarial, line-by-line inspection comparing both deliverables against raw disk files (`paper_src/`, `repo/`, `hf_metadata/`) revealed **14 concrete defects, inaccuracies, and omissions**:
1. **String Literal Escape Collapses**: Generation scripts (`make_code_trace.py` and `make_gaps_lvr.py`) utilized non-raw Python multiline strings, causing escape sequences (`\n`, `\t`) to evaluate into ASCII control characters. This resulted in broken, uncompilable Python code snippets and tab-corrupted LaTeX formulas.
2. **Fabricated and Pseudo-Code Quotes**: Multiple code quotations in `CODE_VS_PAPER_TRACE.md` and `Gaps_LVR.md` present simplified or refactored pseudo-code as verbatim implementation blocks (e.g., rewriting the LVR span loop, collapsing mathematical one-liners, and altering boolean conditionals).
3. **Line Number Inaccuracies & Truncations**: Line citations for M-RoPE in `config.json`, the multimodal projector in `lvr_trainer.py`, the decoding loop in `qwen_lvr_model.py`, and the policy surrogate loss in `grpo_trainer.py` are inaccurate, truncated, or cite docstrings rather than execution logic.
4. **Binary Asset Line-Number Hallucination**: `training_args.bin` is once again cited with a line number (`L83`), despite being a binary pickled artifact (and key 83 corresponds to `optim`, not `loss_lvr_lambda`).
5. **Omission of Equation 6**: Section 2 of `CODE_VS_PAPER_TRACE.md` completely omits Equation 6 (Reward Normalization and Advantage Assignment, `3_lvr.tex` L111–L116).
6. **Omission of Repository Patch Modules**: Crucial architecture and runtime patches ([`src/model/lvr_heads.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/lvr_heads.py), [`src/train/monkey_patch_patch_emb.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_patch_emb.py), and [`src/train/monkey_patch_dataloader.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_dataloader.py)) were entirely absent from the architectural trace.
7. **Phantom Script References**: Both deliverables cite a non-existent file `stage2_3b.sh` instead of the actual `scripts/finetune_lvr_stage2_3b.sh`.

---

## Itemized Checklist of Issues Found

### Issue 1: Unescaped Newline Literals Corrupting Python Code Snippets
- **Files & Locations**:
  - [`CODE_VS_PAPER_TRACE.md` lines 592–594](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L592-L594)
  - [`CODE_VS_PAPER_TRACE.md` lines 679–680](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L679-L680)
  - [`Gaps_LVR.md` lines 339–340](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L339-L340)
- **Deliverable Manifestation**:
  In `CODE_VS_PAPER_TRACE.md` lines 591–594:
  ```python
  def replace_lvr_tokens(input_string, lvr_token_idxs_list, latent_end_token, fixed_num_of_lvr_tokens):
      pattern = r'
  ?' + re.escape(LVR_PLACEHOLDER) + r'
  ?'
  ```
  In `CODE_VS_PAPER_TRACE.md` lines 678–680:
  ```python
  def get_task_instruction(bench_name):
      return "
  Answer with the option's letter from the given choices directly."
  ```
  In `Gaps_LVR.md` lines 337–340:
  ```python
  def get_task_instruction(bench_name):
      if bench_name in ["vstar", "mmvp", "blink"]:
          return "
  Answer with the option's letter from the given choices directly."
  ```
- **Ground Truth Source**:
  - [`src/dataset/data_utils.py` line 35](file:///home/devraj/MTok/external_models/lvr/repo/src/dataset/data_utils.py#L35): `pattern = r'\n?' + re.escape(LVR_PLACEHOLDER) + r'\n?'`
  - [`evaluation/evaluation.py` line 65](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L65): `return "\nAnswer with the option's letter from the given choices directly."`
- **Root Cause**: `make_code_trace.py` and `make_gaps_lvr.py` constructed markdown strings using standard triple-quoted strings (`"""..."""`) without raw string prefixing (`r"""..."""`) or double-escaping (`\\n`), causing Python's string tokenizer to emit literal ASCII byte `0x0A`.
- **Required Fix**: Ensure all string literals containing `\n` in code generator scripts are properly escaped (`\\n`) or use raw strings so the generated markdown preserves `\n` within single-line strings.

---

### Issue 2: Unescaped Tab Literals (`0x09`) Corrupting LaTeX Formatting in `Gaps_LVR.md`
- **Deliverable Location**: [`Gaps_LVR.md` lines 185, 216, 221](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L185)
- **Deliverable Manifestation**:
  - Line 185: `\modelname_{RL} $(4 | 8 | 16)$ & $65.5 | 	extbf{67.0} | 66.5$ & ... \` (contains literal tab `0x09` before `extbf`).
  - Line 216: `*"with a learning rate of $1 	imes 10^{-5}$"*` (contains literal tab `0x09` before `imes`).
  - Line 221: `This is a **$2	imes$ discrepancy** compared to the published paper text.` (contains literal tab `0x09` before `imes`).
- **Ground Truth Source**:
  - [`paper_src/tables/sft_rl.tex` line 15](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex#L15): `\textbf{67.0}`
  - [`paper_src/sections/4_experimetns.tex` line 27](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/4_experimetns.tex#L27): `$1\times 10^{-5}$`
- **Root Cause**: In `make_gaps_lvr.py`, string sequences `\textbf` and `\times` inside standard multiline strings evaluated `\t` as ASCII byte `0x09`.
- **Required Fix**: Double escape all LaTeX backslashes (`\\textbf`, `\\times`) in `make_gaps_lvr.py`.

---

### Issue 3: Inaccurate M-RoPE Line Citation for `config.json` in `CODE_VS_PAPER_TRACE.md`
- **Deliverable Location**: [`CODE_VS_PAPER_TRACE.md` line 337](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L337)
- **Deliverable Claim**:
  *"with section sizes `[16, 24, 24]` ([`config.json` lines 24–28](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L24-L28))."*
- **Actual File Reality ([`hf_metadata/LVR-7B/config.json`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json))**:
  Lines 24–28 are:
  ```json
  24:   "num_attention_heads": 28,
  25:   "num_hidden_layers": 28,
  26:   "num_key_value_heads": 4,
  27:   "rms_norm_eps": 1e-06,
  28:   "rope_scaling": {
  ```
  The M-RoPE section sizes `[16, 24, 24]` actually reside on **lines 28–33**:
  ```json
  28:   "rope_scaling": {
  29:     "mrope_section": [
  30:       16,
  31:       24,
  32:       24
  33:     ],
  ```
- **Required Fix**: Update citation and anchor to [`config.json` lines 28–33](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L28-L33).

---

### Issue 4: Inaccurate Projector Line Citation in Table 1.1 of `CODE_VS_PAPER_TRACE.md`
- **Deliverable Location**: [`CODE_VS_PAPER_TRACE.md` line 22](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L22)
- **Deliverable Claim**:
  `Multimodal Projector` $\rightarrow$ `src/trainer/lvr_trainer.py#L42-L44`, `Lines 42–44, 63–66`, citing `self.visual.merger; frozen in Stage 1 & Stage 2 (freeze_merger=True)`.
- **Actual File Reality ([`src/trainer/lvr_trainer.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/lvr_trainer.py))**:
  - Lines 42–44 are the docstring of `def create_optimizer(self):`:
    ```python
    42:         """
    43:         Setup the optimizer.
    44:         We provide a reasonable default that works well...
    ```
  - Lines 63–65 map custom learning rates: `lr_mapper["merger"] = self.args.merger_lr`.
  - The actual architectural module `self.visual.merger` and parameter freezing (`freeze_merger=True`) is executed in [`src/train/train_lvr.py` lines 43, 49–50](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) and [`src/train/train_grpo.py` line 41](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_grpo.py#L41):
    ```python
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)
    ```
- **Required Fix**: Replace the citation in Table 1.1 with [`src/train/train_lvr.py` lines 43, 49–50](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) and [`src/trainer/lvr_trainer.py` lines 63–65](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/lvr_trainer.py#L63-L65).

---

### Issue 5: Inaccurate Line Citation and Fabricated Snippet for GRPO Surrogate Loss
- **Deliverable Location**: [`CODE_VS_PAPER_TRACE.md` lines 152–158](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L152-L158)
- **Deliverable Claim**:
  Cites `repo/src/trainer/grpo_trainer.py L750-L758, L1025-L1035` for Equation 4 and quotes:
  ```python
  per_token_logps = selective_log_softmax(logits, input_tok_ids)
  per_token_loss = - torch.exp(per_token_logps - old_per_token_logps) * advantages.unsqueeze(1)
  ```
- **Actual File Reality ([`src/trainer/grpo_trainer.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py))**:
  - Lines 1025–1035 only align completion logits and compute log probabilities:
    ```python
    1025:             logits = outputs.logits  # (B, T, V)
    1026:             logits = logits[:, :-1, :].contiguous()
    1027:             input_tok_ids = prompt_completion_ids[:, 1:].contiguous()
    ...
    1034:             per_token_logps = selective_log_softmax(logits, input_tok_ids)
    ```
  - The PPO/GRPO clipped surrogate objective `per_token_loss` is NOT computed in lines 1025–1035. It is computed in **lines 1058–1085**:
    ```python
    1059:         adv = advantages.to(device).unsqueeze(1)
    1061:         coef_1 = torch.exp(per_token_logps - old_logps)
    1062:         coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    1064:         per_token_loss1 = coef_1 * adv
    1065:         per_token_loss2 = coef_2 * adv
    1066:         per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    ```
  - The formula `per_token_loss = - torch.exp(per_token_logps - old_per_token_logps) * advantages.unsqueeze(1)` is a simplified fabrication that ignores clipping and dual coefficient evaluation.
- **Required Fix**: Cite [`src/trainer/grpo_trainer.py` lines 1025–1035, 1058–1085](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1058-L1085) and quote the verbatim clipped surrogate code from lines 1059–1066.

---

### Issue 6: Pseudo-Code Modification Presented as Verbatim Quote for LVR Span Masking
- **Deliverable Location**: [`CODE_VS_PAPER_TRACE.md` lines 448–463](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L448-L463)
- **Deliverable Quote**:
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
- **Actual File Reality ([`src/trainer/grpo_trainer.py` lines 743–753](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L743-L753))**:
  ```python
  743:         for b in range(prompt_completion_ids.size(0)):
  744:             active = False
  745:             for t in range(prompt_completion_ids.size(1)):
  746:                 tok = prompt_completion_ids[b, t].item()
  747:                 if tok == self.model.config.lvr_start_id:
  748:                     active = True
  749:                 elif tok == self.model.config.lvr_end_id:
  750:                     active = False
  751:                 if active:
  752:                     lvr_mask[b, t] = False  # zero out inside LVR span
  ```
  The builder replaced `prompt_completion_ids.size(0)` with `B`, `prompt_completion_ids.size(1)` with `T`, `prompt_completion_ids[b, t].item()` with `completion_ids[b, t]`, and changed `elif` to `if`.
- **Required Fix**: Replace the pseudo-code with the verbatim code from [`src/trainer/grpo_trainer.py` lines 743–753](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L743-L753).

---

### Issue 7: Hallucinated Line Number `L83` on Binary Checkpoint File `training_args.bin`
- **Deliverable Location**: [`CODE_VS_PAPER_TRACE.md` line 129](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L129)
- **Deliverable Claim**:
  `loss_lvr_lambda = 0.1 (verified in repo/src/params.py L29, repo/scripts/finetune_lvr_stage1_7b.sh L32, and deserialized hf_metadata/LVR-7B/training_args.bin L83).`
- **Actual File Reality ([`hf_metadata/LVR-7B/training_args.bin`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/training_args.bin))**:
  - `training_args.bin` is a binary PyTorch pickled artifact. It has no line numbers.
  - When deserialized with `torch.load(..., weights_only=False)` into a dictionary (`args.to_dict()`), `loss_lvr_lambda` is attribute/key index **132** (`loss_lvr_lambda: 0.1`).
  - Key index 83 in the deserialized dictionary is `optim = 'adamw_torch'`!
- **Required Fix**: Remove `L83` and cite as `deserialized attribute loss_lvr_lambda = 0.1 in hf_metadata/LVR-7B/training_args.bin`.

---

### Issue 8: Truncated Line Range for `_lvr_deocding_by_steps` in `CODE_VS_PAPER_TRACE.md`
- **Deliverable Location**: [`CODE_VS_PAPER_TRACE.md` lines 34 and 640](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L34)
- **Deliverable Claim**:
  Cites `src/model/qwen_lvr_model.py#L855-L1050`, `Lines 855–1050`.
- **Actual File Reality ([`src/model/qwen_lvr_model.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py))**:
  - Function `def _lvr_deocding_by_steps` begins at line 855.
  - Line 1050 is inside the `while` loop (`lvr_mode_switch = new_mode_switch & (lvr_remaining_steps > 0)`).
  - The loop continues until line 1071, streamer shutdown runs lines 1072–1074, and output generation objects (`GenerateDecoderOnlyOutput`) return at line **1095**.
- **Required Fix**: Update cited range to [`src/model/qwen_lvr_model.py` lines 855–1095](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L855-L1095).

---

### Issue 9: Complete Omission of Equation 6 from Section 2 Mathematical Mapping Matrix
- **Deliverable Location**: [`CODE_VS_PAPER_TRACE.md` Section 2 (lines 40–195)](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L40-L195)
- **Deliverable Claim**:
  Header states: *"Below is the line-by-line verification of every mathematical formula presented in the paper text and LaTeX sources."*
  However, Section 2 only contains boxes for Equation 1 (L46), Equation 2 (L78), Equation 3 (L112), Equation 4 (L138), and Equation 5 (L172).
- **Actual File Reality ([`paper_src/sections/3_lvr.tex` lines 111–116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116))**:
  The paper LaTeX source explicitly includes **Equation 6**:
  ```latex
  \begin{align}
  \tilde{R}_i \;=\; \frac{R(y_i)-\mathrm{mean}(R(y_{1}),\dots,R(y_{G}))}
  {\mathrm{std}(R(y_{1}),\dots,R(y_{G}))}, \qquad
  \hat{A}_{i,t} \;=\; \tilde{R}_i \quad (\forall t\in\{1,\dots,|y_i|\}).
  \label{eq:adv_text_only}
  \end{align}
  ```
  This is implemented in [`src/trainer/grpo_trainer.py` lines 882–895](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L882-L895):
  ```python
  mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
  std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
  advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
  ```
- **Required Fix**: Add an Equation 6 box to Section 2 detailing the reward normalization and advantage broadcast formula and its implementation in `grpo_trainer.py`.

---

### Issue 10: Omission of Critical Architecture & Patch Modules in Trace
- **Deliverable Target**: [`CODE_VS_PAPER_TRACE.md` Section 1.1 and Section 3](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L17)
- **Deliverable Omission**:
  The trace fails to catalog or cite three key architecture and runtime patch files:
  1. [`src/model/lvr_heads.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/lvr_heads.py) (62 lines): Defines `LVRHead` (simple MLP, lines 5–20) and `LVRHeadGLU` (GLU MLP with SiLU, lines 24–41), directly realizing the ablation models `\modelname_{MLP Head}` and `\modelname_{GLU Head}` in [`paper_src/tables/ablation.tex` lines 14–15](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/ablation.tex#L14-L15).
  2. [`src/train/monkey_patch_patch_emb.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_patch_emb.py) (23 lines): Monkey-patches `Qwen2_5_VisionPatchEmbed.forward` to force fp32 autocast on 3D patch projection, executed in [`train_lvr.py` line 146](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L146), [`train_sft.py` line 131](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_sft.py#L131), and [`train_grpo.py` line 180](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_grpo.py#L180).
  3. [`src/train/monkey_patch_dataloader.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_dataloader.py) (60 lines): Monkey-patches Trainer's `get_train_dataloader` to bypass HuggingFace samplers for `PackedDataset`.
- **Required Fix**: Integrate these three modules into Section 1.1 Structural Component Mapping Table and Section 3 Forward Pass Trace.

---

### Issue 11: Fabricated Code Line in `evaluation.py` Quote in `Gaps_LVR.md`
- **Deliverable Location**: [`Gaps_LVR.md` lines 335–341](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L335-L341)
- **Deliverable Quote**:
  ```python
  def get_task_instruction(bench_name):
      if bench_name in ["vstar", "mmvp", "blink"]:
          return "\nAnswer with the option's letter from the given choices directly."
  ```
- **Actual File Reality ([`evaluation/evaluation.py` lines 63–72](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L63-L72))**:
  ```python
  63: def get_task_instruction(bench_name):
  64:     if bench_name == "vstar":
  65:         return "\nAnswer with the option's letter from the given choices directly."
  66:     elif bench_name == "mmvp":
  67:         return "\nAnswer with the option's letter from the given choices directly."
  68:     elif bench_name == "blink":
  69:         return "\nAnswer with the option's letter from the given choices directly."
  70:     else:
  71:         raise ValueError(f"Unknown benchmark: {bench_name}")
  ```
  The line `if bench_name in ["vstar", "mmvp", "blink"]:` does not exist in the codebase.
- **Required Fix**: Quote the verbatim `if ... elif ...` chain from [`evaluation/evaluation.py` lines 63–71](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L63-L71).

---

### Issue 12: Bash `export` Syntax Falsely Attributed to Python File `evaluation.py`
- **Deliverable Location**: [`Gaps_LVR.md` lines 377–387](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L377-L387)
- **Deliverable Claim**:
  Quotes bash `export` syntax and claims it exists in both `scripts/vanilla_sft_7b.sh` and `evaluation/evaluation.py`:
  ```bash
  export ACCESS_KEY_ID="cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
  export SECRET_ACCESS_KEY="bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
  export ENDPOINT_URL="https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
  export BUCKET_NAME="bangzhengli"
  export REGION_NAME="us-ashburn-1"
  ```
- **Actual File Reality ([`evaluation/evaluation.py` lines 17–21](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L17-L21))**:
  `evaluation.py` is a Python script and defines variables in Python:
  ```python
  17: access_key_id = "cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
  18: secret_access_key = "bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
  19: endpoint_url = "https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
  20: bucket_name = "bangzhengli"
  21: region_name = "us-ashburn-1"
  ```
- **Required Fix**: Explicitly differentiate between the bash export block in `scripts/vanilla_sft_7b.sh:8-12` and the Python variable definitions in `evaluation/evaluation.py:17-21`.

---

### Issue 13: Non-Existent Filename `stage2_3b.sh` Cited Across Deliverables
- **Deliverable Locations**:
  - [`CODE_VS_PAPER_TRACE.md` line 715](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L715): ``LR=5e-6` (`stage2_3b.sh` L19)`
  - [`CODE_VS_PAPER_TRACE.md` line 716](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md#L716): ``2560 * 28 * 28` (`stage2_3b.sh` L51)`
  - [`Gaps_LVR.md` line 369](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L369): ``2560 * 28 * 28` (`stage2_3b.sh` L51)`
- **Actual File Reality ([`repo/scripts/`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/))**:
  There is no file named `stage2_3b.sh`. The actual file is [`scripts/finetune_lvr_stage2_3b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh).
- **Required Fix**: Replace all instances of `stage2_3b.sh` with `scripts/finetune_lvr_stage2_3b.sh`.

---

### Issue 14: Missing & Over-Broad Anchor URLs in `Gaps_LVR.md`
- **Deliverable Locations**:
  - [`Gaps_LVR.md` line 16](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L16): Text cites `3_lvr.tex lines 20–24, 52–58`, but link URL is [`3_lvr.tex#L52-L58`](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L52-L58), omitting lines 20–24.
  - [`Gaps_LVR.md` line 273](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L273): Text cites `src/model/qwen_lvr_model.py lines 953–954, 1045–1049`, but link URL is [`qwen_lvr_model.py#L953-L954`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954), omitting lines 1045–1049.
  - [`Gaps_LVR.md` line 284](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md#L284): Text cites `lines 1013, 1055`, but link URL anchor spans the entire 43-line block [`#L1013-L1055`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L1013-L1055).
- **Required Fix**: Update link texts or split anchors to accurately target the cited lines.

---

## Adversarial Verification of the 5 Core Gaps

The 5 core gaps detailed in `Gaps_LVR.md` were independently tested and verified against raw disk assets:

### Gap 1: ROI Copying Paradox (Tautological Latent Reconstruction)
- **Status**: **CONFIRMED & FULLY SUPPORTED**
- **Verification Evidence**:
  - In [`src/train/monkey_patch_forward_lvr.py` lines 181–209](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L181-L209), the entire input image is mapped to ViT patch embeddings $\mathbf{V}_T$ and scattered into the causal prefix context under `<|image_pad|>`.
  - In lines 241–244, the input embeddings at the LVR slots are overwritten with the exact ground-truth ViT embeddings of the target ROI: `inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds`.
  - In lines 329–334, the preceding hidden state `seq_positions - 1` is supervised via MSE loss against `selected_lvr_embeds`.
  - Because causal self-attention gives position `seq_positions - 1` unconstrained access to all earlier tokens—including the visual prefix $\mathbf{V}_T$—the network merely executes an auto-associative memory copy of existing visual features rather than generative mental reasoning.

### Gap 2: Latent Budget vs Adaptive Exit Collapse
- **Status**: **CONFIRMED & FULLY SUPPORTED**
- **Verification Evidence**:
  - **Latent End Token Collapse**: [`paper_src/tables/ablation.tex` lines 12–15](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/ablation.tex#L12-L15) proves catastrophic collapse on LVR-7B: $V^*$ overall collapses from **81.7% to 39.8%** (-41.9%), MMVP collapses from **71.7% to 19.0%** (-52.7%), IQ-Test collapses from **29.3% to 6.7%** (-22.6%), and JigSaw collapses from **52.0% to 13.3%** (-38.7%). In code ([`src/model/qwen_lvr_model.py` lines 81–92](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L81-L92)), Euclidean distance thresholding in 3584-dimensional space wandered without triggering.
  - **Mode Switching Loss Collapse**: Paper Section 4.3 ([`4_experimetns.tex` line 74](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/4_experimetns.tex#L74)) admits it failed and collapsed to 0 LVR steps. In released checkpoint [`hf_metadata/LVR-7B/training_args.bin`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/training_args.bin), `mode_switch_loss = False` is deactivated.
  - **Fixed Budget Shackling**: The model relies entirely on fixed step counts ($K \in \{4, 8, 16\}$) via `_lvr_deocding_by_steps` ([`src/model/qwen_lvr_model.py` lines 855–1095](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L855-L1095)).
  - **Vocabulary Leakage**: Line 1055 simultaneously samples `next_tokens = torch.argmax(next_token_scores)` and appends them to `input_ids` during latent mode, corrupting sequence history with garbage tokens.

### Gap 3: Data Leakage & Baseline Suppression in Table 2
- **Status**: **CONFIRMED & FULLY SUPPORTED (Critical Forensic Finding)**
- **Verification Evidence**:
  - In [`paper_src/tables/main_table.tex` lines 20–29](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/main_table.tex#L20-L29) and [`paper_src/tables/sft_rl.tex` lines 10–11](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex#L10-L11), the authors evaluated both off-the-shelf Base Qwen2.5-VL 3B and Vanilla SFT 3B, but commented them out in LaTeX:
    ```latex
    % Qwen2.5-VL & 74.87 & 82.61 & 63.16 & 59 & 69.17 & 28.0 & 50.0 & 41.79 & 81.82 \\
    % SFT & 75.92 & 85.22 & 61.84 & 62.0 & 63.33 & 22.67 & 50.67 & 41.79 & 86.01\\
    ```
  - The base Qwen2.5-VL 3B model outperforms LVR 3B and LVR_RL 3B across every benchmark:
    - $V^*$ Overall: Base **74.87%** vs LVR_RL 3B **67.02%** (**-7.85% regression**)
    - $V^*_{D.A.}$: Base **82.61%** vs LVR_RL 3B **72.17%** (**-10.44% regression**)
    - BLINK Counting: Base **69.17%** vs LVR_RL 3B **55.83%** (**-13.34% regression**)
  - By hiding the base model behind LaTeX `%` comments and reporting only PAPO (31.94%), the authors concealed that LVR caused severe regression relative to the base model.

### Gap 4: Hyperparameter Divergence & Detached Gradients in RL
- **Status**: **CONFIRMED & FULLY SUPPORTED**
- **Verification Evidence**:
  - RL learning rate: Paper claims $1 \times 10^{-5}$ ([`4_experimetns.tex` line 27](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/4_experimetns.tex#L27)). Scripts [`scripts/finetune_lvr_stage2_3b.sh` line 19](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh#L19) and `finetune_lvr_stage2_7b.sh` line 19 hardcode `LR=5e-6` ($2\times$ smaller).
  - RL max image resolution: Cut by 50% from 5,120 patches to 2,560 patches (`--image_max_pixels $((2560 * 28 * 28))` at line 51).
  - 7B RL script: Fully functional script exists in [`scripts/finetune_lvr_stage2_7b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh) despite paper claiming RL was not extended to 7B due to compute limitations.
  - Detached gradients: In [`src/trainer/grpo_trainer.py` line 1482](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1482), `_replay_lvr_to_collect_states` is decorated with `@torch.no_grad()`. In lines 750–758, the LVR span is masked out of `final_mask`. Consequently, $\frac{\partial J_{\mathrm{GRPO}}}{\partial \theta_{\mathrm{latent}}} \equiv 0$. Zero policy gradient flows into latent reasoning.

### Gap 5: Evaluation Biases & Non-Standard Harness
- **Status**: **CONFIRMED & FULLY SUPPORTED**
- **Verification Evidence**:
  - Standard compliance claim: Paper claims *"all evaluations follow the standardized metrics provided by LMMs-Eval"* ([`4_experimetns.tex` line 35](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/4_experimetns.tex#L35)).
  - In reality, [`evaluation/evaluation.py`](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py) does not use `lmms-eval`.
  - Parsing relies on aggressive single-character extraction: `given_answer[0]` ([lines 55–60](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L55-L60)).
  - Queries are forcibly rewritten to change option formats (`(a)` $\rightarrow$ `A.`) and mandatory prompt directives are injected ([lines 65, 262–264](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L262-L264)).

---

## Actionable Remediation Checklist for `forensic-builder`

To achieve an unassailable `VERDICT: PASS (0 MISTAKES)` in Round 2, `forensic-builder` must apply the following specific corrections:

1. [ ] **Escape String Literals**: In `make_code_trace.py` and `make_gaps_lvr.py`, fix all unescaped `\n` and `\t` literals so that:
   - `pattern = r'\n?' + re.escape(LVR_PLACEHOLDER) + r'\n?'` in `CODE_VS_PAPER_TRACE.md` line 592 is on a single, valid line.
   - `return "\nAnswer with the option's letter from the given choices directly."` in `CODE_VS_PAPER_TRACE.md` line 679 and `Gaps_LVR.md` line 339 is on a single, valid line.
   - Corrupted LaTeX strings in `Gaps_LVR.md` lines 185, 216, 221 are restored to `\textbf{67.0}` and `$1 \times 10^{-5}$`.
2. [ ] **Fix M-RoPE Line Citation**: In `CODE_VS_PAPER_TRACE.md` line 337, change `config.json lines 24–28` to `config.json lines 28–33`.
3. [ ] **Fix Projector Freezing Citation**: In `CODE_VS_PAPER_TRACE.md` Table 1.1 line 22, cite `src/train/train_lvr.py` lines 43, 49–50 and `src/trainer/lvr_trainer.py` lines 63–65 instead of docstring lines 42–44.
4. [ ] **Correct GRPO Loss Citation & Quote**: In `CODE_VS_PAPER_TRACE.md` line 152, cite `src/trainer/grpo_trainer.py` lines 1025–1035 and 1058–1085, and replace the fabricated one-line loss formula with the actual clipped surrogate code from lines 1059–1066.
5. [ ] **Restore Verbatim Masking Code**: In `CODE_VS_PAPER_TRACE.md` lines 448–463, replace pseudo-code with verbatim code from `src/trainer/grpo_trainer.py` lines 743–753 (`prompt_completion_ids.size(0)`, `prompt_completion_ids[b, t].item()`, `elif`).
6. [ ] **Remove Binary Line Number**: In `CODE_VS_PAPER_TRACE.md` line 129, remove `L83` and reference `training_args.bin` as a deserialized dictionary attribute (`loss_lvr_lambda = 0.1`).
7. [ ] **Update Decoding Function Range**: In `CODE_VS_PAPER_TRACE.md` lines 34 and 640, update line range to `src/model/qwen_lvr_model.py lines 855–1095`.
8. [ ] **Add Equation 6 Box**: In `CODE_VS_PAPER_TRACE.md` Section 2, insert Equation 6 (`3_lvr.tex` lines 111–116) and its mapping to `src/trainer/grpo_trainer.py` lines 882–895.
9. [ ] **Trace Missing Modules**: In `CODE_VS_PAPER_TRACE.md` Section 1.1 and Section 3, document `src/model/lvr_heads.py` (lines 5–41), `src/train/monkey_patch_patch_emb.py` (lines 10–22), and `src/train/monkey_patch_dataloader.py` (lines 20–55).
10. [ ] **Fix `evaluation.py` Task Instruction Quote**: In `Gaps_LVR.md` lines 337–341, quote the verbatim `if ... elif ...` chain from `evaluation/evaluation.py` lines 63–71 instead of the fabricated `in ["vstar", "mmvp", "blink"]`.
11. [ ] **Differentiate Cloud Credential Formats**: In `Gaps_LVR.md` lines 377–387, show bash `export` syntax for `scripts/vanilla_sft_7b.sh:8-12` and Python variable assignments for `evaluation/evaluation.py:17-21`.
12. [ ] **Fix Script Filename**: In `CODE_VS_PAPER_TRACE.md` lines 715–716 and `Gaps_LVR.md` line 369, replace all references to `stage2_3b.sh` with `scripts/finetune_lvr_stage2_3b.sh`.
13. [ ] **Correct Multi-Range Anchors**: In `Gaps_LVR.md`, fix links at line 16, line 273, and line 284 to accurately target cited lines.
