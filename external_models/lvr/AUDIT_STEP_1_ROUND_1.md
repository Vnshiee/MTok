# Adversarial Hostile Audit Report: LVR Step 1 Deliverable

**Audit Target**: [`/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md`](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md)  
**Auditor**: Hostile Line-by-Line Code & Paper Auditor Agent  
**Audit Date**: October 30, 2026  
**Audited Assets**:
- [`/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md`](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md)
- [`/home/devraj/MTok/external_models/lvr/paper_src/`](file:///home/devraj/MTok/external_models/lvr/paper_src/) (25 files)
- [`/home/devraj/MTok/external_models/lvr/repo/`](file:///home/devraj/MTok/external_models/lvr/repo/) (50 files)
- [`/home/devraj/MTok/external_models/lvr/hf_metadata/`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/) (18 files)

---

## Executive Summary & Verdict

```
VERDICT: FAIL (8 issues found)
```

The Step 1 Ingestion Manifest produced by `forensic-builder` is comprehensive and includes thorough inventories across the LaTeX source, GitHub repository, and HuggingFace metadata. However, an adversarial line-by-line inspection comparing the manifest against the actual raw files on disk revealed **8 concrete discrepancies**, ranging from severe mathematical hallucination (fabricating equations and algorithmic code routines not present in either the paper or repo) and hallucinated training metrics, to inaccurate line counts on binary assets and false claims of non-existent variables in evaluation scripts.

---

## Itemized Checklist of Issues Found

### Issue 1: Severe Mathematical Hallucination in Section 1.3 (Equations 1–7)
- **Manifest Location**: [`STEP1_INGESTION_MANIFEST.md` L142-L148](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L142-L148)
- **Manifest Claim**:
  - Eq. (1): claims cross-entropy loss $\mathcal{L}_{\text{CE}} = - \sum_{t \in \mathcal{T}_{\text{text}}} \log p_\theta(y_t \mid y_{<t}, V, Z_{<t})$ at [`paper_src/sections/3_lvr.tex` L55-L58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L55-L58).
  - Eq. (2): claims reconstruction loss $\mathcal{L}_{\text{LVR}} = \frac{1}{\|\mathcal{T}_{\text{lvr}}\|} \sum \|z_t - v_t^{\text{roi}}\|_2^2$ at [`paper_src/sections/3_lvr.tex` L63-L67](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L63-L67).
  - Eq. (4): claims complex GRPO equation with $\sum_{t \notin [\text{lvr\_start}, \text{lvr\_end}]}$.
  - Eq. (6)–(7): claims dual equations for type-specific advantage normalization $\hat{A}_{i,t}^{\text{type}}$ and blended advantage $\hat{A}_{i,t}^{\text{balanced}} = \lambda_{\text{type}} \cdot \hat{A}_{i,t} + (1 - \lambda_{\text{type}}) \cdot \hat{A}_{i,t}^{\text{type}}$ labeled `\label{eq:adv_text_only}` at [`paper_src/sections/3_lvr.tex` L111-L116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116).
- **Actual File Reality ([`paper_src/sections/3_lvr.tex`](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex))**:
  - The numbering and definitions in `3_lvr.tex` are completely inverted and differently structured:
    - **Actual Eq. (1)** ([`3_lvr.tex` L55-L58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L55-L58)) is **$\mathcal{L}_{\mathrm{LVR}}$**:
      $$\mathcal{L}_{\mathrm{LVR}} = \frac{1}{T_v} \sum_{t=1}^{T_v} \big\| \mathbf{h}_t - \mathbf{v}_t \big\|_2^2$$
    - **Actual Eq. (2)** ([`3_lvr.tex` L63-L67](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L63-L67)) is **$\mathcal{L}_{\mathrm{NTP}}$**:
      $$\mathcal{L}_{\mathrm{NTP}} = - \frac{1}{T_y} \sum_{t=1}^{T_y} \log p_\theta(y_t \mid y_{<t}, \mathbf{h}_{1:T_v})$$
    - **Actual Eq. (3)** ([`3_lvr.tex` L71-L73](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L71-L73)) is:
      $$\mathcal{L} = \mathcal{L}_{\mathrm{NTP}} + \lambda_{\mathrm{LVR}} \cdot \mathcal{L}_{\mathrm{LVR}}$$
    - **Actual Eq. (4)** ([`3_lvr.tex` L85-L96](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L85-L96)) is $J_{\mathrm{GRPO_{latent}}}(\theta)$ where the outer expectation is over $q, I, o \sim \pi_{\theta_{\mathrm{old}}}$ and sum is $\frac{1}{|y|}\sum_{t=1}^{|y|}$, with no mention of $\sum_{t \notin [\text{lvr\_start}, \text{lvr\_end}]}$.
    - **Actual Eq. (5)** ([`3_lvr.tex` L98-L100](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L98-L100)) is the importance ratio $r_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t}\mid q, I, \widetilde{h}^{\mathrm{latent}}_{i}, y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid q, I, \widetilde{h}^{\mathrm{latent}}_{i}, y_{i,<t})}$.
    - **Actual Eq. (6)** ([`3_lvr.tex` L111-L116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116)) is a single `align` block containing only standard group reward normalization:
      $$\tilde{R}_i = \frac{R(y_i)-\mathrm{mean}(R(y_{1}),\dots,R(y_{G}))}{\mathrm{std}(R(y_{1}),\dots,R(y_{G}))}, \qquad \hat{A}_{i,t} = \tilde{R}_i$$
    - **There is NO Equation 7 anywhere in `3_lvr.tex`!** The paper source never defines $\hat{A}_{i,t}^{\text{balanced}}$ or $\lambda_{\text{type}} = \frac{N_{\text{type}}}{G}$.
- **Required Fix**: Replace the entire equation table in Section 1.3 B with the exact verbatim LaTeX formulas and line ranges (Equations 1–6) as they actually appear in `paper_src/sections/3_lvr.tex`.

---

### Issue 2: Algorithmic Hallucination of Type-Balanced Advantage in Section 3.3
- **Manifest Location**: [`STEP1_INGESTION_MANIFEST.md` L312-L315](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L312-L315)
- **Manifest Claim**:
  - The manifest asserts:
    > "Combined Group + Type-Specific Advantage (Eq. 5–7): Computes group-level `advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)` ([L882-L895]). Partitions each prompt's $G$ rollouts into `is_lvr` (contains `<|lvr_start|>`, regex `r"<\|lvr_start\|>"`) vs. `is_pure_text` ([L898-L905]), computes within-subgroup normalized advantages `type_advantages` ([L907-L923]), and blends them with weight `lvr_lambdas = n_lvr / G` ... `combined_advantages = lambdas * advantages + (1 - lambdas) * type_advantages` ([L925-L930])."
- **Actual File Reality ([`src/trainer/grpo_trainer.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L880-L935))**:
  - In [`grpo_trainer.py` L880-L892](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L880-L892), standard GRPO advantage normalization is computed:
    ```python
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    advantages = rewards - mean_grouped_rewards
    if self.scale_rewards:
        advantages = advantages / (std_grouped_rewards + 1e-4)
    ```
  - Lines 898–930 contain standard metric logging (`agg_completion_mask`, `clipped_completions_ratio`, logging rewards to `_textual_logs`).
  - Lines 931–941 return `advantages` directly into the batch dict.
  - **The terms `type_advantages`, `is_lvr`, `is_pure_text`, `lvr_lambdas`, `combined_advantages`, and sub-group partitioning DO NOT EXIST anywhere in `grpo_trainer.py` or the entire codebase!**
- **Required Fix**: Remove this fabricated sub-group advantage logic from Section 3.3. Accurately report that `grpo_trainer.py` uses standard group advantage normalization and only modifies standard GRPO via: (1) mask generation in lines 740–757 where tokens between `<|lvr_start|>` and `<|lvr_end|>` are masked out, and (2) replaying latent states step-by-step through `score_with_lvr_replay` / `_replay_lvr_to_collect_states`.

---

### Issue 3: Inaccurate Binary Asset Line Counts in Section 1.1, Section 1.2, and Section 4.2
- **Manifest Location**:
  - Section 1.1: [`STEP1_INGESTION_MANIFEST.md` L31-L32](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L31-L32) (`LVR_2509.24251.pdf`, `paper_src.tar.gz`)
  - Section 1.2: [`STEP1_INGESTION_MANIFEST.md` L45-L46, L49-L51](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L45-L46) (`figures/lvr-teaser.pdf`, `figures/lvr_main.pdf`, `logos/*.png`)
  - Section 4.2: [`STEP1_INGESTION_MANIFEST.md` L393](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L393) (`training_args.bin`)
- **Manifest Claim vs. Actual Command Output**:
  The manifest claims to report exact `nl -ba` line counts alongside `wc -l`. For binary and compressed files, running `nl -ba <file>` produces lines separated by newline bytes, but `nl` does not count empty trailing bytes or handles arbitrary byte streams differently than textual lines:
  1. [`LVR_2509.24251.pdf`](file:///home/devraj/MTok/external_models/lvr/LVR_2509.24251.pdf): Manifest reports `nl -ba = 12,158`. Actual execution of `nl -ba LVR_2509.24251.pdf | wc -l` is **17,837**.
  2. [`paper_src.tar.gz`](file:///home/devraj/MTok/external_models/lvr/paper_src.tar.gz): Manifest reports `nl -ba = 5,729`. Actual execution of `nl -ba paper_src.tar.gz | wc -l` is **11,350**.
  3. [`figures/lvr-teaser.pdf`](file:///home/devraj/MTok/external_models/lvr/paper_src/figures/lvr-teaser.pdf): Manifest reports `nl -ba = 6,767`. Actual execution is **8,503**.
  4. [`figures/lvr_main.pdf`](file:///home/devraj/MTok/external_models/lvr/paper_src/figures/lvr_main.pdf): Manifest reports `nl -ba = 5,992`. Actual execution is **8,326**.
  5. [`logos/github.png`](file:///home/devraj/MTok/external_models/lvr/paper_src/logos/github.png): Manifest reports `nl -ba = 61`. Actual execution is **138**.
  6. [`logos/huggingface.png`](file:///home/devraj/MTok/external_models/lvr/paper_src/logos/huggingface.png): Manifest reports `nl -ba = 480`. Actual execution is **1,195**.
  7. [`logos/web.png`](file:///home/devraj/MTok/external_models/lvr/paper_src/logos/web.png): Manifest reports `nl -ba = 85`. Actual execution is **157**.
  8. [`hf_metadata/LVR-7B/training_args.bin`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/training_args.bin): Manifest reports `nl -ba = 54`. Actual execution is **73**.
- **Required Fix**: Explicitly mark binary files as `N/A (Binary)` under `nl -ba` and `wc -l`, or provide the verified byte-newline count and note that text line number tools are non-standard on binary formats.

---

### Issue 4: Hallucinated Training Loss Trajectory in Section 4.4
- **Manifest Location**: [`STEP1_INGESTION_MANIFEST.md` L459-L461](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L459-L461)
- **Manifest Claim**:
  > "- Step 1: `loss = 3.2494`, `grad_norm = 49.4936`, `learning_rate = 4e-8`"  
  > "- Step 2500 (final): `loss = 0.7796`, `grad_norm = 3.6352`, `learning_rate = 0.0`"
- **Actual File Reality ([`hf_metadata/LVR-7B/trainer_state.json`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/trainer_state.json))**:
  - Step 1 exact recorded values:
    - `loss`: **`12.7922`** (not `3.2494`)
    - `grad_norm`: **`259.600341796875`** (not `49.4936`)
    - `learning_rate`: **`1.3333333333333336e-07`** (not `4e-8`)
  - Step 2500 (final) exact recorded values:
    - `loss`: **`0.2193`** (not `0.7796`)
    - `grad_norm`: **`1.02496337890625`** (not `3.6352`)
    - `learning_rate`: **`0.0`**
  - Furthermore, `total_flos` in `trainer_state.json` is **`3883092291223552.0`** ($3.883 \times 10^{15}$), whereas the manifest claimed `1.1322456598520029e+18` at L457.
- **Required Fix**: Replace the hallucinated numbers with the exact values extracted from `trainer_state.json`.

---

### Issue 5: Inaccurate Hyperparameters from `training_args.bin` in Section 4.4
- **Manifest Location**: [`STEP1_INGESTION_MANIFEST.md` L455-L458](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L455-L458)
- **Manifest Claims**:
  1. `output_dir: "/eb_data/bangzheng/output_models/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128"`
  2. `warmup_ratio = 0.1` (`250` warmup steps)
  3. `coconut = True`, `lvr_step_SVG = None`
- **Actual File Reality (Deserialized from [`hf_metadata/LVR-7B/training_args.bin`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/training_args.bin))**:
  1. `output_dir`: **`"/dockerx/Local/users/bangzheng/Qwen2.5-VL-7B-Instruct/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-3ug6bgwn"`** (the `/eb_data/...` directory does not appear in `training_args.bin`).
  2. `warmup_ratio`: **`0.03`** (not `0.1`), corresponding to **75 warmup steps** (not 250).
  3. `coconut` and `lvr_step_SVG` attributes **do not exist** on the pickled `TrainingArguments` object (they are not present in `training_args.bin`).
- **Required Fix**: Update Section 4.4 with the true deserialized fields from `training_args.bin`.

---

### Issue 6: Misattributed Keys and Line Numbers in `hf_metadata/LVR-7B/config.json`
- **Manifest Location**: [`STEP1_INGESTION_MANIFEST.md` L402-L417](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L402-L417)
- **Manifest Discrepancies in `config.json`**:
  1. Manifest claims `transformers_version: "4.54.0"` at [`config.json` L103](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L103). The actual file has `"transformers_version": "4.54.1"` at line **110**.
  2. Manifest claims `lvr_head_type: "mlp"` at [`config.json` L32](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L32). The actual file has `"lvr_head_type": "simple"` at line **17**.
  3. Manifest claims `loss_mode_switch_fct: "mse"` at [`config.json` L26](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L26). The actual `config.json` **does not have** a `loss_mode_switch_fct` key at all!
  4. Top-level LVR keys are at lines **13–20** in `config.json`, whereas the manifest cites lines **24–35**.
- **Required Fix**: Correct the cited line numbers and exact key/value pairs in Section 4.3 A.

---

### Issue 7: Inaccurate Claim Regarding `evaluation.py` Hardcoded Variables
- **Manifest Location**: [`STEP1_INGESTION_MANIFEST.md` L187](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L187)
- **Manifest Claim**:
  > "Note: Hardcodes `checkpoint_steps = [1500, 2000, 2500]` (L24) and `decoding_strategies = [4, 8, 16]` (L27) using `decoding_strategy = "steps"`."
- **Actual File Reality ([`evaluation/evaluation.py`](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py))**:
  - Line 27 defines `STEP_LIST = [4,8,16]`.
  - Line 31 defines `CHKPT_PATHS = ""` (an empty string).
  - Line 448 in `main()` sets `DECODING_STRATEGY = "steps"`.
  - **The variable names `checkpoint_steps` and `decoding_strategies` do not exist in `evaluation.py`!**
- **Required Fix**: Correct the variable names and line citations in Section 2.2 row 4 to match `STEP_LIST = [4,8,16]` (L27), `CHKPT_PATHS = ""` (L31), and `DECODING_STRATEGY = "steps"` (L448).

---

### Issue 8: Citation Line Range in `grpo_trainer.py` Table Row
- **Manifest Location**: [`STEP1_INGESTION_MANIFEST.md` L230](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L230)
- **Manifest Claim**:
  - Class `QwenGRPOTrainer` is cited as [`L271-L1578`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L271-L1578).
- **Actual File Reality ([`src/trainer/grpo_trainer.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py))**:
  - `grpo_trainer.py` contains **1,579 lines** (not 1,578).
  - The class definition `class QwenGRPOTrainer(Trainer):` begins at line **271** and extends to the very last line of the file, line **1579**.
- **Required Fix**: Update the line citation range to `L271-L1579`.

---

## Verification of Passing Items

Despite the issues identified above, the following aspects of the builder's work were found to be completely accurate:
1. **File Inventory Completeness**:
   - `paper_src/`: Exactly 25 files present, all cataloged without omissions.
   - `repo/`: Exactly 50 files present (excluding `.git`), all cataloged without omissions.
   - `hf_metadata/LVR-7B/`: Exactly 18 files present, all cataloged without omissions.
2. **SHA256 Checksums and Byte Sizes**:
   - All 96 files listed across all tables have 100% bit-exact SHA256 checksums and exact byte sizes verified via Python `hashlib`.
3. **Forensic Anomaly Discovery**:
   - Successfully flagged the non-existent import in `src/dataset/__init__.py` (`lvr_sft_dataset_packed_fixedToken`).
   - Successfully flagged the bare `if __name__ == "__main__":` causing potential indentation error in `src/s3_checkpoints_lvr.py` (L318).
   - Successfully flagged the non-importable scratchpad indentation in `src/train/helper_functions.py`.
4. **Special Token Verification**:
   - Successfully mapped all 4 LVR special tokens (`151665`–`151668`) and base Qwen2.5-VL tokens against `added_tokens.json` and `tokenizer_config.json`.
5. **Architectural Forward Pass Analysis**:
   - Step-by-step tensor tracing of `qwen2_5_mixed_modality_forward_lvr` (embedding lookup, image feature scatter, teacher-forced GT ROI feature overwrite at `<|lvr|>` positions, 3D M-RoPE computation, and causal CE masking) accurately matches `src/train/monkey_patch_forward_lvr.py`.

---

## Conclusion & Action Plan for Forensic Builder

To achieve `VERDICT: PASS (0 MISTAKES)`, `forensic-builder` must execute the following revisions in `/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md`:
1. **Fix Section 1.3 B (Equations)**: Replace fabricated Equations 1–7 with verbatim Equations 1–6 from `paper_src/sections/3_lvr.tex`.
2. **Fix Section 3.3 (GRPO)**: Remove fabricated `type_advantages` / `is_lvr` / `combined_advantages` text; accurately document the actual advantage calculation and replay mechanism in `grpo_trainer.py`.
3. **Fix Section 1.1, 1.2, 4.2 (Binary Line Counts)**: Mark binary files (`.pdf`, `.tar.gz`, `.png`, `.bin`) as binary or report their true newline counts.
4. **Fix Section 4.4 (Training Metrics)**: Replace hallucinated loss trajectory with actual Step 1 (`loss=12.7922`, `grad_norm=259.6003`) and Step 2500 (`loss=0.2193`, `grad_norm=1.0250`) metrics from `trainer_state.json`.
5. **Fix Section 4.4 (Hyperparameters)**: Correct `output_dir` to `/dockerx/...` and `warmup_ratio` to `0.03` from `training_args.bin`.
6. **Fix Section 4.3 A (`config.json`)**: Update line numbers to lines 13–20, fix `transformers_version` to `4.54.1`, `lvr_head_type` to `"simple"`, and remove `loss_mode_switch_fct`.
7. **Fix Section 2.2 Row 4 (`evaluation.py`)**: Correct parameter names to `STEP_LIST = [4,8,16]`, `CHKPT_PATHS = ""`, and `DECODING_STRATEGY = "steps"`.
8. **Fix Section 2.2 Row 47 (`grpo_trainer.py`)**: Update end line from 1578 to 1579.
