# LVR (Latent Visual Reasoning) — Comprehensive Theoretical Ablation Study & Mechanistic Falsification Design

**Target System:** `vincentleebang/LVR-7B` & `LVR-3B` ([`VincentLeebang/lvr`](https://github.com/VincentLeebang/lvr), `arXiv:2509.24251`)  
**Base Backbone:** `Qwen/Qwen2.5-VL-7B-Instruct` ($L=28$ layers, $D=3584$, $H_Q=28$, $H_{KV}=4$) & `Qwen/Qwen2.5-VL-3B-Instruct`  
**Certified Forensic Evidence Base:**  
- [STEP1_INGESTION_MANIFEST.md](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md) (SHA-256 verified weights, vocabulary, scripts, and data manifests)  
- [CODE_VS_PAPER_TRACE.md](file:///home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md) (Line-by-line equation-to-tensor execution trace)  
- [Gaps_LVR.md](file:///home/devraj/MTok/external_models/lvr/Gaps_LVR.md) (14 structural, mathematical, and empirical discrepancies)  
- [LVR_EMPIRICAL_REPORT.md](file:///home/devraj/MTok/external_models/lvr/LVR_EMPIRICAL_REPORT.md) (Live GPU-0 replication telemetry on `vincentleebang/LVR-7B`)

---

## 0. Audit Changelog: Critical Corrections Over Initial Draft

A rigorous line-by-line verification of the initial ablation design against the source code, LaTeX tables, and checkpoint metadata exposed **seven foundational errors and omissions** that have been corrected in this definitive specification:

1. **Corrected Evaluation Benchmark Suite (Section 4)**:
   - *Prior Error*: The initial draft listed `VSR`, `RefCOCOg`, `MathVista`, `MathVerse`, `OlympiadBench-MM`, `MMBench v1.1`, `ScienceQA-IMG`, and `HallusionBench`—none of which exist in the LVR paper or [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py).
   - *Ground-Truth Correction*: Replaced with the exact LVR evaluation suite from [main_table.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/main_table.tex#L59-L80) and [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L33-L46): **$V^*$ Bench** (`craigwu/vstar_bench`, $N=191$: Direct Attributes $N=115$, Relative Position $N=76$), **MMVP** ($N=300$), and **BLINK** (`BLINK-Benchmark/BLINK` `val` split: 5 core subtasks $N=697$ (`Counting` $120$, `IQ_Test` $150$, `Jigsaw` $150$, `Relative_Reflectance` $134$, `Spatial_Relation` $143$) plus 6 extended subtasks in [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L340)).
2. **Exact Stage 1 SFT Loss Equation, Causal Index Shift & Normalization (Section 2.1 & Axis 1)**:
   - *Prior Error*: Wrote $\mathcal{L}_{\mathrm{MSE}} = \frac{1}{|R|}\sum \|h_{t,i}^{(L)} - v_i\|_2^2$ without accounting for the causal $-1$ index shift (`seq_positions_start = seq_positions - 1`), `float32` upcasting, or feature-dimension normalization ($D=3584$).
   - *Ground-Truth Correction*: Derived the exact tensor equation implemented in [monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L106-L107) and [L241-L244](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L241-L244): `torch.nn.MSELoss(reduction="mean")` averages over both $T_v$ tokens and $D=3584$ hidden dimensions ($\frac{1}{T_v \cdot D}$), reducing effective gradient magnitude by $3584\times$ relative to unnormalized vector L2 loss, while evaluating $\mathbf{h}_{s_k - 1}^{(L)}$ against $\mathbf{v}_k$ under **parallel teacher-forcing** (where clean ground-truth ViT patches $\mathbf{v}_{1:T_v}$ overwrite `<|lvr|>` input embeddings during SFT, creating severe **exposure bias** when switched to autoregressive rollout $\hat{\mathbf{h}}_{k-1}^{(L)}$ at test time).
3. **1D M-RoPE Spatial Grid Destruction ($H_{1c}$, Section 2.1 & Axis 1A)**:
   - *Prior Omission*: Failed to capture that [get_rope_index](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L269-L275) only assigns 2D $(y,x)$ spatial M-RoPE coordinates to `<|image_pad|>` (`151655`), whereas `<|lvr|>` (`151666`) tokens receive 1D monotonic text positions $(t,t,t)$, flattening 2D bounding-box geometry into a 1D strip.
4. **Stage 2 GRPO Masking Interval, Frozen Merger/Head & Suppressed 3B Baselines ($H_2$, Section 2.2 & Axis 2)**:
   - *Prior Errors*:
     1. Claimed `final_mask` zeros the closed interval $[t_{\mathrm{start}}, t_{\mathrm{end}}]$. In [grpo_trainer.py](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L746-L752), `lvr_mask` zeros the **half-open interval** $[t_{\mathrm{start}}, t_{\mathrm{end}})$ in `prompt_completion_ids` (which, after slicing `final_mask[:, -logits_to_keep:]` against shifted target tokens `prompt_completion_ids[:, 1:]` at [L1026-L1031](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1026-L1031), masks `<|lvr_start|>` and all interior latent steps while leaving `<|lvr_end|>` unmasked).
     2. Referenced a trainable projector $\frac{\partial \mathcal{L}_{\mathrm{GRPO}}}{\partial W_{\mathrm{proj}}}$. In reality, `freeze_vision_tower=True` and `freeze_merger=True` in both Stage 1 and Stage 2 ([finetune_lvr_stage2_3b.sh](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh#L45-L47)), and `lvr_head=False` in `vincentleebang/LVR-7B` ([config.json](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L16)). The detached path is w.r.t. the LLM backbone parameters $\theta_{\mathrm{LLM}}$ across recurrent hidden states $\mathbf{H}_{\mathrm{LVR}}$ due to `@torch.no_grad()` on [`_replay_lvr_to_collect_states`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1482) and `.detach()` in [`score_with_lvr_replay`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1427).
     3. Omitted that `vincentleebang/LVR-7B` is a **Stage 1 SFT-only checkpoint** (`stage_2=False`), that Stage 2 RL was only reported on the 3B model ([sft_rl.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex)), and that the authors commented out the Base Qwen2.5-VL-3B (`74.87%` on $V^*$) and Vanilla SFT-3B (`75.92%` on $V^*$) rows in [sft_rl.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex#L10-L11) because both outperform LVR-3B (`65.45%`) and $\text{LVR}_{\text{RL}}$-3B (`67.02%`) by **$-7.85\%$ to $-8.90\%$**.
5. **Exact Step Quotas, Decoding Strategies & Token IDs ($H_3$, Section 2.3 & Axis 3)**:
   - *Prior Error*: Used $K \in \{0, 1, 3, 5, 10, 20, 50\}$ with $K=5$ labeled as default, and mislabeled token `151643` as `<|pad|>`.
   - *Ground-Truth Correction*: Aligned step grid with the paper's `STEP_LIST = [4, 8, 16]` ([evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L27)): $K \in \{0, 1, 2, 4, 8, 16, 32, 64\}$. Corrected token IDs (`151643` is `<|endoftext|>`, used as `bos_token_id` and `pad_token_id`; `151665` `<|lvr_start|>`, `151666` `<|lvr|>`, `151667` `<|lvr_latent_end|>`, `151668` `<|lvr_end|>`). Added explicit ablation of all three decoding loops in [qwen_lvr_model.py](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py): [`_lvr_deocding_by_steps`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L891), [`_lvr_deocding_with_latentend`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L619) (`39.8%` collapse on $V^*$), and [`_lvr_deocding`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L395) (`mode_switch_loss` collapse to 0 steps).
6. **Evaluation Parser & Prompt Mutation Confounds ($H_5$, Section 2.5 & Axis 5)**:
   - *Prior Omission*: Omitted the brittle first-character answer parser (`given_answer = given_answer.strip().split(' ')[0]; return given_answer[0] == ground_truth` in [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L57-L61)) and prompt/option-format mutations (`(a)` $\to$ `A.` in [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L262-L263)).
7. **Paired Statistical Testing & Discordance Protocol (Section 4.3)**:
   - Added exact Paired McNemar's test, exact binomial small-sample handling ($b+c < 25$), $B=10,000$ paired bootstrap 95% CIs, and Holm-Bonferroni family-wise error correction across $V^*$, MMVP, and BLINK.

---

## 1. Forensic Architecture & Anomaly Inventory

Before specifying the experimental matrices, Table 1 codifies the exact ground-truth implementation facts of `vincentleebang/LVR-7B` verified against its HuggingFace weights and source repository.

| Component / Parameter | Paper Claim (`arXiv:2509.24251`) | Actual Code / Checkpoint Implementation | Exact Source Reference |
| :--- | :--- | :--- | :--- |
| **Released 7B Checkpoint** | Trained via Stage 1 SFT + Stage 2 GRPO RL | **Stage 1 SFT-only** (`"stage_2": false` in `config.json`; no 7B RL script exists in repo) | [config.json](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L25) |
| **Latent Projection Head** | Optional reconstruction head | **Disabled (`"lvr_head": false`)**; raw last-layer norm output $\mathbf{h}^{(28)} \in \mathbb{R}^{3584}$ fed directly into Layer 0 | [config.json](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L16), [qwen_lvr_model.py](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L180-L185) |
| **Vision Tower & Merger** | Jointly adapted | **Strictly Frozen** (`freeze_vision_tower=True`, `freeze_merger=True`) in both Stage 1 and Stage 2 | [finetune_lvr_stage1_7b.sh](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage1_7b.sh#L33-L35), [finetune_lvr_stage2_3b.sh](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh#L45-L47) |
| **Added Special Tokens** | `<\|lvr_start\|>`, `<\|lvr\|>`, `<\|lvr_end\|>` | 4 tokens: `151665` (`<\|lvr_start\|>`), `151666` (`<\|lvr\|>`), `151667` (`<\|lvr_latent_end\|>`), `151668` (`<\|lvr_end\|>`) | [added_tokens.json](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/added_tokens.json#L1-L26) |
| **Stage 1 SFT Latent Rollout** | Autoregressive latent generation ("COCONUT-style") | **Single-pass parallel teacher-forcing**: clean ground-truth ViT ROI patches overwrite `<\|lvr\|>` input embeddings (`inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds`) | [monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L241-L244) |
| **Stage 1 SFT MSE Scaling** | $\mathcal{L}_{\mathrm{LVR}} = \frac{1}{T_v}\sum_{t=1}^{T_v}\|\hat{\mathbf{v}}_t - \mathbf{v}_t\|_2^2$ | `nn.MSELoss(reduction="mean")` divides by $T_v \cdot D$ ($D=3584$), scaling gradients down by $3584\times$ | [monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L106-L107) |
| **Latent M-RoPE Coordinates** | Preserves spatial region reasoning | `<\|lvr\|>` (`151666`) receives **1D text positions $(t,t,t)$** instead of 2D spatial $(y,x)$ grid indices | [monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L269-L275) |
| **Stage 2 GRPO Gradient Flow** | Optimizes latent visual reasoning policy | **Zero gradient through latent recurrent states**: `@torch.no_grad()` replay + `.detach()` + `lvr_mask=0` on $[t_{\mathrm{start}}, t_{\mathrm{end}})$ | [grpo_trainer.py](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L746-L752), [L1427](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1427), [L1482](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1482) |
| **Inference Termination** | Learned adaptive `<\|lvr_end\|>` switch | **Hardcoded step counter** $K \in \{4, 8, 16\}$ (`decoding_strategy="steps"`); learned switch collapsed by $-41.9\%$ (`39.8%` on $V^*$) | [qwen_lvr_model.py](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L891-L1005), [ablation.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/ablation.tex#L13) |

---

## 2. Formal Mechanistic Hypotheses

We formulate five falsifiable hypotheses ($H_1$–$H_5$) targeting every stage of the LVR pipeline:

### 2.1 Hypothesis $H_1$: Prefix Identity Copying, Teacher-Forcing Exposure Bias & 1D M-RoPE Distortion (Stage 1 SFT)

In Stage 1 SFT ([monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L202-L343)), the full image is encoded by the frozen ViT + Merger into $N$ patch embeddings $\mathbf{V}_T = [\mathbf{v}_1^{\mathrm{img}}, \dots, \mathbf{v}_N^{\mathrm{img}}] \in \mathbb{R}^{N \times D}$ ($D=3584$), which are placed into the prompt prefix at `<|image_pad|>` (`151655`) positions $p_1, \dots, p_N$. A bounding box defines an index subset $\mathbf{I} = (I_1, \dots, I_{T_v}) \subseteq \{1, \dots, N\}$ of $T_v$ ROI patches ([data_utils.py](file:///home/devraj/MTok/external_models/lvr/repo/src/dataset/data_utils.py#L31-L55)). Inside the target response, $T_v$ placeholder `<|lvr|>` (`151666`) tokens occupy sequence indices $s_1, \dots, s_{T_v}$ (where $s_1 - 1$ is the position of `<|lvr_start|>` `151665`, and $s_k - 1 = s_{k-1}$ for $k \in \{2, \dots, T_v\}$).

During SFT forward execution:
1. **Parallel Teacher-Forcing Overwrite** ([L241-L244](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L241-L244)):
   $$\mathbf{X}_{\mathrm{embed}}[s_k] \leftarrow \mathbf{v}_k \triangleq \mathbf{V}_T[I_k], \quad \forall k \in \{1, \dots, T_v\}$$
2. **Causally Shifted Feature Regression** ([L106-L107](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L106-L107), [L329-L334](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L329-L334)):
   $$\mathcal{L}_{\mathrm{LVR}}^{\mathrm{code}} = \frac{1}{T_v \cdot D} \sum_{k=1}^{T_v} \left\| \mathrm{float32}\!\left(\mathbf{h}_{s_k - 1}^{(L)}\right) - \mathrm{float32}\!\left(\mathbf{v}_k\right) \right\|_2^2, \quad \mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{NTP}} + \lambda_{\mathrm{LVR}} \mathcal{L}_{\mathrm{LVR}}^{\mathrm{code}} \quad (\lambda_{\mathrm{LVR}} = 0.1)$$

This exact formulation induces three distinct sub-hypotheses:
- **$H_{1a}$ (Contextual Prefix Identity Copying)**: Because $\mathbf{v}_k = \mathbf{V}_T[I_k]$ already exists verbatim at prefix index $p_{I_k}$ ($p_{I_k} < s_k - 1$), minimizing $\mathcal{L}_{\mathrm{LVR}}^{\mathrm{code}}$ reduces to an attention-head lookup/copy operation from prefix position $p_{I_k}$ (or, for $k \ge 2$, copying the teacher-forced input $\mathbf{X}_{\mathrm{embed}}[s_{k-1}] = \mathbf{v}_{k-1}$ of an adjacent patch) rather than synthesizing higher-order visual abstractions.
- **$H_{1b}$ (Teacher-Forcing vs. Autoregressive Exposure Bias)**: Despite the codebase naming the flag `coconut=True` ([L119](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L119)), SFT **never unrolls its own predicted states $\mathbf{h}_{s_k - 1}^{(L)}$**. At test time ([_lvr_deocding_by_steps](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L999-L1005)), the model feeds its own noisy last-layer hidden state $\hat{\mathbf{h}}_{k-1}^{(L)}$ into Layer 0 (`inputs_embeds[lvr_mode_switch, -1, :] = last_position_hidden_state[lvr_mode_switch]`). Furthermore, at test time the model has **no bounding box $\mathbf{I}$** provided in the prompt! Thus, the first latent state $\hat{\mathbf{h}}_0^{(L)}$ (at `<|lvr_start|>`) must guess which patch $\mathbf{v}_{I_1}$ to reconstruct without coordinate conditioning, and subsequent steps suffer compounding autoregressive rollout drift.
- **$H_{1c}$ (1D M-RoPE Spatial Grid Destruction)**: In [monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L269-L275), `get_rope_index` assigns 2D $(y,x)$ spatial coordinates only to `<|image_pad|>` (`151655`). The `<|lvr|>` (`151666`) tokens receive 1D sequential text coordinates $(t,t,t)$. When a 2D ROI box of grid shape $H_{\mathrm{roi}} \times W_{\mathrm{roi}}$ is raster-scanned into $T_v = H_{\mathrm{roi}} W_{\mathrm{roi}}$ tokens, row wraps destroy vertical patch adjacency in M-RoPE space.

---

### 2.2 Hypothesis $H_2$: "Ghost RL" — Detached Latent Replay, Half-Open Token Masking & Suppressed 3B Baselines (Stage 2 GRPO)

In Stage 2 GRPO ([grpo_trainer.py](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py)), policy optimization is structurally severed from the continuous latent reasoning states through three mechanisms:
1. **Non-Differentiable Replay (`@torch.no_grad()` + `.detach()`)**: During training rollouts, [`_replay_lvr_to_collect_states`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1482-L1579) is decorated with `@torch.no_grad()`, and [`score_with_lvr_replay`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1425-L1432) injects cached states into `inputs_embeds` via `lvr_states.to(...).detach()`. Consequently, the recurrent Jacobian across latent steps is identically zero:
   $$\frac{\partial \mathbf{h}_{t_{\mathrm{start}}+k}^{(L)}}{\partial \mathbf{h}_{t_{\mathrm{start}}+k-1}^{(L)}} \equiv \mathbf{0} \quad (\text{in the autograd graph of } \mathcal{L}_{\mathrm{GRPO}})$$
2. **Half-Open Log-Probability Masking**: In [`_generate_and_score_completions`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L742-L756), `lvr_mask` sets `False` (`0`) on the half-open interval $[t_{\mathrm{start}}, t_{\mathrm{end}})$ in `prompt_completion_ids` (from `<|lvr_start|>` inclusive to `<|lvr_end|>` exclusive). When `final_mask = (loss_mask & lvr_mask)[:, -logits_to_keep:]` is applied to `per_token_loss` (which scores shifted target tokens `prompt_completion_ids[:, 1:]` at [L1026-L1031](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1026-L1031)):
   - Target token `<|lvr_start|>` (`151665`), all interior latent tokens $t \in (t_{\mathrm{start}}, t_{\mathrm{end}}-1)$, and the last latent token $t_{\mathrm{end}}-1$ are **masked out** (`final_mask = 0`), except for a single 1-token boundary offset where `lvr_mask` transitions to `True` at $t_{\mathrm{end}}$ (`<|lvr_end|>`).
   - No continuous action density $p_\theta(\mathbf{h}_k \mid s_k)$ is ever computed or differentiated; GRPO only differentiates discrete token log-probabilities $\log \pi_\theta(y_t \mid \text{context}, \mathrm{sg}(\mathbf{H}_{\mathrm{LVR}}))$ on the text answer tokens after `<|lvr_end|>`.
3. **Suppressed 3B Baselines & Rigid Prefix Regex**:
   - In [sft_rl.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex#L10-L18), the authors commented out Base Qwen2.5-VL-3B (`74.87%` on $V^*$) and Vanilla SFT-3B (`75.92%` on $V^*$) because LVR-3B (`65.45%`) and $\text{LVR}_{\text{RL}}$-3B (`67.02%`) suffer a **$-7.85\%$ to $-10.47\%$ catastrophic regression** on $V^*$!
   - Furthermore, [`format_reward`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/reward_funcs.py#L49-L56) enforces `^<\|lvr_start\|>.*?<\|lvr_end\|>\s*<answer>.*?</answer>$` (anchored with `^`), strictly forbidding the model from generating any natural-language Chain-of-Thought before `<|lvr_start|>`, while [finetune_lvr_stage2_3b.sh](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh#L49) halves maximum image resolution (`max_pixels = 2560*28*28` vs. `5120*28*28` in SFT).

---

### 2.3 Hypothesis $H_3$: Step Quota Invariance, Unsupervised Termination & Adaptive Stopping Collapse

At inference, LVR uses [`_lvr_deocding_by_steps`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L891-L1005) (`decoding_strategy="steps"`) with hardcoded quotas $K \in \{4, 8, 16\}$ ([evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L27)).
- **Why Learned Stopping Collapsed**: In SFT ([monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L218-L219)), `labels[labels == 151666] = -100` (`IGNORE_INDEX`). Thus, during Stage 1 SFT, when the model's input is a visual hidden state $\mathbf{v}_k$, the **only** unmasked vocabulary target it ever sees in $\mathcal{L}_{\mathrm{NTP}}$ is `<|lvr_end|>` (`151668`) at the end of the ROI sequence! Consequently, `lm_head` places overwhelming probability mass on `<|lvr_end|>` (`151668`) at **every** latent step $k \ge 1$.
- When the authors tested [`_lvr_deocding`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L395) with `mode_switch_loss` ([monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L1345-L1351)), reasoning collapsed to $0$ steps ([4_experimetns.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/4_experimetns.tex#L74)). When they tested [`_lvr_deocding_with_latentend`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L619) (`<|lvr_latent_end|>` `151667`), $V^*$ accuracy plummeted from `81.7%` to `39.8%` ([ablation.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/ablation.tex#L13)).
- Under fixed step quotas (`_lvr_deocding_by_steps`), once the quota `lvr_steps` expires (`lvr_Remaining_steps == 0`), `lvr_mode_switch` turns `False`, and on step $K+1$ the model naturally emits `<|lvr_end|>` (`151668`) from `lm_head` because `<|lvr_end|>` is the sole unmasked token associated with visual hidden states during SFT.
- **Live Empirical Confirmation**: In our GPU-0 verification on `vincentleebang/LVR-7B` ([LVR_EMPIRICAL_REPORT.md](file:///home/devraj/MTok/external_models/lvr/LVR_EMPIRICAL_REPORT.md)), predictions across $K \in \{1, 2, 4, 8, 16\}$ were **100% identical**, with pairwise step-to-step latent cosine similarity $\cos(\mathbf{h}_t, \mathbf{h}_{t+1}) \in [0.692, 0.858]$, indicating rapid convergence to a stationary attractor after step 1.

---

### 2.4 Hypothesis $H_4$: Syntactic Format-Locking vs. Continuous Latent Semantics

In Stage 1 SFT, 100% of Visual-CoT training examples follow the template `<|lvr_start|><|lvr|>...<|lvr|><|lvr_end|><answer>...</answer>`. We hypothesize that `vincentleebang/LVR-7B` is **syntactically format-locked**:
- If prevented from emitting `<|lvr_start|>`...`<|lvr_end|>` (`Direct Answer`), accuracy degrades not because continuous visual computation is lost, but because the language model is forced out-of-distribution relative to its SFT response prefix.
- Conversely, if we satisfy the syntactic prefix gate via an **Empty Delimiter Prefill** (`<|lvr_start|><|lvr_end|>` with $K=0$ continuous steps) or replace the recurrent hidden states $\mathbf{h}_{1:K}^{(L)}$ with discrete `<|lvr|>` token embeddings (`151666`), isotropic Gaussian noise, or the mean prefix image embedding $\bar{\mathbf{v}}$, downstream accuracy will match the default $K=8$ continuous rollout within statistical noise.

---

### 2.5 Hypothesis $H_5$: Evaluation Parser Fragility & Prompt Mutation Confounds

In [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L52-L71):
1. **First-Character Extraction Bug** ([L57-L61](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L57-L61)):
   ```python
   given_answer = given_answer.strip().split(' ')[0]
   return given_answer[0] == ground_truth
   ```
   Any model output starting with `"Answer: B"` or `"The answer is C"` evaluates `given_answer[0]` as `'A'` or `'T'`, systematically penalizing baseline instruct models (`Qwen2.5-VL-7B/3B-Instruct`) that do not wrap single-letter outputs in `<answer>X</answer>`, while favoring SFT/RL models trained on `<answer>X</answer>`.
2. **Option Label Mutation**: [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L262-L263) rewrites MMVP options from `(a) ... (b) ...` to `A. ... B. ...` and appends `"\nAnswer with the option's letter from the given choices directly."` ([L70](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L70)).

---

## 3. The 6-Axis Experimental Ablation Matrix

To achieve complete mechanistic closure without unnecessary compute waste, the experiments are structured into **zero-retraining inference interventions** on the official `vincentleebang/LVR-7B` checkpoint (**Axes 1A, 3, 4, 5**) and **controlled training counterfactuals** (**Axes 1B, 2**).

---

### Axis 1A: Inference-Time Causal Interventions on Prefix Copying, Exposure Bias & M-RoPE (Zero Retraining on `vincentleebang/LVR-7B`)

This axis tests $H_{1a}$, $H_{1b}$, and $H_{1c}$ directly on the released `vincentleebang/LVR-7B` weights by intervening on the forward attention mask, input embeddings, and M-RoPE position IDs during [`_lvr_deocding_by_steps`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L891-L1005) ($K=8$ default):

| Condition ID | Intervention Category | Exact Forward-Pass Modification in `qwen_lvr_model.py` | Target Hypothesis & Mechanistic Purpose |
| :--- | :--- | :--- | :--- |
| **`A1A-BASE`** | Unmodified Default | Standard `vincentleebang/LVR-7B` inference ($K=8$, 1D M-RoPE on LVR tokens, full causal attention to prefix `<\|image_pad\|>` tokens). | Reference baseline (`81.68%` $V^*$, `68.33%` MMVP, `64.13%` BLINK-5). |
| **`A1A-MASK-ALL-IMG`** | Prefix Knockout (All Vision) | During the $K$ latent steps ($t \in [t_{\mathrm{start}}, t_{\mathrm{start}}+K-1]$), set attention mask $M_{t, p_j} = -\infty$ for **all** prefix visual tokens $p_1, \dots, p_N$ (`<\|image_pad\|>` `151655`). Normal vision attention is restored at `<\|lvr_end\|>` and `<answer>`. | Tests whether latent steps must attend to prefix image patches to function, or if `<answer>` generation reads directly from prefix `<\|image_pad\|>` tokens anyway. |
| **`A1A-MASK-ANS-TO-LVR`** | Latent Readout Knockout | Allow normal latent rollout for $K$ steps, but during subsequent text generation ($t \ge t_{\mathrm{end}}$), set attention mask $M_{t, s_k} = -\infty$ for all latent positions $s_1, \dots, s_K$. | **Direct Causal Readout Test**: If accuracy does not drop when `<answer>` tokens are forbidden from attending to the $K$ latent states, the latent states have **zero causal effect** on the final answer! |
| **`A1A-MASK-ANS-TO-IMG`** | Prefix Vision Knockout at Answer | Allow latent states $s_1, \dots, s_K$ to attend to prefix `<\|image_pad\|>` tokens $p_1, \dots, p_N$, but mask $M_{t, p_j} = -\infty$ for $t \ge t_{\mathrm{end}}$ so the answer tokens can **only** access visual information through the $K$ latent states. | Measures the true **Information Bottleneck Capacity** of the $K$ latent vectors when direct visual prefix bypass is blocked. |
| **`A1A-ORACLE-ROI`** | Exposure Bias Elimination ($H_{1b}$) | On $V^*$ / Visual-CoT items with ground-truth ROI boxes $\mathbf{I} = (I_1, \dots, I_{T_v})$, replace autoregressive hidden-state feedback $\hat{\mathbf{h}}_{k-1}^{(L)}$ at inference with teacher-forced ground-truth ViT ROI embeddings $\mathbf{v}_{I_k} = \mathbf{V}_T[I_k]$ (matching SFT training exactly). | Isolates **Teacher-Forcing Exposure Bias ($H_{1b}$)** + missing test-time bounding-box conditioning: $\Delta_{\mathrm{exposure}} = \mathrm{Acc}(\texttt{A1A-ORACLE-ROI}) - \mathrm{Acc}(\texttt{A1A-BASE})$. |
| **`A1A-ORACLE-BKG`** | Wrong-Region Teacher Forcing | Same as `A1A-ORACLE-ROI`, but inject $K$ ViT patch embeddings sampled from **outside** the ground-truth ROI ($\mathbf{I}_{\mathrm{bkg}} \subseteq \{1,\dots,N\} \setminus \mathbf{I}$). | Tests whether the model is sensitive to the *spatial relevance* of injected ViT patches at test time. |
| **`A1A-2D-MROPE`** | 2D M-RoPE Restoration ($H_{1c}$) | For each latent step $k$, assign the 2D M-RoPE spatial coordinates $(t, y_{\hat{I}_k}, x_{\hat{I}_k})$ of the highest-similarity prefix patch $\hat{I}_k = \arg\max_j \cos(\hat{\mathbf{h}}_{k-1}^{(L)}, \mathbf{v}_j^{\mathrm{img}})$ instead of 1D text coordinates $(t,t,t)$. | Tests whether restoring 2D spatial M-RoPE alignment during latent rollout improves spatial reasoning ($V^*_{R.P.}$ and BLINK `Spatial_Relation`). |

---

### Axis 1B: Stage 1 SFT Supervision Counterfactuals (Controlled SFT Training on Visual-CoT)

To test whether Stage 1 SFT requires ground-truth ROI ViT targets $\mathbf{v}_k = \mathbf{V}_T[I_k]$ or merely benefits from Visual-CoT dataset adaptation and auxiliary regularization, we train controlled Stage 1 SFT variants on the exact Visual-CoT split ([finetune_lvr_stage1_7b.sh](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage1_7b.sh), `2500` steps, `lr=1e-5`, `freeze_vision_tower=True`, `freeze_merger=True`):

| Condition ID | SFT Input Overwrite $\mathbf{X}_{\mathrm{embed}}[s_k]$ | Auxiliary Loss $\mathcal{L}_{\mathrm{LVR}}$ Target | Loss Scaling / Geometry | Mechanistic Hypothesis Tested |
| :--- | :--- | :--- | :--- | :--- |
| **`A1B-ORIG`** | $\mathbf{v}_k = \mathbf{V}_T[I_k]$ (Teacher-forced ROI) | $\mathbf{v}_k = \mathbf{V}_T[I_k]$ | $\lambda=0.1$, `MSELoss(mean)` ($\frac{1}{T_v D}$) | Official Stage 1 SFT reproduction baseline. |
| **`A1B-NO-MSE`** | Discrete `<\|lvr\|>` token embedding (`151666`) | None ($\lambda_{\mathrm{LVR}} = 0$) | $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{NTP}}$ only | **Dataset vs. Architecture Control**: Tests if fine-tuning on Visual-CoT QA pairs with $T_v$ discrete `<\|lvr\|>` pause tokens achieves the same gain without any ViT feature regression. |
| **`A1B-VANILLA-SFT`** | No `<\|lvr\|>` tokens (`Direct Answer` SFT) | None ($\lambda_{\mathrm{LVR}} = 0$) | $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{NTP}}$ only | **The Missing 7B Baseline**: Matches the `Vanilla SFT` baseline that beat LVR-3B (`75.92%` vs. `65.45%` on $V^*$) in the commented-out rows of [sft_rl.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex#L11)! |
| **`A1B-PERM-ROI`** | $\mathbf{v}_{\pi(k)}$ (Random permutation $\pi$ of ROI patches) | $\mathbf{v}_{\pi(k)}$ | $\lambda=0.1$, `MSELoss(mean)` | Destroys 2D raster order within the bounding box while preserving the set of ROI features ($H_{1c}$). |
| **`A1B-BKG-PATCH`** | $\mathbf{v}_k^{\mathrm{bkg}} = \mathbf{V}_T[J_k]$ ($J_k \notin \mathbf{I}$, non-ROI background) | $\mathbf{v}_k^{\mathrm{bkg}}$ | $\lambda=0.1$, `MSELoss(mean)` | **Spatial Grounding Falsification**: Supervises latent tokens to copy *irrelevant background patches* instead of the annotated bounding box. |
| **`A1B-NOISE`** | $\boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma_V^2 \mathbf{I}_D)$ (Fixed random seed per sample) | $\boldsymbol{\epsilon}_k$ | $\lambda=0.1$, `MSELoss(mean)` | Tests whether $\mathcal{L}_{\mathrm{LVR}}$ acts purely as a hidden-state norm regularizer devoid of visual semantics. |
| **`A1B-TRUE-COCONUT`** | $\mathbf{h}_{s_k - 1}^{(L)}$ (Sequential unrolled hidden state, $K=4$) | $\mathbf{v}_k = \mathbf{V}_T[I_k]$ | $\lambda=0.1$, `MSELoss(mean)` | **Eliminating Teacher-Forcing Exposure Bias ($H_{1b}$)**: Unrolls the model's own predicted hidden states during SFT so training matches inference recurrence. |
| **`A1B-COSINE-UNNORM`** | $\mathbf{v}_k = \mathbf{V}_T[I_k]$ | $\mathbf{v}_k = \mathbf{V}_T[I_k]$ | `lvr_loss_fct="cosine"` or $\frac{1}{T_v}$ sum-over-$D$ MSE | Tests whether the $1/D$ ($1/3584$) dilution in `MSELoss(reduction='mean')` under-weighed visual regression relative to directional cosine loss ([monkey_patch_forward_lvr.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L108-L114)). |

---

### Axis 2: Stage 2 GRPO "Ghost RL" Falsification & 3B Baseline Recovery (`Qwen2.5-VL-3B`)

Because Stage 2 RL is only implemented and reported in the paper for the 3B model ([finetune_lvr_stage2_3b.sh](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh), [sft_rl.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/sft_rl.tex)), Axis 2 executes on the 3B pipeline (and optionally a matched 7B Stage 2 run) to test $H_2$:

| Condition ID | `_replay_lvr_to_collect_states` & `score_with_lvr_replay` | `lvr_mask` on $[t_{\mathrm{start}}, t_{\mathrm{end}})$ | Reward & Resolution Config | Mechanistic Hypothesis Tested |
| :--- | :--- | :--- | :--- | :--- |
| **`A2-BASE-3B`** | N/A (Zero-shot `Qwen2.5-VL-3B-Instruct`) | N/A | Max pixels $5120 \times 28^2$ | **Restoring the Commented-Out Baseline**: Paper's hidden `sft_rl.tex` row scored `74.87%` on $V^*$ (beating LVR-3B `65.45%` by $+9.42\%$). |
| **`A2-VANILLA-SFT-3B`** | N/A (Vanilla SFT on Visual-CoT without LVR tokens) | N/A | Max pixels $5120 \times 28^2$ | **Restoring the Commented-Out SFT Baseline**: Paper's hidden `sft_rl.tex` row scored `75.92%` on $V^*$ (beating $\text{LVR}_{\text{RL}}$-3B `67.02%` by $+8.90\%$). |
| **`A2-GHOST-DEFAULT`** | `@torch.no_grad()` + `.detach()` (Official Code) | `lvr_mask = False` on $[t_{\mathrm{start}}, t_{\mathrm{end}})$ | $R_{\mathrm{acc}} + 0.5 R_{\mathrm{fmt}}$ (`^<\|lvr_start\|>...`), $2560 \times 28^2$ px | Official Stage 2 GRPO implementation ([grpo_trainer.py](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L746-L752)): zero gradient through continuous latent states $\mathbf{H}_{\mathrm{LVR}}$. |
| **`A2-STRICT-MASK-END`** | `@torch.no_grad()` + `.detach()` | `lvr_mask = False` on closed $[t_{\mathrm{start}}, t_{\mathrm{end}}]$ | Same as `A2-GHOST-DEFAULT` | Fixes the 1-token half-open shift bug in [grpo_trainer.py](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L746-L752) so `<\|lvr_end\|>` is also masked. Tests if `A2-GHOST-DEFAULT` is purely optimizing text answer formatting. |
| **`A2-DIFF-REPLAY`** | **Remove `@torch.no_grad()` and `.detach()`**: unroll [`_replay_lvr_to_collect_states`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L1482) with `torch.enable_grad()` so $\frac{\partial \log \pi_\theta(\text{answer})}{\partial \mathbf{H}_{\mathrm{LVR}}} \frac{\partial \mathbf{H}_{\mathrm{LVR}}}{\partial \theta_{\mathrm{LLM}}} \neq \mathbf{0}$ | `lvr_mask = False` on $[t_{\mathrm{start}}, t_{\mathrm{end}})$ (supervise via answer log-probs backpropagating through $\mathbf{H}_{\mathrm{LVR}}$) | Same as `A2-GHOST-DEFAULT` | **True Differentiable Latent RL**: Enables end-to-end policy gradient backpropagation from answer tokens through the recurrent continuous latent states $\mathbf{H}_{\mathrm{LVR}}$ into $\theta_{\mathrm{LLM}}$. |
| **`A2-UNMASK-LOGP`** | `@torch.no_grad()` + `.detach()` | **`lvr_mask = True` everywhere** (unmask $[t_{\mathrm{start}}, t_{\mathrm{end}})$) | Same as `A2-GHOST-DEFAULT` | Tests whether unmasking discrete surrogate token log-probabilities during latent steps destabilizes training (exploding KL / entropy collapse). |
| **`A2-FMT-ONLY`** | `@torch.no_grad()` + `.detach()` | `lvr_mask = False` on $[t_{\mathrm{start}}, t_{\mathrm{end}})$ | **$R_{\mathrm{acc}} = 0$, $R = R_{\mathrm{fmt}}$ only** | **Pure Format-Polishing Test**: Tests if the $+1.57\%$ gain from Stage 1 (`65.45%`) to Stage 2 (`67.02%`) on $V^*$ comes entirely from `format_reward` enforcing `<answer>X</answer>` syntax for `evaluation.py`. |
| **`A2-INTERLEAVED-REGEX`** | `@torch.no_grad()` + `.detach()` | `lvr_mask = False` on $[t_{\mathrm{start}}, t_{\mathrm{end}})$ | Replace anchored `^<\|lvr_start\|>` in [reward_funcs.py](file:///home/devraj/MTok/external_models/lvr/repo/src/train/reward_funcs.py#L51) with `r"(?:<\|lvr_start\|>.*?<\|lvr_end\|>.*?)+<answer>.*?</answer>"` + $5120 \times 28^2$ px | Tests whether allowing free-form text CoT before/around latent blocks and restoring full $5120 \times 28^2$ image resolution recovers the $-8.90\%$ gap vs. Vanilla SFT. |

---

### Axis 3: Inference Step Quota Scaling & Termination Strategy Ablation (`vincentleebang/LVR-7B`)

This axis tests $H_3$ on `vincentleebang/LVR-7B` by sweeping the step budget $K$ around the paper's `STEP_LIST = [4, 8, 16]` ([evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L27)) and directly comparing all three decoding implementations in [qwen_lvr_model.py](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py):

#### 3A. Step Quota Scaling Sweep (`decoding_strategy="steps"`, [`_lvr_deocding_by_steps`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L891))

We evaluate `vincentleebang/LVR-7B` across 8 step budgets:
$$K \in \{0, 1, 2, 4, 8, 16, 32, 64\}$$
- **$K = 0$ (Zero-Step Immediate Exit)**: Set `lvr_steps = 0` so `lvr_mode_switch` is immediately `False` upon encountering `<|lvr_start|>` (`151665`), or prefill `<|lvr_start|><|lvr_end|>`.
- **$K \in \{1, 2\}$ (Sub-Paper Minimal Quotas)**: Tests whether a single continuous vector ($K=1$) captures 100% of the accuracy of $K=16$ (as observed on our 5-sample smoke test in [LVR_EMPIRICAL_REPORT.md](file:///home/devraj/MTok/external_models/lvr/LVR_EMPIRICAL_REPORT.md)).
- **$K \in \{4, 8, 16\}$ (Official Paper Quotas)**: Exact replication of [main_table.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/main_table.tex#L78-L80) (`Steps=4`, `Steps=8`, `Steps=16`).
- **$K \in \{32, 64\}$ (Over-Horizon Rollout)**: Tests whether unrolling beyond the typical SFT bounding-box patch count induces hidden-state norm explosion, fixed-point attractor collapse ($\cos(\mathbf{h}_k, \mathbf{h}_{k+1}) \to 1$), or failure to emit `<|lvr_end|>` (`151668`) upon quota expiration.

#### 3B. Decoding Strategy & Termination Mechanism Comparison

| Condition ID | Decoding Function in `qwen_lvr_model.py` | Termination Rule | Expected Mechanistic Behavior |
| :--- | :--- | :--- | :--- |
| **`A3B-STEPS-8`** | [`_lvr_deocding_by_steps`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L891) | Hardcoded counter $K=8$; on step $9$, `lvr_mode_switch=False` and `lm_head` emits `<\|lvr_end\|>` (`151668`). | Reference (`80.10%` $V^*$ at $K=8$, `81.68%` best across $K \in \{4,8,16\}$). |
| **`A3B-FORCED-END`** | Modified `_lvr_deocding_by_steps` | Explicitly inject token ID `151668` (`<\|lvr_end\|>`) when `lvr_Remaining_steps == 0` instead of relying on `lm_head` argmax. | Tests how often `lm_head` fails to emit `<\|lvr_end\|>` at large $K \in \{32, 64\}$. |
| **`A3B-VANILLA-LOGIT`** | [`_lvr_deocding`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L395) | Autonomous exit when $\arg\max_v \mathrm{logits}_t = 151668$ (`<\|lvr_end\|>`). | Because `<\|lvr\|>` (`151666`) was masked with `-100` in SFT, `<\|lvr_end\|>` is predicted immediately after 1 step ($K_{\mathrm{actual}} = 1$). |
| **`A3B-LATENT-END`** | [`_lvr_deocding_with_latentend`](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L619) | Exit when $\|\mathbf{h}_t^{(L)} - \mathbf{e}_{151667}\|_2 < \tau$ (`lvr_latent_end_emb`). | Since `vincentleebang/LVR-7B` was not trained with `loss_latent_end`, `lvr_latent_end_emb` is random initialization (norm `0.795` vs. $\|\mathbf{h}_t\|_2 \approx 123.5$), running until `max_new_tokens` or collapsing (`39.8%` in [ablation.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/ablation.tex#L13)). |

---

### Axis 4: Latent Representation Substrate & Format-Locking Matrix (The 9-Condition E1 Suite on `vincentleebang/LVR-7B`)

Directly mirroring our rigorous E1 protocol for Mull-Tokens ([PREREGISTRATION.md](file:///home/devraj/MTok/PREREGISTRATION.md)), we hold `vincentleebang/LVR-7B` weights fixed and manipulate **only the representation injected into `inputs_embeds` during the latent phase** ($K=8$ steps unless noted):

| Condition ID | Latent Phase Substrate ($k = 1, \dots, K$) | Exact Mathematical Definition of $\mathbf{X}_{\mathrm{embed}}[t_{\mathrm{start}}+k]$ | Hypothesis Tested |
| :--- | :--- | :--- | :--- |
| **`C1-CONT-DEFAULT`** | Autoregressive Continuous Hidden State | $\mathbf{h}_{t_{\mathrm{start}}+k-1}^{(L)}$ (Output of `self.model.norm` at Layer 28) | Official LVR continuous recurrence baseline. |
| **`C2-EMPTY-TAG`** | Empty Delimiter Prefill ($K=0$) | Force generation prefix `<\|lvr_start\|><\|lvr_end\|>` (`[151665, 151668]`) with **zero** latent steps | **Pure Syntactic Format-Unlocking ($H_4$)**: Satisfies the `<\|lvr_start\|>...<\|lvr_end\|>` SFT prefix gate with zero continuous computation. |
| **`C3-DISCRETE-LVR`** | Discrete `<\|lvr\|>` Token Embedding | $\mathbf{E}_{\mathrm{tok}}[151666]$ (Fixed vocabulary embedding of `<\|lvr\|>` repeated $K=8$ times, bypassing hidden-state overwrite) | **Discrete Pause-Token Control**: Tests if 8 identical discrete `<\|lvr\|>` token embeddings match continuous hidden-state feedback. |
| **`C4-MEAN-IMG-POOL`** | Mean Prefix Image Embedding | $\bar{\mathbf{v}} = \frac{1}{N}\sum_{j=1}^N \mathbf{v}_j^{\mathrm{img}}$ repeated for $K=8$ steps | Tests if feeding a static global average of the prefix image patches matches dynamic step-by-step recurrence. |
| **`C5-NN-QUANT-IMG`** | Nearest-Neighbor Prefix Patch Quantization | $\mathbf{v}_{\hat{j}_k}^{\mathrm{img}}$ where $\hat{j}_k = \arg\min_{j \in \{1,\dots,N\}} \|\mathbf{h}_{t_{\mathrm{start}}+k-1}^{(L)} - \mathbf{v}_j^{\mathrm{img}}\|_2$ | **Prefix Copy Discretization Test ($H_{1a}$)**: Snaps every continuous state to its nearest existing prefix ViT patch vector. |
| **`C6-ZERO-VEC`** | Zero Vector Ablation | $\mathbf{0} \in \mathbb{R}^{3584}$ for $K=8$ steps | Tests whether only the positional shift of $K=8$ dummy steps matters. |
| **`C7-GAUSS-NOISE`** | Norm-Matched Isotropic Gaussian Noise | $\boldsymbol{\epsilon}_k \sim \mathcal{N}\!\left(\mathbf{0}, \frac{\|\mathbf{h}_{t_{\mathrm{start}}+k-1}^{(L)}\|_2^2}{D}\mathbf{I}_D\right)$ (3 seeds) | Destroys all directional semantics while preserving exact L2 norm (~$123.5$) and sequence length ($K=8$). |
| **`C8-SHUFFLE-BATCH`** | Cross-Sample Latent Transplant | Inject cached latent trajectory $\mathbf{H}_{\mathrm{LVR}}^{(b')}$ from a *different* benchmark image $b' \neq b$ | **Image-Specificity Falsification**: If accuracy does not drop when receiving latent vectors from the wrong image, the answer decoder ignores latent visual content! |
| **`C9-DIRECT-ANS`** | Bypass `<\|lvr_start\|>` Entirely | Prefill `<answer>` immediately after user prompt (forbid `151665`) | Measures the syntactic format-locking penalty when `<\|lvr_start\|>` is omitted. |

---

### Axis 5: Evaluation Harness, Parser & Prompt Confound Audit ($2 \times 2 \times 2$ Factorial)

To test $H_5$ and guarantee that reported comparisons between Base `Qwen2.5-VL-7B/3B`, Vanilla SFT, and LVR are not artifacts of [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py), we evaluate all models across a $2 \times 2 \times 2$ factorial evaluation grid:

1. **Factor A — Answer Extraction Parser**:
   - **A1 (`Parser-FirstChar`, Official Repo Bug)**: [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L52-L61) (`str.split('<answer>')[-1]`, `split(' ')[0][0] == ground_truth`).
   - **A2 (`Parser-Robust-Regex`)**: Cascaded extractor that checks (i) `<answer>\s*\(?([A-F])\)?\s*</answer>`, (ii) explicit choice patterns (`(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-F])\)?`), (iii) standalone leading/trailing option letters, and (iv) exact option-text substring fallback (`lmms-eval` standard).
2. **Factor B — Prompt Suffix & Option Formatting**:
   - **B1 (`Prompt-LVR-Eval`)**: Appends `"\nAnswer with the option's letter from the given choices directly."` and rewrites MMVP `(a)/(b)` to `A./B.` ([evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L70), [L262-L263](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L262-L263)).
   - **B2 (`Prompt-Training-Aligned`)**: Appends the Stage 1 SFT instruction `" Please provide the bounding box coordinate of the region that can help you answer the question better."` ([STEP1_INGESTION_MANIFEST.md](file:///home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md#L359)) vs. native benchmark prompt without mutation.
3. **Factor C — Circular Option Permutation (`CircularEval`)**:
   - **C1 (`Canonical-Order`)**: Single pass with original option ordering.
   - **C2 (`Circular-Permuted`)**: Cyclic shift of option labels $(A, B, C, D) \to (B, C, D, A)$ across all $M$ choices; an item is scored correct under `CircularEval` iff answered correctly across all cyclic shifts, eliminating position-bias inflation.

---

## 4. Ground-Truth Benchmark Protocol & Statistical Rigor

### 4.1 Exact LVR Evaluation Benchmarks ([main_table.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/main_table.tex), [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py))

Every ablation condition across Axes 1–5 is evaluated on the exact three benchmarks and subtask splits used in the LVR paper:

| Benchmark | HuggingFace / Source ID | Total $N$ | Subtasks / Splits & Exact Sample Counts | Paper Reported `Qwen2.5-VL-7B` | Paper Reported `LVR-7B` ($K=4 / 8 / 16$ / Best) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **$V^*$ Bench** | `craigwu/vstar_bench` (`test` split) | **191** | • **Direct Attributes ($V^*_{D.A.}$)**: `question_id <= 114` ($N=115$)<br>• **Relative Position ($V^*_{R.P.}$)**: `question_id > 114` ($N=76$) | `76.44%` (`74.78%` D.A., `78.95%` R.P.) | `79.58%` ($K=4$) / `80.10%` ($K=8$) / `79.58%` ($K=16$) / **`81.68%`** (Best) |
| **MMVP** | `MMVP` (`Questions.csv` + `MMVP_Images`) | **300** | 150 paired fine-grained visual contrast instances ($N=300$ questions) | `66.33%` | `66.00%` ($K=4$) / `67.33%` ($K=8$) / `66.67%` ($K=16$) / **`68.33%`** (Best) |
| **BLINK (Core 5)** | `BLINK-Benchmark/BLINK` (`val` split) | **697** | • `Counting`: $N=120$<br>• `IQ_Test`: $N=150$<br>• `Jigsaw`: $N=150$<br>• `Relative_Reflectance`: $N=134$<br>• `Spatial_Relation`: $N=143$ | `62.21%` (Unweighted Macro Avg of 5 subtasks) | `62.94%` ($K=4$) / `63.61%` ($K=8$) / `62.45%` ($K=16$) / **`64.13%`** (Best) |
| **BLINK (Extended 6)** | `BLINK-Benchmark/BLINK` (`val` split, in [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L340)) | **859** | `Multi-view_Reasoning` ($133$), `Object_Localization` ($122$), `Relative_Depth` ($124$), `Semantic_Correspondence` ($139$), `Visual_Correspondence` ($172$), `Visual_Similarity` ($169$) | Unreported in paper (audited for selective subtask reporting) | Unreported in paper (audited to test if LVR degrades on non-selected BLINK splits) |

> **Critical Reporting Anomaly Note ("Best-of-$K$ Oracle Selection")**: In [main_table.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/main_table.tex#L59), the headline `LVR-7B` numbers (`81.68%` $V^*$, `68.33%` MMVP, `64.13%` BLINK) are **strictly higher** than every fixed-step row (`Steps=4`, `Steps=8`, `Steps=16` in [main_table.tex](file:///home/devraj/MTok/external_models/lvr/paper_src/tables/main_table.tex#L78-L80))! Our evaluation protocol explicitly separates **Fixed-$K$ Accuracy** from **Post-Hoc Per-Subtask / Per-Item Oracle $\max_{K \in \{4,8,16\}}$ Selection** to quantify exactly how much of the headline gain is post-hoc step cherry-picking.

---

### 4.2 Deterministic Inference Hyperparameters

To eliminate sampling variance during inference ablations (Axes 1A, 3, 4, 5), all evaluations use deterministic greedy decoding matching [evaluation.py](file:///home/devraj/MTok/external_models/lvr/repo/evaluation/evaluation.py#L140-L157):
- `do_sample = False`, `temperature = 0.0` (greedy argmax)
- `max_new_tokens = 2048` (`max_tokens = 2048` in `evaluation.py`)
- `attn_implementation = "flash_attention_2"` (or `"sdpa"` when extracting explicit $L \times H_Q \times T \times T$ attention weight tensors in Axis 1A)
- `torch_dtype = torch.bfloat16` (with `float32` upcast inside `MSELoss` during Stage 1 SFT)

---

### 4.3 Paired Statistical Testing & Discordance Protocol

Because $V^*$ has only $N=191$ items ($1\,\text{item} = 0.524\%$) and MMVP has $N=300$ items ($1\,\text{item} = 0.333\%$), raw percentage differences of $1\%\text{–}2\%$ represent only **2 to 6 items**! Every comparison between two conditions $A$ and $B$ must report:
1. **Contingency Table & Discordant Pairs $(n_{00}, n_{01}, n_{10}, n_{11})$**:
   - $b = n_{10}$: items where condition $A$ is correct and condition $B$ is wrong.
   - $c = n_{01}$: items where condition $A$ is wrong and condition $B$ is correct.
2. **Paired McNemar's Test**:
   - For $b + c \ge 25$, compute McNemar's chi-squared statistic with Edwards' continuity correction:
     $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad p = 1 - F_{\chi^2_1}(\chi^2)$$
   - For $b + c < 25$, compute the **Exact Two-Sided Binomial Test** $p$-value under $\mathrm{Binomial}(b+c, 0.5)$.
3. **Paired Bootstrap 95% Confidence Intervals**:
   - Resample $B = 10,000$ paired item indices with replacement to compute the 95% percentile confidence interval $[\Delta_{2.5\%}, \Delta_{97.5\%}]$ on $\Delta \mathrm{Acc}(A, B) = \mathrm{Acc}(A) - \mathrm{Acc}(B)$.
4. **Multiple-Comparison Control**:
   - Apply **Holm-Bonferroni** step-down correction across all primary hypothesis tests within each experimental axis ($\alpha = 0.05$).

---

## 5. Quantitative Mechanistic Diagnostics

Beyond task accuracy, every run logs five continuous mechanistic telemetry metrics:

1. **Prefix ROI vs. Background Cosine Alignment ($H_{1a}$)**:
   For each latent step $k \in \{1, \dots, K\}$ with normalized hidden state $\mathbf{h}_k^{(L)} \in \mathbb{R}^{3584}$:
   $$\mathrm{CS}_{\mathrm{max}}(\mathbf{h}_k^{(L)}, \mathbf{V}_T) = \max_{j \in \{1,\dots,N\}} \frac{\langle \mathbf{h}_k^{(L)}, \mathbf{v}_j^{\mathrm{img}} \rangle}{\|\mathbf{h}_k^{(L)}\|_2 \|\mathbf{v}_j^{\mathrm{img}}\|_2}, \quad \mathrm{CS}_{\mathrm{ROI}} = \max_{j \in \mathbf{I}} \cos(\mathbf{h}_k^{(L)}, \mathbf{v}_j^{\mathrm{img}}), \quad \mathrm{CS}_{\mathrm{bkg}} = \max_{j \notin \mathbf{I}} \cos(\mathbf{h}_k^{(L)}, \mathbf{v}_j^{\mathrm{img}})$$
   - **ROI Retrieval Recall@m**: Fraction of latent steps $k$ whose top-$m$ ($m \in \{1, 5\}$) nearest prefix ViT patches $\mathbf{v}_j^{\mathrm{img}}$ fall inside the ground-truth bounding box $\mathbf{I}$.
2. **Layer-Wise Attention Mass Allocation ($\alpha_{\mathrm{ROI}}$ vs. $\alpha_{\mathrm{bkg}}$ vs. $\alpha_{\mathrm{LVR}}$)**:
   - During latent generation ($t \in [t_{\mathrm{start}}, t_{\mathrm{start}}+K-1]$): proportion of attention weight directed to ground-truth ROI prefix patches ($\sum_{j \in \mathbf{I}} A_{l,h,t,p_j}$) vs. non-ROI background patches ($\sum_{j \notin \mathbf{I}} A_{l,h,t,p_j}$).
   - During answer generation ($t \ge t_{\mathrm{end}}$): proportion of attention weight directed to the $K$ latent states ($\sum_{k=1}^K A_{l,h,t,s_k}$) vs. directly to the prefix image patches ($\sum_{j=1}^N A_{l,h,t,p_j}$).
3. **Teacher-Forcing vs. Autoregressive Rollout Drift ($H_{1b}$)**:
   $$\delta_k = \left\| \hat{\mathbf{h}}_k^{\mathrm{free},(L)} - \mathbf{h}_k^{\mathrm{TF},(L)} \right\|_2 \quad \text{and} \quad \cos\!\left(\hat{\mathbf{h}}_k^{\mathrm{free},(L)}, \mathbf{v}_{I_k}\right) \text{ vs. } \cos\!\left(\mathbf{h}_k^{\mathrm{TF},(L)}, \mathbf{v}_{I_k}\right)$$
   measuring how rapidly free-running inference states diverge from teacher-forced SFT states as $k$ increases from $1$ to $K$.
4. **Latent Trajectory Effective Rank & Stationary Attractor Convergence ($H_3$)**:
   Given the centered latent trajectory matrix $\tilde{\mathbf{H}} \in \mathbb{R}^{K \times D}$ with singular values $\sigma_1 \ge \dots \ge \sigma_K$ and $p_i = \frac{\sigma_i}{\sum_j \sigma_j}$:
   $$\mathrm{erank}(\tilde{\mathbf{H}}) = \exp\!\left(-\sum_{i=1}^K p_i \ln p_i\right), \quad \rho_{\mathrm{adj}}(k) = \cos\!\left(\mathbf{h}_k^{(L)}, \mathbf{h}_{k+1}^{(L)}\right)$$
   If $\mathrm{erank}(\tilde{\mathbf{H}}) \ll K$ and $\rho_{\mathrm{adj}}(k) \to 1$, the multi-step rollout collapses onto a low-dimensional fixed point.
5. **Stage 2 RL Latent-Path Gradient Norm ($H_2$)**:
   $$\mathcal{G}_{\mathrm{latent}} = \left\| \frac{\partial \mathcal{L}_{\mathrm{GRPO}}}{\partial \mathbf{H}_{\mathrm{LVR}}} \right\|_F \quad \text{and} \quad \mathcal{G}_{\theta,\mathrm{latent}} = \left\| \frac{\partial \mathcal{L}_{\mathrm{GRPO}}}{\partial \mathbf{H}_{\mathrm{LVR}}} \cdot \frac{\partial \mathbf{H}_{\mathrm{LVR}}}{\partial \theta_{\mathrm{LLM}}} \right\|_2$$
   Under default `A2-GHOST-DEFAULT`, $\mathcal{G}_{\theta,\mathrm{latent}} \equiv 0.0$ identically by mathematical construction; under `A2-DIFF-REPLAY`, $\mathcal{G}_{\theta,\mathrm{latent}} > 0$ quantifies the true policy gradient signal flowing through continuous visual thoughts.

---

## 6. Pre-Registered Adjudication & Falsification Matrix

Table 6 establishes the strict, pre-registered pass/fail criteria that adjudicate each claim of the LVR framework:

| Hypothesis | Primary Experimental Comparison | Metric & Statistical Test | Outcome Supporting LVR Paper Claim | Outcome Falsifying LVR Claim (Confirming Forensic Critique) |
| :--- | :--- | :--- | :--- | :--- |
| **$H_{1a}$: Prefix Identity Copying** | `A1A-MASK-ANS-TO-LVR` vs. `A1A-BASE` & `A1B-NO-MSE` vs. `A1B-ORIG` | $\Delta \mathrm{Acc}$ on $V^*$, MMVP, BLINK (McNemar $p < 0.05$) + $\mathrm{CS}_{\mathrm{ROI}}$ | Masking answer-to-latent attention drops accuracy by $\ge 3.0\%$ ($p < 0.01$); `A1B-ORIG` beats `A1B-NO-MSE` and `A1B-BKG-PATCH` by $\ge 2.5\%$. | Masking answer-to-latent attention (`A1A-MASK-ANS-TO-LVR`) causes $< 1.0\%$ drop ($p > 0.10$), or `A1B-NO-MSE` / `A1B-BKG-PATCH` matches `A1B-ORIG` within $1.5\%$. |
| **$H_{1b}$: Teacher-Forcing Exposure Bias** | `A1A-ORACLE-ROI` vs. `A1A-BASE` & `A1B-TRUE-COCONUT` vs. `A1B-ORIG` | Rollout drift $\delta_k$ & $\Delta \mathrm{Acc}$ on $V^*$ | Free-running rollout $\hat{\mathbf{h}}_k$ maintains high ROI alignment ($\mathrm{CS}_{\mathrm{ROI}} > 0.6$) across $k=1\dots 8$ without needing oracle box injection. | Free-running $\mathrm{CS}_{\mathrm{ROI}}$ decays sharply after $k=1$, or `A1A-ORACLE-ROI` reveals a large exposure-bias gap obscured at test time. |
| **$H_{1c}$: 1D M-RoPE Distortion** | `A1A-2D-MROPE` & `A1B-PERM-ROI` vs. `A1B-ORIG` | $\Delta \mathrm{Acc}$ on $V^*_{R.P.}$ ($N=76$) & BLINK `Spatial_Relation` ($N=143$) | `A1B-ORIG` significantly outperforms spatially permuted `A1B-PERM-ROI` ($p < 0.05$). | `A1B-PERM-ROI` matches `A1B-ORIG` ($|\Delta| < 1.0\%$), proving 1D M-RoPE $(t,t,t)$ already destroyed 2D patch grid geometry. |
| **$H_2$: "Ghost RL" Disconnection** | `A2-GHOST-DEFAULT` vs. `A2-FMT-ONLY`, `A2-DIFF-REPLAY`, and `A2-VANILLA-SFT-3B` | $\mathcal{G}_{\theta,\mathrm{latent}}$ & $\Delta \mathrm{Acc}$ on $V^*$, MMVP, BLINK | `A2-GHOST-DEFAULT` significantly outperforms `A2-FMT-ONLY` and `A2-VANILLA-SFT-3B`. | $\mathcal{G}_{\theta,\mathrm{latent}} \equiv 0$; `A2-FMT-ONLY` matches `A2-GHOST-DEFAULT` within $1.0\%$; and `A2-VANILLA-SFT-3B` (`75.92%`) beats $\text{LVR}_{\text{RL}}$-3B (`67.02%`) by $> 7\%$ ($p < 0.001$). |
| **$H_3$: Step Quota Invariance** | `A3A` sweep $K \in \{0, 1, 2, 4, 8, 16, 32, 64\}$ | Slope of $\mathrm{Acc}(K)$ for $K \ge 1$, McNemar $K=16$ vs. $K=1$, & $\mathrm{erank}(\tilde{\mathbf{H}})$ | Monotonic, statistically significant accuracy gain from $K=1$ to $K=16$ ($\Delta \ge +2.5\%, p < 0.05$) with $\mathrm{erank}(\tilde{\mathbf{H}}) \propto K$. | Flat scaling curve ($|\mathrm{Acc}(K=16) - \mathrm{Acc}(K=1)| < 1.0\%, p > 0.20$), $\rho_{\mathrm{adj}}(k) > 0.75$, and paper's headline `81.68%` is reproduced only by post-hoc best-of-$\{4,8,16\}$ selection. |
| **$H_4$: Format-Locking vs. Semantics** | `C1-CONT-DEFAULT` vs. `C2-EMPTY-TAG`, `C3-DISCRETE-LVR`, `C7-GAUSS-NOISE`, `C8-SHUFFLE-BATCH` | Paired McNemar $p$-value & 95% bootstrap CI on $V^*$, MMVP, BLINK | `C1-CONT-DEFAULT` significantly beats `C2-EMPTY-TAG`, `C3-DISCRETE-LVR`, `C7-GAUSS-NOISE`, and `C8-SHUFFLE-BATCH` ($\Delta \ge +2.5\%, p < 0.05$). | `C2-EMPTY-TAG` (`<\|lvr_start\|><\|lvr_end\|>`), `C3-DISCRETE-LVR`, or `C8-SHUFFLE-BATCH` (wrong-image latents) matches `C1-CONT-DEFAULT` within statistical noise ($p > 0.10$). |
| **$H_5$: Parser & Prompt Confound** | `Parser-FirstChar` (`A1`) vs. `Parser-Robust-Regex` (`A2`) across Base 7B/3B vs. LVR 7B/3B | Relative gain $\Delta_{\mathrm{LVR - Base}}$ under `A1` vs. `A2` | $\Delta_{\mathrm{LVR - Base}}$ remains statistically significant ($p < 0.05$) under `Parser-Robust-Regex` (`A2`) and `CircularEval` (`C2`). | Base `Qwen2.5-VL-7B/3B` accuracy jumps significantly under `Parser-Robust-Regex` (`A2`), shrinking $\Delta_{\mathrm{LVR - Base}}$ into statistical noise ($p > 0.10$). |
