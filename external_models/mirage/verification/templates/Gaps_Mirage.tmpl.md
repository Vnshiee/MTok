# Mirage — Verified Gaps

**Scope:** arXiv 2506.17218; `repo/` @ `53f26de2682025e146781c5b198ec93bdfe4c4d6` (with the vendored transformers fork); checkpoint `Miiche/vsp_spatial_planning_direct_sft` @ `d1b6fdce`.
**Companion documents:** `STEP1_INGESTION_MANIFEST.md` (inventory), `CODE_VS_PAPER_TRACE.md` (line-level trace), `diagnostics/verify_mirage_autograd.py` (29 CPU checks, all passing; IDs **[T…]** below).
**Rule:** every item states what the code does, cites the lines, and says how it was verified. Severity reflects effect on the paper's reported numbers or on reproducing them.

This document replaces the previous `Gaps_Mirage.md`. Its claims are assessed in §3.

---

## 1. Gaps

### G1. Only VSP planning can be evaluated; RL is not released (reproducibility: high)

`repo/src/test.py` scores every task with `simulate_vsp(map_desc, …)`:

{{CODE repo/src/test.py 73 79}}

COMT, BLINK-Jigsaw, SAT and VSP spatial reasoning items have no `map_desc`; `simulate_vsp([])` raises at {{LINE repo/src/task.py 138 "raise ValueError"}}. Tables 1–3 and the 3B table therefore cannot be reproduced with released code except the VSP-planning direct column, which is also the only released checkpoint. The GRPO stage described in the paper (`paper_src/sec/3_framework.tex` L105) has no code in the repository, so the "+ w/ GRPO" rows are not reproducible.

### G2. Reported accuracies are single stochastic draws (statistics: medium)

{{CODE repo/src/test.py 60 68}}

Decoding samples (T = 0.7, top-p 0.9; the checkpoint's `generation_config.json` adds `repetition_penalty: 1.05`) with one seed (`seed_everything(seed=42)`, {{LINE repo/src/test.py 14 "seed_everything(seed=42)"}}). The paper reports no variance across seeds or decoding settings. With 100 maps per VSP level, a binomial 95% interval at p = 0.76 is about ±8.4 points per level and ±4.2 points for the 400-map average, comparable to several reported margins (e.g. Direct SFT 0.72 vs Ours (Direct) 0.76 on VSP planning).

### G3. Test-time latent count is decided by the model, capped at 8; training uses exactly 4 (method/eval: medium)

Training sequences always contain `latent_size` = 4 pads (Stage 2 raises otherwise, {{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2171 "if max_n_latents != self.config.latent_size:"}}). At inference the loop leaves latent mode when the selected token is `<|latent_end|>`, or after 8 latents:

{{CODE repo/transformers/src/transformers/generation/utils.py 3390 3396}}

The CE target at the 4th latent position is `<|latent_end|>` **[T4h]**; earlier latent positions have no CE target, so whether the model exits at 4 is learned only indirectly, and under `repo/src/test.py` sampling the exit is itself random. Neither the paper nor `repo/src/test.py` reports how many latents were used. **[T5a]** reproduces the cap. Effect on accuracy is measured by Axis 3 of the ablation design.

### G4. Latent targets are means of contiguous raster bands, not a 2D pooling (method description: low–medium)

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2109 2120}}

The paper says "average pooling over the original patch features" ({{LINE paper_src/sec/3_framework.tex 52 "simple average pooling over the original patch features"}}). The code averages K contiguous chunks of the row-major merged-token sequence; with K = 4 each target is the mean of one horizontal quarter of the helper image, and up to K − 1 trailing tokens are discarded. Verified exactly **[T3a]**. For the VSP helper image (the start map with a red path) this means each latent target encodes one horizontal band of the map.

### G5. Hyperparameters in the paper disagree with each other and with the released run (reporting: medium)

| Item | Paper | Released |
|---|---|---|
| γ | 0.1 (`paper_src/sec/4_experiments_arxiv.tex` L26) and 10 (`paper_src/sec/7_supp.tex` L275) | 0.1 (`repo/src/trainer.py` L25) |
| Batch size | 8 (`paper_src/sec/4_experiments_arxiv.tex` L25, `paper_src/sec/7_supp.tex` L271) | 1 × 1 GPU, grad-accum 1 (`training_args.bin`) |
| Grad accumulation | 2 (`paper_src/sec/7_supp.tex` L272) | 1 |
| LR schedule | cosine (`paper_src/sec/4_experiments_arxiv.tex` L25) | linear |
| Adam β₂ | 0.95 (`paper_src/sec/7_supp.tex` L273) | 0.999 |
| Epochs | 10 (`paper_src/sec/7_supp.tex` L274) | 15 (README, released run); 10 in `repo/train.sh` |
| VSP SFT size | 1k (`paper_src/sec/4_experiments_arxiv.tex` L14), 3,000 (`paper_src/sec/7_supp.tex` L244) | 1,000 (`train_direct.jsonl`) |

All values are listed with exact quotes in `CODE_VS_PAPER_TRACE.md` §6; the released values come from `training_args.bin` (`STEP1_INGESTION_MANIFEST.md` §6).

### G6. Arithmetic error in the k/γ ablation table (reporting: low)

{{TEX paper_src/sec/4_experiments_arxiv.tex 159 "2 & 0.1 & 0.85 & 0.86 & 0.89 & \textbf{0.93} & 0.86"}}

The average of 0.85, 0.86, 0.89 and 0.93 is 0.8825, not 0.86. The other five rows' averages agree with their level values within rounding (0.8725, 0.8825, 0.745, 0.8425, 0.835 reported as 0.87, 0.88, 0.75, 0.84, 0.83). With 0.88, k = 2 ties k = 6 for best rather than ranking below k = 4.

### G7. Generation-loop state is batch-global and is not reset after a latent segment (implementation: low for released use, high for batched or multi-segment use)

{{CODE repo/transformers/src/transformers/generation/utils.py 3375 3388}}

- `in_latent_mode` is one flag for the whole batch: once any row samples `<|latent_start|>`, every other row's token is replaced by `<|latent_pad|>` and every row is fed hidden states **[T5d]**.
- `latent_end` is never set back to False and `latent_end_num` is never reset, and exit requires `latent_end_num == cur_bs`. After a second `<|latent_start|>` in a sequence, the loop feeds hidden states for every remaining step and a further `<|latent_end|>` does not end latent mode **[T5c]**.

The released evaluation uses batch size 1, and training data has exactly one helper image per answer ({{LINE paper_src/sec/3_framework.tex 37 "\bm o = \bm o_{\rm pre} \oplus I \oplus \bm o_{\rm post}"}}), so the released numbers are unaffected unless the model emits a second `<|latent_start|>`. The ablation runner (`ablation_study/run_mirage_ablation.py`) tracks segments per question and reproduces the fork's single-segment behaviour token-for-token **[T6]**.

### G8. VSP scoring details (evaluation: low)

{{CODE repo/src/task.py 115 120}}

- Any `U/D/R/L` character inside the box is a move, so non-move words inside `\boxed{}` add moves.
- The `"Invalid action"` branch ({{LINE repo/src/task.py 142 "if action not in ACTION_MAP:"}}) is unreachable because the parser emits only keys of `ACTION_MAP`, so the "Invalid" rate logged by `repo/src/test.py` is always 0.
- Off-grid moves are ignored ({{LINE repo/src/task.py 156 "continue"}}) rather than counted as errors.

### G9. Latent-position schedule in Stage 2 assumes all rows share latent positions (implementation: low)

{{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2176 "next_compute_range = (0, latent_indices[:, 1].min().item())"}} starts pass 1 at the batch minimum, and every later pass advances by one position for all rows. This is exact for batch size 1, which is what `repo/src/main.py` uses (`per_device_train_batch_size=1`) and what the released single-GPU run used. **[T4f]** confirms equality with a single causal pass at batch size 1.

### G10. Smaller items (low)

- Image-feature surplus is truncated instead of raising (`repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2046–L2054); the released pipelines never hit it **[T2d]**.
- Unreachable code: `latent_projection` for `compress_strategy == "linear"` (the CLI only allows `average`), and the `latent_hidden_states` scatter branch (`repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2146).
- Token ids 151665/151666/151667 are hard-coded in the decoding loop ({{LINE repo/transformers/src/transformers/generation/utils.py 3370 "latent_start_idx = 151666"}}–L3372); they match the released tokenizer **[T1b]** but would silently mismatch a tokenizer with other ids.
- `MAX_LATENT_LEN = 8` is hard-coded ({{LINE repo/transformers/src/transformers/generation/utils.py 3295 "MAX_LATENT_LEN = 8"}}) rather than derived from `config.latent_size`.
- README: "a sample dataset of 100 examples"; `repo/data/examples/sample.jsonl` holds 10 records.

---

## 2. Verified as consistent with the paper (not gaps)

| Property | Evidence |
|---|---|
| Stage-1 loss = 1 − cos(h_{p−1}, ê_p) + 0.1·CE, prediction from the preceding position | `repo/src/trainer.py` L13–L25; **[T3b]** |
| Stage-1 targets carry no trainable path (vision tower frozen) | `repo/src/main.py` L68–L69; **[T3c, T3e]** |
| Stage 2 = CE only, latents generated by the model, gradient flows through every latent | `repo/src/trainer.py` L34–L38; `repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2248–L2290; **[T4d, T4e, T4g]** |
| Multi-pass Stage-2 forward equals a single causal pass | **[T4f]** (max abs logit difference 2.7e-7) |
| Latents use 1D text RoPE positions consistently in Stage 1, Stage 2 and inference | `CODE_VS_PAPER_TRACE.md` §3; **[T4f]** |
| Inference feeds the previous last-layer hidden state, as in Stage 2 | `repo/transformers/src/transformers/generation/utils.py` L3416–L3419; **[T5b]** |
| Labels supervise `<|latent_start|>`, `<|latent_end|>` and the answer; pads unsupervised | `repo/src/utils.py` L294–L315; **[T2b, T4h]** |

---

## 3. Assessment of the previous `Gaps_Mirage.md`

| Previous claim | Verdict | Evidence |
|---|---|---|
| Header: "CVPR 2026 Submission" | Unsupported | `paper_src/neurips_2025_arxiv.tex` L5 uses `\usepackage[preprint]{neurips_2025}`; the README says "arXiv 2025". |
| §1.1 Stage 1 loss is `0.1·CE + (1 − cos)` (trainer L6–L26) | Correct | `repo/src/trainer.py` L25; **[T3b]** |
| §1.1 Stage 2 removes the cosine loss; `loss = ce_loss` at L38 | Correct, and stated by the paper | {{LINE paper_src/sec/3_framework.tex 83 "we remove the cosine loss altogether"}}; it is a disclosed design choice, not a hidden finding. |
| §1.1 "Catastrophic representational drift … arbitrary prompt-tuning prefixes and format-lock tokens" | Not tested by the previous audit | No experiment was run. The paper reports the opposite for the two-stage model (latents "stay close to the visual manifold", {{LINE paper_src/sec/5_analysis_arxiv.tex 29 "stay close to the visual manifold"}}) and drift only without Stage 1 ({{LINE paper_src/sec/4_experiments_arxiv.tex 145 "the latent vectors drift into regions"}}). This is hypothesis H2/H3 in `PREREGISTRATION_MIRAGE.md`, tested by Axis 1 and by the latent–image cosine telemetry. |
| §2.1 Latent tokens added at `repo/src/main.py` L36–L38, ids read at L61–L63 | Correct | — |
| §2.1 Latent length fixed by the data collators | Correct for training | `process_batch(..., args.latent_size, ...)`, `repo/src/main.py` L108–L109 and L150–L151 |
| §2.1 "The model lacks dynamic early stopping for mental imagery rollouts" | Incorrect at inference | `<|latent_end|>` ends latent mode (`repo/transformers/src/transformers/generation/utils.py` L3380–L3384), with a cap of 8 (G3). |
| Paths `/home/devraj/...`; no commit recorded | Not reproducible as written | This document pins commit `53f26de` and uses repository-relative paths. |
| `forensic_audit_mirage.py`, `check_mirage.py` | Not evidence | They count words (`stage`, `detach`, …) and list files. |

---

## 4. Open questions (not answerable from code)

Whether the latent vectors carry task information, or whether the gains come from the trained weights and the answer format, cannot be settled by reading code. `PREREGISTRATION_MIRAGE.md` states hypotheses H1–H3 with decision rules, and `MIRAGE_ABLATION_STUDY_DESIGN.md` specifies the GPU experiments (zero, noise, donor and pause latents; K sweep; base model under matched prompts) that decide them.
