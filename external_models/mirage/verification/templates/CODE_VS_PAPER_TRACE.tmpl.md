# Mirage — Code-vs-Paper Trace

**Paper:** arXiv 2506.17218 (`paper_src/`, identical to the arXiv e-print; see `STEP1_INGESTION_MANIFEST.md` §2).
**Code:** `repo/` = UMass-Embodied-AGI/Mirage @ `53f26de2682025e146781c5b198ec93bdfe4c4d6`, including the vendored transformers fork `repo/transformers/src/transformers` (4.50.0.dev0).
**Released checkpoint:** `Miiche/vsp_spatial_planning_direct_sft` @ `d1b6fdce` (`config.stage = "stage2"`, `latent_size = 4`).

Every code block below is copied verbatim from the cited lines by `verification/render_docs.py`, which fails if a range or quoted substring does not match. Paper quotes are asserted to occur on the cited LaTeX line. Behaviour marked **[T…]** is executed and checked on CPU by `diagnostics/verify_mirage_autograd.py` (29/29 checks pass), which runs the repository's own collate functions, trainers, fork forward and fork `generate()` on a tiny randomly initialised model built from the released config, tokenizer and processor.

---

## 1. Forward-pass tensor flow and embedding injection points

### 1.1 Token sequence

Each training answer is `[helper image, text_output]` (`repo/src/task.py` L18–L23). The assistant-side image placeholder is renamed to latent delimiters:

{{CODE repo/src/utils.py 116 123}}

`process_batch` then replaces everything between `<|latent_start|>` and `<|latent_end|>` (the expanded helper-image `<|image_pad|>` run) with `latent_size` copies of `<|latent_pad|>`:

{{CODE repo/src/utils.py 184 198}}

The user image keeps its normal `<|vision_start|><|image_pad|>…<|vision_end|>` run; its pixels are taken from a user-only processor pass, and the helper image's pixels go to `pixel_values_latent` (Stage 1 only):

{{CODE repo/src/main.py 94 100}}

Result for K = 4: `… <|im_start|>assistant \n <|latent_start|> <|latent_pad|>×4 <|latent_end|> \boxed{…} <|im_end|>` **[T2a]**. Token ids from the released tokenizer: `<|latent_pad|>` 151665, `<|latent_start|>` 151666, `<|latent_end|>` 151667, equal to `config.json` and to the constants hard-coded in the fork's decoding loop **[T1a, T1b]**:

{{CODE repo/transformers/src/transformers/generation/utils.py 3370 3372}}

### 1.2 Stage 1: ground-truth latent inputs

The paper defines the Stage-1 targets as compressed helper-image features:

{{TEX paper_src/sec/3_framework.tex 52 "we realize ${\rm Compress}(\cdot)$ with simple average pooling over the original patch features"}}

The fork computes them with the frozen vision tower, splits the patch sequence into K contiguous chunks in the vision tower's output order, averages each chunk, drops any remainder that does not divide by K, and scatters the K means into the `<|latent_pad|>` input positions:

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2103 2122}}

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2136 2143}}

Verified **[T3a]**: the four pad inputs equal the four contiguous-chunk means of `model.visual(pixel_values_latent)` exactly. Because the merged visual tokens are in row-major order, each latent target is the mean of a horizontal band of the helper image (K = 4 → four bands). "Average pooling over the original patch features" in the paper does not specify this banding.

### 1.3 Stage 2: model-generated latents (multi-pass forward)

{{TEX paper_src/sec/3_framework.tex 88 "\bm e_{j}=f_{\bm \theta}\bigl(\bm x, \bm o_{\rm pre},\, \bm e_{<j} \bigr)."}}

When `config.stage == 'stage2'` and `generate_mode` is False, the forward runs K + 1 passes with a KV cache. Pass 0 covers positions `[0, L0)` (L0 = first latent position); pass i (1 ≤ i < K) covers the single position holding latent i−1; after each pass the last hidden state is written into the next latent position:

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2248 2290}}

The final pass covers the rest of the sequence, and logits are computed on the concatenation of all pass outputs:

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2324 2327}}

Verified: logits cover the full sequence **[T4c]**; latent i equals the last-layer post-norm hidden state at position `pad_i − 1` **[T4d]**; the multi-pass logits equal one causal pass over the filled sequence to 2.7e-7 **[T4f]**. The pass schedule uses the batch-global minimum latent position (`next_compute_range = (0, latent_indices[:, 1].min().item())`, L2176) and the same `+1` steps for every row, so it is exact only when all rows have their latents at the same positions; training uses `per_device_train_batch_size=1` (`repo/src/main.py` L179) and the released run used 1 GPU (`_n_gpu = 1`).

### 1.4 Inference: the fork's `_sample` loop

{{CODE repo/transformers/src/transformers/generation/utils.py 3286 3296}}

Every forward at inference passes `generate_mode=True`, so the Stage-2 multi-pass branch is never taken at test time:

{{CODE repo/transformers/src/transformers/generation/utils.py 3316 3320}}

Latent-mode state machine (token selection → latent bookkeeping):

{{CODE repo/transformers/src/transformers/generation/utils.py 3374 3402}}

Feedback of the hidden state as the next input embedding:

{{CODE repo/transformers/src/transformers/generation/utils.py 3416 3419}}

Behaviour, verified on the fork itself:
- After `<|latent_start|>` is sampled and processed, each step appends `<|latent_pad|>` and feeds the previous step's last-layer post-norm hidden state **[T5b]**, the same quantity Stage 2 trains with (§1.3). Positions match Stage 2: latent j sits at `start + 1 + j`.
- Latent mode ends when the token chosen at a latent step is `<|latent_end|>`, or after `MAX_LATENT_LEN = 8` latents (L3295, L3393–L3396), when `<|latent_end|>` is forced **[T5a]**. Training uses K = 4, so the number of latents at test time is decided by the model, capped at 8.
- `in_latent_mode`, `latent_end` and `latent_end_num` are batch-global and never reset. With batch size > 1, a row that never produced `<|latent_start|>` is forced to `<|latent_pad|>` **[T5d]**. After a second `<|latent_start|>` in the same sequence, `latent_end` is already True and `latent_end_num` no longer equals the batch size, so the loop keeps feeding hidden states for every remaining step and a second `<|latent_end|>` does not end latent mode **[T5c]**. `repo/src/test.py` runs batch size 1, and the released direct-answer model has one latent segment, so neither path is exercised by the released evaluation; both affect any interleaved (multi-segment) or batched use.

### 1.5 Other modified paths

- Image-feature/token count mismatch: upstream raises; the fork keeps the first `n_image_tokens / B` features per row when there are more features than tokens:

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2046 2054}}

  In the released pipelines the counts match (user-image tokens = user-image features; the helper image is collapsed to K pads), so this branch is not taken **[T2d]**.
- `latent_projection` is created only for `compress_strategy == "linear"` ({{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 1747 "if self.model.config.compress_strategy in [\"linear\"]:"}}), but the CLI only accepts `average` ({{LINE repo/src/utils.py 18 "choices=['average']"}}); the linear path is unreachable.
- The `latent_hidden_states` scatter branch (`elif latent_hidden_states is not None and not generate_mode`, {{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2146 "elif latent_hidden_states is not None and not generate_mode:"}}) is unreachable: training never passes `latent_hidden_states`, and generation passes it only with `generate_mode=True`.

---

## 2. Loss masking, detachment and index shifts

### 2.1 Labels

{{CODE repo/src/main.py 114 120}}

{{CODE repo/src/utils.py 294 315}}

The fourth argument of `generate_labels_after_multi_token_start` is the latent-pad id, so pads are masked (the parameter name `img_token_idx` is reused). Everything up to `<|im_start|>assistant` and every `<|endoftext|>` is masked; `\n`, `<|latent_start|>`, `<|latent_end|>`, the answer and `<|im_end|>` are supervised **[T2b]**. With the CE shift (`shift_labels = labels[..., 1:]`), the logit at the last latent position is trained to predict `<|latent_end|>`, and logits at the other latent positions carry no CE target **[T4h]**.

### 2.2 Stage-1 loss and index alignment

{{TEX paper_src/sec/3_framework.tex 62 "\ell_{\cos}\Bigl(\hat{\bm{e}}_{j},\,g_{\theta}\bigl(\bm o_{\rm pre},\,\hat{\bm{e}}_{1:j-1}\bigr)\Bigr),"}}

{{TEX paper_src/sec/3_framework.tex 77 "$\mathcal{L}_{1}=\mathcal{L}_{\rm visual}+\gamma \mathcal{L}_{\rm text}$"}}

{{CODE repo/src/trainer.py 10 26}}

- `predict_embeddings[..., :-1, :]` with the mask shifted by one selects the hidden state at `p − 1` for each latent position `p`, and `input_embeddings[..., 1:, :]` selects the target at `p`: the prediction of latent j is made from the position before it (from `<|latent_start|>` for j = 1), matching Eq. 1 **[T2c, T3b]**.
- `outputs.hidden_states` is the fork's last-layer post-norm hidden state (`hidden_states=hidden_states`, {{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2438 "hidden_states=hidden_states,"}}), matching "last layer hidden states" ({{LINE paper_src/sec/3_framework.tex 54 "the last layer hidden states at corresponding positions"}}).
- Code weighting: `loss = 0.1 * ce_loss + sim_loss`, i.e. γ = 0.1 on the text loss, as in the main text ({{LINE paper_src/sec/4_experiments_arxiv.tex 26 "$\gamma=0.1$"}}). Verified numerically **[T3b]**.

### 2.3 Detachment

- No `.detach()` appears in `CustomTrainerStage1`; the targets `gt_embeddings` are rows of `outputs.inputs_embeds`. Those rows come from the vision tower, which is frozen in both stages:

{{CODE repo/src/main.py 68 69}}

  so the cosine loss has no trainable path through the targets, and the frozen tower receives no gradient **[T3c, T3e]**.
- The `<|latent_pad|>` input-embedding row receives zero gradient in Stage 1 because every pad input is overwritten by `masked_scatter` **[T3d]**; it is likewise overwritten in Stage 2 and at inference, so its value never enters the computation.
- Stage 2 has no `.detach()` between the pass outputs and the filled latents (§1.3 code). The text CE gradient reaches the producer of every latent **[T4g]** (‖∂L/∂h‖ ≈ 0.11 at each of the 4 producers in the CPU test), consistent with:

{{TEX paper_src/sec/3_framework.tex 99 "the gradient can be propagated to these latent tokens when minimizing the above loss on the textual tokens"}}

- Stage-2 loss is CE only **[T4e]**:

{{CODE repo/src/trainer.py 34 39}}

  as stated in the paper:

{{TEX paper_src/sec/3_framework.tex 83 "in the second stage, we remove the cosine loss altogether and keep only the cross-entropy loss over text tokens"}}

---

## 3. RoPE coordinates (text 1D vs image 2D)

Qwen2.5-VL multimodal RoPE assigns (t, h, w) grid coordinates only to `<|image_pad|>`/`<|video_pad|>` runs; every other token, including the three latent tokens, receives the same sequential index on all three axes:

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 1910 1913}}

{{CODE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 1928 1931}}

- Stage 1: positions come from `get_rope_index` over the full sequence; the K pooled helper-image vectors sit at 1D text positions. Their ViT features were computed with the helper image's own 2D grid (`image_grid_thw_latent`) before pooling, so the 2D layout survives only inside the features, averaged into bands (§1.2).
- Stage 2: pass 0 calls `get_rope_index` on `[0, L0)` and later passes use `cache_position + rope_deltas` (sequential 1D), {{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2215 "cache_position = torch.arange(next_compute_range[0], next_compute_range[1], device=inputs_embeds.device)"}}–{{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2227 "position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)"}}. Identical to a single full pass **[T4f]**.
- Inference: decode steps use the same `cache_position + rope_deltas` rule ({{LINE repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py 2388 "position_ids = torch.arange(seq_length, device=inputs_embeds.device)"}}). Latent positions are therefore consistent across Stage 1, Stage 2 and inference.

---

## 4. Teacher forcing vs autoregressive rollout

| | Stage 1 | Stage 2 | Inference (`_sample`) |
|---|---|---|---|
| Latent input at pad j | pooled helper-image band j (ground truth) | last hidden state at the previous position (model output) | last hidden state at the previous position |
| Supervision on latents | 1 − cos(h_{p−1}, target_p) | none (CE on text only, gradient through latents) | — |
| Number of latents | K = 4 (data) | K = 4 (data; `max_n_latents != latent_size` raises, L2171–L2174) | model-decided, ≤ 8 |
| Exit from latent mode | fixed by data | fixed by data | sampled/argmax `<|latent_end|>` or cap |

- Stage 1 is teacher-forced: the hidden states it learns to produce are never fed back during Stage 1. Stage 2 trains on the model's own rollouts with full backpropagation through the latent chain, so at the end of training the inference-time input distribution (own hidden states) matches Stage 2.
- The one train/test difference that remains is the latent count and exit decision. The model is trained to emit `<|latent_end|>` after exactly 4 latents (§2.1), but at test time it may emit it earlier or later (capped at 8). With `repo/src/test.py`'s sampling (`do_sample=True, temperature=0.7, top_p=0.9`), that decision is itself sampled. The ablation runner records the exit step and reason for every question.
- Measured effect of the remaining mismatch: not available offline; it is Axis 3 of `MIRAGE_ABLATION_STUDY_DESIGN.md` (forced K ∈ {0, 1, 3, 4, 5, 10, 20} vs the model's own exit rule).

---

## 5. Answer extraction and evaluation

{{CODE repo/src/test.py 57 79}}

{{CODE repo/src/task.py 115 120}}

{{CODE repo/src/task.py 140 158}}

- The prompt ends with `<|im_start|>assistant` and no newline; the model generates `\n` and then `<|latent_start|>` itself, as in the training text.
- Decoding is stochastic (`do_sample=True, temperature=0.7, top_p=0.9`, seed 42 via `seed_everything`) and single-sample; the checkpoint's `generation_config.json` additionally applies `repetition_penalty=1.05`. Reported accuracies are one draw.
- `parse_action_sequence` keeps every `U`, `D`, `R`, `L` character of the upper-cased boxed text, so word answers parse (`DOWN` → `[D]`, `RIGHT` → `[R]`, `LEFT` → `[L]`, `UP` → `[U]`, the format of the released training answers, e.g. `\boxed{DOWN}`), but any other word inside the box contributes its U/D/R/L letters as moves.
- Because the parser only emits characters that are keys of `ACTION_MAP`, the `"Invalid action"` branch of `simulate_vsp` (L142–L147) is unreachable, so the "Invalid" rate logged by `repo/src/test.py` (L79, L82, L84) is always 0. A missing `\boxed{}` gives `"None"` → no moves → "Did not reach goal".
- Moves off the grid are ignored (`continue`, L155–L156) instead of failing, and success requires ending on the goal; a path that passes over the goal and leaves fails.
- Only VSP planning has evaluation code. `repo/src/test.py` scores every task with `simulate_vsp(map_desc, …)`; COMT, BLINK-Jigsaw, SAT and VSP spatial reasoning (which have no `map_desc`) cannot be evaluated with the released code (`simulate_vsp([])` raises `ValueError` at L137–L138).

---

## 6. Paper statements vs released code and checkpoint

| Item | Paper | Released code / checkpoint |
|---|---|---|
| Compression | "average pooling over the original patch features" (`paper_src/sec/3_framework.tex` L52) | mean of K contiguous raster chunks, remainder dropped (§1.2) |
| γ (text-loss weight) | 0.1 ({{LINE paper_src/sec/4_experiments_arxiv.tex 26 "$\gamma=0.1$"}}); **10** in the supplement table ({{LINE paper_src/sec/7_supp.tex 275 "loss weight $\gamma$ & 10"}}) | 0.1 (`repo/src/trainer.py` L25) |
| Batch size | 8 ({{LINE paper_src/sec/4_experiments_arxiv.tex 25 "batch size of 8"}}; {{LINE paper_src/sec/7_supp.tex 271 "batch size &  8"}}) | per-device 1, 1 GPU; grad-accum 1 in the released run (`training_args.bin`), 8 / 1 in README stage 1 / stage 2 |
| Gradient accumulation | 2 ({{LINE paper_src/sec/7_supp.tex 272 "gradient accumulation steps & 2"}}) | 1 (released run) |
| LR schedule | cosine ({{LINE paper_src/sec/4_experiments_arxiv.tex 25 "cosine learning rate scheduler"}}) | `linear` (SFTConfig default; `repo/src/main.py` L176–L198 does not set it) |
| Adam β₂ | 0.95 ({{LINE paper_src/sec/7_supp.tex 273 "0.95"}}) | 0.999 (released run) |
| Epochs | 10 ({{LINE paper_src/sec/7_supp.tex 274 "training epochs & 10"}}) | 15 (released run; README); 10 in `repo/train.sh` |
| VSP SFT samples | 1k ({{LINE paper_src/sec/4_experiments_arxiv.tex 14 "we sample 1k training instances for fine-tuning"}}); 3,000 ({{LINE paper_src/sec/7_supp.tex 244 "VSP Spatial Planning & 3,000"}}) | 1,000 lines in `train_direct.jsonl` |
| k-ablation, k = 2 row | Avg 0.86 ({{LINE paper_src/sec/4_experiments_arxiv.tex 159 "2 & 0.1 & 0.85 & 0.86 & 0.89 & \textbf{0.93} & 0.86"}}) | mean of the four levels is 0.8825 (all other rows' averages are consistent) |
| RL (GRPO, §3.4) | "we explicitly optimize the probabilities of textual tokens while allowing gradients to flow through the latent tokens" ({{LINE paper_src/sec/3_framework.tex 105 "allowing gradients to flow through the latent tokens"}}) | no RL code in the repository |
| Evaluation of COMT / Jigsaw / SAT / VSP reasoning | Tables 1–3 | no evaluation code (§5) |
| Sample data | — | README: "a sample dataset of 100 examples"; `repo/data/examples/sample.jsonl` has 10 records |
