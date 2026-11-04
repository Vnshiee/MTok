# Mirage — Ablation Study Design

**Scope:** the released checkpoint `Miiche/vsp_spatial_planning_direct_sft` @ `d1b6fdce` (Stage 2, `latent_size = 4`), code `repo/` @ `53f26de2682025e146781c5b198ec93bdfe4c4d6`, base model `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898`.
**Hypotheses, decision rules and statistics:** `PREREGISTRATION_MIRAGE.md` (this document defines the experiments; that one fixes how they are judged).
**Implementation:** `external_models/mirage/model_loader.py`, `external_models/mirage/ablation_study/run_mirage_ablation.py`, `external_models/mirage/ablation_study/run_all_ablations.sh`, `external_models/mirage/ablation_study/summarize_ablation_results.py`.
**CPU verification:** `external_models/mirage/diagnostics/verify_mirage_autograd.py` (29/29 checks; IDs **[T…]** below).

Every code block is copied verbatim from the cited lines by `external_models/mirage/verification/render_docs.py` and re-checked by `external_models/mirage/verification/verify_doc_snippets.py`.

---

## 0. Object of study: what is fed at a latent position

Notation. x = prompt tokens (system, user text, image tokens), K = number of latent positions, h_t ∈ ℝ^3584 = last-layer hidden state after the final RMSNorm at position t (the tensor captured on `model.model.norm`), E(·) = input-embedding lookup, V = visual-merger output for the question's image (one vector per image token, in the LM input space).

During inference the fork's `_sample` loop decides the next token and, inside latent mode, overwrites it with `<|latent_pad|>`, counts latents, and forces `<|latent_end|>` after 8:

`external_models/mirage/repo/transformers/src/transformers/generation/utils.py` L3374–L3396:

```python
            #start from earliest and end after last in batch
            for i, token_str in enumerate(next_token_strs):
                if token_str == latent_start_idx:
                    in_latent_mode = True
                    latent_start = True
                
                elif token_str == latent_end_idx:
                    latent_end_num += 1
                    if latent_end_num == cur_bs:    
                        latent_end = True
                        in_latent_mode = False
                
                elif in_latent_mode:
                    next_tokens[i, None] = latent_pad_idx
                    
                    
            if in_latent_mode and not latent_end and not latent_start:
                latent_num += 1

                if latent_num > MAX_LATENT_LEN:
                    latent_end = True
                    in_latent_mode = False
                    next_tokens[:, None] = latent_end_idx
```

and feeds the previous step's hidden state as the next input embedding:

`external_models/mirage/repo/transformers/src/transformers/generation/utils.py` L3416–L3419:

```python
            # in latent mode, use latent embeds
            if in_latent_mode and not latent_start:
                cur_latent_embeds = outputs.hidden_states
                inputs_embeds = cur_latent_embeds
```

So for latent step j = 0, …, K−1 with the block starting at position s (the `<|latent_start|>` token):

  e_{s+1+j} = h_{s+j}   (C0, the model's own latent)

and the answer is decoded after `<|latent_end|>` at position s+K+1. The ablation replaces e_{s+1+j} and changes K; nothing else in the forward pass is altered.

The runner reproduces this loop at batch size 1 with a KV cache. Under `--quota auto` it applies the fork's two exit rules (predicted `<|latent_end|>`; cap of 8); under an integer quota it feeds exactly K latents and records, without acting on them, the steps at which the model would have ended the block:

`external_models/mirage/ablation_study/run_mirage_ablation.py` L361–L375:

```python
        if phase == "latent":
            k = seg["steps"]
            lg = logits_hook(len(ids), logits) if logits_hook else logits
            would = choose_token(lg, ids, args, gen)
            if args.quota == "auto":
                if would == lat.end:
                    exit_now, reason = True, "model_latent_end"
                elif k + 1 > args.max_latent_auto:
                    exit_now, reason = True, "max_latent_cap"
                else:
                    exit_now, reason = False, None
            else:
                if would == lat.end and k < args.quota:
                    seg["would_exit_steps"].append(k)
                exit_now, reason = (k >= args.quota), "quota_reached"
```

**[T6]** shows the runner's `--quota auto` output equals the fork's `generate()` token for token, both for an early `<|latent_end|>` (3 latents) and for the 8-latent cap.

---

## 1. Axis 1 — Causal interventions on the latent vectors (C0–C3 and controls)

`external_models/mirage/model_loader.py` L259–L281:

```python
    def apply(self, qid: str, step: int, natural):
        import torch
        if self.mode == "none":
            return natural, {}
        if self.mode == "zero":
            return torch.zeros_like(natural), {}
        if self.mode == "noise":
            g = torch.Generator(device="cpu").manual_seed(stable_seed(qid, self.noise_salt * 1000 + step))
            noise = torch.randn(natural.shape, generator=g, dtype=torch.float32)
            noise = noise / noise.norm() * natural.float().norm().cpu()
            return noise.to(natural.device, natural.dtype), {}
        if self.mode == "pause":
            return self.pause_vec.to(natural.dtype).view_as(natural), {}
        if self.mode == "mean":
            s = min(step, len(self.mean_by_step) - 1)
            return self.mean_by_step[s].to(natural.device, natural.dtype).view_as(natural), {"mean_step_used": s}
        if self.mode == "donor":
            did = self.donor_for(qid)
            bank = self.donor_bank.get(did)
            if bank is None or bank.shape[0] <= step:
                return torch.zeros_like(natural), {"donor_id": did, "donor_missing_step": True}
            return bank[step].to(natural.device, natural.dtype).view_as(natural), {"donor_id": did}
        raise AssertionError
```

| ID | `--intervention` | Fed vector e_{s+1+j} | Question it answers |
|---|---|---|---|
| C0 | `none` | h_{s+j} | reference trajectory |
| C1 | `zero` | 0 | does the answer need any latent content? |
| C2 | `noise` | ‖h_{s+j}‖ · ε/‖ε‖, ε ~ N(0, I), generator seed `int(sha256(f"{1000·seed + j}:{qid}")[:8], 16)` | does the answer need content beyond the norm scale? |
| C3 | `donor` | h^{(d)}_{s+j}: the j-th latent that C0 fed for donor question d at the same K | is the content question-specific, and does it carry the donor's answer? |
| pause | `pause` | E(".") | pause-token control: an in-vocabulary embedding with no content |
| mean | `mean` | (1/|Q|) Σ_{q∈Q} h^{(q)}_{s+j} over the C0 bank at the same K | dataset-average latent: correct scale and direction, no per-question information |
| suppress | `suppress` | — (`<|latent_start|>` banned in decoding; no block) | does the output format itself matter? |

Details that affect interpretation:

* Under C1–C3, pause and mean, every latent position is replaced, so the hidden states h_{s+j} for j ≥ 1 that the runner logs as `natural_*` are computed on the intervened trajectory. They describe what the model would have fed next, not the C0 latents.
* C3 and mean read the latents saved by the C0 run at the same K (`outputs/latents/<benchmark>__target__K<K>__none__forced/`); `run_all_ablations.sh` always runs C0 before them. Donor pairs are fixed by `build_donor_map` (seed 20261103): each question receives a different question of the same benchmark, preferring one whose ground-truth key differs (VSP: the map; multiple choice: the correct letter). **[T7]** checks that fed donor latents equal the saved C0 latents bit for bit, that noise is norm-matched and seeded, and that pause equals the token embedding.
* K = 0 with an intervention other than `none`/`suppress` is rejected, because no vector is fed (`external_models/mirage/ablation_study/run_mirage_ablation.py` L100).

Telemetry per latent step (`latent_steps` in each JSON line): `natural_norm`, `fed_norm`, `cos_fed_natural`, `cos_natural_prev_natural`, `cos_fed_prev_fed`, `max_cos_natural_image_token`, `argmax_image_token`, `cos_natural_image_centroid`, `max_cos_natural_null_image_token` (against the image tokens of the previously processed question, the H3 null), `would_be_token`, plus `donor_id` or `mean_step_used` where applicable.

---

## 2. Axis 2 — Autograd paths and loss dynamics

No GPU training is run. Axis 2 establishes, on CPU and with the repository's own collate functions and trainers, which gradients exist in each stage, so that the inference-time interventions of Axis 1 can be interpreted against what training could have shaped.

### 2.1 Stage 1 (teacher-forced latents, cosine alignment)

Let P be the K latent positions and H ∈ ℝ^{M×d} the helper image's visual-merger features in raster order. With g = ⌊M/K⌋ (the last M mod K rows are dropped), the input at the j-th latent position is

  ê_j = (1/g) Σ_{i = jg}^{(j+1)g − 1} H_i,

computed here:

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2105–L2120:

```python
                        latent_image_embeds = latent_image_embeds.view(batch_size, -1, hidden_states)

                        assert n_image_tokens // batch_size == self.model.config.latent_size

                        res = latent_image_embeds.shape[1] % self.model.config.latent_size
                        if res > 0: latent_image_embeds = latent_image_embeds[:, :-res, :]

                        assert latent_image_embeds.shape[1] % self.model.config.latent_size == 0

                        group_size = latent_image_embeds.shape[1] // self.model.config.latent_size

                        chunks = torch.split(latent_image_embeds, group_size, dim=1)  # list of length `small_dim`

                        chunk_means = [c.mean(dim=1, keepdim=True) for c in chunks]

                        latent_image_embeds = torch.cat(chunk_means, dim=1).contiguous()
```

and the loss is L₁ = 0.1 · CE + (1 − (1/|P|) Σ_{p∈P} cos(h_{p−1}, ê_p)):

`external_models/mirage/repo/src/trainer.py` L13–L25:

```python
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        sim_loss = 1 - sim_loss

        loss = 0.1 * ce_loss + sim_loss
```

Verified: **[T3a]** pooling into contiguous raster chunks; **[T3b]** L₁ equals this formula on a real batch; **[T3c]** the LM receives gradient while the frozen vision tower receives none; **[T3d]** the `<|latent_pad|>` embedding row gets zero gradient (every pad input is overwritten); **[T3e]** the cosine targets are not detached in code but have no trainable path.

### 2.2 Stage 2 (model-generated latents, CE only)

L₂ = CE on the text labels; the latent inputs are produced by a multi-pass forward in which pass i feeds h at the position before latent i:

`external_models/mirage/repo/src/trainer.py` L34–L38:

```python
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
```

Verified: **[T4d]** latent i equals the last-layer post-norm hidden state at position pad_i − 1; **[T4f]** the multi-pass logits equal one causal pass over the filled sequence (max |diff| 2.7e-7); **[T4g]** the text CE gradient reaches the producer of every latent, so nothing in the latent chain is detached; **[T4h]** pad positions carry no CE target and `<|latent_end|>` is predicted from the last latent position.

### 2.3 What Axis 2 implies for Axis 1

* After Stage 2, latents are shaped only through their effect on the answer CE. If that effect were negligible, Stage 2 would exert no pressure on latent content, and C1–C3 would leave accuracy unchanged (H2 or H2-null in `PREREGISTRATION_MIRAGE.md` §4).
* `<|latent_end|>` is supervised only at the 4th latent position, so the stop decision can be learned as a position count. The stop-gate telemetry (Axis 3) tests this directly.
* The Stage-1 cosine objective pulls latents toward pooled image features, which makes a positive latent-to-image cosine expected by construction; the H3 decision therefore relies on interventions, not on that cosine.

### 2.4 Out of scope here (requires GPU training)

Re-training with the latent chain detached in Stage 2, Stage 1 only versus Stage 1 + 2, and different γ or K: these would test the training dynamics directly but violate the no-GPU-training constraint of this audit. They are listed as follow-ups in §8.

---

## 3. Axis 3 — Latent quota K

| Setting | Run | Meaning |
|---|---|---|
| `--quota auto --entry natural` | `…__Kauto__none__natural` | fork protocol: the model opens and closes the block itself, capped at 8 |
| `--quota auto --entry natural --decoding test_py` | `…__Kauto__none__natural__testpy` | the paper's decoding (T = 0.7, top-p 0.9; `external_models/mirage/repo/src/test.py` L65) |
| `--quota K --entry forced`, K ∈ {0, 1, 3, 4, 5, 10, 20} | `…__K<K>__none__forced` | exactly K latents; K = 4 is the training value; K > 8 exceeds the fork's cap; K = 0 is `<|latent_start|><|latent_end|>` with no vector |
| C1–C3 × K ∈ {1, 3, 4, 5, 10, 20} (VSP only) | `…__K<K>__{zero,noise,donor}__forced` | whether intervention effects grow with K |

Forced entry appends `\n<|latent_start|>` after `<|im_start|>assistant`, which is the token sequence the training data contains at that point:

`external_models/mirage/ablation_study/run_mirage_ablation.py` L311–L312:

```python
    if args.entry == "forced":
        prompt += "\n<|latent_start|>"
```

Measurements: accuracy versus K; for K > 4, the share of questions whose model would emit `<|latent_end|>` right after the 4th latent and the step of its first such prediction (`stop_gate` in the summarizer, `external_models/mirage/ablation_study/summarize_ablation_results.py` L150); for auto runs, the latent-count histogram and the share of blocks ended by the model rather than by the cap. Comparing these under `none` and under C1–C3 at K = 10 and K = 20 separates a gate that counts positions (fires after 4 regardless of content) from one that reads the latent content.

Confound: every latent occupies one sequential 1D RoPE position (`CODE_VS_PAPER_TRACE.md` §3), so K also shifts the answer's position indices by K − 4 relative to training. The K = 0 and suppress conditions bound this effect from the other side.

---

## 4. Axis 4 — Base model under matched prompts

`--model_mode base` loads `Qwen/Qwen2.5-VL-7B-Instruct` with the same runner, the same `build_prompt` (chat template + the repository's `place_input_image`, `external_models/mirage/ablation_study/run_mirage_ablation.py` L296), the same image preprocessing (`--min_pixels/--max_pixels` applied to both the processor and `qwen_vl_utils`), the same decoding (greedy, repetition penalty 1.05) and the same scorer. The base model has no latent tokens, so it runs only with `--quota 0 --intervention none --entry natural` (`external_models/mirage/ablation_study/run_mirage_ablation.py` L92).

Comparisons (exploratory, `PREREGISTRATION_MIRAGE.md` §9):

* base vs target-auto (`…__Kauto__none__natural`): total effect of the Mirage fine-tune under each model's own protocol;
* base vs target-suppress: effect of the fine-tuned weights without any latent block;
* target-suppress vs R: effect of the latent block given the fine-tuned weights.

On VSP the instruction text is the dataset's own prompt and the answer is the boxed action string. On BLINK Jigsaw, V\* and MMVP the runner appends one instruction asking for the option letter in `\boxed{}`; the scorer `parse_mc` accepts boxed letters, `<answer>` tags, answer phrases, a leading option marker, unique option text and, last, the final standalone letter (`parse_method` records which rule fired, so the rule mix can be compared between models).

---

## 5. Run matrix

`external_models/mirage/ablation_study/run_all_ablations.sh` (all four benchmarks, `DRY_RUN=1` lists 76 conditions):

| Phase | Benchmark | Conditions | Count |
|---|---|---|---:|
| 1a | VSP planning (400) | auto greedy; auto `test_py` | 2 |
| 1b | VSP planning | K ∈ {0, 1, 3, 4, 5, 10, 20}, `none`, forced | 7 |
| 1c | VSP planning | {zero, noise, donor} × K ∈ {1, 3, 4, 5, 10, 20}; pause, mean at K = 4 | 20 |
| 1d | VSP planning | suppress; base | 2 |
| 2 | BLINK Jigsaw (150), V\* (191), MMVP (300), each | base; auto; K sweep (7); {zero, noise, donor, pause, mean} at K = 4; suppress | 3 × 15 |
| 3 | all | `summarize_ablation_results.py` → `outputs/ABLATION_SUMMARY.md`, `outputs/ablation_summary.json` | — |

Total: 31 + 45 = 76 runs. Each run writes `outputs/<run>.jsonl` (one JSON object per question), `outputs/<run>.meta.json` (arguments, model revision, library and GPU information), `outputs/<run>.COMPLETE` on success, `logs/<run>.log`, and for C0 target runs the fed latents in `outputs/latents/<run>/`. Runs are resumable at question granularity and skipped once complete.

Per-question JSON fields: `question_id`, `category`, `ground_truth`, `parsed_option`, `parse_method`, `correct` (VSP adds `vsp_status`, `vsp_invalid`), `generated_text`, `answer_text`, `generated_token_ids`, `n_prompt_tokens`, `n_image_tokens`, `n_generated_tokens`, `stop_reason`, `n_latent_segments`, `latent_steps_total`, `latent_segments` (start position, steps, `would_exit_steps`, `exit_reason`, entry), `latent_steps` (§1), `latent_norms`, `natural_latent_norms`, `step_cosine`, `seconds`, and `pair_id` (MMVP) or `donor_id` (C3).

---

## 6. Threats to validity and how the design handles them

| Threat | Handling |
|---|---|
| Runner differs from the fork's generation | **[T6]** token-for-token equality under `--quota auto`; batch size 1 avoids the fork's batch-global latent state (`Gaps_Mirage.md` G7, **[T5d]**) |
| Forced entry is not the model's own choice | auto runs record how often the model enters the block itself (`entered_latent_frac` in `ablation_summary.json`), and Δ of the auto greedy run versus R is in the summary table |
| Second latent block after the first | if the model emits `<|latent_start|>` again, the runner opens a new segment with the same exit rule and logs it separately (`n_latent_segments`, `latent_segments`). The fork does not leave a second block on `<|latent_end|>` (**[T5c]**, `Gaps_Mirage.md` G7), so the runner–fork equality of **[T6]** covers single-block outputs only; the frequency of second blocks is reported |
| Intervention changes the norm scale | C2 is norm-matched; mean preserves the per-step average; `fed_norm` is logged |
| Donor shares the answer by chance | adoption is computed only over pairs whose C0 answers differ |
| H3 null image | uses the previous question's image tokens; the first question of a run (and after a resume) has no null and is excluded from the contrast |
| GPU non-determinism in bf16 | single greedy pass per condition; `.meta.json` records torch, CUDA, GPU and attention implementation |
| Multiple comparisons | Holm within the 7-condition confirmatory family, and separately over all conditions of a benchmark |
| Out-of-distribution benchmarks | exploratory only; the checkpoint was fine-tuned on VSP planning |
| V\* image size | processor default (as `external_models/mirage/repo/src/test.py`) unless `VSTAR_MAX_PIXELS` is set; the same setting applies to base and target |

---

## 7. Execution

From the MTok repository root:

```bash
git clone https://github.com/UMass-Embodied-AGI/Mirage.git external_models/mirage/repo
git -C external_models/mirage/repo checkout 53f26de2682025e146781c5b198ec93bdfe4c4d6
pip install -r external_models/mirage/ablation_study/requirements_ablation.txt
cd external_models/mirage/ablation_study
SMOKE=1 bash run_all_ablations.sh
mkdir -p logs && tmux new-session -d -s mirage_ablation "bash run_all_ablations.sh 2>&1 | tee -a logs/run_all.log"
```

`SMOKE=1` runs every condition on 5 questions into `outputs_smoke/` first; the full run then writes to `outputs/`. Phase 0 checks the repository commit, extracts the VSP test set if needed, checks CUDA, and runs the CPU diagnostic.

---

## 8. Follow-ups that need GPU training (not part of this study)

1. Stage 2 with `detach()` on each fed latent: isolates the contribution of backpropagation through the latent chain.
2. Stage 1 only versus Stage 1 + Stage 2 checkpoints under C1–C3.
3. Training with K ∈ {2, 8} and testing the stop gate at K > training value.
4. Supervising `<|latent_end|>` at every latent position (stop/continue labels) to test whether a content-dependent gate can be learned.
