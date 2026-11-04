# Mirage — Code-vs-Paper Trace

**Paper:** arXiv 2506.17218 (`paper_src/`, identical to the arXiv e-print; see `STEP1_INGESTION_MANIFEST.md` §2).
**Code:** `repo/` = UMass-Embodied-AGI/Mirage @ `53f26de2682025e146781c5b198ec93bdfe4c4d6`, including the vendored transformers fork `external_models/mirage/repo/transformers/src/transformers` (4.50.0.dev0).
**Released checkpoint:** `Miiche/vsp_spatial_planning_direct_sft` @ `d1b6fdce` (`config.stage = "stage2"`, `latent_size = 4`).

Every code block below is copied verbatim from the cited lines by `external_models/mirage/verification/render_docs.py`, which fails if a range or quoted substring does not match. Paper quotes are asserted to occur on the cited LaTeX line. Behaviour marked **[T…]** is executed and checked on CPU by `external_models/mirage/diagnostics/verify_mirage_autograd.py` (29/29 checks pass), which runs the repository's own collate functions, trainers, fork forward and fork `generate()` on a tiny randomly initialised model built from the released config, tokenizer and processor.

---

## 1. Forward-pass tensor flow and embedding injection points

### 1.1 Token sequence

Each training answer is `[helper image, text_output]` (`external_models/mirage/repo/src/task.py` L18–L23). The assistant-side image placeholder is renamed to latent delimiters:

`external_models/mirage/repo/src/utils.py` L116–L123:

```python
def replace_visual_spectial_tokens(texts):

    update_texts = []
    for i, text in enumerate(texts):
        prev, after = text.split("<|im_start|>assistant")
        update_texts.append(prev + "<|im_start|>assistant" + after.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>"))
        
    return update_texts
```

`process_batch` then replaces everything between `<|latent_start|>` and `<|latent_end|>` (the expanded helper-image `<|image_pad|>` run) with `latent_size` copies of `<|latent_pad|>`:

`external_models/mirage/repo/src/utils.py` L184–L198:

```python
    for (s_pos, e_pos) in zip(start_positions, end_positions):
        # Add everything before this image part as-is, BUT include start_token itself
        new_seq_pieces.append(seq[prev_end : s_pos + 1])
        
        # Replace the entire chunk [s_pos+1 .. e_pos) with N copies of replacement_token
        replacement_span = torch.tensor(
            [replacement_token] * replacement_length, 
            dtype=seq.dtype, 
            device=seq.device
        )
        new_seq_pieces.append(replacement_span)

        # Move past the end_token
        prev_end = e_pos
    
```

The user image keeps its normal `<|vision_start|><|image_pad|>…<|vision_end|>` run; its pixels are taken from a user-only processor pass, and the helper image's pixels go to `pixel_values_latent` (Stage 1 only):

`external_models/mirage/repo/src/main.py` L94–L100:

```python
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    batch['pixel_values_latent'] = assistant_batch['pixel_values']
    batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']
```

Result for K = 4: `… <|im_start|>assistant \n <|latent_start|> <|latent_pad|>×4 <|latent_end|> \boxed{…} <|im_end|>` **[T2a]**. Token ids from the released tokenizer: `<|latent_pad|>` 151665, `<|latent_start|>` 151666, `<|latent_end|>` 151667, equal to `config.json` and to the constants hard-coded in the fork's decoding loop **[T1a, T1b]**:

`external_models/mirage/repo/transformers/src/transformers/generation/utils.py` L3370–L3372:

```python
            latent_start_idx = 151666
            latent_end_idx = 151667
            latent_pad_idx = 151665
```

### 1.2 Stage 1: ground-truth latent inputs

The paper defines the Stage-1 targets as compressed helper-image features:

> `external_models/mirage/paper_src/sec/3_framework.tex` L52: “we realize ${\rm Compress}(\cdot)$ with simple average pooling over the original patch features”

The fork computes them with the frozen vision tower, splits the patch sequence into K contiguous chunks in the vision tower's output order, averages each chunk, drops any remainder that does not divide by K, and scatters the K means into the `<|latent_pad|>` input positions:

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2103–L2122:

```python
                    if self.model.config.compress_strategy == "average":
                        
                        latent_image_embeds = latent_image_embeds.view(batch_size, -1, hidden_states)

                        assert n_image_tokens // batch_size == self.model.config.latent_size

                        res = latent_image_embeds.shape[1] % self.model.config.latent_size
                        if res > 0: latent_image_embeds = latent_image_embeds[:, :-res, :]

                        assert latent_image_embeds.shape[1] % self.model.config.latent_size == 0

                        group_size = latent_image_embeds.shape[1] // self.model.config.latent_size

                        chunks = torch.split(latent_image_embeds, group_size, dim=1)  # list of length `small_dim`

                        chunk_means = [c.mean(dim=1, keepdim=True) for c in chunks]

                        latent_image_embeds = torch.cat(chunk_means, dim=1).contiguous()

                        latent_image_embeds = latent_image_embeds.view(-1, hidden_states)
```

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2136–L2143:

```python
                mask = input_ids == self.config.latent_token_id
                
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                latent_image_embeds = latent_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, latent_image_embeds)
```

Verified **[T3a]**: the four pad inputs equal the four contiguous-chunk means of `model.visual(pixel_values_latent)` exactly. Because the merged visual tokens are in row-major order, each latent target is the mean of a horizontal band of the helper image (K = 4 → four bands). "Average pooling over the original patch features" in the paper does not specify this banding.

### 1.3 Stage 2: model-generated latents (multi-pass forward)

> `external_models/mirage/paper_src/sec/3_framework.tex` L88: “\bm e_{j}=f_{\bm \theta}\bigl(\bm x, \bm o_{\rm pre},\, \bm e_{<j} \bigr).”

When `config.stage == 'stage2'` and `generate_mode` is False, the forward runs K + 1 passes with a KV cache. Pass 0 covers positions `[0, L0)` (L0 = first latent position); pass i (1 ≤ i < K) covers the single position holding latent i−1; after each pass the last hidden state is written into the next latent position:

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2248–L2290:

```python
                next_compute_range = (
                    next_compute_range[1],
                    (
                        input_ids.shape[1]
                        if latent_i + 1 >= max_n_latents
                        else next_compute_range[1] + 1
                    ),
                )
                
                hidden_states = outputs[0]
                all_hidden_states.append(hidden_states)
                kv_cache = outputs.past_key_values
                
                hidden_states = hidden_states[:, -1:, :]
                assert hidden_states.shape[1] == 1
                
                #replace the latent token with the output thought
                filling_indices = [
                    (instance_idx, mask_list[latent_i])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > latent_i
                ]
 
                tensor_list = [
                    [
                        inputs_embeds[batch_idx, pos, :]
                        for pos in range(inputs_embeds.shape[1])
                    ]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
  
                # replace some of them with continuous thoughts
                for idx_pair in filling_indices:
                    batch_idx, token_idx = idx_pair  
                    tensor_list[batch_idx][token_idx] = hidden_states[batch_idx][0]

                #TODO: Multi GPU support
                inputs_embeds = torch.stack(
                    [
                        torch.stack(tensor_list[batch_idx])
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]
                )
```

The final pass covers the rest of the sequence, and logits are computed on the concatenation of all pass outputs:

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2324–L2327:

```python
            hidden_states = outputs[0]
            all_hidden_states.append(hidden_states)
            hidden_states = torch.cat(all_hidden_states, dim=1)
            logits = self.lm_head(hidden_states)
```

Verified: logits cover the full sequence **[T4c]**; latent i equals the last-layer post-norm hidden state at position `pad_i − 1` **[T4d]**; the multi-pass logits equal one causal pass over the filled sequence to 2.7e-7 **[T4f]**. The pass schedule uses the batch-global minimum latent position (`next_compute_range = (0, latent_indices[:, 1].min().item())`, L2176) and the same `+1` steps for every row, so it is exact only when all rows have their latents at the same positions; training uses `per_device_train_batch_size=1` (`external_models/mirage/repo/src/main.py` L179) and the released run used 1 GPU (`_n_gpu = 1`).

### 1.4 Inference: the fork's `_sample` loop

`external_models/mirage/repo/transformers/src/transformers/generation/utils.py` L3286–L3296:

```python
        #Latent mode init
        in_latent_mode = False
        after_latent_mode = False
        latent_hidden_states = None
        is_prefill = True
        latent_start = False
        latent_end = False
        latent_num = 0
        latent_end_num = 0
        MAX_LATENT_LEN = 8
        cur_bs = input_ids.shape[0]
```

Every forward at inference passes `generate_mode=True`, so the Stage-2 multi-pass branch is never taken at test time:

`external_models/mirage/repo/transformers/src/transformers/generation/utils.py` L3316–L3320:

```python
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True, generate_mode=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True, generate_mode=True, latent_hidden_states=latent_hidden_states)
```

Latent-mode state machine (token selection → latent bookkeeping):

`external_models/mirage/repo/transformers/src/transformers/generation/utils.py` L3374–L3402:

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
                
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)              
                if latent_hidden_states is None:
                    latent_hidden_states = outputs.hidden_states
                else:
                    latent_hidden_states = torch.cat([latent_hidden_states, outputs.hidden_states], dim=1)
```

Feedback of the hidden state as the next input embedding:

`external_models/mirage/repo/transformers/src/transformers/generation/utils.py` L3416–L3419:

```python
            # in latent mode, use latent embeds
            if in_latent_mode and not latent_start:
                cur_latent_embeds = outputs.hidden_states
                inputs_embeds = cur_latent_embeds
```

Behaviour, verified on the fork itself:
- After `<|latent_start|>` is sampled and processed, each step appends `<|latent_pad|>` and feeds the previous step's last-layer post-norm hidden state **[T5b]**, the same quantity Stage 2 trains with (§1.3). Positions match Stage 2: latent j sits at `start + 1 + j`.
- Latent mode ends when the token chosen at a latent step is `<|latent_end|>`, or after `MAX_LATENT_LEN = 8` latents (L3295, L3393–L3396), when `<|latent_end|>` is forced **[T5a]**. Training uses K = 4, so the number of latents at test time is decided by the model, capped at 8.
- `in_latent_mode`, `latent_end` and `latent_end_num` are batch-global and never reset. With batch size > 1, a row that never produced `<|latent_start|>` is forced to `<|latent_pad|>` **[T5d]**. After a second `<|latent_start|>` in the same sequence, `latent_end` is already True and `latent_end_num` no longer equals the batch size, so the loop keeps feeding hidden states for every remaining step and a second `<|latent_end|>` does not end latent mode **[T5c]**. `external_models/mirage/repo/src/test.py` runs batch size 1, and the released direct-answer model has one latent segment, so neither path is exercised by the released evaluation; both affect any interleaved (multi-segment) or batched use.

### 1.5 Other modified paths

- Image-feature/token count mismatch: upstream raises; the fork keeps the first `n_image_tokens / B` features per row when there are more features than tokens:

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2046–L2054:

```python
                if n_image_tokens != n_image_features:
                    assert n_image_tokens < n_image_features
                    batch_size = input_ids.shape[0]
                    hidden_states = image_embeds.shape[-1]

                    assert n_image_tokens % batch_size == 0
                    image_embeds = image_embeds.view(batch_size, -1, hidden_states)
                    image_embeds = image_embeds[:, :n_image_tokens // batch_size, :].contiguous()
                    image_embeds = image_embeds.view(-1, hidden_states)
```

  In the released pipelines the counts match (user-image tokens = user-image features; the helper image is collapsed to K pads), so this branch is not taken **[T2d]**.
- `latent_projection` is created only for `compress_strategy == "linear"` (`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L1747), but the CLI only accepts `average` (`external_models/mirage/repo/src/utils.py` L18); the linear path is unreachable.
- The `latent_hidden_states` scatter branch (`elif latent_hidden_states is not None and not generate_mode`, `external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2146) is unreachable: training never passes `latent_hidden_states`, and generation passes it only with `generate_mode=True`.

---

## 2. Loss masking, detachment and index shifts

### 2.1 Labels

`external_models/mirage/repo/src/main.py` L114–L120:

```python
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask
```

`external_models/mirage/repo/src/utils.py` L294–L315:

```python
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            row[:] = -100
        else:
            # The sub-sequence length
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len  # the position *after* the sub-sequence
            
            # Mask everything up to (and including) the sub-sequence
            row[:end_of_subseq] = -100
        
        # Mask pad tokens
        row[row == pad_token_idx] = -100
        # Mask image tokens
        row[row == img_token_idx] = -100
    
    return labels
```

The fourth argument of `generate_labels_after_multi_token_start` is the latent-pad id, so pads are masked (the parameter name `img_token_idx` is reused). Everything up to `<|im_start|>assistant` and every `<|endoftext|>` is masked; `\n`, `<|latent_start|>`, `<|latent_end|>`, the answer and `<|im_end|>` are supervised **[T2b]**. With the CE shift (`shift_labels = labels[..., 1:]`), the logit at the last latent position is trained to predict `<|latent_end|>`, and logits at the other latent positions carry no CE target **[T4h]**.

### 2.2 Stage-1 loss and index alignment

> `external_models/mirage/paper_src/sec/3_framework.tex` L62: “\ell_{\cos}\Bigl(\hat{\bm{e}}_{j},\,g_{\theta}\bigl(\bm o_{\rm pre},\,\hat{\bm{e}}_{1:j-1}\bigr)\Bigr),”

> `external_models/mirage/paper_src/sec/3_framework.tex` L77: “$\mathcal{L}_{1}=\mathcal{L}_{\rm visual}+\gamma \mathcal{L}_{\rm text}$”

`external_models/mirage/repo/src/trainer.py` L10–L26:

```python
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        sim_loss = 1 - sim_loss

        loss = 0.1 * ce_loss + sim_loss
        return (loss, outputs) if return_outputs else loss
```

- `predict_embeddings[..., :-1, :]` with the mask shifted by one selects the hidden state at `p − 1` for each latent position `p`, and `input_embeddings[..., 1:, :]` selects the target at `p`: the prediction of latent j is made from the position before it (from `<|latent_start|>` for j = 1), matching Eq. 1 **[T2c, T3b]**.
- `outputs.hidden_states` is the fork's last-layer post-norm hidden state (`hidden_states=hidden_states`, `external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2438), matching "last layer hidden states" (`external_models/mirage/paper_src/sec/3_framework.tex` L54).
- Code weighting: `loss = 0.1 * ce_loss + sim_loss`, i.e. γ = 0.1 on the text loss, as in the main text (`external_models/mirage/paper_src/sec/4_experiments_arxiv.tex` L26). Verified numerically **[T3b]**.

### 2.3 Detachment

- No `.detach()` appears in `CustomTrainerStage1`; the targets `gt_embeddings` are rows of `outputs.inputs_embeds`. Those rows come from the vision tower, which is frozen in both stages:

`external_models/mirage/repo/src/main.py` L68–L69:

```python
for param in model.visual.parameters():
    param.requires_grad = False
```

  so the cosine loss has no trainable path through the targets, and the frozen tower receives no gradient **[T3c, T3e]**.
- The `<|latent_pad|>` input-embedding row receives zero gradient in Stage 1 because every pad input is overwritten by `masked_scatter` **[T3d]**; it is likewise overwritten in Stage 2 and at inference, so its value never enters the computation.
- Stage 2 has no `.detach()` between the pass outputs and the filled latents (§1.3 code). The text CE gradient reaches the producer of every latent **[T4g]** (‖∂L/∂h‖ ≈ 0.11 at each of the 4 producers in the CPU test), consistent with:

> `external_models/mirage/paper_src/sec/3_framework.tex` L99: “the gradient can be propagated to these latent tokens when minimizing the above loss on the textual tokens”

- Stage-2 loss is CE only **[T4e]**:

`external_models/mirage/repo/src/trainer.py` L34–L39:

```python
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        return (loss, outputs) if return_outputs else loss
```

  as stated in the paper:

> `external_models/mirage/paper_src/sec/3_framework.tex` L83: “in the second stage, we remove the cosine loss altogether and keep only the cross-entropy loss over text tokens”

---

## 3. RoPE coordinates (text 1D vs image 2D)

Qwen2.5-VL multimodal RoPE assigns (t, h, w) grid coordinates only to `<|image_pad|>`/`<|video_pad|>` runs; every other token, including the three latent tokens, receives the same sequential index on all three axes:

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L1910–L1913:

```python
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
```

`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L1928–L1931:

```python
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
```

- Stage 1: positions come from `get_rope_index` over the full sequence; the K pooled helper-image vectors sit at 1D text positions. Their ViT features were computed with the helper image's own 2D grid (`image_grid_thw_latent`) before pooling, so the 2D layout survives only inside the features, averaged into bands (§1.2).
- Stage 2: pass 0 calls `get_rope_index` on `[0, L0)` and later passes use `cache_position + rope_deltas` (sequential 1D), `external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2215–`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2227. Identical to a single full pass **[T4f]**.
- Inference: decode steps use the same `cache_position + rope_deltas` rule (`external_models/mirage/repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L2388). Latent positions are therefore consistent across Stage 1, Stage 2 and inference.

---

## 4. Teacher forcing vs autoregressive rollout

| | Stage 1 | Stage 2 | Inference (`_sample`) |
|---|---|---|---|
| Latent input at pad j | pooled helper-image band j (ground truth) | last hidden state at the previous position (model output) | last hidden state at the previous position |
| Supervision on latents | 1 − cos(h_{p−1}, target_p) | none (CE on text only, gradient through latents) | — |
| Number of latents | K = 4 (data) | K = 4 (data; `max_n_latents != latent_size` raises, L2171–L2174) | model-decided, ≤ 8 |
| Exit from latent mode | fixed by data | fixed by data | sampled/argmax `<|latent_end|>` or cap |

- Stage 1 is teacher-forced: the hidden states it learns to produce are never fed back during Stage 1. Stage 2 trains on the model's own rollouts with full backpropagation through the latent chain, so at the end of training the inference-time input distribution (own hidden states) matches Stage 2.
- The one train/test difference that remains is the latent count and exit decision. The model is trained to emit `<|latent_end|>` after exactly 4 latents (§2.1), but at test time it may emit it earlier or later (capped at 8). With `external_models/mirage/repo/src/test.py`'s sampling (`do_sample=True, temperature=0.7, top_p=0.9`), that decision is itself sampled. The ablation runner records the exit step and reason for every question.
- Measured effect of the remaining mismatch: not available offline; it is Axis 3 of `MIRAGE_ABLATION_STUDY_DESIGN.md` (forced K ∈ {0, 1, 3, 4, 5, 10, 20} vs the model's own exit rule).

---

## 5. Answer extraction and evaluation

`external_models/mirage/repo/src/test.py` L57–L79:

```python
    inputs = processor(text=[t+'<|im_start|>assistant' for t in texts], images=image_inputs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            tokenizer=processor.tokenizer,
        )
    
    decoded_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = decoded_output.split('<|im_start|>assistant')[-1]

    map_desc = sample.get("map_desc", [])
    path_str = extract_boxed_content(answer)

    result = simulate_vsp(map_desc, path_str)

    if result['success']: correct += 1
    elif result['invalid']: invalid += 1
```

`external_models/mirage/repo/src/task.py` L115–L120:

```python
def parse_action_sequence(path_str):

    path_str = path_str.upper()
    action_sequence = [char for char in list(path_str) if char in ['U', 'D', 'R', 'L']]

    return action_sequence
```

`external_models/mirage/repo/src/task.py` L140–L158:

```python
    current_position = start
    for action in action_sequence:
        if action not in ACTION_MAP:
            return {
                "success": False,
                "status": "Invalid action",
                "invalid": True,
            }
            raise ValueError(f"Invalid action: {action}. Valid actions are {list(ACTION_MAP.keys())}.")
        
        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc

        # Check boundaries; ignore the move if it goes out-of-bounds.
        if new_r < 0 or new_r >= len(map_desc) or new_c < 0 or new_c >= len(map_desc[0]):
            continue
        
        current_position = (new_r, new_c)
```

- The prompt ends with `<|im_start|>assistant` and no newline; the model generates `\n` and then `<|latent_start|>` itself, as in the training text.
- Decoding is stochastic (`do_sample=True, temperature=0.7, top_p=0.9`, seed 42 via `seed_everything`) and single-sample; the checkpoint's `generation_config.json` additionally applies `repetition_penalty=1.05`. Reported accuracies are one draw.
- `parse_action_sequence` keeps every `U`, `D`, `R`, `L` character of the upper-cased boxed text, so word answers parse (`DOWN` → `[D]`, `RIGHT` → `[R]`, `LEFT` → `[L]`, `UP` → `[U]`, the format of the released training answers, e.g. `\boxed{DOWN}`), but any other word inside the box contributes its U/D/R/L letters as moves.
- Because the parser only emits characters that are keys of `ACTION_MAP`, the `"Invalid action"` branch of `simulate_vsp` (L142–L147) is unreachable, so the "Invalid" rate logged by `external_models/mirage/repo/src/test.py` (L79, L82, L84) is always 0. A missing `\boxed{}` gives `"None"` → no moves → "Did not reach goal".
- Moves off the grid are ignored (`continue`, L155–L156) instead of failing, and success requires ending on the goal; a path that passes over the goal and leaves fails.
- Only VSP planning has evaluation code. `external_models/mirage/repo/src/test.py` scores every task with `simulate_vsp(map_desc, …)`; COMT, BLINK-Jigsaw, SAT and VSP spatial reasoning (which have no `map_desc`) cannot be evaluated with the released code (`simulate_vsp([])` raises `ValueError` at L137–L138).

---

## 6. Paper statements vs released code and checkpoint

| Item | Paper | Released code / checkpoint |
|---|---|---|
| Compression | "average pooling over the original patch features" (`external_models/mirage/paper_src/sec/3_framework.tex` L52) | mean of K contiguous raster chunks, remainder dropped (§1.2) |
| γ (text-loss weight) | 0.1 (`external_models/mirage/paper_src/sec/4_experiments_arxiv.tex` L26); **10** in the supplement table (`external_models/mirage/paper_src/sec/7_supp.tex` L275) | 0.1 (`external_models/mirage/repo/src/trainer.py` L25) |
| Batch size | 8 (`external_models/mirage/paper_src/sec/4_experiments_arxiv.tex` L25; `external_models/mirage/paper_src/sec/7_supp.tex` L271) | per-device 1, 1 GPU; grad-accum 1 in the released run (`training_args.bin`), 8 / 1 in README stage 1 / stage 2 |
| Gradient accumulation | 2 (`external_models/mirage/paper_src/sec/7_supp.tex` L272) | 1 (released run) |
| LR schedule | cosine (`external_models/mirage/paper_src/sec/4_experiments_arxiv.tex` L25) | `linear` (SFTConfig default; `external_models/mirage/repo/src/main.py` L176–L198 does not set it) |
| Adam β₂ | 0.95 (`external_models/mirage/paper_src/sec/7_supp.tex` L273) | 0.999 (released run) |
| Epochs | 10 (`external_models/mirage/paper_src/sec/7_supp.tex` L274) | 15 (released run; README); 10 in `external_models/mirage/repo/train.sh` |
| VSP SFT samples | 1k (`external_models/mirage/paper_src/sec/4_experiments_arxiv.tex` L14); 3,000 (`external_models/mirage/paper_src/sec/7_supp.tex` L244) | 1,000 lines in `train_direct.jsonl` |
| k-ablation, k = 2 row | Avg 0.86 (`external_models/mirage/paper_src/sec/4_experiments_arxiv.tex` L159) | mean of the four levels is 0.8825 (all other rows' averages are consistent) |
| RL (GRPO, §3.4) | "we explicitly optimize the probabilities of textual tokens while allowing gradients to flow through the latent tokens" (`external_models/mirage/paper_src/sec/3_framework.tex` L105) | no RL code in the repository |
| Evaluation of COMT / Jigsaw / SAT / VSP reasoning | Tables 1–3 | no evaluation code (§5) |
| Sample data | — | README: "a sample dataset of 100 examples"; `external_models/mirage/repo/data/examples/sample.jsonl` has 10 records |
