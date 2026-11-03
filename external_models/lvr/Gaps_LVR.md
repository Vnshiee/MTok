# Technical Review and Forensic Gap Analysis: Latent Visual Reasoning (LVR)

> [!WARNING]
> **Verification status (2026-11-03):** This document has been revised against the primary sources (the official repository [`VincentLeebang/lvr`](external_models/lvr/repo/) at commit `637a9f7`, arXiv:2509.24251v2 LaTeX source, and released checkpoint metadata). Overstated claims from prior versions have been corrected, and findings N1–N9 from [`external_models/lvr/VERIFICATION_REPORT.md`](external_models/lvr/VERIFICATION_REPORT.md) have been incorporated.

**Target Paper**: *Latent Visual Reasoning* (`arXiv:2509.24251v2`, ICLR 2026 Submission)  
**Target Repository**: [`external_models/lvr/repo/`](external_models/lvr/repo/)  
**LaTeX Source**: [`external_models/lvr/paper_src/`](external_models/lvr/paper_src/)  
**Checkpoint Artifacts**: [`external_models/lvr/hf_metadata/LVR-7B/`](external_models/lvr/hf_metadata/LVR-7B/)  

---

## 1. High-Level Claims and Architectural Design Choices

### 1.1 SFT Supervision on Region-of-Interest (ROI) Visual Features
- **Paper Framing**:
  In Section 1 ([`external_models/lvr/paper_src/sections/1_intro.tex` lines 30–42](external_models/lvr/paper_src/sections/1_intro.tex#L30-L42)) and Section 3 ([`external_models/lvr/paper_src/sections/3_lvr.tex` lines 20–24](external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24)), the authors present LVR as a mechanism where an MLLM internally reconstructs query-relevant visual semantics within latent hidden states.
- **Code Reality**:
  In [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` lines 241–244](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L241-L244), the forward pass injects ground-truth ViT embeddings at the `<|lvr|>` token positions:
```python
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
```
  The hidden state at the preceding position (`seq_positions - 1`) is supervised with mean squared error (MSE) loss against `selected_lvr_embeds` in [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` lines 329–334](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L329-L334):
```python
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)
```
- **Analysis**:
  The target patches $\mathbf{v}_t$ are a question-dependent subset of image patch tokens already encoded in the context prefix. While causal self-attention allows attending to prompt tokens, whether this mechanism performs auto-associative memory copying or localization requires empirical testing (see E-L1/E-L3).

### 1.2 Multi-Block Interleaving Structure in Data Pipeline and Generation
- **Code Implementation**:
  In [`external_models/lvr/repo/src/dataset/data_utils.py` lines 45–50](external_models/lvr/repo/src/dataset/data_utils.py#L45-L50), `replace_lvr_tokens` constructs one LVR block for each placeholder in the input string:
```python
            for seg,idxs in zip(input_segments,lvr_token_idxs_list):
                if latent_end_token is not None:    #latent end token mode will append a stopping token as the last
                    replacement = LVR_START_TOKEN + LVR_TOKEN*len(idxs) + LVR_LATENT_END_TOKEN + LVR_END_TOKEN
                else:
                    replacement = LVR_START_TOKEN + LVR_TOKEN*len(idxs) + LVR_END_TOKEN
                output_segments.append(replacement+seg)
```
  During inference in [`external_models/lvr/repo/src/model/qwen_lvr_model.py` lines 1039–1043](external_models/lvr/repo/src/model/qwen_lvr_model.py#L1039-L1043), the generation loop checks for `<|lvr_start|>` at each step and can re-enter LVR mode.
- **Constraints in Practice**:
  In [`external_models/lvr/repo/src/train/reward_funcs.py` lines 53–56](external_models/lvr/repo/src/train/reward_funcs.py#L53-L56), the format reward regex enforces a single sequence structure:
```python
    pattern = r"^<\|lvr_start\|>.*?<\|lvr_end\|>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```
  Because the regex is anchored with `^<|lvr_start|>`, completions that produce initial text reasoning before `<|lvr_start|>` receive a format reward of 0.0. Furthermore, because `re.match` is called without `re.DOTALL`, any newline character within the completion causes the pattern match to return 0.0.

### 1.3 Vision Tower and Multimodal Projector Configuration
- **Implementation**:
  In [`external_models/lvr/repo/scripts/finetune_lvr_stage1_7b.sh` lines 57–58](external_models/lvr/repo/scripts/finetune_lvr_stage1_7b.sh#L57-L58), the vision tower and projector are frozen during Stage 1 SFT:
  `--freeze_vision_tower True` and `--freeze_merger True`.
  This configuration is disclosed in [`external_models/lvr/paper_src/sections/4_experimetns.tex` line 18](external_models/lvr/paper_src/sections/4_experimetns.tex#L18).
- **Positional Embeddings**:
  In [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` lines 269–275](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L269-L275), `get_rope_index` assigns 2D spatial grid positions only to tokens matching `image_token_id`. The `<|lvr|>` tokens are assigned 1D monotonic text sequence coordinates $(t, t, t)$.

---

## 2. Supervised Fine-Tuning Execution

### 2.1 Input Embedding Mismatch Between Training and Inference
- **Training Stage**:
  During SFT ([`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` lines 241–244](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L241-L244)), input embeddings at `<|lvr|>` positions are replaced with clean ground-truth ViT embeddings.
- **Inference Stage**:
  During rollout ([`external_models/lvr/repo/src/train/monkey_patch_forward_lvr_rl.py` lines 129–130](external_models/lvr/repo/src/train/monkey_patch_forward_lvr_rl.py#L129-L130)), the model feeds its own prior hidden state as input:
```python
    if last_position_hidden_state is not None:
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
```
  The model encounters recurrent hidden states at test time rather than ground-truth ViT features.

### 2.2 Sequence Packing and Dynamic Batching in Data Collator
- **Buffer Accumulation**:
  In [`external_models/lvr/repo/src/dataset/lvr_sft_dataset_packed.py` lines 554–583](external_models/lvr/repo/src/dataset/lvr_sft_dataset_packed.py#L554-L583), `PackedDataset` accumulates sequences up to `max_packed_tokens = 16384` (`4 * 4096`).
- **Collator Re-Padding**:
  In [`external_models/lvr/repo/src/dataset/lvr_sft_dataset_packed.py` lines 889–906](external_models/lvr/repo/src/dataset/lvr_sft_dataset_packed.py#L889-L906), the batch collator unpacks sequences and pads each to `max_len`:
```python
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        for i in range(len(all_input_ids)):
            seq_len = len(all_input_ids[i])
            padding_needed = max_len - seq_len

            # Pad on the right side.
            padded_input_ids.append(torch.nn.functional.pad(
                all_input_ids[i], (0, padding_needed), value=self.pad_token_id))
            
            padded_attention_masks.append(torch.nn.functional.pad(
                all_attention_masks[i], (0, padding_needed), value=0))
            
            padded_labels.append(torch.nn.functional.pad(
                all_labels[i], (0, padding_needed), value=IGNORE_INDEX))
```
  FlashAttention-2 unpads padded sequences using `attention_mask` during forward passes, avoiding full quadratic attention over pad tokens.

### 2.3 DeepSpeed ZeRO-3 Dummy Image Workaround
- In [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` lines 169–177](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L169-L177):
```python
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
```
  When a batch item contains text only, DeepSpeed ZeRO-3 requires all ranks to participate in parameter all-gathers for the vision tower. The forward pass runs a dummy zero tensor through the visual module to prevent deadlocks.

---

## 3. Reinforcement Learning Implementation

### 3.1 Gradient Backpropagation in GRPO Latent Training
- **Paper Statement**:
  Section 3.2.2 ([`external_models/lvr/paper_src/sections/3_lvr.tex` lines 97–106](external_models/lvr/paper_src/sections/3_lvr.tex#L97-L106)) notes that latent representations are recorded during generation and replayed during the loss computation.
- **Code Implementation**:
  In [`external_models/lvr/repo/src/trainer/grpo_trainer.py` lines 998–1005](external_models/lvr/repo/src/trainer/grpo_trainer.py#L998-L1005):
```python
            lvr_states, lvr_mask, had_lvr, model_kwargs_after_lvr = self._replay_lvr_to_collect_states(
                model, 
                prompt_ids, 
                prompt_mask, 
                prompt_completion_ids, 
                multimodal_inputs,
                lvr_steps=self.generation_config.lvr_steps
            )
```
The helper is decorated with `@torch.no_grad()` in [`external_models/lvr/repo/src/trainer/grpo_trainer.py` line 1482](external_models/lvr/repo/src/trainer/grpo_trainer.py#L1482).
- **Gradient Flow**:
  The replayed states enter the policy forward pass at lines 1022–1023 without `torch.no_grad()`. Because `freeze_llm=False`, the loss evaluated on answer tokens backpropagates into all language model weights. The gradient path through the recurrent generation of the latents themselves is detached.

### 3.2 Masking of Latent Reasoning Steps in Policy Loss
- In [`external_models/lvr/repo/src/trainer/grpo_trainer.py` lines 746–757](external_models/lvr/repo/src/trainer/grpo_trainer.py#L746-L757):
```python
                tok = prompt_completion_ids[b, t].item()
                if tok == self.model.config.lvr_start_id:
                    active = True
                elif tok == self.model.config.lvr_end_id:
                    active = False
                if active:
                    lvr_mask[b, t] = False  # zero out inside LVR span


        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        final_mask = attention_mask.bool() & lvr_mask  ### << only count valid + non-LVR tokens
```
  Tokens inside the latent reasoning span are assigned `final_mask = False`, excluding them from the policy loss objective.

### 3.3 Comparative Baseline Performance of 3B Variants
- **Commented Baseline Data in LaTeX**:
  In [`external_models/lvr/paper_src/tables/sft_rl.tex` lines 10–11](external_models/lvr/paper_src/tables/sft_rl.tex#L10-L11):
```latex
             % Qwen2.5-VL & 74.87 & 82.61 & 63.16 & 59 & 69.17 & 28.0 & 50.0 & 41.79	 & 81.82 \\
             % SFT & 75.92	& 85.22	 &61.84	& 62.0 & 63.33	&22.67	&50.67	&41.79	&86.01\\
```
- **Comparison**:
  The published Table 2 does not include the base Qwen2.5-VL 3B model.
  Base Qwen2.5-VL 3B outperforms LVR_RL 3B (8 steps) on:
  - $V^*$ Overall: 74.87% vs. 67.02% (-7.85%)
  - MMVP: 59.00% vs. 55.33% (-3.67%)
  - BLINK Counting: 69.17% vs. 55.83% (-13.34%)
  LVR 3B and LVR_RL 3B outperform base Qwen2.5-VL 3B on:
  - BLINK IQ-Test: 30.67% / 32.00% vs. 28.00% (+2.67% / +4.00%)
  - BLINK JigSaw: 52.67% vs. 50.00% (+2.67%)

### 3.4 Unreported 7B Reinforcement Learning Launch Script
- In [`external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh`](external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh), a launch script for 7B RL is configured with DeepSpeed ZeRO-2 and `STAGE1_STEPS=2500`.
- Section 4 ([`external_models/lvr/paper_src/sections/4_experimetns.tex` lines 27–28](external_models/lvr/paper_src/sections/4_experimetns.tex#L27-L28)) states that RL was not applied to the 7B variant due to limited computational resources. The released 7B checkpoint has `stage_2 = False`.

### 3.5 Hyperparameter Discrepancies in RL Scripts
- **Learning Rate**:
  The paper states a learning rate of $1 \times 10^{-5}$ ([`external_models/lvr/paper_src/sections/4_experimetns.tex` line 27](external_models/lvr/paper_src/sections/4_experimetns.tex#L27)). In both [`external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh` line 19](external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh#L19) and `finetune_lvr_stage2_7b.sh` line 19:
  `LR=5e-6`.
- **Image Resolution**:
  In [`external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh` line 51](external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh#L51), image resolution is capped at `--image_max_pixels $((2560 * 28 * 28))`, compared to 5,120 in Stage 1 SFT.

---

## 4. Decoding and Termination Mechanisms

### 4.1 Latent End Token Performance and Hard Step Budget Exit
- **Empirical Performance**:
  In [`external_models/lvr/paper_src/tables/ablation.tex` lines 12–15](external_models/lvr/paper_src/tables/ablation.tex#L12-L15), LVR with Latent End Token drops from 81.7% to 39.8% on $V^*$ Bench and 71.7% to 19.0% on MMVP.
- **Budget Exit in Code**:
  In [`external_models/lvr/repo/src/model/qwen_lvr_model.py` lines 796–804](external_models/lvr/repo/src/model/qwen_lvr_model.py#L796-L804), the function `_lvr_deocding_with_latentend` implements a hard step budget:
```python
                # Force exit if budget exceeded
                lvr_budget_exceeded = lvr_step_counter >= lvr_max_steps

                '''
                    Goal: lvr_mode_switch = lvr_mode_switch + lvr_start_switch - lvr_end_switch
                    Update: exit lvr when lvr_budget_exceeded 

                '''
                # lvr_mode_switch = ((lvr_mode_switch | lvr_start_switch) & (~lvr_end_switch)).to(torch.bool)
                lvr_mode_switch = ((lvr_mode_switch | lvr_start_switch) & (~lvr_end_switch) & (~lvr_budget_exceeded)).to(torch.bool)
```
  The decoder terminates upon reaching `lvr_max_steps`.

### 4.2 Mode Switching Loss Objective Formulation
- In [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` lines 1344–1350](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L1344-L1350):
```python
        lvr_end_logits = logits[..., self.config.lvr_end_id]  # [batch_size, seq_len]

        # Apply mask to focus only on <lvr_start>,<lvr> token positions
        masked_logits = lvr_end_logits[lvr_or_lvrstart_mask]  # [num_lvr_tokens]
        masked_targets = targets[lvr_or_lvrstart_mask]        # [num_lvr_tokens]

        loss_mode_switch = F.binary_cross_entropy_with_logits(masked_logits, masked_targets)
```
  The loss applies binary cross-entropy to `lvr_end_logits` in isolation. At inference time, token emission is determined by softmax argmax over the entire vocabulary (see N6). In the released 7B checkpoint, `mode_switch_loss = False`.

### 4.3 Fixed Step Quota Decoding
- In [`external_models/lvr/repo/src/model/qwen_lvr_model.py` lines 953–954](external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954), the decoding routine enforces fixed quotas:
  `lvr_steps_orig = torch.tensor(lvr_steps, dtype=torch.int, device=self.device)`.
  The evaluation evaluates $K \in \{4, 8, 16\}$.

### 4.4 Generation Appending of Argmax Tokens
- In [`external_models/lvr/repo/src/model/qwen_lvr_model.py` lines 1013–1014](external_models/lvr/repo/src/model/qwen_lvr_model.py#L1013-L1014):
```python
                next_tokens = torch.argmax(next_token_scores, dim=-1)
```
  And in line 1055, `input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)`. During continuous latent steps, discrete tokens are sampled from the language head and appended to `input_ids`.

---

## 5. Data Pipeline and Evaluation Protocols

### 5.1 Dataset Overlap and Visual Near-Duplicate Analysis
- **Benchmark Question Text Overlap**:
  Comparing BLINK Spatial Relation ($N=143$) against Visual-CoT's VSR training subset reveals that 51 of 143 questions (35.7%) share identical text phrasing due to templated synthetic spatial questions.
- **Perceptual Hash Image Analysis**:
  Running difference hashing (dHash) across the matching question pairs yields a minimum Hamming distance of 90/256 (where duplicates typically score $\le 20$). No identical or near-duplicate images are present between the training set and the evaluation split.

### 5.2 Evaluation Answer Parsing Sensitivity
- In [`external_models/lvr/repo/evaluation/evaluation.py` lines 52–61](external_models/lvr/repo/evaluation/evaluation.py#L52-L61):
```python
def accuracy_reward(response: str, ground_truth: str) -> float:
    # content_match = re.search(r"<answer>(.*?)</answer>", response)
    # given_answer = content_match.group(1).strip() if content_match else response.strip()
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) >1:
        given_answer = given_answer[0]
    return given_answer == ground_truth
```
  If `<answer>` tags are absent, `response.split('<answer>')[-1]` defaults to the raw completion. If the string contains multiple characters, `given_answer[0]` extracts the initial character of the string.

### 5.3 Benchmark Evaluation Implementation
- The evaluation suite in [`external_models/lvr/repo/evaluation/evaluation.py`](external_models/lvr/repo/evaluation/evaluation.py) implements standalone evaluation loops directly rather than calling the `lmms-eval` package.

### 5.4 Evaluation Prompt Modifications
- In [`external_models/lvr/repo/evaluation/evaluation.py` lines 63–71](external_models/lvr/repo/evaluation/evaluation.py#L63-L71):
```python
def get_task_instruction(bench_name):
    if bench_name == "vstar":
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "mmvp":
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "blink":
        return "\nAnswer with the option's letter from the given choices directly."
    else:
        raise ValueError(f"Unknown benchmark: {bench_name}")
```
  The script appends explicit single-letter instructions to benchmark prompts.

### 5.5 Multi-Image Evaluation Performance
- On BLINK Relative Reflectance (`paper_src/tables/main_table.tex` lines 74–80), LVR 7B (16 steps) scores 41.8% compared to Vision-R1 (44.8%) and PixelReasoner (42.5%). Section 4.1 ([`external_models/lvr/paper_src/sections/4_experimetns.tex` line 58](external_models/lvr/paper_src/sections/4_experimetns.tex#L58)) notes this gap on multi-image tasks.

---

## 6. Infrastructure and Training Parameter Configuration

### 6.1 Discrepancy Matrix

| Feature / Parameter | Published Paper Text | Training Scripts (`finetune_lvr_*.sh`) | Released `training_args.bin` | Code Implementation | Status / Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 Sequence Length** | Imbalanced instance lengths (`4_experimetns.tex` L21) | `MAX_PACKED_TOKENS = 16384` (`4 * 4096`) | `max_packed_tokens = 16384` | Unpacked and padded in collator | Consistent: 4096 is the `LST` script threshold; paper never states 4,096 packing |
| **Stage 2 RL on 7B** | Not applied to 7B (`4_experimetns.tex` L28) | [`scripts/finetune_lvr_stage2_7b.sh`](external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh) exists | `stage_2 = False` | Functional script present | Script exists; released weights are Stage 1 only |
| **Stage 2 Learning Rate** | $1 \times 10^{-5}$ (`4_experimetns.tex` L27) | `LR=5e-6` (`finetune_lvr_stage2_3b.sh` L19) | N/A | Script configures `5e-6` | Disclosed $1 \times 10^{-5}$ vs. script $5 \times 10^{-6}$ |
| **Stage 2 Max Image Tokens** | 5,120 max pixels | `2560 * 28 * 28` (`finetune_lvr_stage2_3b.sh` L51) | N/A | Script halves resolution | Resolution lowered in RL script |
| **Base Model in Table 2** | Not reported in Table 2 | Base model numbers in LaTeX comments | N/A | Commented in `tables/sft_rl.tex` | Base 3B model outperforms LVR 3B on V* and MMVP |
| **Evaluation Harness** | Mentions LMMs-Eval metrics | Custom script `evaluation.py` | N/A | Standalone parsing logic | Standalone evaluation without `lmms_eval` library |
| **Latent State Replay** | Recorded states patched in | `_replay_lvr_to_collect_states` | N/A | Off-by-one context shift (see N1) | Replay forward diverges 0.20–0.37 nats from rollout |
| **Adaptive Termination** | Latent End / BCE ablated | Both scripts support flags | `mode_switch_loss = False` | Hard budget exit at L796 | Deactivated in released 7B checkpoint |

### 6.2 Credential Redaction in Repository
- Hardcoded Oracle Cloud Infrastructure (OCI) credentials present in upstream files [`scripts/vanilla_sft_7b.sh`](external_models/lvr/repo/scripts/vanilla_sft_7b.sh) (lines 8–12) and [`evaluation/evaluation.py`](external_models/lvr/repo/evaluation/evaluation.py) (lines 17–21) have been redacted in documentation:
```bash
export ACCESS_KEY_ID="<REDACTED_ACCESS_KEY_ID>"
export SECRET_ACCESS_KEY="<REDACTED_SECRET_ACCESS_KEY>"
export ENDPOINT_URL="https://<REDACTED_NAMESPACE>.compat.objectstorage.us-ashburn-1.oraclecloud.com"
export BUCKET_NAME="bangzhengli"
export REGION_NAME="us-ashburn-1"
```

---

## 7. Additional Architectural and Optimization Findings (N1–N9)

### N1. Context Mismatch in GRPO Latent Replay
The paper (`3_lvr.tex` lines 97–106) states that recorded latent states are replayed into latent positions to restore the exact context preceding text generation.
- **Rollout Forward**: In `_lvr_deocding_by_steps`, position $j \in \{s+1, \dots, s+K\}$ receives hidden state $h_{j-1}$, where $s$ is the index of `<|lvr_start|>`.
- **Replay Implementation**: In [`external_models/lvr/repo/src/trainer/grpo_trainer.py` lines 1531–1547](external_models/lvr/repo/src/trainer/grpo_trainer.py#L1531-L1547):
```python
            in_span  = ((pos > lvr_start_idx) & (pos <= configured_end_idx)).to(device=device)

            '''
                We collect hidden states here to avoid off-by-one issues
                the inital step containing lvr_start will be skipped

            '''
            # Create a mask for samples that are in span and not the end token
            is_lvr_token_mask = (prompt_completion_ids[:, pos] != self.model.config.lvr_end_id)
            update_mask = in_span.cpu() & is_lvr_token_mask.cpu()

            # collect hidden state for *this* step if token at pos is an LVR token (not <|lvr_end|>)
            if update_mask.any():
                # Select the relevant hidden states on GPU, move to CPU, then assign
                states_to_store = last_position_hidden_state[update_mask].cpu()
                lvr_states[update_mask, step] = states_to_store
                lvr_mask[update_mask, step] = True
```
  And in [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr_rl.py` lines 135–142](external_models/lvr/repo/src/train/monkey_patch_forward_lvr_rl.py#L135-L142):
```python
    if lvr_states is not None and lvr_mask is not None:
        comp_embeds = inputs_embeds[:, prompt_length:, :]  # (B, C, H)
        comp_embeds = torch.where(
            lvr_mask.unsqueeze(-1),   # (B, C, 1)
            lvr_states,               # (B, C, H)
            comp_embeds               # (B, C, H)
        )
        inputs_embeds = torch.cat([inputs_embeds[:, :prompt_length, :], comp_embeds], dim=1)
```
  Due to the timing of `last_position_hidden_state` updates and `in_span` evaluation, the replayed states are offset by one step, and the embedding at `<|lvr_start|>` is overwritten.
- **Harness Verification**: Testing the repo forward implementation against rollout logits (`verification/lvr_rl_replay_harness.py`) yields a maximum absolute log-probability divergence on loss tokens of:
  - $K=4$: 0.201 nats
  - $K=8$: 0.245 nats
  - $K=16$: 0.302 nats
  With the corrected 2-line indexing alignment, the divergence reduces to $4.77 \times 10^{-7}$.

### N2. Exclusion of `<|lvr_start|>` from Policy Gradient Loss
In [`external_models/lvr/repo/src/trainer/grpo_trainer.py` lines 746–752](external_models/lvr/repo/src/trainer/grpo_trainer.py#L746-L752), when `tok == lvr_start_id`, `active` is set to `True` before `if active: lvr_mask[b, t] = False`. Consequently, `<|lvr_start|>` itself is assigned `lvr_mask = False` and excluded from the loss. The format reward does not directly backpropagate into the generation probability of `<|lvr_start|>`.

### N3. Training vs Inference Latent Length Mismatch
- **Training**: One `<|lvr|>` token is generated per visual patch in the bounding box ([`external_models/lvr/repo/src/dataset/data_utils.py` lines 45–50](external_models/lvr/repo/src/dataset/data_utils.py#L45-L50)). Across all 434,265 Visual-CoT training bboxes:
  - Median $T_v = 28$ tokens
  - Mean $T_v = 66$ tokens
  - 90th percentile $T_v = 143$ tokens
  - Maximum $T_v = 5,096$ tokens
- **Inference**: Truncated to fixed $K \in \{4, 8, 16\}$. For $K=8$, only 21.4% of training samples have $T_v \le 8$. Because ROI patches follow row-major raster order, an 8-step budget covers only the top row(s) of the bounding box.

### N4. Baseline Score Adjustments Between Draft and Published Table 1
In [`external_models/lvr/paper_src/tables/main_table.tex` lines 33–40](external_models/lvr/paper_src/tables/main_table.tex#L33-L40), an earlier draft of Table 1 is preserved in comments:
- Base Qwen2.5-VL 7B on JigSaw changed from 53.33% to 52.0% (-1.33%, equivalent to -2 items out of 150).
- PAPO on IQ-Test changed from 30.0% to 29.3% (-0.70%, equivalent to -1 item out of 150).
- In the draft, base Qwen led LVR 4-step (53.33% vs 52.67%) on JigSaw, and PAPO led LVR 8-step (30.0% vs 29.33%) on IQ-Test. In the published table, LVR is bolded as leading or tied.

### N5. Baseline Answer Parsing Sensitivity on Chain-of-Thought Responses
In Table 1 ([`external_models/lvr/paper_src/tables/main_table.tex` line 76](external_models/lvr/paper_src/tables/main_table.tex#L76)), Vision-R1 scores on binary benchmarks:
- BLINK JigSaw: 27.3% (random chance = 50.0%)
- MMVP: 46.7% (random chance = 50.0%)
- BLINK Spatial Relation: 66.4% (draft was 83.22%)
Because Vision-R1 outputs `<think>...</think>` tags and the parser falls back to string slicing `given_answer[0]` when `<answer>` tags are absent, parsing failures contribute to below-chance accuracy.

### N6. Objective Formulation of Mode-Switch Loss
In [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` lines 1344–1350](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L1344-L1350), `loss_mode_switch` trains a single logit via binary cross-entropy with logits. At inference time, decoding decisions are made by argmax over the entire vocabulary, allowing logit magnitude shifts without altering the winning token.

### N7. Temperature Discrepancy in Policy and Reference Log-Probabilities
In [`external_models/lvr/repo/src/trainer/grpo_trainer.py` line 1334](external_models/lvr/repo/src/trainer/grpo_trainer.py#L1334), `score_with_lvr_replay` uses `apply_temperature = 1.0` by default to score reference and old log-probabilities. In line 1033, policy log-probabilities are scaled by `self.temperature` (0.9 in training scripts), introducing a baseline KL divergence of $\approx 7.1 \times 10^{-4}$ nats/token even when policy equals reference.

### N8. Normalization Scaling in Mean Squared Error Loss
Equation 1 of the paper defines $\mathcal{L}_{\mathrm{LVR}} = \frac{1}{T_v}\sum_{t=1}^{T_v} \|\mathbf{h}_t - \mathbf{v}_t\|_2^2$. The code in [`external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py` line 334](external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py#L334) uses `nn.MSELoss(reduction="mean")`, which normalizes over both tokens and hidden dimension $H=3584$, scaling the effective loss magnitude by $1/3584$ relative to Equation 1.

### N9. Dual Dependency Version Specifications
- [`external_models/lvr/repo/environment.yaml` line 135](external_models/lvr/repo/environment.yaml#L135) specifies `transformers==4.51.3`.
- [`external_models/lvr/repo/requirements.txt` line 99](external_models/lvr/repo/requirements.txt#L99) specifies `transformers==4.54.0`.
- The RL forward implementation utilizes pre-4.52 attribute interfaces (`self.visual`, `_get_initial_cache_position`), requiring compatibility handling across versions.
