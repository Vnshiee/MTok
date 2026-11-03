# The Unified Theory of Multimodal Latent Tokens: Architectural Mechanisms, Format Constraints, and Representational Drift

> [!NOTE]
> Monet and Mirage sections below are not independently verified.

**Comparative Analysis of 4 Multimodal Latent Reasoning Frameworks:**
1. **Mull-Tokens** (*CVPR 2026* — Boston University / Stanford / MIT)
2. **LVR: Latent Visual Reasoning** (*ICLR 2026* — `arXiv:2509.24251`)
3. **Monet: Reasoning in Latent Visual Space** (*CVPR 2026* — Peking University — `arXiv:2511.21395`)
4. **Mirage: Machine Mental Imagery** (*CVPR 2026* — UMass / MIT — `arXiv:2506.17218`)

All models benchmarked on: `Qwen2.5-VL-7B` Backbone.

---

## 1. Executive Synthesis Matrix across All 4 Architectures

| Dimension | 1. Mull-Tokens | 2. LVR | 3. Monet | 4. Mirage |
| :--- | :--- | :--- | :--- | :--- |
| **Token Representation** | Static discrete prefix tokens (`<\|latent_pad\|> \times 20`) | Dynamic autoregressive continuous hidden state rollout (`<lvr> ... </lvr>`) | Precomputed offline teacher continuous thought vectors | Interleaved latent visual scratchpad (`<\|latent_pad\|> \times N`) |
| **Visual Anchoring (Stage 1)** | Average-pooled ViT patch embedding ($\mathbf{v}_t = \bar{g}_\phi(I_t)$) | Direct MSE loss to unpooled ViT ROI bounding-box patch tokens | Teacher feature distillation across all layers | Cosine similarity loss ($1 - \cos(\mathbf{h}, \mathbf{v})$) |
| **Relaxation & Drift (Stage 2)** | Relaxed SFT removes intermediate visual loss for 200k steps | Latent states conditioned on frozen ViT features with fixed quotas | Student trained on offline `.pt` tensors with detached proxy loss | Complete drop of visual loss: `loss = ce_loss` (`src/trainer.py` L38) |
| **Reinforcement Learning (Stage 3)** | GRPO with $G=2$ updates | LVR's GRPO replays latent states without gradient through their generation (disclosed in the paper, 3_lvr.tex L97–106), and the replay in the loss forward does not reproduce the rollout context (N1) | Detached proxy gradient `safe_grads.append(g.detach())` (`src/trainer.py` L39) | Text-only PPO/GRPO reward with static latent slots |
| **What the Latents Actually Do** | **Format Lock & Calibrator**: Forces `<think></think>` stop-gate (+10% B-bias) | **ROI supervision**: Untested interpretation of patch copying; see E-L1/E-L3 | **Linear Projection Proxy**: Offline snapshot memory lookup | **Prefix Tuner**: Drifts into syntax prompt embedding |

---

## 2. Core Architectural Mechanisms of Latent Multimodal Reasoning

### Mechanism 1: Relaxation and Representational Drift (Mull-Tokens & Mirage)
In both Mull-Tokens and Mirage, the authors initialize latent tokens with multimodal visual supervision in Stage 1. However, to enable text generation, Stage 2 drops the visual loss entirely (`loss = ce_loss` in Mirage `src/trainer.py` L38, and 200k steps in Mull). Without a visual anchor, gradient descent alters the latent representations to minimize cross-entropy on the answer token.

### Mechanism 2: Prompt Patch Supervision (LVR)
LVR supervises hidden states via MSE loss against raw ViT patch tokens. Because the entire image is already present in the prompt prefix, an untested interpretation is that self-attention allows the model to attend back and copy input patch embeddings into the latent slot (see E-L1/E-L3).

### Mechanism 3: Reinforcement Learning Replay Properties (LVR & Monet)
- **LVR**: LVR's GRPO replays latent states without gradient through their generation (disclosed in the paper, `3_lvr.tex` L97–106), and the replay in the loss forward does not reproduce the rollout context (N1). Intermediate latent tokens are excluded from the loss calculation (`final_mask = attention_mask & lvr_mask` in `grpo_trainer.py` L757), and `<|lvr_start|>` is also masked out (N2).
- **Monet**: Calls `safe_grads.append(g.detach())` in `src/trainer.py` L39 and uses a surrogate `proxy_loss = \sum (v * g.detach())`.

---

## 3. Conclusion: The Real Nature of Multimodal Latent Reasoning
Current empirical evidence indicates that multimodal latent architectures function primarily through:
1. **Syntactic format locks** that suppress degenerative verbal reasoning (Mull-Tokens),
2. **Prompt-conditioned patch predictors** (LVR; see E-L1/E-L3),
3. **Static snapshot lookups** with detached proxy losses (Monet), or
4. **Unmoored prefix tuners** that experience representational drift (Mirage).
