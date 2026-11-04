# The Unified Theory of Multimodal Latent Tokens: Architectural Mechanisms, Format Constraints, and Representational Drift

**Comparative Analysis of 4 Multimodal Latent Reasoning Frameworks:**
1. **Mull-Tokens** (*CVPR 2026* — Boston University / Stanford / MIT)
2. **LVR: Latent Visual Reasoning** (*ICLR 2026* — `arXiv:2509.24251`)
3. **Monet: Reasoning in Latent Visual Space** (*CVPR 2026* — Peking University — `arXiv:2511.21395`)
4. **Mirage: Machine Mental Imagery** (*CVPR 2026* — UMass / MIT — `arXiv:2506.17218`)

All models benchmarked on the identical base architecture: `Qwen2.5-VL-7B` on dedicated NVIDIA A100-PCIE-40GB hardware.

---

## 1. Executive Synthesis Matrix across All 4 Architectures

| Dimension | 1. Mull-Tokens | 2. LVR | 3. Monet | 4. Mirage |
| :--- | :--- | :--- | :--- | :--- |
| **Token Representation** | Static discrete prefix tokens (`<\|latent_pad\|> \times 20`) | Dynamic autoregressive continuous hidden state rollout (`<lvr> ... </lvr>`) | Autoregressively generated continuous latents (last hidden state fed back as next input) | Interleaved continuous latent visual scratchpad (`<\|latent_pad\|> \times K`) |
| **Visual Anchoring (Stage 1)** | Average-pooled ViT patch embedding ($\mathbf{v}_t = \bar{g}_\phi(I_t)$) | Direct MSE loss to unpooled ViT ROI bounding-box patch tokens | Teacher feature distillation across all layers | Cosine similarity loss ($1 - \cos(\mathbf{h}, \mathbf{v})$) |
| **Relaxation & Drift (Stage 2)** | Relaxed SFT removes intermediate visual loss for 200k steps | Latent states conditioned on frozen ViT features with fixed quotas | Student trained on offline `.pt` tensors with detached proxy loss | Complete drop of visual loss: `loss = ce_loss` (`src/trainer.py` L38) |
| **Reinforcement Learning (Stage 3)** | GRPO with $G=2$ updates | Replays latent states without gradient through generation (`3_lvr.tex` L97–106); masked loss | Continuous VLPO with Gaussian log-prob; disabled in released code | Text-only PPO/GRPO reward with static latent slots |
| **Causal Agency (Ablations)** | **Inert Delimiter**: Random noise/zeros yield identical accuracy ($p > 0.99$) | **Inert Delimiter**: Zero/noise interventions leave accuracy unchanged | **Causally Active**: Zeros drop accuracy by $-6.28\%$; noise drops $-4.19\%$ | **Inert Delimiter**: Gaussian noise/donors yield identical accuracy ($\pm 0.25\%$, $p > 0.99$) |
| **Thinking Quota ($K=0$ vs. Target)** | $K=0$ matches $K=20$ (+0.10%) | $K=0$ matches $K=4$ ($-0.32\%$) | **$K=0$ outperforms $K=10$ (+1.05%)** | **$K=0$ matches $K=4$ (74.75% vs. 75.50%, $p > 0.50$)** |
| **Semantic Donor Transfer** | 0.0% Adoption | 0.0% Adoption | Distractor effect | **0.00% Adoption on VSP & V\* ($\ge 98.5\%$ retention)** |
| **What the Latents Actually Do** | **Format Lock & Calibrator**: Forces `<think>` stop-gate | **Prompt-Patch Copying**: Repackages input ViT tokens | **Contextual Bias / Over-smoother**: Low-pass filter | **Delimiter & Compute Anchor**: SFT format calibration |

---

## 2. Core Architectural Mechanisms Revealed by GPU Falsification

### Mechanism 1: Relaxation and Representational Drift (Mull-Tokens & Mirage)
In both Mull-Tokens and Mirage, the authors initialize latent tokens with multimodal visual supervision in Stage 1. However, to enable downstream text generation, Stage 2 drops the visual alignment loss entirely (`loss = ce_loss` in Mirage `src/trainer.py` L38, and 200k steps in Mull-Tokens). 

**Empirical Reality**:
Without continuous visual anchoring, backpropagation from the target answer tokens rapidly washes out the initial visual alignment. On Mirage, empirical measurement of the cosine similarity between generated latents and the prompt image tokens versus distractor image tokens reveals an alignment differential of only $+0.0002$ to $+0.0005$ across 4 benchmarks (95% bootstrap CIs span zero). The latent states experience complete representational drift, functioning as unmoored deliberation delimiters rather than mental imagery.

### Mechanism 2: Prompt Patch Supervision (LVR)
LVR supervises hidden states via MSE loss against raw ViT patch tokens. Because the entire image is already encoded in the multimodal prefix:
1. Self-attention layers allow the model to attend back to the prefix image tokens and copy input patch embeddings directly into the latent slots.
2. Intervening on the latent representations does not impair performance because the decoder attends back directly to the original image tokens in the prompt prefix.

### Mechanism 3: Over-Smoothing Recurrent Dynamics (Monet)
Monet implements true continuous latent recurrent feedback, making its latents causally active (zeroing them drops accuracy by $-6.28\%$). However:
1. The recurrent loop converges to an asymptotic attractor by step 9 ($\cos = 0.9995$, with $97.36\%$ variance in the top-3 singular vectors).
2. Rather than discovering new visual features, this recurrence acts as an over-smoothing low-pass filter. Consequently, direct answering without any latent thinking ($K=0$) achieves **$84.82\%$** accuracy on $V^*$ Bench, surpassing the full 10-step latent thinking budget (**$83.77\%$**).

### Mechanism 4: The SFT Format Lock Illusion (All 4 Models)
A universal finding across all four papers is that the dramatic performance delta between the "un-finetuned base model" and the "latent model" is an artifact of output formatting rather than latent reasoning:
- On VSP Spatial Planning (Mirage), the base model achieves only **8.25%** accuracy solely because it outputs unconstrained natural language rather than formatted coordinate tuples `[x, y, ...]`.
- Once fine-tuning teaches the model the required syntax, stripping all latent tokens entirely ($K=0$) delivers **74.75%** accuracy—statistically identical to the 4-latent model (**75.50%**).
- Similarly, on Mull-Tokens, $K=0$ matches $K=20$ within $0.1\%$, and substituting continuous Gaussian noise reproduces baseline accuracy bit-for-bit.

---

## 3. Definitive Conclusion: The True Nature of Multimodal Latent Reasoning

Across four independent architectures evaluated under rigorous identical causal interventions on NVIDIA A100 GPUs, continuous and discrete multimodal latent tokens do not execute autonomous "latent reasoning", "mental simulation", or "visual chain-of-thought".

Instead, they function through three distinct non-reasoning roles:
1. **Syntactic and format delimiters** that enforce structural compliance and suppress conversational degeneration (Mull-Tokens, Mirage).
2. **Redundant prompt-patch copying mechanisms** that echo already-present visual tokens (LVR).
3. **Contextual bias energy** that acts as an over-smoothing recurrent filter without expanding problem-solving capability over standard language CoT (Monet).
