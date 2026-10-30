# Pre-Registration: Mechanistic Adjudication of Latent Visual Reasoning (LVR)

**Target Model**: LVR (`vincentleebang/LVR-7B`, `arXiv:2509.24251`)  
**Backbone**: `Qwen2.5-VL-7B-Instruct`  
**Theoretical Target**: Mechanistic Disentanglement of Latent Visual Reconstruction vs. Attention Copying & Evaluation Bias

---

## 1. Primary Hypotheses

### Hypothesis H1: The Tautological ROI Copying Shortcut
- *Core Claim*: Latent states supervised with MSE loss against ground-truth ViT ROI patches do not compute novel spatial transformations; they function as identity copy vectors of patch embeddings already present in the prompt prefix.
- *Test*: Cosine similarity and attention ablation between latent tokens and preceding visual tokens. If H1 holds, cosine similarity exceeds 0.92, and zeroing out self-attention to the image patches causes latent rollout MSE to collapse.

### Hypothesis H2: Disconnected Policy Gradient Optimization
- *Core Claim*: The reported GRPO gains are purely language format and answer-token recalibration; latent generation steps receive zero policy gradient due to span masking (`final_mask = 0`) and detached replay tensors (`@torch.no_grad()`).
- *Test*: Gradient backpropagation trace during RL training pass confirms $\frac{\partial \mathcal{J}_{\mathrm{GRPO}}}{\partial \mathbf{z}_{\mathrm{latent}}} \equiv \mathbf{0}$.

### Hypothesis H3: Brittle Step-Quota Dependence
- *Core Claim*: Adaptive termination mechanisms (Latent End Token) collapse accuracy by >40%, forcing the model to rely entirely on fixed step quotas (4, 8, 16) that lead to repetitive vector generation.

---

## 2. Experimental Diagnostic Protocols
All diagnostic scripts run deterministically on CPU under `CUDA_VISIBLE_DEVICES=""`.
- `diagnostics/test_roi_copying.py`: Mathematical verification of identity copying under MSE supervision.
- `diagnostics/test_grpo_zero_grad.py`: Verifies zero loss assignment across the LVR reasoning span.
- `diagnostics/test_adaptive_exit_entropy.py`: Evaluates transition token logit entropy across rollout depth.
