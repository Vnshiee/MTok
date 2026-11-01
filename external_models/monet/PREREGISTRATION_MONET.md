# Preregistration: Forensic Audit & Empirical Diagnostics for Monet (`arXiv:2511.21395`)
**Paper Title:** Reasoning in Latent Visual Space Beyond Images and Language (Monet)  
**Authors:** Haoran Hao et al. (CVPR 2026, Paper ID `7696`)  
**Repository:** `https://github.com/NOVAglow646/Monet`  
**Hub Weights:** `NOVAglow646/Monet-7B` (commit `39c1e32`)  
**Auditor:** Forensic Multi-Model Replication Team  
**Preregistration Date:** 2026-11-01  
**Hardware Environment:** NVIDIA A100-PCIE-40GB (GPU 0), CUDA 12.2, Driver 535.183.01  

---

## 1. Executive Summary & Objective

Monet claims to bridge vision and language reasoning through continuous latent visual thoughts generated during autoregressive chain-of-thought generation. Unlike Mull/LVR which emit discrete pseudo-tokens with continuous embedding swaps, Monet trains a dual-stage framework:
1. **Stage 1 & 2 SFT:** Offline representation alignment where latent tokens $\mathbf{z}_t$ are pulled toward image feature subspaces via affine projection $\mathcal{L}_{\text{align-img}}$ and text-observation teacher hidden states via a proxy gradient $\mathcal{L}_{\text{align-obs}}$.
2. **Stage 3 RL (VLPO - Vision-Language Policy Optimization):** Continuous latent policy optimization using an exponential ratio $r_t = \exp(-\frac{1}{2\sigma^2}\|\mathbf{h}_t^{\text{old}} - \mathbf{h}_t^\theta\|^2)$.

The goal of this preregistration is to formulate rigorous, mathematically provable and empirically testable hypotheses regarding:
1. The mathematical integrity of the continuous policy gradient (VLPO).
2. The exact autograd behavior of the detached proxy loss in SFT Stage 2.
3. The empirical necessity of the continuous latent vectors $\mathbf{z}_t$ versus zeros/noise during generation on GPU 0.
4. The role of the hardcoded external quota (`LATENT_SIZE=10`) versus autonomous termination.

---

## 2. Preregistered Hypotheses

### Hypothesis 1 ($H_1$): VLPO Ratio Degeneracy and Absence of Exploratory Policy Gradient
* **Mathematical Claim in Paper:** Section 3.3 defines VLPO for continuous latent tokens as optimizing a surrogate objective $\mathbb{E}[r_{i,t}(\theta) \hat{A}_{i,t}]$ where $r_{i,t}(\theta)$ represents the probability ratio of generating continuous vector $\mathbf{h}_t$ under policy $\pi_\theta$ versus reference $\pi_{\theta_{\text{old}}}$.
* **Code Implementation (`RL/verl/workers/actor/dp_actor.py` L135–158):**
  $$r_{i,t}(\theta) = \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{h}_{i,t}^{\text{old}} - \mathbf{h}_{i,t}^\theta\|_2^2\right)$$
  where $\sigma=1.0$ is hardcoded, and the rollout is generated deterministically via greedy argmax / teacher forced forward pass.
* **Preregistered Predictions:**
  1. **Strict Upper Bound:** Since $\|\mathbf{h}^{\text{old}} - \mathbf{h}^\theta\|_2^2 \ge 0$, the ratio is strictly bounded:
     $$\forall \theta, \quad 0 < r_{i,t}(\theta) \le 1.0$$
     Unlike true PPO/GRPO importance sampling ratios which satisfy $\mathbb{E}_{\pi_{\text{old}}}[r] \approx 1$ and can take values $> 1$ to encourage probability increase on positive-advantage actions, $r_{i,t}(\theta)$ can **never** exceed $1.0$.
  2. **Zero-Reward / Positive-Advantage Inversion:**
     When advantage $\hat{A}_{i,t} > 0$, the surrogate loss $-\mathbb{E}[r_{i,t}\hat{A}_{i,t}]$ minimizes when $r_{i,t} \to 1.0$, which implies $\|\mathbf{h}^{\text{old}} - \mathbf{h}^\theta\|_2^2 \to 0$. Thus, a positive reward *penalizes* any deviation from the old policy!
  3. **Collapse to L2 Regularization:**
     $$\nabla_\theta \left(r_{i,t} \hat{A}_{i,t}\right) = -\frac{\hat{A}_{i,t}}{\sigma^2} r_{i,t} (\mathbf{h}^\theta - \mathbf{h}^{\text{old}}) \nabla_\theta \mathbf{h}^\theta$$
     For small drift $(\mathbf{h}^\theta \approx \mathbf{h}^{\text{old}})$, $r_{i,t} \approx 1$, so the gradient is exactly equivalent to an L2 weight decay / anchor pull toward $\mathbf{h}^{\text{old}}$, scaled by $-\hat{A}$. It does not perform density estimation or exploration in continuous space.

---

### Hypothesis 2 ($H_2$): Detached Proxy-Loss Zeroes Direct Observation Gradients
* **Mathematical Claim in Paper:** Section 3.2 states that latent vectors are aligned with subsequent observation tokens via $\mathcal{L}_{\text{align-obs}}$, creating bidirectional grounding between latent visualization and text reasoning.
* **Code Implementation (`src/trainer.py` L11–42, L198–203):**
  The author defines:
  ```python
  g = torch.autograd.grad(l_obs, z, retain_graph=True)[0]
  loss = (z * g.detach()).sum()
  ```
* **Preregistered Predictions:**
  1. Since `g.detach()` severs the backward graph from $g$ back to the observation hidden states and cross-attention maps, $\frac{\partial \text{loss}}{\partial \mathbf{h}_{\text{obs}}} \equiv \mathbf{0}$.
  2. The observation tokens themselves receive **zero** direct gradient from this loss term. The loss acts only as a first-order directional projection on $\mathbf{z}$ along the frozen directional derivative $g$.

---

### Hypothesis 3 ($H_3$): Generation Dynamics and External Quota Dependency
* **Paper vs Code Reality:**
  The paper implies the model dynamically decides how long to reason in latent space.
  However, in `inference/load_and_gen_vllm.py` and `inference/vllm_inference_example.py`:
  Generation is driven by an external shell variable:
  ```bash
  export LATENT_SIZE=10
  ```
  The model never generates an autonomous end-of-thought delimiter `</abs_vis_token>` to stop; rather, the outer Python driver repeatedly injects `<abs_vis_token>` exactly `LATENT_SIZE` times.
* **Preregistered Predictions:**
  1. Under `LATENT_SIZE=0`, Monet behaves as a standard text-CoT model.
  2. The cosine similarity between successive latent states $\mathbf{z}_t, \mathbf{z}_{t+1}$ during rollout will show high collinearity ($\cos(\mathbf{z}_t, \mathbf{z}_{t+1}) > 0.85$), indicating slow representation drift within an anchor manifold rather than distinct multi-step visual reasoning hops.

---

### Hypothesis 4 ($H_4$): Causal Sensitivity of Continuous Latents to Intervention
* **Intervention Protocol:**
  On benchmark visual reasoning probe samples:
  1. **Baseline ($C_0$):** Standard autoregressive execution with full continuous latent propagation ($\mathbf{z}_1 \dots \mathbf{z}_{10}$).
  2. **Zero Vector ($C_{\text{zero}}$):** Replace each continuous latent $\mathbf{z}_t$ with all-zero tensor $\mathbf{0}$.
  3. **Gaussian Noise ($C_{\text{noise}}$):** Replace each continuous latent $\mathbf{z}_t$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\mu}_\mathbf{z}, \boldsymbol{\sigma}_\mathbf{z}^2)$.
  4. **Donor Swapping ($C_{\text{donor}}$):** Replace $\mathbf{z}_t$ with latents extracted from an unrelated image/prompt.
* **Preregistered Metric:**
  Compare token generation probabilities, answer parity ($P(\text{Ans}(C_{\text{intervention}}) == \text{Ans}(C_0))$), and output token overlap.

---

## 3. Empirical Test Plan on GPU 0

1. **Step 3.1: Autograd & Mathematical Diagnostic (`verify_monet_math_autograd.py`):**
   * Execute synthetic forward-backward pass verifying $H_1$ ($r_t \le 1.0$ bound and gradient direction) and $H_2$ (`g.detach()` gradient isolation).
2. **Step 3.2: Weights Download & Architecture Verification (`download_monet_weights.py`):**
   * Download `NOVAglow646/Monet-7B` safetensors to HuggingFace Hub cache.
   * Verify layer weights, vocabulary size (151,670), and custom token embeddings.
3. **Step 3.3: GPU Empirical Diagnostic (`eval_monet_empirical.py`):**
   * Load `NOVAglow646/Monet-7B` in `bfloat16` on GPU 0.
   * Run inference on probe samples from $V^*$ Bench / MMVP.
   * Log latent trajectory statistics: norm $\|\mathbf{z}_t\|_2$, pairwise cosine similarities, and intervention outputs.
4. **Step 3.4: Synthesis & Reporting (`MONET_EMPIRICAL_REPORT.md`):**
   * Tabulate all numerical outputs and verify or falsify $H_1$–$H_4$.
