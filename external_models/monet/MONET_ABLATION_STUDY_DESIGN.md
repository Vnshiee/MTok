# Monet (CVPR 2026): Rigorous Ablation Study Design & Adversarial Verification Protocol

**Target Paper**: *Reasoning in Latent Visual Space Beyond Images and Language*  
**Conference / Status**: CVPR 2026 (Paper ID `7696`, arXiv: `2511.21395`)  
**Backbone Architecture**: `NOVAglow646/Monet-7B` (8.29B parameters, base `Qwen2.5-VL-7B`)  
**Evaluation Hardware**: Dedicated NVIDIA A100-PCIE-40GB GPU, CUDA 12.2, PyTorch 2.5.1+cu121  
**Execution Environment**: Strict `tmux` session `monet_pipeline` on GPU 0  
**Design Author**: Antigravity Autonomous Research Agent  
**Date**: November 1, 2026  

---

## 1. Executive Protocol & Design Objectives

Following the multi-model forensic audit protocol established for the MTok research program, this document formalizes the complete experimental design for the ablation study of **Monet**. 

In Step 3 empirical diagnostics (`MONET_EMPIRICAL_REPORT.md`), we established three foundational empirical facts:
1. Special tokens (`<abs_vis_token>`, `</abs_vis_token>`, `<observation>`, `</observation>`) possess near-zero L2 norms ($\le 0.007$ vs. text $0.88$), proving they serve solely as structural control switches rather than lexical embeddings.
2. In the probe sample test, zeroing out continuous latents ($z_k = \mathbf{0}$) or injecting Gaussian noise completely flipped the reasoning decision (from *"a road"* to *"the bus"*), proving that continuous latents are **causally active**.
3. Autograd verification confirmed that the continuous policy gradient (VLPO) suffers from dimensional underflow ($r_{i,t} \equiv 0.0$ when coordinate drift $> 0.22$ in $\mathbb{R}^{3584}$), and the Stage 2 proxy loss completely decouples observation weights from the vision encoder via `.detach()`.

The objective of this Ablation Study is to systematically quantify Monet's performance across four rigorous axes:
- **Axis 1: Causal Intervention & Latent Ablation**: Does Monet's visual reasoning truly depend on continuous latents, or can equivalent performance be obtained with zeroed tokens, isotropic Gaussian noise, or unrelated donor latents?
- **Axis 2: Autograd & Loss Objective Dynamics**: What is the mathematical and empirical impact of the VLPO ratio underflow failure mode, and does feature-dimension normalization ($\frac{1}{D}$) restore policy gradient stability?
- **Axis 3: Latent Budget / Thinking Quota Scaling**: How does downstream reasoning accuracy scale as the latent thinking budget $K$ varies across $\{0, 1, 3, 5, 10, 20\}$? Does performance saturate, collapse, or monotonically improve?
- **Axis 4: Direct Benchmark Replication & Baseline Comparison**: How does `Monet-7B` compare directly against its un-finetuned parent model `Qwen2.5-VL-7B-Instruct` across standard vision-language benchmarks:
  - **V\* Bench (Direct Attributes)**: $N = 191$ visual search and high-resolution attribute discovery questions.
  - **MMVP (Multimodal Visual Patterns)**: $N = 300$ CLIP-blind visual reasoning paired challenges.

---

## 2. Experimental Axis Breakdown

```
+---------------------------------------------------------------------------------------------------+
|                                MONET ABLATION STUDY DESIGN ARCHITECTURE                           |
+---------------------------------------------------------------------------------------------------+
                                                  |
         +--------------------+-------------------+--------------------+--------------------+
         |                    |                                        |                    |
         v                    v                                        v                    v
  [AXIS 1: CAUSAL]     [AXIS 2: OBJECTIVE]                      [AXIS 3: QUOTA]      [AXIS 4: BENCHMARK]
  - C0: Natural        - Autograd Trace of VLPO Ratio           - K = 0 (No Latent)  - V* Bench (N=191)
  - C1: Zero Latents   - Dimensional Underflow (D=3584)         - K = 1 (Minimal)    - MMVP (N=300)
  - C2: Noise Injection- Normalized VLPO vs Unnormalized        - K = 3 (Short CoT)  - Monet-7B vs
  - C3: Donor Swapping - Detached Proxy Alignment Gradient      - K = 5 (Medium CoT)   Qwen2.5-VL-7B-Instruct
                                                                - K = 10 (Paper Def)
                                                                - K = 20 (Extended)
```

### 2.1 Axis 1: Causal Intervention & Latent Perturbation
We isolate the information content transmitted through the continuous latent channel during autoregressive generation. For every test prompt:
1. **Condition C0 (Natural Latent Rollout)**: The model emits `<abs_vis_token>`, followed by $K-1$ autoregressive forward passes where each step's output hidden state $\mathbf{h}_{t}$ is fed directly into the input embeddings of step $t+1$, closed by `</abs_vis_token>`.
2. **Condition C1 (Zero-Vector Ablation)**: At every latent step $k \in \{1, \dots, K-1\}$, the input embedding is replaced with the zero vector $\mathbf{0} \in \mathbb{R}^{3584}$.
3. **Condition C2 (Gaussian Noise Ablation)**: At every step, the input embedding is replaced with an isotropic Gaussian sample matching the empirical coordinate mean and variance of natural latents: $\mathbf{z}_k \sim \mathcal{N}(\mu_{\mathbf{z}}, \sigma_{\mathbf{z}}^2 \mathbf{I})$.
4. **Condition C3 (Donor Swapping Ablation)**: At every step, the input embedding is replaced with latent vectors precomputed from an unrelated, random donor image from a completely different category.

**Hypothesis**: If Monet has learned a genuine latent reasoning process, downstream benchmark accuracy under C1, C2, and C3 must exhibit a statistically significant drop relative to C0 ($\Delta \text{Acc} > 5\%$). If performance does not drop, the latent representations are inert and the model's performance stems solely from prompt scaffolding.

---

### 2.2 Axis 2: Autograd & Loss Objective Dynamics
We evaluate the structural flaws uncovered in Monet's training objectives:

1. **Continuous Policy Gradient (VLPO) Ratio Collapse**:
   $$r_{i,t}(\theta) = \exp\left( - \frac{\|\mathbf{h}_{i,t}^\theta - \mathbf{h}_{i,t}^{\text{old}}\|^2}{2\sigma^2} \right)$$
   - We evaluate the empirical distribution of $r_{i,t}$ across parameter updates.
   - We compare standard unnormalized VLPO against the dimension-normalized variant:
     $$\widetilde{r}_{i,t}(\theta) = \exp\left( - \frac{\|\mathbf{h}_{i,t}^\theta - \mathbf{h}_{i,t}^{\text{old}}\|^2}{2 D \sigma^2} \right), \quad D = 3584$$
   - We verify whether $\widetilde{r}_{i,t}$ restores gradient survival across drift regimes $\overline{\delta} \in [0.1, 1.0]$.
2. **Observation Detachment in Stage 2 SFT**:
   - We verify whether the detached proxy loss $\mathbf{g}_j = \nabla_{\mathbf{z}_j} \mathcal{L}_{\text{align-obs}}$ prevents end-to-end representation alignment between the vision tower and language model.

---

### 2.3 Axis 3: Latent Budget / Thinking Quota Scaling ($K$)
Monet enforces a hardcoded quota via `LATENT_SIZE = 10` in `monet_gpu_model_runner.py`. We test the hypothesis that reasoning capacity is strictly contingent on this hyperparameter:
- Quotas evaluated: $K \in \{0, 1, 3, 5, 10, 20\}$.
- For each quota $K$:
  - Measure accuracy on V\* Bench and MMVP.
  - Measure output token length and repetition rate.
  - Measure average trajectory cosine similarity $\overline{\cos}(\mathbf{z}_{t}, \mathbf{z}_{t-1})$ and SVD subspace concentration.
- **Expected Outcome**:
  - $K=0$: Base direct answering without latent visual thinking.
  - $K=1$: Prone to degenerate 1-word collapse as observed in empirical diagnostics.
  - $K=10$: Optimal paper baseline.
  - $K=20$: Test whether latents continue to refine or saturate due to asymptotic convergence ($\cos \to 0.9995$).

---

### 2.4 Axis 4: Downstream Benchmark Accuracy & Baseline Comparison
To assess whether Monet delivers genuine visual reasoning improvements, we benchmark `NOVAglow646/Monet-7B` against its direct un-finetuned parent model `Qwen2.5-VL-7B-Instruct` across two challenging benchmarks:

1. **V\* Bench (Visual Search / Direct Attributes)** ($N = 191$):
   - Benchmark evaluating high-resolution detail discovery and spatial object attribute identification.
   - Cached locally at: `~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/`
   - Evaluation Metric: Exact Match (EM) accuracy on target attributes and final boxed predictions (`\boxed{}`).
2. **MMVP (Multimodal Visual Patterns)** ($N = 300$):
   - Benchmark evaluating visual perception questions specifically designed to test visual patterns where standard VLMs fail due to CLIP visual shortcuts.
   - Cached locally at: `external_models/monet/benchmarks/mmvp/` or downloaded via HuggingFace.
   - Evaluation Metric: Pairwise accuracy and individual question accuracy.

---

## 3. Detailed Experimental Execution Matrix

| Experiment ID | Evaluation Benchmark | Model Backbone | Latent Quota ($K$) | Latent Condition | Target Metric | Purpose |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-1.0** | V\* Bench ($N=191$) | `Monet-7B` | $K=10$ | Natural (C0) | EM Accuracy (%) | **Monet Baseline Replication** |
| **EXP-1.1** | V\* Bench ($N=191$) | `Monet-7B` | $K=10$ | Zero Vector (C1) | EM Accuracy (%) | Causal Latent Ablation (Zeros) |
| **EXP-1.2** | V\* Bench ($N=191$) | `Monet-7B` | $K=10$ | Gaussian Noise (C2) | EM Accuracy (%) | Causal Latent Ablation (Noise) |
| **EXP-1.3** | V\* Bench ($N=191$) | `Monet-7B` | $K=10$ | Donor Swapping (C3) | EM Accuracy (%) | Causal Latent Ablation (Donor) |
| **EXP-2.0** | V\* Bench ($N=191$) | `Monet-7B` | $K=0$ | None ($K=0$) | EM Accuracy (%) | Quota Sweep: Zero Thinking |
| **EXP-2.1** | V\* Bench ($N=191$) | `Monet-7B` | $K=1$ | Natural | EM Accuracy (%) | Quota Sweep: Minimal Budget |
| **EXP-2.2** | V\* Bench ($N=191$) | `Monet-7B` | $K=3$ | Natural | EM Accuracy (%) | Quota Sweep: Low Budget |
| **EXP-2.3** | V\* Bench ($N=191$) | `Monet-7B` | $K=5$ | Natural | EM Accuracy (%) | Quota Sweep: Medium Budget |
| **EXP-3.0** | V\* Bench ($N=191$) | `Qwen2.5-VL-7B-Instruct` | — | Text Only (CoT) | EM Accuracy (%) | Parent Model Comparative Baseline |
| **EXP-4.0** | MMVP ($N=300$) | `Monet-7B` | $K=10$ | Natural (C0) | Pairwise Acc (%) | Monet Baseline on Perception |
| **EXP-4.1** | MMVP ($N=300$) | `Monet-7B` | $K=10$ | Zero Vector (C1) | Pairwise Acc (%) | Causal Latent Ablation on MMVP |
| **EXP-4.2** | MMVP ($N=300$) | `Qwen2.5-VL-7B-Instruct` | — | Text Only (CoT) | Pairwise Acc (%) | Parent Model Baseline on MMVP |

---

## 4. Adversarial Verification & Failure Mode Guardrails

To prevent metric corruption, data leakage, or deceptive artifacts, the experimental runner must enforce the following guardrails:

1. **Strict Caching Isolation**:
   - `DynamicCache` in PyTorch modifies KV cache tensors in place.
   - *Guardrail*: For every test query and intervention condition, `past_key_values` must be freshly created from prefill. No cached tensors may be shared across conditions $C_0, C_1, C_2, C_3$.
2. **Determinism & Greediness ($T = 0$)**:
   - All evaluations must run under greedy decoding ($T = 0$, `do_sample=False`) to eliminate stochastic jitter.
   - Differences in downstream accuracy between $C_0$ and $C_1$ must be $100\%$ attributable to the latent vector values.
3. **Strict Answer Normalization & Parsing**:
   - Standard regex extraction targeting `\boxed{...}`, followed by lowercase normalization and string stripping.
   - If `\boxed{}` is missing, fallback regex extracts the final answer phrase according to benchmark evaluation conventions.
4. **Execution Inside Dedicated tmux Session**:
   - All benchmark processes run inside `tmux` session `monet_pipeline` on GPU 0.
   - Outputs are continuously logged to JSON and plain text logs to ensure zero progress loss during long runs.

---

## 5. Deliverables & Acceptance Criteria

Step 4 will be certified complete when:
1. `ablation_study/` evaluation scripts are fully implemented, verified, and checked against failure modes.
2. The benchmark evaluation suite completes across all planned experiments in `monet_pipeline`.
3. Complete telemetry and benchmark predictions are serialized to JSON in `external_models/monet/ablation_study/outputs/`.
4. The final [`MONET_ABLATION_RESULTS_REPORT.md`](file:///home/devraj/MTok/external_models/monet/MONET_ABLATION_RESULTS_REPORT.md) is compiled with exact numbers, tables, and theoretical interpretation.
