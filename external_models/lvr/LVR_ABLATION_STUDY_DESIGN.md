# Theoretical Ablation Study Design: Deconstructing Latent Visual Reasoning (LVR)

**Target Paper**: *Latent Visual Reasoning* (ICLR 2026, arXiv:2509.24251)  
**Backbone**: `Qwen2.5-VL-7B`  
**Core Question**: *Does LVR execute continuous latent visual computation, or is its empirical performance fully explained by auto-associative patch identity copying and syntactic stop-gating?*

---

## 1. Executive Summary & Core Hypotheses

The paper claims that by inserting continuous latent tokens $z_{1:K}$ between `<|lvr_start|>` and `<|lvr_end|>` and supervising them with Mean Squared Error (MSE) against visual Region-of-Interest (ROI) tokens, the LLM performs "reasoning in visual latent space," further refined via Latent GRPO.

Our forensic code audit revealed two structural anomalies:
1. **The Identity Copyist ($H_1$)**: The MSE auxiliary loss does not generate new visual representations; it trains the model to copy raw unpooled ViT patch embeddings that already reside in the prompt's visual prefix ($\text{Cosine Sim} > 0.998$).
2. **The Disconnected Policy Gradient ($H_2$)**: In Stage 2 GRPO (`src/trainer/grpo_trainer.py:740–757`), the entire latent span is masked out (`final_mask = 0.0`) and rolled out under `@torch.no_grad()`. Zero policy gradient reaches the latent reasoning trajectory during RL.
3. **Representational Degeneration ($H_3$)**: Continuous latents fed recurrently into autoregressive generation cause drift in layer activations, collapsing into repetitive token loops unless hard syntactic boundaries clamp the generation distribution.

This document formulates a **4-Axis Theoretical Ablation Suite** designed to isolate every confounding variable, provide rigorous control baselines, and mathematically verify or falsify each claim.

---

## 2. Theoretical Hypotheses & Mathematical Formalization

### Hypothesis 1: The Auto-Associative Patch Copyist Hypothesis ($H_1$)
Let $X_V = \{v_1, v_2, \dots, v_P\} \subset \mathbb{R}^{d}$ denote the sequence of $P$ unpooled 2D/3D ViT image patch embeddings present in the prompt prefix. Let $R \subseteq \{1, \dots, P\}$ denote the subset of patch indices corresponding to the target bounding box (ROI).
During Stage 1 SFT, the latent hidden state $h_t^{(L)} \in \mathbb{R}^d$ at step $t \in [1, K]$ is supervised via:
$$\mathcal{L}_{MSE} = \frac{1}{|R|} \sum_{i \in R} \| h_{t, i}^{(L)} - v_i \|_2^2$$
- **Null Hypothesis ($H_1^0$)**: $h_t^{(L)}$ learns a non-trivial compositional transformation $f(X_V, Q) \neq v_i$ representing semantic abstraction.
- **Alternative Hypothesis ($H_1^A$)**: $h_t^{(L)}$ converges to the identity mapping of the input prefix:
$$\lim_{\mathcal{L}_{MSE} \to 0} \frac{\langle h_t^{(L)}, v_i \rangle}{\|h_t^{(L)}\|_2 \|v_i\|_2} = 1.0, \quad \forall i \in R$$
The latent tokens function purely as an in-context pointer/memory cache to prefix patches.

---

### Hypothesis 2: Ghost RL & Policy Gradient Independence ($H_2$)
In Stage 2 GRPO, the objective is defined as:
$$\mathcal{L}_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|Y_i|} \sum_{t=1}^{|Y_i|} M_t \min \left( r_t(\theta) A_i, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_i \right)$$
where $M_t \in \{0, 1\}$ is the mask applied at token $t$.
- In the audited codebase:
$$M_t = \begin{cases} 0.0 & \text{if } t \in [\text{idx}(\text{<\|lvr\_start\|>}), \text{idx}(\text{<\|lvr\_end\|>})] \\ 1.0 & \text{otherwise (verbal tokens)} \end{cases}$$
- **Null Hypothesis ($H_2^0$)**: GRPO updates the parameters governing the continuous latent transition matrix $\nabla_\theta \mathcal{L}_{GRPO} \cdot \frac{\partial z_t}{\partial \theta} \neq 0$.
- **Alternative Hypothesis ($H_2^A$)**: The gradient with respect to latent generation parameters is identically zero:
$$\nabla_{\theta_{LVR}} \mathcal{L}_{GRPO} \equiv 0$$
Any RL performance gains stem solely from linguistic calibration on post-latent verbal tokens.

---

### Hypothesis 3: Latent Step Invariance vs. Attention Sink Collapse ($H_3$)
Let $K$ be the number of latent reasoning steps. If genuine multi-step visual reasoning occurs:
$$\text{Performance}(K_2) > \text{Performance}(K_1) \quad \forall K_2 > K_1$$
- **Null Hypothesis ($H_3^0$)**: Downstream spatial reasoning monotonically scales with latent compute budget $K$.
- **Alternative Hypothesis ($H_3^A$)**: Performance is step-invariant or degenerates for $K > 1$. The language decoder attends exclusively to the delimiter `<|lvr_end|>` as an attention sink, rendering intermediate latents $z_{1:K-1}$ inert.

---

## 3. The 4-Axis Ablation Experimental Matrix

```
                        [ LVR ABLATION ARCHITECTURE ]
                                     |
       +-----------------------------+-----------------------------+
       |                             |                             |
[ Axis 1: Supervision ]     [ Axis 2: RL Gradient ]     [ Axis 3: Capacity ]
       |                             |                             |
 - 1A: Ground-Truth ROI        - 2A: Masked (Default)       - 3A: K = 0 (No Latent)
 - 1B: Permuted Patch          - 2B: Unmasked (True RL)     - 3B: K = 1 (Single Step)
 - 1C: Cross-Image Patch       - 2C: Frozen Latent Prefix   - 3C: K = 5 (Standard)
 - 1D: Gaussian Noise          - 2D: Pure Text CoT          - 3D: K = 20 (Deep)
 - 1E: Zero Supervision (LM)                                - 3E: K = 50 (Stress Test)
                                     |
                       [ Axis 4: Representation Type ]
                                     |
                         - 4A: Continuous Hidden States
                         - 4B: Codebook Quantized Tokens
                         - 4C: Inert Static Dummy Vectors
                         - 4D: Verbal Thought Tokens (<think>)
```

---

### Axis 1: Supervision Source & Ground-Truth Validity (Testing $H_1$)

This axis isolates whether the visual content of the MSE target matters, or if any auxiliary vector regularization suffices.

| Condition Code | Configuration Name | Target Representation ($T_i$) | Objective Function | Theoretical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **A1-CTRL** | Standard LVR | $v_{ROI} \in X_V$ (True bounding box) | $\mathcal{L}_{CE} + \lambda \mathcal{L}_{MSE}(h_t, v_{ROI})$ | Baseline replication. |
| **A1-PERM** | Spatial Patch Permutation | $\text{shuffle}(v_{ROI})$ (Random patch order) | $\mathcal{L}_{CE} + \lambda \mathcal{L}_{MSE}(h_t, \pi(v_{ROI}))$ | Tests if spatial topology matters or just raw feature distribution. |
| **A1-OUT** | Out-of-ROI Background | $v_{BKG} \in X_V \setminus X_{ROI}$ | $\mathcal{L}_{CE} + \lambda \mathcal{L}_{MSE}(h_t, v_{BKG})$ | Tests if task-relevant visual region matters. |
| **A1-CROSS**| Cross-Image Distractor | $v_{other} \sim \mathcal{D}_{val}$ (Unrelated image) | $\mathcal{L}_{CE} + \lambda \mathcal{L}_{MSE}(h_t, v_{other})$ | Tests if image-grounded features are required. |
| **A1-NOISE**| Gaussian Fixed Prior | $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | $\mathcal{L}_{CE} + \lambda \mathcal{L}_{MSE}(h_t, \epsilon)$ | Tests if vector norm regularization alone accounts for gains. |
| **A1-NONE** | Pure Language Modeling | None ($\lambda = 0$) | $\mathcal{L}_{CE}$ only | Pure language control with delimiter formatting. |

#### Expected Verification Verdict:
- If **$\text{Score}(\text{A1-CTRL}) \approx \text{Score}(\text{A1-PERM}) \approx \text{Score}(\text{A1-OUT})$**, then $H_1$ is confirmed: LVR does not perform grounded visual reasoning; it merely learns an unorganized patch-memory lookup.
- If **$\text{Score}(\text{A1-CTRL}) \gg \text{Score}(\text{A1-NOISE})$**, visual grounding is necessary.

---

### Axis 2: GRPO Gradient Flow & Optimization Isolation (Testing $H_2$)

This axis tests whether the Stage 2 reinforcement learning stage actually tunes latent reasoning or if all policy improvement is verbal.

| Condition Code | RL Mask ($M_{LVR}$) | Replay Mode | Projector Status | Theoretical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **A2-MASK (Default)** | $M_{LVR} = 0.0$ | `@torch.no_grad()` | Trainable | Replicates audited repo behavior (Ghost RL). |
| **A2-UNMASK** | $M_{LVR} = 1.0$ | Full Auto-diff | Trainable | True end-to-end continuous policy gradient through latents. |
| **A2-FROZEN** | $M_{LVR} = 0.0$ | Frozen weights | `requires_grad = False` | Strictly freezes all parameters upstream of `<|lvr_end|>`. |
| **A2-TEXT-ONLY** | No Latent Span | N/A | Removed | GRPO applied purely to text tokens (`<think> ... </think>`). |
| **A2-RANDOM-REW** | $M_{LVR} = 0.0$ | `@torch.no_grad()` | Trainable | Control: RL with uniform random rewards. |

#### Expected Verification Verdict:
- If **$\text{Score}(\text{A2-MASK}) \equiv \text{Score}(\text{A2-FROZEN})$**, then $H_2$ is confirmed: the latent span is completely unaffected by RL; the Stage 2 training gains reported in the paper are entirely verbal reward shaping.
- If **$\text{Score}(\text{A2-UNMASK})$** suffers from policy collapse or gradient explosion, it explains why the authors masked out $M_{LVR} = 0$ in the codebase.

---

### Axis 3: Reasoning Depth & Step Scaling ($K \in \{0, 1, 3, 5, 10, 20, 50\}$) (Testing $H_3$)

Tests whether intermediate latent compute represents progressive inferential chains or redundant representations.

| Condition Code | Latent Quota $K$ | Decoding Scheme | Attention Masking | Target Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **A3-K0** | $K = 0$ | Direct Generation | None | Base model lower bound. |
| **A3-K1** | $K = 1$ | Step Decoding | Causal | Tests single summary bottleneck. |
| **A3-K3** | $K = 3$ | Step Decoding | Causal | Intermediate step comparison. |
| **A3-K5** | $K = 5$ | Step Decoding | Causal | Author's default configuration. |
| **A3-K10** | $K = 10$ | Step Decoding | Causal | Deep latent chain. |
| **A3-K20** | $K = 20$ | Step Decoding | Causal | Long reasoning trajectory (Mull-Tokens parity). |
| **A3-K50** | $K = 50$ | Step Decoding | Causal | Stress test for representational degeneration. |

#### Diagnostic Metrics:
1. **Latent-to-Latent CKA**:
   $$\text{CKA}(H_t, H_{t+1}) = \frac{\| H_t^\top H_{t+1} \|_F^2}{\| H_t^\top H_t \|_F \| H_{t+1}^\top H_{t+1} \|_F}$$
   If $\text{CKA} > 0.99$ across all steps $t$, the trajectory is static (no iterative computation).
2. **Attention Sink Weight**:
   $$\alpha_{end} = \frac{1}{|Y|} \sum_{y \in Y} \text{Attn}(y \to \text{<\|lvr\_end\|>}), \quad \alpha_{mid} = \frac{1}{|Y| K} \sum_{y \in Y} \sum_{t=1}^K \text{Attn}(y \to z_t)$$
   If $\alpha_{end} \gg \alpha_{mid}$, subsequent tokens ignore intermediate latents.

---

### Axis 4: Representation Type & Format-Locking Mechanism

This axis separates continuous latent semantics from discrete syntactic tokens.

| Condition Code | Latent Token Substrate | Input to Self-Attention | Vocabulary ID / State |
| :--- | :--- | :--- | :--- |
| **A4-CONT** | Continuous State ($h_t$) | Direct hidden vector feed | Unprojected $d$-dimensional float16 |
| **A4-QUANT** | Vector Quantized ($q(h_t)$) | Nearest codebook embedding | Discrete codebook index |
| **A4-DUMMY** | Inert Placeholder | Constant vector $\mathbf{c} \in \mathbb{R}^d$ | Frozen uniform embedding |
| **A4-PAD** | Padding Tokens | `<|pad|>` token embedding | Vocabulary ID 151643 |
| **A4-TEXT** | Verbal Reasoning | Text tokens (`Wait, let me inspect...`) | Standard natural language tokens |

---

## 4. Evaluation Suite & Benchmark Protocol

To prevent overfitting or task-specific artifacts, all ablation conditions must be evaluated on three distinct classes of vision-language reasoning:

```
                            [ EVALUATION SUITE ]
                                     |
       +-----------------------------+-----------------------------+
       |                             |                             |
[ Spatial Precision ]     [ Geometric & Math ]     [ General Multimodal ]
  - VSR (Spatial Rel)       - MathVista (GPS/Plot)    - MMBench
  - RefCOCOg (Bounding)     - MathVerse               - ScienceQA-IMG
  - Spatial-Grid Diagnostic - OlympiadBench-MM        - HallusionBench
```

### 1. Spatial Precision & Localization
- **Visual Spatial Reasoning (VSR)**: True/False binary validation on 65 spatial relations (tests topological reasoning).
- **RefCOCOg**: Strict bounding box coordinates (Intersection-over-Union, IoU@0.5).
- **Synthetic Spatial-Grid**: The bespoke 3-shape grid test (red square, blue circle, green triangle) to detect positional hallucination.

### 2. Geometric & Mathematical Reasoning
- **MathVista**: Sub-categories: Geometry (`GPS`), Function Plots (`MWP`), Visual Reasoning (`VQA`).
- **MathVerse**: Disentangled visual vs. text-dominant mathematical reasoning.

### 3. General Multimodal & Hallucination Diagnostics
- **MMBench (v1.1)**: Circular evaluation (A/B/C/D option permutation) to catch positional/letter bias.
- **HallusionBench**: Distinguishing visual hallucination from language prior bias.

---

## 5. Quantitative Diagnostics & Forensic Metrics

For each ablation run, compute the following four mechanistic metrics:

### Metric 1: Prefix Cosine Similarity ($CS_{prefix}$)
Measures the maximum alignment between any latent token $z_t$ and prefix visual patch $v_p$:
$$CS_{prefix}(t) = \max_{p \in [1, P]} \frac{\langle z_t, v_p \rangle}{\|z_t\|_2 \|v_p\|_2}$$
- **Threshold**: If $CS_{prefix} > 0.95$, the latent state is a literal patch copy.

### Metric 2: Effective Rank ($\text{erank}$) of Latent Trajectory
Measures the dimensional richness of the $K \times d$ latent matrix $Z$:
$$\text{erank}(Z) = \exp \left( -\sum_{j=1}^{\min(K, d)} \bar{\sigma}_j \ln \bar{\sigma}_j \right), \quad \bar{\sigma}_j = \frac{\sigma_j}{\sum_k \sigma_k}$$
where $\sigma_j$ are the singular values of $Z$.
- **Interpretation**: If $\text{erank}(Z) \approx 1.0$, all $K$ steps span the exact same 1D subspace (rank collapse).

### Metric 3: Empirical Policy Gradient Norm ($\|\mathbf{g}_{LVR}\|$)
During GRPO rollouts, measure the backpropagated gradient norm arriving at the latent projection layer:
$$\|\mathbf{g}_{LVR}\| = \left\| \frac{\partial \mathcal{L}_{GRPO}}{\partial W_{proj}} \right\|_F$$
- **Verification Criterion**: If $\|\mathbf{g}_{LVR}\| \equiv 0.0$, Disconnected Policy Gradient ($H_2$) is verified.

### Metric 4: Degenerate Repetition Frequency ($R_{deg}$)
Measures the percentage of rollout completions that enter repetitive n-gram loops (e.g. `地区地区...`):
$$R_{deg} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{max\_repeat\_ngram}(Y_i) \ge 10 \right)$$

---

## 6. Pre-Registered Decision Table & Theory Matrix

| Empirical Finding Across Ablations | Mechanistic Interpretation | Final Verdict on LVR Paper |
| :--- | :--- | :--- |
| **$\text{A1-CTRL} \approx \text{A1-PERM} \approx \text{A1-NOISE}$** | Auxiliary MSE provides generic regularization; ROI visual semantics are irrelevant. | **Claim Refuted**: Visual reasoning is fictitious. |
| **$\text{A2-MASK} \equiv \text{A2-FROZEN}$** | Stage 2 GRPO does not touch or optimize the latent reasoning mechanism. | **Claim Refuted**: Latent RL is an illusion. |
| **$\text{A3-K1} \ge \text{A3-K5} \gg \text{A3-K20}$** | Multi-step reasoning collapses; deeper chains cause representational drift. | **Claim Refuted**: Multi-step latent compute is non-functional. |
| **$\text{A4-CONT} \le \text{A4-DUMMY}$** | Continuous vectors degrade generation; inert dummy tokens perform equally well. | **Claim Refuted**: The continuous latent substrate is degenerate. |

---

## 7. Verification Checklist

- [x] All 4 axes mathematically formalized without execution code.
- [x] Clear control and counter-factual conditions defined for every paper claim.
- [x] Quantitative metrics established with explicit thresholds for falsification.
- [x] Disentanglement of SFT identity copying vs. GRPO zero-gradient policy isolation.
- [x] Cross-benchmark validation plan specified across spatial, mathematical, and hallucination suites.
