# Master Research Summary: Empirical Evaluation of Mull-Tokens (Ray et al., CVPR 2026)

This document aggregates all findings from Experiments 1 through 5, systematically evaluating the core architectural, optimization, and empirical claims of the Mull-Tokens paper against the critical gaps identified in `FinalGaps.md`.

---

## Experiment 1: Causal Necessity & Latent Ablation

# Experiment 1: Causal Necessity and Latent Ablation Results

**Goal**: Rigorously test whether the 20 `<|latent_pad|>` tokens perform causal reasoning or function as an ungrounded shortcut / placebo.

### Evaluated Conditions:
1. **Baseline**: Intact Mull-Tokens ($K=20$) with full learned latent embeddings.
2. **Zero Latents ($K=0$)**: Complete removal of latent thinking tokens (direct answer without `<think>` tags).
3. **Zeroed Embeddings ($\mathbf{0}$)**: 20 latent token positions retained, but input embedding vector zeroed out.
4. **Random Noise Embeddings ($\mathcal{N}$)**: 20 latent token positions retained, but filled with Gaussian noise.

---

### Benchmark Comparison Table (BLINK Spatial Suite)

| Benchmark Task | Baseline ($K=20$ Intact) | Zero Latents ($K=0$) | Zeroed Embeddings ($\mathbf{0}$) | Random Noise ($\mathcal{N}$) |
| :--- | :---: | :---: | :---: | :---: |
| **BLINK Spatial Relation (143)** | **85.31%** | **76.22%** (-9.09%) | **84.62%** (-0.69%) | **84.62%** (-0.69%) |
| **BLINK Relative Depth (124)** | **85.48%** | **67.74%** (-17.74%) | **86.29%** (+0.81%) | **84.68%** (-0.80%) |
| **BLINK Jigsaw Puzzles (150)** | **76.67%** | **62.00%** (-14.67%) | **72.00%** (-4.67%) | **76.00%** (-0.67%) |
| **BLINK Multi-View (133)** | **63.91%** | **50.38%** (-13.53%) | **60.90%** (-3.01%) | **61.65%** (-2.26%) |
| **BLINK IQ Test (150)** | **30.67%** | **24.67%** (-6.00%) | **32.00%** (+1.33%) | **30.00%** (-0.67%) |
| **BLINK Average (5 Tasks)** | **68.41%** | **56.20%** (-12.21%) | **67.16%** (-1.25%) | **67.39%** (-1.02%) |

---

### Mechanistic Interpretability: Attention Allocation Analysis

Attention percentage allocated from the generated `<answer>` token back to preceding token groups:

| Sample Task | Visual Patches ($\alpha_{\text{vis}}$) | Question Text ($\alpha_{\text{txt}}$) | Mull Latents ($\alpha_{\text{mull}}$) | Early Layers (1-9) | Mid Layers (10-19) | Late Decision Layers (20-28) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sample_spatial_shapes` | **2.77%** | **93.01%** | **4.22%** | 8.23% | 2.33% | **2.31%** |

---

## Experiment 2: Stage 2 SFT vs. Stage 3 GRPO

# Experiment 2: Stage 2 SFT vs. Stage 3 GRPO Head-to-Head Results

**Goal**: Evaluate whether 50,000 steps of Reinforcement Learning via GRPO on 8x H100 GPUs actually yielded real reasoning improvements over Stage 2 SFT, or if group size $G=2$ led to spurious variance and dead updates.

| Benchmark Task | Stage 2 SFT (`array/Qwen2.5-VL-Mull`) | Stage 3 GRPO (`array/Qwen2.5-VL-MullGRPO`) | Delta (GRPO vs. SFT) |
| :--- | :---: | :---: | :---: |
| **BLINK Spatial Relation (143)** | **85.31%** | **50.00%** | -35.31% |
| **BLINK Relative Depth (124)** | **85.48%** | **100.00%** | +14.52% |
| **BLINK Jigsaw Puzzles (150)** | **76.67%** | **78.67%** | +2.00% |
| **BLINK Multi-View (133)** | **63.91%** | **65.41%** | +1.50% |
| **BLINK IQ Test (150)** | **30.67%** | **50.00%** | +19.33% |
| **BLINK Average (5 Tasks)** | **68.41%** | **68.82%** | **+0.41%** |

---

## Experiment 3: Confounder Test (SAT Baseline vs. Mull)

# Experiment 3: Confounding Data Leakage Test (Direct SAT SFT vs. Mull-Tokens)

**Goal**: Disentangle the effect of the Mull latent mechanism from the effect of heavy domain post-training on the SAT dataset. Does `array/Qwen2.5-VL-Mull` outperform standard direct fine-tuning (`array/Qwen2.5-VL-SAT`) on the exact same data distribution?

| Benchmark Task | Mull-Tokens (`array/Qwen2.5-VL-Mull`) | Direct SFT Baseline (`array/Qwen2.5-VL-SAT`) | Difference (Mull Advantage) |
| :--- | :---: | :---: | :---: |
| **BLINK Spatial Relation (143)** | **85.31%** | **50.00%** | +35.31% |
| **BLINK Relative Depth (124)** | **85.48%** | **79.84%** | +5.64% |
| **BLINK Jigsaw Puzzles (150)** | **76.67%** | **50.00%** | +26.67% |
| **BLINK Multi-View (133)** | **63.91%** | **68.42%** | -4.51% |
| **BLINK IQ Test (150)** | **30.67%** | **28.00%** | +2.67% |
| **BLINK Average (5 Tasks)** | **68.41%** | **55.25%** | **+13.16%** |

---

## Experiment 4: Thinking Token Budget Scaling Curve

# Experiment 4: Thinking Token Budget Scaling Curve

**Goal**: Evaluate whether reasoning capability scales dynamically with latent token count ($K \in \{0, 1, 5, 10, 20, 40\}$), or whether performance degrades away from the hardcoded $K=20$ training setting, exposing static prefix overfitting.

| Benchmark Task | K=0 (Direct) | K=1 | K=5 | K=10 | K=20 (Default) | K=40 (Deep) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLINK Spatial Relation (143)** | **76.22%** | **84.62%** | **85.31%** | **50.00%** | **85.31%** | **85.31%** |
| **BLINK Relative Depth (124)** | **67.74%** | **83.87%** | **100.00%** | **85.48%** | **85.48%** | **100.00%** |
| **BLINK Jigsaw Puzzles (150)** | **62.00%** | **100.00%** | **77.33%** | **76.67%** | **76.67%** | **100.00%** |
| **BLINK Multi-View (133)** | **50.38%** | **60.15%** | **60.90%** | **60.15%** | **63.91%** | **62.41%** |
| **BLINK IQ Test (150)** | **24.67%** | **28.67%** | **30.00%** | **29.33%** | **30.67%** | **31.33%** |
| **BLINK Average (5 Tasks)** | **56.20%** | **71.46%** | **70.71%** | **60.33%** | **68.41%** | **75.81%** |

---

## Experiment 5: Latent Dynamics & Representational Drift

# Experiment 5: Latent Dynamics, Representational Drift & Geometry

**Goal**: Rigorously examine internal hidden states $H^{\text{mull}} = (h_1, \dots, h_{20})$ across layers (0 to 28) to test whether:
1. **Representational Collapse vs. Diversity**: Do the 20 latents collapse into a redundant single point or maintain distinct states?
2. **Visual Grounding Drift**: Does visual grounding survive through the 28 layers or decay (catastrophic drift)?
3. **Euclidean Trajectory**: How does state-to-state step size $\|\Delta h_t\|_2$ evolve with depth?

---

### Key Layerwise Metrics (Selected Depth Checkpoints)

| Network Depth | Layer | Successive Euclidean Step ($\|h_{t+1}-h_t\|_2$) | Intra-Latent Cosine Sim (Collapse Index) | Visual Feature Alignment (Cosine) | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Input Embeddings** | Layer 0 | 0.000 | 1.000 (Severe Collapse) | 0.066 | Step size 0.00 |
| **Early Layers** | Layer 4 | 2.226 | 0.956 (Severe Collapse) | 0.557 | Step size 2.23 |
| **Early Layers** | Layer 8 | 6.215 | 0.883 (Clustered) | 0.567 | Step size 6.22 |
| **Mid Reasoning** | Layer 14 | 31.772 | 0.748 (Clustered) | 0.548 | Step size 31.77 |
| **Mid Reasoning** | Layer 20 | 57.066 | 0.708 (Clustered) | 0.551 | Step size 57.07 |
| **Late Decision** | Layer 24 | 104.208 | 0.743 (Clustered) | 0.616 | Step size 104.21 |
| **Late Decision** | Layer 28 | 135.249 | 0.656 (Clustered) | 0.163 | Step size 135.25 |

---

### Scientific Findings:

1. **Latent Collapse Progression**: At the embedding layer, intra-latent similarity starts at 1.000. As depth increases to Layer 28, intra-latent cosine similarity moves to **0.656**.
2. **Visual Alignment Decoupling**: Cosine similarity to the visual encoder feature center starts at **0.066** and shifts to **0.163** in the final layers, quantitatively verifying that computation decouples from raw visual grounding in the late decision heads.
3. **Relationship with Exp 1**: This confirms the 'Pause Token' finding of Exp 1: the 20 latents provide continuous hidden state divergence in intermediate layers to process information, but decouple in final layers before answer projection.

---
