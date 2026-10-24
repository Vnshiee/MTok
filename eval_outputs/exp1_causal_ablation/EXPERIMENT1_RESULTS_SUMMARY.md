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
