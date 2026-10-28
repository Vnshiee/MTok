# Experiment 3: Confounding Data Leakage Test (Direct SAT SFT vs. Mull-Tokens)

**Goal**: Disentangle the effect of the Mull latent mechanism from the effect of heavy domain post-training on the SAT dataset. Does `array/Qwen2.5-VL-Mull` outperform standard direct fine-tuning (`array/Qwen2.5-VL-SAT`) on the exact same data distribution?

| Benchmark Task | Mull-Tokens (`array/Qwen2.5-VL-Mull`) | Direct SFT Baseline (`array/Qwen2.5-VL-SAT`) | Difference (Mull Advantage) |
| :--- | :---: | :---: | :---: |
| **BLINK Spatial Relation (143)** | **85.31%** | **81.82%** | +3.49% |
| **BLINK Relative Depth (124)** | **85.48%** | **79.84%** | +5.64% |
| **BLINK Jigsaw Puzzles (150)** | **76.67%** | **68.00%** | +8.67% |
| **BLINK Multi-View (133)** | **63.91%** | **68.42%** | -4.51% |
| **BLINK IQ Test (150)** | **30.67%** | **28.00%** | +2.67% |
| **BLINK Average (5 Tasks)** | **68.41%** | **65.22%** | **+3.19%** |
