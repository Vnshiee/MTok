# Experiment 4: Thinking Token Budget Scaling Curve

**Goal**: Evaluate whether reasoning capability scales dynamically with latent token count ($K \in \{0, 1, 5, 10, 20, 40\}$), or whether performance degrades away from the hardcoded $K=20$ training setting, exposing static prefix overfitting.

| Benchmark Task | K=0 (Direct) | K=1 | K=5 | K=10 | K=20 (Default) | K=40 (Deep) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLINK Spatial Relation (143)** | **76.22%** | **84.62%** | **85.31%** | **85.31%** | **85.31%** | **85.31%** |
| **BLINK Relative Depth (124)** | **67.74%** | **83.87%** | **86.29%** | **85.48%** | **85.48%** | **83.87%** |
| **BLINK Jigsaw Puzzles (150)** | **62.00%** | **78.00%** | **77.33%** | **76.67%** | **76.67%** | **76.67%** |
| **BLINK Multi-View (133)** | **50.38%** | **60.15%** | **60.90%** | **60.15%** | **63.91%** | **62.41%** |
| **BLINK IQ Test (150)** | **24.67%** | **28.67%** | **30.00%** | **29.33%** | **30.67%** | **31.33%** |
| **BLINK Average (5 Tasks)** | **56.20%** | **67.06%** | **67.97%** | **67.39%** | **68.41%** | **67.92%** |
