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
