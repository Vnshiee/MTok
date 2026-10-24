# Research Roadmap: Mull-Tokens Replication & Mechanistic Investigation

This roadmap outlines the milestones, current completion status, and future research directions for the **Mull-Tokens (MTok)** project.

---

## Phase 1: Environment Setup, Bug Fixes & Replication (✅ 100% Complete)

- [x] **Framework & Dependency Architecture**:
  - [x] Custom `transformers` fork integration (`transformers==4.49.0.dev0`).
  - [x] PyTorch 2.5.1 + CUDA 12.2 + Flash Attention 2 setup on NVIDIA A100.
  - [x] `lmms_eval` harness configuration and missing package installations.
- [x] **Critical Bug Patches**:
  - [x] Chat template turn delimiter bug: fixed global stripping of `<|im_end|>\n`.
  - [x] Stream Hugging Face datasets dynamically instead of hardcoded `/home/jupyter/...` cluster paths (BLINK, ERQA, SAT, VSI-Bench).
  - [x] Patched SAT visual loader to read embedded PIL images from `doc["images"]`.
  - [x] Patched ERQA loader to decode byte streams from `doc["images_encoded"]`.
  - [x] Fixed VSI-Bench aggregation metric calculations and logging paths.
- [x] **Benchmark Replication**:
  - [x] Single-sample inference validation (`mull_infer.py`) on spatial, OCR, and scene tasks.
  - [x] Evaluated all 8 Core Spatial Benchmarks (~2,400 samples).
  - [x] Evaluated full Video Spatial Reasoning benchmark (VSI-Bench, 5,130 samples).
  - [x] Generated detailed replication writeup ([`MULL_TOKENS_REPLICATION_REPORT.md`](MULL_TOKENS_REPLICATION_REPORT.md)).

---

## Phase 2: Empirical Gap Analysis & 5 Master Experiments (✅ 100% Complete)

- [x] **Critical Gap Analysis**:
  - [x] Authored comprehensive critique [`FinalGaps.md`](FinalGaps.md) analyzing training contradictions, shortcut learning, group size bottlenecks, and confounders.
- [x] **Experiment 1: Causal Necessity & Latent Ablation**:
  - [x] Evaluated Intact Baseline ($K=20$) vs. Zero Latents ($K=0$) vs. Zeroed Embeddings ($\mathbf{0}$) vs. Gaussian Noise ($\mathcal{N}$).
  - [x] Developed PyTorch attention allocation diagnostic (`experiment1_attention_diagnostic.py`).
  - [x] Documented findings: 20 latents function as a *Pause Token / Compute Buffer* rather than semantic visual vectors.
- [x] **Experiment 2: Stage 2 SFT vs. Stage 3 GRPO**:
  - [x] Evaluated `array/Qwen2.5-VL-Mull` vs. `array/Qwen2.5-VL-MullGRPO`.
  - [x] Quantified that 50,000 RL steps yielded only +0.41% net gain due to $G=2$ dead updates.
- [x] **Experiment 3: Confounding Data Leakage Test**:
  - [x] Evaluated `array/Qwen2.5-VL-Mull` vs. `array/Qwen2.5-VL-SAT` on identical test splits.
  - [x] Proved +13.16% advantage of Mull latent architecture over direct domain fine-tuning.
- [x] **Experiment 4: Thinking Token Budget Scaling Curve**:
  - [x] Evaluated token budget performance across $K \in \{0, 1, 5, 10, 20, 40\}$.
  - [x] Discovered scaling up to $K=40$ achieves peak accuracy of 75.81%.
- [x] **Experiment 5: Latent Dynamics & Representational Drift**:
  - [x] Measured Euclidean step size, intra-latent cosine similarity, and visual alignment across layers 0 to 28.
  - [x] Documented decoupling of visual alignment in late decision heads (layers 24–28).
- [x] **Master Synthesis**:
  - [x] Consolidated all 5 experiment summaries into [`eval_outputs/ALL_EXPERIMENTS_MASTER_SUMMARY.md`](eval_outputs/ALL_EXPERIMENTS_MASTER_SUMMARY.md).

---

## Phase 3: Architectural Extensions & Mechanistic Interventions (⏳ Next Steps)

- [ ] **Milestone 3.1: Adaptive / Dynamic Thinking Budget ($K_{\text{adaptive}}$)**
  - *Hypothesis*: Easy questions can terminate thinking in $K=2$ to $5$ tokens, while complex spatial puzzles benefit from scaling to $K=40$.
  - *Action*: Train or calibrate a dynamic stopping token `<|end_of_thought|>` or monitor state-to-state convergence $\|\Delta h_t\|_2 < \tau$ during generation.
  - *Target Metric*: Reduce average latency by >40% with zero loss in BLINK benchmark accuracy.

- [ ] **Milestone 3.2: High-Group-Size GRPO Scaling ($G \ge 8$)**
  - *Hypothesis*: The marginal +0.41% gain of Stage 3 RL was caused by the severe memory-constrained group size $G=2$.
  - *Action*: Implement DeepSpeed ZeRO-3 Offload or parameter-efficient LoRA training to scale group size to $G=8$ and $G=16$.
  - *Target Metric*: Eliminate dead updates and achieve statistically significant RL gains on spatial benchmarks.

- [ ] **Milestone 3.3: Unfreezing Vision Backbone & Patch Adapters**
  - *Hypothesis*: Negative transfer on fine-grained depth (e.g. BLINK Relative Depth) stems from the frozen vision encoder.
  - *Action*: Add trainable Low-Rank Adapters (LoRA) to the vision transformer and image projection layer.
  - *Target Metric*: Recover positive transfer on BLINK Relative Depth (>88%).

- [ ] **Milestone 3.4: Linear Probing & Latent Geometry Decodability**
  - *Hypothesis*: Are intermediate latent vectors truly storing spatial coordinates or purely abstract tokens?
  - *Action*: Train linear probes on hidden states $H^{\text{mull}}$ at Layer 14 to decode 2D bounding boxes and object relative positions.

---

## Phase 4: Model Scale & Publication (⏳ Future Work)

- [ ] **Milestone 4.1: Cross-Scale Generalization (32B / 72B)**:
  - Test whether discrete latent thinking benefits larger parameter regimes (e.g., Qwen2.5-VL-72B) where text CoT visual drift is already reduced.
- [ ] **Milestone 4.2: Open-Source Benchmark Suite & Technical Report**:
  - Package `mull-eval` as a modular Python package with pip installability.
  - Draft comprehensive technical report synthesizing the causal necessity findings and gap resolutions.
