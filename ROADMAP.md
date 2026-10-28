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
  - [x] Observed that zeroing or adding noise to embeddings produces minimal degradation (~1 point), supporting the hypothesis that latent tokens act as compute buffers or format conditions rather than semantic visual vectors.
- [x] **Experiment 2: Stage 2 SFT vs. Stage 3 GRPO**:
  - [x] Evaluated `array/Qwen2.5-VL-Mull` vs. `array/Qwen2.5-VL-MullGRPO`.
  - [x] Measured +0.81% average gain across 5 BLINK tasks (+1.39% on reasoning tasks), consistent with the paper's reported small delta.
- [x] **Experiment 3: Confounding Data Leakage Test**:
  - [x] Evaluated `array/Qwen2.5-VL-Mull` vs. `array/Qwen2.5-VL-SAT` on identical test splits.
  - [x] Measured +3.19% overall advantage for Mull, while the SAT baseline outperformed Mull by +4.51% on Multi-View reasoning.
- [x] **Experiment 4: Thinking Token Budget Scaling Curve**:
  - [x] Evaluated token budget performance across $K \in \{0, 1, 5, 10, 20, 40\}$.
  - [x] Observed a flat scaling curve between $K=1$ (67.06%) and $K=40$ (67.92%), indicating that a single latent token recovers ~89% of the K=0 to K=20 gap.
- [x] **Experiment 5: Latent Dynamics & Pilot Geometry**:
  - [x] Measured Euclidean step size, intra-latent cosine similarity, and visual alignment across layers 0 to 28 on a preliminary pilot query.
  - [x] Identified that the layer 28 shift coincides with the final RMSNorm boundary in `Qwen2_5_VL`.
- [x] **Master Synthesis**:
  - [x] Consolidated all 5 experiment summaries into [`eval_outputs/ALL_EXPERIMENTS_MASTER_SUMMARY.md`](eval_outputs/ALL_EXPERIMENTS_MASTER_SUMMARY.md) and logged corrections in [`ERRATA.md`](ERRATA.md).

---

## Phase 3: The E1 Run Matrix & Format-vs-Content Diagnostics (⏳ In Progress)

- [ ] **Milestone 3.1: Resolution Standardization & Image Audit**
  - *Action*: Standardize on 4.19 MP across all evaluations and audit BLINK image resolutions.
- [ ] **Milestone 3.2: The 9-Row Diagnostic Matrix (4.19 MP, All Logged)**
  - *Hypotheses*: 
    - H1 (Scratchpad computation): Performance scales with token budget and requires learned embeddings.
    - H3 (Format conditioning / mode switching): Prefill tags suppress harmful verbal reasoning; empty think tags (`<think></think>\n`) or punctuation pause tokens recover performance.
  - *Runs*:
    - Row 1: Mull | latents | think + 20 latents (Clean 4.19 MP reference)
    - Row 2: Mull | latents | empty (K=0 repeat with full token logging)
    - Row 3: Mull | latents | `<think></think>\n` (Tags only)
    - Row 4: Mull | latents | think + 20x`" ."` (Text pause tokens)
    - Row 5: Mull | latents | think + 20x`<|latent_end|>` (Alternative special token)
    - Row 6: Mull | sft | empty (Prompt sensitivity of K=0 drop)
    - Row 7: Mull | sft | think + 20 latents (Prompt sensitivity of Mull)
    - Row 8: SAT | sft | empty (True DirAns baseline setting)
    - Row 9: SAT | latents | empty (SAT under Mull prompt)
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
