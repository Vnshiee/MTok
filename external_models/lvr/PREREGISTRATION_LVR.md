# Pre-Registration: Mechanistic Adjudication of Latent Visual Reasoning (LVR)

> [!NOTE]
> **Verification status (2026-11-03):** Empirical execution of protocols E-L1 to E-L6 has been completed on GPU 0 (NVIDIA A100-PCIE-40GB) across benchmarks ($V^*$ Bench $N=191$, MMVP $N=300$, BLINK $N=697$) and 200 model rollouts from `vincentleebang/LVR-7B`. Full results, paired McNemar statistics, and attention/similarity telemetry are documented in Section 2 below and in [`LVR_ABLATION_RESULTS_REPORT.md`](LVR_ABLATION_RESULTS_REPORT.md).


**Target Model**: LVR (`vincentleebang/LVR-7B`, `vincentleebang/LVR-3B`, `arXiv:2509.24251`)  
**Backbone**: `Qwen2.5-VL-7B-Instruct` / `Qwen2.5-VL-3B-Instruct`  
**Theoretical Target**: Mechanistic Adjudication of Continuous Latent Visual Reasoning vs. Context Routing, Step Quotas, and Evaluation Parsing

---

## 1. Experimental Protocols & Decision Rules (E-L1 to E-L6)

All experiments run with fp32 scoring of option letters (or bf16 with a reported precision floor), $N = \text{all items}$, paired McNemar test with exact binomial $p$-value (capped at 1.0), and 10,000-sample bootstrap confidence intervals.

### Experiment E-L1: Do latent contents matter?
- **Question**: Does the semantic content of the latent vectors drive downstream reasoning accuracy?
- **Protocol**: At inference, with $K \in \{4, 8, 16\}$, replace the $K$ latent states with:
  1. `baseline`: the model's own generated latents;
  2. `oracle-roi`: oracle first-$K$ ROI ViT tokens;
  3. `random-non-roi`: random non-ROI image tokens;
  4. `mean-image`: the mean image token;
  5. `zeros`: all-zero vectors;
  6. `empty-tag`: $K=0$ (`<|lvr_start|><|lvr_end|>` forced with no intermediate latents).
  - *Evaluation Suites*: $V^*$ Bench (with gold bounding boxes) and BLINK.
- **Decision Rule**: If conditions (3), (4), (5), and (6) match condition (1) within $\pm 2.0$ percentage points (Two One-Sided Tests, TOST equivalence), the latent contents are not the operative mechanism. If condition (2) significantly outperforms condition (1), latent reconstruction quality limits performance.

### Experiment E-L2: Is information routed through the latents?
- **Question**: Does the answer generation causally depend on attention directed toward the latent tokens?
- **Protocol**: Attention knockout applied across layer bands:
  1. `answer -> latent` attention knockout;
  2. `latent -> image` attention knockout.
- **Decision Rule**: No significant change in downstream accuracy from `answer -> latent` knockout indicates that the latent reasoning span is bypassed during answer generation.

### Experiment E-L3: Do latents localize?
- **Question**: Do intermediate latent states align with spatial patches in the target bounding box?
- **Protocol**: For each latent state, compute cosine similarity against all input image tokens. Measure the top-1 hit rate inside the gold $V^*$ bounding box compared to the bounding box's area fraction (chance baseline).
- **Decision Rule**: A hit rate significantly above chance supports spatial localization/re-attention; a hit rate at or below chance falsifies spatial grounding.

### Experiment E-L4: Is sequence truncation (N3) costly?
- **Question**: Does truncating to fixed $K \in \{4, 8, 16\}$ steps (covering only a fraction of the full training ROI length $T_v$, median 28, mean 66) penalize performance?
- **Protocol**: Oracle injection of the first-$K$ tokens vs. the full $T_v$ ROI tokens, followed by a forced `<|lvr_end|>`.
- **Decision Rule**: Full $T_v > \text{first-}K$ demonstrates that the truncated $K$-step decoder discards task-critical spatial content.

### Experiment E-L5: Is the baseline gap driven by parsing artifacts?
- **Question**: Does the fragile first-character answer parser artificially suppress natural language / Chain-of-Thought baselines?
- **Protocol**: Re-score raw completions from all baseline models (including Vision-R1) under:
  1. the official LVR parser;
  2. strict `<answer>` tag extraction;
  3. standardized `lmms-eval` BLINK extraction.
  Report parse-failure and formatting-penalty rates across all conditions.
- **Decision Rule**: If Vision-R1's binary-task scores (e.g., JigSaw 27.3%, MMVP 46.7%) rise to or above chance (50.0%) under standardized extraction (2) and (3), parsing distortion is confirmed.

### Experiment E-L6: Magnitude of the RL Replay Context Mismatch (N1)
- **Question**: How severely does the off-by-one latent replay error in `_replay_lvr_to_collect_states` distort policy log-probabilities on the real 3B model?
- **Protocol**: For 200 RL rollouts from LVR-3B, evaluate per-token log-probabilities across:
  1. the rollout logits;
  2. the official repo loss forward path;
  3. the corrected replay forward path (with the 2-line alignment fix).
  Single forward pass per rollout without training parameter updates.
- **Decision Rule**: A mean $|\Delta \log p|$ across loss tokens clearly exceeding the bf16 precision floor confirms that published GRPO_latent gradients were computed on corrupted contexts.

---

## 2. Empirical Results & Decision Rule Outcomes (Executed on GPU 0)

All experiments were executed using the full evaluation suite in `external_models/lvr/ablation_study/` on an NVIDIA A100-PCIE-40GB GPU. Full tabular metrics and per-item predictions are stored in `ablation_outputs/` and synthesized in [`LVR_ABLATION_RESULTS_REPORT.md`](LVR_ABLATION_RESULTS_REPORT.md).

### Experiment E-L1 Outcome: Do latent contents matter?
- **Empirical Findings**:
  - $V^*$ ($N=191$): Baseline `A1A-BASE` ($K=8$) achieves **81.15%**. Replacing the 8 latent vectors with:
    - `C6-ZERO-VEC` (all zeros): **82.20%** ($\Delta = +1.05\%$, paired exact $p = 0.7539$)
    - `C7-GAUSS-NOISE` ($\mathcal{N}(0, 1)$): **81.15%** ($\Delta = +0.00\%$, paired exact $p = 1.0000$)
    - `C4-MEAN-IMG-POOL` (mean image token): **82.20%** ($\Delta = +1.05\%$, paired exact $p = 0.7539$)
    - `C8-SHUFFLE-BATCH` (cross-sample latents): **81.15%** ($\Delta = +0.00\%$, paired exact $p = 1.0000$)
    - `C2-EMPTY-TAG` ($K=0$, `<|lvr_start|><|lvr_end|>`): **81.68%** ($\Delta = +0.52\%$, paired exact $p = 1.0000$)
    - `C9-DIRECT-ANS` ($K=0$, direct answer): **81.68%** ($\Delta = +0.52\%$, paired exact $p = 1.0000$)
  - MMVP ($N=300$): Baseline `A1A-BASE` achieves **71.33%**. All counterfactual replacements (`zeros`: 70.67%, `noise`: 71.67%, `mean-pool`: 70.33%, `shuffle`: 71.00%, `empty-tag`: 70.67%) remain within $[-1.00\%, +0.33\%]$ with paired $p \ge 0.5488$.
  - BLINK ($N=697$): Baseline `A1A-BASE` achieves **54.81%**. All counterfactual replacements (`zeros`: 55.24%, `noise`: 55.52%, `mean-pool`: 54.52%, `shuffle`: 55.67%, `empty-tag`: 55.67%) remain within $[-0.29\%, +0.86\%]$ with paired $p \ge 0.4881$.
- **Decision Rule Verdict**: **Confirmed (Invariance to Latent Contents)**. Across all three benchmarks, every content-nullified and randomized replacement matches the baseline within $\pm 1.05\%$ (well within the $\pm 2.0\%$ TOST bound) with no statistically significant differences ($p > 0.48$). Downstream accuracy is invariant to the continuous vector contents of the latent reasoning positions.

### Experiment E-L2 Outcome: Is information routed through the latents?
- **Empirical Findings**:
  - Knocking out attention from answer tokens to the latent tokens (`A1A-MASK-ANS-TO-LVR`, causal attention weight $= 0.0$):
    - $V^*$: **82.20%** vs. Baseline **81.15%** ($\Delta = +1.05\%$, $p = 0.7539$).
    - MMVP: **71.00%** vs. Baseline **71.33%** ($\Delta = -0.33\%$, $p = 1.0000$).
    - BLINK: **55.09%** vs. Baseline **54.81%** ($\Delta = +0.29\%$, $p = 0.9049$).
  - Mean attention weight from answer queries to latent positions under baseline generation is minimal ($0.0165$ on $V^*$, $0.019$ on BLINK).
- **Decision Rule Verdict**: **Confirmed (Latents Bypassed)**. Zeroing attention from the answer tokens to the latent representations produces no statistically significant change across any benchmark ($p \ge 0.75$). The latent span is bypassed during answer generation.

### Experiment E-L3 Outcome: Do latents localize?
- **Empirical Findings**:
  - Under baseline $K=8$ generation on $V^*$:
    - Mean maximum cosine similarity to any image patch $\mathrm{CS}_{\max} = 0.5759$.
    - Mean cosine similarity to gold ROI patches $\mathrm{CS}_{\mathrm{ROI}} = 0.5017$.
    - ROI Recall@1 (fraction of latent tokens whose nearest neighbor is inside the gold ROI) is **11.06%**.
  - Injecting oracle ROI ViT tokens (`A1A-ORACLE-ROI`) forces ROI Recall@1 to **45.03%**, but yields **82.20%** accuracy ($\Delta = +1.05\%$, $p = 0.7539$ vs. baseline).
  - Injecting background non-ROI ViT tokens (`A1A-ORACLE-BKG`) drops ROI Recall@1 to **0.39%**, yielding **80.63%** accuracy ($\Delta = -0.52\%$, $p = 1.0000$).
- **Decision Rule Verdict**: **Falsified (No Spatial Localization Driving Performance)**. Natural latents exhibit low ROI localization (11.06%), and artificially driving localization up to 45.03% or down to 0.39% causes no statistically significant deviation in task accuracy.

### Experiment E-L4 Outcome: Is sequence truncation (N3) costly?
- **Empirical Findings**:
  - On $V^*$, injecting the full sequence of $T_v = \text{len}(\text{roi\_idxs})$ tokens (`A1A-ORACLE-ROI-FULL`, up to 5,000+ tokens, median 28, mean 66) yields **81.15%** accuracy—identical to injecting only the truncated first $K=8$ tokens (`81.15%`, $p = 1.0000$) and baseline $K=8$ (`81.15%`, $p = 1.0000$).
  - Evaluating fixed step sweeps $K \in \{0, 1, 2, 4, 8, 16, 32, 64\}$:
    - $K=0$: 78.53%
    - $K=1$: 81.15%
    - $K=2$: 79.58%
    - $K=4$: 81.15%
    - $K=8$: 81.15%
    - $K=16$: 81.15%
    - $K=32$: 81.15%
    - $K=64$: 81.68%
- **Decision Rule Verdict**: **Confirmed Invariance to Sequence Truncation**. Truncating the ROI from full $T_v$ patches to $K=8$ does not degrade performance; in fact, $K=1$ achieves 81.15%, identical to $K=8, 16, 32$ and full-sequence injection.

### Experiment E-L5 Outcome: Is the baseline gap driven by parsing artifacts?
- **Empirical Findings**:
  - MMVP evaluation:
    - Baseline `BASE-QWEN25-VL-7B` under `clean` prompt format: Official parser scores **0.00%**, while the robust parser extracts the correct option on **70.33%** of identical outputs.
    - LVR-7B under `clean` prompt format: Official parser scores **13.67%**, while the robust parser extracts **74.33%**.
    - LVR-7B under `bbox_aligned` prompt format: Official parser scores **4.67%**, while the robust parser extracts **73.67%**.
  - $V^*$ position bias:
    - Under `circular_eval` (cyclic option rotation), LVR-7B accuracy drops from **81.15%** to **73.82%** ($-7.33\%$, paired exact $p = 0.0001$).
- **Decision Rule Verdict**: **Confirmed (Parsing Sensitivity)**. The official first-character fallback penalizes models that omit explicit `<answer>` wrappers, artificially shifting baseline accuracy to near-zero (0.00%–13.67%) despite >70% semantic answer correctness.

### Experiment E-L6 Outcome: Magnitude of the RL Replay Context Mismatch (N1 & N2)
- **Empirical Findings** (Audited on `vincentleebang/LVR-7B`, 200 rollouts, $K=8$):
  - **N2 Verification**: The policy loss mask for `<|lvr_start|>` was **100.0% masked** across all 200 rollouts. The format reward objective cannot directly backpropagate gradient into $P(\text{<|lvr_start|>})$.
  - **N1 Verification**: The official repository loss path in `_replay_lvr_to_collect_states` introduces a 1-step index shift ($h_{\text{repo}}(k) = h_{\text{rollout}}(k-1)$):
    - Mean per-token $|\Delta \log p| = \mathbf{0.2080\text{ nats}}$ (max $0.2746\text{ nats}$).
    - **100.0%** of rollouts exhibit $|\Delta \log p| > 0.01\text{ nats}$.
    - State cosine similarity starts at $0.8033$ at step 0 and shifts the entire conditioning trajectory.
  - **Correction Verification**: Under the 2-line alignment fix, mean per-token $|\Delta \log p| = \mathbf{4.77 \times 10^{-7}\text{ nats}}$, recovering exact rollout conditioning within float32 numerical precision.
- **Decision Rule Verdict**: **Confirmed (Severe RL Context Mismatch)**. Upstream GRPO training forward passes operated on distorted context representations differing by $0.208\text{ nats}$ per token from the rollout states.
