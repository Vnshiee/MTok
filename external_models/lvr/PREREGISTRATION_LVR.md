# Pre-Registration: Mechanistic Adjudication of Latent Visual Reasoning (LVR)

> [!WARNING]
> **Verification status (2026-11-03):** empirical adjudication registered as proposed protocols (E-L1 to E-L6). No GPU runs have been completed or claimed yet. See [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md) §6.


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
