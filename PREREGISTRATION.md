# Pre-Registration Note: Experiment E1 Matrix
**Date:** October 2026  
**Repository:** [Vnshiee/MTok](https://github.com/Vnshiee/MTok)  
**Target:** Mechanistic Disentanglement of Latent Reasoning Compute vs. Format Conditioning on BLINK (N=700)

---

## 1. Background & Motivation

Initial experiments with Mull (`array/Qwen2.5-VL-Mull`) showed a pronounced performance gap between $K=20$ latents (68.41% overall) and $K=0$ latents (56.20% overall). Concurrently, budget scaling across $K \in \{1, 5, 10, 20, 40\}$ revealed an essentially flat trajectory (67.06% to 68.41%), where a single latent token ($K=1$) recovered ~89% of the full $K=0 \rightarrow K=20$ gain, and ablating the latent token embedding to $\mathbf{0}$ or random Gaussian noise $\mathcal{N}$ yielded 67.16% and 67.39% respectively.

These findings suggest that the $K=0$ drop is not simply a linear reduction in thinking capacity, but may reflect **format conditioning**: the assistant prefill `<think>...</think>\n` forces the model into an immediate answering mode, suppressing verbose, unconstrained verbal chain-of-thought that degrades spatial perception accuracy.

This pre-registration defines the 9-condition experimental matrix designed to formally adjudicate between these hypotheses before running the full evaluation.

---

## 2. Competing Hypotheses & Theoretical Predictions

### Hypothesis H1: Latent Scratchpad / Compute Scaling
*The latent tokens function as continuous cognitive scratchpad representations where spatial reasoning is performed across depth.*
* **Prediction H1.1:** Row 1 ($K=20$ latents) significantly outperforms Rows 3, 4, and 5 (McNemar test $p < 0.05$).
* **Prediction H1.2:** Rows 3, 4, and 5 (empty tags `<think></think>`, dot pauses, generic special tokens) lack latent spatial representations and will degrade towards Row 2 ($K=0$).
* **Prediction H1.3:** Output representations in Row 1 exhibit distinct representational dynamics that are absent in generic pauses.

### Hypothesis H3: Format Conditioning / Verbal Reasoning Suppression
*The assistant prefill `<think>...</think>\n` acts as a syntactic stop-gate that prevents the base LLM from emitting unconstrained verbal CoT ("let me think...", rambling steps) which confuses spatial token extraction.*
* **Prediction H3.1:** Rows 3, 4, and 5 will land close to Row 1 (~67–68%), well within statistical margin of error.
* **Prediction H3.2:** Paired McNemar tests between Row 1 and Rows 3, 4, 5 will show no statistically significant divergence ($p > 0.05$).
* **Prediction H3.3:** Output forensics will show Row 2 generates long, discursive text sequences (avg length > 150 tokens), whereas Rows 1, 3, 4, and 5 generate short answers (avg length < 25 tokens) immediately following the closed tag.

---

## 3. The E1 Run Matrix

All runs evaluate all 5 spatial BLINK benchmarks (Total $N=700$):
- `blink_sprel` ($N=143$)
- `blink_reldepth` ($N=124$)
- `blink_jigsaw` ($N=150$)
- `blink_mv` ($N=133$)
- `blink_iqtest` ($N=150$)

All runs standardize image resolution at 4.19 MP (`max_pixels=4194304`). All runs log per-sample generated token IDs, length, latency, and full decoded text.

| Row # | Model | User Prompt (`BLINK_PROMPT_MODE`) | Assistant Prefill String | Expected Prefill Tokens | Theoretical Target |
|---|---|---|---|:---:|---|
| **1** | Mull | `latents` | `<think>` + 20×`<\|latent_pad\|>` + `</think>\n` | 26 | Clean reference baseline ($K=20$) |
| **2** | Mull | `latents` | `""` (empty) | 0 | Unconstrained generation baseline ($K=0$) |
| **3** | Mull | `latents` | `<think></think>\n` | 5 | Tag-only test: are empty think tags sufficient? |
| **4** | Mull | `latents` | `<think>` + 20×`" ."` + `</think>\n` | 26 | Generic pause tokens: does arbitrary delay suffice? |
| **5** | Mull | `latents` | `<think>` + 20×`<\|latent_end\|>` + `</think>\n` | 26 | Special token control: syntax vs. latent semantics |
| **6** | Mull | `sft` | `""` (empty) | 0 | Prompt confound test: is $K=0$ drop driven by prompt? |
| **7** | Mull | `sft` | `<think>` + 20×`<\|latent_pad\|>` + `</think>\n` | 26 | Mull sensitivity to SFT prompt |
| **8** | SAT | `sft` | `""` (empty) | 0 | Direct SAT SFT baseline (paper's canonical DirAns) |
| **9** | SAT | `latents` | `""` (empty) | 0 | SAT evaluated under latent user prompt |

### 2×2 Decomposition (Rows 6–9)
Rows 6, 7, 8, and 9 form a full $2 \times 2$ factorial design of **Model** ($\text{Mull} \text{ vs. } \text{SAT}$) $\times$ **Prompt** ($\text{SFT} \text{ vs. } \text{Latents}$). This directly quantifies:
1. The prompt-induced performance delta on Mull without prefill ($\text{Row 6} - \text{Row 2}$).
2. The prompt-induced performance delta on the SAT baseline ($\text{Row 8} - \text{Row 9}$).
3. The true model-to-model advantage under matched SFT conditions ($\text{Row 7} - \text{Row 8}$ and $\text{Row 6} - \text{Row 8}$).

---

## 4. Statistical Testing Protocol

1. **Paired McNemar's Test**:
   For each condition $i \in \{2, \dots, 9\}$, we construct the $2 \times 2$ contingency table of paired correctness with Row 1:
   $$b = \text{Count}(\text{Row 1 Correct}, \text{Row } i \text{ Incorrect})$$
   $$c = \text{Count}(\text{Row 1 Incorrect}, \text{Row } i \text{ Correct})$$
   For $b + c \ge 25$, we compute the continuity-corrected McNemar statistic:
   $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
   For $b + c < 25$, we compute the two-sided exact binomial test.

2. **Bootstrap Confidence Intervals**:
   10,000 paired bootstrap resamples ($B=10,000$) over the $N=700$ items to compute 95% percentile bootstrap confidence intervals for both absolute accuracy and difference vs. Row 1.

3. **Output Forensics**:
   - Generated token length distributions (mean, median, 95th percentile).
   - Frequency of verbal thinking tokens (`"let me think"`, `"wait"`, `"reasoning"`) in Row 2 vs Rows 1, 3–5.
   - Per-task item discordance matrix (which specific questions flip correct $\leftrightarrow$ incorrect).

---

## 5. Decision Rule for Follow-up Experiments

* **If Hypothesis H1 is supported** (Row 1 significantly beats Rows 3–5, $p < 0.05$):
  Proceed to **Experiment E2: Layer-wise Attention Knockout** to locate the exact transformer layers and attention heads routing spatial information through `<|latent_pad|>`.
* **If Hypothesis H3 is supported** (Rows 3–5 match Row 1, $p > 0.05$, and Row 2 generates verbose CoT):
  Formalize the **Format Conditioning Finding**: document that the reported latent reasoning gain over $K=0$ is primarily a suppression of verbal reasoning degradation in visual QA, and evaluate if simple prompt engineering or minimal punctuation tokens achieve the same empirical benefit.
