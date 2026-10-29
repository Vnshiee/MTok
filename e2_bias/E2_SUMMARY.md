# Experiment E2: Answer Bias vs. Discrimination Analysis
**Generated:** outputs | **Evaluations:** N=700 (2-Way N=550)

## 1. Master Calibration & Debiasing Matrix

| Cond | Model | Prefill | Raw Acc (N=700) | Calibrated M2 | Macro Raw | Macro Calib | 2-Way Pred B | Debiased M3 (N=550) | Mean L | Mean P |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T1** | mull | `latents` / `<think><|latent...` | **66.71%** | 53.57% | 67.42% | 54.09% | 60.1% | 74.91% | +0.098 | +0.191 |
| **T10** | grpo | `latents` / `<think><|latent...` | **68.71%** | 57.43% | 69.36% | 58.04% | 53.8% | 75.09% | +0.024 | +0.282 |
| **T11** | mull | `latents` / `<think><|latent...` | **66.86%** | 53.14% | 67.52% | 53.65% | 60.5% | 75.09% | +0.120 | +0.198 |
| **T12** | mull | `latents` / `<think><|latent...` | **67.71%** | 53.86% | 68.37% | 54.37% | 59.7% | 74.91% | +0.103 | +0.186 |
| **T13** | mull | `latents` / `<think><|latent...` | **66.43%** | 53.14% | 67.14% | 53.60% | 62.5% | 74.73% | +0.113 | +0.201 |
| **T14** | mull | `latents` / `<think><|latent...` | **67.00%** | 53.14% | 67.67% | 53.57% | 56.6% | 74.18% | +0.085 | +0.178 |
| **T2F** | mull | `latents` / `...` | **66.86%** | 53.43% | 67.57% | 53.85% | 56.8% | 73.82% | +0.069 | +0.190 |
| **T3** | mull | `latents` / `<think></think>...` | **66.86%** | 53.86% | 67.52% | 54.34% | 59.9% | 73.82% | +0.120 | +0.183 |
| **T4** | mull | `latents` / `<think> . . . ....` | **67.00%** | 53.86% | 67.68% | 54.34% | 59.7% | 75.64% | +0.121 | +0.184 |
| **T5** | mull | `latents` / `<think><|latent...` | **67.00%** | 54.29% | 67.68% | 54.78% | 60.4% | 75.09% | +0.109 | +0.184 |
| **T6F** | mull | `sft` / `...` | **66.14%** | 54.43% | 66.76% | 54.86% | 59.2% | 73.45% | +0.088 | +0.187 |
| **T7** | mull | `sft` / `<think><|latent...` | **67.00%** | 53.57% | 67.66% | 54.08% | 60.1% | 74.18% | +0.127 | +0.183 |
| **T8** | sat | `sft` / `...` | **63.57%** | 56.14% | 64.27% | 56.41% | 50.0% | 72.55% | -0.125 | +0.185 |
| **T9** | sat | `latents` / `...` | **64.29%** | 57.71% | 65.06% | 57.98% | 46.6% | 72.18% | -0.128 | +0.146 |

## 2. Primary Hypothesis Q1: Does Mull's Edge Survive Bias Calibration?

| Metric | T1 (Mull) | T8 (SAT) | Delta | 95% Bootstrap CI | McNemar p | Survives? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw Accuracy (M1)** | 66.71% | 63.57% | +3.14% | [+0.00%, +6.29%] | p=0.0630 | Baseline |
| **Content-Free Calibrated (M2)** | 53.57% | 56.14% | -2.57% | [-5.29%, +0.14%] | p=0.0790 | NO |
| **Debiased 2x2 (M3, N=550)** | 74.91% | 72.55% | +2.36% | [-1.09%, +5.82%] | p=0.2181 | NO |

> **Share of Published Advantage Explained by Bias:** `35.0%`

## 3. Continuous Logit Margins: T3 (<think></think>) vs. T1 (20 Latents)

- **T1 Mean Margin:** `0.7163`
- **T3 Mean Margin:** `0.7130` (Delta: `+0.0033`, 95% CI: `[-0.0074, +0.0141]`)
- **Wilcoxon Signed-Rank Test:** W=46497.0, **p = 0.9833**
- **Paired Cohen's d_z:** `0.0230`
- **Two One-Sided Equivalence Test (TOST, ±3pp):** **CONFIRMED EQUIVALENT**
