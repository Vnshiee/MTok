# Experiment E1: Mechanistic Matrix Results
**Standardized Image Resolution:** 4.19 MP (max_pixels=4194304)  
**Expected Benchmark Total:** N=700 across 5 BLINK tasks  

## 1. Overall Performance Matrix

| # | Model | Prompt | Assistant Prefill | SpRel | RelDepth | Jigsaw | Multi-View | IQ-Test | Reasoning Avg | BLINK Overall | Delta vs Row 1 |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Mull | `latents` | `think + 20 latents` | 85.31% | 85.48% | 76.67% | 63.91% | 30.67% | **60.02%** | **67.71%** | Reference |
| **2** | Mull | `latents` | `empty` | 76.22% | 67.74% | 62.00% | 50.38% | 24.67% | **47.60%** | **55.71%** | -12.00% |
| **3** | Mull | `latents` | `<think></think>\n` | 85.31% | 85.48% | 79.33% | 60.15% | 28.67% | **58.10%** | **67.14%** | -0.57% |
| **4** | Mull | `latents` | `think + 20x' .'` | 84.62% | 84.68% | 75.33% | 60.90% | 30.00% | **58.53%** | **66.43%** | -1.29% |
| **5** | Mull | `latents` | `think + 20x<|latent_end|>` | 84.62% | 86.29% | 72.00% | 60.90% | 32.00% | **59.73%** | **66.43%** | -1.29% |
| **6** | Mull | `sft` | `empty` | 78.32% | 74.19% | 64.67% | 51.13% | 25.33% | **50.22%** | **58.14%** | -9.57% |
| **7** | Mull | `sft` | `think + 20 latents` | 85.31% | 84.68% | 77.33% | 59.40% | 29.33% | **57.80%** | **66.57%** | -1.14% |
| **8** | SAT | `sft` | `empty` | 81.82% | 79.03% | 68.00% | 63.91% | 27.33% | **56.76%** | **63.29%** | -4.43% |
| **9** | SAT | `latents` | `empty` | 81.82% | 79.84% | 68.00% | 68.42% | 28.00% | **58.75%** | **64.43%** | -3.29% |
| **10** | Mull-GRPO | `latents` | `think + 20 latents` | 86.01% | 84.68% | 78.67% | 65.41% | 31.33% | **60.47%** | **68.57%** | +0.86% |
| **11** | Mull | `latents` | `think + 1 latent` | 84.62% | 83.87% | 78.00% | 60.15% | 28.67% | **57.56%** | **66.43%** | -1.29% |
| **12** | Mull | `latents` | `think + 5 latents` | 85.31% | 86.29% | 77.33% | 60.90% | 30.00% | **59.06%** | **67.29%** | -0.43% |
| **13** | Mull | `latents` | `think + 10 latents` | 85.31% | 85.48% | 76.67% | 60.15% | 29.33% | **58.32%** | **66.71%** | -1.00% |
| **14** | Mull | `latents` | `think + 40 latents` | 85.31% | 83.87% | 76.67% | 62.41% | 31.33% | **59.20%** | **67.29%** | -0.43% |
| **15** | Mull | `latents` | `think + 20 latents` | 85.31% | 85.48% | 76.67% | 63.91% | 30.67% | **60.02%** | **67.71%** | +0.00% |

## 2. Statistical Significance vs. Row 1 Reference (N=700)

| Target Row | Condition | Overall Acc | Delta (95% CI) | McNemar b/c | p-value | Significant (p<0.05) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Row 2 | Mull / `latents` / `empty` | 55.71% | -12.00% [-15.86%, -8.57%] | b=133, c=48 | p=0.0000 | **YES (Reject H0)** |
| Row 3 | Mull / `latents` / `<think></think>\n` | 67.14% | -0.57% [-2.00%, +0.71%] | b=14, c=10 | p=0.5413 | No (p >= 0.05) |
| Row 4 | Mull / `latents` / `think + 20x' .'` | 66.43% | -1.29% [-2.57%, -0.14%] | b=14, c=5 | p=0.0636 | No (p >= 0.05) |
| Row 5 | Mull / `latents` / `think + 20x<|latent_end|>` | 66.43% | -1.29% [-2.57%, +0.00%] | b=15, c=6 | p=0.0784 | No (p >= 0.05) |
| Row 6 | Mull / `sft` / `empty` | 58.14% | -9.57% [-12.86%, -6.29%] | b=108, c=41 | p=0.0000 | **YES (Reject H0)** |
| Row 7 | Mull / `sft` / `think + 20 latents` | 66.57% | -1.14% [-2.57%, +0.29%] | b=18, c=10 | p=0.1859 | No (p >= 0.05) |
| Row 8 | SAT / `sft` / `empty` | 63.29% | -4.43% [-7.71%, -1.14%] | b=84, c=53 | p=0.0104 | **YES (Reject H0)** |
| Row 9 | SAT / `latents` / `empty` | 64.43% | -3.29% [-6.57%, +0.00%] | b=82, c=59 | p=0.0639 | No (p >= 0.05) |
| Row 10 | Mull-GRPO / `latents` / `think + 20 latents` | 68.57% | +0.86% [-1.00%, +2.71%] | b=20, c=26 | p=0.4610 | No (p >= 0.05) |
| Row 11 | Mull / `latents` / `think + 1 latent` | 66.43% | -1.29% [-2.71%, +0.00%] | b=16, c=7 | p=0.0931 | No (p >= 0.05) |
| Row 12 | Mull / `latents` / `think + 5 latents` | 67.29% | -0.43% [-1.71%, +0.86%] | b=12, c=9 | p=0.6636 | No (p >= 0.05) |
| Row 13 | Mull / `latents` / `think + 10 latents` | 66.71% | -1.00% [-2.14%, +0.14%] | b=12, c=5 | p=0.1435 | No (p >= 0.05) |
| Row 14 | Mull / `latents` / `think + 40 latents` | 67.29% | -0.43% [-1.86%, +1.00%] | b=15, c=12 | p=0.7003 | No (p >= 0.05) |
| Row 15 | Mull / `latents` / `think + 20 latents` | 67.71% | +0.00% [+0.00%, +0.00%] | b=0, c=0 | p=1.0000 | No (p >= 0.05) |

## 3. Output Forensics & Generation Length Analysis

| Row | Condition | Mean Gen Tokens | Median Gen Tokens | P95 Gen Tokens | Mean Latency (s/sample) |
|---|---|:---:|:---:|:---:|:---:|
| Row 1 | Clean reference baseline | 7.0 | 7.0 | 7.0 | 32.29s |
| Row 2 | K=0 unconstrained reasoning | 138.5 | 140.0 | 270.0 | 175.12s |
| Row 3 | Empty think tags | 7.0 | 7.0 | 7.0 | 0.45s |
| Row 4 | Generic pause tokens | 7.0 | 7.0 | 7.0 | 0.45s |
| Row 5 | Special token control | 7.0 | 7.0 | 7.0 | 0.45s |
| Row 6 | Prompt sensitivity (Mull un-prefilled) | 122.9 | 139.0 | 251.0 | 4.76s |
| Row 7 | Mull with SFT prompt | 7.0 | 7.0 | 7.0 | 0.45s |
| Row 8 | Direct SAT SFT baseline (DirAns) | 7.0 | 7.0 | 7.0 | 0.44s |
| Row 9 | SAT with latent prompt | 7.0 | 7.0 | 7.0 | 0.44s |
| Row 10 | Stage 3 GRPO model | 7.0 | 7.0 | 7.0 | 0.35s |
| Row 11 | Budget scaling K=1 | 7.0 | 7.0 | 7.0 | 0.35s |
| Row 12 | Budget scaling K=5 | 7.0 | 7.0 | 7.0 | 0.35s |
| Row 13 | Budget scaling K=10 | 7.0 | 7.0 | 7.0 | 0.37s |
| Row 14 | Budget scaling K=40 | 7.0 | 7.0 | 7.0 | 0.34s |
| Row 15 | Warm pure GPU reference | 7.0 | 7.0 | 7.0 | 0.36s |
