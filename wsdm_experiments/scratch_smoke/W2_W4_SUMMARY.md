# WSDM Experiments W2 & W4: Causal Attention Knockout, Activation Patching & Pre-Norm Geometry
**Evaluated Items:** N=10 (2-Way Patching N=8) | **SDPA vs. FlashAttention-2 Parity:** 100.00%

## 1. Experiment W2: Causal 4D Attention Mask Knockout (N=700)

| Condition | Intervention | Micro Acc | Macro Acc | Agree w/ Intact | Mean Margin | Δ Margin | McNemar b/c | McNemar p |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **W2_Intact** | No intervention (Baseline T1) | **70.00%** | 70.00% | 100.00% | 0.7250 | +0.0000 | 0/0 | p=1.0000 |
| **W2_KO_LatentToAns_All** | Block `Latents -> Answer` (All Layers 0–27) | **70.00%** | 70.00% | 100.00% | 0.7000 | -0.0250 | 0/0 | p=1.0000 |
| **W2_KO_LatentToAns_Early** | Block `Latents -> Answer` (Early Layers 0–9) | **70.00%** | 70.00% | 100.00% | 0.7500 | +0.0250 | 0/0 | p=1.0000 |
| **W2_KO_LatentToAns_Mid** | Block `Latents -> Answer` (Mid Layers 10–18) | **70.00%** | 70.00% | 90.00% | 0.7375 | +0.0125 | 0/0 | p=1.0000 |
| **W2_KO_LatentToAns_Late** | Block `Latents -> Answer` (Late Layers 19–27) | **70.00%** | 70.00% | 100.00% | 0.6750 | -0.0500 | 0/0 | p=1.0000 |
| **W2_KO_ImgToLatent_All** | Block `Image -> Latents` (All Layers 0–27) | **70.00%** | 70.00% | 100.00% | 0.7750 | +0.0500 | 0/0 | p=1.0000 |
| **W2_KO_Total_Sever** | Block BOTH `Image -> Latents` & `Latents -> Answer` (0–27) | **70.00%** | 70.00% | 100.00% | 0.7000 | -0.0250 | 0/0 | p=1.0000 |

## 2. Experiment W4a: Counterfactual Cross-Image Latent Activation Patching (N=550)

- **Original Unpatched Accuracy:** `75.00%`
- **Patched Accuracy (with opposite-answer image latents across all 28 layers):** `75.00%`
- **Prediction Unchanged Rate:** **`100.00%`** (Flip Rate: `0.00%`)
- **Mean Counterfactual Log-Odds Gap (`|m_cf - m_orig|`)**: `0.5469`
- **Mean Patched Log-Odds Shift (`|m_patched - m_orig|`)**: `0.0312` (`5.71%` of counterfactual signal transferred)

## 3. Experiment W4b: Pre-RMSNorm Layerwise Geometry (Anisotropy-Corrected, N=143)

| Layer | `<|latent_pad|>` Raw Cos | `<|latent_pad|>` Centered Cos | `" ."` Pause Centered Cos | Question Centered Cos | `<|latent_pad|>` Eff. Rank | `" ."` Eff. Rank | `<|latent_pad|>` Vis Align | `" ."` Vis Align |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (Embed)** | 1.0000 | **1.0000** | **1.0000** | 0.9979 | 1.00 | 1.00 | -0.9999 | -0.9991 |
| **1** | 0.9949 | **0.9989** | **0.9787** | 0.7653 | 3.42 | 1.39 | -0.9845 | -0.8902 |
| **4** | 0.9635 | **0.9791** | **0.9207** | 0.5572 | 2.23 | 2.35 | -0.1350 | -0.0663 |
| **8** | 0.8892 | **0.8842** | **0.8568** | 0.3993 | 1.98 | 2.80 | 0.0185 | 0.0779 |
| **12** | 0.8046 | **0.7748** | **0.7484** | 0.3079 | 4.39 | 5.92 | -0.0032 | 0.0801 |
| **14** | 0.7707 | **0.7750** | **0.7630** | 0.3251 | 5.31 | 6.94 | -0.0266 | 0.0553 |
| **18** | 0.7636 | **0.7578** | **0.7550** | 0.3050 | 6.21 | 7.69 | -0.2276 | -0.1814 |
| **20** | 0.7751 | **0.7514** | **0.7721** | 0.3048 | 5.49 | 8.13 | -0.3354 | -0.3293 |
| **24** | 0.8189 | **0.7172** | **0.7105** | 0.1762 | 4.62 | 8.64 | -0.4896 | -0.5010 |
| **27** | 0.8296 | **0.7308** | **0.7475** | 0.2389 | 3.55 | 8.44 | -0.5636 | -0.6401 |
| **28 (Pre-Norm Final)** | 0.7562 | **0.8100** | **0.8431** | 0.3960 | 2.94 | 5.23 | -0.7865 | -0.8049 |
