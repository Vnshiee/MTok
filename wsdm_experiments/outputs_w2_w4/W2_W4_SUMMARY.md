# WSDM Experiments W2 & W4: Causal Attention Knockout, Activation Patching & Pre-Norm Geometry
**Evaluated Items:** N=700 (2-Way Patching N=550) | **SDPA vs. FlashAttention-2 Parity:** 96.86%

## 1. Experiment W2: Causal 4D Attention Mask Knockout (N=700)

| Condition | Intervention | Micro Acc | Macro Acc | Agree w/ Intact | Mean Margin | Δ Margin | McNemar b/c | McNemar p |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **W2_Intact** | No intervention (Baseline T1) | **66.57%** | 67.30% | 100.00% | 0.7146 | +0.0000 | 0/0 | p=1.0000 |
| **W2_KO_LatentToAns_All** | Block `Latents -> Answer` (All Layers 0–27) | **66.00%** | 66.77% | 96.43% | 0.6782 | -0.0363 | 14/10 | p=0.5413 |
| **W2_KO_LatentToAns_Early** | Block `Latents -> Answer` (Early Layers 0–9) | **66.71%** | 67.47% | 96.71% | 0.7111 | -0.0035 | 11/12 | p=1.0000 |
| **W2_KO_LatentToAns_Mid** | Block `Latents -> Answer` (Mid Layers 10–18) | **67.00%** | 67.78% | 96.14% | 0.7137 | -0.0008 | 11/14 | p=0.6900 |
| **W2_KO_LatentToAns_Late** | Block `Latents -> Answer` (Late Layers 19–27) | **67.29%** | 68.00% | 98.00% | 0.6809 | -0.0337 | 4/9 | p=0.2668 |
| **W2_KO_ImgToLatent_All** | Block `Image -> Latents` (All Layers 0–27) | **66.57%** | 67.27% | 97.14% | 0.7141 | -0.0004 | 10/10 | p=1.1762 |
| **W2_KO_Total_Sever** | Block BOTH `Image -> Latents` & `Latents -> Answer` (0–27) | **66.00%** | 66.77% | 96.43% | 0.6782 | -0.0363 | 14/10 | p=0.5413 |

## 2. Experiment W4a: Counterfactual Cross-Image Latent Activation Patching (N=550)

- **Original Unpatched Accuracy:** `76.91%`
- **Patched Accuracy (with opposite-answer image latents across all 28 layers):** `77.09%`
- **Prediction Unchanged Rate:** **`97.27%`** (Flip Rate: `2.73%`)
- **Mean Counterfactual Log-Odds Gap (`|m_cf - m_orig|`)**: `3.4757`
- **Mean Patched Log-Odds Shift (`|m_patched - m_orig|`)**: `0.0427` (`1.23%` of counterfactual signal transferred)

## 3. Experiment W4b: Pre-RMSNorm Layerwise Geometry (Anisotropy-Corrected, N=143)

| Layer | `<|latent_pad|>` Raw Cos | `<|latent_pad|>` Centered Cos | `" ."` Pause Centered Cos | Question Centered Cos | `<|latent_pad|>` Eff. Rank | `" ."` Eff. Rank | `<|latent_pad|>` Vis Align | `" ."` Vis Align |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (Embed)** | 1.0000 | **1.0000** | **1.0000** | 0.9979 | 1.00 | 1.00 | -0.9999 | -0.9992 |
| **1** | 0.9946 | **0.9988** | **0.9783** | 0.7593 | 3.20 | 1.39 | -0.9840 | -0.8873 |
| **4** | 0.9638 | **0.9785** | **0.9211** | 0.5492 | 2.22 | 2.34 | -0.1298 | -0.0659 |
| **8** | 0.8889 | **0.8832** | **0.8577** | 0.4043 | 1.97 | 2.77 | 0.0286 | 0.0778 |
| **12** | 0.8030 | **0.7692** | **0.7512** | 0.3039 | 4.37 | 5.85 | 0.0167 | 0.0922 |
| **14** | 0.7698 | **0.7721** | **0.7678** | 0.3235 | 5.30 | 6.78 | 0.0025 | 0.0713 |
| **18** | 0.7634 | **0.7538** | **0.7585** | 0.3055 | 6.17 | 7.66 | -0.2119 | -0.1740 |
| **20** | 0.7740 | **0.7421** | **0.7686** | 0.2955 | 5.40 | 8.09 | -0.3090 | -0.3051 |
| **24** | 0.8157 | **0.7083** | **0.7057** | 0.1655 | 4.64 | 8.56 | -0.4671 | -0.4873 |
| **27** | 0.8266 | **0.7237** | **0.7398** | 0.2279 | 3.71 | 8.39 | -0.5395 | -0.6261 |
| **28 (Pre-Norm Final)** | 0.7476 | **0.7997** | **0.8356** | 0.3758 | 3.06 | 5.33 | -0.7780 | -0.7919 |

### 5-Fold Cross-Validated Linear Probe Accuracy on `blink_sprel` (N=143)

- **Layer 14 Pooled**: `<|latent_pad|>` = `55.91%` vs. `" ."` Pause = `60.12%`
- **Layer 24 Pooled**: `<|latent_pad|>` = `88.15%` vs. `" ."` Pause = `82.54%`
- **Layer 28 (Final Pre-Norm) Pooled**: `<|latent_pad|>` = `84.58%` vs. `" ."` Pause = `81.75%`
- **Layer 28 `<answer` Decision Token**: **`88.15%`**
