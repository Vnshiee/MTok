# LVR (`vincentleebang/LVR-7B`) — Comprehensive Empirical Ablation & Falsification Report

## 1. Executive Summary of Empirical Findings

---

## 2. Stage 1 SFT Supervision Counterfactuals (Axis 1B Autograd Audit)

| Condition | $\mathcal{L}_{\mathrm{CE}}$ | $\mathcal{L}_{\mathrm{LVR}}$ | $\mathcal{L}_{\mathrm{total}}$ | $\|\nabla_\theta \mathcal{L}_{\mathrm{total}}\|_2$ | Parallel Teacher-Forced Input |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `A1B-ORIG` | `7.987964` | `1.873381` | `8.175302` | `5.383632` | `True` |
| `A1B-NO-MSE` | `7.948569` | `0.0` | `7.948569` | `5.393829` | `False` |
| `A1B-VANILLA-SFT` | `8.090393` | `0.0` | `8.090393` | `6.94269` | `False` |
| `A1B-PERM-ROI` | `8.095262` | `1.896948` | `8.284956` | `5.383016` | `True` |
| `A1B-BKG-PATCH` | `8.095498` | `1.965783` | `8.292076` | `5.373488` | `True` |
| `A1B-NOISE` | `8.018429` | `2.119923` | `8.230421` | `5.356266` | `True` |
| `A1B-TRUE-COCONUT` | `7.883674` | `1.752963` | `8.058969` | `6.160099` | `False` |
| `A1B-COSINE-UNNORM` | `7.987964` | `0.938708` | `8.081835` | `5.383607` | `True` |

---

## 3. Stage 2 GRPO 'Ghost RL' Gradient Flow Audit (Axis 2)

| RL Variant | Unmasked Completion Tokens | `<\|lvr_end\|>` Unmasked | $\mathcal{L}_{\mathrm{GRPO}}$ | $\|\partial \mathcal{L}_{\mathrm{GRPO}} / \partial \mathbf{H}_{\mathrm{LVR}}\|_F$ | $\|(\partial \mathcal{L}_{\mathrm{GRPO}} / \partial \mathbf{H}_{\mathrm{LVR}}) \cdot (\partial \mathbf{H}_{\mathrm{LVR}} / \partial \theta_{\mathrm{LLM}})\|_2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `A2-GHOST-DEFAULT` | `4` | `True` | `11.876366` | `0.166883` | **`0.0`** |
| `A2-STRICT-MASK-END` | `3` | `False` | `11.511697` | `0.031362` | **`0.0`** |
| `A2-DIFF-REPLAY` | `4` | `True` | `11.876366` | `0.315795` | **`5.737087`** |
| `A2-UNMASK-LOGP` | `9` | `True` | `11.477979` | `0.177146` | **`0.0`** |
