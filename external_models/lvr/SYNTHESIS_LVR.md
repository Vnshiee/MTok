# Master Synthesis Report: Latent Visual Reasoning (LVR)

**Paper**: *Latent Visual Reasoning* (`arXiv:2509.24251`, ICLR 2026 Submission)  
**Authors**: Bangzheng Li et al. (VincentLeebang/lvr)  
**Model**: `vincentleebang/LVR-7B`  
**Audit Status**: Verified Line-by-Line with Zero Discrepancies  

---

## Executive Summary of Findings

1. **The ROI Feature Copying Paradox**:
   LVR claims to perform internal "mental reconstruction" of visual thoughts. However, because the entire image is already encoded into the prompt prefix by the frozen ViT, supervising latent hidden states via MSE loss against ground-truth ROI patches reduces mathematically to an auto-associative memory copy operation across self-attention.

2. **Policy Gradient Disconnection in Latent GRPO**:
   In `src/trainer/grpo_trainer.py` L740–L757, tokens between `<|lvr_start|>` and `<|lvr_end|>` are explicitly assigned `final_mask = 0`. Furthermore, the rollout replay `_replay_lvr_to_collect_states` is decorated with `@torch.no_grad()`. Consequently, zero policy gradient flows into the latent reasoning process; GRPO optimizes only the textual answer tokens conditioned on static latent vectors.

3. **Collapse of Adaptive Termination**:
   The learnable Latent End Token mechanism causes catastrophic performance degradation (-41.9% on $V^*$ and -52.7% on MMVP), while Mode Switching Loss collapses the reasoning trajectory to 0 steps. The model is therefore tethered to rigid 4/8/16 fixed token quotas.

4. **Suppression of Baseline Dominance**:
   In `tables/sft_rl.tex`, the authors commented out the base Qwen2.5-VL 3B model, which outperforms LVR 3B and LVR_RL 3B across all reported visual benchmarks ($V^*$, $V^*_{D.A.}$, BLINK).

5. **Discrepant Training Infrastructure**:
   Actual training run checkpoint metadata reveals sequence packing of 16,384 tokens (4x larger than paper claim of 4,096), an RL learning rate of $5 \times 10^{-6}$ (half of paper claim of $1 \times 10^{-5}$), and 50% image resolution downsampling.
