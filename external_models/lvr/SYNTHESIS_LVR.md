# Master Synthesis Report: Latent Visual Reasoning (LVR)

> [!WARNING]
> **Verification status (2026-11-03):** historical summary audited against the source repository and arXiv paper. Verified findings are documented in [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md).


**Paper**: *Latent Visual Reasoning* (`arXiv:2509.24251`, ICLR 2026 Submission)  
**Authors**: Bangzheng Li et al. (`VincentLeebang/lvr`)  
**Target Models**: `vincentleebang/LVR-7B`, `vincentleebang/LVR-3B`  
**Reference**: [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md)

---

## Executive Summary of Verified Findings

1. **Latent Span Supervision and Sequence Length Mismatch**:
   During SFT (`src/train/monkey_patch_forward_lvr.py` L241–244, L329–334), latent input embeddings are overwritten with ground-truth ViT embeddings from the region of interest (ROI) and supervised with MSE loss against subsequent patch embeddings. The training latent length $T_v$ equals the total patch count of the bounding box (median 28, mean 66, max 5,096 across 434,265 Visual-CoT training bboxes). In contrast, inference decoding truncates to fixed quotas $K \in \{4, 8, 16\}$, covering only the initial top-left raster strip of the ROI.

2. **RL Replay Context Mismatch and Token Masking**:
   In `src/trainer/grpo_trainer.py` L1483–1570, `_replay_lvr_to_collect_states` teacher-forces latents under `no_grad` before passing them to the policy forward pass (`src/train/monkey_patch_forward_lvr_rl.py` L135–142). Due to an indexing shift, the replayed states are offset by one position and overwrite `<|lvr_start|>`, creating a 0.20–0.37 nat log-probability divergence between rollout and loss evaluation. Furthermore, the masking loop (L746–752) excludes `<|lvr_start|>` itself from the policy loss objective.

3. **Status of Adaptive Termination Mechanisms**:
   The learnable Latent End Token mechanism drops accuracy significantly (-41.9% on $V^*$ Bench and -52.7% on MMVP, `paper_src/tables/ablation.tex` L37), while Mode Switching Loss collapses the latent count to zero (`4_experimetns.tex` L74). Released checkpoints rely on hardcoded step budgets ($K \in \{4, 8, 16\}$).

4. **Comparative Performance of 3B Variants**:
   In `paper_src/tables/sft_rl.tex` (L10–11), the base Qwen2.5-VL 3B model was commented out in LaTeX. Base Qwen2.5-VL 3B scores higher than LVR 3B and LVR_RL 3B (8 steps) on $V^*$ (74.87% vs. 67.02%), MMVP (59.0% vs. 55.33%), and BLINK Counting (69.17% vs. 55.83%), while LVR 3B scores higher on IQ-Test (30.67% / 32.0% vs. 28.0%) and JigSaw (52.67% vs. 50.0%).

5. **Disclosed and Undisclosed Training Parameters**:
   The released Stage 1 7B checkpoint (`training_args.bin`) matches `scripts/finetune_lvr_stage1_7b.sh` with learning rate $1 \times 10^{-5}$, $\lambda_{\mathrm{LVR}} = 0.1$, and `max_packed_tokens = 16384` (`4 * 4096`). In Stage 2 RL scripts (`finetune_lvr_stage2_3b.sh` L19, L51), the learning rate is configured to $5 \times 10^{-6}$ (compared to $1 \times 10^{-5}$ stated in `4_experimetns.tex` L27) and visual resolution is capped at 2,560 tokens (compared to 5,120 in SFT).
