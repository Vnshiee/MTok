# diagnostics/: status

Neither script tests LVR. Keep them only as historical record.

- **`test_roi_copying.py`** creates random vectors, defines the "LVR reconstruction" as `target + 0.001·noise`, and asserts cosine > 0.95. The conclusion is in the input. It does not load the LVR model or code.
- **`test_grpo_zero_grad.py`** re-types the masking loop from `grpo_trainer.py` and runs it on a hand-written 7-token tensor. It computes no gradient. Its output shows that the `<|lvr_start|>` position is masked too. That observation is real and is reported as N2 in `../VERIFICATION_REPORT.md`.

## Replacements

- **`../verification/lvr_rl_replay_harness.py`** runs the repo's actual generation, replay and loss code on a small model (CPU).
- **Experiments E-L1 to E-L6** in `../VERIFICATION_REPORT.md` §6 need the LVR checkpoint and a GPU.
