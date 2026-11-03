# `verification/` for Monet

These scripts back `../VERIFICATION_REPORT.md`. Everything runs on CPU.

## Setup

1. Re-create the audited code. `../repo/` is gitignored, so clone it and pin the commit:

   ```bash
   git clone https://github.com/NOVAglow646/Monet.git ../repo
   ```

   ```bash
   git -C ../repo checkout 0893999
   ```

2. Create the Python environment:

   ```bash
   python3 -m venv .venv && . .venv/bin/activate
   ```

   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch
   ```

   ```bash
   pip install "transformers==4.51.3" numpy
   ```

## Scripts

Run each command from `external_models/monet/`.

| Script | What it checks | Command | Result (2026-11-03) |
|---|---|---|---|
| `proxy_grad_identity.py` | Runs `compute_latents_only_loss` verbatim and compares its gradient with the latent-path gradient | `python verification/proxy_grad_identity.py repo` | Identical (max \|diff\| 0.0), norm 15.2, direct path excluded |
| `monet_vlpo_harness.py` | Runs the repo's rollout state machine, segment finder, latent insertion and Gaussian log-prob on a tiny Qwen2. Use `INIT=0.3 LAYERS=4` for non-degenerate dynamics. | `INIT=0.3 LAYERS=4 python verification/monet_vlpo_harness.py repo 10 0` | The repo's mean equals the next latent (cos 1.000). ‖∇‖ 12.6–21.3 vs 0.07–0.18 with the correct mean. The `ABS_VIS_*` env vars are unset, which raises `TypeError` |
| `monet_mask_check.py` | Runs the repo's `build_4d_attn` on a synthetic Stage-2 sequence | `python verification/monet_mask_check.py repo` | The auxiliary image is visible only to latents. Image tokens cannot see each other or themselves |
| `table_checks.py` | Checks that every table value is k/N-feasible and that Overall equals the weighted mean of its sub-scores | `python3 verification/table_checks.py paper_src/sec` | 100 values checked, 2 infeasible. All Overall values are consistent |
