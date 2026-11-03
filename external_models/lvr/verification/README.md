# `verification/`: reproducible checks behind `../VERIFICATION_REPORT.md`

Everything here runs on CPU.

## Setup

**1. Re-create the audited LVR repo.** `../repo/` is gitignored, so clone it:

```bash
git clone https://github.com/VincentLeebang/lvr.git ../repo
```

```bash
git -C ../repo checkout 637a9f7e17e865b1372b65f182197fbc4a5a1449
```

**2. Create the Python environment.** This is needed for the harness and the ROI script. `transformers==4.51.3` matches the RL environment in the upstream `environment.yaml`.

```bash
python3 -m venv .venv && . .venv/bin/activate
```

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

```bash
pip install "transformers==4.51.3" numpy pillow
```

## Scripts

| Script | Checks | Command (from `external_models/lvr/`) | Result (2026-11-03) |
|---|---|---|---|
| `verify_manifest.py` | SHA256, bytes and line counts of every manifest row | `python3 verification/verify_manifest.py STEP1_INGESTION_MANIFEST.md . repo` | 87/87 present files match; 8 gitignored |
| `verify_snippets.py` | Code blocks under each `repo/...#Lx-Ly` citation are verbatim in that range | `python3 verification/verify_snippets.py Gaps_LVR.md repo .` | Gaps 44/2/17 · Trace 19/1/7 · Manifest 18 not verbatim |
| `safe_args.py` | Reads `training_args.bin` with a stub unpickler that never imports pickled classes | `python3 verification/safe_args.py hf_metadata/LVR-7B/training_args.bin lvr packed learning_rate` | Matches `finetune_lvr_stage1_7b.sh` exactly |
| `table_draft_vs_final.py` | Commented draft Table 1 vs published Table 1; k/N feasibility | `python3 verification/table_draft_vs_final.py paper_src` | 9 baseline-only changes; IQ-Test and JigSaw winners flip |
| `roi_lengths.py` | Training latent length T_v from the repo's own `bbox_to_token_idxs` over Visual-CoT metadata | `python verification/roi_lengths.py repo <dir with *_cot_train.jsonl>` | N = 434,265; median 28; mean 66; max 5,096 |
| `blink_vsr_overlap.py` | BLINK Spatial-Relation vs Visual-CoT/VSR: question overlap and image near-duplicates (dHash) | `python verification/blink_vsr_overlap.py <metadata dir> <out dir>` | 51/143 questions match; 0/51 images match |
| `lvr_rl_replay_harness.py` | Runs the repo's own generation loop, replay functions, mask block and loss block on a tiny random Qwen2 model; compares log-probs with the rollout; tests the proposed fix | `python verification/lvr_rl_replay_harness.py repo <K> <seed>` | Repo loss path deviates 0.20–0.37 nats; fixed path ≤ 5e-7 |

Visual-CoT metadata comes from <https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/main/metadata>. Use the 12 `*_cot_train.jsonl` files, renamed to `<source>.jsonl` (e.g. `vsr.jsonl`).
