I'm attaching two files: `lvr_verification.patch` and `lvr_verification_files.zip`. They come from an independent verification of `external_models/lvr/` in the MTok repository. That verification found factual errors in our LVR audit, and also real issues the audit missed.

Do the tasks below in order, commit after each one, and push to my branch.

Ground rules:
- Do not soften or restate any finding without re-checking it against the source.
- Do not add any claim you have not verified with code or data.
- Never write words such as "illusion", "lie", "fiction", "conceal" or "paradox". Report facts and line numbers.

## Task 1: Apply the patch

Run this from the repo root:

```bash
git am lvr_verification.patch
```

If it doesn't apply cleanly, unzip `lvr_verification_files.zip` into `external_models/lvr/` instead, then make these changes by hand:

- Add the warning banners described in `external_models/lvr/VERIFICATION_REPORT.md` to the top of:
  - `Gaps_LVR.md`
  - `SYNTHESIS_LVR.md`
  - `CODE_VS_PAPER_TRACE.md`
  - `STEP1_INGESTION_MANIFEST.md`
  - `PREREGISTRATION_LVR.md`
  - `external_models/PIPELINE_STATE.md`
- Redact the third-party OCI credentials everywhere outside `.git`:
  - the access key that starts `cd2fce68…` becomes `<REDACTED_ACCESS_KEY_ID>`;
  - the secret that starts `bLInsfvX…` becomes `<REDACTED_SECRET_ACCESS_KEY>`;
  - the namespace `idhpuomb10ix` becomes `<REDACTED_NAMESPACE>`.

Then confirm with this command; it must print 0:

```bash
grep -rlE "cd2fce68e716|bLInsfvXCRP1T8|idhpuomb10ix" . --exclude-dir=.git | wc -l
```

## Task 2: Re-run the verification scripts

Clone the LVR repo into `external_models/lvr/repo`:

```bash
git clone https://github.com/VincentLeebang/lvr.git external_models/lvr/repo
```

Pin it to the audited commit:

```bash
git -C external_models/lvr/repo checkout 637a9f7e17e865b1372b65f182197fbc4a5a1449
```

Then follow `external_models/lvr/verification/README.md` and run every script. Confirm you get these results:

- **Manifest:** 87/87 present files match.
- **Table diff:** 9 cells changed, all baselines; IQ-Test and JigSaw winners flip.
- **Replay harness:** run it for K = 4, 8 and 16. The repo loss path should deviate from the rollout by 0.2–0.4 nats, and the fixed path by at most 1e-6.
- **ROI lengths:** median T_v = 28, mean ≈ 66, N = 434,265.
- **BLINK/VSR overlap:** 51/143 questions match; 0/51 images match.

If any number differs, stop and tell me rather than editing the report.

## Task 3: Rewrite `external_models/lvr/Gaps_LVR.md` from the verified facts

Use `VERIFICATION_REPORT.md` §2 (verdict table) and §3 (new findings N1–N9) as the only source.

- Keep the claims marked ✅, with their exact citations.
- Rewrite the claims marked ⚠️ to the narrower, correct form given in the report.
- Delete or correct the claims marked ❌:
  - §1.2: "interleaving impossible".
  - §3.1: "zero gradient" and "only the LM head is tuned".
  - §3.3: "every single benchmark". Add IQ-Test (LVR 30.67/32.0 vs base 28.0) and JigSaw (52.67 vs 50.0).
  - §4.1: the root cause. The budget exit is at `qwen_lvr_model.py` L796 and L804.
  - §4.2: the mechanism.
  - §6.1: "paper claims 4,096". The paper never states a packing length; 4096 is the `LST` script variable.
- Change §5.1 (contamination) to the measured result: 51/143 question-template overlap, 0/51 image near-duplicates.
- Add N1–N9 as new sections, with the harness numbers.

Every code block must be copied verbatim from the cited lines. Then check snippet fidelity:

```bash
python3 external_models/lvr/verification/verify_snippets.py external_models/lvr/Gaps_LVR.md external_models/lvr/repo external_models/lvr
```

It must report 0 "not_in_file" and 0 "in_file_wrong_lines". Change all `/home/devraj/...` links to repo-relative paths.

## Task 4: Fix the other LVR documents

- **`STEP1_INGESTION_MANIFEST.md` rows 6–7:** the stage-1 scripts actually use `MAX_INSTANCE_PER_BATCH=4`, `MAX_PACKED_TOKENS=$((4*4096))` (=16384) and `GRAD_ACCUM_STEPS=8`. Fix the row text, but keep the SHA256 values; they are correct.
- **`STEP1_INGESTION_MANIFEST.md` L292–301:** replace the invented `compute_loss` block with the real `repo/src/trainer/lvr_trainer.py` L235–243, verbatim.
- **`SYNTHESIS_LVR.md`:** rewrite it to match the corrected Gaps document. Remove "zero policy gradient" and "4× larger than paper claim of 4,096".
- **`PREREGISTRATION_LVR.md`:** replace H1–H3 and the nonexistent `test_adaptive_exit_entropy.py` with experiments E-L1 to E-L6 from `VERIFICATION_REPORT.md` §6. Keep their decision rules.
- **`CODE_VS_PAPER_TRACE.md`:** make every code block verbatim. Check it with `verify_snippets.py`; the target is 0 errors.
- **`PIPELINE_STATE.md`:** change the LVR rows from "Certified PASS (0 MISTAKES)" to "Formatting audited; factual verification: see `lvr/VERIFICATION_REPORT.md`".

## Task 5: Fix the cross-model documents that repeat LVR errors

In `external_models/UNIFIED_THEORY_OF_LATENT_TOKENS.md` and `MASTER_RESEARCH_PLAN.md`:

- Replace "RL policy gradients into latent reasoning are literally zero" and every similar statement with: "LVR's GRPO replays latent states without gradient through their generation (disclosed in the paper, 3_lvr.tex L97–106), and the replay in the loss forward does not reproduce the rollout context (N1)."
- Remove "ROI Feature Copying Paradox" as an established fact. Say instead: "untested interpretation; see E-L1/E-L3."
- Remove all "illusion", "lie" and "fiction" wording.
- Do not change the Monet or Mirage sections unless you re-verify them against their own sources the same way. If you don't re-verify them, add a banner: "not independently verified."

## Task 6: Replace the diagnostic scripts

Keep `diagnostics/README.md`. Move `diagnostics/test_roi_copying.py` and `diagnostics/test_grpo_zero_grad.py` into `diagnostics/legacy_tautological/`. Do not present them as evidence anywhere.

## Task 7: Report back

Give me:
- a table listing each task, done or not, and the commit hash;
- the outputs of every verification command;
- the final `verify_snippets.py` output for `Gaps_LVR.md` and `CODE_VS_PAPER_TRACE.md`;
- `git log --oneline -10`.

Do not run any GPU experiment, and do not claim E-L1 to E-L6 results unless you actually ran them.

Push to `claude/compassionate-fermi-m2zb7j`.
