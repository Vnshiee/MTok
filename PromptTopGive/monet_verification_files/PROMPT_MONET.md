I'm attaching `monet_verification.patch` and `monet_verification_files.zip`. They come from an independent verification of `external_models/monet/` in the MTok repository.

That verification found that every claim in `Gaps_Monet.md` is wrong or unsupported. It also found real issues in Monet that the audit missed.

Apply the LVR patch (`lvr_verification.patch`) before this one if you haven't already.

Do the tasks below in order, commit after each one, and push to my branch.

Ground rules:
- Do not restate any finding without re-checking it against the source.
- Never write words such as "illusion", "lie", "fiction", "severing" or "conceal". Report facts and line numbers.
- Do not claim any GPU result you did not run.

## Task 1: Apply the patch

Run this from the repo root:

```bash
git am monet_verification.patch
```

If it doesn't apply cleanly, unzip `monet_verification_files.zip` into `external_models/monet/`. Then add the banner text described in `VERIFICATION_REPORT.md` to the top of `external_models/monet/Gaps_Monet.md`.

## Task 2: Re-run the verification scripts

Clone the Monet repo into `external_models/monet/repo`:

```bash
git clone https://github.com/NOVAglow646/Monet.git external_models/monet/repo
```

Pin it to the audited commit:

```bash
git -C external_models/monet/repo checkout 0893999
```

Follow `external_models/monet/verification/README.md` and run all four scripts. Confirm you get these results:

- **`proxy_grad_identity.py`:** max |diff| 0.0 and a nonzero gradient norm.
- **`monet_vlpo_harness.py`** (with `INIT=0.3 LAYERS=4`, seeds 0, 1 and 2):
  - cos(latent_{j+1}, repo mean_j) = 1.000;
  - repo gradient norm about 12–22;
  - aligned gradient norm about 0.07–0.2;
  - Q4 raises `TypeError`.
- **`monet_mask_check.py`:** AUX_IMG rows see no AUX_IMG columns, and LATENT rows do see them.
- **`table_checks.py`:** 100 values checked, 2 infeasible (78.28 and 54.28).

If any number differs, stop and report it. Do not edit the report.

## Task 3: Rewrite `external_models/monet/Gaps_Monet.md`

Use `VERIFICATION_REPORT.md` §2 and §3 as the only source.

- Remove these claims:
  - §1.1: "VLPO severs gradients via `g.detach()`". It is SFT Stage 2 code (`src/trainer.py` L198–199) and it passes the latent-path gradient exactly.
  - §1.1.1: "student latents loaded offline". `load_offline_tensor` loads teacher targets.
  - §2.2: "Stage 3 drops teacher representations". Stage 3 aligns to them at L273, L301 and L312.
  - §3.1: "aligns to ViT intermediate blocks". The targets are LLM hidden states, and `alignment_layer` is a CLI choice.
- Rewrite §2.1 as unsupported.
- Add findings M1–M10, each with its exact file and line citations and the harness numbers.
- Every code block must be copied verbatim from the cited lines of `repo/` at `0893999`. Record that commit hash at the top.
- Use repo-relative links, not `/home/devraj/...`.

## Task 4: Fix the cross-model documents

**In `external_models/UNIFIED_THEORY_OF_LATENT_TOKENS.md`:**

- Monet column, "Token Representation": change it to "Autoregressively generated continuous latents (last hidden state fed back as next input); trained in SFT against precomputed teacher targets".
- Monet column, RL row: change it to "VLPO: Gaussian log-prob on latents. As implemented, the mean is the hidden state that produces the next latent (M1); disabled in the released code because `ABS_VIS_START_ID`/`ABS_VIS_END_ID` are never set (M2)".
- Monet column, "What the latents do": change it to "Not established; see experiments E-M1 and E-M6".
- Remove Monet from "Trap 3", and rename the trap without the word "Lie".
- In the conclusion, remove "Static snapshot lookups with severed policy gradients (Monet)".

**In `MASTER_RESEARCH_PLAN.md`:**

- Replace the Monet bullet (L17) with: "Monet: VLPO's latent log-prob uses the next latent's producer as the Gaussian mean (M1); VLPO is disabled in the released code (M2); latent positions differ between SFT and inference/RL (M3). The previously claimed 'detached proxy loss' is SFT Stage-2 latent-only backprop and is correct."
- Fix the Monet line in the directory tree (L49).

**In `external_models/PIPELINE_STATE.md`:** change the Monet row's verdict to "Factual verification: see `monet/VERIFICATION_REPORT.md`".

## Task 5: Clean up the Monet scripts

- Move `external_models/monet/forensic_audit_monet.py` and `check_monet.py` into `external_models/monet/legacy/`.
- Add a one-line README there: "These scripts list files and count words; they are not evidence."

## Task 6: Do not touch Mirage yet

Do not modify `external_models/mirage/`. Add this banner to the top of `external_models/mirage/Gaps_Mirage.md`:

> **Not independently verified. The Mull/LVR/Monet audits in this repo contained substantive errors; treat this document as unverified.**

## Task 7: Report back

Give me:
- a table listing each task, done or not, and the commit hash;
- the full output of every verification command;
- `git log --oneline -10`.

Push to `claude/compassionate-fermi-m2zb7j`.
