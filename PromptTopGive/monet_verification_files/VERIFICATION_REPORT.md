# Monet Folder: Independent Verification Report

**Scope:** every file in `external_models/monet/`, checked against the primary sources:

- **Code:** the official repo `NOVAglow646/Monet`, cloned at HEAD `0893999`. That is the latest commit (2026-03-19) and the only state that existed when `Gaps_Monet.md` was written; the audit recorded no commit.
- **Paper:** the arXiv 2511.21395 v2 e-print and PDF.

**Date:** 2026-11-03.

**Method:** No claim is accepted from the existing documents. Each was re-derived from source, or tested with a script in [`verification/`](verification/). Everything here runs on CPU; GPU experiments are designed in §6 and **not run**.

---

## 0. Bottom line

1. **What the folder contains.** It holds less than the LVR folder:
   - one 3.4 KB gaps document with 5 claims;
   - two scripts that only list files or count words such as "detach" (against `/home/devraj/...` paths);
   - the LaTeX source;
   - a PDF text extraction.

   There is no code snapshot, manifest, checkpoint metadata or experiment.
2. **Every substantive claim in `Gaps_Monet.md` is wrong or unsupported.**
   - The headline claim says Monet's RL "severs" gradients via `g.detach()`. That code (`src/trainer.py` L11–42) belongs to **SFT Stage 2**, not to VLPO/RL.
   - It implements exactly the gradient the paper describes (supplement L77–82). I verified it numerically: the gradient equals the latent-path gradient, max |diff| = 0.0.
   - The same wrong claim is repeated in `UNIFIED_THEORY_OF_LATENT_TOKENS.md` and `MASTER_RESEARCH_PLAN.md`.
3. **The audit missed real, verified problems in Monet, mostly in VLPO, the paper's headline RL contribution.**
   - **M1:** the VLPO Gaussian log-prob scores each latent against the hidden state that produces the *next* latent. On-policy, the correctly specified term would be ≈ gradient-free, because the rollout adds no noise.
   - **M2:** the released code never sets the env vars that enable VLPO in the actor. It silently falls back to GRPO, with latent slots filled by token embeddings.
   - **M3:** latents occupy different positions in SFT and in inference/RL.
   - **M4:** the released Stage-3 script cannot run.
   - **M5–M10:** undisclosed reward and evaluation choices, test-set selection of K, loss-weight discrepancies, and table inconsistencies.

---

## 1. Artifact verification

| Check | Result |
|---|---|
| `paper_src/` vs arXiv v2 e-print | All 24 shared files byte-identical. 5 figure PDFs (`overview`, `SFT2`, `dataset_construction`, `obs_acc`, `latent_size`) are missing because MTok's `.gitignore` excludes `external_models/**/*.pdf`. Not an audit error, but the K-scaling figure is unavailable in the repo. |
| `paper_text.txt` vs `pdftotext` of arXiv v2 | Same content (whitespace-normalized similarity 0.996); differences are token order from the extractor version. |
| Code snapshot | **None in MTok.** `repo/` is gitignored and the audit recorded no commit. Re-created at `0893999`; see `verification/README.md`. |
| Paper vs RL script | LR 1e-6, batch 64, 8 rollouts, temperature 0.5, σ = 10, K = 10 and accuracy threshold 0.6 all match `RL/examples/vlpo_train.sh` and `config_monet.yaml`. |
| Paper vs SFT scripts | LR 1e-5 and grad-accum 16 match. Mismatches are in M8. |

---

## 2. Verdicts on `Gaps_Monet.md`

| § | Claim | Verdict | Evidence |
|---|---|---|---|
| 1.1 | VLPO avoids true policy gradients; `compute_latents_only_loss` with `g.detach()` (trainer L39) and the proxy (L41) "completely sever" gradient flow | ❌ | **Wrong component:** `compute_latents_only_loss` is called only in `CustomTrainerSFT_STAGE2.compute_loss` (L198–199). VLPO lives in `RL/verl/workers/actor/dp_actor.py`. **Wrong math:** for `proxy = Σ v·stop_grad(∂L/∂v)`, ∂proxy/∂θ = (∂L/∂v)(∂v/∂θ), i.e. the gradient flows *through the latents* into θ, and only the direct non-latent path is removed. That is the paper's stated "latent-only BP" (`X_suppl.tex` L77–82; ablation "w/o latent-only BP"). Verified with the repo function verbatim (`verification/proxy_grad_identity.py`): identical to the latent-path gradient (max \|diff\| 0.0), norm 15.2, direct path excluded. |
| 1.1.1 | "Continuous latent representations are computed offline and loaded from disk (`load_offline_tensor`)" | ❌ | `load_offline_tensor` loads **teacher targets**: observation-token reps (Stage 2) and target latents (Stage 3). The student's latents are generated **online** in a latent-mode forward (trainer L160–163, L278–290; model L1914–1918). |
| 2.1 | Offline teacher snapshots "disable adaptive dynamic reasoning" | ❓ | Caching fixed distillation targets is standard and stated in the paper ("we fix h* and maximize cosine similarity"). No evidence for the conclusion. |
| 2.2 | "Stage 3 drops the ground-truth teacher representations → massive drift" | ❌ | Stage 3 **aligns** student latents to precomputed teacher latents: `load_offline_tensor(..., rep_type="latent")` at L273, with `loss_type = ['ce','alignment']` and `loss = ce + alignment_weight·alignment` at L301 and L312. The paper says the same (L150–163). |
| 3.1 | "Hard-codes `alignment_layer='all_layers'`, forcing latents to match ViT intermediate blocks" | ❌ | `all_layers` is a *default argument* of `load_offline_tensor`; the real value is a CLI choice (`all_layers`/`last_layer`, `utils.py`). The targets are teacher **LLM** hidden states, not ViT blocks. Aligning all layers is the paper's stated design (L155). |
| header | "CVPR 2026 submission" | ⚠️ | The source only carries a CVPR author-kit template comment (`arxiv_main.tex` L1); acceptance isn't established by the source. |
| `forensic_audit_monet.py`, `check_monet.py` | Presented as forensic audit | — | They list files and count words (`detach`, `policy`, …) in a repo that isn't in MTok. They test nothing. |

---

## 3. New findings (verified, missed by the audit)

### M1. VLPO's Gaussian log-prob uses the wrong mean (severity: high)

**Paper (Eq. 9–10).** The latent at position *t* is treated as a sample from N(h^θ_t, σ²I), where h^θ_t is the latent the policy produces in the same context. So the ratio is exp(−‖h^old_t − h^θ_t‖²/2σ²).

**Code.**
- *Rollout* (`RL/monet_models/vllm/monet_gpu_model_runner.py` L1099–1121, L1305–1339): the hidden state output at position q−1 is fed as the input embedding at position q.
- *Actor:*
  - It inserts the recorded latents at the same positions (`dp_actor.py` L114: `arange(s_pos, e_pos)`; model L1858). This is correct: text log-probs match the rollout to 2e-5 (bf16-level 6e-3 with fp16-stored latents).
  - It then computes `compute_latent_log_probs(latent_poss, latents, hidden_states[-1])`, using `last_hidden_state[0, latent_poss]` (L151), the output **at** q, as the mean for the latent **input at** q.
  - That output is what produces the **next** latent.

**Harness.** `verification/monet_vlpo_harness.py` runs on a tiny Qwen2 model. It uses the repo's state machine, segment finder, insertion line and log-prob function verbatim; 3 seeds, non-degenerate dynamics.

| Measure | Repo | Correct mean (`latent_poss − 1`) |
|---|---|---|
| cos(latent_j, mean used for it) | −0.21 … 0.28 | 1.000 |
| cos(latent_{j+1}, repo mean for j) | **1.000** | — |
| On-policy latent log-prob (σ = 10) | −0.56 … −0.67 per latent | ≈ −1e-6 |
| ‖∇θ‖ of the latent term at θ = θ_old | 12.6 – 21.3 | 0.07 – 0.18 (fp16 rounding only) |

**Implications.**

1. **What the term actually optimizes.** The implemented term is ∝ −‖h_q − z_q‖². Here z_q is the previous step's output, which equals the latent input at q. With positive advantage it pulls each latent toward its predecessor; with negative advantage it pushes them apart. In other words it regularizes **consecutive-latent similarity**, not "pulling the policy toward a good-action latent" (paper L199).
2. **Why the stated objective would be inert.** If the mean were correct, the term would be essentially gradient-free on-policy:
   - The rollout latent is the mean itself. Noise is commented out at L1331: `sampled = mu  # + torch.randn_like(mu)`, although the paper describes the latent as "a sample drawn from a latent Gaussian".
   - Its gradient at θ_old is (z − h_θ)/σ² ≈ 0, except for fp16 storage (L1337) and vLLM/HF numerical differences, or drift across the 4 minibatches of each rollout batch (64 × 8 rollouts vs global batch 16 × 8 GPUs).

### M2. VLPO is disabled in the released code (severity: high, reproducibility)

`dp_actor.py` L195–196 read `int(os.getenv("ABS_VIS_START_ID"))` and `int(os.getenv("ABS_VIS_END_ID"))`. **No file in the repo sets these.** `monet_rl_patch.py` L63–64 sets `LATENT_START_ID`/`LATENT_END_ID`, a different name, and `vlpo_train.sh` sets neither.

What happens:
- `int(None)` raises `TypeError` (reproduced).
- The bare `except Exception` at L232 prints a warning and continues with `latents = None`.
- So the actor (1) inserts **no** latents (the latent slots get the token embeddings of whatever tokens vLLM sampled there), and (2) computes **no** VLPO term.

In the harness, text log-probs then differ from the rollout by 0.14–0.41 nats with default initialization and 6–11 nats with init 0.3. The paper's results may have been produced with the variables set privately, but as released, `vlpo_train.sh` runs GRPO on a context the rollout never used.

(`vllm_rollout_spmd.py` L174 also reads a third name, `ABS_VIS_LATENT_SIZE`, which is unused.)

### M3. Latent positions differ between SFT and inference/RL (severity: high)

| | SFT (all stages) | Inference and RL rollout |
|---|---|---|
| Layout | `<abs_vis_token>` + K × `<abs_vis_token_pad>` + `</abs_vis_token>` | `<abs_vis_token>` then latents |
| Where latents go | **pad** positions s+1 … s+K | positions s … s+K−1 |
| First latent | h_s, the output at the start token | h_{s−1} |
| Start token | keeps its learned embedding | its input is **overwritten** |
| Sources | `src/main.py` L187–188, L258; model L1736, L1914–1918 | runner L1327–1333 sets `pending` on the activation step, and the next step overrides the start token's own row (L1099–1116; same logic in `inference/vllm/monet_gpu_model_runner.py` L1655–1666) |

At deployment the model never sees the `<abs_vis_token>` embedding it was trained with, and the latent block is shifted by one position relative to SFT. The RL actor follows the rollout convention, so VLPO trains under a layout that SFT never used.

Related: the inference runner's `LATENT_SIZE` defaults to **0** (L329). A start token then yields one latent, not zero. `export LATENT_SIZE=10` was only added to `inference/example.sh` on 2026-03-19 (commit `211a8f8`).

### M4. Released Stage-3 script fails immediately (severity: medium, reproducibility)

`script_examples/sft_stage3.sh` passes `--stage "avt_v5_stage2"`. `src/utils.py` L19 restricts `--stage` to `choices=['sft_stage1','sft_stage2','sft_stage3']`, so the repo's own parser exits with code 2 (reproduced).

The 2026-02-04 commit `ebcaba8` edited the same block (`--epochs 3→2`) without fixing it.

### M5. Reward and evaluation details not in the paper (severity: medium)

| What the paper says | What the code does |
|---|---|
| "accuracy reward (1/0) + format reward" (L197) | `0.9·acc + 0.1·format − 0.001·(len − ref_len)⁺`. With `repetition_penalty=true`, acc can be **−1**. Correctness: rule-based, then an **LLM judge (Gemini-2.5-Pro)** for rule failures (`monet_reward_function.py` L176–249; `vlpo_train.sh`). |
| "We adopt VLMEvalKit for fair evaluation" | The README (L181) says exact-match judging was **replaced by an API judge**. The paper doesn't say whether the non-starred baselines were judged the same way; starred rows are copied from other papers. |

### M6. Test-time K chosen per benchmark on the test set (severity: medium)

`4_experiment.tex` L41: "At test time, we select the best latent size from {8, 10, 12, 16}". This is test-set hyperparameter selection, and the baselines get no equivalent choice.

On VisualPuzzles, Monet's +2.31 over base is 27 items of 1,168. A max over 4 configurations biases that upward. Report a fixed K (the training K) alongside the selected one.

### M7. Stage-2 attention mask goes beyond the paper (severity: medium)

The paper (L112) says auxiliary-image embeddings are attended **only** by latent embeddings, "not by subsequent text tokens". Running the repo's `build_4d_attn` (`verification/monet_mask_check.py`) confirms that part. It also shows something the paper doesn't describe:

- auxiliary-image tokens cannot attend to **each other or themselves**, because the diagonal is masked (`utils.py` L570–575: every non-latent row is blocked from the image columns, including the image rows themselves);
- the `<abs_vis_token>` start token also cannot see the image.

The comment "only A_i (and not others) can see I_i" suggests the intra-image blocking is unintended. Either way, the Stage-2 student encodes auxiliary images without intra-image attention, unlike the Stage-1 teacher it is aligned to.

### M8. Loss weights and schedule differ from the paper (severity: low–medium)

| Item | Paper | Code / scripts |
|---|---|---|
| Stage-2 alignment weight | α = 2.0 on Σ_layers (1 − cos) | `emphasize_latent_weight` 2.0 × proxy(`alignment_weight` 2.0 × **mean** over layers and positions) → effective 4.0 per layer-mean (`trainer.py` L198–200; model L247) |
| Stage-3 alignment | β = 2.0 on Σ_layers | 2.0 × layer-**mean** |
| Observation-token CE up-weighting | not mentioned | `ce_emphasize_factor` 2.0 (Stage 1), 4.0 (Stages 2–3) |
| Stage-1 length | "4 epochs" (L41) vs "3885 steps (3 epochs)" (supplement table) | `--epochs 4` |
| Stages 2–3 length | 1000 steps (≈1.03 epoch) | `--epochs 2` (Stage 3 was 3 before `ebcaba8`) |
| `--latent_can_see_all_previous` | — | `store_true` with `default=True`, so it can never be disabled (`utils.py` L64) |

### M9. Truncated backpropagation across latent steps (severity: low, disclosure)

In the SFT latent forward, each latent is `prev_hidden.clone()`, which keeps its graph (model L1916). But it is fed to the next step as `latent_embed.detach()` (L1924). Gradients therefore reach θ through each latent's own step, but **not through the chain of earlier latents**. That is a one-step truncation the paper doesn't mention.

### M10. Table inconsistencies (severity: low–medium)

- **The same model has two scores.** "Monet-SFT" on MME-RealWorld-Lite is **55.03** in the SFT ablation (`3_method.tex` L140) and **52.68** in the component ablation (`4_experiment.tex`). The other four columns are identical. With 55.03, VLPO's gain on that benchmark is +0.47 rather than +2.82.
- **Two values are impossible given N** (`verification/table_checks.py`, 100 values checked):
  - vanilla SFT HRBench4K-FSP 78.28 (N=400 only allows multiples of 0.25; the row's overall 68.38 implies 78.25);
  - DeepEyes MME-RealWorld-Lite 54.28 (not k/1919).
- **Bolding:**
  - VisualPuzzles–Spatial bolds vanilla SFT 29.02 while Pangea-7B's 32.5* is higher;
  - Deductive bolds Monet's 47.50 but not the tied base model;
  - V*–Attribute bolds both 83.48 and 84.35.
- **Positive check:** every "Overall" equals the weighted mean of its sub-scores (V* 115/76; HRBench 400/400).

---

## 4. Errors in MTok documents about Monet

| Document | Error |
|---|---|
| `Gaps_Monet.md` | All five claims (§2). Links point to `/home/devraj/...`, and no commit is recorded. |
| `UNIFIED_THEORY_OF_LATENT_TOKENS.md` | "Precomputed offline teacher continuous thought vectors" as Monet's token representation; "Detached proxy gradient (`src/trainer.py` L39)" in the **RL** column; "Static snapshot lookups with severed policy gradients"; "Broken Policy Gradient Lie". All are wrong (§2, M1–M2). |
| `MASTER_RESEARCH_PLAN.md` L17, L49 | Same claim ("avoids backpropagating through time… offline static proxy loss"). |
| `PIPELINE_STATE.md` | "Monet: 🟢 Certified PASS" after one round, on a document whose claims don't hold. |

---

## 5. What should be said about Monet instead

Monet's SFT pipeline implements what the paper describes:

- latent-only backpropagation via the gradient-injection proxy;
- Stage-3 alignment to Stage-2 target latents;
- auxiliary image → latent attention.

There are also several undisclosed choices: M7–M9, an effective alignment weight of 4 rather than 2, and 2–4× up-weighting of observation tokens.

The problems that matter are in **VLPO** and **deployment**:

- **M1:** the implemented latent term regularizes consecutive latents rather than scoring rollout latents under their own policy mean. With the mean fixed, it would be nearly inert because the rollout is noise-free.
- **M2:** as released, VLPO doesn't run at all.
- **M3:** SFT, RL and inference place latents differently.
- **M6:** test-time K is selected on the benchmarks.

Each of these is concrete, line-referenced and reproducible on CPU.

---

## 6. Experiments designed (need Monet-7B and a GPU; **not run**)

| ID | Question | Protocol | Decision rule |
|---|---|---|---|
| E-M1 | Does the VLPO latent term matter? | Retrain VLPO from Monet-SFT with: (a) the repo term; (b) the correct mean `latent_poss − 1`; (c) the correct mean with real sampling `z = μ + σε`; (d) no latent term (GRPO with latents inserted). Fixed K = 10, 3 seeds. | If (a) ≈ (d) and (b) ≈ (d), VLPO's reported gains do not come from latent policy gradients. |
| E-M2 | How large are M2 and M3 in practice? | On 200 rollouts, compare text log-probs: rollout vs actor with env vars set, unset, and with the SFT layout. | Mismatch above the bf16 floor ⇒ the released pipeline trains on mismatched contexts. |
| E-M3 | Does the layout shift hurt Monet-SFT? | Evaluate Monet-SFT with the inference runner as released vs patched to the SFT layout (latents on pad positions s+1…s+K). | A significant difference (McNemar) ⇒ M3 costs accuracy. |
| E-M4 | Is K-selection inflating results? | Report all K ∈ {8, 10, 12, 16} per benchmark, plus fixed K = K_train, with 10k-bootstrap CIs. | If the selected-K gain over the baseline loses significance at fixed K, revise the claims. |
| E-M5 | Is the judge doing work? | Re-score Monet and every non-starred baseline with VLMEvalKit's default exact match and with the API judge. | Report both; the gain should hold under both. |
| E-M6 | Do latent contents matter at inference? | Replace latents with zeros, the mean image token, or latents from another image; and K = 0 (forced `</abs_vis_token>`). | If ≈ baseline, the latents are not the operative mechanism (same logic as LVR E-L1). |

---

## 7. Honest assessment

The existing Monet audit is not usable. It attributes an SFT function to RL, misreads a standard gradient-injection trick as "severing", calls Stage 3 alignment-free when it aligns explicitly, and presents word counts as forensics.

Monet's code does have real, consequential problems:
- VLPO as implemented does not compute what the paper says (M1);
- VLPO as released does not run (M2);
- training and deployment disagree on where latents go (M3).

None of these needs rhetoric. They need the line numbers and the harness outputs above, plus E-M1/E-M3 to measure their effect. Replace `Gaps_Monet.md` with §2–§3 of this report, and correct the Monet rows in the two cross-model documents.
