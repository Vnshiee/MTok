# LVR Folder: Independent Verification Report

**Scope:** every file in `external_models/lvr/`, checked against the primary sources:
- the official repo `VincentLeebang/lvr`, re-cloned at the audited HEAD `637a9f7e17e865b1372b65f182197fbc4a5a1449`;
- the arXiv v2 PDF and e-print;
- the HF checkpoint `vincentleebang/LVR-7B`;
- Visual-CoT metadata and BLINK.

**Date:** 2026-11-03.

**Method:** No claim is accepted from the existing documents. Each one was re-derived from source, or tested with a script in [`verification/`](verification/). Experiments requiring the 7B model and GPU were designed in §6 and executed in [`ablation_study/`](ablation_study/) with full empirical findings synthesized in [`LVR_ABLATION_RESULTS_REPORT.md`](LVR_ABLATION_RESULTS_REPORT.md).

---

## 0. Bottom line

1. **Ingestion is exact.**
   - All 87 present files listed in `STEP1_INGESTION_MANIFEST.md` match their recorded SHA256, byte count and line counts.
   - The 6 gitignored artifacts I re-downloaded (arXiv PDF + e-print, `trainer_state.json`, `vocab.json`, `merges.txt`, `tokenizer.json`) also match their recorded hashes.
2. **The analysis layer (`Gaps_LVR.md`, `CODE_VS_PAPER_TRACE.md`, `SYNTHESIS_LVR.md`) is a mix.**
   - Some findings are correct and useful: RL learning-rate and resolution discrepancies, the missing 3B base reference, the fragile answer parser, reliance on fixed steps, and leaked credentials.
   - Several headline claims are wrong or overstated: "interleaving is impossible", "RL gradient is identically zero / only the LM head is tuned", "worse than base on every single benchmark", "paper claims 4,096 packed tokens", the latent-end root cause, contamination.
   - Some code blocks presented as verbatim are not in the code. One has different logic.
3. **None of the "experiments" in this folder tests anything about LVR.**
   - `diagnostics/test_roi_copying.py` builds `reconstruction = ground truth + 0.001·noise` and asserts cosine > 0.95.
   - `diagnostics/test_grpo_zero_grad.py` re-implements a mask loop on a hand-written 7-token tensor.
   - The pre-registered `test_adaptive_exit_entropy.py` does not exist.
   - 16 of the 20 `.py` files generate or patch Markdown.
   - "Certified PASS (0 MISTAKES)" therefore reflects formatting rounds, not factual verification.
4. **The most important problems in LVR are ones the audit missed.** I verified these:
   - **N1:** the GRPO_latent loss forward does **not** restore the rollout's latent context. Answer-token log-probs deviate by 0.20–0.37 nats in a controlled harness. A two-change fix makes them exact.
   - **N2:** `<|lvr_start|>` is excluded from the policy loss, which contradicts the paper's mechanism for "encouraging LVR".
   - **N3:** training latent length is the ROI size (median 28, mean 66, max 5,096 tokens over 434,265 Visual-CoT samples), but inference truncates at K ∈ {4, 8, 16}.
   - **N4/N5:** between the commented draft and the published Table 1, only baseline cells changed. Two of those changes turn LVR losses into ties or wins. The final Vision-R1 numbers are far below chance on binary tasks, which is consistent with answer-parsing failure.

---

## 1. Verification of artifacts (positive results)

| Check | Result | Evidence |
|---|---|---|
| Manifest SHA256 / bytes / `wc -l` / `nl -ba` | **87/87 match** | `verification/verify_manifest.py` |
| Gitignored artifacts re-downloaded from arXiv/HF | **6/6 hashes match** | `sha256sum` vs manifest |
| Released checkpoint config vs launch script | `training_args.bin` matches `scripts/finetune_lvr_stage1_7b.sh` **exactly**: LR 1e-5, 2,500 steps, λ_LVR 0.1, MSE, `max_packed_tokens` 16384 = 4×4096, `max_instance_per_batch` 4, grad-accum 8, `mode_switch_loss` False, `stage_2` False | `verification/safe_args.py` (stub unpickler, executes no pickled code) |
| Paper vs checkpoint (SFT) | LR 1e-5 and 2,500 steps match `4_experimetns.tex` L23 | — |
| Every LVR/baseline number in Table 1 (draft and final) is achievable as k/N | yes; 3 final cells are double-rounded upward (e.g. 97/115 = 84.348% → draft 84.35 → final 84.4) | `verification/table_draft_vs_final.py` |
| Training pool size | 434,265 Visual-CoT train bboxes, consistent with the paper's "438k" | `verification/roi_lengths.py` |

---

## 2. Claim-by-claim verdicts on `Gaps_LVR.md`

Legend: ✅ verified · ⚠️ partly true / overstated · ❌ false · ❓ unsupported (no evidence either way)

| § | Claim | Verdict | What the source actually shows |
|---|---|---|---|
| 1.1 | SFT is an "identity copy" of ROI patches already in context | ⚠️ | **Mechanics correct.** `monkey_patch_forward_lvr.py` L241–244 overwrites LVR inputs with ROI ViT embeddings, and L329–334 applies MSE between the hidden state at s−1 and the embedding at s. **The interpretation is untested.** Targets are a *question-dependent subset* of the image tokens, so choosing them is localization, which is the non-trivial part. The paper itself names Argus (re-injection of ROI tokens) as its closest comparator (L43). The diagnostic offered as proof is tautological (§5). |
| 1.2 | Interleaving is "architecturally impossible"; regex allows only one block; inference is single-stage | ❌ (mostly) | `data_utils.py` L36–51 builds one LVR block **per placeholder**, each followed by its text segment, so multi-bbox data interleaves. The format regex (`reward_funcs.py` L53) uses lazy `.*?`, which **does** match several LVR blocks. Its real defects are different: it is anchored at `^` (any leading text → 0), and without `re.DOTALL` any newline → 0. At inference, `qwen_lvr_model.py` L1039–1043 re-enters LVR whenever `<\|lvr_start\|>` is emitted. Fair version: the data (one bbox per sample) and the reward *favour* one leading block, and the paper never demonstrates interleaving. |
| 1.3 / 2.2 | Frozen ViT and merger contradict "unification"; LVR tokens get 1D RoPE | ⚠️ | The frozen modules are the paper's *stated* design (`4_experimetns.tex` L18). It is true that `<\|lvr\|>` tokens are text tokens for `get_rope_index`, so they get 1D positions. "Destroys 2D geometry / severely handicaps" is untested. |
| 2.1 | Teacher forcing → "catastrophic" exposure bias that "compounds exponentially" | ⚠️ | Train/test input mismatch is real: ground-truth ViT embeddings in training, own hidden states at inference. "Catastrophic" is contradicted by the paper's flat 4/8/16-step results (Table 1). The bigger, unmentioned mismatch is latent *length* (N3). |
| 2.3 | Packing "subverts efficiency"; full quadratic attention over pads | ⚠️ | The collator does split and right-pad (L880–906). But training uses FlashAttention-2 (`train_lvr.py` L132), whose HF path unpads using `attention_mask`, so pads add no attention cost. It is length-bucketed batching, with no scientific consequence. |
| 2.4 | Dummy-image ZeRO-3 workaround | ✅ (not a gap) | Standard practice; adds negligible compute. |
| 3.1 | `@torch.no_grad()` replay ⇒ gradient "identically ZERO"; RL "only tunes the language head"; "empirical fiction" | ❌ | `no_grad` replay is real (`grpo_trainer.py` L1482). The replayed states then enter a gradient-carrying forward (L1012–1023), and the answer-token loss backpropagates into **all** LLM weights (`freeze_llm False`). Only the path *through the generation of* the latents is cut, and the paper describes exactly that replay (`3_lvr.tex` L97–106). The real defect is that the replay is *wrong* (N1). |
| 3.2 | LVR span masked from the loss | ✅ (design, not a flaw) | L742–757. Excluding latent-slot tokens is correct, since they are not the inputs actually used. Missed: `<\|lvr_start\|>` is masked too (N2). |
| 3.3 | 3B base hidden; LVR-3B worse "on every single benchmark" | ⚠️ | Numbers are correctly copied from the commented draft rows. **"Every single" is false:** LVR-3B beats base-3B on IQ-Test (30.67/32.0 vs 28.0) and JigSaw (52.67 vs 50.0), the two Table 2 columns the audit's table leaves out. Valid core: the published paper has no 3B base reference, and LVR_RL-3B (8 steps) trails base-3B by 7.85 pts on V*, 3.67 on MMVP and 13.34 on Counting (draft numbers). "To conceal" is speculation about intent. |
| 3.4 | 7B RL was run and hidden | ❓ | `finetune_lvr_stage2_7b.sh` differs from the 3B script only in names. A script's existence is not evidence it was run. The released checkpoint has `stage_2 = False`. |
| 3.5 | RL LR 5e-6 vs paper 1e-5; RL max pixels 2560 vs 5120 | ✅ | `finetune_lvr_stage2_{3b,7b}.sh` L19 and L51 vs `4_experimetns.tex` L27 and L18. |
| 4.1 | Latent-End collapse because the loop runs "until max_new_tokens = 512" | ⚠️ / ❌ | Numbers verified (`ablation.tex` L37). **Root cause contradicted by the code:** `_lvr_deocding_with_latentend` has a hard budget exit (`qwen_lvr_model.py` L796, L804). The failure to terminate is reported by the paper itself (L74). "Curse of dimensionality" is speculation. |
| 4.2 | Mode-switch BCE pushed `<lvr_end>` to 0 so the model "immediately emitted `<lvr_end>`" | ❌ (mechanism) | The explanation contradicts itself. The paper reports the collapse. The real design flaw is that the BCE acts on a raw logit through a sigmoid while inference uses softmax argmax (N6). |
| 4.3 | Fixed 4/8/16 step budgets | ✅ (disclosed) | It is the paper's declared method (`3_lvr.tex` L124, L129). |
| 4.4 | Argmax "garbage" tokens are appended to `input_ids` | ✅ | L1013 and L1055. The sharper consequence is in N3: after K steps, the next *real* input is whatever was predicted at the last latent step. |
| 5.1 | Benchmark contamination via Visual-CoT | ❓ → tested | `verification/blink_vsr_overlap.py`: 51/143 BLINK Spatial-Relation *questions* appear verbatim in Visual-CoT's VSR train subset (0/143 in GQA), but **0/51 images** are near-duplicates (min dHash distance 90/256; duplicates ≤ 20). This is in-domain template overlap, not leakage. (Heavy crops are not excluded.) |
| 5.2 | Fragile answer parser | ✅ | `evaluation.py` L52–61. See N5 for evidence that it matters. |
| 5.3 | "Standardized LMMs-Eval" is false | ⚠️ | The harness is custom (no `lmms_eval` import) and its parser differs from lmms-eval's. The paper says it follows LMMs-Eval *metrics* (accuracy), so "false" is too strong. |
| 5.4 | Evaluation prompts rewritten | ✅ | L63–71, L262–264. Routine for the model's own evaluation; it matters when applied to chain-of-thought baselines (N5). |
| 5.5 | Multi-image regression | ✅ | Admitted by the paper (L58). |
| 6.1 | "Paper claims 4,096 packed tokens" (4× discrepancy) | ❌ **fabricated** | The paper never states a packing length (`grep 4096\|4,096` over `paper_src/` finds nothing). 4096 is `LST`, the long-sequence *threshold* variable in the script. |
| 6.1 | Other matrix rows | mixed | LR and pixel rows ✅; 7B-RL row ❓; "zero policy gradient" ❌ (3.1); packing ⚠️ (2.3). |
| 6.2 | Plaintext cloud credentials in the upstream repo | ✅ | `scripts/vanilla_sft_7b.sh` L8–11, `evaluation/evaluation.py` L17–20. **MTok re-published the full secrets in 7 files.** They are now redacted in the working tree but remain in git history (commit `60f9d67`). |

---

## 3. New findings (verified, missed by the existing audit)

### N1. GRPO_latent's loss forward does not reproduce the rollout context (severity: high)

The paper (`3_lvr.tex` L97–106) says recorded latent states are "patched into the latent-reasoning positions … restoring the exact context preceding text generation and ensuring consistent conditional log-probabilities."

**How generation works.** In `_lvr_deocding_by_steps`, input position *j* ∈ {s+1, …, s+K} receives *h*<sub>j−1</sub>, where *s* is the index of `<|lvr_start|>`.

**How the replay differs.** `_replay_lvr_to_collect_states` (L1524–1570) runs the forward for position `pos−1` but gates it with `in_span(pos)`, and stores the state *before* that forward. This has two effects:

1. The input embedding at the `<|lvr_start|>` position itself gets overwritten. Generation never does this.
2. Every patched state is taken one step too early. The first patched state is exactly *h*(token before `<|lvr_start|>`).

**Harness.** `verification/lvr_rl_replay_harness.py` takes the repo's **own** `generate`/`_lvr_deocding_by_steps`, `_replay_lvr_to_collect_states`, `score_with_lvr_replay`, the L742–757 mask block and the L996–1034 loss block, all loaded from source text. It also uses the RL forward's patch rules from `monkey_patch_forward_lvr_rl.py` L129–142 verbatim. These run on a tiny random Qwen2 decoder under transformers 4.51.3, the RL environment in `environment.yaml`.

| K | max \|Δ log p\| on loss tokens: repo loss path vs rollout | `score_with_lvr_replay` | proposed fix |
|---|---|---|---|
| 4 (3 seeds) | 0.20 / 0.26 / 0.33 | 0.0 | 4.8e-7 |
| 8 (3 seeds) | 0.24 / 0.25 / 0.31 | 0.0 | 4.8e-7 |
| 16 (3 seeds) | 0.30 / 0.27 / 0.37 | 0.0 | 4.8e-7 |

Cosine similarity between the repo-patched states and the true rollout states at the 8 latent positions (K=8) is 0.43–0.50.

**Consequence.** The policy-gradient term is ∇ log π(y | a context that never occurred). Two separate effects make the objective inconsistent:
- **Context:** old and ref log-probs are computed with the exact `score_with_lvr_replay`, while the policy uses the faulty path.
- **Temperature:** see N7.

The magnitude on the real 3B model is unknown; it is cheap to measure (§6, E-L6).

**Fix (two changes, verified exact in the harness):**
1. Pass `lvr_mode_switch = ((pos-1 > lvr_start_idx) & (pos-1 <= configured_end_idx))` to the forward.
2. Move the `if update_mask.any(): … lvr_states[update_mask, step] = …` block to *after* `last_position_hidden_state` is updated.

### N2. `<|lvr_start|>` is never in the policy loss (severity: medium-high)

In the mask loop (L746–752), `active = True` is set *before* `if active: lvr_mask[b,t] = False`, so the `<|lvr_start|>` position itself is masked. The harness confirms this for all 9 configurations.

The paper credits GRPO with "encouraging the generation of the `<|lvr_start|>` token" (`3_lvr.tex` L77; `4_experimetns.tex` L61). With that token's log-prob excluded, the format reward cannot raise P(`<|lvr_start|>`) directly; it can only do so through side effects on shared weights.

### N3. Training latent length ≠ inference latent length (severity: high, conceptual)

Training uses one `<|lvr|>` per ROI ViT token (`data_utils.py` L49), and `fixed_num_of_lvr_tokens` is never passed by any caller. Running the repo's own `bbox_to_token_idxs` over all Visual-CoT train bboxes, with the training pixel limits (min 128×28², max 5120×28²), gives:

| Statistic | T_v |
|---|---|
| N | 434,265 |
| median | 28 |
| mean | 66 |
| 10th percentile | 6 |
| 90th percentile | 143 |
| max | 5,096 |

| K at inference | Training samples with T_v ≤ K | Median share of the ROI that K steps cover |
|---|---|---|
| 4 | 9.7% | 14.3% |
| 8 | 21.4% | 28.6% |
| 16 | 38.0% | 57.1% |

By source, median T_v ranges from 8 (InfographicsVQA) to 130 (VSR).

Two consequences:
- Targets are in raster order, so K steps can at most reconstruct the **top rows** of the ROI.
- The model is trained to emit `<|lvr_end|>` only after T_v steps. At step K it is usually at a position where the next-token label was masked during training, and that predicted token becomes the next real input.

The fact that accuracy is flat across K = 4/8/16 (Table 1), with most of each ROI never reconstructed, is itself evidence against "mental reconstruction of ROI semantics" as the operative mechanism. E-L1 and E-L4 in §6 test this directly.

### N4. Draft vs published Table 1: only baselines changed (severity: medium)

`paper_src/tables/main_table.tex` keeps the earlier 7B table as comments (L33–40, 2-decimal values). Comparing it to the published table (L71–80) with `verification/table_draft_vs_final.py`:

- No LVR cell changed beyond rounding. Nine baseline cells changed:
  - Qwen2.5-VL JigSaw: 53.33 → 52.0 (−2 items of 150).
  - PAPO IQ-Test: 30.00 → 29.3 (−1 item).
  - All seven Vision-R1 cells moved by 14–23 points.
- These two baseline changes flip the column winners:
  - **IQ-Test:** draft best is PAPO 30.0 vs LVR 29.33. Final is a tie at 29.3, with only LVR bolded.
  - **JigSaw:** draft best is base 53.33 vs LVR 52.67. Final is LVR 52.7, tied with PixelReasoner 52.7, with only LVR bolded.
- The text claims "leading results on Counting, IQ-Test, JigSaw" (L56).

The changes may well be legitimate re-evaluations, but they are undocumented. Either way, differences of 1–2 items out of 150 cannot support "leading" claims.

### N5. Final Vision-R1 numbers are below chance on binary tasks (severity: medium)

| Benchmark (2 options) | Draft | Final | Chance |
|---|---|---|---|
| JigSaw | 50.0 | **27.3** | 50 |
| MMVP | 61.33 | **46.7** | 50 |
| Spatial Relation | 83.22 | 66.4 | 50 |

A 7B model scoring well below chance on 2-way tasks points to answer extraction or formatting, not perception.

`evaluation.py` L55–60 returns the **first character** of the output when no `<answer>` tag is present, and `skip_special_tokens=False` keeps tokens such as `<think>`. So a tag-less chain-of-thought answer often parses as `<`, `T`, `B`, and so on. This is a hypothesis consistent with the numbers; test it with E-L5.

### N6. Mode-switch BCE trains a raw logit, but inference uses argmax (severity: low-medium)

`monkey_patch_forward_lvr.py` L1344–1350 applies `binary_cross_entropy_with_logits` to `logits[..., lvr_end_id]` alone. Exit at inference is decided by softmax argmax over the whole vocabulary, so the sigmoid of one raw logit can be driven to the target without changing the argmax. This is a plausible contributor to the reported collapse (paper L74) that the audit did not identify.

### N7. KL term compares two temperatures (severity: low)

`score_with_lvr_replay` uses `apply_temperature = 1.0` by default, and no caller overrides it; this produces the ref and old log-probs. The policy log-probs use `logits / self.temperature`, which is 0.9 in the scripts (L1033).

As a result, `per_token_kl` (L1072) is nonzero even when policy == ref (≈8e-4 nats/token in the harness). With β = 0.04 this is a small but systematic bias.

### N8. Reconstruction loss is the paper's Eq. 1 divided by H (severity: low)

The paper's Eq. 1 is (1/T_v) Σ_t ‖h_t − v_t‖². The code uses `nn.MSELoss()`, which averages over all T_v·H elements, so code = paper / H (verified: ratio 2.790e-4 = 1/3584). With λ = 0.1, the effective weight relative to the paper's equation is ≈ 2.8e-5. This matters for anyone reimplementing from the paper.

### N9. Two transformers versions (severity: low, reproducibility)

- `environment.yaml` pins 4.51.3; `requirements.txt` pins 4.54.0.
- The RL forward (`monkey_patch_forward_lvr_rl.py`) uses the <4.52 API (`self.visual`, `self.model` as the text model, `get_rope_index` on the model class), and `_replay_lvr_to_collect_states` uses the 4.51 signature of `_get_initial_cache_position`.
- Yet the README (L210) says RL may require ≥ 4.54.

---

## 4. Errors in the MTok LVR documents themselves

| # | Document | Error |
|---|---|---|
| D1 | `STEP1_INGESTION_MANIFEST.md` rows 6–7 | Lists `MAX_PACKED_TOKENS=8192`, `MAX_INSTANCE_PER_BATCH=8`, `GRAD_ACCUM_STEPS=4` for both stage-1 scripts. Both actually use 4×4096 = 16384, 4, and 8, even though the SHA256s in the same rows are correct. |
| D2 | `STEP1_INGESTION_MANIFEST.md` L291–301 | The "verbatim" `compute_loss` block is invented and has **different logic**: it gates on `loss_mode_switch is not None` and `loss_lvr is not None`, whereas the real `lvr_trainer.py` L240–243 gates on `self.args.mode_switch_loss` and `loss_lvr_lambda > 0`. |
| D3 | All three analysis docs | Snippet fidelity (`verification/verify_snippets.py`): many cited code lines are not present verbatim. Most are reformatted (joined lines, changed spacing, `export` added to Python assignments). Counts are in the table below. |
| D4 | `Gaps_LVR.md` §6.1 | "Paper claims 4,096 packed tokens": fabricated (see §2). |
| D5 | `Gaps_LVR.md` §3.3 | "On every single benchmark… substantially worse": false, because the table omits the two columns where LVR-3B wins. |
| D6 | `Gaps_LVR.md` §3.1, `SYNTHESIS_LVR.md` §2, `UNIFIED_THEORY_OF_LATENT_TOKENS.md` | "Zero policy gradient", "only tunes the language head": false (see §2). |
| D7 | `Gaps_LVR.md` §4.1 | Root cause contradicted by the hard budget exit at L796/L804. |
| D8 | `diagnostics/*.py` | Tautological. Neither imports the LVR code nor measures a gradient or a model state. |
| D9 | `PREREGISTRATION_LVR.md` | References a script that does not exist; none of the H1–H3 tests were run. H2's quantity (∂J/∂z_latent ≡ 0) is the wrong target; the right question is whether the replayed context equals the rollout's (N1). |
| D10 | `PIPELINE_STATE.md`, `AUDIT_STEP_*` | "Certified PASS (0 MISTAKES)": the audit rounds track formatting, links and LaTeX escapes. The Round-2 report body is a hard-coded string in `generate_round2_audit.py`. |
| D11 | `Gaps_LVR.md`, `AUDIT_STEP_2_ROUND_{1,2}.md`, five `.py` patch scripts | Re-published third-party cloud secrets in full (now redacted in the working tree; still in history). |
| D12 | `Gaps_LVR.md` paths | All links point to `/home/devraj/...`, and `repo/` is gitignored, so no reader of MTok could check any code claim. `verification/` now documents how to re-create `repo/` at the audited commit. |

D3 counts:

| Document | Snippet lines verbatim in cited range | Mis-cited lines | Not verbatim |
|---|---|---|---|
| `Gaps_LVR.md` | 44 | 2 | 17 |
| `CODE_VS_PAPER_TRACE.md` | 19 | 1 | 7 |
| Manifest | — | — | 18 |

---

## 5. What the folder's "experiments" actually test

| File | What it does | What it tests |
|---|---|---|
| `diagnostics/test_roi_copying.py` | Random vectors; sets "reconstruction" = target + 0.001·noise; asserts cosine > 0.95 | Nothing about LVR. The conclusion is built into the input. |
| `diagnostics/test_grpo_zero_grad.py` | Re-types the mask loop and runs it on a 7-token tensor | That a copy of the loop masks the span. (It also shows the `<\|lvr_start\|>` position masked, i.e. N2, without noticing.) No gradient is computed. |
| `check_*.py`, `scratch_audit*.py` | String/citation checks against `/home/devraj/...` | Document formatting |
| `make_*.py`, `fix_*.py`, `patch_*.py`, `apply_r2_fixes.py`, `clean_final_fixes.py`, `insert_aux.py`, `generate_round2_audit.py` | Write or patch Markdown from hard-coded strings | — |

---

## 6. Experiments designed and executed (E-L1 to E-L6)

> **Execution Update (2026-11-03):** All six experiments (E-L1 through E-L6) have now been executed on GPU 0 (NVIDIA A100-PCIE-40GB) using `vincentleebang/LVR-7B`. Full tabular metrics, attention weights, cosine similarity trajectories, and paired McNemar statistics are reported in [`LVR_ABLATION_RESULTS_REPORT.md`](LVR_ABLATION_RESULTS_REPORT.md) and [`PREREGISTRATION_LVR.md`](PREREGISTRATION_LVR.md) §2.

Run each with fp32 scoring of option letters (or bf16 with a reported precision floor), N = all items, paired McNemar with exact binomial p (capped at 1), and 10k-bootstrap CIs.

| ID | Question | Protocol | Decision rule |
|---|---|---|---|
| **E-L1** | Do latent *contents* matter? | At inference, with K ∈ {4, 8, 16}, replace the K latent states with: (a) the model's own (baseline); (b) the oracle first-K ROI ViT tokens; (c) random non-ROI image tokens; (d) the mean image token; (e) zeros; (f) K=0 (`<\|lvr_start\|><\|lvr_end\|>` forced). Data: V* (has bboxes) and BLINK. | If (c), (d), (e), (f) ≈ (a) within ±2 pts (TOST), the contents are not the operative mechanism. If (b) > (a) significantly, reconstruction quality limits performance. |
| **E-L2** | Is information routed through the latents? | Attention knockout per layer band: answer→latent, and latent→image. | No change from answer→latent knockout means the latents are bypassed. |
| **E-L3** | Do latents *localize*? | For each latent state, find the nearest image token (cosine). Score the hit rate inside the gold V* bbox against the bbox's area fraction (chance). | A hit rate above chance supports "re-attention/localization", the reframed form of the "copy" hypothesis. |
| **E-L4** | Is truncation (N3) costly? | Oracle injection of the first-K vs the full-T_v ROI tokens, followed by a forced `<\|lvr_end\|>`. | Full > first-K means the K-step decoder discards useful content. |
| **E-L5** | Is the baseline gap partly parsing? | Re-score every baseline's raw outputs with (i) the LVR parser, (ii) strict `<answer>` extraction, (iii) lmms-eval's BLINK extraction. Report parse-failure rates. | If Vision-R1's binary-task scores rise above chance under (ii)/(iii), N5 is confirmed. |
| **E-L6** | How large is N1 on the real model? | For about 200 RL rollouts from LVR-3B, compare per-token log-probs from the rollout logits vs the repo loss path vs the fixed path. One forward per rollout, no training. | A mean \|Δ\| clearly above the bf16 floor means the published GRPO_latent gradients were computed on corrupted contexts. Optionally retrain GRPO-3B with the fix and compare against Table 2. |

---

## 7. Honest assessment

- **Strongest part of the folder:** the ingestion. It is bit-exact and makes everything above reproducible.
- **Weakest part:** the step from code reading to conclusions. Several headline claims are rhetorical ("paradox", "illusion", "lie", "empirical fiction", "conceal"), several are wrong, and nothing was tested on the model. The same over-claiming carried into `UNIFIED_THEORY_OF_LATENT_TOKENS.md` and `MASTER_RESEARCH_PLAN.md`.
- **What I'd lead with in a paper or review:** N1 (RL replay faithfulness), N3 (train/test latent-length mismatch, together with flat-in-K accuracy), and E-L1/E-L5 once run. These are concrete, verifiable and new. N4/N5 belong in a careful, neutral "reporting and evaluation" note, not as allegations of intent.
- **Recommendation:** treat `Gaps_LVR.md` as superseded by this report. Rewrite its conclusions from §2–§3, delete or replace the two diagnostics with E-L1/E-L6, and re-run the pre-registration with the corrected hypotheses.
