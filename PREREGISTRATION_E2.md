# E2 Design: Separating Answer Bias from Discrimination

**Status:** APPROVED and committed before any E2 GPU run (2026-10-28). Execution order: Tier A first, then checkpoint report, then Tiers B/C. Phase E5 is optional and not yet authorised.

---

## 0. What the paper claims, and what we're testing

**Paper (Mull-Tokens, Table 1):** Qwen2.5-VL-7B gets two-stage training: (1) a multimodal warm-up where the `<|latent_pad|>` tokens are supervised to hold image or text content, then (2) answer-only SFT with 20 latents prefilled as `<think>` + 20×`<|latent_pad|>` + `</think>\n`. The headline is that Mull (row f, BLINK avg **66.80**) beats the direct-answer fine-tune **DirAns** (row b, **61.40**), so **+5.40**. On the reasoning splits the gain is +7.78, and Jigsaw alone is +15.34. The paper attributes this to *latent reasoning*.

**What E1 already showed (15 conditions, N=700 each):** the gain depends on format, not on the latents. `<think></think>\n` with no latents is indistinguishable from 20 latents (p=0.54). K=0 collapses because the model falls back to long text CoT.

**The new threat (Claude's finding 1, which I verified on the raw logs):** accuracy mixes **discrimination** (can the model tell A from B?) with **bias** (a prior toward one letter or one position). On the four 2-way splits:

| Condition | Predicted "B" (2-way, N=550) | Gold "B" |
|---|:---:|:---:|
| Mull rows 1, 3, 4, 5, 7, 11–13, 15 | 59.6–61.8% | 49.3% |
| Mull row 14 (K=40) | 56.9% | 49.3% |
| **Mull-GRPO row 10** | **53.8%** | 49.3% |
| Mull rows 2, 6 (no prefill) | 62.9–64.4% | 49.3% |
| SAT rows 8, 9 | 46.2–49.6% | 49.3% |

*New detail, not in Claude's message:* GRPO cuts the B-bias by about 6 points. So part of the +0.86 GRPO "gain" could be a bias shift too.

### 0.1 A structural fact about BLINK that shapes the whole design

I checked the dataset directly. **Three of the five splits use one identical question text for every item** (Relative Depth: 1 distinct prompt, Jigsaw: 1, Multi-View: 1). IQ-Test has 2. Only Spatial Relation varies (141 distinct). On those splits the *only* thing that differs between items is the image. Two consequences:

- The "blind baseline" (Step 4) gets an identical input on every Multi-View item, since all MV images are 480×640. Blind accuracy there **equals the gold share of whichever letter the model prefers**. The language-prior question reduces exactly to letter bias × gold imbalance.
- Gold imbalance per split, i.e. the accuracy of "always answer B": SpRel 46.2%, RelDepth 48.4%, Jigsaw 47.3%, **Multi-View 55.6%**. A B-bias **helps only on Multi-View** and hurts slightly on the other three. So whether calibration shrinks Mull's lead over SAT is a real open question. It could go either way, and this design is built to answer it without assuming the result.

---

## 1. Verification of Claude's 5 points (done now, CPU only, from raw logs)

| # | Claude's claim | My independent check | Verdict |
|---|---|---|---|
| 1 | Mull ≈10 pt B-bias; SAT calibrated | Reproduced exactly (table above). Gold B = 271/550 = 49.3% | ✅ Confirmed. Plus the new GRPO detail |
| 2 | K=0 collapse includes degeneration | Row 2: **1** output hits the 1024 cap (IQ #52, "forty-eighth image…" loop), **3** with no `<answer>` tag, 5 with CJK runs. Row 6: **1** "便便便…" collapse (RelDepth #12) that hits the cap, 1 with no `<answer>`, 23 with CJK runs | ✅ Real but **small**: ≤4 of the 84 lost items in row 2. Needs its own line in the write-up. **Correction:** an earlier statement in this project ("zero outputs hit the 1024 cap") was wrong. One does in each of rows 2 and 6 |
| 3 | Total format lock: exactly 7 tokens | Rows 1, 3, 5, 8–15: 7 tokens on all 700, SD = 0. Rows 4 and 7 have a few 11-token outputs (`<answer>B. B is closer</answer>`) | ✅ Confirmed (rows 4 and 7 near-total) |
| 4 | Exp 2 / Exp 4 lack per-sample logs, so no significance tests | Old `exp2_*`/`exp4_*` folders do lack them, **but E1 rows 10–14 re-ran GRPO and K∈{1,5,10,40} with full per-sample logs and McNemar** (GRPO +0.86, p=0.46; all K p>0.09) | ⚠️ Already resolved. Claude seems to have looked at the superseded folders. The old "+0.81" figure should be retired |
| 5 | Exp 5 still reads post-RMSNorm layer 28, no centering, no question baseline | Confirmed: `experiment5_latent_dynamics.py` L125–130, L197 | ✅ Still unfixed. Proposed as optional Phase E5 (§8) |

---

## 2. Core technique: teacher-forced letter scoring

Every rigorous question here (calibration, margins, letter vs. position) needs the model's **probability over option letters**, not just its greedy output. A single forward pass gives it exactly. Here is why it's valid for this checkpoint:

1. **The latents are plain tokens at inference.** The model card states the latent code paths "are gated on a training-only flag, so they never run during generation". I confirmed this in `mmlatentdiscrete_qwen_vl.py` L2108: latent replacement only runs if `latent_hidden_states` is passed, and eval never passes it. One forward pass is therefore mathematically the same computation as greedy decoding's prefill.
2. **The decision is one token.** Every short generation in all 15 rows starts with token ids `[27 "<", 9217 "answer"]`. The letter is then a *single merged BPE token*: `>A`=23465, `>B`=36721, `>C`=25630, `>D`=36495. These ids are identical in the Mull and SAT tokenizers (verified).
3. **Procedure:** build the exact generation input (same chat template, same images, same `max_pixels=4194304`, same prefill), append `<answer` → `[27, 9217]`, run one forward, and read the logits at the last position for the candidate letters (A/B, or A–D for IQ).

**Per item we record:** full-vocab log-probs of each candidate, `logsumexp` mass on the candidates (should be ≈1), the overall top-1 token, and the option-restricted normalized log-probs.

### Hard gates: the run aborts if any fails
| Gate | Check |
|---|---|
| **G1 Prompt fidelity** | For every doc, the reconstructed user text must equal the logged `input` string of the matching E1 row, byte for byte. |
| **G2 Tokenization** | Assert `<answer>X</answer>` → `[27, 9217, id_X, 522, 9217, 29]` for every X, and that the prefill token count matches E1 (26 / 5 / 0 …). |
| **G3 Equivalence to generation** | On real inputs, teacher-forced argmax must match E1's greedy letter on **≥99%** of items for every format-locked condition. Every mismatch gets logged. Median candidate mass must be ≥0.95. |
| **G4 Integrity** | Exact N per (condition, variant, task). No duplicate `(task, doc_id)`. Outputs keyed and checksummed. Smoke runs (`--limit 5`) go to `scratch/` only, never to the results directory. |

---

## 3. Conditions

Same checkpoints, prompts, and prefills as E1, via the same `BLINK_PROMPT_MODE` / prefill strings.

| ID | Model | Prompt | Prefill | Mirrors | Tier |
|---|---|---|---|---|:---:|
| T1 | Mull | latents | think + 20 latents | Row 1 | **A** |
| T3 | Mull | latents | `<think></think>\n` | Row 3 | **A** |
| T8 | SAT | sft | empty | Row 8 (DirAns proxy) | **A** |
| T9 | SAT | latents | empty | Row 9 | B |
| T10 | Mull-GRPO | latents | think + 20 latents | Row 10 | B |
| T4 / T5 / T7 | Mull | … | dots / `latent_end` / sft+latents | Rows 4, 5, 7 | B |
| T11–T14 | Mull | latents | K = 1, 5, 10, 40 | Rows 11–14 | C |
| T2F / T6F | Mull | latents / sft | **empty, then forced `<answer`** | *new* | C |

> [!IMPORTANT]
> **T2F/T6F are not the same measurement as rows 2/6.** Rows 2 and 6 let the model free-run into text CoT, which can't be teacher-forced. T2F/T6F force an immediate answer with no tags at all. That's a new condition, and it answers a sharp question directly: *with neither latents nor tags, does Mull still answer as well as row 1 when it's prevented from rambling?* If yes, the K=0 collapse is 100% generation mode.

> [!WARNING]
> **T8 is a proxy for the paper's DirAns, not a verified copy.** The `array/Qwen2.5-VL-SAT` card says "60% SAT, 40% Video-R1 answers". The paper's DirAns used "the training datasets listed above", which plausibly includes Zebra-CoT. Our row 8 per-split scores also differ a lot from paper row b (SpRel 81.8 vs 74.1, RelDep 79.0 vs 87.1, Jigsaw 68.0 vs 58.7, MV 63.9 vs 57.1). Every Mull-vs-DirAns statement must say "vs. the released SAT baseline". `array/Qwen2.5-VL-SATBase` is in the cache, but only its README, no weights.

---

## 4. Input variants (per condition)

| Variant | Applies to | Images | Text | Purpose |
|---|---|---|---|---|
| **V0 real** | all 700 | original | original | Step 3 margins; gate G3 |
| **V1 blind** | all 700 | uniform grey (128) canvases, **same count and size** as the item's images, so the visual token count and RoPE layout are unchanged | original | Step 4 language-prior baseline |
| **V2 swap** | 550 two-way | original | option *contents* exchanged, letters stay A-then-B | position/letter/content decomposition |
| **V3 relabel** | 550 two-way | original | original content order, **letters reversed**: `(B) c1` / `(A) c2` | separates the letter "B" from "second position" |
| **V4 swap+relabel** | 550 two-way | original | `(B) c2` / `(A) c1` | completes the 2×2 |
| **CF content-free** | 27 probes per task | {grey, black, white} at the sizes of 3 seeded-random items in that task, same image count | question stem and every option text replaced by {"N/A", "", "[MASK]"}. Each task's exact scaffold is kept ("Select from the following choices.", blank lines, number of options) | Step 1, the Zhao et al. (2021) intrinsic prior `p_cf(letter)` |

IQ-Test is excluded from V2–V4. Its options ("picture A" … "picture D") refer to labels drawn *inside* the image, so swapping the text would make the item incoherent. IQ still gets V0, V1 and CF.

**Cost:** about 700 + 700 + 3×550 + 135 ≈ 3,185 forward passes per condition, roughly 16–20 min each. Tier A ≈ 1 h, all 14 conditions ≈ 4–4.5 h on the free A100.

### 4.1 The exact 2×2 decomposition (V0, V2, V3, V4)

For a 2-way item with contents c1, c2, let `m = log p(B) − log p(A)` (option-restricted). Write **L** = literal preference for letter B, **P** = preference for the second position, **s** = preference for content c2 over c1. Under an additive model:

| Variant | Layout | m = |
|---|---|---|
| V0 | A:c1, B:c2 | s + L + P |
| V2 | A:c2, B:c1 | −s + L + P |
| V3 | B:c1, A:c2 | −s + L − P |
| V4 | B:c2, A:c1 | s + L − P |

Solving per item: **L = (m0+m2+m3+m4)/4**, **P = (m0+m2−m3−m4)/4**, **s = (m0−m2−m3+m4)/4**.

`s` with its sign set by the gold content (call it `s_gold`) is a **bias-free discrimination score**. *Debiased accuracy* = fraction of items with `s_gold > 0`. Caveat to report: layouts V3/V4 ("(B) before (A)") are unusual for the model, and Relative Depth's options mention the in-image point labels "A"/"B", which confounds letter and content on that split. Results are reported per split, with RelDepth flagged.

---

## 5. Metrics (all pre-specified)

| # | Metric | Scope | Answers |
|---|---|---|---|
| M1 | Raw TF accuracy (V0 argmax) | 700 | sanity vs. E1 (gate G3) |
| M2 | **Content-free calibrated accuracy**: `argmax p(k\|x) / p_cf(k)`, with `p_cf` per (condition, task), averaged over 27 probes (Zhao et al. 2021) | 700 | Claude's Step 2 |
| M3 | **Debiased 2×2 accuracy** (`s_gold > 0`) | 550 | strongest bias-free accuracy |
| M4 | **AUC** of `m0` for gold-B vs gold-A, per split, then N-weighted mean | 550 | threshold-free, *invariant to any constant bias* by construction |
| M5 | Balanced accuracy (mean of per-class recall) from the E1 generation logs | 550 | zero-GPU cross-check |
| M6 | Correct-vs-incorrect margin `d = ℓ(gold) − max_{k≠gold} ℓ(k)` | 700 | Claude's Step 3, continuous power |
| M7 | Blind accuracy and AUC (V1); image-necessity gap = M1 − blind | per split | Step 4 |
| M8 | L, P (mean ± bootstrap CI) and `p_cf(B)` | per condition | letter vs. position, intrinsic prior |
| M9 | Label-free batch calibration (subtract each letter's mean log-prob over the split; Zhou et al. 2024) | 700 | robustness for M2. Caveat: it partly absorbs true gold imbalance (MV) |

Everything is reported **micro (N=700)** and **macro (mean of 5 splits, the paper's "Avg")**. We've been reporting micro. Macro is what Table 1 uses.

> [!CAUTION]
> **Margins (M6) are compared only within a model.** Logit scale differs between checkpoints, so a Mull margin and a SAT margin aren't on the same scale. Cross-model comparisons (Mull vs. SAT, Mull vs. GRPO) use only scale-free metrics: M2, M3, M4, M5.

---

## 6. Pre-registered questions, tests, and decision rules

Shared settings: paired bootstrap with 10,000 resamples, stratified by split, seed 20261028. Exact McNemar for paired binary outcomes. Wilcoxon signed-rank for paired continuous outcomes. Holm correction within each family. α = 0.05.

**Q1 (primary). Does Mull's lead over the SAT baseline survive bias removal?** T1 vs. T8.
- Tests: McNemar on M3 correctness (N=550) and on M2 correctness (N=700). Paired-bootstrap CI of the AUC difference (M4).
- Report **share of gap explained by bias** = 1 − (debiased gap / raw gap), computed on the *same* 550 items.
- Outcomes: **"survives"** if the M3 gap > 0 with p < 0.05 *and* the M4 CI excludes 0. **"Explained by bias"** if the M3 CI includes 0 and contains less than half the raw gap. Anything else is **"partially explained"**.
- *My honest prior (recorded before any data):* because the B-bias only aligns with gold on Multi-View, I expect the gap to **mostly survive overall but shrink on MV**. This could easily be wrong, and that's the point of running it.

**Q2. H1 re-test with continuous power (within Mull): T3, T4, T5 vs. T1.**
- Wilcoxon on paired `d` (M6), Holm over 3. Effect size `d_z`.
- **Equivalence (TOST):** "equivalent" only if the 90% paired-bootstrap CI of the accuracy difference lies within **±3 pp** *and* |d_z| < 0.1. This replaces the weaker "not significant" statement with a positive equivalence claim, or a refutation of one.

**Q3. Letter vs. position.** One-sample tests of L ≠ 0 and P ≠ 0 per condition (M8), plus `p_cf(B)` from CF.

**Q4. GRPO.** T10 vs. T1 on M3/M4 (scale-free) and on L/P. Is the +0.86 a discrimination gain or a bias shift?

**Q5. Blind baseline.** M7 per split for T1, T8. Expectation by construction: on RelDepth, Jigsaw and MV, blind ≈ the gold share of the preferred letter. SpRel is the only split where blind can show real language-prior knowledge.

**Q6. K=0 mechanism.** T2F/T6F vs. T1 on M1/M3. If T2F ≈ T1, the row 2 collapse comes entirely from generation mode (text CoT plus the small degeneration share counted in §1).

---

## 7. Implementation plan (files)

| File | Role |
|---|---|
| `e2_bias/score_letters.py` | Loads the model **through the same `Qwen2_5_VL_MMLatent_simple.__init__` path** (same token additions and memory logic). Rebuilds messages exactly like `generate_until` (L401–523: images then text, no system message, template, `<\|im_end\|>` strip, prefill asserts). Variant builders V0–V4/CF. Enforces gates G1–G4. Writes `e2_bias/outputs/<cond>/<variant>.jsonl`. Resumable, deduplicated by key |
| `e2_bias/run_e2.sh` | Tier-ordered runner (A → B → C), `--limit` smoke mode to `scratch/`, one model load per checkpoint |
| `e2_bias/analyze_e2.py` | M1–M9, all tests in §6, writes `E2_SUMMARY.md` + `e2_results.json`. Phase-0 CPU analyses (M5, degeneration table, format-lock table) run on the existing E1 logs |
| `PREREGISTRATION_E2.md` | This document, committed **before** the first GPU run |

**Order:** commit pre-reg → Phase 0 (CPU, minutes) → smoke (`--limit 5`, gates only) → Tier A (~1 h) → **checkpoint report to you** → Tiers B and C (~3.5 h) → analysis → commit and push as Vanshika Gupta.

---

## 8. Optional Phase E5: fix the latent-dynamics experiment (Claude point 5)

Hidden-state index 28 is the post-final-RMSNorm output, so it isn't comparable to layers 0–27. Fix: analyze layers 0–27 (pre-norm residual stream), mean-center the vectors per layer before cosine similarity (raw cosine in LLM residual streams is dominated by a shared anisotropic direction), add a **question-token baseline** (same statistics on the user-text tokens) and a **dot-pause baseline** (T4's `" ."` tokens at the same positions), and use all layers instead of 7 checkpoints. About 30 min. It's independent of E2 and can run after it.
