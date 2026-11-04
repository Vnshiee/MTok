# Mirage — Pre-registration of the Latent-Token Ablation Study

**Registered:** 2026-11-03, before any GPU run. No accuracy of the released checkpoint under any condition below has been computed by us at the time of writing; the only executions so far are CPU runs on a tiny randomly initialised model (`diagnostics/verify_mirage_autograd.py`, 29/29 checks) and CPU dry runs of the runner.
**Companion documents:** `MIRAGE_ABLATION_STUDY_DESIGN.md` (conditions, telemetry, rationale), `CODE_VS_PAPER_TRACE.md`, `Gaps_Mirage.md`.
**Deviation policy:** any change to this document after the first GPU run is logged in §10 with date, reason, and whether it was made before or after seeing outcome data.

---

## 1. Frozen inputs

| Item | Pin |
|---|---|
| Paper | arXiv 2506.17218 (`paper_src/`, identical to the e-print; `STEP1_INGESTION_MANIFEST.md` §2) |
| Code | `repo/` = UMass-Embodied-AGI/Mirage @ `53f26de2682025e146781c5b198ec93bdfe4c4d6` (vendored transformers fork 4.50.0.dev0) |
| Target model | `Miiche/vsp_spatial_planning_direct_sft` @ `d1b6fdce3c30ac99283a49945a73877de36fe36a` (`stage2`, `latent_size = 4`) |
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| VSP planning test set | `repo/data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz` → `test_direct.jsonl` (400 maps, levels 3–6, 100 each) |
| BLINK | `BLINK-Benchmark/BLINK` @ `a3666eb249237ba3d5eca8db21176cc47967e040`, subtask Jigsaw, split `val` (150) |
| V\* | `craigwu/vstar_bench` @ `d9ae62c903da0c98336e85c5ee89cd863b04b4da` (191) |
| MMVP | `MMVP/MMVP` @ `37eafecab8a3940c50c2ade5b36de69dbc99a8cf` (300 questions, 150 pairs) |

Analysis code, frozen by SHA-256 (a changed hash is a deviation, §10):

| File | SHA-256 |
|---|---|
| `ablation_study/run_mirage_ablation.py` | `{{SHA ablation_study/run_mirage_ablation.py}}` |
| `ablation_study/summarize_ablation_results.py` | `{{SHA ablation_study/summarize_ablation_results.py}}` |
| `ablation_study/run_all_ablations.sh` | `{{SHA ablation_study/run_all_ablations.sh}}` |
| `model_loader.py` | `{{SHA model_loader.py}}` |

---

## 2. Research question

Mirage trains Qwen2.5-VL-7B to emit a short block `<|latent_start|>` + K latent positions + `<|latent_end|>` before the answer; at each latent position the model's own last-layer post-norm hidden state is fed back as the next input embedding ({{LINE repo/transformers/src/transformers/generation/utils.py 3418 "cur_latent_embeds = outputs.hidden_states"}}–L3419). The released checkpoint was trained with K = 4. The study asks which of three mechanisms accounts for the checkpoint's VSP planning accuracy.

| ID | Hypothesis | Mechanistic statement |
|---|---|---|
| **H1** | Genuine latent compute | The vectors fed at the latent positions carry question-specific information that the answer depends on. Replacing them changes accuracy. |
| **H2** | Format / stop-gate lock | Accuracy depends on the presence of the `<|latent_start|> … <|latent_end|>` block (the output format the model was trained to produce, including a positional count of latent slots before `<|latent_end|>`), not on the vectors inside it. |
| **H3** | Feature copying / shortcut | The latent vectors approximately reproduce visual features of the question's own image, which are already in context. Removing them costs little (the source stays in context); inserting features of a different image interferes. |

The three hypotheses are made mutually exclusive by their predictions for the zero and donor conditions (§4).

---

## 3. Conditions, outcome and unit of analysis

All runs use the runner `ablation_study/run_mirage_ablation.py` at batch size 1 with KV cache; run names are `<benchmark>__<model_mode>__K<quota>__<intervention>__<entry>`.

**Reference R** = `vsp_planning__target__K4__none__forced`: target model, forced latent entry, exactly 4 latents (the training latent size), each the model's own hidden state, greedy decoding, repetition penalty 1.05 (checkpoint `generation_config.json`).

**Confirmatory family on VSP planning** (each compared with R on the same 400 questions):

| Condition | Run name suffix | What is fed at the 4 latent positions |
|---|---|---|
| C1 zero | `target__K4__zero__forced` | zero vector |
| C2 noise | `target__K4__noise__forced` | isotropic Gaussian rescaled to the norm of the model's own latent, seeded by question id and step |
| C3 donor | `target__K4__donor__forced` | the 4 latents R produced for a different question (VSP: a different map; multiple choice: a different correct option where one exists), fixed by a seeded assignment |
| pause | `target__K4__pause__forced` | input embedding of the text token `.` |
| mean | `target__K4__mean__forced` | per-step mean of R's latents over all 400 questions |
| K0 | `target__K0__none__forced` | no latent position: `<|latent_start|><|latent_end|>` only |
| suppress | `target__Kauto__suppress__natural` | no latent block: `<|latent_start|>` is banned during decoding |

The intervention definitions are in `model_loader.py` ({{LINE model_loader.py 259 "def apply(self, qid: str, step: int, natural):"}}–L281); donor assignment is `build_donor_map` ({{LINE model_loader.py 284 "def build_donor_map"}}); the K = 0 guard is {{LINE ablation_study/run_mirage_ablation.py 100 "if a.quota == 0 and a.intervention not in"}}.

**Primary outcome:** per-question binary correctness. VSP: the boxed action sequence is executed by the repository's simulator `simulate_vsp` on the question's `map_desc` ({{LINE repo/src/task.py 122 "def simulate_vsp"}}); correct iff it reaches the goal without falling into a hole. Unparsed answers (no `\boxed{}`) count as incorrect. Multiple-choice benchmarks: `parse_mc` ({{LINE ablation_study/run_mirage_ablation.py 216 "def parse_mc"}}); an unparsed answer counts as incorrect.

**Effect:** Δ_X = acc(X) − acc(R) on paired questions, in accuracy points.

---

## 4. Decision rules

For each condition X in the confirmatory family:

* **X < R** ("X lowers accuracy"): two-sided exact McNemar p, Holm-adjusted within the confirmatory family, < 0.05, **and** the upper bound of the 95% paired-bootstrap CI of Δ_X < 0.
* **X ≈ R** ("X is equivalent to R"): the 90% paired-bootstrap CI of Δ_X falls within [−3, +3] points (two one-sided tests at α = 0.05).
* **X ? R** (inconclusive): neither of the above. An inconclusive comparison is reported as inconclusive and is never counted as support for H2 or H3.
* A significant increase (Holm p < 0.05, CI lower bound > 0) is reported as such and is inconsistent with all three hypotheses' predictions for that condition.

Predicted pattern per hypothesis (✓ = required for the hypothesis to be declared supported):

| Condition | H1 | H2 | H3 |
|---|---|---|---|
| C1 zero | < R ✓ | ≈ R ✓ | ≈ R ✓ |
| C2 noise | < R ✓ | ≈ R ✓ | — |
| C3 donor | < R ✓ | ≈ R ✓ | < R ✓ |
| mean | < R ✓ | ≈ R ✓ | ≈ R ✓ |
| pause | < R | ≈ R ✓ | — |
| K0 | < R ✓ | ≈ R ✓ | ≈ R |
| suppress | < R | < R ✓ | — |
| H3 alignment contrast in R (§6) | — | — | 95% CI lower bound > 0 ✓ |

* **H1 supported** iff every ✓ in the H1 column holds. It is additionally labelled **content-carrying** if, in C3 donor, the donor-answer adoption rate (§6) is ≥ 0.10 with a 95% Wilson lower bound > 0.05.
* **H2 supported** iff every ✓ in the H2 column holds. If all H2 rows hold except suppress, which is also ≈ R, the outcome is recorded as **H2-null: the latent block has no measurable effect**, distinct from H2.
* **H3 supported** iff every ✓ in the H3 column holds.
* Exclusivity: H1 needs zero < R, while H2 and H3 need zero ≈ R; H2 needs donor ≈ R, while H3 needs donor < R. So at most one hypothesis can be declared supported.
* **No hypothesis supported:** any other pattern. The full pattern of <, ≈, ? per condition is reported.

---

## 5. Statistical procedures (as implemented in `ablation_study/summarize_ablation_results.py`)

| Quantity | Definition | Code |
|---|---|---|
| McNemar | b = #(R correct, X wrong), c = #(R wrong, X correct); p = min(1, 2·P(Bin(b+c, ½) ≤ min(b, c))); p = 1 when b + c = 0 | {{LINE ablation_study/summarize_ablation_results.py 48 "def binom_two_sided(b, c):"}} |
| Holm | step-down, monotone; confirmatory family = the 7 conditions of §3 (`--primary`, default `PRIMARY_DEFAULT`); a second Holm over all conditions of the benchmark is reported as a more conservative secondary column | {{LINE ablation_study/summarize_ablation_results.py 90 "def holm(pvals):"}}, {{LINE ablation_study/summarize_ablation_results.py 229 "PRIMARY_DEFAULT = ("}} |
| Paired bootstrap | resample questions with replacement, B = 10,000, numpy `default_rng(20261103)`; percentile CIs (95%: 2.5/97.5; 90%: 5/95) of mean(d), d = correct_X − correct_R | {{LINE ablation_study/summarize_ablation_results.py 68 "def paired_bootstrap(x, y, B=10000, seed=20261103, alpha=0.05):"}} |
| Equivalence | 90% bootstrap CI ⊂ [−m, +m], m = 0.03 (`--equiv_margin 0.03`) | {{LINE ablation_study/summarize_ablation_results.py 216 "equivalent=bool(n and -margin <= ci90[0] and ci90[1] <= margin)"}} |
| Per-condition interval | Wilson score 95%, z = 1.959964 | {{LINE ablation_study/summarize_ablation_results.py 58 "def wilson(k, n, z=1.959964):"}} |
| Standard error | sqrt(acc·(1 − acc)/n) | `describe` |

The summary command is fixed in `ablation_study/run_all_ablations.sh` (Phase 3): `--reference target__K4__none__forced --equiv_margin 0.03 --bootstrap 10000`, seed default 20261103.

---

## 6. Secondary measures used in the decision rules

**Donor adoption (C3).** Among questions q whose donor d satisfies "R's parsed answer for q ≠ R's parsed answer for d, both parsed", adoption = share of q for which the C3 answer equals R's answer for d; retention = share equal to R's answer for q ({{LINE ablation_study/summarize_ablation_results.py 129 "def donor_adoption(run, natural):"}}). For VSP the compared answer is the boxed action string. A 95% Wilson interval on adoption is computed from the reported n and adoption count.

**H3 alignment contrast (R).** For each question and latent step: max cosine between the model's latent and any image-token embedding of the question's own image (visual-merger output, the vectors the LM receives), minus the same quantity for the image of the previously processed question; averaged over steps, then a one-sample bootstrap 95% CI of the mean over questions (B = 10,000, seed 20261103) ({{LINE ablation_study/summarize_ablation_results.py 117 "def h3_contrast(run, B, seed):"}}). The first question of a run, and the first question after a resume, have no previous image and are excluded. Stage 1 trains latents toward pooled helper-image features with a cosine loss ({{LINE repo/src/trainer.py 25 "loss = 0.1 * ce_loss + sim_loss"}}), so a positive contrast is expected by construction and is necessary, not sufficient, for H3; the decision for H3 rests on the intervention pattern.

---

## 7. Sample size and sensitivity (computed before data collection)

N is fixed by the benchmarks; there is no optional stopping. Minimum true accuracy drop detectable with 80% power by the exact two-sided McNemar test at α = 0.05 (no multiplicity correction), by exact enumeration over the number of discordant pairs, as a function of the discordant-pair rate ψ:

| N | ψ = 5% | ψ = 10% | ψ = 20% |
|---|---:|---:|---:|
| 400 (VSP) | 3.25 pts | 4.75 pts | 6.50 pts |
| 300 (MMVP) | 3.75 pts | 5.50 pts | 7.50 pts |
| 191 (V\*) | 4.75 pts | 6.75 pts | 9.50 pts |
| 150 (BLINK Jigsaw) | not reachable (Δ ≤ ψ) | 7.50 pts | 10.75 pts |

Holm over the 7-condition family lowers the per-test threshold to as little as 0.05/7 ≈ 0.0071 for the smallest p, which raises these values.

Probability of declaring X ≈ R with the ±3-point margin when the true Δ = 0 (normal approximation, SE = sqrt(ψ/N)):

| N | ψ = 2% | ψ = 5% | ψ = 8% | ψ = 10% |
|---|---:|---:|---:|---:|
| 400 | 0.99 | 0.70 | 0.37 | 0.20 |
| 300 | 0.96 | 0.50 | 0.15 | 0.00 |
| 191 | 0.80 | 0.17 | 0.00 | 0.00 |
| 150 | 0.66 | 0.00 | 0.00 | 0.00 |

Equivalence can be established on VSP only if fewer than about 8% of questions change correctness under an intervention. The margin was set at ±3 points because ±2 points is not attainable at N = 400 for ψ ≥ 5%. These limits are the reason H2 and H3 are confirmatory only on VSP.

---

## 8. Reproduction check (run before interpreting §4)

`vsp_planning__target__Kauto__none__natural__testpy`: the paper's decoding (temperature 0.7, top-p 0.9, {{LINE repo/src/test.py 65 "temperature=0.7"}}–L66; the model enters and leaves latent mode by itself, capped at 8 latents as in {{LINE repo/transformers/src/transformers/generation/utils.py 3295 "MAX_LATENT_LEN = 8"}}). The paper reports 0.76 average VSP planning accuracy for "Ours (Direct)" ({{LINE paper_src/sec/4_experiments_arxiv.tex 55 "Ours (Direct)"}}).

* **Reproduced** if |acc − 0.76| ≤ 0.042 (binomial 95% half-width at p = 0.76, N = 400).
* Not reproduced otherwise. The confirmatory analysis is still run, because it compares conditions of the same checkpoint and does not depend on the paper's number. The report states which outcome occurred. The repository does not state that this checkpoint is the one behind the paper's table, so a mismatch is not by itself attributed to an error.

Also reported, not used in decisions: the greedy run `vsp_planning__target__Kauto__none__natural`, its Δ versus R, and the latent-count histogram (§9).

---

## 9. Exploratory analyses (no confirmatory claims)

1. **Latent-count curve (VSP):** accuracy for K ∈ {0, 1, 3, 4, 5, 10, 20} (`none`, forced), and C1–C3 × K ∈ {1, 3, 4, 5, 10, 20}. Reported with Δ versus R and Holm over all VSP conditions.
2. **Stop-gate telemetry (H2 context):** for forced K > 4, the share of questions whose model would emit `<|latent_end|>` right after the 4th latent and the mean step of the first such prediction, under `none` and under C1–C3 (`summarize_ablation_results.py` `stop_gate`, {{LINE ablation_study/summarize_ablation_results.py 150 "def stop_gate(run, rows):"}}); for auto runs, the share leaving latent mode by a predicted `<|latent_end|>` versus the 8-latent cap. A gate that fires after 4 latents equally under `none` and `zero` indicates position counting rather than content.
3. **Out-of-distribution benchmarks** (BLINK Jigsaw, V\*, MMVP): the released checkpoint was fine-tuned on VSP only, so every result there is exploratory. Same conditions at K = 4 plus the K sweep; Holm within benchmark.
4. **Axis 4 — base versus target under matched prompts:** `base__K0__none__natural` versus `target__Kauto__none__natural` and versus R on every benchmark: identical user text, image preprocessing, decoding and scoring; the only difference is the weights (and, for the target, the latent block). Reported as Δ with McNemar and bootstrap CI.
5. **MMVP pair accuracy** (both questions of a pair correct), per condition.
6. **Telemetry summaries:** latent norm, consecutive-step cosine, max cosine to image tokens, unparsed rate, generated length.

---

## 10. Data handling and deviations

* **Exclusions:** none. All 400 VSP questions are analysed. A question missing from a condition's output (incomplete run) is excluded from that paired comparison only, and the summary marks the run *(incomplete)*. A confirmatory conclusion requires all 7 family runs and R to be complete.
* **Unparsed answers** are incorrect. Their rate is reported per condition.
* **Sampling run:** one draw with seed 42 (`--seed` default, matching `seed_everything(seed=42)`, {{LINE repo/src/test.py 14 "seed_everything(seed=42)"}}); greedy runs are deterministic up to GPU kernel non-determinism.
* **Hardware or library deviations** (GPU type, CUDA, torch, attention implementation) are recorded from the runner's `.meta.json` and reported.
* **Deviation log:** none at registration.

| Date | Change | Before / after outcome data | Reason |
|---|---|---|---|
| — | — | — | — |
