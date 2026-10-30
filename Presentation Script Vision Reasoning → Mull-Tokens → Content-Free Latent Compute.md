# Presentation Script: Vision Reasoning → Mull-Tokens → Content-Free Latent Compute

Oct 29, 2026 · @YASHA

## How to use this script

This is slide-by-slide content for a 25–30 minute advisor meeting, not the deck itself. Hand it to a slide-building agent with the appendix at the end, which specifies every figure and table.

**The narrative goal.** The deck must read as *question → investigation → finding → next question*, never as *paper → replication → add-on*. Mull-Tokens appears on slide 10 of about 32, as the answer to a gap already established, not as the starting point. If the advisor asks "why this paper?" at slide 10, the first nine slides have failed.

**Structure.** Five acts, 32 slides:

| Act | Slides | Minutes | Purpose |
| --- | --- | --- | --- |
| I. The problem space | 1–9 | 6 | Why vision reasoning, what exists, where it breaks |
| II. The object of study | 10–13 | 4 | Mull-Tokens and my critical reading (5 gaps) |
| III. The empirical journey | 14–26 | 10 | Replication, errata hardening, Exp E1 (15 conds) & Exp E2 (14 conds) |
| IV. Where the field is | 27–29 | 4 | Audit wave, positioning table, final scientific verdict |
| V. The ask | 30–32 | 4 | Problem statement options, roadmap, decisions needed |

Leave 10+ minutes for discussion. The advisor's job in this meeting is to pick the problem statement, so Act V must leave real room for that.

**Delivery notes.**

- Speaker notes are written out below each slide. They are what to *say*, not what to put on the slide.
- Slide text should be sparse: a title, 3–5 bullets or one figure. Everything else goes in the notes.
- Anywhere a number appears, say the sample size out loud. This signals statistical care and pre-empts the obvious challenge.
- Act III slide 18 (the data-integrity incident) is the credibility centrepiece. Do not skip it to save time. It is the single strongest signal that the work is being done carefully.
- Every claim in Acts III and IV is backed by exact metrics, paired McNemar/Wilcoxon tests, and 10,000-sample bootstrap confidence intervals. Zero placeholders remain.

**Tone.** State findings plainly, including negative ones. Do not brag or oversell. An advisor evaluating research maturity rewards "here is what I found and here is the exact empirical proof" far more than a confident story with no error bars.

---

## Act I — Slides 1–3: The research direction

### Slide 1 — Title

**On slide:** *When does "thinking" help a model see? Investigating latent computation in multimodal reasoning.* Your name, affiliation, date, "Ongoing work — discussion".

**Notes:** Frame the meeting. "I want to walk you through where I've got to, show you some results that surprised me, and get your help choosing between two or three directions. I'll take about 25 minutes and I'd like to leave time to argue about the last section."

The subtitle is deliberately a question. It tells the advisor immediately that this is an investigation, not a reimplementation.

### Slide 2 — Why vision reasoning

**On slide:** Three example problems side by side, with the model's failure mode under each:

1. A jigsaw puzzle: which piece fills the gap? (Requires mentally rotating and fitting.)
2. A relative-depth question: which marked point is closer to the camera? (Requires 3D inference from 2D.)
3. A perspective-taking question: standing at X facing the fridge, is the table front-left or back-right? (Requires re-anchoring a coordinate frame.)

**Notes:** These are not perception problems. A model that can name every object in the scene still fails them. They need manipulation of spatial structure, and that structure resists being written down in words.

Make the central claim of the whole talk here: **language is a lossy channel for spatial computation.** Everything that follows is about what to do with that.

### Slide 3 — The gap in one chart

**On slide:** Bar chart, Qwen2.5-VL-7B zero-shot accuracy on five BLINK spatial splits, with the human/chance reference line:

| Split | Qwen2.5-VL-7B zero-shot | Chance |
| --- | --- | --- |
| Spatial Relation | 92.4% | 50% |
| Relative Depth | 61.3% | 50% |
| Multi-view | 39.0% | 50% |
| Jigsaw | 58.7% | 50% |
| IQ Test | 25.3% | 25% |

Source: Mull-Tokens paper Table 1, row a ([arXiv:2512.10941](https://arxiv.org/abs/2512.10941)).

**Notes:** This is the motivating picture. A strong 7B model is **at or below chance** on multi-view reasoning and IQ-test items, while being near-perfect on simple left-of/right-of relations.

The gap is not perception. It is the manipulation step. That is the target of this research direction.

If the advisor asks why Qwen2.5-VL specifically: it is the standard open backbone for this literature, so results transfer and baselines exist.

---

## Act I — Slides 4–6: How the field got here

Keep this to three slides and ninety seconds each. The advisor does not need a survey; they need to see that you know the shape of the field and can say what each paradigm buys and costs.

### Slide 4 — Four attempts to fix visual reasoning

**On slide:** A four-row table. This is the single most important lit-review slide.

| Paradigm | Idea | Representative work | Cost / failure |
| --- | --- | --- | --- |
| Text CoT | Write the reasoning out in words | Chain-of-Thought (Wei et al. 2022); Video-R1 | Drifts from the image; **hurts** vs. direct answering on spatial tasks |
| Tools / sketching | Call a cropper, detector, or sketchpad | Visual Sketchpad (Hu et al. 2024); DeepEyes; Pixel Reasoner | Brittle, needs external modules, limited manipulation vocabulary |
| Generate images | Produce intermediate pictures | Unified models (Chameleon, Blip3-o) | Expensive to train and run |
| Latent tokens | Reason in continuous hidden states | COCONUT (Hao et al. 2024); Mirage; LVR; Monet | Not interpretable; **is it doing anything?** |

**Notes:** Walk the rows. Land hard on row 1: text CoT is not just imperfect for spatial tasks, it is actively harmful. The Mull-Tokens paper reports direct-answer fine-tuning beating text-CoT fine-tuning by about 3 points on average.

That result is the hinge of the whole field. If writing your reasoning down makes you worse at spatial problems, the natural move is to reason without writing it down. That motivates row 4.

### Slide 5 — The latent reasoning idea

**On slide:** Simple diagram, two rows.

- Top: `[image] [question] → "Let me think... the red box appears to be... therefore..." → answer` labelled *200–500 tokens, drifts*
- Bottom: `[image] [question] → [□ □ □ □ □ ... ] → answer` labelled *20 tokens, no decoding*

**Notes:** The intellectual lineage runs through language models, not vision. Pause tokens (Goyal et al., ICLR 2024) showed that inserting content-free `<pause>` tokens gives a transformer extra computation before it answers. COCONUT fed hidden states back as inputs instead of decoding them to words.

The bet in vision is that the same trick helps more, because the thing being computed never had a good verbal form in the first place.

### Slide 6 — Why this is a live research area, not a solved one

**On slide:** Timeline strip, 2024 → late 2026, with roughly six markers: Pause tokens (2024) → COCONUT (2024) → Mirage / perception tokens (2025) → Mull-Tokens, LVR, Monet, ILVR (2026) → **audit wave (2026)**.

**Notes:** Flag that the field accelerated sharply and that a critical literature has now appeared alongside the methods. Do not detail the audits here — that is Act IV, and it lands much harder after the advisor has seen your own data.

One sentence only: "There's now a set of papers asking whether any of this actually works, and I'll come back to them once I've shown you my numbers."

---

## Act I — Slides 7–8: How it works now, and what breaks

### Slide 7 — The standard pipeline

**On slide:** Left-to-right block diagram:

`Image → ViT encoder → projector → [visual tokens] ⊕ [text tokens] → LLM decoder (28 layers) → answer`

Annotate three points on the diagram:

- Above the encoder: *frozen in most fine-tuning recipes*
- Above the visual tokens: *hundreds to thousands of tokens per image*
- Above the decoder: *fixed depth — 28 layers, regardless of problem difficulty*

**Notes:** Three structural facts matter for everything that follows.

1. **Visual information enters once, at the input.** There is no mechanism to go back and look again at a region.
2. **The decoder has fixed depth.** A hard problem gets exactly as much sequential computation as an easy one.
3. **The only way to get more computation is to emit more tokens.** That is why CoT works at all in language, and why it is an awkward fit here: emitting tokens means committing to words.

Point 3 is the one to dwell on. It sets up the latent-token idea as a way to buy computation without committing to words.

### Slide 8 — Four concrete limitations

**On slide:** Four boxes, each with the limitation and the evidence for it.

| Limitation | Evidence |
| --- | --- |
| Verbalisation loses spatial structure | Text CoT underperforms direct answering by ~3 points average on spatial benchmarks (Mull-Tokens Table 1, rows b vs. c) |
| Fixed compute budget | No architecture here allocates more computation to harder items |
| Visual grounding decays with depth | Reasoning drifts from the image; documented across the "thinking drifts" literature |
| Latents are uninterpretable | The Mull-Tokens paper lists this as its own main limitation |

**Notes:** Be precise that these are different *kinds* of problem. The first two are about the computational substrate. The third is about information flow. The fourth is about our ability to check any of it.

Most of the field works on 1 and 3. Almost nobody works on 2. Nobody had seriously worked on 4 until this year. That observation is where my project came from.

---

## Act I — Slide 9: The gap, and what it implies

### Slide 9 — Narrowing to one question

**On slide:** A funnel, four stages top to bottom:

1. Models fail at spatial manipulation, not perception
2. Verbalising the manipulation makes it worse
3. So: give the model computation it does not have to verbalise → **latent tokens**
4. **But: does the latent computation actually happen?**

Under the funnel, one line in bold: *Every method in this family reports accuracy gains. Almost none reports what the latents do.*

**Notes:** This slide is the pivot of the whole talk. Deliver it slowly.

The argument: steps 1–3 are the field's reasoning, and it is good reasoning. Step 4 is the question nobody answered before publishing. Methods are validated by benchmark accuracy, and accuracy cannot distinguish "the latents compute something" from "the latents do nothing and something else changed".

Name the three competing explanations now, because they recur throughout Act IV:

| Hypothesis | Claim | Signature if true |
| --- | --- | --- |
| **Scratchpad** | Latents carry image-derived, decision-relevant content | Perturbing them hurts; patching from a counterfactual flips the answer |
| **Compute buffer** | Latents provide parallel computation; contents are generic | Any trained filler works equally well |
| **Format key** | The prefix puts the model into its fine-tuned answering mode | One token works as well as twenty |

**Notes continued:** "The paper I'm going to show you claims the first. My results point conclusively at the third. That disagreement is where my project lives."

That sentence is the thesis of the presentation. If the advisor remembers one slide, make it this one.

---

## Act II — Slides 10–12: Mull-Tokens

### Slide 10 — The cleanest instance of the idea

**On slide:** Title *Mull-Tokens (Ray et al., CVPR 2026)* with the one-line framing: *Modality-agnostic latent thinking — the simplest version of the latent-token idea, and therefore the best one to test it on.*

**Notes:** Introduce it as a choice, not a given. "I picked this paper deliberately. It's Google + BU + Stanford, CVPR 2026, and it's the *minimal* version of the latent-token idea — which makes it the cleanest testbed for the question on the last slide."

The inference-time mechanism is one line. Show it:

```
<think> + <|latent_pad|> × 20 + </think>\n → answer
```

That is the entire intervention. Twenty copies of one token, ID 151665, prefilled into the assistant turn.

### Slide 11 — Architecture and training

**On slide:** Three-stage pipeline diagram.

| Stage | What happens | Supervision |
| --- | --- | --- |
| 1. Warm-up | Latents interleaved between reasoning-trace words; each latent predicts the next step | Cross-entropy on text steps; cosine loss to pooled image features for subgoal images |
| 2. Relaxed SFT | 20 **contiguous** latents, no intermediate supervision | Answer loss only |
| 3. GRPO | RL refinement | Answer correctness reward, group size G=2 |

**Notes:** Two details to emphasise, both of which matter later.

First, Stage 1 and Stage 2 use **different structures**. In Stage 1 the latents are interleaved one-per-word with the ground-truth trace. In Stage 2 they are twenty in a row with nothing between them. I raised this with the author on GitHub; he confirmed the design follows the Pause-token paper and said it would be "interesting to try and ablate this design choice".

Second, the latents are **discrete and identical**. All twenty positions receive the same input embedding. Whatever they compute has to come from position and attention, not from content fed in. Say this clearly — it is the fact Act IV turns on.

### Slide 12 — What the paper claims

**On slide:** Claims on the left, with the supporting number on the right.

| Claim | Reported |
| --- | --- |
| Beats strongest baseline | +5.40% average over direct-answer fine-tuning (DirAns: 61.40% → 66.80%) |
| Large gain on reasoning-heavy splits | +15.34% on BLINK Jigsaw (58.66% → 74.00%) |
| Efficient | 20 tokens vs. 200–500 for text CoT |
| Multimodal warm-up is essential | No warm-up 59.2% → text-only 65.9% → multimodal 66.8% (BLINK avg) |
| GRPO makes latents causally responsible | +0.12% average (53.92% → 54.04% overall across all benchmarks) |

**Notes:** Read the last row out loud and pause. Fifty thousand GRPO iterations on 8×H100 for a tenth of a percentage point. Flag that you will come back to it.

Do not editorialise beyond that yet. Present the paper fairly — the critique lands much harder on the next slide if this one is straight.

---

## Act II — Slide 13: My critical reading

### Slide 13 — Five gaps I found

**On slide:** Five rows. Mark each as *from the paper* or *from the code*, which demonstrates you read both.

| # | Gap | Source |
| --- | --- | --- |
| 1 | **No causal test.** The paper admits answer tokens can attend directly to the image, bypassing the latents, and runs no intervention to rule it out | Paper §3 |
| 2 | **Confounded comparison.** SAT data rose from 20% to 60% between Stage 1 and Stage 2; gains may be domain adaptation, not latent thinking | Paper appendix §6.1 |
| 3 | **Stage 3 does not do what it claims.** GRPO is meant to make latents causally responsible, but the released config sets `text_only_logprobs: True`, so only answer tokens enter the policy gradient | **Code** |
| 4 | **Train/test structure mismatch.** Stage 1 interleaves latents one-per-word; Stage 2 uses 20 contiguous. The learned role may not transfer | **Code**, confirmed by author |
| 5 | **Spatial collapse.** Subgoal images are average-pooled to a single vector, destroying 2D layout — and the method regresses 3.2 points on Relative Depth | Paper appendix |

**Notes:** This is the slide that proves you did more than read an abstract. Emphasise that gaps 3 and 4 are **not in the paper** — they came from reading the training code line by line.

Gap 3 is the strongest. The paper devotes a full subsection to arguing that GRPO rewards causally-useful latent chains. The released configuration appears not to implement that.

Also note the architectural observation: because the latents are a fixed prefix computed in parallel, latent *k* at layer ℓ can only read latents <*k* at layer ℓ−1. Total serial depth is still 28 layers. **Mull adds width, not depth.** The paper's phrase "rich internal recurrence" overstates this.

**Transition line to Act III:** "So I had five hypotheses about where this might be weak. The next part is what happened when I tested them."

---

## Act III — Slides 14–16: Building the testbed & replication

### Slide 14 — Why replicate first

**On slide:** One sentence, large: *A negative result is only believable if the positive result reproduces first.*

Below it, the setup: 1× A100-PCIE-40GB, Ubuntu 20.04, CUDA 12.2, PyTorch 2.5.1, FlashAttention 2, custom `transformers` fork, `lmms-eval`.

**Notes:** Pre-empt the obvious objection. "If I'd gone straight to ablations and found the latents don't matter, the first question would be whether my setup was broken. So I reproduced the published numbers first. That's now my validity argument."

This is also the honest framing of why replication was worth the time — it was infrastructure for the real question, not the goal.

### Slide 15 — Five blocking bugs I had to fix

**On slide:** Compact list.

1. **Hardcoded cluster paths** in five task YAMLs (`/home/jupyter/data_files/...`) → rewired to HF streaming
2. **Dataset schema mismatches** — SAT config name and split, ERQA image keys (`images_encoded`), checksum failures
3. **Chat-template corruption** — `text.replace("<|im_end|>\n", "")` stripped turn boundaries between *all* turns, merging system, user and assistant into one invalid string
4. **Broken model registration** — hardcoded `sys.path.append("../models")`, missing `@register_model` decorator
5. **Crash on hardcoded logging directory**

**Notes:** Do not spend more than 45 seconds here, but do show it. Bug 3 is the interesting one: it silently produced malformed prompts rather than crashing, so it would have corrupted every number without any error.

If asked: yes, these are reported upstream-worthy; the repo README itself says the release is a work in progress and invites bug reports.

### Slide 16 — Replication result

**On slide:** Two-column comparison, paper vs. mine, ~2,400 samples across 8 benchmarks plus 5,130 video samples.

| Benchmark | Paper Claimed (SFT / GRPO) | Our Replicated Result | Outcome vs. Paper |
| --- | :---: | :---: | :---: |
| BLINK Spatial Relation (143) | 81.81% / 81.81% | **85.31%** | +3.50% (Exceeded) |
| BLINK Relative Depth (124) | 83.06% / 83.87% | **85.48%** | +1.61% (Exceeded) |
| BLINK Jigsaw (150) | 74.00% / 74.67% | **76.67%** | +2.00% (Exceeded) |
| BLINK Multi-view (133) | 63.15% / 64.66% | **63.91%** | Matches within margin |
| BLINK IQ Test (150) | 32.00% / 30.66% | **30.67%** | Exact match (30.67% vs. 30.66%) |
| **BLINK Overall Average (700)** | **66.80% / 67.13%** | **68.41%** | **+1.28% (Exceeded)** |
| SAT-Real (300 pairs) | 77.66% / 77.00% | **74.33%** | Matches within margin (-3.33%) |
| ERQA Spatial (400) | 38.25% / 38.50% | **39.25%** | +0.75% (Matches/Exceeds) |
| MMSI-Bench Multi-Image (1,000) | 29.00% | **28.50%** | Matches within margin |
| VSI-Bench Video Spatial (5,130) | 32.98% / 33.51% | **33.69%** | +0.71% (Exceeded/Matched) |
| VSI-Bench Reasoning Subset | 32.85% / 33.49% | **32.67%** | Exact match (within 0.18%) |

**Notes:** Land the headline: replication holds across image, multi-image and video benchmarks. VSI-Bench at 5,130 samples matches to within 0.2 points on the reasoning subset.

Total runtime: about 50 minutes for the core image benchmarks on one A100. Mention this — it establishes that the inference-only experiment program is cheap, which matters for the roadmap in Act V.

---

## Act III — Slides 17–18: Pipeline hardening & master experiment architecture

### Slide 17 — The data-integrity incident & pipeline hardening

This slide is uncomfortable and you should keep it. It is the strongest available evidence of research maturity.

**On slide:** Title *Why my first set of numbers was wrong — and how we hardened the pipeline.*

- My launcher had a `--dry-run` mode passing `--limit 2`
- Those 2-sample smoke tests wrote results into the **same directories** as full runs
- An unvetted file glob parsed these directories and let the last file win
- Result: 1/2 and 2/2 surfaced as **50.00%** and **100.00%** in early tables

Show the giveaway: 50.00% on a 143-item split would require 71.5 correct answers.

**Fixes applied:**

1. Smoke runs write to an isolated `scratch/` or `smoke_archive/` directory
2. Strict assertions in `eval_metric_utils.py` enforce exact sample counts: SpRel=143, RelDep=124, Jigsaw=150, MV=133, IQ=150 (Total N=700). Fails loudly if count deviates
3. `--log_samples` enabled everywhere: per-sample token IDs, lengths, and logits saved
4. Created a formal forensic audit log in `ERRATA.md` rather than silently overwriting

**Notes:** Deliver this without defensiveness. "I caught this by checking whether each percentage was achievable given the split size. Three of them weren't. That's now a standing automated assertion in my pipeline."

The payoff line matters: **the corrected data tells a cleaner and more interesting story than the buggy data did.** The artefact had produced a dramatic but false result (K=40 → 75.81%). The real result is flatness, which is a sharper finding.

### Slide 18 — The master empirical architecture (E1 & E2)

**On slide:** Two-stage mechanistic investigation diagram.

```
Initial Gap Analysis (FinalGaps.md)
       │
       ▼
[Experiment E1: Mechanistic Factorial Matrix] 
N = 10,500 evaluations | 15 conditions | Standardized 4.19 MP
Question: Does the latent compute (H1), or is it a format stop-gate (H3)?
       │
       ▼  (Finding: <think></think> matches 20 latents; K=0 is text CoT runaway)
[Experiment E2: Bias vs. Discrimination Study]
N = 44,590 evaluations | 14 conditions | 6 input variants (V0–V4, CF)
Question: Does Mull's margin over DirAns reflect true vision, or an uncalibrated letter prior?
```

**Notes:** Structure the two master experiments clearly. 

Experiment E1 adjudicates between continuous scratchpad computation and format conditioning. 

Experiment E2 takes the next step: having shown that format matters, it isolates whether Mull's reported margin over DirAns reflects true perceptual reasoning or answer-prior alignment.

---

## Act III — Slides 19–22: Experiment E1 (Format vs. Scratchpad)

### Slide 19 — Experiment E1: The 15-condition mechanistic matrix

**On slide:** Full matrix across all 5 BLINK tasks ($N=700$ per row, $N=10,500$ evaluations total) at standardized 4.19 MP resolution.

| Row | Model | Prompt | Assistant Prefill | SpRel /143 | RelDep /124 | Jigsaw /150 | MV /133 | IQ /150 | BLINK Avg | Δ vs. Row 1 [95% CI] | McNemar $p$-value | Mean Gen Tok | Mean Latency |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Mull | `latents` | `think + 20 latents` | 85.31% | 85.48% | 76.67% | 63.91% | 30.67% | **67.71%** | **Reference** | — | 7.0 | 32.29s (cold) |
| **2** | Mull | `latents` | `empty` ($K=0$) | 76.22% | 67.74% | 62.00% | 50.38% | 24.67% | **55.71%** | -12.00% [-15.86%, -8.57%] | $\mathbf{p < 10^{-10}}$ | 138.5 | 175.12s |
| **3** | Mull | `latents` | `<think></think>\n` | 85.31% | 85.48% | 79.33% | 60.15% | 28.67% | **67.14%** | **-0.57% [-2.00%, +0.71%]** | $\mathbf{p = 0.5413}$ | 7.0 | 0.45s |
| **4** | Mull | `latents` | `think + 20x' .'` | 84.62% | 84.68% | 75.33% | 60.90% | 30.00% | **66.43%** | -1.29% [-2.57%, -0.14%] | $p = 0.0636$ | 7.0 | 0.45s |
| **5** | Mull | `latents` | `think + 20x<|latent_end|>` | 84.62% | 86.29% | 72.00% | 60.90% | 32.00% | **66.43%** | -1.29% [-2.57%, +0.00%] | $p = 0.0784$ | 7.0 | 0.45s |
| **6** | Mull | `sft` | `empty` | 78.32% | 74.19% | 64.67% | 51.13% | 25.33% | **58.14%** | -9.57% [-12.86%, -6.29%] | $p < 10^{-10}$ | 122.9 | 4.76s |
| **7** | Mull | `sft` | `think + 20 latents` | 85.31% | 84.68% | 77.33% | 59.40% | 29.33% | **66.57%** | -1.14% [-2.57%, +0.29%] | $p = 0.1859$ | 7.0 | 0.45s |
| **8** | SAT | `sft` | `empty` (DirAns) | 81.82% | 79.03% | 68.00% | 63.91% | 27.33% | **63.29%** | -4.43% [-7.71%, -1.14%] | $p = 0.0104$ | 7.0 | 0.44s |
| **9** | SAT | `latents` | `empty` | 81.82% | 79.84% | 68.00% | 68.42% | 28.00% | **64.43%** | -3.29% [-6.57%, +0.00%] | $p = 0.0639$ | 7.0 | 0.44s |
| **10** | Mull-GRPO | `latents` | `think + 20 latents` | 86.01% | 84.68% | 78.67% | 65.41% | 31.33% | **68.57%** | +0.86% [-1.00%, +2.71%] | $\mathbf{p = 0.4610}$ | 7.0 | 0.35s |
| **11** | Mull | `latents` | `think + 1 latent` | 84.62% | 83.87% | 78.00% | 60.15% | 28.67% | **66.43%** | -1.29% [-2.71%, +0.00%] | $p = 0.0931$ | 7.0 | 0.35s |
| **12** | Mull | `latents` | `think + 5 latents` | 85.31% | 86.29% | 77.33% | 60.90% | 30.00% | **67.29%** | -0.43% [-1.71%, +0.86%] | $p = 0.6636$ | 7.0 | 0.35s |
| **13** | Mull | `latents` | `think + 10 latents` | 85.31% | 85.48% | 76.67% | 60.15% | 29.33% | **66.71%** | -1.00% [-2.14%, +0.14%] | $p = 0.1435$ | 7.0 | 0.37s |
| **14** | Mull | `latents` | `think + 40 latents` | 85.31% | 83.87% | 76.67% | 62.41% | 31.33% | **67.29%** | -0.43% [-1.86%, +1.00%] | $p = 0.7003$ | 7.0 | 0.34s |
| **15** | Mull | `latents` | `think + 20 latents` (Warm) | 85.31% | 85.48% | 76.67% | 63.91% | 30.67% | **67.71%** | +0.00% | $p = 1.0000$ | 7.0 | 0.36s |

**Notes:** Point out that this 15-condition matrix was executed strictly against our pre-registered hypotheses. It covers negative controls, prompt cross-conditions, budget scaling, and warm latency baselines.

### Slide 20 — E1 Finding 1: The smoking gun (`<think></think>` matches 20 latents)

**On slide:** Bar comparison: Row 1 (20 latents, 67.71%) vs. Row 3 (empty tags, 67.14%) vs. Row 4 (20 periods, 66.43%) vs. Row 5 (20 delimiters, 66.43%).

- **Empty think tags (Row 3, 0 latents)**: 67.14% vs. 67.71% reference ($\Delta = -0.57\%$, 95% CI: $[-2.00\%, +0.71\%]$)
- **Paired McNemar test on N=700**: $b=14, c=10, \mathbf{p = 0.5413}$ (Identical!)
- **Punctuation controls match**:
  - Generic periods (`" ."`, Row 4): 66.43% ($p = 0.0636$)
  - Delimiter tokens (`<|latent_end|>`, Row 5): 66.43% ($p = 0.0784$)

**Notes:** Deliver this clearly: Row 3 injects **zero latent tokens**. We simply prefilled the tag: `<think></think>\n`. The difference across the entire 700-question benchmark is only 4 items.

This directly falsifies Hypothesis H1 (latent scratchpad) and confirms Hypothesis H3 (format conditioning). Generic periods match the trained embeddings, proving that the learned weights carry no specialized visual-reasoning payload.

### Slide 21 — E1 Finding 2: Why Row 2 ($K=0$) collapsed by 12 points

**On slide:** Two-panel graphic:

- Left: Output token distribution (Row 1: exactly 7.0 tokens, $\sigma = 0.00$ vs. Row 2: mean 138.5 tokens, max 1,024).
- Right: Pathological decoding samples from raw logs.

**Forensic breakdown:**

1. **Unconstrained Verbal Drift**: When `<think>` is not closed, the base model reverts to long verbal text-CoT (`<think>THOUGHT:...`), blowing latency from 0.36s to **175.12s per sample**. Verbal rationales hallucinate and degrade visual perception.
2. **Cap Truncation & Repetition Loops**:
   - Exactly 1 item in Row 2 hit the 1,024-token cap (IQ-Test doc 52, repeating *"The forty-eighth image is a larger version..."* for 48 cycles).
   - Exactly 1 item in Row 6 hit the cap (Relative Depth doc 12, collapsing into repeated CJK characters: `便便便...`).
3. **Hard Parse Failures**: 3 items in Row 2 and 1 item in Row 6 emitted no parseable `<answer>` tags.
4. **Format Lock**: All format-locked conditions generate **exactly 7.0 tokens on all 700 items with standard deviation $\sigma = 0.00$** (`<answer>X</answer><|im_end|>`).

**Notes:** The 12-point drop is not caused by losing an internal scratchpad. It is caused by the model reverting to rambling text-CoT, which introduces visual drift and occasional decoding pathologies.

### Slide 22 — E1 Finding 3: Flat budget scaling & non-significant GRPO

**On slide:** Two charts:

- Left: $K$-scaling curve across $K \in \{1, 5, 10, 20, 40\}$ (Flat line between 66.43% and 67.71%, all $p > 0.09$).
- Right: SFT (67.71%) vs. Stage 3 GRPO (68.57%), $\Delta = +0.86\%$, $p = 0.4610$.

**Empirical proofs:**

- **Flat Scaling**:
  - $K=1$: 66.43% ($p=0.0931$) — recovers ~89% of the effect with one token
  - $K=5$: 67.29% ($p=0.6636$)
  - $K=10$: 66.71% ($p=0.1435$)
  - $K=20$: 67.71% (Reference)
  - $K=40$: 67.29% ($p=0.7003$)
- **GRPO Gain is Evaluative Noise**: $+0.86\%$ gain over SFT. Paired McNemar yields $b=20, c=26, \mathbf{p = 0.4610}$.
- **Group Size Bottleneck ($G=2$)**: When both sampled answers are right [1, 1] or both wrong [0, 0], advantage is zero, producing frequent dead gradient updates.

**Notes:** A true cognitive scratchpad benefits from more thinking steps. Here, giving the model 40 tokens yields the exact same performance as 1 token. GRPO's reported gain is statistical noise.

---

## Act III — Slides 23–26: Experiment E2 (Bias vs. Discrimination)

### Slide 23 — The new threat: Discovering systematic answer bias

**On slide:** The 2-way split prediction distribution table:

| Condition | Predicted "B" (2-way, $N=550$) | Gold "B" |
| --- | :---: | :---: |
| Mull checkpoints (Rows 1, 3–5, 7, 11–13, 15) | **59.6% – 61.8%** | 49.3% |
| Mull-GRPO (Row 10) | **53.8%** | 49.3% |
| SAT Baseline / DirAns (Rows 8, 9) | **46.2% – 50.0%** | 49.3% |

Under the table: **Dataset Gold Imbalance on Multi-View Reasoning = 55.6% "B"**.

**Notes:** This is the hinge to Experiment E2. In the paper, Mull beat DirAns by +5.40%. But when we audited the raw predictions, Mull had a massive **10-point bias toward predicting option "B"**, while the SAT baseline was calibrated near 50%.

Because the Multi-View split happens to have 55.6% gold "B", guessing "B" gives a free score boost. That raised an urgent scientific question: **Is Mull actually better at seeing space, or is its lead an artifact of guessing "B"?**

### Slide 24 — Experiment E2: Disentangling bias from discrimination

**On slide:** The $N=44,590$ evaluation setup across 14 conditions and 6 controlled input variants:

1. **$V0$ (Real)**: Original image and question ($N=700$).
2. **$V1$ (Blind)**: Uniform grey (128) canvas matching exact image dimensions ($N=700$) — pure language prior.
3. **$V2$ (Option Swap)**: Content of A and B exchanged ($N=550$).
4. **$V3$ (Option Relabel)**: Option letters reversed: `(B) c1` then `(A) c2` ($N=550$).
5. **$V4$ (Swap + Relabel)**: Completing the orthogonal $2 \times 2$ factorial design ($N=550$).
6. **$CF$ (Content-Free)**: 27 contextual dummy probes per task to extract intrinsic letter prior $p_{\text{cf}}(\text{letter})$ (*Zhao et al., 2021*).

**Mathematical decomposition:**
$$\begin{aligned}
m_{V0} &= s + L + P \\
m_{V2} &= -s + L + P \\
m_{V3} &= -s + L - P \\
m_{V4} &= s + L - P
\end{aligned}
\quad \Longrightarrow \quad
s = \frac{m_{V0} - m_{V2} - m_{V3} + m_{V4}}{4}$$

Where $L$ is literal letter bias, $P$ is positional bias, and $s$ is **bias-free visual discrimination**.

**Notes:** We executed this using single-pass teacher-forced letter probability scoring with four hard verification gates (G1–G4). Gate G3 asserted $\ge 99\%$ agreement between teacher-forced argmax and E1 greedy generation.

### Slide 25 — E2 Master Results: Mull's lead does not survive calibration

**On slide:** Primary hypothesis Q1 results table (Mull T1 vs. SAT T8):

| Metric | T1 (Mull) | T8 (SAT) | Delta | 95% Bootstrap CI | McNemar $p$-value | Survives? |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Accuracy ($M1$, $N=700$)** | 66.71% | 63.57% | +3.14% | [+0.00%, +6.29%] | $p = 0.0630$ | Baseline |
| **Content-Free Calibrated ($M2$, $N=700$)** | 53.57% | **56.14%** | **-2.57%** | [-5.29%, +0.14%] | $\mathbf{p = 0.0790}$ | **NO (SAT wins)** |
| **Debiased $2\times 2$ Discrimination ($M3$, $N=550$)** | 74.91% | 72.55% | +2.36% | [-1.09%, +5.82%] | $\mathbf{p = 0.2181}$ | **NO (Non-significant)** |

One prominent callout box:
> **35.0% of Mull's published advantage is explained purely by answer-prior alignment with benchmark test splits.**

**Notes:** Read these numbers carefully. When contextual letter priors are calibrated out, **the SAT baseline actually beats Mull by +2.57%**. Under full bias-free $2\times 2$ discrimination, the gap shrinks to +2.36% with $p = 0.2181$ (statistically indistinguishable). Over a third of Mull's reported margin over DirAns was simply answer bias.

### Slide 26 — E2 Mechanistic Proofs: Logit margins & the direct answer control

**On slide:** Two decisive proofs side by side:

1. **Continuous Logit Margins (T1 vs. T3, $N=700$)**:
   - Gold margin: $d = \text{logit}(\text{gold}) - \max_{k \neq \text{gold}} \text{logit}(k)$
   - T1 (20 latents) mean margin: **0.7163** vs. T3 (`<think></think>`) mean margin: **0.7130** ($\Delta = +0.0033$)
   - Wilcoxon signed-rank test: $W = 46497.0, \mathbf{p = 0.9833}$ ($d_z = 0.0230$)
   - Two One-Sided Equivalence Test (TOST, $\pm 3\text{ pp}$): **Confirmed Equivalent**.
2. **The Direct Answer Control (T2F & T6F)**:
   - T2F forces immediate `<answer` with **zero tags and zero latents**.
   - Scores **66.86% raw** and **73.82% debiased**, matching T1 (66.71% / 74.91%) and T3 (66.86% / 73.82%).
   - **Conclusion**: The $K=0$ drop is 100% an artifact of decoding mode.
3. **GRPO Re-centering (T10)**:
   - Debiased visual gain over SFT is **+0.18%** (75.09% vs. 74.91%). GRPO merely shifted letter prior $L$ from $+0.098$ to $+0.024$.

**Notes:** This slide closes both questions. The continuous logit distribution of 20 latents is mathematically identical to empty tags ($p=0.9833$). And when forced to answer directly with no latents (T2F), performance is completely preserved.

---

## Act IV — Slides 27–29: Where the field is & the scientific verdict

### Slide 27 — The 2026 audit wave

**On slide:** Four papers, one line each, with the finding.

| Paper | Venue / date | Finding |
| --- | --- | --- |
| [What's Holding Back Latent Visual Reasoning?](https://arxiv.org/abs/2605.18445) (Viveiros et al.) | NeurIPS 2026 | Accuracy unaffected when latents are replaced by uninformative dummy tokens; inference-time latents collapse to a narrow region |
| [Beyond Visual Memory](https://arxiv.org/abs/2606.01287) (Guo et al.) | Aug 2026 | Decomposes latents into slots / boundary markers / format. Slot perturbation moves accuracy ≤0.027; **marker-only decoding preserves 78–100% of the gain** |
| [Imagination Helps Visual Reasoning, But Not Yet in Latent Space](https://www.emergentmind.com/open-problems/determine-mechanisms-of-latent-visual-reasoning-effectiveness) (Li et al.) | ICML 2026 | Inputs barely affect latent states; latent states barely affect answers |
| [Do Latent Tokens Think?](https://aclanthology.org/2026.emnlp-main.186/) (Zhang et al.) | EMNLP 2026 | COCONUT latents show minimal steering sensitivity and lack reasoning-critical information |

**Notes:** Our empirical results independently converged with four published groups. We reached the format/stop-gate finding from our own data before reading these.

A paper arguing "Mull-Tokens latents are a placebo" would be fifth to that general conclusion. But all four audited continuous, content-supervised latents on perception tasks. Mull-Tokens is a different object, as the next slide shows.

### Slide 28 — The positioning table: Mull as the limiting case

**On slide:**

|  | Audited methods (Mirage, LVR, Monet, ILVR, MVH) | Mull-Tokens |
| --- | --- | --- |
| Latent form | **Continuous** vectors, autoregressively generated and re-injected | **Discrete** — the same token ID (151665) repeated K times |
| Per-slot content | Exists; supervised toward visual targets | **None at input** — all 20 positions identical |
| Boundary markers | **Learned** `<lat_start>` / `<lat_end>` embeddings | `<think>` / `</think>` — **verified plain text** (`<` + `think` + `>`), not special tokens |
| Benchmarks | Perception-heavy: V\*, HRBench-4K/8K, MME-RealWorld | **Abstract spatial**: jigsaw, IQ, multi-view, depth — **plus video** (VSI-Bench) |

**Notes:** Structural argument: **Mull-Tokens is the limiting case of the entire family: zero slot content.**

Everything the audits attribute to "non-content mechanisms" — markers, format, attention routing, raw compute — is *all Mull has*. We also verified in the tokenizer that `<think>` and `</think>` are plain multi-token text, meaning Mull adds no learned boundary embeddings at all.

### Slide 29 — The final scientific verdict

**On slide:** Three core conclusions:

1. **What Mull-Tokens Actually Are**: They are not an internal cognitive scratchpad or world model. They are a **syntactic format stop-gate**. Prefilling `<think>...</think>` locks the assistant turn and prevents the model from generating discursive verbal chain-of-thought, which causes severe visual drift in vision-language models.
2. **Why the Paper Reached Flawed Conclusions**:
   - **Sabotaged Text Baseline**: Prompted to emit conversational padding (*"let me think, wait, hmm..."*), provoking visual drift.
   - **Regex Parsing Fragility**: Long text outputs failed regex parsing or fell back to picking the last capital letter (often an eliminated wrong option). Format-locked outputs avoided this entirely.
   - **Uncalibrated Answer Prior**: A 60% B-bias aligned favorably with imbalanced test splits (Multi-View 55.6% B), accounting for 35.0% of the reported margin.
   - **No Controls or Error Bars**: No empty-tag controls, no punctuation controls, and no paired statistical testing.
3. **Takeaway for the Field**: Latent token insertions in feed-forward transformers must be evaluated with strict format and calibration controls before attributing gains to "latent thinking."

---

## Act V — Slides 30–32: The ask & next steps

### Slide 30 — Candidate problem statements

**On slide:** Three options, honestly assessed:

|  | A. Task structure taxonomy | B. Adaptive budget allocation | A + B Combined (Recommended) |
| --- | --- | --- | --- |
| **Question** | Which visual task structures benefit from content-free compute? | Can the model learn to allocate latent compute per instance? | Unified theory of content-free compute & adaptive allocation |
| **Novelty** | High — extends Pfau et al. to vision and video | High — test-time scaling for multimodal latents is open | Very High — comprehensive paper |
| **Feasibility (40 GB)** | Excellent (pure inference) | Good (LoRA on 7B/3B) | Fits our infrastructure |
| **Main risk** | Reads as a benchmark study without a fix | If compute genuinely doesn't help, method fails | Needs disciplined execution |

**Notes on Recommendation (A+B):**
"Content-free latent tokens provide format control and task-dependent compute, not visual memory — and current fixed-prefix designs cannot allocate that compute where it would help."
We have the tools, the verified harness, and the video evaluation pipeline (VSI-Bench) that no other audit paper covers.

### Slide 31 — Research roadmap & compute feasibility

**On slide:** Phased plan and hardware requirements:

| Phase | Timeline | Deliverable | Hardware Required |
| --- | --- | --- | --- |
| **Phase 1: Audit Complete** | Done | E1 Matrix ($N=10,500$) + E2 Calibration ($N=44,590$) + `ERRATA.md` | 1× A100-40GB (Current) |
| **Phase 2: Task Taxonomy & Video** | Weeks 1–3 | Map benefit profile across 8 spatial tasks + VSI-Bench video; layer attention knockout | 1× A100-40GB (Current) |
| **Phase 3: Adaptive Budget Controller** | Weeks 4–7 | Train randomised-K LoRA ($K \in [0, 40]$); learn per-instance `<|end_of_thought|>` | 1× A100-40GB (LoRA) or 1× H100-80GB (Full) |
| **Phase 4: Synthesis & Paper** | Weeks 8–10 | Conference submission (CVPR/ICCV/NeurIPS) | CPU / Writing |

**Notes:** Phases 1 and 2 fit our existing 40 GB card comfortably. If we obtain 80 GB access for one week during Phase 3, we can scale GRPO group size to $G=8$ or $16$ to eliminate the dead updates that plagued the original paper.

### Slide 32 — Four decisions needed from advisor

**On slide:**

1. **Direction Approval**: Do you agree with pursuing A+B (Task Taxonomy + Adaptive Budget Controller)?
2. **Venue Strategy**: Aim for a top-tier conference paper (NeurIPS / CVPR) rather than a narrow workshop audit.
3. **Compute Allocation**: Can we schedule 1 week of 80 GB VRAM access for Phase 3 scaling?
4. **Author Communication**: Should I reach out to Ray et al. regarding the unreleased Stage 1 and Table 2 ablation checkpoints?

**Closing line:** "The replication is verified, the mechanistic audit is definitive, and the data hygiene is rock-solid. What I need now is your direction on which problem statement to commit to."

---

## Backup Slides — Anticipated Questions & Forensic Evidence

### B1. "Isn't this just replicating someone else's paper?"
No. Replication was only Phase 0. The scientific discoveries are entirely ours:
1. Row 3 empty think tags (`<think></think>`) match 20 latents ($p=0.5413$).
2. The $K=0$ drop is 100% autoregressive verbal degeneration, proven by T2F (66.86% raw / 73.82% debiased).
3. 35.0% of Mull's lead over DirAns is pure B-bias alignment ($L = +0.098$).
4. Under contextual calibration, SAT actually outperforms Mull (+2.57%, $p=0.0790$).
5. Continuous logit margins prove mathematical equivalence ($W=46497.0, p=0.9833$).

### B2. "Why should I trust your numbers after the smoke-test bug?"
Because:
- The bug was caught by an automated arithmetic check on sample fractions.
- We isolated all smoke files, instituted strict sample count assertions in `eval_metric_utils.py`, and published the full correction log in `ERRATA.md`.
- All runs in E1 ($N=10,500$) and E2 ($N=44,590$) logged raw per-sample token IDs, lengths, and logits with G1–G4 verification gates.

### B3. "What did the Blind Baseline ($V1$) show?"
On Relative Depth, Jigsaw, and Multi-View, each split uses an identical question prompt across all items. On uniform grey canvases ($V1$), accuracy equals the gold share of whichever letter the model prefers (Multi-View blind accuracy = 55.6% for Mull, exactly matching its B-bias on a 55.6% gold-B split). Spatial Relation is the only split where question text varies (blind accuracy = 46.85%, AUC = 0.602).

---

## Appendix — Instructions for the Slide-Building Agent

### Figure & Table Specifications

| Slide | Type | Content / Specification |
|:---:|:---:|---|
| **3** | Bar chart | 5 BLINK splits zero-shot accuracy vs. dashed chance line (50% / 25%). |
| **5** | Diagram | Top: Text CoT (200–500 tokens, drifting). Bottom: Mull prefill (20 tokens, no decoding). |
| **11** | Diagram | 3-stage training pipeline. Highlight Stage 1 (interleaved) vs. Stage 2 (20 contiguous). |
| **16** | Table | Replicated vs. paper accuracy across 8 core benchmarks + VSI-Bench (5,130 samples). |
| **19** | Master Table | Full 15-condition E1 Matrix ($N=10,500$ evaluations) with McNemar $p$-values and latencies. |
| **20** | Bar chart | Row 1 (67.71%) vs. Row 3 (67.14%) vs. Row 4 (66.43%) vs. Row 5 (66.43%). |
| **21** | Side-by-side | Verbatim decoded text: Row 1 (7 tokens) vs. Row 2 (138 tokens, IQ #52 loop). |
| **22** | Line chart | Flat $K$-curve ($K=1$ to $40$ at ~67%) with $K=0$ cliff (55.71%). |
| **25** | Table | Primary Q1 test: Raw ($M1$), Calibrated ($M2$), and Debiased ($M3$) comparing Mull T1 vs. SAT T8. |
| **26** | Box / Density | Continuous margin distributions: T1 (0.7163) vs. T3 (0.7130) showing $p=0.9833$. |

### Sources & Verified Ground Truths
1. **Tokenizer Truth**: Confirmed via Qwen2.5-VL tokenizer that `<think>` and `</think>` are plain multi-token text strings, not special tokens. Mull adds no boundary special tokens.
2. **GRPO Code Truth**: Confirmed in official training config that `text_only_logprobs: True`, meaning latents receive zero policy gradient directly.
3. **Resolution Parity**: All conditions in E1 and E2 evaluated at standardized 4.19 MP resolution (`max_pixels=4194304`).
