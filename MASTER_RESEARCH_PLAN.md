# Master Multi-Model Forensic Research Plan & Autonomous Agent Guide

> **Target Audience**: Autonomous AI Coding Agents (Antigravity, Devin, Claude Code, Cursor, SWE-bench) and Human Co-Investigators.  
> **Repository**: [https://github.com/Vnshiee/MTok](https://github.com/Vnshiee/MTok)  
> **Mission**: Conduct publication-grade, line-by-line forensic audits, code-vs-paper discrepancy analyses, and mechanistic falsification experiments across the top 2026 Multimodal Latent Reasoning architectures (`Qwen2.5-VL-7B` backbone).

---

## 1. Executive Overview & Research Thesis

In 2025–2026, leading AI research labs claimed that Multimodal Large Language Models (MLLMs) can achieve human-like "visual thinking" by predicting continuous latent vectors (e.g., Mull-Tokens, LVR, Monet, Mirage) instead of verbose text Chain-of-Thought (CoT).

**Our Central Empirical Finding**:  
Across all 4 flagship frameworks, **the claimed "continuous visual reasoning" is an architectural illusion**:
1. **Mull-Tokens** (*CVPR 2026*): Does not perform visual thinking; it acts as a **syntactic format lock** (`<think></think>\n` stop-gate) that suppresses degenerative verbal CoT and exploits a +10% multiple-choice option `B` calibration shift.
2. **LVR** (*ICLR 2026*): Suffers from the **ROI Feature Copying Paradox**—supervising latent hidden states via MSE against raw ViT patch tokens simply trains an auto-associative identity copy across self-attention of visual tokens already sitting in the prompt prefix. Furthermore, RL policy gradients into latent reasoning are literally zero (`final_mask = 0`, `@torch.no_grad()`).
3. **Monet** (*CVPR 2026*): Claims continuous policy gradients (VLPO), but actually avoids backpropagating through time by computing first-order gradients, detaching them (`safe_grads.append(g.detach())`), and updating latents via an offline static proxy loss.
4. **Mirage** (*CVPR 2026*): Suffers from **Catastrophic Representational Drift** because Stage 2 completely drops visual loss (`loss = ce_loss`), unmooring the "mental imagery" tokens into generic prompt embeddings.

---

## 2. Directory Structure & Complete Forensic Inventory

When cloning this repository, any autonomous agent or researcher should inspect these exact directories:

```
MTok/
├── AGENT_GUIDE.md                               # Operational guide for running Mull-Tokens code & benchmarks
├── FinalGaps.md                                 # Line-by-line paper critique & gap analysis for Mull-Tokens
├── PREREGISTRATION.md                           # E1 Matrix Pre-registration (Format lock vs Compute)
├── PREREGISTRATION_E2.md                        # E2 Matrix Pre-registration (B-Bias & Teacher-Forced Scoring)
├── MASTER_RESEARCH_PLAN.md                      # [THIS FILE] Unified multi-paper research plan & agent instructions
│
├── external_models/                             # Forensic audits of comparative 2026 frameworks
│   ├── PIPELINE_STATE.md                        # Master state machine tracking certification status (Pass/Fail)
│   ├── UNIFIED_THEORY_OF_LATENT_TOKENS.md       # Master comparative synthesis paper across all 4 architectures
│   │
│   ├── lvr/                                     # Paper 1: LVR (Latent Visual Reasoning, arXiv:2509.24251)
│   │   ├── STEP1_INGESTION_MANIFEST.md          # 100% verified line-by-line inventory of 25 LaTeX files & 50 repo files
│   │   ├── CODE_VS_PAPER_TRACE.md               # Mathematical & tensor execution trace (Forward, M-RoPE, GRPO, Collator)
│   │   ├── Gaps_LVR.md                          # Comprehensive gap analysis (ROI copying paradox, step collapse)
│   │   ├── PREREGISTRATION_LVR.md               # Pre-registered hypotheses & mechanistic test protocols
│   │   ├── SYNTHESIS_LVR.md                     # Executive summary of findings for LVR
│   │   ├── diagnostics/                         # CPU-executable verification scripts (test_roi_copying, test_grpo_zero_grad)
│   │   ├── paper_src/                           # Full extracted LaTeX source & figures from arXiv
│   │   └── hf_metadata/LVR-7B/                  # Config, tokenizers, args, index from HuggingFace
│   │
│   ├── monet/                                   # Paper 2: Monet (Reasoning in Latent Visual Space, arXiv:2511.21395)
│   │   ├── Gaps_Monet.md                        # Forensic analysis of detached proxy loss (src/trainer.py L39)
│   │   └── paper_src/                           # Extracted arXiv LaTeX source
│   │
│   └── mirage/                                  # Paper 3: Mirage (Machine Mental Imagery, arXiv:2506.17218)
│       ├── Gaps_Mirage.md                       # Forensic analysis of Stage 2 visual loss drop (src/trainer.py L38)
│       └── paper_src/                           # Extracted arXiv LaTeX source
│
├── wsdm_experiments/                            # Empirical evaluation scripts & outputs
│   ├── outputs_w1_w3/                           # SAT/Mull lineage benchmark matrices
│   └── outputs_w2_w4/                           # Attention knockout & activation patching matrices
└── mull/                                        # Core Mull-Tokens model code and lmms-eval harness
```

---

## 3. The 4-Step Forensic Protocol for Autonomous Agents

To ensure scientific rigor, avoid hallucinations, and eliminate superficial benchmarking, every framework is audited using this strict 4-step pipeline:

### Step 1: Paper & Codebase Ingestion (`STEP1_INGESTION_MANIFEST.md`)
1. Download official arXiv PDF, extract text (`pdftotext`), and unpack all LaTeX `.tex` sources into `paper_src/`.
2. Clone official GitHub repository into `repo/`.
3. Download HuggingFace checkpoint metadata (configs, tokenizers, training arguments, index) without downloading multi-GB weights.
4. Produce `STEP1_INGESTION_MANIFEST.md` cataloging every file, SHA256 checksum, exact line count, and structural outline.

### Step 2: Line-by-Line Code-vs-Paper Forensic Trace (`Gaps_<Model>.md` & `CODE_VS_PAPER_TRACE.md`)
1. Cross-reference every mathematical equation in `paper_src/` against the PyTorch forward pass, loss calculations, and training collators in `repo/`.
2. Identify:
   - Tautological shortcuts (e.g., reconstructing tokens already present in context).
   - Broken or detached gradients (`.detach()`, `@torch.no_grad()`, span zero-masking).
   - Discrepancies between paper claims and actual training configs (e.g., sequence length, learning rate).
   - Biased answer parsers or custom evaluation regexes.
3. Require adversarial audit verification until certified with `VERDICT: PASS (0 MISTAKES)`.

### Step 3: Bespoke Mechanistic Experiment Design (`PREREGISTRATION_<Model>.md`)
1. Formulate 2–3 competing falsification hypotheses targeting the model's specific architectural seam.
2. Build standalone, reproducible Python test scripts in `diagnostics/` (executable on CPU under `CUDA_VISIBLE_DEVICES=""`).
3. Pre-register statistical decision boundaries (e.g., McNemar paired test, cosine similarity threshold > 0.95, gradient norm $\equiv 0$).

### Step 4: Cross-Model Master Synthesis (`UNIFIED_THEORY_OF_LATENT_TOKENS.md`)
1. Synthesize empirical findings across all studied architectures into a unified taxonomy of latent multimodal failure modes.

---

## 4. How to Resume or Run Tasks via Autonomous Agents

If you clone this repository to an autonomous agent (Antigravity, Devin, Claude Code), instruct the agent:

```bash
# 1. Read the state of the project
cat external_models/PIPELINE_STATE.md
cat MASTER_RESEARCH_PLAN.md

# 2. Inspect the Master Synthesis
cat external_models/UNIFIED_THEORY_OF_LATENT_TOKENS.md

# 3. Run CPU diagnostic verifications
python external_models/lvr/diagnostics/test_roi_copying.py
python external_models/lvr/diagnostics/test_grpo_zero_grad.py
```

All code citations are referenced with bit-exact file paths and line numbers.
