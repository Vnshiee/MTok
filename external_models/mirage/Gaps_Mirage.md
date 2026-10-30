# Critical Review and Forensic Gap Analysis: Mirage (Machine Mental Imagery)

**Target Paper**: *Mirage: Machine Mental Imagery* (`arXiv:2506.17218`, CVPR 2026 Submission)  
**Target Repository**: `/home/devraj/MTok/external_models/mirage/repo/`  
**LaTeX Source**: `/home/devraj/MTok/external_models/mirage/paper_src/`  
**Audit Standard**: Publication-Grade Forensic Rigor (Equivalent to `FinalGaps.md`)  
**Auditor**: Forensic Builder & Hostile Auditor (Paper 3 Certification)

---

## 1. The Catastrophic Stage-1 to Stage-2 Representational Collapse

### 1.1 Complete Dropping of Visual Supervision in Stage 2 (`src/trainer.py` lines 6–39)
The code reveals an exact parallel to the **Mull-Tokens relaxation collapse**:
1. **Stage 1 (`CustomTrainerStage1` lines 6–26)**:
   - Uses a joint loss combining Cross-Entropy and Cosine Similarity against ground-truth image patch embeddings:
     $$\mathcal{L}_{\text{Stage1}} = 0.1 \cdot \mathcal{L}_{\text{CE}} + (1 - \operatorname{CosineSimilarity}(\mathbf{h}_{\text{pred}}, \mathbf{h}_{\text{gt}}))$$
   - This anchors the latent tokens directly to the visual feature manifold.
2. **Stage 2 (`CustomTrainerStage2` lines 28–39)**:
   - The visual cosine similarity loss is **completely removed**:
     $$\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{CE}}$$
   - Verified verbatim at line 38: `loss = ce_loss`.
3. **Forensic Verdict**:
   During Stage 2 fine-tuning, with zero gradient penalty on visual similarity, the "mental imagery" tokens suffer **Catastrophic Representational Drift**, unmooring from visual space and degenerating into arbitrary prompt-tuning prefixes and format-lock tokens.

---

## 2. Hardcoded Token Sequences & Architecture Seams

### 2.1 The Latent Token Sequence Structure
In `src/main.py` lines 36–38 and 61–63, special tokens `<|latent_start|>`, `<|latent_pad|>`, and `<|latent_end|>` are added to the tokenizer.
- Just like Mull-Tokens and LVR, the latent length is fixed by data collators (`collate_fn_stage1` line 73 and `collate_fn_stage2` line 124).
- The model lacks dynamic early stopping for mental imagery rollouts.
