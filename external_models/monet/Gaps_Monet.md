# Critical Review and Forensic Gap Analysis: Monet (Reasoning in Latent Visual Space)

**Target Paper**: *Monet: Reasoning in Latent Visual Space* (`arXiv:2511.21395`, CVPR 2026 Submission)  
**Target Repository**: `/home/devraj/MTok/external_models/monet/repo/`  
**LaTeX Source**: `/home/devraj/MTok/external_models/monet/paper_src/`  
**Audit Standard**: Publication-Grade Forensic Rigor (Equivalent to `FinalGaps.md`)  
**Auditor**: Forensic Builder & Hostile Auditor (Paper 2 Certification)

---

## 1. High-Level Claims, Framing & Conceptual Contradictions

### 1.1 The "Continuous RL Policy Gradient" Fallacy (The Detached Proxy-Loss Trick)
- **The Paper's Framing**:
  The paper claims that standard RL (GRPO) fundamentally fails for continuous visual latents because continuous hidden states do not possess discrete token probabilities $\log \pi_\theta(z_t)$. The authors claim to invent **VLPO (Visual-Latent Policy Optimization)**, deriving policy gradients directly over continuous visual trajectories.
- **The Forensic Code Reality (`src/trainer.py` lines 11–42, lines 195–203)**:
  Inspection of the actual implementation reveals how Monet bypasses true continuous policy gradient backpropagation through time:
  1. The continuous latent representations are computed offline and loaded from pre-saved disk tensors (`load_offline_tensor` lines 44–67).
  2. In `compute_latents_only_loss` (lines 11–42), Monet computes first-order gradients `grads = torch.autograd.grad(outputs=loss_for_latents, inputs=ce_vec_list, retain_graph=True, create_graph=False)`.
  3. Crucially, at **line 39**, Monet calls **`safe_grads.append(g.detach())`**!
  4. It then constructs a surrogate scalar:
     $$\text{proxy\_loss} = \sum (\mathbf{v} \cdot \mathbf{g}.\text{detach}())$$
     (line 41).
  5. **Consequence**: This stops higher-order graph construction (`create_graph=False`) and detaches the gradient vector $\mathbf{g}$! The latent embeddings are updated solely via a linear inner-product proxy with a detached constant gradient, completely severing recurrent gradient flow through time and through the vision-language backbone.

---

## 2. Multi-Stage Distillation SFT vs. Base Model Superiority

### 2.1 Teacher-Student Reliance on Offline Precomputed Latents
- In `src/trainer.py` lines 50–65, alignment is not computed end-to-end between dynamic student and teacher models.
- Instead, teacher representations (`teacher_reps_dir`) must be pre-extracted to `.pt` files (`rep_{alignment_layer}_{dataset_name}_{sample_id}.pt`).
- This means the student continuous thoughts are locked to static offline snapshot vectors, completely disabling adaptive dynamic reasoning during training.

### 2.2 SFT Stage 1 vs Stage 2 vs Stage 3 Representation Drift
- In Stage 1, the model learns standard token autoregression.
- In Stage 2, alignment loss is added between observation tokens and teacher representations.
- In Stage 3, the model attempts to generate latent thoughts without explicit observation tokens.
- Just like Mull-Tokens and LVR, dropping the ground-truth teacher representations in Stage 3 causes massive drift in the unconstrained latent space.

---

## 3. Data Formatting & Evaluation Quirks

### 3.1 Hardcoded Layer-Wise Projections
- Monet hardcodes `alignment_layer = "all_layers"` or specific intermediate layers, forcing the latent representations to match ViT intermediate blocks regardless of question complexity.
