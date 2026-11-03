# Technical Review and Forensic Gap Analysis: Monet (Reasoning in Latent Visual Space)

> [!WARNING]
> **Verification status (2026-11-03):** This document has been revised against the primary sources: the official repository [`external_models/monet/repo/`](external_models/monet/repo/) at commit `08939998d3d643a73a316e349faa34f420429153` (`0893999`), the arXiv:2511.21395v2 LaTeX source, and reproducible execution harnesses in [`external_models/monet/verification/`](external_models/monet/verification/). Prior claims alleging that RL gradients are disconnected via `g.detach()`, student latents are precomputed offline, Stage 3 drops teacher representations, and latents align to ViT intermediate blocks have been verified to be false (see [`external_models/monet/VERIFICATION_REPORT.md`](external_models/monet/VERIFICATION_REPORT.md) §2). Findings M1–M10 document the verified architectural and empirical characteristics of Monet.

**Target Paper**: *Monet: Reasoning in Latent Visual Space* (`arXiv:2511.21395v2`, CVPR 2026 Submission)  
**Target Repository**: [`external_models/monet/repo/`](external_models/monet/repo/) at commit `08939998d3d643a73a316e349faa34f420429153` (`0893999`)  
**LaTeX Source**: [`external_models/monet/paper_src/`](external_models/monet/paper_src/)  
**Verification Suite**: [`external_models/monet/verification/`](external_models/monet/verification/)  
**Verification Report**: [`external_models/monet/VERIFICATION_REPORT.md`](external_models/monet/VERIFICATION_REPORT.md)  

---

## 1. Evaluation of Prior Audit Claims Against Primary Sources

### 1.1 Latent-Only Backpropagation in SFT Stage 2 (`src/trainer.py` lines 11–42, 198–200)
- **Prior Claim**: The audit claimed that Monet's RL algorithm (VLPO) avoids policy gradient backpropagation by calling `g.detach()` and computing a surrogate proxy loss that eliminates gradient flow.
- **Source Verification (False)**:
  1. **Component Context**: `compute_latents_only_loss` belongs exclusively to **SFT Stage 2** supervised distillation ([`external_models/monet/repo/src/trainer.py` lines 198–200](external_models/monet/repo/src/trainer.py#L198-L200)), not to VLPO or RL training. VLPO is implemented in [`external_models/monet/repo/RL/verl/workers/actor/dp_actor.py`](external_models/monet/repo/RL/verl/workers/actor/dp_actor.py).
  2. **Mathematical Equivalence**: The proxy loss implements the gradient described in Section X of the supplementary material ([`external_models/monet/paper_src/sec/X_suppl.tex` lines 77–85](external_models/monet/paper_src/sec/X_suppl.tex#L77-L85)). For latents $\mathbf{z}$ produced by model parameters $\theta$ and alignment loss $\mathcal{L}$, setting $\text{proxy} = \sum \mathbf{z} \cdot [\nabla_{\mathbf{z}} \mathcal{L}].\text{detach}()$ yields:
     $$\frac{\partial\,\text{proxy}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \frac{\partial \mathbf{z}}{\partial \theta}$$
     This passes gradients through the latent pathway into $\theta$ while excluding direct non-latent paths through earlier activations.
  3. **Numerical Verification**: In [`external_models/monet/verification/proxy_grad_identity.py`](external_models/monet/verification/proxy_grad_identity.py), running the repo's function verbatim confirms that the gradient of `proxy_loss` with respect to model weights $W_1$ matches the latent-path gradient with $\max |\text{diff}| = 0.0$ and a nonzero gradient norm of $15.16$.

Verbatim implementation from [`external_models/monet/repo/src/trainer.py` lines 11–42](external_models/monet/repo/src/trainer.py#L11-L42):
```python
def compute_latents_only_loss(latents, loss_for_latents):
    '''
    Compute a loss (`loss_for_latents`) that backpropagates only through the latent embeddings `latents`.
    '''
    def _flatten_tensors(x):
                # Flatten nested [list/tuple of Tensors] into a flat list of Tensors
                if isinstance(x, (list, tuple)):
                    out = []
                    for y in x:
                        out.extend(_flatten_tensors(y))
                    return out
                return [x]

    ce_vec_list = _flatten_tensors(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,   # we won't reuse the 3rd graph
        create_graph=False,   # stop higher-order graph
        allow_unused=True     # in case some ce vectors are not used
    )

    # Replace None with zeros for unused elements
    safe_grads = []
    for v, g in zip(ce_vec_list, grads):
        if g is None:
            # Create a zero tensor on the same device/dtype/shape
            g = torch.zeros_like(v)
        safe_grads.append(g.detach())  # detach to stop any 3rd-forward param pathg

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    return proxy_loss
```

Usage in Stage 2 trainer ([`external_models/monet/repo/src/trainer.py` lines 198–200](external_models/monet/repo/src/trainer.py#L198-L200)):
```python
        if self.args.emphasize_latent_weight != 0.0 and alignment_loss.item() != 0.0: # latent-only backpropagation for alignment loss
            latent_only_loss = compute_latents_only_loss(outputs.ce_patch_vec, self.args.alignment_weight * alignment_loss)
            loss = self.args.emphasize_latent_weight * latent_only_loss + teacher_ce_loss
```

### 1.2 Offline Tensor Loading Functionality (`src/trainer.py` lines 44–67)
- **Prior Claim**: The audit stated that the student's continuous latent representations are precomputed offline and loaded from disk via `load_offline_tensor`.
- **Source Verification (False)**:
  `load_offline_tensor` loads precomputed **teacher representations** used as fixed distillation targets: observation-token representations in Stage 2, and target latent vectors in Stage 3. The student model's continuous latents are computed **online** during the forward pass in [`external_models/monet/repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py` lines 1914–1918](external_models/monet/repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py#L1914-L1918) and stored in `ce_patch_vec`.

Verbatim function definition from [`external_models/monet/repo/src/trainer.py` lines 44–47](external_models/monet/repo/src/trainer.py#L44-L47):
```python
def load_offline_tensor(tensor_dir, batch_metadata, alignment_layer="all_layers", rep_type="rep", align_poss="obs"):
    '''
    Load precomputed teacher representations (observation tokens for the alignment in SFT stage 2 or the latent embeddings for SFT stage 3)
    '''
```

### 1.3 Offline Teacher Snapshots and Dynamic Adaptation (`src/trainer.py` lines 50–65)
- **Prior Claim**: The audit asserted that using precomputed offline teacher snapshots completely disables adaptive dynamic reasoning.
- **Source Verification (Unsupported)**:
  Caching teacher representations for distillation is a standard offline distillation method explicitly documented in the paper (Section 3.2, lines 150–163: "we fix $\mathbf{h}^*$ and maximize cosine similarity"). No experimental evidence was provided to support the assertion that this practice prevents dynamic reasoning.

### 1.4 Stage 3 Target Representation Alignment (`src/trainer.py` lines 270–315)
- **Prior Claim**: The audit claimed that Stage 3 drops teacher representations, causing unconstrained representation drift.
- **Source Verification (False)**:
  In SFT Stage 3, the student's generated latents are explicitly aligned to precomputed teacher latents. Line 273 loads teacher latents, and lines 301 and 312 compute joint cross-entropy and alignment loss:

Verbatim lines from [`external_models/monet/repo/src/trainer.py` lines 272–273, 301, 311–312](external_models/monet/repo/src/trainer.py#L272-L273):
```python
        # Load precomputed teacher latents
        teacher_latents = load_offline_tensor(self.teacher_latent_dir, batch_metadata=inputs['metadata'], alignment_layer=self.args.alignment_layer, rep_type="latent")
```
```python
        inputs['loss_type'] = ['ce', 'alignment']
```
```python
        alignment_loss = student_outputs.loss_dict['alignment']
        loss = student_ce_loss + self.alignment_weight * alignment_loss
```

### 1.5 Alignment Target Layer Configuration (`src/utils.py` lines 59–60)
- **Prior Claim**: The audit stated that Monet hardcodes `alignment_layer = "all_layers"`, forcing latents to match ViT intermediate blocks.
- **Source Verification (False)**:
  `"all_layers"` is the default argument in CLI parsing (`src/utils.py`); the alignment target layer is configurable (`"all_layers"` vs `"last_layer"`). Furthermore, the distillation targets are **LLM hidden states**, not ViT visual transformer blocks.

---

## 2. Verified Architectural & Empirical Findings (M1–M10)

### M1. VLPO Gaussian Log-Prob Scores Latents Against the Next Latent's Producer (Severity: High)
- **Paper Formulation (Equations 9–10)**:
  The paper defines the policy probability for a continuous latent at position $t$ as a Gaussian $\mathcal{N}(\mathbf{h}_t^\theta, \sigma^2 \mathbf{I})$, where $\mathbf{h}_t^\theta$ is the hidden state generated in the same context by the current policy. The importance weight ratio between current policy $\theta$ and rollout policy $\theta_{\text{old}}$ is:
  $$\frac{\pi_\theta(z_t)}{\pi_{\theta_{\text{old}}}(z_t)} = \exp\left(-\frac{\|\mathbf{h}_t^\theta - z_t\|^2 - \|\mathbf{h}_t^{\theta_{\text{old}}} - z_t\|^2}{2\sigma^2}\right)$$
- **Repository Implementation**:
  1. *Rollout*: In [`external_models/monet/repo/RL/monet_models/vllm/monet_gpu_model_runner.py` lines 1099–1121](external_models/monet/repo/RL/monet_models/vllm/monet_gpu_model_runner.py#L1099-L1121) and lines 1305–1339, the hidden state output produced at step $q-1$ is registered into `st["pending"]` and fed as the input embedding for step $q$.
  2. *Actor Segment Indexing*: In [`external_models/monet/repo/RL/verl/workers/actor/dp_actor.py` line 114](external_models/monet/repo/RL/verl/workers/actor/dp_actor.py#L114):
```python
        for (s_pos, e_pos) in matched[:]:
            inner = torch.arange(s_pos, e_pos, device=device, dtype=torch.long)  # (Lseg,)
```
     The actor identifies latent token indices across the range $[s_{\text{pos}}, e_{\text{pos}})$.
  3. *Actor Latent Insertion*: In [`external_models/monet/repo/RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py` line 1858](external_models/monet/repo/RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py#L1858):
```python
        if latent_poss is not None and latents is not None:
            #breakpoint()
            inputs_embeds[0, latent_poss] = latents
```
  4. *Actor Mean Assignment*: In [`external_models/monet/repo/RL/verl/workers/actor/dp_actor.py` lines 150–152](external_models/monet/repo/RL/verl/workers/actor/dp_actor.py#L150-L152):
```python
    # Shape: [L, D]
    latent_outputs = last_hidden_state[0, latent_poss, :]
    latents = latents.to(latent_outputs)
```
     The actor takes `last_hidden_state[0, latent_poss, :]` (the output hidden state **at** index $q$) as the Gaussian mean $\mu_q$ for the latent vector **input at** index $q$. Because the output at index $q$ produces the latent vector for index $q+1$, the implementation evaluates latent $j$ against the producer of latent $j+1$.
  5. *Rollout Noise*: In [`external_models/monet/repo/RL/monet_models/vllm/monet_gpu_model_runner.py` lines 1330–1331](external_models/monet/repo/RL/monet_models/vllm/monet_gpu_model_runner.py#L1330-L1331):
```python
                        mu = last_token_h[i].detach()
                        sampled = mu  # + torch.randn_like(mu)
```
     Gaussian perturbation noise is commented out in rollout generation.

- **Empirical Verification (`monet_vlpo_harness.py`)**:
  Testing the verbatim functions on a controlled model (`INIT=0.3 LAYERS=4`, $K=10$, seeds 0, 1, 2) produces:

| Metric | Repo Implementation | Aligned Producer Mean (`latent_poss - 1`) |
|---|:---:|:---:|
| $\cos(z_j, \mu_j)$ | $-0.21$ to $0.43$ | **1.000** |
| $\cos(z_{j+1}, \mu_j)$ | **1.000** | — |
| On-policy latent log-prob ($\sigma = 10$) | $-0.37$ to $-0.85$ per latent | $\approx -1\times 10^{-6}$ |
| $\|\nabla_\theta \mathcal{L}_{\text{latent}}\|$ at $\theta = \theta_{\text{old}}$ | $12.59$ to $21.32$ | $0.071$ to $0.180$ (fp16 precision limit) |

- **Consequences**:
  1. The objective regularizes cosine distance between adjacent latents: $-\|h_q - z_q\|^2 = -\|h_q - h_{q-1}\|^2$. Under positive advantage, it encourages consecutive latents to remain identical; under negative advantage, it pushes consecutive latents apart.
  2. If the mean were aligned to the latent's actual producer ($h_{q-1}$), the term would be nearly gradient-free on-policy because rollout generation operates deterministically without noise.

---

### M2. VLPO Environment Variables Never Set in Released Training Code (Severity: High, Reproducibility)
- **Code Reality**:
  In [`external_models/monet/repo/RL/verl/workers/actor/dp_actor.py` lines 193–201](external_models/monet/repo/RL/verl/workers/actor/dp_actor.py#L193-L201):
```python
            try:
                latent_poss = []
                start_id = int(os.getenv("ABS_VIS_START_ID"))
                end_id = int(os.getenv("ABS_VIS_END_ID"))
                _, varlen_by_batch = collect_varlen_segment_indices(
                    input_ids=micro_batch["input_ids"], # (micro_batch["input_ids"][1]==151666).nonzero() (micro_batch["input_ids"][1]==151667).nonzero()
                    attention_mask=micro_batch["attention_mask"],
                    start_id=start_id, end_id=end_id,
                )
```
- **Error Propagation**:
  1. No repository file sets `ABS_VIS_START_ID` or `ABS_VIS_END_ID`. The patch script [`external_models/monet/repo/monet_rl_patch.py` lines 63–64](external_models/monet/repo/monet_rl_patch.py#L63-L64) sets `LATENT_START_ID` and `LATENT_END_ID`, while `RL/examples/vlpo_train.sh` sets neither.
  2. When unset, `int(None)` raises a `TypeError` (reproduced in `monet_vlpo_harness.py` Q4).
  3. The exception is caught by the fallback block at lines 232–235:
```python
            except Exception:
                print(f"[WARNING] Unexpected error before the latent importance sampling. Fall back to vanilla prob computation for this mirco batch.")
                output_hidden_states = False
                pass
```
  4. As a result, `latents` remains `None`, no continuous latent vectors are injected into `inputs_embeds` during actor forward passes, and no VLPO latent importance weights are calculated. The training pipeline silently executes standard GRPO on discrete token embeddings.

---

### M3. Positional Layout Discrepancy Between SFT Training and Inference/RL (Severity: High)
- **Token Positioning Across Stages**:

| Aspect | SFT Training (Stages 1–3) | Inference & RL Rollout |
|---|---|---|
| Sequence Layout | `<abs_vis_token>` + $K \times$ `<abs_vis_token_pad>` + `</abs_vis_token>` | `<abs_vis_token>` followed by dynamic latents |
| Latent Positions | Pad positions $s+1 \dots s+K$ | Positions $s \dots s+K-1$ |
| First Latent Input | Generated from output at position $s$ (`<abs_vis_token>`) | Replaces input embedding at position $s$ |
| Start Token Handling | Retains learned vocabulary embedding | Embedding is overwritten at $s$ |
| Code Locations | `src/main.py` lines 187–188, 258; `modeling_qwen2_5_vl_monet.py` lines 1736, 1914–1918 | `monet_gpu_model_runner.py` lines 1099–1116, 1327–1333; `inference/vllm/monet_gpu_model_runner.py` lines 1655–1666 |

In SFT ([`external_models/monet/repo/src/main.py` lines 187–188](external_models/monet/repo/src/main.py#L187-L188)):
```python
    # add `<abs_vis_token><abs_vis_token_pad>...</abs_vis_token>` after each `<|vision_start|><|image_pad|><|vision_end|>` for each `<|im_start|>assistant` content
    texts = add_latent_pad_after_auxiliary_img(texts, args.latent_size, "<abs_vis_token_pad>")
```
In deployment/rollout ([`external_models/monet/repo/inference/vllm/monet_gpu_model_runner.py` lines 1655–1666](external_models/monet/repo/inference/vllm/monet_gpu_model_runner.py#L1655-L1666)), the activation token sets `pending`, and the subsequent step overwrites position $s$. In addition, `inference/vllm/monet_gpu_model_runner.py` line 329 sets `latent_size = int(os.getenv("LATENT_SIZE", 0))`, which defaults to 0 unless manually overridden via environment variables.

---

### M4. SFT Stage 3 Script Argparse Validation Failure (Severity: Medium, Reproducibility)
- In [`external_models/monet/repo/script_examples/sft_stage3.sh` line 42](external_models/monet/repo/script_examples/sft_stage3.sh#L42):
```bash
  --stage "avt_v5_stage2" \
```
- In [`external_models/monet/repo/src/utils.py` line 19](external_models/monet/repo/src/utils.py#L19):
```python
    parser.add_argument("--stage", type=str, default="sft_stage1", choices=['sft_stage1', 'sft_stage2', 'sft_stage3'])
```
- Passing `"avt_v5_stage2"` violates the choices constraint, causing `src.main` to exit immediately with code 2.

---

### M5. Undisclosed Reward Function and API Judge Configuration (Severity: Medium)
- **Reward Formula**:
  [`external_models/monet/repo/RL/examples/reward_function/monet_reward_function.py` line 244](external_models/monet/repo/RL/examples/reward_function/monet_reward_function.py#L244) computes:
```python
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score - length_penalty_weight * length_penalty[i],
```
  with `format_weight = 0.1` and `length_penalty_weight = 0.001` (penalizing responses exceeding reference length).
- **Repetition Penalty**:
  Lines 240–241 allow `accuracy_score` to evaluate to `-1.0` when repetition penalty is enabled.
- **LLM Judge**:
  Lines 176–219 implement `rule_then_api_batch_judge`, which falls back to `api_name = 'gemini-2.5-pro'` whenever rule-based verification does not match. In the repository README (line 181), exact-match evaluation was replaced by an API-based judge.

---

### M6. Test-Time Latent Budget Selected Per-Benchmark on Test Sets (Severity: Medium)
- In [`external_models/monet/paper_src/sec/4_experiment.tex` line 41](external_models/monet/paper_src/sec/4_experiment.tex#L41):
  "At test time, we select the best latent size from $\{8, 10, 12, 16\}$."
- Evaluating multiple values of $K$ on test benchmarks and reporting the maximum introduces hyperparameter selection bias on the test set, whereas comparative baselines evaluate under fixed configurations.

---

### M7. Stage-2 Attention Mask Disallows Intra-Image Attention in Auxiliary Images (Severity: Medium)
- In [`external_models/monet/repo/src/utils.py` lines 570–575](external_models/monet/repo/src/utils.py#L570-L575):
```python
                # Ensure only A_i (and not others) can see I_i
                if I_idx.numel() and not not_mask_image:
                    not_A = torch.ones(Lb, dtype=torch.bool, device=device)
                    not_A[A_idx] = False
                    not_A_idx = torch.nonzero(not_A, as_tuple=False).squeeze(-1)
                    if not_A_idx.numel():
                        allowed[b][not_A_idx[:, None], I_idx] = False
```
- Running [`external_models/monet/verification/monet_mask_check.py`](external_models/monet/verification/monet_mask_check.py) demonstrates that all non-latent positions (`not_A_idx`), including the auxiliary image tokens themselves, have their attention masks to auxiliary image tokens set to `False`. Auxiliary image tokens cannot attend to one another or to themselves.

---

### M8. Loss Weighting and Epoch Schedules Differ From Paper Specifications (Severity: Low–Medium)
- **Effective Alignment Weight**:
  In [`external_models/monet/repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py` line 247](external_models/monet/repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py#L247), alignment loss computes `.mean()` over layers and positions rather than the unnormalized sum $\sum_{\text{layers}} (1 - \cos)$ stated in paper Section 3.2. With `emphasize_latent_weight = 2.0` and `alignment_weight = 2.0` (`src/trainer.py` lines 198–200), the effective multiplier is 4.0 per layer-mean.
- **CE Observation Token Weighting**:
  `ce_emphasize_factor` scales observation token cross-entropy by $2.0\times$ in Stage 1 and $4.0\times$ in Stages 2–3.
- **Training Duration**:
  `script_examples/sft_stage3.sh` sets `--epochs 2`, whereas paper Section 4.1 specifies 1000 steps ($\approx 1.03$ epochs).
- **Masking Flag**:
  In `src/utils.py` line 64, `--latent_can_see_all_previous` has `action='store_true', default=True`, which cannot be toggled to `False` via CLI.

---

### M9. Truncated Backpropagation Across Latent Steps in SFT Forward Pass (Severity: Low, Disclosure)
- In [`external_models/monet/repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py` lines 1915–1924](external_models/monet/repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py#L1915-L1924):
```python
                    else: # set the latent token embedding as the previous token's hidden state
                        prev_hidden = batch_last_hidden_state[b, pos - 1, :].unsqueeze(0).unsqueeze(0).contiguous()
                        latent_embed = prev_hidden.clone() if self.training else prev_hidden.detach()   
                        ce_patch_pos[b].append(pos)
                        ce_patch_vec[b].append(latent_embed[0, 0])

                    step_pos_ids = pos_ids_s[:, :, pos : pos + 1]

                    step_out = self.language_model(
                        input_ids=None,
                        inputs_embeds=latent_embed.detach(),
```
- While `latent_embed` is created via `.clone()` to retain its local computation graph for alignment loss at position $pos$, it is passed to the next autoregressive step as `latent_embed.detach()`. Gradients propagate through each latent step into $\theta$, but do not propagate recurrently across the sequential chain of earlier latents.

---

### M10. Results Table Inconsistencies and Non-Feasible Values (Severity: Low–Medium)
- **Dual Benchmark Entries**:
  In [`external_models/monet/paper_src/sec/3_method.tex` line 142](external_models/monet/paper_src/sec/3_method.tex#L142), Monet-SFT on MME-RealWorld-Lite is reported as **55.03**. In the main results table in [`external_models/monet/paper_src/sec/4_experiment.tex`](external_models/monet/paper_src/sec/4_experiment.tex), the identical model is reported as **52.68**. Relative to 55.03, VLPO's published gain on MME-RealWorld-Lite is $+0.47$ rather than $+2.82$.
- **Mathematically Infeasible Percentages** ([`external_models/monet/verification/table_checks.py`](external_models/monet/verification/table_checks.py)):
  Across 100 values checked against evaluation split counts $N$:
  1. Vanilla SFT on HRBench4K-FSP: reported as $78.28$ (for $N=400$, possible values must be multiples of $0.25$; nearest feasible value is $78.25$).
  2. DeepEyes on MME-RealWorld-Lite: reported as $54.28$ (not expressible as $k/1919$).
- **Bolding Discrepancies**:
  In Table 1, VisualPuzzles–Spatial bolds vanilla SFT ($29.02$) over Pangea-7B ($32.5^*$), and Deductive bolds Monet-7B ($47.50$) while omitting bolding on the tied base model ($47.50$).

---

## 3. Pre-Registered Mechanistic Evaluation Protocols (E-M1 to E-M6)

*Note: The following experimental protocols are pre-registered designs requiring full model weights and GPU compute; they have not been executed.*

| ID | Research Question | Method & Variations | Decision Boundary |
|---|---|---|---|
| **E-M1** | Does the VLPO continuous latent objective contribute to downstream performance? | Train VLPO on Monet-SFT with: (a) verbatim repo objective; (b) producer-aligned mean `latent_poss - 1`; (c) producer-aligned mean with Gaussian sampling $\mathbf{z} = \mu + \sigma \epsilon$; (d) latent loss omitted (GRPO with latent slots inserted). Fixed $K=10$, 3 seeds. | If (a) $\approx$ (d) and (b) $\approx$ (d), observed RL gains do not stem from continuous latent policy gradients. |
| **E-M2** | What is the impact of unset environment variables and layout mismatch on actor log-probabilities? | On 200 rollouts, compare text-token log-probs between rollout vs actor with environment variables set, unset, and aligned to SFT layout. | Log-prob divergence exceeding fp16 precision bounds confirms train-test distribution shift. |
| **E-M3** | Does the token layout shift between SFT and rollout degrade downstream accuracy? | Evaluate Monet-SFT under released inference runner vs runner patched to place latents at pad positions $s+1 \dots s+K$. | Statistically significant accuracy difference via McNemar test indicates positional distribution shift. |
| **E-M4** | Does test-time selection of latent budget $K$ inflate benchmark gains? | Report evaluation accuracy across all $K \in \{8, 10, 12, 16\}$ alongside fixed $K = K_{\text{train}}$ with 10k bootstrap confidence intervals. | If fixed-$K$ performance fails to maintain statistical significance over baseline, gains are attributable to test-set search. |
| **E-M5** | What proportion of reported accuracy is attributable to the LLM judge? | Evaluate Monet and non-starred baselines using exact match vs Gemini-2.5-Pro API judge. | The relative advantage of Monet over baselines must remain consistent across both judges. |
| **E-M6** | Does the model depend on continuous latent representations during inference? | Intervene at test time by replacing latent vectors with zeros, random noise, or tokens from alternate images; and test $K=0$. | If performance remains within confidence bounds of intact latents, continuous representations are non-operative. |
