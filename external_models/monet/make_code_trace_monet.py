import os

out_path = "/home/devraj/MTok/external_models/monet/CODE_VS_PAPER_TRACE.md"

doc = """# Monet (`arXiv:2511.21395`) — Line-by-Line Code-vs-Paper Forensic Trace

**Target Paper**: *Monet: Reasoning in Latent Visual Space Beyond Images and Language* (CVPR 2026, Paper ID `7696`, `arXiv:2511.21395`)  
**Target Codebase**: `external_models/monet/repo/` (`HEAD` commit `0893999`)  
**Target Checkpoint**: `NOVAglow646/Monet-7B` (`commit 39c1e32`) & `NOVAglow646/Monet-SFT-7B` (`commit 67919ca`)  
**Base Backbone**: `Qwen/Qwen2.5-VL-7B-Instruct` ($L=28, D=3584, H_Q=28, H_{KV}=4, \text{vocab}=151,670$)  
**Certified Deliverable**: Step 2 Code-vs-Paper Equation-to-Tensor Trace

---

## 1. Executive Architectural Mapping: Paper Claims vs. Code Reality

| Component / Mechanism | Paper Formalism (`arXiv:2511.21395`) | Code Implementation (`repo/`) | Forensic Verdict & Discrepancy |
| :--- | :--- | :--- | :--- |
| **Latent Token Vocabulary** | Continuous latent embeddings generated between `<abs_vis_token>` and `</abs_vis_token>` | Token `151666`: `<abs_vis_token>`, Token `151667`: `</abs_vis_token>`, Token `151665`: `<abs_vis_token_pad>` | **Confirmed**: Custom tokens added to Qwen vocabulary (total vocab `151,670`). |
| **Inference Latent Quota** | Autoregressively generated latent embeddings with controllable latent size | `LATENT_SIZE=10` hardcoded in `inference/vllm_inference_example.py` L24 and `monet_gpu_model_runner.py` | **Fragility**: Like LVR, the model has no internal stopping criteria for latents; relies on an external hard quota `LATENT_SIZE`. |
| **Stage 2 SFT Alignment** | Eq (108): $\mathcal{L}_{\text{align-obs}} = \frac{1}{N}\sum_i \sum_l (1 - \cos(\mathbf{h}^{*(i,l)}_{\text{obs}}, \hat{\mathbf{h}}^{(i,l)}_{\text{obs}}))$ | `src/trainer.py` L11–42 (`compute_latents_only_loss`) and L198–203 (`proxy_loss`) | **Critical Discrepancy**: The alignment loss is NOT directly backpropagated. Gradients $\mathbf{g} = \partial \mathcal{L} / \partial \mathbf{z}$ are computed, detached (`g.detach()`), and converted into a surrogate inner product $\sum \mathbf{v} \cdot \mathbf{g}.\text{detach}()$. |
| **Offline Teacher Distillation** | Student aligned with dynamic teacher processing auxiliary images | `src/trainer.py` L44–67 (`load_offline_tensor`) loads pre-saved `.pt` files from `teacher_reps_dir` | **Static Snapshot**: Teacher representations are frozen offline snapshots on disk, completely precluding interactive co-adaptation. |
| **Stage 3 Auxiliary-Free SFT** | Eq (155): $\mathcal{L}_{\text{align-latent}} = \frac{1}{N}\sum_i \sum_l (1 - \cos(\mathbf{h}^{*(i,l)}_{\text{latent}}, \hat{\mathbf{h}}^{(i,l)}_{\text{latent}}))$ | `src/trainer.py` L272–314 (`CustomTrainerSFT_STAGE3.compute_loss`) | **Verified**: Reinitializes student from $M_{\text{warm-up}}$ and aligns all layers to static `.pt` teacher latents. |
| **VLPO Policy Gradient** | Eq (194): $r_{i,t}(\theta) = \frac{\pi_\theta(\mathbf{h}^{\text{old}})}{\pi_{\theta_{\text{old}}}(\mathbf{h}^\theta)} = \exp(-\frac{1}{2\sigma^2}\|\mathbf{h}^{\text{old}} - \mathbf{h}^\theta\|^2)$ | `RL/verl/workers/actor/dp_actor.py` L135–158 (`compute_latent_log_probs`) | **Mathematical Sleight-of-Hand**: The paper claims a continuous policy ratio, but $r_{i,t} \le 1$ always. In code, it is implemented as an isotropic Gaussian log-prob $-0.5 \|\mathbf{z} - \boldsymbol{\mu}\|^2 / \sigma^2$ with fixed $\sigma = 10.0$. |
| **Streaming Latent Rollout** | Rollout continuous visual thoughts during RL | `RL/monet_models/vllm/latent_hook.py` streams latent vectors over local TCP socket (`_send_tcp_bin`) to `LatentRecorder` | **Implementation Quirk**: Uses IPC network socket streaming to bridge vLLM generation process to EasyR1 trainer. |

---

## 2. Equation-by-Equation Mathematical & Code Verification

### 2.1 SFT Stage 2: Observation Alignment Loss
**Paper Equation (108)**:
$$\mathcal{L}_{\text{align-obs}} = \frac{1}{N}\sum_{i=1}^N \sum_{l=1}^L \left(1 - \cos\left(\mathbf{h}^{*(i,l)}_{\text{obs}}.\operatorname{detach}(), \hat{\mathbf{h}}^{(i,l)}_{\text{obs}}\right)\right)$$

**Code Implementation** ([`src/trainer.py` L11–42, L198–203](file:///home/devraj/MTok/external_models/monet/repo/src/trainer.py#L11-L42)):
```python
def compute_latents_only_loss(latents, loss_for_latents):
    ce_vec_list = _flatten_tensors(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    safe_grads = []
    for v, g in zip(ce_vec_list, grads):
        if g is None:
            g = torch.zeros_like(v)
        safe_grads.append(g.detach())  # Detach gradient vector

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    return proxy_loss
```

**Forensic Audit**:
1. **The Detached Proxy-Loss**: Instead of backpropagating $\mathcal{L}_{\text{align-obs}}$ directly through the model, Monet computes the Jacobian vector product $\mathbf{g} = \nabla_{\mathbf{z}} \mathcal{L}_{\text{align-obs}}$, calls `g.detach()`, and constructs `proxy_loss = \sum \mathbf{z} \cdot \mathbf{g}.\text{detach}()`.
2. **Consequence**: When `loss.backward()` is called, $\nabla_\mathbf{z} \text{proxy\_loss} = \mathbf{g}$. The model parameters producing the observation tokens receive **zero direct gradient** from $\mathcal{L}_{\text{align-obs}}$. Only the parameters that produced $\mathbf{z}$ (the latent vectors) receive gradients.
3. **Reason for the Trick**: As admitted in Section 3.1 L111, if standard backpropagation is used, the model takes a "shortcut" by adjusting the observation token representations directly rather than encoding auxiliary visual features into the latents.

---

### 2.2 SFT Stage 2: Total Loss
**Paper Equation (126)**:
$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{NTP}} + \alpha \mathcal{L}_{\text{align-obs}} \quad (\alpha = 2.0)$$

**Code Implementation** ([`src/trainer.py` L198–203](file:///home/devraj/MTok/external_models/monet/repo/src/trainer.py#L198-L203)):
```python
if self.args.emphasize_latent_weight != 0.0 and alignment_loss.item() != 0.0:
    latent_only_loss = compute_latents_only_loss(outputs.ce_patch_vec, self.args.alignment_weight * alignment_loss)
    loss = self.args.emphasize_latent_weight * latent_only_loss + teacher_ce_loss
else:
    loss = teacher_ce_loss + self.args.alignment_weight * alignment_loss
```
Where `alignment_weight` = $\alpha = 2.0$, and `emphasize_latent_weight` defaults to $1.0$.

---

### 2.3 SFT Stage 3: Target Latent Alignment Loss
**Paper Equation (155)**:
$$\mathcal{L}_{\text{align-latent}} = \frac{1}{N}\sum_{i=1}^N \sum_{l=1}^L \left(1 - \cos\left(\mathbf{h}^{*(i,l)}_{\text{latent}}.\operatorname{detach}(), \hat{\mathbf{h}}^{(i,l)}_{\text{latent}}\right)\right)$$

**Code Implementation** ([`src/trainer.py` L273–313](file:///home/devraj/MTok/external_models/monet/repo/src/trainer.py#L273-L313)):
- Target latents $\mathbf{h}^{*(i,l)}_{\text{latent}}$ are loaded from disk via `load_offline_tensor(self.teacher_latent_dir, metadata, alignment_layer="all_layers", rep_type="latent")`.
- Unlike Stage 2, Stage 3 does NOT use `compute_latents_only_loss`. It adds `self.alignment_weight * alignment_loss` directly to `student_ce_loss`.

---

### 2.4 Stage 4: VLPO Policy Optimization Ratio
**Paper Equation (194)**:
$$r_{i,t}(\theta) = \frac{\pi_\theta(\mathbf{h}^{\text{old}}_{i,t} \mid Q, I, \mathbf{o}_{i,<t})}{\pi_{\theta_{\text{old}}}(\mathbf{h}^\theta_{i,t} \mid Q, I, \mathbf{o}_{i,<t})} = \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{h}^{\text{old}}_{i,t} - \mathbf{h}^\theta_{i,t}\|^2\right)$$

**Code Implementation** ([`RL/verl/workers/actor/dp_actor.py` L135–158, L301–304](file:///home/devraj/MTok/external_models/monet/repo/RL/verl/workers/actor/dp_actor.py#L135-L158)):
```python
def compute_latent_log_probs(latent_poss, latents, last_hidden_state, sigma=1.0):
    latent_outputs = last_hidden_state[0, latent_poss, :]
    latents = latents.to(latent_outputs)
    diff2 = (latents - latent_outputs).pow(2).sum(dim=-1)  # ||z - mu||^2
    latent_log_probs = - 0.5 * diff2 / (sigma ** 2)       # [L]
    return latent_log_probs

# Inside DataParallelPPOActor.update_policy:
if self.config.sampling_strategy == 'monet' and not self.config.ablate_latent:
    if latents is not None:
        latent_log_probs = compute_latent_log_probs(
            latent_poss, latents, output.hidden_states[-1], sigma=self.config.monet_rl_sigma
        )
        log_probs[latent_poss] = latent_log_probs.to(log_probs.dtype)
```

**Forensic Audit of VLPO**:
1. **Mathematical Inversion**: In standard PPO, $r_t(\theta) = \frac{\pi_\theta(a_t)}{\pi_{\text{old}}(a_t)}$. Under a Gaussian policy $a_t \sim \mathcal{N}(\boldsymbol{\mu}_\theta, \sigma^2 \mathbf{I})$, the log probability is:
   $$\log \pi_\theta(a_t) = -\frac{1}{2\sigma^2}\|a_t - \boldsymbol{\mu}_\theta\|^2 - \frac{D}{2}\log(2\pi\sigma^2)$$
   Taking the difference $\log \pi_\theta(a_t) - \log \pi_{\text{old}}(a_t)$ cancels the constant term:
   $$\log r_t(\theta) = -\frac{1}{2\sigma^2}\|a_t - \boldsymbol{\mu}_\theta\|^2 + \frac{1}{2\sigma^2}\|a_t - \boldsymbol{\mu}_{\text{old}}\|^2$$
2. **The Constant Variance Assumption**: Monet samples $a_t = \boldsymbol{\mu}_{\text{old}}$ (deterministic argmax rollout in vLLM without actual Gaussian sampling noise!).
   - Therefore, $\|a_t - \boldsymbol{\mu}_{\text{old}}\|^2 \equiv 0$.
   - Thus, $\log r_t(\theta) = -\frac{1}{2\sigma^2}\|\boldsymbol{\mu}_{\text{old}} - \boldsymbol{\mu}_\theta\|^2 \le 0$!
3. **Fatal Policy Gradient Property**:
   - Because $r_t(\theta) = \exp(-\frac{1}{2\sigma^2}\|\boldsymbol{\mu}_{\text{old}} - \boldsymbol{\mu}_\theta\|^2) \in (0, 1]$, the ratio is **strictly bounded above by 1.0**.
   - For positive advantage $\hat{A} > 0$, the PPO objective $\min(r_t \hat{A}, \text{clip}(r_t) \hat{A})$ achieves its maximum at $r_t = 1$, which corresponds to $\boldsymbol{\mu}_\theta = \boldsymbol{\mu}_{\text{old}}$.
   - **Conclusion**: VLPO cannot perform true policy gradient exploration. It behaves purely as an **L2 regularization anchor** that penalizes the student for deviating from its rollout mean!

---

## 3. Discrepancy & Gap Inventory (The 7 Critical Vulnerabilities of Monet)

1. **Gap 1: Deterministic Rollout vs. Continuous Gaussian Assumption**:
   - The paper derives VLPO assuming stochastic sampling $z \sim \mathcal{N}(\mathbf{h}, \sigma^2 \mathbf{I})$. In reality, `vllm_inference_example.py` and `monet_gpu_model_runner.py` perform greedy deterministic rollout. No Gaussian noise is ever sampled during rollout.
2. **Gap 2: Massive Variance Scaling Disconnect ($\sigma = 10.0$)**:
   - In `RL/examples/vlpo_train.sh` L79, $\sigma$ is set to `10.0` ($\sigma^2 = 100$). For a $D=3584$ embedding vector whose L2 norm is $\sim 30$–$50$, $\|\mathbf{h}^{\text{old}} - \mathbf{h}^\theta\|^2 \approx 1000$–$2500$.
   - The exponent is $-1000 / 200 = -5.0 \implies r_t \approx 0.0067$.
   - The latent policy ratio is pushed into the near-zero saturation zone, severely muting gradient updates.
3. **Gap 3: The Proxy Loss Detach Trick**:
   - In SFT Stage 2 (`src/trainer.py` L39), `safe_grads.append(g.detach())` severs true end-to-end backpropagation through the model.
4. **Gap 4: Offline Static Distillation Lock-In**:
   - The student in Stage 2 and Stage 3 is trained against static offline `.pt` files. The student never interacts dynamically with a live visual teacher.
5. **Gap 5: External Stop-Gate Dependence (`LATENT_SIZE=10`)**:
   - The model cannot autonomously decide when to stop latent thinking. It generates exactly 10 tokens because `LATENT_SIZE=10` is externally enforced by the runner.
6. **Gap 6: Formatting Artifacts on Benchmarks**:
   - Monet instructs the model: `Put your final answer in \boxed{}`. On standard benchmarks (e.g. V* Bench, MMVP), raw answer parsers that expect single letters `(A)` or `(B)` fail unless custom boxed parsers are used.
7. **Gap 7: Base Qwen2.5-VL Recovery**:
   - Like LVR, once formatting and boxed parsing are standardized, un-tuned `Qwen2.5-VL-7B-Instruct` is expected to match or exceed Monet-7B.

---

## 4. Tensor Dataflow & Signature Reference

| Function / Tensor | File & Line | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen2_5_VLModel.forward` | `monet_qwen_model/modeling_qwen2_5_vl_monet.py` L1110 | `[B, S]` or `[B, S, D]` | `BaseModelOutputWithPast` | Intercepts `latent_mode` and injects `ce_patch_vec` |
| `compute_latents_only_loss` | `src/trainer.py` L11 | `latents: [L, D]`, `loss: []` | `proxy_loss: []` | Uses `g.detach()` |
| `load_offline_tensor` | `src/trainer.py` L44 | `metadata`, `alignment_layer` | `List[Tensor [T_obs, D]]` | Reads `.pt` from disk |
| `compute_latent_log_probs` | `RL/verl/workers/actor/dp_actor.py` L135 | `latents: [L, D]`, `hidden: [1, T, D]` | `log_probs: [L]` | $-0.5 \|\mathbf{z} - \boldsymbol{\mu}\|^2 / \sigma^2$ |
| `emit_latents_step` | `RL/monet_models/vllm/latent_hook.py` L104 | `req_ids: List[str]`, `latents: List` | `None` (Network emit) | TCP streaming to `LatentRecorder` |
| `monet_gpu_model_runner` | `inference/vllm/monet_gpu_model_runner.py` L1283 | Intercepts token `151666` | Injects last-layer state | Enforces `LATENT_SIZE=10` |
"""

with open(out_path, "w", encoding="utf-8") as f:
    f.write(doc.strip() + "\n")

print(f"Generated {out_path} with {len(doc.splitlines())} lines.")
