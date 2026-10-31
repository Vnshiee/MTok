"""
Hooked Inference, Causal Intervention & Mechanistic Telemetry Engine
for `vincentleebang/LVR-7B` and `Qwen/Qwen2.5-VL-7B/3B-Instruct`.

Implements all zero-retraining experimental conditions with exact step-by-step
KV-cache, attention-mask, M-RoPE, and input-embedding control:

- Axis 1A (Inference-Time Causal Interventions on Prefix Copying, Exposure Bias & M-RoPE):
  * `A1A-BASE`: Unmodified default K=8 continuous latent rollout.
  * `A1A-MASK-ALL-IMG`: Mask attention from latent steps (k=1..K) to all prefix `<|image_pad|>` tokens.
  * `A1A-MASK-ANS-TO-LVR`: Mask attention from `<|lvr_end|>` & `<answer>` tokens to the K latent vectors.
  * `A1A-MASK-ANS-TO-IMG`: Mask attention from `<|lvr_end|>` & `<answer>` tokens to prefix `<|image_pad|>` tokens
    (forcing answer generation to read visual information exclusively through the K latent states).
  * `A1A-ORACLE-ROI`: Teacher-force ground-truth bounding-box ViT patch embeddings `V_T[I_ROI]` during
    latent steps (matching Stage 1 SFT teacher-forcing to isolate exposure bias `H_1b`).
  * `A1A-ORACLE-BKG`: Teacher-force non-ROI background ViT patch embeddings `V_T[I_bkg]` during latent steps.
  * `A1A-2D-MROPE`: Restore 2D spatial M-RoPE coordinates `(t_cur, y_patch, x_patch)` of the nearest
    prefix visual patch during latent steps instead of 1D text M-RoPE `(t_cur, t_cur, t_cur)` (`H_1c`).

- Axis 3 (Step Quota Scaling & Decoding Strategy Comparison):
  * `A3A-STEPS-{K}` for K in {0, 1, 2, 4, 8, 16, 32, 64}.
  * `A3B-STEPS-8`: Standard fixed-step counter exit (`_lvr_deocding_by_steps`).
  * `A3B-FORCED-END`: Explicitly inject token 151668 (`<|lvr_end|>`) at step K+1 when quota expires.
  * `A3B-VANILLA-LOGIT`: Autonomous exit when `argmax(logits) == 151668` (`_lvr_deocding`).
  * `A3B-LATENT-END`: Distance threshold exit against `lvr_latent_end_emb` (`_lvr_deocding_with_latentend`).

- Axis 4 (9-Condition Latent Representation Substrate & Format-Locking Matrix, K=8):
  * `C1-CONT-DEFAULT`: Standard continuous hidden-state feedback `h_{k-1}^{(L)}`.
  * `C2-EMPTY-TAG`: Prefill `<|lvr_start|><|lvr_end|>` (`[151665, 151668]`) with K=0 latent steps.
  * `C3-DISCRETE-LVR`: Feed discrete `<|lvr|>` token embedding (`E_tok[151666]`) for K=8 steps.
  * `C4-MEAN-IMG-POOL`: Feed static mean prefix image patch embedding `mean(V_T)` for K=8 steps.
  * `C5-NN-QUANT-IMG`: Quantize each `h_{k-1}^{(L)}` to its nearest prefix ViT patch `V_T[j*]`.
  * `C6-ZERO-VEC`: Feed zero vector `0` in R^3584 for K=8 steps.
  * `C7-GAUSS-NOISE`: Feed norm-matched isotropic Gaussian noise `N(0, (||h||^2 / D) I_D)` for K=8 steps.
  * `C8-SHUFFLE-BATCH`: Transplant cached latent vectors `H_LVR^{(b')}` from a different image `b' != b`.
  * `C9-DIRECT-ANS`: Prefill `<answer>` immediately after user prompt, bypassing `<|lvr_start|>`.
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

REPO_PATH = "/home/devraj/MTok/external_models/lvr/repo"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from ablation_study.data_loader import bbox_norm_to_token_idxs
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLSdpaAttention,
    Qwen2_5_VLVisionSdpaAttention,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    repeat_kv,
)


def _flash_chunked_vision_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor = None,
) -> torch.Tensor:
    """
    Memory-safe exact replacement for `Qwen2_5_VLVisionSdpaAttention.forward`.
    Instead of allocating a dense `[1, seq_length, seq_length]` boolean mask (which OOMs
    with 14+ GiB on high-resolution V* Bench images), evaluates `F.scaled_dot_product_attention`
    with `attn_mask=None` on each contiguous block `[cu_seqlens[i-1] : cu_seqlens[i]]`,
    dispatching directly to PyTorch's fused FlashAttention-2 kernel with 0 mask memory.
    """
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    q_t = q.transpose(0, 1).unsqueeze(0)  # [1, heads, seq_len, head_dim]
    k_t = k.transpose(0, 1).unsqueeze(0)
    v_t = v.transpose(0, 1).unsqueeze(0)

    chunks = []
    for i in range(1, len(cu_seqlens)):
        s = int(cu_seqlens[i - 1].item())
        e = int(cu_seqlens[i].item())
        chunks.append(
            F.scaled_dot_product_attention(
                q_t[:, :, s:e, :],
                k_t[:, :, s:e, :],
                v_t[:, :, s:e, :],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        )
    attn_output = torch.cat(chunks, dim=2).squeeze(0).transpose(0, 1).reshape(seq_length, -1)
    return self.proj(attn_output)


def _non_invasive_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """
    Non-invasive replacement for `Qwen2_5_VLSdpaAttention.forward`:
    Always computes `attn_output` via `F.scaled_dot_product_attention` so that enabling
    `output_attentions=True` (for attention telemetry) never falls back to low-precision
    eager bfloat16 matmul and stays 100% bit-identical to `output_attentions=False`.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # If no positions are masked (all 0.0), pass None so SDPA uses the exact same kernel as output_attentions=False
        if not torch.any(causal_mask != 0):
            causal_mask = None

    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_weights = None
    if output_attentions:
        # Compute exact read-only attention weights in float32 without altering attn_output
        logits = torch.matmul(query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            logits = logits + causal_mask.to(torch.float32)
        attn_weights = F.softmax(logits, dim=-1, dtype=torch.float32)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


Qwen2_5_VLVisionSdpaAttention.forward = _flash_chunked_vision_sdpa_forward
Qwen2_5_VLSdpaAttention.forward = _non_invasive_sdpa_forward

IMAGE_PAD_TOKEN_ID = 151655
LVR_START_TOKEN_ID = 151665
LVR_TOKEN_ID = 151666
LVR_LATENT_END_TOKEN_ID = 151667
LVR_END_TOKEN_ID = 151668
IM_END_TOKEN_ID = 151645
ENDOFTEXT_TOKEN_ID = 151643


def compute_effective_rank(states: torch.Tensor) -> float:
    """
    Computes the Effective Rank `erank(H)` (Section 5.4) of a trajectory matrix `H` in R^{K x D}:
      p_i = sigma_i / sum_j sigma_j
      erank(H) = exp(-sum_i p_i ln p_i)
    """
    if states is None or states.shape[0] <= 1:
        return 1.0
    mat = states.detach().to(torch.float32)
    if mat.shape[0] >= 2:
        mat_centered = mat - mat.mean(dim=0, keepdim=True)
        if torch.norm(mat_centered) > 1e-6:
            mat = mat_centered
    s = torch.linalg.svdvals(mat)
    s = s[s > 1e-9]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p)).item()
    return float(math.exp(entropy))


def create_qwen_messages(img_input: Any, text: str) -> List[Dict[str, Any]]:
    if not isinstance(img_input, list):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_input},
                    {"type": "text", "text": text},
                ],
            }
        ]
    vision_content = [{"type": "image", "image": ip} for ip in img_input]
    vision_content.append({"type": "text", "text": text})
    return [{"role": "user", "content": vision_content}]


@torch.no_grad()
def run_lvr_intervention_single(
    model: torch.nn.Module,
    processor: Any,
    img_input: Any,
    prompt_text: str,
    condition: str = "A1A-BASE",
    lvr_steps: int = 8,
    max_new_tokens: int = 128,
    bboxes_norm: Optional[List[Tuple[float, float, float, float]]] = None,
    donor_latent_states: Optional[torch.Tensor] = None,
    collect_attention_telemetry: bool = False,
    seed: int = 42,
    latent_end_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Executes a single deterministic greedy forward/decoding run on `QwenWithLVR` under the specified
    intervention `condition`, returning the generated text, token IDs, injected latent states, and
    mechanistic telemetry metrics (Section 5).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    messages = create_qwen_messages(img_input, prompt_text)
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Handle explicit prefill conditions (`C2-EMPTY-TAG`, `C9-DIRECT-ANS`, or `A3A-STEPS-0` prefill)
    if condition == "C2-EMPTY-TAG":
        chat_text = chat_text + "<|lvr_start|><|lvr_end|>"
        effective_lvr_steps = 0
    elif condition == "C9-DIRECT-ANS":
        chat_text = chat_text + "<answer>"
        effective_lvr_steps = 0
    else:
        effective_lvr_steps = lvr_steps

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)

    batch_size, prompt_len = input_ids.shape
    assert batch_size == 1, "Single-sample intervention engine requires batch_size=1"

    # 1. Extract exact prefix visual patch embeddings V_T in R^{N_vis x D} and their sequence indices
    vis_positions = torch.where(input_ids[0] == IMAGE_PAD_TOKEN_ID)[0]
    n_vis = int(vis_positions.numel())

    if pixel_values is not None and n_vis > 0:
        vis_feats_list = model.model.get_image_features(pixel_values, image_grid_thw)
        v_t = torch.cat(vis_feats_list, dim=0).to(device=device, dtype=torch.float32)  # [N_vis, D]
        v_t_norm = F.normalize(v_t, p=2, dim=-1)
        mean_v_t = v_t.mean(dim=0, keepdim=True).to(dtype=dtype)  # [1, D]
    else:
        v_t = None
        v_t_norm = None
        mean_v_t = None

    # 2. Identify Ground-Truth ROI vs. Background patch indices (when bboxes_norm is available)
    roi_idxs: List[int] = []
    bkg_idxs: List[int] = []
    if bboxes_norm and image_grid_thw is not None and n_vis > 0:
        thw_tuple = tuple(int(x) for x in image_grid_thw[0].tolist())
        roi_idxs = [i for i in bbox_norm_to_token_idxs(bboxes_norm, thw_tuple) if 0 <= i < n_vis]
        roi_set = set(roi_idxs)
        bkg_idxs = [i for i in range(n_vis) if i not in roi_set]

    # 3. Compute prefill 3D M-RoPE position_ids so we know exact (t, y, x) of every visual patch
    prefill_pos_ids, rope_deltas = model.model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=attention_mask,
    )
    vis_pos_ids_3d = prefill_pos_ids[:, 0, vis_positions] if n_vis > 0 else None  # [3, N_vis]

    # 4. Initialize KV cache & loop state
    from transformers.cache_utils import DynamicCache

    past_key_values = DynamicCache()
    cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

    lvr_mode_switch = torch.zeros(1, dtype=torch.bool, device=device)
    lvr_remaining_steps = torch.tensor([effective_lvr_steps], dtype=torch.long, device=device)
    last_position_hidden_state: Optional[torch.Tensor] = None

    # Track sequence positions of latent steps for `A1A-MASK-ANS-TO-LVR`
    lvr_kv_positions: List[int] = []
    entered_lvr_once = False
    exited_lvr = False

    # Telemetry accumulators
    raw_output_latent_states: List[torch.Tensor] = []  # h_{k-1}^{(L)} produced by transformer
    injected_latent_states: List[torch.Tensor] = []  # actual vector injected into inputs_embeds
    cs_max_per_step: List[float] = []
    cs_roi_per_step: List[float] = []
    cs_bkg_per_step: List[float] = []
    roi_recall_at_1: List[float] = []
    roi_recall_at_5: List[float] = []
    latent_norms: List[float] = []
    latent_attn_to_roi: List[float] = []
    latent_attn_to_bkg: List[float] = []
    latent_attn_to_vis: List[float] = []
    ans_attn_to_lvr: List[float] = []
    ans_attn_to_vis: List[float] = []

    gen_rng = torch.Generator(device=device)
    gen_rng.manual_seed(seed)

    discrete_lvr_embed = (
        model.model.get_input_embeddings()(
            torch.tensor([[LVR_TOKEN_ID]], dtype=torch.long, device=device)
        )[:, 0, :]
        .detach()
        .clone()
    )  # [1, D]

    is_prefill = True
    latent_step_idx = 0

    for step_idx in range(max_new_tokens):
        cur_seq_len = input_ids.shape[1]

        # Build step-specific attention_mask (cloning so we only mask on the targeted steps)
        step_attention_mask = attention_mask.clone()
        if lvr_mode_switch.item():
            # Inside latent phase (steps k = 1 .. K)
            if condition == "A1A-MASK-ALL-IMG" and n_vis > 0:
                step_attention_mask[:, vis_positions] = 0
        elif exited_lvr:
            # Post-latent phase (`<|lvr_end|>` and `<answer>` tokens)
            if condition == "A1A-MASK-ANS-TO-LVR" and len(lvr_kv_positions) > 0:
                step_attention_mask[:, lvr_kv_positions] = 0
            elif condition == "A1A-MASK-ANS-TO-IMG" and n_vis > 0:
                step_attention_mask[:, vis_positions] = 0

        # Prepare step-specific position_ids (default None lets monkey_patch_forward_lvr compute 1D M-RoPE)
        step_position_ids = None

        # Determine what `last_position_hidden_state` is injected if `lvr_mode_switch` is True
        injected_vec: Optional[torch.Tensor] = None
        if lvr_mode_switch.item() and last_position_hidden_state is not None:
            raw_h = last_position_hidden_state.detach()  # [1, D]
            raw_output_latent_states.append(raw_h[0].to(torch.float32).cpu())
            h_f32 = raw_h.to(torch.float32)
            h_norm_val = float(torch.norm(h_f32, p=2).item())
            latent_norms.append(h_norm_val)

            # Compute cosine similarity telemetry against prefix ViT patches V_T
            nearest_vis_idx = 0
            if v_t_norm is not None and n_vis > 0:
                h_unit = F.normalize(h_f32, p=2, dim=-1)  # [1, D]
                sims = torch.matmul(h_unit, v_t_norm.T).squeeze(0)  # [N_vis]
                cs_max_per_step.append(float(sims.max().item()))
                nearest_vis_idx = int( torch.argmax(sims).item() )

                if len(roi_idxs) > 0:
                    cs_roi_per_step.append(float(sims[roi_idxs].max().item()))
                    top1_idx = nearest_vis_idx
                    top5_idxs = torch.topk(sims, k=min(5, n_vis)).indices.tolist()
                    roi_set = set(roi_idxs)
                    roi_recall_at_1.append(1.0 if top1_idx in roi_set else 0.0)
                    roi_recall_at_5.append(1.0 if any(ix in roi_set for ix in top5_idxs) else 0.0)
                if len(bkg_idxs) > 0:
                    cs_bkg_per_step.append(float(sims[bkg_idxs].max().item()))

            # Select injected vector according to experimental condition
            if condition in (
                "A1A-BASE",
                "A1A-MASK-ALL-IMG",
                "A1A-MASK-ANS-TO-LVR",
                "A1A-MASK-ANS-TO-IMG",
                "A3B-STEPS-8",
                "A3B-FORCED-END",
                "A3B-VANILLA-LOGIT",
                "A3B-LATENT-END",
                "C1-CONT-DEFAULT",
            ) or condition.startswith("A3A-STEPS-"):
                injected_vec = raw_h

            elif condition == "A1A-2D-MROPE":
                injected_vec = raw_h
                if vis_pos_ids_3d is not None and n_vis > 0:
                    # Temporal coordinate stays monotonic; (y, x) restored from nearest ViT patch
                    delta = (cache_position[0] + model.model.rope_deltas).to(device=device, dtype=torch.long)
                    t_coord = delta.view(1, 1)
                    y_coord = vis_pos_ids_3d[1, nearest_vis_idx].view(1, 1)
                    x_coord = vis_pos_ids_3d[2, nearest_vis_idx].view(1, 1)
                    step_position_ids = torch.stack([t_coord, y_coord, x_coord], dim=0)  # [3, 1, 1]

            elif condition == "A1A-ORACLE-ROI":
                if v_t is not None and len(roi_idxs) > 0:
                    pick_idx = roi_idxs[latent_step_idx % len(roi_idxs)]
                    injected_vec = v_t[pick_idx : pick_idx + 1].to(dtype=dtype)
                elif v_t is not None and n_vis > 0:
                    injected_vec = v_t[nearest_vis_idx : nearest_vis_idx + 1].to(dtype=dtype)
                else:
                    injected_vec = raw_h

            elif condition == "A1A-ORACLE-BKG":
                if v_t is not None and len(bkg_idxs) > 0:
                    # Deterministic spread across background patches
                    stride = max(1, len(bkg_idxs) // max(1, effective_lvr_steps))
                    pick_idx = bkg_idxs[(latent_step_idx * stride) % len(bkg_idxs)]
                    injected_vec = v_t[pick_idx : pick_idx + 1].to(dtype=dtype)
                else:
                    injected_vec = raw_h

            elif condition == "C3-DISCRETE-LVR":
                injected_vec = discrete_lvr_embed

            elif condition == "C4-MEAN-IMG-POOL":
                injected_vec = mean_v_t if mean_v_t is not None else raw_h

            elif condition == "C5-NN-QUANT-IMG":
                if v_t is not None and n_vis > 0:
                    injected_vec = v_t[nearest_vis_idx : nearest_vis_idx + 1].to(dtype=dtype)
                else:
                    injected_vec = raw_h

            elif condition == "C6-ZERO-VEC":
                injected_vec = torch.zeros_like(raw_h)

            elif condition == "C7-GAUSS-NOISE":
                d_dim = raw_h.shape[-1]
                std = h_norm_val / math.sqrt(float(d_dim))
                noise = torch.randn(raw_h.shape, generator=gen_rng, device=device, dtype=torch.float32) * std
                injected_vec = noise.to(dtype=dtype)

            elif condition == "C8-SHUFFLE-BATCH":
                if donor_latent_states is not None and donor_latent_states.shape[0] > 0:
                    d_idx = min(latent_step_idx, donor_latent_states.shape[0] - 1)
                    injected_vec = donor_latent_states[d_idx : d_idx + 1].to(device=device, dtype=dtype)
                else:
                    injected_vec = torch.zeros_like(raw_h)

            else:
                raise ValueError(f"Unknown intervention condition: {condition}")

            injected_latent_states.append(injected_vec[0].detach().to(torch.float32).cpu())
            lvr_kv_positions.append(cur_seq_len - 1)
            latent_step_idx += 1

        # Prepare model inputs for this decoding step
        step_output_attentions = bool(collect_attention_telemetry and not is_prefill)
        if is_prefill:
            step_input_ids = input_ids
            step_pixel_values = pixel_values
            step_image_grid_thw = image_grid_thw
            is_prefill = False
        else:
            step_input_ids = input_ids[:, -1:]
            step_pixel_values = None
            step_image_grid_thw = image_grid_thw

        outputs = model(
            input_ids=step_input_ids,
            attention_mask=step_attention_mask,
            position_ids=step_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            pixel_values=step_pixel_values,
            image_grid_thw=step_image_grid_thw,
            cache_position=cache_position,
            lvr_mode_switch=lvr_mode_switch,
            last_position_hidden_state=injected_vec,
            output_attentions=step_output_attentions,
            return_dict=True,
        )

        # Record attention telemetry if enabled (`step_output_attentions=True`)
        if step_output_attentions and outputs.attentions is not None:
            # outputs.attentions is a tuple of L tensors, each [1, H_Q, q_len, kv_len]
            last_query_attns = torch.stack(
                [layer_attn[0, :, -1, :].to(torch.float32) for layer_attn in outputs.attentions],
                dim=0,
            )  # [L, H_Q, kv_len]
            mean_attn = last_query_attns.mean(dim=(0, 1))  # [kv_len]
            if lvr_mode_switch.item():
                if n_vis > 0:
                    latent_attn_to_vis.append(float(mean_attn[vis_positions].sum().item()))
                if len(roi_idxs) > 0:
                    roi_kv = vis_positions[roi_idxs]
                    latent_attn_to_roi.append(float(mean_attn[roi_kv].sum().item()))
                if len(bkg_idxs) > 0:
                    bkg_kv = vis_positions[bkg_idxs]
                    latent_attn_to_bkg.append(float(mean_attn[bkg_kv].sum().item()))
            elif exited_lvr:
                if len(lvr_kv_positions) > 0:
                    ans_attn_to_lvr.append(float(mean_attn[lvr_kv_positions].sum().item()))
                if n_vis > 0:
                    ans_attn_to_vis.append(float(mean_attn[vis_positions].sum().item()))

        next_token_logits = outputs.logits[:, -1, :].to(torch.float32)
        next_tokens = torch.argmax(next_token_logits, dim=-1)  # Greedy argmax

        # Update LVR mode switch according to decoding strategy
        last_tokens = input_ids[:, -1]
        lvr_start_switch = last_tokens == LVR_START_TOKEN_ID

        if condition == "A3B-VANILLA-LOGIT":
            # `_lvr_deocding` (repo/src/model/qwen_lvr_model.py L498-L503):
            # Exits LVR mode as soon as `next_tokens == 151668` (<|lvr_end|>)
            lvr_end_switch = next_tokens == LVR_END_TOKEN_ID
            was_in_lvr = bool(lvr_mode_switch.item())
            lvr_mode_switch = (lvr_mode_switch | lvr_start_switch) & (~lvr_end_switch)
            if lvr_start_switch.item():
                entered_lvr_once = True
            if was_in_lvr and not lvr_mode_switch.item():
                exited_lvr = True

        elif condition == "A3B-LATENT-END":
            # `_lvr_deocding_with_latentend` (repo/src/model/qwen_lvr_model.py L763-L790):
            # Compares output hidden state against `lvr_latent_end_emb` (token 151667)
            was_in_lvr = bool(lvr_mode_switch.item())
            new_mode_switch = lvr_mode_switch | lvr_start_switch
            if lvr_start_switch.item():
                entered_lvr_once = True
            cur_h = outputs.last_position_hidden_state
            latent_end_emb = model.model.get_input_embeddings()(
                torch.tensor([LVR_LATENT_END_TOKEN_ID], device=device)
            )
            dist = torch.norm(cur_h.to(torch.float32) - latent_end_emb.to(torch.float32), p=2, dim=-1)
            reached_end = dist < latent_end_threshold
            if was_in_lvr and reached_end.item():
                lvr_mode_switch = torch.zeros_like(lvr_mode_switch)
                next_tokens = torch.tensor([LVR_END_TOKEN_ID], device=device, dtype=torch.long)
                exited_lvr = True
            else:
                lvr_mode_switch = new_mode_switch

        else:
            # Standard quota-based `_lvr_deocding_by_steps` (repo/src/model/qwen_lvr_model.py L1063-L1086)
            was_in_lvr = bool(lvr_mode_switch.item())
            new_mode_switch = lvr_mode_switch | lvr_start_switch
            just_entered = (~lvr_mode_switch) & new_mode_switch
            if just_entered.item():
                entered_lvr_once = True
                lvr_remaining_steps = torch.tensor([effective_lvr_steps], dtype=torch.long, device=device)
            lvr_remaining_steps = lvr_remaining_steps - lvr_mode_switch.long()
            lvr_mode_switch = new_mode_switch & (lvr_remaining_steps > 0)
            if was_in_lvr and not lvr_mode_switch.item():
                exited_lvr = True
                if condition == "A3B-FORCED-END":
                    next_tokens = torch.tensor([LVR_END_TOKEN_ID], device=device, dtype=torch.long)
            elif just_entered.item() and effective_lvr_steps == 0:
                exited_lvr = True

        last_position_hidden_state = outputs.last_position_hidden_state

        # Append next token and advance cache_position & attention_mask
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=-1,
        )
        cache_position = torch.tensor([input_ids.shape[1] - 1], dtype=torch.long, device=device)

        # Check stopping condition (only stop when not inside LVR mode)
        tok_id = int(next_tokens.item())
        if not lvr_mode_switch.item() and tok_id in (IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID):
            break

    gen_ids = input_ids[0, prompt_len:]
    output_text = processor.decode(
        gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if condition == "C2-EMPTY-TAG":
        output_text = "<|lvr_start|><|lvr_end|>" + output_text
    elif condition == "C9-DIRECT-ANS":
        output_text = "<answer>" + output_text

    # Compute trajectory-level telemetry (effective rank & adjacent cosine similarity)
    if len(raw_output_latent_states) > 0:
        h_stack = torch.stack(raw_output_latent_states, dim=0)  # [K_actual, D]
        erank_val = compute_effective_rank(h_stack)
        adj_cos_list = []
        for idx in range(len(raw_output_latent_states) - 1):
            c_val = F.cosine_similarity(
                raw_output_latent_states[idx].unsqueeze(0),
                raw_output_latent_states[idx + 1].unsqueeze(0),
            ).item()
            adj_cos_list.append(float(c_val))
    else:
        h_stack = torch.empty((0, 3584), dtype=torch.float32)
        erank_val = 0.0
        adj_cos_list = []

    telemetry = {
        "actual_latent_steps": len(injected_latent_states),
        "entered_lvr_once": entered_lvr_once,
        "exited_lvr": exited_lvr,
        "n_vis_tokens": n_vis,
        "n_roi_tokens": len(roi_idxs),
        "n_bkg_tokens": len(bkg_idxs),
        "cs_max_mean": round(float(np.mean(cs_max_per_step)), 4) if cs_max_per_step else None,
        "cs_max_per_step": [round(x, 4) for x in cs_max_per_step],
        "cs_roi_mean": round(float(np.mean(cs_roi_per_step)), 4) if cs_roi_per_step else None,
        "cs_bkg_mean": round(float(np.mean(cs_bkg_per_step)), 4) if cs_bkg_per_step else None,
        "roi_recall_at_1_mean": round(float(np.mean(roi_recall_at_1)), 4) if roi_recall_at_1 else None,
        "roi_recall_at_5_mean": round(float(np.mean(roi_recall_at_5)), 4) if roi_recall_at_5 else None,
        "latent_norm_mean": round(float(np.mean(latent_norms)), 4) if latent_norms else None,
        "latent_norm_per_step": [round(x, 4) for x in latent_norms],
        "effective_rank": round(erank_val, 4),
        "adj_cosine_mean": round(float(np.mean(adj_cos_list)), 4) if adj_cos_list else None,
        "adj_cosine_per_step": [round(x, 4) for x in adj_cos_list],
        "latent_attn_to_vis_mean": round(float(np.mean(latent_attn_to_vis)), 6) if latent_attn_to_vis else None,
        "latent_attn_to_roi_mean": round(float(np.mean(latent_attn_to_roi)), 6) if latent_attn_to_roi else None,
        "latent_attn_to_bkg_mean": round(float(np.mean(latent_attn_to_bkg)), 6) if latent_attn_to_bkg else None,
        "ans_attn_to_lvr_mean": round(float(np.mean(ans_attn_to_lvr)), 6) if ans_attn_to_lvr else None,
        "ans_attn_to_vis_mean": round(float(np.mean(ans_attn_to_vis)), 6) if ans_attn_to_vis else None,
    }

    return {
        "condition": condition,
        "requested_lvr_steps": lvr_steps,
        "output_text": output_text,
        "generated_token_ids": gen_ids.cpu().tolist(),
        "raw_latent_states": h_stack,  # CPU float32 tensor [K_actual, D] for C8 transplant / drift analysis
        "telemetry": telemetry,
    }
