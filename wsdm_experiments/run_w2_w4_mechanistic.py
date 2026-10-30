#!/usr/bin/env python3
"""
WSDM Experiments W2 & W4 (Fast Precomputed-ViT Implementation):
1. W2: Causal 4D Attention Mask Knockout on all N=700 BLINK items:
   - W2_Intact: Standard SDPA forward pass (verified against E2 T1 V0)
   - W2_KO_LatentToAns_All: Block post-latent tokens (</think>\n<answer) from attending to the 20 <|latent_pad|> tokens across ALL 28 layers (0..27).
   - W2_KO_LatentToAns_Early: Block post-latent tokens from attending to latents in Early layers (0..9).
   - W2_KO_LatentToAns_Mid: Block post-latent tokens from attending to latents in Mid layers (10..18).
   - W2_KO_LatentToAns_Late: Block post-latent tokens from attending to latents in Late layers (19..27).
   - W2_KO_ImgToLatent_All: Block the 20 <|latent_pad|> tokens from attending to ANY visual <|image_pad|> tokens across ALL 28 layers (0..27) (Visual Starvation of Latents).
   - W2_KO_Total_Sever: Block BOTH Image->Latents AND Latents->PostLatent across ALL 28 layers (Complete Causal Isolation of Latents).

2. W4a: Counterfactual Cross-Item Latent Activation Patching on all 2-way BLINK tasks (N=550).
3. W4b: Fixed Phase E5 — Pre-RMSNorm Layers 0..27 Anisotropy-Corrected Latent Geometry & Linear Probing (N=143 blink_sprel).
"""

import argparse
import json
import math
import os
import sys
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "mull", "models")
LMMS_DIR = os.path.join(PROJECT_DIR, "mull", "lmms-eval")
E2_DIR = os.path.join(PROJECT_DIR, "e2_bias")
for p in (MODELS_DIR, LMMS_DIR, E2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from score_letters import (
    MAX_PIXELS, MIN_PIXELS, ANSWER_PREFIX_IDS, SEED,
    TASKS, EXPECTED_N, doc_to_text, doc_images, latents
)
import importlib

OUT_DIR = os.path.join(PROJECT_DIR, "wsdm_experiments", "outputs_w2_w4")


class MechanisticAnalyzer:
    def __init__(self, pretrained="array/Qwen2.5-VL-Mull"):
        self.pretrained = pretrained
        Q = importlib.import_module("mmlatentdiscrete_qwen_vl").Qwen2_5_VLForConditionalGeneration
        print(f"Loading {pretrained} with SDPA for exact 4D attention mask & residual hooks...")
        self.model = Q.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval()
        # Use FlashAttention-2 varlen in the frozen Vision Encoder so multi-image tasks (e.g. blink_iqtest)
        # never allocate dense O(N^2) vision attention masks (which tried to allocate 26.71 GiB VRAM),
        # while keeping SDPA in the 28 language model layers for exact 4D causal attention knockout masks.
        vis_fa2_cls = importlib.import_module("mmlatentdiscrete_qwen_vl").Qwen2_5_VLVisionFlashAttention2
        for blk in self.model.visual.blocks:
            blk.attn.__class__ = vis_fa2_cls
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
        tk = self.processor.tokenizer
        n_added = 0
        for t in ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>", "<|latent_image|>"]:
            n_added += tk.add_tokens(t, special_tokens=True)
        cfg = self.model.config
        cfg.latent_token_id = int(tk("<|latent_pad|>")["input_ids"][0])
        cfg.latent_start_id = int(tk("<|latent_start|>")["input_ids"][0])
        cfg.latent_end_id = int(tk("<|latent_end|>")["input_ids"][0])
        cfg.imagelatent_token_id = int(tk("<|latent_image|>")["input_ids"][0])
        self.image_token_id = int(tk.convert_tokens_to_ids("<|image_pad|>"))
        self.latent_token_id = cfg.latent_token_id
        if n_added > 0:
            self.model.resize_token_embeddings(len(tk))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.letter_ids = {}
        self.id_to_letter = {}
        for L in "ABCD":
            ids = self.tokenizer.encode(f"<answer>{L}</answer>", add_special_tokens=False)
            assert len(ids) == 6 and ids[:2] == ANSWER_PREFIX_IDS
            self.letter_ids[L] = ids[2]
            self.id_to_letter[ids[2]] = L
        self.num_layers = len(self.model.model.layers)
        print(f"Loaded {pretrained}: {self.num_layers} layers, latent_id={self.latent_token_id}, image_id={self.image_token_id}")

    @torch.no_grad()
    def prepare_precomputed_state(self, images, user_text, prefill):
        """
        Tokenizes inputs and runs the Vision Encoder ONCE to produce:
          - input_ids (1, S)
          - inputs_embeds (1, S, D)
          - position_ids (3, 1, S)
          - attention_mask (1, S)
        """
        messages = [
            {"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if text.endswith("<|im_end|>\n"):
            text = text[:-len("<|im_end|>\n")]
        elif text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        extra = torch.tensor([ANSWER_PREFIX_IDS], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(extra)], dim=1)
        inputs = inputs.to("cuda:0")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"].type(self.model.visual.dtype)
        image_grid_thw = inputs["image_grid_thw"]

        inputs_embeds = self.model.model.embed_tokens(input_ids)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)

        position_ids, _ = self.model.get_rope_index(input_ids, image_grid_thw, None, None, attention_mask)
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @torch.no_grad()
    def swap_span_token_embeds(self, state, old_token_id, new_token_id):
        """Replaces the 20 span token embeddings in `inputs_embeds` without re-running ViT."""
        input_ids = state["input_ids"].clone()
        span_mask = (input_ids == old_token_id)
        input_ids[span_mask] = new_token_id
        inputs_embeds = state["inputs_embeds"].clone()
        new_emb = self.model.model.embed_tokens(torch.tensor([new_token_id], device="cuda:0"))[0]
        inputs_embeds[0, span_mask[0], :] = new_emb
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": state["position_ids"],
            "attention_mask": state["attention_mask"],
        }

    @torch.no_grad()
    def forward_from_state(self, state):
        out = self.model.model(
            input_ids=None,
            position_ids=state["position_ids"],
            attention_mask=state["attention_mask"],
            inputs_embeds=state["inputs_embeds"],
            use_cache=False,
            return_dict=True,
        )
        logits_last = self.model.lm_head(out[0][:, -1:, :])[0, 0]
        return logits_last

    def extract_logits_dict(self, logits_last, letters):
        logits = logits_last.float()
        logp = torch.log_softmax(logits, dim=-1)
        cand = {L: float(logp[self.letter_ids[L]]) for L in letters}
        cand_logits = {L: float(logits[self.letter_ids[L]]) for L in letters}
        top1 = int(torch.argmax(logits))
        top1_letter = self.id_to_letter.get(top1, None)
        cand_argmax = max(cand, key=cand.get)
        tf_pred = top1_letter if (top1_letter is not None and top1_letter in letters) else cand_argmax
        return dict(logp=cand, logit=cand_logits, tf_pred=tf_pred)

    @torch.no_grad()
    def run_w2_and_capture_intact(self, state, letters, capture_full_geometry=False):
        """
        Runs all 7 W2 conditions using precomputed `state` (ViT already done).
        Simultaneously captures the 20 latent vectors across all 28 layers during `W2_Intact`
        (and full layer geometry if `capture_full_geometry=True`).
        """
        ids = state["input_ids"][0]
        seq_len = ids.shape[0]
        latent_mask = (ids == self.latent_token_id)
        latent_positions = latent_mask.nonzero(as_tuple=True)[0]
        image_positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
        assert len(latent_positions) == 20, f"Expected 20 latents, got {len(latent_positions)}"
        last_latent_pos = int(latent_positions[-1].item())
        post_latent_start = last_latent_pos + 1

        min_val = torch.finfo(torch.bfloat16).min

        def build_custom_mask(mtype):
            if mtype is None:
                return None
            m = torch.triu(torch.full((seq_len, seq_len), min_val, device="cuda:0", dtype=torch.bfloat16), diagonal=1)
            m = m.unsqueeze(0).unsqueeze(0)
            if mtype in ("lat_to_ans", "total_sever"):
                m[:, :, post_latent_start:, latent_positions] = min_val
            if mtype in ("img_to_lat", "total_sever"):
                m[:, :, latent_positions[:, None], image_positions[None, :]] = min_val
            return m

        configs = {
            "W2_Intact": (set(), None),
            "W2_KO_LatentToAns_All": (set(range(0, 28)), "lat_to_ans"),
            "W2_KO_LatentToAns_Early": (set(range(0, 10)), "lat_to_ans"),
            "W2_KO_LatentToAns_Mid": (set(range(10, 19)), "lat_to_ans"),
            "W2_KO_LatentToAns_Late": (set(range(19, 28)), "lat_to_ans"),
            "W2_KO_ImgToLatent_All": (set(range(0, 28)), "img_to_lat"),
            "W2_KO_Total_Sever": (set(range(0, 28)), "total_sever"),
        }

        out_res = {}
        captured_latents = {}
        full_layer_inputs = {}

        for cname, (active_layers, mask_type) in configs.items():
            hooks = []
            custom_mask = build_custom_mask(mask_type)
            if cname == "W2_Intact":
                # Capture latent states at every layer (and full states if needed for W4b)
                for l_idx in range(self.num_layers):
                    def make_cap_hook(idx):
                        def hook(module, args):
                            captured_latents[idx] = args[0][0, latent_mask, :].detach().cpu()
                            if capture_full_geometry:
                                full_layer_inputs[idx] = args[0][0].detach().float()
                        return hook
                    hooks.append(self.model.model.layers[l_idx].register_forward_pre_hook(make_cap_hook(l_idx)))
                if capture_full_geometry:
                    def final_pre_norm_hook(module, args):
                        full_layer_inputs[28] = args[0][0].detach().float()
                    hooks.append(self.model.model.norm.register_forward_pre_hook(final_pre_norm_hook))
            elif active_layers and custom_mask is not None:
                def make_pre_hook(m):
                    def hook(module, args, kwargs):
                        kwargs["attention_mask"] = m
                        return args, kwargs
                    return hook
                for l_idx in active_layers:
                    h = self.model.model.layers[l_idx].self_attn.register_forward_pre_hook(
                        make_pre_hook(custom_mask), with_kwargs=True
                    )
                    hooks.append(h)

            try:
                logits_last = self.forward_from_state(state)
                out_res[cname] = self.extract_logits_dict(logits_last, letters)
            finally:
                for h in hooks:
                    h.remove()
                del custom_mask

        geom_res = None
        probes_res = None
        if capture_full_geometry:
            geom_res, probes_res = self.compute_geometry_from_layer_inputs(state["input_ids"][0], full_layer_inputs, self.latent_token_id)
            del full_layer_inputs

        return out_res, captured_latents, geom_res, probes_res

    @torch.no_grad()
    def run_patched_latents(self, state, patched_states_cpu, letters="AB"):
        ids = state["input_ids"][0]
        latent_mask = (ids == self.latent_token_id)
        hooks = []
        for l_idx in range(self.num_layers):
            p_gpu = patched_states_cpu[l_idx].to("cuda:0", non_blocking=True)
            def make_patch_hook(vec):
                def hook(module, args):
                    h_in = args[0].clone()
                    h_in[0, latent_mask, :] = vec
                    return (h_in,) + args[1:]
                return hook
            hooks.append(self.model.model.layers[l_idx].register_forward_pre_hook(make_patch_hook(p_gpu)))
        try:
            logits_last = self.forward_from_state(state)
            res = self.extract_logits_dict(logits_last, letters)
        finally:
            for h in hooks:
                h.remove()
        return res

    @torch.no_grad()
    def extract_geometry_for_state(self, state, span_token_id):
        layer_inputs = {}
        hooks = []
        for l_idx in range(self.num_layers):
            def make_hook(idx):
                def hook(module, args):
                    layer_inputs[idx] = args[0][0].detach().float()
                return hook
            hooks.append(self.model.model.layers[l_idx].register_forward_pre_hook(make_hook(l_idx)))

        def final_pre_norm_hook(module, args):
            layer_inputs[28] = args[0][0].detach().float()
        hooks.append(self.model.model.norm.register_forward_pre_hook(final_pre_norm_hook))

        try:
            _ = self.forward_from_state(state)
        finally:
            for h in hooks:
                h.remove()

        return self.compute_geometry_from_layer_inputs(state["input_ids"][0], layer_inputs, span_token_id)

    @torch.no_grad()
    def compute_geometry_from_layer_inputs(self, ids, layer_inputs, span_token_id):
        span_mask = (ids == span_token_id)
        image_mask = (ids == self.image_token_id)
        img_pos = image_mask.nonzero(as_tuple=True)[0]
        last_img = int(img_pos[-1].item()) if len(img_pos) > 0 else 0
        span_pos = span_mask.nonzero(as_tuple=True)[0]
        first_span = int(span_pos[0].item())
        q_mask = torch.zeros_like(span_mask)
        q_mask[last_img + 2 : first_span - 5] = True

        metrics_per_layer = []
        n_span = int(span_mask.sum().item())
        eye_mask = ~torch.eye(n_span, dtype=torch.bool, device="cuda:0")

        for l_idx in range(29):
            H = layer_inputs[l_idx]
            h_span = H[span_mask]
            h_img = H[image_mask]
            h_q = H[q_mask]

            h_span_norm = F.normalize(h_span, dim=-1)
            raw_cos = (h_span_norm @ h_span_norm.T)[eye_mask].mean().item()

            seq_mean = H.mean(dim=0, keepdim=True)
            h_span_c = F.normalize(h_span - seq_mean, dim=-1)
            centered_cos = (h_span_c @ h_span_c.T)[eye_mask].mean().item()

            n_q = h_q.shape[0]
            if n_q > 1:
                q_eye = ~torch.eye(n_q, dtype=torch.bool, device="cuda:0")
                h_q_c = F.normalize(h_q - seq_mean, dim=-1)
                q_centered_cos = (h_q_c @ h_q_c.T)[q_eye].mean().item()
            else:
                q_centered_cos = 0.0

            h_span_zero = h_span - h_span.mean(dim=0, keepdim=True)
            s_vals = torch.linalg.svdvals(h_span_zero)
            eig_vals = (s_vals ** 2)
            eig_sum = eig_vals.sum().item()
            eff_rank = float((eig_sum ** 2) / ((eig_vals ** 2).sum().item() + 1e-12)) if eig_sum > 0 else 1.0

            img_center = F.normalize((h_img - seq_mean).mean(dim=0, keepdim=True), dim=-1)
            span_center = F.normalize((h_span - seq_mean).mean(dim=0, keepdim=True), dim=-1)
            vis_align = (span_center @ img_center.T).item()

            metrics_per_layer.append({
                "layer": l_idx,
                "raw_intra_cos": round(raw_cos, 4),
                "centered_intra_cos": round(centered_cos, 4),
                "q_centered_intra_cos": round(q_centered_cos, 4),
                "effective_rank": round(eff_rank, 4),
                "centered_vis_align": round(vis_align, 4),
            })

        probes = {
            "span_L14": layer_inputs[14][span_mask].mean(dim=0).cpu().numpy(),
            "span_L24": layer_inputs[24][span_mask].mean(dim=0).cpu().numpy(),
            "span_L28": layer_inputs[28][span_mask].mean(dim=0).cpu().numpy(),
            "ans_L28": layer_inputs[28][-1].cpu().numpy(),
        }
        return metrics_per_layer, probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out if args.limit is None else os.path.join(PROJECT_DIR, "wsdm_experiments", "scratch_smoke")
    os.makedirs(out_dir, exist_ok=True)

    analyzer = MechanisticAnalyzer("array/Qwen2.5-VL-Mull")
    prefill_20 = latents(20)
    dot_token_id = int(analyzer.tokenizer.encode(" .", add_special_tokens=False)[0])

    w2_records = []
    w4a_records = []
    w4b_lat_layers = []
    w4b_dot_layers = []
    probe_data = {"lat_L14": [], "lat_L24": [], "lat_L28": [], "ans_L28": [],
                  "dot_L14": [], "dot_L24": [], "dot_L28": [], "y": []}

    t_start = time.time()
    per_task_dir = os.path.join(out_dir, "per_task")
    os.makedirs(per_task_dir, exist_ok=True)

    for task, (cfg_name, letters, two_way) in TASKS.items():
        task_done_marker = os.path.join(per_task_dir, f"{task}.COMPLETE")
        task_w2_file = os.path.join(per_task_dir, f"{task}_w2.jsonl")
        task_w4a_file = os.path.join(per_task_dir, f"{task}_w4a.jsonl")
        task_w4b_file = os.path.join(per_task_dir, f"{task}_w4b.pt")

        if args.limit is None and os.path.exists(task_done_marker) and os.path.exists(task_w2_file):
            print(f"\n>>> [Resume] {task} already COMPLETE, loading from {per_task_dir}...")
            with open(task_w2_file) as f:
                w2_records.extend(json.loads(line) for line in f if line.strip())
            if two_way and os.path.exists(task_w4a_file):
                with open(task_w4a_file) as f:
                    w4a_records.extend(json.loads(line) for line in f if line.strip())
            if task == "blink_sprel" and os.path.exists(task_w4b_file):
                saved_w4b = torch.load(task_w4b_file, weights_only=False)
                w4b_lat_layers = saved_w4b["w4b_lat_layers"]
                w4b_dot_layers = saved_w4b["w4b_dot_layers"]
                probe_data = saved_w4b["probe_data"]
            continue

        ds = load_dataset("BLINK-Benchmark/BLINK", cfg_name, split="val")
        n_docs = len(ds) if args.limit is None else min(args.limit, len(ds))
        print(f"\n>>> Running W2 & W4 on {task} (n={n_docs})...")

        e2_v0_path = os.path.join(E2_DIR, "outputs", "T1", "V0.jsonl")
        e2_ref = {}
        if os.path.exists(e2_v0_path):
            for ln in open(e2_v0_path):
                r = json.loads(ln)
                if r["task"] == task:
                    e2_ref[r["doc_id"]] = r

        task_items_cache = []
        task_w2_records = []
        task_w4a_records = []

        for doc_id in range(n_docs):
            doc = ds[doc_id]
            gold = doc["answer"].replace("(", "").replace(")", "")
            user_text = doc_to_text(doc, "latents")
            imgs = doc_images(doc)

            # Single ViT forward pass per item!
            state = analyzer.prepare_precomputed_state(imgs, user_text, prefill_20)
            is_sprel = (task == "blink_sprel")
            w2_out, lat_states_cpu, lat_geom, lat_probes = analyzer.run_w2_and_capture_intact(
                state, letters, capture_full_geometry=is_sprel
            )

            rec_w2 = {
                "key": f"{task}|{doc_id}",
                "task": task,
                "doc_id": doc_id,
                "gold": gold,
                "letters": letters,
                "e2_t1_pred": e2_ref[doc_id]["tf_pred"] if doc_id in e2_ref else None,
                "conditions": w2_out,
            }
            w2_records.append(rec_w2)
            task_w2_records.append(rec_w2)

            if is_sprel:
                state_dots = analyzer.swap_span_token_embeds(state, analyzer.latent_token_id, dot_token_id)
                dot_geom, dot_probes = analyzer.extract_geometry_for_state(state_dots, dot_token_id)
                w4b_lat_layers.append(lat_geom)
                w4b_dot_layers.append(dot_geom)
                probe_data["lat_L14"].append(lat_probes["span_L14"])
                probe_data["lat_L24"].append(lat_probes["span_L24"])
                probe_data["lat_L28"].append(lat_probes["span_L28"])
                probe_data["ans_L28"].append(lat_probes["ans_L28"])
                probe_data["dot_L14"].append(dot_probes["span_L14"])
                probe_data["dot_L24"].append(dot_probes["span_L24"])
                probe_data["dot_L28"].append(dot_probes["span_L28"])
                probe_data["y"].append(1 if gold == "B" else 0)
                del state_dots

            if two_way:
                base_res = w2_out["W2_Intact"]
                m0 = base_res["logp"]["B"] - base_res["logp"]["A"]
                task_items_cache.append({
                    "doc_id": doc_id,
                    "gold": gold,
                    "state_cpu": {k: v.cpu() for k, v in state.items()},
                    "lat_states_cpu": lat_states_cpu,
                    "base_res": base_res,
                    "m0": m0,
                })

            del state
            if (doc_id + 1) % 25 == 0:
                torch.cuda.empty_cache()

        if two_way and len(task_items_cache) >= 2:
            sorted_by_m0 = sorted(task_items_cache, key=lambda x: x["m0"])
            n_c = len(sorted_by_m0)
            for i, item in enumerate(task_items_cache):
                if item["base_res"]["tf_pred"] == "A":
                    cf_item = sorted_by_m0[-1 - (i % max(1, n_c // 5))]
                else:
                    cf_item = sorted_by_m0[i % max(1, n_c // 5)]

                state_gpu = {k: v.to("cuda:0", non_blocking=True) for k, v in item["state_cpu"].items()}
                patched_res = analyzer.run_patched_latents(state_gpu, cf_item["lat_states_cpu"], letters="AB")
                m_patched = patched_res["logp"]["B"] - patched_res["logp"]["A"]

                rec_w4a = {
                    "key": f"{task}|{item['doc_id']}",
                    "task": task,
                    "doc_id": item["doc_id"],
                    "gold": item["gold"],
                    "orig_pred": item["base_res"]["tf_pred"],
                    "cf_doc_id": cf_item["doc_id"],
                    "cf_pred": cf_item["base_res"]["tf_pred"],
                    "patched_pred": patched_res["tf_pred"],
                    "flipped_to_cf": (patched_res["tf_pred"] != item["base_res"]["tf_pred"]),
                    "m_orig": round(item["m0"], 4),
                    "m_cf": round(cf_item["m0"], 4),
                    "m_patched": round(m_patched, 4),
                    "delta_m": round(m_patched - item["m0"], 4),
                }
                w4a_records.append(rec_w4a)
                task_w4a_records.append(rec_w4a)
                del state_gpu

            del task_items_cache
        torch.cuda.empty_cache()

        if args.limit is None:
            with open(task_w2_file, "w") as f:
                for r in task_w2_records:
                    f.write(json.dumps(r) + "\n")
            if two_way:
                with open(task_w4a_file, "w") as f:
                    for r in task_w4a_records:
                        f.write(json.dumps(r) + "\n")
            if task == "blink_sprel":
                torch.save({
                    "w4b_lat_layers": w4b_lat_layers,
                    "w4b_dot_layers": w4b_dot_layers,
                    "probe_data": probe_data,
                }, task_w4b_file)
            with open(task_done_marker, "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

        print(f"  Completed {task} in {time.time() - t_start:.1f}s elapsed")

    with open(os.path.join(out_dir, "w2_attention_knockout.jsonl"), "w") as f:
        for r in w2_records:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(out_dir, "w4a_activation_patching.jsonl"), "w") as f:
        for r in w4a_records:
            f.write(json.dumps(r) + "\n")

    summarize_w2_w4(w2_records, w4a_records, w4b_lat_layers, w4b_dot_layers, probe_data, out_dir)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(2.0 * min(stats.binom.cdf(min(b, c), n, 0.5), 1.0 - stats.binom.cdf(max(b, c) - 1, n, 0.5)))


def summarize_w2_w4(w2_records, w4a_records, w4b_lat_layers, w4b_dot_layers, probe_data, out_dir):
    cond_names = [
        "W2_Intact",
        "W2_KO_LatentToAns_All",
        "W2_KO_LatentToAns_Early",
        "W2_KO_LatentToAns_Mid",
        "W2_KO_LatentToAns_Late",
        "W2_KO_ImgToLatent_All",
        "W2_KO_Total_Sever",
    ]
    summary = {"W2": {}, "W4a": {}, "W4b": {}}

    e2_matches = [r["conditions"]["W2_Intact"]["tf_pred"] == r["e2_t1_pred"] for r in w2_records if r["e2_t1_pred"] is not None]
    sdpa_parity = float(np.mean(e2_matches)) if e2_matches else 1.0
    summary["W2"]["sdpa_vs_fa2_parity"] = round(sdpa_parity, 6)

    ref_corr = [r["conditions"]["W2_Intact"]["tf_pred"] == r["gold"] for r in w2_records]
    ref_margins = []
    for r in w2_records:
        c = r["conditions"]["W2_Intact"]
        g = r["gold"]
        alt = max(c["logit"][L] for L in r["letters"] if L != g)
        ref_margins.append(c["logit"][g] - alt)

    for cname in cond_names:
        corr = [r["conditions"][cname]["tf_pred"] == r["gold"] for r in w2_records]
        agree_with_intact = [r["conditions"][cname]["tf_pred"] == r["conditions"]["W2_Intact"]["tf_pred"] for r in w2_records]
        margins = []
        per_task_acc = {}
        for t in TASKS:
            t_corr = [r["conditions"][cname]["tf_pred"] == r["gold"] for r in w2_records if r["task"] == t]
            if t_corr:
                per_task_acc[t] = round(float(np.mean(t_corr)) * 100.0, 2)
        for r in w2_records:
            c = r["conditions"][cname]
            g = r["gold"]
            alt = max(c["logit"][L] for L in r["letters"] if L != g)
            margins.append(c["logit"][g] - alt)

        b = sum(x and not y for x, y in zip(ref_corr, corr))
        c_cnt = sum(not x and y for x, y in zip(ref_corr, corr))
        p_mcn = mcnemar_p(b, c_cnt)
        if cname != "W2_Intact" and len(margins) > 1 and not np.allclose(ref_margins, margins):
            _, p_wilcox = stats.wilcoxon(ref_margins, margins)
        else:
            p_wilcox = 1.0

        summary["W2"][cname] = {
            "micro_acc_pct": round(float(np.mean(corr)) * 100.0, 2),
            "macro_acc_pct": round(float(np.mean(list(per_task_acc.values()))), 2),
            "per_task_acc_pct": per_task_acc,
            "agree_with_intact_pct": round(float(np.mean(agree_with_intact)) * 100.0, 2),
            "mean_margin": round(float(np.mean(margins)), 4),
            "margin_delta_vs_intact": round(float(np.mean(margins) - np.mean(ref_margins)), 4),
            "mcnemar_b": int(b),
            "mcnemar_c": int(c_cnt),
            "mcnemar_p": round(p_mcn, 4),
            "wilcoxon_p": round(float(p_wilcox), 4),
        }

    if w4a_records:
        orig_acc = float(np.mean([r["orig_pred"] == r["gold"] for r in w4a_records])) * 100.0
        patched_acc = float(np.mean([r["patched_pred"] == r["gold"] for r in w4a_records])) * 100.0
        flip_rate = float(np.mean([r["flipped_to_cf"] for r in w4a_records])) * 100.0
        mean_abs_delta_m = float(np.mean([abs(r["delta_m"]) for r in w4a_records]))
        mean_cf_gap = float(np.mean([abs(r["m_cf"] - r["m_orig"]) for r in w4a_records]))
        summary["W4a"] = {
            "n_items": len(w4a_records),
            "orig_acc_pct": round(orig_acc, 2),
            "patched_acc_pct": round(patched_acc, 2),
            "flip_rate_pct": round(flip_rate, 2),
            "unchanged_rate_pct": round(100.0 - flip_rate, 2),
            "mean_abs_log_odds_shift": round(mean_abs_delta_m, 4),
            "mean_counterfactual_log_odds_gap": round(mean_cf_gap, 4),
            "fraction_of_cf_gap_transferred": round(mean_abs_delta_m / (mean_cf_gap + 1e-9), 4),
        }

    if w4b_lat_layers:
        n_layers = len(w4b_lat_layers[0])
        geom_table = []
        for l_idx in range(n_layers):
            geom_table.append({
                "layer": l_idx,
                "lat_raw_cos": round(float(np.mean([x[l_idx]["raw_intra_cos"] for x in w4b_lat_layers])), 4),
                "lat_centered_cos": round(float(np.mean([x[l_idx]["centered_intra_cos"] for x in w4b_lat_layers])), 4),
                "dot_centered_cos": round(float(np.mean([x[l_idx]["centered_intra_cos"] for x in w4b_dot_layers])), 4),
                "q_centered_cos": round(float(np.mean([x[l_idx]["q_centered_intra_cos"] for x in w4b_lat_layers])), 4),
                "lat_eff_rank": round(float(np.mean([x[l_idx]["effective_rank"] for x in w4b_lat_layers])), 2),
                "dot_eff_rank": round(float(np.mean([x[l_idx]["effective_rank"] for x in w4b_dot_layers])), 2),
                "lat_vis_align": round(float(np.mean([x[l_idx]["centered_vis_align"] for x in w4b_lat_layers])), 4),
                "dot_vis_align": round(float(np.mean([x[l_idx]["centered_vis_align"] for x in w4b_dot_layers])), 4),
            })
        summary["W4b"]["layer_geometry"] = geom_table

        y = np.array(probe_data["y"])
        if len(y) >= 20 and len(np.unique(y)) == 2:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import StandardScaler

            probe_scores = {}
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for k_name in ["lat_L14", "dot_L14", "lat_L24", "dot_L24", "lat_L28", "dot_L28", "ans_L28"]:
                X = np.array(probe_data[k_name])
                fold_accs = []
                for train_idx, test_idx in skf.split(X, y):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X[train_idx])
                    X_te = scaler.transform(X[test_idx])
                    clf = LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
                    clf.fit(X_tr, y[train_idx])
                    fold_accs.append(clf.score(X_te, y[test_idx]))
                probe_scores[k_name] = round(float(np.mean(fold_accs)) * 100.0, 2)
            summary["W4b"]["linear_probe_5fold_acc_pct"] = probe_scores

    with open(os.path.join(out_dir, "w2_w4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# WSDM Experiments W2 & W4: Causal Attention Knockout, Activation Patching & Pre-Norm Geometry",
        f"**Evaluated Items:** N={len(w2_records)} (2-Way Patching N={len(w4a_records)}) | **SDPA vs. FlashAttention-2 Parity:** {sdpa_parity*100:.2f}%\n",
        "## 1. Experiment W2: Causal 4D Attention Mask Knockout (N=700)\n",
        "| Condition | Intervention | Micro Acc | Macro Acc | Agree w/ Intact | Mean Margin | Δ Margin | McNemar b/c | McNemar p |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    desc_map = {
        "W2_Intact": "No intervention (Baseline T1)",
        "W2_KO_LatentToAns_All": "Block `Latents -> Answer` (All Layers 0–27)",
        "W2_KO_LatentToAns_Early": "Block `Latents -> Answer` (Early Layers 0–9)",
        "W2_KO_LatentToAns_Mid": "Block `Latents -> Answer` (Mid Layers 10–18)",
        "W2_KO_LatentToAns_Late": "Block `Latents -> Answer` (Late Layers 19–27)",
        "W2_KO_ImgToLatent_All": "Block `Image -> Latents` (All Layers 0–27)",
        "W2_KO_Total_Sever": "Block BOTH `Image -> Latents` & `Latents -> Answer` (0–27)",
    }
    for cname in cond_names:
        s = summary["W2"][cname]
        md.append(
            f"| **{cname}** | {desc_map[cname]} | **{s['micro_acc_pct']:.2f}%** | {s['macro_acc_pct']:.2f}% | "
            f"{s['agree_with_intact_pct']:.2f}% | {s['mean_margin']:.4f} | {s['margin_delta_vs_intact']:+.4f} | "
            f"{s['mcnemar_b']}/{s['mcnemar_c']} | p={s['mcnemar_p']:.4f} |"
        )

    if summary["W4a"]:
        w4a = summary["W4a"]
        md.extend([
            "\n## 2. Experiment W4a: Counterfactual Cross-Image Latent Activation Patching (N=550)\n",
            f"- **Original Unpatched Accuracy:** `{w4a['orig_acc_pct']:.2f}%`",
            f"- **Patched Accuracy (with opposite-answer image latents across all 28 layers):** `{w4a['patched_acc_pct']:.2f}%`",
            f"- **Prediction Unchanged Rate:** **`{w4a['unchanged_rate_pct']:.2f}%`** (Flip Rate: `{w4a['flip_rate_pct']:.2f}%`)",
            f"- **Mean Counterfactual Log-Odds Gap (`|m_cf - m_orig|`)**: `{w4a['mean_counterfactual_log_odds_gap']:.4f}`",
            f"- **Mean Patched Log-Odds Shift (`|m_patched - m_orig|`)**: `{w4a['mean_abs_log_odds_shift']:.4f}` (`{w4a['fraction_of_cf_gap_transferred']*100:.2f}%` of counterfactual signal transferred)",
        ])

    if "layer_geometry" in summary["W4b"]:
        md.extend([
            "\n## 3. Experiment W4b: Pre-RMSNorm Layerwise Geometry (Anisotropy-Corrected, N=143)\n",
            "| Layer | `<|latent_pad|>` Raw Cos | `<|latent_pad|>` Centered Cos | `\" .\"` Pause Centered Cos | Question Centered Cos | `<|latent_pad|>` Eff. Rank | `\" .\"` Eff. Rank | `<|latent_pad|>` Vis Align | `\" .\"` Vis Align |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for row in summary["W4b"]["layer_geometry"]:
            if row["layer"] in [0, 1, 4, 8, 12, 14, 18, 20, 24, 27, 28]:
                lbl = "0 (Embed)" if row["layer"] == 0 else (f"{row['layer']} (Pre-Norm Final)" if row["layer"] == 28 else str(row["layer"]))
                md.append(
                    f"| **{lbl}** | {row['lat_raw_cos']:.4f} | **{row['lat_centered_cos']:.4f}** | **{row['dot_centered_cos']:.4f}** | "
                    f"{row['q_centered_cos']:.4f} | {row['lat_eff_rank']:.2f} | {row['dot_eff_rank']:.2f} | "
                    f"{row['lat_vis_align']:.4f} | {row['dot_vis_align']:.4f} |"
                )

    if "linear_probe_5fold_acc_pct" in summary["W4b"]:
        ps = summary["W4b"]["linear_probe_5fold_acc_pct"]
        md.extend([
            "\n### 5-Fold Cross-Validated Linear Probe Accuracy on `blink_sprel` (N=143)\n",
            f"- **Layer 14 Pooled**: `<|latent_pad|>` = `{ps['lat_L14']:.2f}%` vs. `\" .\"` Pause = `{ps['dot_L14']:.2f}%`",
            f"- **Layer 24 Pooled**: `<|latent_pad|>` = `{ps['lat_L24']:.2f}%` vs. `\" .\"` Pause = `{ps['dot_L24']:.2f}%`",
            f"- **Layer 28 (Final Pre-Norm) Pooled**: `<|latent_pad|>` = `{ps['lat_L28']:.2f}%` vs. `\" .\"` Pause = `{ps['dot_L28']:.2f}%`",
            f"- **Layer 28 `<answer` Decision Token**: **`{ps['ans_L28']:.2f}%`**",
        ])

    report_path = os.path.join(out_dir, "W2_W4_SUMMARY.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
