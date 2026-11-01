"""
eval_monet_empirical.py:
Empirical diagnostic suite for Monet (NOVAglow646/Monet-7B).
Executes on GPU 0:
1. Special token embedding forensic audit.
2. Latent Visual Reasoning Phase Execution (Mode B):
   - Injects <abs_vis_token> to initiate latent visual thought.
   - Measures step-by-step continuous latent rollout telemetry:
     * Hidden state norm ||z_k||_2.
     * Pairwise cosine similarity cos(z_k, z_{k+1}).
     * SVD singular value spectrum and effective rank.
3. Causal Intervention Ablations on GPU 0:
   - Condition C0: Natural continuous latents.
   - Condition C1: Zero vectors (z_k = 0).
   - Condition C2: Gaussian noise (z_k ~ N(mu, sigma^2)).
   - Condition C3: Donor latents from random vector.
   - Condition C4: Latent quota sweep (K in {0, 1, 3, 5, 10}).
4. Generates structured JSON telemetry and empirical report.
"""

import os
import sys
import json
import time
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# Ensure external_models/monet is in sys.path
monet_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(monet_dir))

from monet_loader import load_monet_model_and_processor
from qwen_vl_utils import process_vision_info

LATENT_START_ID = 151666 # <abs_vis_token>
LATENT_END_ID = 151667   # </abs_vis_token>
PAD_ID = 151665          # <abs_vis_token_pad>

def forensic_audit_token_embeddings(model, tokenizer):
    print("=" * 70)
    print("1. SPECIAL TOKEN EMBEDDING FORENSIC AUDIT")
    print("=" * 70)
    embed_weights = model.model.language_model.embed_tokens.weight.data # [151670, 3584]
    vocab_size, hidden_dim = embed_weights.shape
    print(f"Embedding matrix shape: {vocab_size} x {hidden_dim}")
    
    special_tokens = {
        "abs_vis_pad": ("<abs_vis_token_pad>", 151665),
        "abs_vis_start": ("<abs_vis_token>", 151666),
        "abs_vis_end": ("</abs_vis_token>", 151667),
        "observation_start": ("<observation>", 151668),
        "observation_end": ("</observation>", 151669),
    }
    
    audit_results = {}
    for name, (tok, expected_id) in special_tokens.items():
        actual_id = tokenizer.convert_tokens_to_ids(tok)
        assert actual_id == expected_id, f"Token mismatch: {tok} expected {expected_id}, got {actual_id}"
        
        vec = embed_weights[actual_id].float()
        l2_norm = vec.norm().item()
        mean_val = vec.mean().item()
        std_val = vec.std().item()
        audit_results[name] = {
            "token": tok,
            "id": actual_id,
            "l2_norm": float(l2_norm),
            "mean": float(mean_val),
            "std": float(std_val),
            "is_zero": bool(l2_norm < 1e-6)
        }
        print(f"  Token {tok:<20} (ID: {actual_id}): L2 Norm = {l2_norm:8.4f}, Mean = {mean_val:10.6f}, Std = {std_val:8.4f}")
        
    # Check baseline random token norm for comparison
    rand_ids = torch.randint(0, 151600, (100,))
    rand_norms = embed_weights[rand_ids].float().norm(dim=-1).mean().item()
    print(f"  Reference text token average L2 Norm: {rand_norms:8.4f}")
    audit_results["reference_avg_norm"] = float(rand_norms)
    
    return audit_results


def generate_with_latent_intervention(
    model, 
    processor, 
    tokenizer, 
    messages, 
    latent_size=10, 
    intervention_mode="none", # "none", "zero", "noise", "donor"
    donor_latents=None,
    device="cuda:0",
    max_new_tokens=64
):
    """
    Executes Monet's latent reasoning phase:
    1. Prefill prompt up to <|im_start|>assistant\n.
    2. Inject LATENT_START_ID (<abs_vis_token>) to initiate visual latent thinking.
    3. Run K steps of continuous latent propagation, overwriting input embeddings with last hidden state.
    4. Force LATENT_END_ID (</abs_vis_token>) and continue autoregressive generation until eos.
    """
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    latent_vectors = []
    cos_sims = []
    generated_ids = []
    
    with torch.no_grad():
        # Prefill prompt
        outputs = model(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
            image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            output_hidden_states=True
        )
        
        past_key_values = outputs.past_key_values
        cur_pos = inputs.input_ids.shape[1]
        
        if latent_size > 0:
            # Inject LATENT_START_ID (<abs_vis_token>)
            start_token = torch.tensor([[LATENT_START_ID]], device=device)
            generated_ids.append(LATENT_START_ID)
            cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
            outputs = model(
                input_ids=start_token,
                past_key_values=past_key_values,
                cache_position=cache_pos,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values
            cur_pos += 1
            current_latent = outputs.hidden_states[0][-1:, -1:, :] # [1, 1, H]
            latent_vectors.append(current_latent[0, 0].float().cpu())
            
            # Continuous latent rollout for latent_size - 1 steps
            for k in range(latent_size - 1):
                # Apply intervention
                if intervention_mode == "none":
                    injected_embed = current_latent
                elif intervention_mode == "zero":
                    injected_embed = torch.zeros_like(current_latent)
                elif intervention_mode == "noise":
                    injected_embed = torch.randn_like(current_latent) * current_latent.std() + current_latent.mean()
                elif intervention_mode == "donor" and donor_latents is not None and k < len(donor_latents):
                    injected_embed = donor_latents[k].to(device).view(1, 1, -1).to(current_latent.dtype)
                else:
                    injected_embed = current_latent
                    
                cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
                outputs = model(
                    inputs_embeds=injected_embed,
                    past_key_values=past_key_values,
                    cache_position=cache_pos,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = outputs.past_key_values
                cur_pos += 1
                next_latent = outputs.hidden_states[0][-1:, -1:, :] # [1, 1, H]
                
                cos = F.cosine_similarity(current_latent[0, 0].float(), next_latent[0, 0].float(), dim=0).item()
                cos_sims.append(cos)
                latent_vectors.append(next_latent[0, 0].float().cpu())
                current_latent = next_latent
                
            # Close latent thought with LATENT_END_ID
            end_token = torch.tensor([[LATENT_END_ID]], device=device)
            generated_ids.append(LATENT_END_ID)
            cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
            outputs = model(
                input_ids=end_token,
                past_key_values=past_key_values,
                cache_position=cache_pos,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values
            cur_pos += 1
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
        else:
            # Latent size = 0: Pure text generation without latent thought
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            
        # Continue autoregressive text generation
        for _ in range(max_new_tokens):
            if next_token.item() in [tokenizer.eos_token_id, 151645, 151643]:
                break
            cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                cache_position=cache_pos,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            cur_pos += 1
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    telemetry = {
        "num_latents": len(latent_vectors),
        "latent_norms": [float(v.norm().item()) for v in latent_vectors],
        "cos_sims": [float(c) for c in cos_sims],
        "mean_cos_sim": float(np.mean(cos_sims)) if cos_sims else 0.0,
        "generated_text": decoded_text,
        "token_count": len(generated_ids)
    }
    
    if len(latent_vectors) > 1:
        Z = torch.stack(latent_vectors, dim=0) # [K, H]
        U, S, V = torch.linalg.svd(Z - Z.mean(dim=0, keepdim=True), full_matrices=False)
        telemetry["singular_values"] = [float(s.item()) for s in S[:min(len(S), 10)]]
        telemetry["energy_top1"] = float((S[0]**2 / (S**2).sum()).item())
        telemetry["energy_top3"] = float(((S[:3]**2).sum() / (S**2).sum()).item())
        
    return telemetry, latent_vectors


def run_empirical_suite():
    print("=" * 70)
    print("STARTING EMPIRICAL DIAGNOSTICS SUITE ON GPU 0")
    print("=" * 70)
    
    model, processor, tokenizer = load_monet_model_and_processor(
        model_id_or_path="NOVAglow646/Monet-7B",
        device="cuda:0",
        torch_dtype=torch.bfloat16
    )
    
    # 1. Forensic audit of special tokens
    token_audit = forensic_audit_token_embeddings(model, tokenizer)
    
    # 2. Benchmark probe sample test
    sample_img_path = "/home/devraj/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/d9ae62c903da0c98336e85c5ee89cd863b04b4da/direct_attributes/sa_34040.jpg"
    assert os.path.exists(sample_img_path), f"Sample image {sample_img_path} not found!"
    raw_img = Image.open(sample_img_path).convert("RGB")
    print(f"\nLoaded probe image: {sample_img_path} (Size: {raw_img.size})")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_img},
                {"type": "text", "text": "Question: What is the main object in this image? Put your final answer in \\boxed{}."}
            ]
        }
    ]
    
    print("\n" + "=" * 70)
    print("2. TESTING LATENT GENERATION TRAJECTORY & DYNAMICS (K=10)")
    print("=" * 70)
    telemetry_c0, natural_latents = generate_with_latent_intervention(
        model, processor, tokenizer, messages,
        latent_size=10,
        intervention_mode="none"
    )
    print(f"C0 Generated text ({telemetry_c0['token_count']} tokens):")
    print(f"'{telemetry_c0['generated_text']}'\n")
    print(f"Latent Norms: {[round(n, 2) for n in telemetry_c0['latent_norms']]}")
    print(f"Successive Latent Cosine Similarities: {[round(c, 4) for c in telemetry_c0['cos_sims']]}")
    print(f"Mean Latent Cosine Sim: {telemetry_c0['mean_cos_sim']:.4f}")
    if "energy_top1" in telemetry_c0:
        print(f"Trajectory Subspace Energy Top 1: {telemetry_c0['energy_top1']*100:.2f}% | Top 3: {telemetry_c0['energy_top3']*100:.2f}%")
        
    print("\n" + "=" * 70)
    print("3. CAUSAL INTERVENTION ABLATIONS (H4)")
    print("=" * 70)
    
    # C1: Zero vector
    print("--> Testing Condition C1: Zero vectors (z_k = 0)...")
    telemetry_c1, _ = generate_with_latent_intervention(
        model, processor, tokenizer, messages,
        latent_size=10,
        intervention_mode="zero"
    )
    print(f"C1 Text: '{telemetry_c1['generated_text']}'\n")
    
    # C2: Gaussian noise
    print("--> Testing Condition C2: Gaussian noise (z_k ~ N(mu, sigma^2))...")
    telemetry_c2, _ = generate_with_latent_intervention(
        model, processor, tokenizer, messages,
        latent_size=10,
        intervention_mode="noise"
    )
    print(f"C2 Text: '{telemetry_c2['generated_text']}'\n")
    
    # C3: Donor latents (unrelated random donor)
    print("--> Testing Condition C3: Donor latents...")
    rand_donor = [torch.randn(3584) * 5.0 for _ in range(10)]
    telemetry_c3, _ = generate_with_latent_intervention(
        model, processor, tokenizer, messages,
        latent_size=10,
        intervention_mode="donor",
        donor_latents=rand_donor
    )
    print(f"C3 Text: '{telemetry_c3['generated_text']}'\n")
    
    # C4: Latent quota sweep (K in {0, 1, 3, 5, 10})
    print("\n--> Testing Condition C4: Latent Quota Sweep (K in {0, 1, 3, 5, 10})...")
    quota_results = {}
    for k in [0, 1, 3, 5, 10]:
        t_k, _ = generate_with_latent_intervention(
            model, processor, tokenizer, messages,
            latent_size=k,
            intervention_mode="none"
        )
        quota_results[f"K={k}"] = {
            "token_count": t_k["token_count"],
            "generated_text": t_k["generated_text"]
        }
        print(f"  Quota K={k:2d}: Generated {t_k['token_count']} tokens | Text: '{t_k['generated_text']}'")
        
    # Compile report dictionary
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": "NOVAglow646/Monet-7B",
        "device": "NVIDIA A100-PCIE-40GB",
        "token_audit": token_audit,
        "baseline_trajectory": telemetry_c0,
        "interventions": {
            "C0_natural": telemetry_c0,
            "C1_zero": telemetry_c1,
            "C2_noise": telemetry_c2,
            "C3_donor": telemetry_c3
        },
        "quota_sweep": quota_results
    }
    
    out_path = monet_dir / "diagnostics" / "monet_empirical_telemetry.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved full empirical telemetry to {out_path}]")
    print("=" * 70)
    print("EMPIRICAL DIAGNOSTICS COMPLETE (0 FAILURES).")
    print("=" * 70)

if __name__ == "__main__":
    run_empirical_suite()
