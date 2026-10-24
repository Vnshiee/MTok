#!/usr/bin/env python3
"""
Experiment 1 Diagnostic: Attention Allocation & Causal Attribution Analysis.

This script loads the replicated Mull-Tokens model (array/Qwen2.5-VL-Mull),
performs forward passes with attention extraction enabled, and computes the
exact distribution of self-attention weights from the answer tokens back to:
  1. Visual image patch tokens (<|image_pad|>)
  2. Text / Question prompt tokens
  3. The 20 Mull thinking tokens (<|latent_pad|>)

Outputs a detailed layer-by-layer breakdown and average attention percentages.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure packages and paths
transformers_path = os.path.join(BASE_DIR, "Video-R1/transformers-main/src")
if os.path.exists(transformers_path):
    sys.path.insert(0, transformers_path)

import qwen_vl_utils
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def run_attention_analysis(output_file=None):
    if output_file is None:
        output_file = os.path.join(BASE_DIR, "eval_outputs/exp1_attention_distribution.json")
    print("=" * 70)
    print("EXPERIMENT 1 DIAGNOSTIC: Latent Token Attention Attribution")
    print("=" * 70)

    model_id = "array/Qwen2.5-VL-Mull"
    num_latents = 20

    print(f"[1/3] Loading {model_id} with eager attention for attention extraction...")
    # output_attentions=True requires eager or sdpa (flash_attention_2 does not output dense attention matrices)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    latent_token_id = processor.tokenizer.convert_tokens_to_ids("<|latent_pad|>")
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"Token IDs: <|latent_pad|> = {latent_token_id}, <|image_pad|> = {image_token_id}")

    test_samples = [
        {
            "id": "sample_spatial_shapes",
            "image": os.path.join(BASE_DIR, "spatial_test.png"),
            "question": "Which shape is located in the middle, between the red square and the green triangle, and what color is it? Please choose between: A. Blue circle B. Yellow star C. Black pentagon D. Purple cross. Please think about this question deeply. Provide your final answer between the <answer> </answer> tags. Please provide only the single option letter within the <answer> </answer> tags.",
        },
        {
            "id": "sample_shapes_visual",
            "image": os.path.join(BASE_DIR, "mull/images/shapes.png"),
            "question": "Describe the spatial layout and relative positions of all shapes. Please think about this question deeply. Provide your final answer between the <answer> </answer> tags.",
        },
        {
            "id": "sample_red_left_blue_right",
            "image": os.path.join(BASE_DIR, "mull/images/red_left_blue_right.png"),
            "question": "Is the red box to the left of the blue cylinder? (A) Yes (B) No. Please think about this question deeply. Provide your final answer between the <answer> </answer> tags. Please provide only the single option letter within the <answer> </answer> tags.",
        }
    ]

    results = []

    print(f"\n[2/3] Running attention attribution across {len(test_samples)} representative spatial tasks...")
    for idx, sample in enumerate(test_samples):
        img_path = sample["image"]
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} (not found)")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<think>" + "<|latent_pad|>" * num_latents + "</think>\n<answer>"
                    }
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if text.endswith("<|im_end|>\n"):
            text = text[:-len("<|im_end|>\n")]
        elif text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

        input_ids = inputs.input_ids[0]
        seq_len = input_ids.shape[0]

        # Identify token masks
        mull_mask = (input_ids == latent_token_id)
        image_mask = (input_ids == image_token_id)
        # All other tokens before the answer token
        text_mask = ~(mull_mask | image_mask)
        # Zero out the answer token itself from text mask
        text_mask[-1] = False

        num_mull = mull_mask.sum().item()
        num_image = image_mask.sum().item()
        num_text = text_mask.sum().item()

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions is a tuple of (batch=1, num_heads, seq_len, seq_len) per layer
        num_layers = len(outputs.attentions)
        layer_breakdown = []

        for layer_idx in range(num_layers):
            # Attention from the last token (<answer>) to all preceding tokens
            # Shape: (num_heads, seq_len)
            attn_last = outputs.attentions[layer_idx][0, :, -1, :].float()

            # Sum attention allocated to each token type (averaged across all attention heads)
            mull_attn = attn_last[:, mull_mask].sum(dim=-1).mean().item()
            image_attn = attn_last[:, image_mask].sum(dim=-1).mean().item()
            text_attn = attn_last[:, text_mask].sum(dim=-1).mean().item()
            total = mull_attn + image_attn + text_attn + 1e-9

            layer_breakdown.append({
                "layer": layer_idx + 1,
                "mull_pct": round(100.0 * mull_attn / total, 2),
                "image_pct": round(100.0 * image_attn / total, 2),
                "text_pct": round(100.0 * text_attn / total, 2),
            })

        # Calculate averages for lower (1-9), middle (10-19), and upper (20-28) layers
        lower_mull = np.mean([l["mull_pct"] for l in layer_breakdown[:9]])
        mid_mull = np.mean([l["mull_pct"] for l in layer_breakdown[9:19]])
        upper_mull = np.mean([l["mull_pct"] for l in layer_breakdown[19:]])
        overall_mull = np.mean([l["mull_pct"] for l in layer_breakdown])

        overall_image = np.mean([l["image_pct"] for l in layer_breakdown])
        overall_text = np.mean([l["text_pct"] for l in layer_breakdown])

        sample_res = {
            "sample_id": sample["id"],
            "seq_len": seq_len,
            "token_counts": {"mull": num_mull, "image": num_image, "text": num_text},
            "overall_attention": {
                "mull_tokens_pct": round(float(overall_mull), 2),
                "image_tokens_pct": round(float(overall_image), 2),
                "text_tokens_pct": round(float(overall_text), 2),
            },
            "mull_attention_by_depth": {
                "early_layers_1_to_9": round(float(lower_mull), 2),
                "mid_layers_10_to_19": round(float(mid_mull), 2),
                "late_decision_layers_20_to_28": round(float(upper_mull), 2),
            },
            "layers": layer_breakdown,
        }
        results.append(sample_res)

        print(f"\n--- Results for {sample['id']} ---")
        print(f"Total Tokens: {seq_len} (Image: {num_image}, Text: {num_text}, Mull Latents: {num_mull})")
        print(f"Overall Attention Allocation from <answer>:")
        print(f"  • Visual Tokens:  {overall_image:.2f}%")
        print(f"  • Question Text:  {overall_text:.2f}%")
        print(f"  • Mull Latents:   {overall_mull:.2f}%")
        print(f"Mull Attention by Depth:")
        print(f"  • Early Layers (1-9):   {lower_mull:.2f}%")
        print(f"  • Mid Layers (10-19):   {mid_mull:.2f}%")
        print(f"  • Late Layers (20-28):  {upper_mull:.2f}%")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[3/3] Detailed attention attribution saved to {output_file}")


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "eval_outputs/exp1_attention_distribution.json")
    run_attention_analysis(out_path)
