#!/usr/bin/env python3
"""
Mull-Tokens (Modality-agnostic Latent Thinking) Minimal CLI Inference Script.

Paper: Mull-Tokens: Modality-Agnostic Latent Thinking (Ray et al., 2025) - https://arxiv.org/abs/2512.10941
Repository: https://github.com/arijitray1993/mull
Model: array/Qwen2.5-VL-Mull

Key Mechanism:
At inference, Mull-Tokens pre-fills the assistant turn with:
  "<think>" + "<|latent_pad|>" * 20 + "</think>\n"
This prompts the model to perform discrete latent thinking (compressed multimodal reasoning)
before generating its final textual response.
"""

import os
import sys
import argparse
import torch

# Ensure the installed package's qwen_vl_utils (version 0.0.14) is used
import qwen_vl_utils
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Mull-Tokens CLI Inference")
    parser.add_argument(
        "--model_id",
        type=str,
        default="array/Qwen2.5-VL-Mull",
        help="HuggingFace model ID or local path (default: array/Qwen2.5-VL-Mull)",
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--image_path",
        type=str,
        default=os.path.join(base_dir, "spatial_test.png"),
        help="Path to input image for inference",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Which shape is located in the middle, between the red square and the green triangle, and what color is it? Please choose between: A. Blue circle B. Yellow star C. Black pentagon D. Purple cross.",
        help="Question text for the model",
    )
    parser.add_argument(
        "--question_type",
        type=str,
        default="multiple choice",
        choices=["multiple choice", "numerical", "OCR", "free-form", "regression"],
        help="Question type formatting template",
    )
    parser.add_argument(
        "--num_latents",
        type=int,
        default=20,
        help="Number of <|latent_pad|> thinking tokens to prepend (default: 20)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation ('flash_attention_2' or 'sdpa')",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Mull-Tokens Inference")
    print(f"Model: {args.model_id}")
    print(f"Image: {args.image_path}")
    print(f"Question: {args.question}")
    print(f"Latent thinking tokens: {args.num_latents}x <|latent_pad|>")
    print(f"Attention implementation: {args.attn_implementation}")
    print(f"qwen_vl_utils location: {qwen_vl_utils.__file__}")
    print("=" * 60)

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found at {args.image_path}")

    # Prompt templates as specified in Mull README
    QUESTION_TEMPLATE_LATENT = (
        "{Question}\n"
        "Please think about this question deeply. "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your final answer between the <answer> </answer> tags."
    )
    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    }

    prompt = (
        QUESTION_TEMPLATE_LATENT.format(Question=args.question)
        + TYPE_TEMPLATE[args.question_type]
    )

    # 1. Load Model and Processor
    try:
        print(f"\n[1/4] Loading model with attn_implementation='{args.attn_implementation}'...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map="auto",
        )
    except Exception as e:
        print(f"Warning: Loading with '{args.attn_implementation}' failed: {e}")
        print("Falling back to attn_implementation='sdpa'...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )

    print("\n[2/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Verify latent token
    latent_pad_id = processor.tokenizer.convert_tokens_to_ids("<|latent_pad|>")
    print(f"Token '<|latent_pad|>' ID: {latent_pad_id}")

    # 2. Prepare Messages with Pre-filled Latent Thinking
    # KEY STEP: Mull-Tokens requires latent thinking tokens before answer generation
    # Pre-fill assistant turn with "<think>" + "<|latent_pad|>" * NUM_LATENTS + "</think>\n"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<think>" + "<|latent_pad|>" * args.num_latents + "</think>\n",
                }
            ],
        },
    ]

    print("\n[3/4] Formatting input with chat template and vision processing...")
    # Apply chat template with add_generation_prompt=False (prefill already added)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Strip <|im_end|>\n so model continues generation seamlessly
    text = text.replace("<|im_end|>\n", "")

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"Input tokens length: {inputs['input_ids'].shape[1]}")

    # 3. Generate response
    print("\n[4/4] Generating response (do_sample=False, max_new_tokens=512)...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # Decode output (skipping prompt/prefill input tokens)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "=" * 60)
    print("MULL-TOKENS MODEL RESPONSE:")
    print("=" * 60)
    print(response.strip())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
