import os
import sys
import json
import time
import re
import argparse
import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

# Ensure path includes monet directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MONET_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, MONET_DIR)

from monet_loader import load_monet_model_and_processor
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

LATENT_START_ID = 151666
LATENT_END_ID = 151667

VSTAR_BENCH_DIR = os.path.expanduser("~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/d9ae62c903da0c98336e85c5ee89cd863b04b4da")
MMVP_DIR = os.path.expanduser("~/.cache/huggingface/hub/datasets--MMVP--MMVP/snapshots/37eafecab8a3940c50c2ade5b36de69dbc99a8cf")


def parse_boxed_option(text: str, prompt_text: str = "") -> str:
    """Extracts option letter (A, B, C, D) from boxed text or final pattern."""
    # 1. Look for \boxed{A}
    boxed = re.findall(r'\\boxed\{([A-Ea-e])\}', text)
    if boxed:
        return boxed[-1].upper()
    
    # 2. Look for \boxed{...} containing letter or matching option text
    boxed_gen = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_gen:
        content = boxed_gen[-1].strip()
        m = re.search(r'\b([A-Ea-e])\b', content)
        if m:
            return m.group(1).upper()
        if prompt_text:
            options = re.findall(r'\(([A-Ea-e])\)\s*([^(\n\r]+)', prompt_text)
            content_lower = content.lower()
            for opt_letter, opt_text in options:
                opt_text_clean = opt_text.strip().lower()
                if opt_text_clean and (opt_text_clean in content_lower or content_lower in opt_text_clean):
                    return opt_letter.upper()
            
    # 3. Look for "the answer is (A)" or "option (A)" or "(A)"
    m = re.findall(r'(?:answer is|choice is|option is|option:?)\s*\(?([A-Ea-e])\)?', text, re.IGNORECASE)
    if m:
        return m[-1].upper()
        
    # 4. Look for isolated (A) or A. at the end of string
    m = re.findall(r'\(([A-Ea-e])\)', text)
    if m:
        return m[-1].upper()
        
    # 5. Look for standalone capital letter at the end
    m = re.findall(r'\b([A-E])\b', text)
    if m:
        return m[-1].upper()
        
    # 6. Fallback: search prompt options in text
    if prompt_text:
        options = re.findall(r'\(([A-Ea-e])\)\s*([^(\n\r]+)', prompt_text)
        text_lower = text.lower()
        for opt_letter, opt_text in options:
            opt_text_clean = opt_text.strip().lower()
            if opt_text_clean and len(opt_text_clean) >= 3 and f"is {opt_text_clean}" in text_lower:
                return opt_letter.upper()
                
    return "UNKNOWN"


def load_benchmark_data(benchmark_name: str, limit: int = None):
    samples = []
    if benchmark_name == "vstar":
        q_path = os.path.join(VSTAR_BENCH_DIR, "test_questions.jsonl")
        assert os.path.exists(q_path), f"V* Bench questions not found at {q_path}"
        with open(q_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                img_path = os.path.join(VSTAR_BENCH_DIR, data["image"])
                samples.append({
                    "id": str(data["question_id"]),
                    "category": data.get("category", "general"),
                    "image_path": img_path,
                    "prompt": data["text"],
                    "ground_truth": data["label"].upper()
                })
    elif benchmark_name == "mmvp":
        csv_path = os.path.join(MMVP_DIR, "Questions.csv")
        assert os.path.exists(csv_path), f"MMVP Questions.csv not found at {csv_path}"
        df = pd.read_csv(csv_path)
        img_dir = os.path.join(MMVP_DIR, "MMVP Images")
        for _, row in df.iterrows():
            idx = int(row["Index"])
            img_path = os.path.join(img_dir, f"{idx}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, f"{idx}.png")
            
            # MMVP format: Options string "(a) ... (b) ...", Correct Answer "(a)"
            gt = str(row["Correct Answer"]).strip().replace("(", "").replace(")", "").upper()
            prompt = f"{row['Question']}\n{row['Options']}\nAnswer with the option's letter from the given choices directly. Put your final answer in \\boxed{{LETTER}}."
            samples.append({
                "id": str(idx),
                "category": f"pair_{int((idx-1)//2)}",
                "image_path": img_path,
                "prompt": prompt,
                "ground_truth": gt
            })
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
    if limit is not None and limit > 0:
        samples = samples[:limit]
    return samples


def generate_monet_prediction(
    model, processor, tokenizer,
    image: Image.Image,
    question_text: str,
    latent_size: int = 10,
    intervention_mode: str = "none",
    donor_latents = None,
    device: str = "cuda:0",
    max_new_tokens: int = 128
):
    system_prompt = (
        "You are a helpful multimodal assistant. You are required to answer the question based on the image provided. "
        "Put your final answer in \\boxed{LETTER}."
    )
    
    if "\\boxed" not in question_text:
        formatted_question = f"{question_text}\nAnswer with the option's letter directly in \\boxed{{LETTER}}."
    else:
        formatted_question = question_text
        
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": formatted_question}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    generated_ids = []
    latent_norms = []
    cos_sims = []
    
    with torch.no_grad():
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
            latent_norms.append(float(current_latent.norm().item()))
            
            for k in range(latent_size - 1):
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
                next_latent = outputs.hidden_states[0][-1:, -1:, :]
                
                cos = F.cosine_similarity(current_latent[0, 0].float(), next_latent[0, 0].float(), dim=0).item()
                cos_sims.append(cos)
                latent_norms.append(float(next_latent.norm().item()))
                current_latent = next_latent
                
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
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            
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
            
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    parsed_option = parse_boxed_option(raw_text, question_text)
    
    return {
        "raw_text": raw_text,
        "parsed_option": parsed_option,
        "latent_norms": latent_norms,
        "mean_cos_sim": float(np.mean(cos_sims)) if cos_sims else 0.0,
        "generated_token_count": len(generated_ids)
    }


def generate_qwen_base_prediction(
    model, processor, tokenizer,
    image: Image.Image,
    question_text: str,
    device: str = "cuda:0",
    max_new_tokens: int = 128
):
    system_prompt = (
        "You are a helpful multimodal assistant. You are required to answer the question based on the image provided. "
        "Put your final answer in \\boxed{LETTER}."
    )
    if "\\boxed" not in question_text:
        formatted_question = f"{question_text}\nAnswer with the option's letter directly in \\boxed{{LETTER}}."
    else:
        formatted_question = question_text
        
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": formatted_question}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
    gen_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
    raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    parsed_option = parse_boxed_option(raw_text, question_text)
    
    return {
        "raw_text": raw_text,
        "parsed_option": parsed_option,
        "generated_token_count": len(gen_tokens)
    }


def main():
    parser = argparse.ArgumentParser(description="Monet Ablation Benchmark Runner")
    parser.add_argument("--benchmark", type=str, default="vstar", choices=["vstar", "mmvp"])
    parser.add_argument("--model_mode", type=str, default="monet", choices=["monet", "qwen_base"])
    parser.add_argument("--quota", type=int, default=10, help="Latent quota K (for Monet)")
    parser.add_argument("--intervention", type=str, default="none", choices=["none", "zero", "noise", "donor"])
    parser.add_argument("--limit", type=int, default=None, help="Sample count limit (for rapid testing)")
    parser.add_argument("--output_dir", type=str, default="external_models/monet/ablation_study/outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tag = f"{args.benchmark}_{args.model_mode}_K{args.quota}_{args.intervention}"
    if args.limit:
        tag += f"_N{args.limit}"
    out_json_path = os.path.join(args.output_dir, f"{tag}.json")
    
    print("=" * 70)
    print(f"RUNNING MONET ABLATION: {tag}")
    print(f"Benchmark: {args.benchmark} | Model Mode: {args.model_mode} | Quota: {args.quota} | Intervention: {args.intervention}")
    print("=" * 70)
    
    # 1. Load Model
    if args.model_mode == "monet":
        model, processor, tokenizer = load_monet_model_and_processor(
            model_id_or_path="NOVAglow646/Monet-7B",
            device=args.device,
            torch_dtype=torch.bfloat16
        )
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"[Loader] Loading base Qwen2.5-VL-7B-Instruct onto {args.device}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=args.device
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
    # 2. Load Benchmark Samples
    samples = load_benchmark_data(args.benchmark, limit=args.limit)
    print(f"Loaded {len(samples)} evaluation samples.")
    
    donor_latents = None
    if args.intervention == "donor":
        donor_latents = [torch.randn(3584, device=args.device, dtype=torch.bfloat16) * 4.8 for _ in range(25)]
        
    predictions = []
    correct_count = 0
    t0 = time.time()
    
    for i, item in enumerate(tqdm(samples, desc=f"Evaluating {tag}")):
        try:
            raw_img = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error reading image {item['image_path']}: {e}")
            continue
            
        if args.model_mode == "monet":
            pred = generate_monet_prediction(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=raw_img,
                question_text=item["prompt"],
                latent_size=args.quota,
                intervention_mode=args.intervention,
                donor_latents=donor_latents,
                device=args.device
            )
        else:
            pred = generate_qwen_base_prediction(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=raw_img,
                question_text=item["prompt"],
                device=args.device
            )
            
        is_correct = (pred["parsed_option"] == item["ground_truth"])
        if is_correct:
            correct_count += 1
            
        record = {
            "id": item["id"],
            "category": item["category"],
            "ground_truth": item["ground_truth"],
            "parsed_option": pred["parsed_option"],
            "is_correct": is_correct,
            "raw_text": pred["raw_text"],
            "token_count": pred["generated_token_count"],
        }
        if "mean_cos_sim" in pred:
            record["mean_cos_sim"] = pred["mean_cos_sim"]
        predictions.append(record)
        
    elapsed = time.time() - t0
    acc = (correct_count / len(predictions)) * 100.0 if predictions else 0.0
    
    # MMVP pairwise accuracy
    pair_acc = None
    if args.benchmark == "mmvp" and len(predictions) > 0:
        pair_dict = {}
        for p in predictions:
            pair_key = p["category"]
            pair_dict.setdefault(pair_key, []).append(p["is_correct"])
        pair_correct = sum(1 for v in pair_dict.values() if len(v) == 2 and all(v))
        pair_acc = (pair_correct / len(pair_dict)) * 100.0
        
    summary = {
        "tag": tag,
        "benchmark": args.benchmark,
        "model_mode": args.model_mode,
        "quota": args.quota,
        "intervention": args.intervention,
        "num_evaluated": len(predictions),
        "correct_count": correct_count,
        "accuracy": acc,
        "pair_accuracy": pair_acc,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE FOR {tag}")
    print(f"Accuracy: {acc:.2f}% ({correct_count}/{len(predictions)})")
    if pair_acc is not None:
        print(f"MMVP Pairwise Accuracy: {pair_acc:.2f}%")
    print(f"Time Elapsed: {elapsed:.2f} s ({elapsed/len(predictions):.2f} s/sample)")
    print("=" * 70)
    
    final_payload = {
        "summary": summary,
        "predictions": predictions
    }
    
    with open(out_json_path, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"[Saved results to {out_json_path}]")


if __name__ == "__main__":
    main()
