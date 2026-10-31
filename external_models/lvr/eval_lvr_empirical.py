import sys
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer

# Add repo to python path
sys.path.append("/home/devraj/MTok/external_models/lvr/repo")
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from src.model.qwen_lvr_model import QwenWithLVR

print("=== LVR Empirical Evaluation & Mechanistic Testing ===")
print("CUDA Available:", torch.cuda.is_available(), "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Activate LVR forward patch for inference
replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=False, rl=False)

model_id = "vincentleebang/LVR-7B"
out_dir = "/home/devraj/MTok/external_models/lvr/experiment_outputs"
os.makedirs(out_dir, exist_ok=True)

print(f"Loading {model_id} onto GPU 0...")
processor = AutoProcessor.from_pretrained(model_id)
model = QwenWithLVR.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
model.eval()

print("Model successfully loaded onto GPU!")

# Mechanistic Test 1: Testing Cosine Similarity on Spatial Test Image
from PIL import Image
test_img_path = "/home/devraj/MTok/spatial_test.png"
if os.path.exists(test_img_path):
    print("\n--- Running Empirical Test: Latent Hidden State vs. Input ViT Patches ---")
    image = Image.open(test_img_path)
    
    question = "Which shape is located in the middle, between the red square and the green triangle?"
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question + "\nAnswer with the option's letter from the given choices directly."}
        ]}
    ]
    
    from qwen_vl_utils import process_vision_info
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            decoding_strategy="steps",
            lvr_steps=[5]
        )
    
    response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print("Generated LVR Response:")
    print(response)
    
    # Save output log
    with open(os.path.join(out_dir, "lvr_single_sample_response.json"), "w") as f:
        json.dump({"prompt": prompt, "response": response}, f, indent=2)

print("Evaluation run completed.")
