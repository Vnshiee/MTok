"""
Rigorous GPU Verification of `lvr_interventions.py` on `vincentleebang/LVR-7B`.

Verifies:
1. Bit-exact token equivalence between `run_lvr_intervention_single(condition="A1A-BASE", lvr_steps=K)`
   and `model.generate(decoding_strategy="steps", lvr_steps=[K])` for K in {4, 8}.
2. Exact zero attention weight (`0.000000`) on masked KV positions under:
   - `A1A-MASK-ALL-IMG`: `latent_attn_to_vis_mean == 0.0`
   - `A1A-MASK-ANS-TO-LVR`: `ans_attn_to_lvr_mean == 0.0`
   - `A1A-MASK-ANS-TO-IMG`: `ans_attn_to_vis_mean == 0.0`
3. Execution & telemetry across all Axis 1A, Axis 3, and Axis 4 conditions on V* Sample 0.
"""

import os
import sys
import torch
from transformers import AutoConfig, AutoProcessor

ROOT_DIR = "/home/devraj/MTok/external_models/lvr"
REPO_DIR = os.path.join(ROOT_DIR, "repo")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, REPO_DIR)

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from qwen_vl_utils import process_vision_info

from ablation_study.data_loader import load_vstar_bench
from ablation_study.lvr_interventions import create_qwen_messages, run_lvr_intervention_single


def main():
    model_id = "vincentleebang/LVR-7B"
    config = AutoConfig.from_pretrained(model_id)
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True, lvr_head=config.lvr_head, rl=False
    )

    print(f"Loading {model_id} with attn_implementation='sdpa'...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = QwenWithLVR.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()

    vstar = load_vstar_bench()
    sample = vstar[0]
    sample_donor = vstar[1]
    print(f"Testing on V* Sample 0: id={sample['question_id']}, category={sample['category']}, label={sample['label']}, bboxes_norm={sample['bboxes_norm']}")

    # 1. Check exact equivalence against official `model.generate`
    for k_test in [0, 4, 8]:
        messages = create_qwen_messages(sample["image"], sample["text_official_eval"])
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            official_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                decoding_strategy="steps",
                lvr_steps=[k_test],
            )
        official_gen_ids = official_ids[0, inputs.input_ids.shape[1]:].cpu().tolist()

        our_res = run_lvr_intervention_single(
            model=model,
            processor=processor,
            img_input=sample["image"],
            prompt_text=sample["text_official_eval"],
            condition="A1A-BASE",
            lvr_steps=k_test,
            max_new_tokens=64,
            bboxes_norm=sample["bboxes_norm"],
            collect_attention_telemetry=False,
        )
        our_gen_ids = our_res["generated_token_ids"]
        print(f"[Equivalence K={k_test}] Official={official_gen_ids} vs Ours={our_gen_ids}")
        assert official_gen_ids == our_gen_ids, f"Token mismatch at K={k_test}: {official_gen_ids} vs {our_gen_ids}"

    print(">>> PASS 1: Exact token-for-token equivalence with official `model.generate` verified!")

    # 2. Verify exact 0.0 attention weight on masked KV positions AND bit-exactness with collect_attention_telemetry=False
    donor_res = run_lvr_intervention_single(
        model=model,
        processor=processor,
        img_input=sample_donor["image"],
        prompt_text=sample_donor["text_official_eval"],
        condition="A1A-BASE",
        lvr_steps=8,
        max_new_tokens=64,
        bboxes_norm=sample_donor["bboxes_norm"],
        collect_attention_telemetry=False,
    )
    donor_states = donor_res["raw_latent_states"]

    for mask_cond in ["A1A-BASE", "A1A-MASK-ALL-IMG", "A1A-MASK-ANS-TO-LVR", "A1A-MASK-ANS-TO-IMG"]:
        res_no_tel = run_lvr_intervention_single(
            model=model,
            processor=processor,
            img_input=sample["image"],
            prompt_text=sample["text_official_eval"],
            condition=mask_cond,
            lvr_steps=8,
            max_new_tokens=64,
            bboxes_norm=sample["bboxes_norm"],
            collect_attention_telemetry=False,
        )
        res = run_lvr_intervention_single(
            model=model,
            processor=processor,
            img_input=sample["image"],
            prompt_text=sample["text_official_eval"],
            condition=mask_cond,
            lvr_steps=8,
            max_new_tokens=64,
            bboxes_norm=sample["bboxes_norm"],
            collect_attention_telemetry=True,
        )
        assert res["generated_token_ids"] == res_no_tel["generated_token_ids"], (
            f"Telemetry perturbed generation under {mask_cond}: {res['generated_token_ids']} vs {res_no_tel['generated_token_ids']}"
        )
        tel = res["telemetry"]
        print(
            f"[{mask_cond}] output={res['output_text']!r} | "
            f"latent_attn_vis={tel['latent_attn_to_vis_mean']} | "
            f"ans_attn_lvr={tel['ans_attn_to_lvr_mean']} | "
            f"ans_attn_vis={tel['ans_attn_to_vis_mean']}"
        )
        if mask_cond == "A1A-MASK-ALL-IMG":
            assert tel["latent_attn_to_vis_mean"] == 0.0, f"Expected 0.0, got {tel['latent_attn_to_vis_mean']}"
        elif mask_cond == "A1A-MASK-ANS-TO-LVR":
            assert tel["ans_attn_to_lvr_mean"] == 0.0, f"Expected 0.0, got {tel['ans_attn_to_lvr_mean']}"
        elif mask_cond == "A1A-MASK-ANS-TO-IMG":
            assert tel["ans_attn_to_vis_mean"] == 0.0, f"Expected 0.0, got {tel['ans_attn_to_vis_mean']}"

    print(">>> PASS 2: Exact 0.0 attention masking AND bit-exact non-invasive telemetry verified!")

    # 3. Verify remaining Axis 1A, Axis 3, and Axis 4 conditions
    all_other_conds = [
        ("A1A-ORACLE-ROI", 8),
        ("A1A-ORACLE-BKG", 8),
        ("A1A-2D-MROPE", 8),
        ("A3A-STEPS-0", 0),
        ("A3A-STEPS-1", 1),
        ("A3A-STEPS-16", 16),
        ("A3B-FORCED-END", 8),
        ("A3B-VANILLA-LOGIT", 8),
        ("A3B-LATENT-END", 8),
        ("C2-EMPTY-TAG", 0),
        ("C3-DISCRETE-LVR", 8),
        ("C4-MEAN-IMG-POOL", 8),
        ("C5-NN-QUANT-IMG", 8),
        ("C6-ZERO-VEC", 8),
        ("C7-GAUSS-NOISE", 8),
        ("C8-SHUFFLE-BATCH", 8),
        ("C9-DIRECT-ANS", 0),
    ]
    for cond, k_val in all_other_conds:
        res = run_lvr_intervention_single(
            model=model,
            processor=processor,
            img_input=sample["image"],
            prompt_text=sample["text_official_eval"],
            condition=cond,
            lvr_steps=k_val,
            max_new_tokens=64,
            bboxes_norm=sample["bboxes_norm"],
            donor_latent_states=donor_states,
            collect_attention_telemetry=False,
        )
        tel = res["telemetry"]
        print(
            f"[{cond:18s} K={k_val:2d}] actual_K={tel['actual_latent_steps']:2d} | "
            f"cs_max={tel['cs_max_mean']} | cs_roi={tel['cs_roi_mean']} | "
            f"erank={tel['effective_rank']} | out={res['output_text']!r}"
        )

    print(">>> PASS 3: All Axis 1A, Axis 3, and Axis 4 conditions verified on GPU!")


if __name__ == "__main__":
    main()
