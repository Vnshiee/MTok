import torch

print("--- LVR Diagnostic 2: Latent Span GRPO Zero-Gradient Verification ---")
# Simulating the token masking in grpo_trainer.py L740-L757
prompt_completion_ids = torch.tensor([[101, 151665, 151666, 151666, 151668, 202, 303]])
lvr_start_token_id = 151665
lvr_end_token_id = 151668
final_mask = torch.ones_like(prompt_completion_ids, dtype=torch.float32)

for i in range(prompt_completion_ids.size(0)):
    is_lvr = False
    for j in range(prompt_completion_ids.size(1)):
        if prompt_completion_ids[i, j] == lvr_start_token_id:
            is_lvr = True
            final_mask[i, j] = 0
        elif prompt_completion_ids[i, j] == lvr_end_token_id:
            is_lvr = False
            final_mask[i, j] = 0
        elif is_lvr:
            final_mask[i, j] = 0

print("Prompt Completion IDs:", prompt_completion_ids.tolist())
print("Resulting Policy Loss Mask:", final_mask.tolist())
assert final_mask[0, 1].item() == 0 and final_mask[0, 2].item() == 0 and final_mask[0, 3].item() == 0 and final_mask[0, 4].item() == 0
assert final_mask[0, 0].item() == 1 and final_mask[0, 5].item() == 1
print("Result: Verified that the entire LVR latent span receives exactly ZERO policy loss weight.")
