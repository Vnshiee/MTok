import re
import os

print("=== FORENSIC AUDIT: MONET (arXiv:2511.21395) ===")

# 1. Inspect VLPO Policy Gradient calculation in repo/RL
rl_files = [
    "/home/devraj/MTok/external_models/monet/repo/RL/monet_rl_patch.py",
    "/home/devraj/MTok/external_models/monet/repo/RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py",
    "/home/devraj/MTok/external_models/monet/repo/RL/monet_models/vllm/latent_hook.py"
]

for r in rl_files:
    if os.path.exists(r):
        print(f"\n--- Checking {os.path.basename(r)} ---")
        with open(r, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        print("  Size:", len(code), "lines:", len(code.splitlines()))
        
        # Check for policy gradient / log_prob / detach
        for term in ["log_prob", "logp", "detach", "gaussian", "perturb", "policy", "reward", "advantage"]:
            matches = len(re.findall(r"\b" + term + r"\b", code, re.I))
            print(f"  Occurrences of '{term}': {matches}")

# 2. Inspect Student vs Teacher Distillation in src/trainer.py
trainer_path = "/home/devraj/MTok/external_models/monet/repo/src/trainer.py"
if os.path.exists(trainer_path):
    print(f"\n--- Checking {trainer_path} ---")
    with open(trainer_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    print("  Lines:", len(code.splitlines()))
    for term in ["distill", "teacher", "mse", "l2", "loss", "detach"]:
        matches = len(re.findall(r"\b" + term + r"\b", code, re.I))
        print(f"  Occurrences of '{term}': {matches}")

