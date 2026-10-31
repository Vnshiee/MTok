#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

LOG_DIR="/home/devraj/MTok/external_models/lvr/experiment_logs"
mkdir -p "$LOG_DIR"
EXEC_LOG="$LOG_DIR/lvr_execution_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$EXEC_LOG") 2>&1

echo "================================================================================"
echo "STARTING COMPLETE EMPIRICAL AUDIT PIPELINE FOR LVR (arXiv:2509.24251)"
echo "Timestamp: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Log File: $EXEC_LOG"
echo "================================================================================"

# 1. Run CPU Architectural Diagnostics
echo "[Phase 1/3] Running CPU Verification Diagnostics..."
/home/devraj/MTok/.mtok/bin/python /home/devraj/MTok/external_models/lvr/diagnostics/test_roi_copying.py
/home/devraj/MTok/.mtok/bin/python /home/devraj/MTok/external_models/lvr/diagnostics/test_grpo_zero_grad.py
echo ">>> Phase 1 Diagnostics Certified: Identity Copying & GRPO Zero-Gradient confirmed."

# 2. Setup Dedicated Evaluation & Diagnostic Script
echo "[Phase 2/3] Preparing LVR-7B Model Weights & Inference Suite..."
/home/devraj/MTok/.mtok/bin/python -u /home/devraj/MTok/external_models/lvr/eval_lvr_empirical.py

echo "[Phase 3/3] Generating Empirical Summary Report..."
/home/devraj/MTok/.mtok/bin/python -u /home/devraj/MTok/external_models/lvr/summarize_lvr_experiments.py

echo "================================================================================"
echo "LVR EMPIRICAL RESEARCH PIPELINE COMPLETED SUCCESSFULLY"
echo "Timestamp: $(date)"
echo "================================================================================"
