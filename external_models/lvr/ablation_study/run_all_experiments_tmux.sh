#!/usr/bin/env bash
# Master Automation Script for LVR Complete Empirical Experiments (E-L1 to E-L6)
# Runs the full ablation and mechanistic suite on GPU 0 inside a tmux session.

set -euo pipefail

ROOT_DIR="/home/devraj/MTok/external_models/lvr"
ABL_DIR="${ROOT_DIR}/ablation_study"
OUT_DIR="${ROOT_DIR}/ablation_outputs"
LOG_FILE="${OUT_DIR}/lvr_master_run.log"
PYTHON_BIN="/home/devraj/MTok/.mtok/bin/python"

mkdir -p "${OUT_DIR}"

echo "=================================================================" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] STARTING LVR COMPLETE EXPERIMENTAL PIPELINE (E-L1 to E-L6) ON GPU 0" | tee -a "${LOG_FILE}"
echo "Hardware: NVIDIA A100-PCIE-40GB" | tee -a "${LOG_FILE}"
echo "Output Directory: ${OUT_DIR}" | tee -a "${LOG_FILE}"
echo "Master Log File: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# --- PHASE 0: MATHEMATICAL & UNIT VERIFICATIONS ---
echo "" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] Phase 0: Running Unit & Mathematical Verifications..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/parsers_and_stats.py" 2>&1 | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/verify_interventions_unit.py" 2>&1 | tee -a "${LOG_FILE}"

# --- PHASE 1: EXPERIMENT E-L6 (RL REPLAY CONTEXT MISMATCH AUDIT) ---
echo "" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] Phase 1: Running Experiment E-L6 (RL Replay Context Mismatch on 200 Rollouts)..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/run_e_l6_rl_replay_audit.py" \
    --num_samples 200 \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

# --- PHASE 2: EXPERIMENTS E-L1 to E-L5 ON V* BENCH ---
echo "" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] Phase 2: Running Full V* Bench Ablation Matrix (E-L1 to E-L5 + Attention Telemetry)..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/run_ablation_suite.py" \
    --benchmarks vstar \
    --axes axis1a axis3a axis3b axis4 axis5 axis2_base \
    --collect_attention_telemetry \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u "${ABL_DIR}/summarize_ablation_results.py" 2>&1 | tee -a "${LOG_FILE}"

# --- PHASE 3: EXPERIMENTS E-L1, E-L2, E-L5 ON BLINK BENCH ---
echo "" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] Phase 3: Running Full BLINK Bench Ablation Matrix..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/run_ablation_suite.py" \
    --benchmarks blink \
    --axes axis1a axis3a axis3b axis4 axis5 axis2_base \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u "${ABL_DIR}/summarize_ablation_results.py" 2>&1 | tee -a "${LOG_FILE}"

# --- PHASE 4: EXPERIMENTS ON MMVP BENCH ---
echo "" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] Phase 4: Running Full MMVP Bench Ablation Matrix..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/run_ablation_suite.py" \
    --benchmarks mmvp \
    --axes axis1a axis3a axis3b axis4 axis5 axis2_base \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

# --- PHASE 5: COMPILE MASTER SYNTHESIS & HYPOTHESIS DECISION VERDICTS ---
echo "" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] Phase 5: Compiling Final Publication-Grade Statistical Synthesis..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/summarize_ablation_results.py" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] ALL LVR COMPLETE EXPERIMENTS (E-L1 to E-L6) COMPLETED SUCCESSFULLY!" | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"
