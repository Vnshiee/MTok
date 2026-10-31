#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/devraj/MTok/external_models/lvr"
ABL_DIR="${ROOT_DIR}/ablation_study"
OUT_DIR="${ROOT_DIR}/ablation_outputs"
LOG_FILE="${OUT_DIR}/ablation_master.log"
PYTHON_BIN="/home/devraj/MTok/.mtok/bin/python"

mkdir -p "${OUT_DIR}"

echo "=================================================================" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] Starting LVR Rigorous Ablation Suite on GPU 0" | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Phase 0: Running Unit & Mathematical Verifications..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/parsers_and_stats.py" 2>&1 | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/training_and_rl_verification.py" 2>&1 | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/verify_interventions_unit.py" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date -Is)] Phase 1: Running Full V* Bench Ablations (All Axes + Attention Telemetry)..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/run_ablation_suite.py" \
    --benchmarks vstar \
    --axes axis1a axis3a axis3b axis4 axis5 axis2_base \
    --collect_attention_telemetry \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u "${ABL_DIR}/summarize_ablation_results.py" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date -Is)] Phase 2: Running Full MMVP Bench Ablations (All Axes)..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/run_ablation_suite.py" \
    --benchmarks mmvp \
    --axes axis1a axis3a axis3b axis4 axis5 axis2_base \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u "${ABL_DIR}/summarize_ablation_results.py" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date -Is)] Phase 3: Running Full BLINK Bench Ablations (All Axes)..." | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u "${ABL_DIR}/run_ablation_suite.py" \
    --benchmarks blink \
    --axes axis1a axis3a axis3b axis4 axis5 axis2_base \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u "${ABL_DIR}/summarize_ablation_results.py" 2>&1 | tee -a "${LOG_FILE}"

echo "=================================================================" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] ALL LVR ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!" | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"

