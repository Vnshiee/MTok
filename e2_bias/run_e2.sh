#!/usr/bin/env bash
# ==============================================================================
# run_e2.sh: Run E2 bias-vs-discrimination experiments
#
# Tier A: T1 (Mull Ref), T3 (Empty Think), T8 (SAT DirAns)
# Tier B: T9, T10 (GRPO), T4, T5, T7
# Tier C: T11-T14 (K-sweep), T2F, T6F
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.mtok"
E2_OUT="${PROJECT_DIR}/e2_bias/outputs"
LOG_FILE="${PROJECT_DIR}/e2_bias/run_e2.log"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${E2_OUT}"

echo "==============================================================================" | tee -a "${LOG_FILE}"
echo ">>> Starting Experiment E2: Bias vs Discrimination Evaluation" | tee -a "${LOG_FILE}"
echo "    Project Dir : ${PROJECT_DIR}" | tee -a "${LOG_FILE}"
echo "    Output Dir  : ${E2_OUT}" | tee -a "${LOG_FILE}"
echo "    Timestamp   : $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"

# Default to Tier A if no arguments passed
CONDS=("$@")
if [ ${#CONDS[@]} -eq 0 ]; then
    CONDS=("T1" "T3" "T8")
fi

echo ">>> Running conditions: ${CONDS[*]}" | tee -a "${LOG_FILE}"

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/e2_bias/score_letters.py" \
    --conds "${CONDS[@]}" \
    --out "${E2_OUT}" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo ">>> Running statistical analysis..." | tee -a "${LOG_FILE}"
"${VENV_DIR}/bin/python" "${PROJECT_DIR}/e2_bias/analyze_e2.py" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo ">>> Experiment E2 run completed successfully at $(date -Iseconds)" | tee -a "${LOG_FILE}"
