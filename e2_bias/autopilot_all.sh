#!/usr/bin/env bash
# ==============================================================================
# autopilot_all.sh: Full autopilot runner for all 14 conditions of Experiment E2
# Runs detached in background, automatically waiting for active runs, executing
# remaining conditions, computing statistical summaries, and pushing to GitHub.
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.mtok"
E2_OUT="${PROJECT_DIR}/e2_bias/outputs"
LOG_FILE="${PROJECT_DIR}/e2_bias/autopilot.log"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${E2_OUT}"

echo "==============================================================================" >> "${LOG_FILE}"
echo ">>> Autopilot started at $(date -Iseconds)" >> "${LOG_FILE}"
echo "==============================================================================" >> "${LOG_FILE}"

# 1. Wait for any currently running score_letters.py process to finish
while pgrep -f "score_letters.py" > /dev/null; do
    echo "[$(date +'%H:%M:%S')] Active score_letters.py detected. Waiting 30s..." >> "${LOG_FILE}"
    sleep 30
done

echo "[$(date +'%H:%M:%S')] Prior process completed. Commencing full 14-condition execution suite..." >> "${LOG_FILE}"

# All 14 conditions. score_letters.py skips any condition that has a COMPLETE marker.
ALL_CONDS=("T1" "T3" "T8" "T9" "T10" "T4" "T5" "T7" "T11" "T12" "T13" "T14" "T2F" "T6F")

echo "[$(date +'%H:%M:%S')] Condition suite target: ${ALL_CONDS[*]}" >> "${LOG_FILE}"
"${VENV_DIR}/bin/python" "${PROJECT_DIR}/e2_bias/score_letters.py" \
    --conds "${ALL_CONDS[@]}" \
    --out "${E2_OUT}" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date +'%H:%M:%S')] All 14 conditions finished! Running master statistical analysis..." >> "${LOG_FILE}"
"${VENV_DIR}/bin/python" "${PROJECT_DIR}/e2_bias/analyze_e2.py" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date +'%H:%M:%S')] Auto-committing and pushing all results and summaries to GitHub..." >> "${LOG_FILE}"
cd "${PROJECT_DIR}"
git add e2_bias/outputs/*/COMPLETE e2_bias/outputs/*/*.json e2_bias/outputs/*/*.jsonl e2_bias/E2_SUMMARY.md e2_bias/e2_results.json
git commit -m "feat(e2): complete full 14-condition bias-vs-discrimination evaluation and statistical matrix" || true
git push origin main || true

echo "==============================================================================" >> "${LOG_FILE}"
echo ">>> Autopilot completed successfully at $(date -Iseconds)!" >> "${LOG_FILE}"
echo "==============================================================================" >> "${LOG_FILE}"
