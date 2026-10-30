#!/usr/bin/env bash
set -eo pipefail

cd /home/devraj/MTok
source .mtok/bin/activate

LOG_FILE="/home/devraj/MTok/wsdm_experiments/pipeline.log"
echo "=== WSDM EXPERIMENTS PIPELINE STARTED AT $(date) ===" | tee -a "$LOG_FILE"

# Step 1: W2 (Causal Knockout N=700) & W4 (Patching & Pre-Norm Geometry)
echo "[1/3] Running W2 & W4 Mechanistic experiments..." | tee -a "$LOG_FILE"
python wsdm_experiments/run_w2_w4_mechanistic.py 2>&1 | tee -a "$LOG_FILE"
echo "[1/3] W2 & W4 COMPLETE at $(date)" | tee -a "$LOG_FILE"

# Step 2: W1 (Lineage Audit N=700) & W3 (Cross-Benchmark Generalization N=1700)
echo "[2/3] Running W1 & W3 Lineage and Cross-Benchmark experiments..." | tee -a "$LOG_FILE"
python wsdm_experiments/run_w1_w3_lineage_and_benchmarks.py 2>&1 | tee -a "$LOG_FILE"
echo "[2/3] W1 & W3 COMPLETE at $(date)" | tee -a "$LOG_FILE"

# Step 3: Master Analysis & Aggregation (W1-W5)
echo "[3/3] Running Master Analyzer..." | tee -a "$LOG_FILE"
python wsdm_experiments/analyze_wsdm_all.py 2>&1 | tee -a "$LOG_FILE"
echo "[3/3] Master Analyzer COMPLETE at $(date)" | tee -a "$LOG_FILE"

echo "=== WSDM EXPERIMENTS PIPELINE ALL FINISHED AT $(date) ===" | tee -a "$LOG_FILE"

