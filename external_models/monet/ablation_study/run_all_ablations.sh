#!/bin/bash
# Master Automation Script for Monet Ablation Suite
# Runs all preregistered ablation experiments inside tmux session

set -e
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="/home/devraj/MTok/.mtok/bin/python"
SCRIPT="external_models/monet/ablation_study/run_monet_ablation.py"
OUTPUT_DIR="external_models/monet/ablation_study/outputs"
LOG_DIR="external_models/monet/ablation_study/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "======================================================================"
echo "STARTING FULL MONET ABLATION BENCHMARK SUITE ON GPU 0"
echo "Date: $(date)"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================================================"

# --- PHASE 1: V* BENCH BENCHMARK & CAUSAL INTERVENTIONS (EXP-1.0 to EXP-1.3) ---
echo ""
echo ">>> [1/7] Running EXP-1.0: V* Bench Monet K=10 Natural (Baseline)..."
$PYTHON_BIN $SCRIPT --benchmark vstar --model_mode monet --quota 10 --intervention none --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/exp1_0_vstar_natural.log"

echo ""
echo ">>> [2/7] Running EXP-1.1: V* Bench Monet K=10 Zero Vector Ablation (C1)..."
$PYTHON_BIN $SCRIPT --benchmark vstar --model_mode monet --quota 10 --intervention zero --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/exp1_1_vstar_zero.log"

echo ""
echo ">>> [3/7] Running EXP-1.2: V* Bench Monet K=10 Gaussian Noise Ablation (C2)..."
$PYTHON_BIN $SCRIPT --benchmark vstar --model_mode monet --quota 10 --intervention noise --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/exp1_2_vstar_noise.log"

echo ""
echo ">>> [4/7] Running EXP-1.3: V* Bench Monet K=10 Donor Swapping Ablation (C3)..."
$PYTHON_BIN $SCRIPT --benchmark vstar --model_mode monet --quota 10 --intervention donor --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/exp1_3_vstar_donor.log"

# --- PHASE 2: LATENT QUOTA SWEEP ON V* BENCH (EXP-2.0 to EXP-2.3) ---
echo ""
echo ">>> [5/7] Running EXP-2.0: V* Bench Monet K=0 (No Latents / Direct Answer)..."
$PYTHON_BIN $SCRIPT --benchmark vstar --model_mode monet --quota 0 --intervention none --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/exp2_0_vstar_k0.log"

echo ""
echo ">>> [6/7] Running EXP-2.1: V* Bench Monet K=1 (Minimal Latent Quota)..."
$PYTHON_BIN $SCRIPT --benchmark vstar --model_mode monet --quota 1 --intervention none --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/exp2_1_vstar_k1.log"

echo ""
echo ">>> [7/7] Running EXP-2.2: V* Bench Monet K=5 (Medium Latent Quota)..."
$PYTHON_BIN $SCRIPT --benchmark vstar --model_mode monet --quota 5 --intervention none --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/exp2_2_vstar_k5.log"

echo "======================================================================"
echo "ALL MONET ABLATION BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Date: $(date)"
echo "======================================================================"
