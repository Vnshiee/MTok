#!/bin/bash
set -uo pipefail

# ==============================================================================
# MASTER RUNNER: Experiments 2 to 5 Pipeline
#
# Experiment 2: Stage 2 SFT vs. Stage 3 GRPO Head-to-Head (array/Qwen2.5-VL-MullGRPO)
# Experiment 3: Confounding Data Leakage Test (Direct SAT SFT Baseline array/Qwen2.5-VL-SAT)
# Experiment 4: Thinking Token Budget Scaling Curve (K in {1, 5, 10, 40})
# Experiment 5: Latent Dynamics, Representational Drift & Cosine Geometry
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_DIR="${PROJECT_DIR}/mull/lmms-eval"
VENV_DIR="${PROJECT_DIR}/.mtok"
EVAL_DIR="${PROJECT_DIR}/eval_outputs"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HOME}/.cache/huggingface"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Check for dry-run or limit flags
LIMIT_ARG=""
EXP5_LIMIT_ARG=""
IS_DRY_RUN=false

for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        LIMIT_ARG="--limit 2"
        EXP5_LIMIT_ARG="--limit 2"
        IS_DRY_RUN=true
    elif [[ "$arg" == --limit* ]]; then
        LIMIT_ARG="$arg"
        EXP5_LIMIT_ARG="$arg"
        IS_DRY_RUN=true
    fi
done

LOG_FILE="${EVAL_DIR}/experiments_2_to_5_execution.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "STARTING EXPERIMENTS 2 TO 5 PIPELINE"
echo "Dry Run Mode: ${IS_DRY_RUN}"
echo "Limit Argument: ${LIMIT_ARG:-None (Full Run)}"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)"
echo "Log File: ${LOG_FILE}"
echo "=============================================================================="

TASKS=("blink_sprel" "blink_reldepth" "blink_jigsaw" "blink_mv" "blink_iqtest")
cd "${LMMS_EVAL_DIR}"

# ==============================================================================
# EXPERIMENT 2: Stage 2 SFT vs. Stage 3 GRPO Head-to-Head
# ==============================================================================
EXP2_DIR="${EVAL_DIR}/exp2_stage3_grpo"
mkdir -p "${EXP2_DIR}"
echo ""
echo "=============================================================================="
echo ">>> [EXPERIMENT 2] Evaluating Stage 3 GRPO Model (array/Qwen2.5-VL-MullGRPO)..."
echo "Directory: ${EXP2_DIR}"
echo "=============================================================================="

for TASK in "${TASKS[@]}"; do
    echo "  [Exp 2] Running Task: ${TASK} ..."
    "${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
        --model qwen2_5_vl_mmlatentdiscrete \
        --model_args "pretrained=array/Qwen2.5-VL-MullGRPO,max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False" \
        --gen_kwargs "prompt_mode=mmlatent2,prompt_version=new,num_latents=20" \
        --tasks "${TASK}" \
        --batch_size 1 \
        ${LIMIT_ARG} \
        --output_path "${EXP2_DIR}" || echo "Warning: [Exp 2] Task ${TASK} failed with exit code $?"
done

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_exp2_results.py"

# ==============================================================================
# EXPERIMENT 3: Confounding Data Leakage Test (Direct SAT SFT Baseline)
# ==============================================================================
EXP3_DIR="${EVAL_DIR}/exp3_sat_baseline"
mkdir -p "${EXP3_DIR}"
echo ""
echo "=============================================================================="
echo ">>> [EXPERIMENT 3] Evaluating Direct SAT SFT Baseline (array/Qwen2.5-VL-SAT)..."
echo "Directory: ${EXP3_DIR}"
echo "=============================================================================="

for TASK in "${TASKS[@]}"; do
    echo "  [Exp 3] Running Task: ${TASK} ..."
    "${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
        --model qwen2_5_vl_mmlatentdiscrete \
        --model_args "pretrained=array/Qwen2.5-VL-SAT,max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False" \
        --gen_kwargs "prompt_mode=direct,prompt_version=new,num_latents=0" \
        --tasks "${TASK}" \
        --batch_size 1 \
        ${LIMIT_ARG} \
        --output_path "${EXP3_DIR}" || echo "Warning: [Exp 3] Task ${TASK} failed with exit code $?"
done

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_exp3_results.py"

# ==============================================================================
# EXPERIMENT 4: Thinking Token Budget Scaling Curve (K in {1, 5, 10, 40})
# ==============================================================================
EXP4_DIR="${EVAL_DIR}/exp4_length_scaling"
mkdir -p "${EXP4_DIR}"
echo ""
echo "=============================================================================="
echo ">>> [EXPERIMENT 4] Evaluating Thinking Budget Scaling (K in {1, 5, 10, 40})..."
echo "Directory: ${EXP4_DIR}"
echo "=============================================================================="

K_VALUES=(1 5 10 40)
for K in "${K_VALUES[@]}"; do
    K_SUBDIR="${EXP4_DIR}/K_${K}"
    mkdir -p "${K_SUBDIR}"
    echo ""
    echo "  --- [Exp 4] Scaling Condition: K = ${K} Latents ---"
    for TASK in "${TASKS[@]}"; do
        echo "    [Exp 4 | K=${K}] Running Task: ${TASK} ..."
        "${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
            --model qwen2_5_vl_mmlatentdiscrete \
            --model_args "pretrained=array/Qwen2.5-VL-Mull,max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False" \
            --gen_kwargs "prompt_mode=mmlatent2,prompt_version=new,num_latents=${K}" \
            --tasks "${TASK}" \
            --batch_size 1 \
            ${LIMIT_ARG} \
            --output_path "${K_SUBDIR}" || echo "Warning: [Exp 4 | K=${K}] Task ${TASK} failed with exit code $?"
    done
done

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_exp4_results.py"

# ==============================================================================
# EXPERIMENT 5: Latent Representational Drift & Cosine Geometry
# ==============================================================================
EXP5_DIR="${EVAL_DIR}/exp5_latent_dynamics"
echo ""
echo "=============================================================================="
echo ">>> [EXPERIMENT 5] Running Latent Dynamics & Representation Drift Analysis..."
echo "Directory: ${EXP5_DIR}"
echo "=============================================================================="

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/experiment5_latent_dynamics.py" \
    --model_id "array/Qwen2.5-VL-Mull" \
    --output_dir "${EXP5_DIR}" \
    ${EXP5_LIMIT_ARG}

# ==============================================================================
# MASTER SUMMARY COMPILATION
# ==============================================================================
echo ""
echo "=============================================================================="
echo ">>> Compiling Master Summary for Experiments 1 through 5..."
echo "=============================================================================="
"${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_master_report.py"

echo ""
echo "=============================================================================="
echo "ALL EXPERIMENTS 2 TO 5 PIPELINE COMPLETE!"
echo "Finished at: $(date)"
echo "Master Summary saved to: ${EVAL_DIR}/ALL_EXPERIMENTS_MASTER_SUMMARY.md"
echo "=============================================================================="
