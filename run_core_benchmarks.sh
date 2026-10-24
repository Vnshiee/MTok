#!/bin/bash
set -euo pipefail

# ==============================================================================
# Mull-Tokens (array/Qwen2.5-VL-Mull) Core Image & Spatial Benchmarks Evaluation
#
# Evaluates on:
#   1. BLINK Spatial Relation (blink_sprel)
#   2. BLINK IQ Test (blink_iqtest)
#   3. BLINK Multi-view Reasoning (blink_mv)
#   4. BLINK Relative Depth (blink_reldepth)
#   5. BLINK Jigsaw (blink_jigsaw)
#   6. SAT Real-world (sat_real)
#   7. ERQA Egocentric Spatial (erqa)
#   8. MMSI-Bench Multi-image Spatial (mmsi_bench)
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_DIR="${PROJECT_DIR}/mull/lmms-eval"
VENV_DIR="${PROJECT_DIR}/.mtok"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HOME}/.cache/huggingface"
export WANDB_MODE="offline"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/eval_outputs/core_benchmarks_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

TASKS="blink_sprel,blink_iqtest,blink_mv,blink_reldepth,blink_jigsaw,sat_real,erqa,mmsi_bench"
# Allow overriding tasks via first argument
if [ $# -ge 1 ]; then
    TASKS="$1"
fi

# Optional limit for testing (second argument)
LIMIT_ARG=""
if [ $# -ge 2 ]; then
    LIMIT_ARG="--limit $2"
fi

MODEL_ARGS="pretrained=array/Qwen2.5-VL-Mull,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False"
GEN_KWARGS="prompt_mode=mmlatent2,prompt_version=new,num_latents=20"

echo "=============================================================================="
echo "Starting Mull-Tokens Evaluation: Core Image & Spatial Benchmarks"
echo "Tasks: ${TASKS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Gen kwargs: ${GEN_KWARGS}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================================================================="

cd "${LMMS_EVAL_DIR}"

"${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
    --model qwen2_5_vl_mmlatentdiscrete \
    --model_args "${MODEL_ARGS}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    ${LIMIT_ARG} \
    --output_path "${OUTPUT_DIR}"

echo "=============================================================================="
echo "Evaluation Completed Successfully!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "=============================================================================="
