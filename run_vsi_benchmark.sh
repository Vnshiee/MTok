#!/bin/bash
set -euo pipefail

# ==============================================================================
# Mull-Tokens (array/Qwen2.5-VL-Mull) VSI-Bench Evaluation
#
# Evaluates on:
#   VSI-Bench (Video Spatial Reasoning) - 5,130 test instances
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_DIR="${PROJECT_DIR}/mull/lmms-eval"
VENV_DIR="${PROJECT_DIR}/.mtok"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HOME}/.cache/huggingface"
export WANDB_MODE="offline"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/eval_outputs/vsibench_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

TASKS="vsibench"
LIMIT_ARG=""
if [ $# -ge 1 ]; then
    LIMIT_ARG="--limit $1"
fi

MODEL_ARGS="pretrained=array/Qwen2.5-VL-Mull,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False"
GEN_KWARGS="prompt_mode=mmlatent2,prompt_version=new,num_latents=20"

echo "=============================================================================="
echo "Starting Mull-Tokens Evaluation: VSI-Bench Video Spatial Reasoning"
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
echo "VSI-Bench Evaluation Completed Successfully!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "=============================================================================="
