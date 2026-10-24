#!/bin/bash
set -uo pipefail

# ==============================================================================
# EXPERIMENT 1: Causal Necessity and Latent Token Ablation Suite (BLINK Suite)
#
# Tests whether Mull-Tokens (<|latent_pad|>) perform real causal reasoning or
# function as an ungrounded shortcut / placebo on the 5 core BLINK benchmarks.
#
# Conditions:
#   1. Diagnostic: Attention Allocation Analysis (already done)
#   2. Condition 2: Zero Latents (K=0, Direct Answering without <think> tags)
#   3. Condition 3: Zeroed Latent Embeddings (K=20, <|latent_pad|> embedding = 0)
#   4. Condition 4: Random Noise Embeddings (K=20, <|latent_pad|> ~ N(0, sigma^2))
#   5. Aggregation: Generate Consolidated Markdown Report with Delta Comparisons
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_DIR="${PROJECT_DIR}/mull/lmms-eval"
VENV_DIR="${PROJECT_DIR}/.mtok"
EXP_DIR="${PROJECT_DIR}/eval_outputs/exp1_causal_ablation"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HOME}/.cache/huggingface"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${EXP_DIR}"
LOG_FILE="${EXP_DIR}/experiment1_execution.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "STARTING EXPERIMENT 1: Causal Necessity & Latent Ablation Suite"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)"
echo "Output Directory: ${EXP_DIR}"
echo "=============================================================================="

# The 5 Core BLINK benchmarks where Mull claims its reasoning gains (700 items total)
TASKS=("blink_sprel" "blink_reldepth" "blink_jigsaw" "blink_mv" "blink_iqtest")

cd "${LMMS_EVAL_DIR}"

# ------------------------------------------------------------------------------
# STEP 1: Diagnostic Attention Attribution (if not already done)
# ------------------------------------------------------------------------------
ATTN_JSON="${EXP_DIR}/exp1_attention_distribution.json"
if [ ! -f "${ATTN_JSON}" ]; then
    echo ""
    echo "=============================================================================="
    echo "[STEP 1/5] Running Mechanistic Attention Allocation Analysis..."
    echo "=============================================================================="
    "${VENV_DIR}/bin/python" "${PROJECT_DIR}/experiment1_attention_diagnostic.py" "${ATTN_JSON}"
else
    echo "[STEP 1/5] Attention attribution already exists at ${ATTN_JSON}. Skipping."
fi

# ------------------------------------------------------------------------------
# STEP 2: Condition 2 - Zero Latents (K=0 / Direct Answering)
# ------------------------------------------------------------------------------
COND_K0_DIR="${EXP_DIR}/cond_zero_latents_K0"
mkdir -p "${COND_K0_DIR}"
echo ""
echo "=============================================================================="
echo "[STEP 2/5] Running Condition 2: Zero Latents (K=0, Direct Answering)"
echo "Tasks: ${TASKS[*]}"
echo "Directory: ${COND_K0_DIR}"
echo "=============================================================================="

for TASK in "${TASKS[@]}"; do
    echo ">>> [K=0] Evaluating Task: ${TASK} ..."
    "${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
        --model qwen2_5_vl_mmlatentdiscrete \
        --model_args "pretrained=array/Qwen2.5-VL-Mull,max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False" \
        --gen_kwargs "prompt_mode=mmlatent2,prompt_version=new,num_latents=0" \
        --tasks "${TASK}" \
        --batch_size 1 \
        --output_path "${COND_K0_DIR}" || echo "Warning: Task ${TASK} failed with exit code $?"
done

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_exp1_results.py"

# ------------------------------------------------------------------------------
# STEP 3: Condition 3 - Zeroed Latent Embeddings (K=20, Embedding = 0)
# ------------------------------------------------------------------------------
COND_ZEROS_DIR="${EXP_DIR}/cond_zeroed_embeddings"
mkdir -p "${COND_ZEROS_DIR}"
echo ""
echo "=============================================================================="
echo "[STEP 3/5] Running Condition 3: Zeroed Latent Embeddings (K=20, Input Vector = 0)"
echo "Tasks: ${TASKS[*]}"
echo "Directory: ${COND_ZEROS_DIR}"
echo "=============================================================================="

for TASK in "${TASKS[@]}"; do
    echo ">>> [Zeroed Embeddings] Evaluating Task: ${TASK} ..."
    "${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
        --model qwen2_5_vl_mmlatentdiscrete \
        --model_args "pretrained=array/Qwen2.5-VL-Mull,max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False,ablation_mode=zeros" \
        --gen_kwargs "prompt_mode=mmlatent2,prompt_version=new,num_latents=20" \
        --tasks "${TASK}" \
        --batch_size 1 \
        --output_path "${COND_ZEROS_DIR}" || echo "Warning: Task ${TASK} failed with exit code $?"
done

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_exp1_results.py"

# ------------------------------------------------------------------------------
# STEP 4: Condition 4 - Random Gaussian Noise Embeddings (K=20, ~ N(0, sigma^2))
# ------------------------------------------------------------------------------
COND_NOISE_DIR="${EXP_DIR}/cond_random_noise_embeddings"
mkdir -p "${COND_NOISE_DIR}"
echo ""
echo "=============================================================================="
echo "[STEP 4/5] Running Condition 4: Random Noise Latent Embeddings (K=20, Input ~ Gaussian Noise)"
echo "Tasks: ${TASKS[*]}"
echo "Directory: ${COND_NOISE_DIR}"
echo "=============================================================================="

for TASK in "${TASKS[@]}"; do
    echo ">>> [Random Noise] Evaluating Task: ${TASK} ..."
    "${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
        --model qwen2_5_vl_mmlatentdiscrete \
        --model_args "pretrained=array/Qwen2.5-VL-Mull,max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False,ablation_mode=random_noise" \
        --gen_kwargs "prompt_mode=mmlatent2,prompt_version=new,num_latents=20" \
        --tasks "${TASK}" \
        --batch_size 1 \
        --output_path "${COND_NOISE_DIR}" || echo "Warning: Task ${TASK} failed with exit code $?"
done

# ------------------------------------------------------------------------------
# STEP 5: Final Report Compilation
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "[STEP 5/5] Generating Final Consolidated Experiment 1 Report..."
echo "=============================================================================="
"${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_exp1_results.py"

echo ""
echo "=============================================================================="
echo "EXPERIMENT 1 COMPLETE!"
echo "Final report saved to: ${EXP_DIR}/EXPERIMENT1_RESULTS_SUMMARY.md"
echo "Finished at: $(date)"
echo "=============================================================================="
