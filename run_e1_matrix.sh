#!/usr/bin/env bash
# ==============================================================================
# run_e1_matrix.sh: Run the E1 Experimental Matrix (BLINK, 5 tasks, 9 conditions)
#
# Standardized at 4.19 MP (max_pixels=4194304) with --log_samples and per-sample
# latency, token ID, and generated length logging.
# ==============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.mtok"
EVAL_BASE="${PROJECT_DIR}/eval_outputs/e1_matrix"

TASKS=("blink_sprel" "blink_reldepth" "blink_jigsaw" "blink_mv" "blink_iqtest")
CUSTOM_TASKS=()

# Parse CLI flags
LIMIT_ARG=""
SELECTED_ROWS=()
IS_DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            LIMIT_ARG="--limit 1"
            IS_DRY_RUN=true
            shift
            ;;
        --limit)
            LIMIT_ARG="--limit $2"
            shift 2
            ;;
        --row)
            SELECTED_ROWS+=("$2")
            shift 2
            ;;
        --task)
            CUSTOM_TASKS+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ${#CUSTOM_TASKS[@]} -gt 0 ]; then
    TASKS=("${CUSTOM_TASKS[@]}")
fi

if [ ${#SELECTED_ROWS[@]} -eq 0 ]; then
    SELECTED_ROWS=(1 2 3 4 5 6 7 8 9)
fi

echo "=============================================================================="
echo ">>> Starting E1 Experimental Matrix Evaluation"
echo "    Project Dir : ${PROJECT_DIR}"
echo "    Virtualenv  : ${VENV_DIR}"
echo "    Output Dir  : ${EVAL_BASE}"
echo "    Selected Rows: ${SELECTED_ROWS[*]}"
echo "    Limit Arg   : ${LIMIT_ARG:-None (Full N=700)}"
echo "=============================================================================="

run_row() {
    local row_id="$1"
    local model_id="$2"
    local prompt_mode="$3"
    local custom_prefill="$4"
    local num_latents="$5"
    local dir_name="$6"

    local row_out="${EVAL_BASE}/${dir_name}"
    if [ "$IS_DRY_RUN" = true ]; then
        row_out="${EVAL_BASE}/smoke_dryrun/${dir_name}"
    fi

    local sample_log_dir="${row_out}/per_sample_logs"
    mkdir -p "${row_out}" "${sample_log_dir}"

    echo ""
    echo "=============================================================================="
    echo ">>> Running Row ${row_id}: ${dir_name}"
    echo "    Model: ${model_id} | Prompt: ${prompt_mode} | Prefill: ${custom_prefill} (K=${num_latents})"
    echo "    Output: ${row_out}"
    echo "=============================================================================="

    export BLINK_PROMPT_MODE="${prompt_mode}"
    export LMMS_EVAL_SAMPLE_LOG_DIR="${sample_log_dir}"

    for task in "${TASKS[@]}"; do
        # Check if task is already completed with valid sample count
        if [ "$IS_DRY_RUN" = false ] && [ -d "${row_out}" ]; then
            IS_DONE=$("${VENV_DIR}/bin/python" -c "
import sys
from eval_metric_utils import extract_blink_metrics
try:
    scores = extract_blink_metrics('${row_out}', require_all=False)
    print('YES' if '${task}' in scores else 'NO')
except Exception:
    print('NO')
" 2>/dev/null || echo "NO")

            if [ "$IS_DONE" = "YES" ]; then
                echo "  [Row ${row_id}] Task ${task} is already completed with full sample count. Skipping!"
                continue
            fi
        fi

        # If task is not yet completed, clean up any partial per_sample log to avoid duplicate entries
        local task_sample_file="${sample_log_dir}/per_sample_details_${task}.jsonl"
        if [ -f "${task_sample_file}" ]; then
            echo "  [Row ${row_id}] Cleaning incomplete sample log before restart: ${task_sample_file}"
            rm -f "${task_sample_file}"
        fi

        echo "  [Row ${row_id}] Executing Task: ${task} ..."
        "${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
            --model qwen2_5_vl_mmlatentdiscrete \
            --model_args "pretrained=${model_id},max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False" \
            --gen_kwargs "custom_prefill=${custom_prefill},prompt_mode=mmlatent2,prompt_version=new,num_latents=${num_latents}" \
            --tasks "${task}" \
            --batch_size 1 \
            --log_samples \
            ${LIMIT_ARG} \
            --output_path "${row_out}" || echo "Warning: Task ${task} in Row ${row_id} failed with exit code $?"
    done

    if [ "$IS_DRY_RUN" = false ]; then
        echo ""
        echo ">>> Updating E1 Matrix Summary after Row ${row_id}..."
        "${VENV_DIR}/bin/python" "${PROJECT_DIR}/summarize_e1_matrix.py" || true
    fi
}

for row in "${SELECTED_ROWS[@]}"; do
    case "$row" in
        1)
            # Row 1: Mull | latents | think + 20 latents (Reference baseline)
            run_row 1 "array/Qwen2.5-VL-Mull" "latents" "mmlatent2" 20 "row1_mull_ref"
            ;;
        2)
            # Row 2: Mull | latents | empty (K=0 repeat, slow)
            run_row 2 "array/Qwen2.5-VL-Mull" "latents" "empty" 0 "row2_mull_empty"
            ;;
        3)
            # Row 3: Mull | latents | <think></think>\n (Tag-only test)
            run_row 3 "array/Qwen2.5-VL-Mull" "latents" "empty_think" 20 "row3_mull_empty_think"
            ;;
        4)
            # Row 4: Mull | latents | think + 20x" ." (Generic pause tokens)
            run_row 4 "array/Qwen2.5-VL-Mull" "latents" "dot_pause" 20 "row4_mull_dot_pause"
            ;;
        5)
            # Row 5: Mull | latents | think + 20x<|latent_end|> (Special token control)
            run_row 5 "array/Qwen2.5-VL-Mull" "latents" "latent_end" 20 "row5_mull_latent_end"
            ;;
        6)
            # Row 6: Mull | sft | empty (Prompt confound test on Mull, slow)
            run_row 6 "array/Qwen2.5-VL-Mull" "sft" "empty" 0 "row6_mull_sft_empty"
            ;;
        7)
            # Row 7: Mull | sft | think + 20 latents (Mull sensitivity to SFT prompt)
            run_row 7 "array/Qwen2.5-VL-Mull" "sft" "mmlatent2" 20 "row7_mull_sft_latents"
            ;;
        8)
            # Row 8: SAT | sft | empty (Direct SAT SFT baseline)
            run_row 8 "array/Qwen2.5-VL-SAT" "sft" "empty" 0 "row8_sat_sft_empty"
            ;;
        9)
            # Row 9: SAT | latents | empty (SAT evaluated under latent user prompt)
            run_row 9 "array/Qwen2.5-VL-SAT" "latents" "empty" 0 "row9_sat_latents_empty"
            ;;
        *)
            echo "Unknown row: $row"
            ;;
    esac
done

echo ""
echo "=============================================================================="
echo ">>> E1 Run Matrix Execution Completed!"
echo "=============================================================================="
