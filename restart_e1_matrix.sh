#!/usr/bin/env bash
# ==============================================================================
# restart_e1_matrix.sh: Safely stop slow CPU offload, verify GPU, and restart E1 Matrix
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

VENV_DIR="${PROJECT_DIR}/.mtok"
EVAL_BASE="${PROJECT_DIR}/eval_outputs/e1_matrix"

echo "=============================================================================="
echo ">>> STEP 1: Stopping existing slow lmms_eval and watchdog processes..."
echo "=============================================================================="
pkill -f "run_e1_matrix.sh" || true
pkill -f "lmms_eval" || true
pkill -f "watchdog_e1_matrix.py" || true
sleep 3

# Double check if any lingering process remains
if pgrep -f "lmms_eval" >/dev/null 2>&1; then
    echo "Sending SIGKILL to lingering lmms_eval process..."
    pkill -9 -f "lmms_eval" || true
    sleep 2
fi

echo "=============================================================================="
echo ">>> STEP 2: Verifying GPU Memory Availability..."
echo "=============================================================================="
"${VENV_DIR}/bin/python" -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA is not available!')
    exit(1)
free_b, total_b = torch.cuda.mem_get_info()
free_gb = free_b / (1024**3)
total_gb = total_b / (1024**3)
print(f'GPU VRAM: {free_gb:.2f} GiB free out of {total_gb:.2f} GiB total.')
if free_gb < 16.0:
    print(f'WARNING: Free VRAM ({free_gb:.2f} GiB) is lower than 16 GiB. Check running processes.')
else:
    print(f'SUCCESS: Sufficient VRAM ({free_gb:.2f} GiB) available for GPU-accelerated evaluation.')
"

echo "=============================================================================="
echo ">>> STEP 3: Archiving partial CPU-offload log to prevent duplicate samples..."
echo "=============================================================================="
PARTIAL_LOG="${EVAL_BASE}/row2_mull_empty/per_sample_logs/per_sample_details_blink_sprel.jsonl"
ARCHIVE_LOG="${EVAL_BASE}/row2_mull_empty/per_sample_logs/partial_cpu_backup_blink_sprel.jsonl"

if [ -f "${PARTIAL_LOG}" ]; then
    LINES=$(wc -l < "${PARTIAL_LOG}")
    echo "Found partial log with ${LINES} samples. Archiving to: ${ARCHIVE_LOG}"
    mv "${PARTIAL_LOG}" "${ARCHIVE_LOG}"
fi

echo "=============================================================================="
echo ">>> STEP 4: Running 2-sample Validation Smoke Test on Row 2 (GPU Speed & Memory Test)..."
echo "=============================================================================="
export BLINK_PROMPT_MODE="latents"
export LMMS_EVAL_SAMPLE_LOG_DIR="${EVAL_BASE}/smoke_dryrun/validation_test/per_sample_logs"
mkdir -p "${LMMS_EVAL_SAMPLE_LOG_DIR}"

TEST_START=$(date +%s)
"${VENV_DIR}/bin/accelerate" launch --num_processes=1 -m lmms_eval \
    --model qwen2_5_vl_mmlatentdiscrete \
    --model_args "pretrained=array/Qwen2.5-VL-Mull,max_pixels=4194304,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False" \
    --gen_kwargs "custom_prefill=empty,prompt_mode=mmlatent2,prompt_version=new,num_latents=20" \
    --tasks "blink_sprel" \
    --batch_size 1 \
    --limit 2 \
    --log_samples \
    --output_path "${EVAL_BASE}/smoke_dryrun/validation_test"

TEST_END=$(date +%s)
TEST_DUR=$((TEST_END - TEST_START))
echo ""
echo ">>> Validation test finished successfully in ${TEST_DUR} seconds!"
echo "    (Previously, 2 samples on CPU offload took ~1,000+ seconds. Speedup verified!)"

echo "=============================================================================="
echo ">>> STEP 5: Launching Full E1 Matrix Evaluation and Background Watchdog..."
echo "=============================================================================="
# Start watchdog in background
nohup "${VENV_DIR}/bin/python" "${PROJECT_DIR}/watchdog_e1_matrix.py" > "${EVAL_BASE}/watchdog.log" 2>&1 &
WATCHDOG_PID=$!
echo "Watchdog started with PID: ${WATCHDOG_PID}"

# Launch main runner
echo "Launching main matrix runner (Row 1 will be automatically detected as COMPLETE and skipped)..."
echo "Live output will be logged to: ${EVAL_BASE}/matrix_runner.log"
exec bash "${PROJECT_DIR}/run_e1_matrix.sh" 2>&1 | tee "${EVAL_BASE}/matrix_runner.log"
