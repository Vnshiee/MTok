#!/usr/bin/env bash
# Full Mirage ablation matrix (see ../MIRAGE_ABLATION_STUDY_DESIGN.md).
#
# Launch detached (from external_models/mirage/ablation_study):
#   mkdir -p logs && tmux new-session -d -s mirage_ablation "bash run_all_ablations.sh 2>&1 | tee -a logs/run_all.log"
#   tmux attach -t mirage_ablation
#
# Re-running skips every condition whose <OUT_DIR>/<run>.COMPLETE marker exists, and the runner itself skips questions
# already written to a partial .jsonl, so the script is safe to restart after interruption.
#
# Environment overrides (defaults in brackets):
#   PYTHON [python]            interpreter of an env with requirements_ablation.txt installed
#   OUT_DIR [outputs]          LOG_DIR [logs]          HF_CACHE [unset]
#   MODEL_TARGET [Miiche/vsp_spatial_planning_direct_sft]   MODEL_BASE [Qwen/Qwen2.5-VL-7B-Instruct]
#   DEVICE [cuda] DTYPE [bfloat16] CUDA_VISIBLE_DEVICES [0]
#   BENCHMARKS ["vsp_planning blink vstar mmvp"]   BLINK_SUBTASKS [Jigsaw]
#   VSTAR_MAX_PIXELS [unset = processor default, as in src/test.py]
#   MAX_NEW_TOKENS [512]  LIMIT [unset]  SMOKE [0: if 1, LIMIT=5 and OUT_DIR=outputs_smoke]
#   DRY_RUN [0]  SKIP_PREFLIGHT [0]  SKIP_DIAGNOSTIC [0]
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRAGE_DIR="$(dirname "$HERE")"
cd "$HERE"

PYTHON="${PYTHON:-python}"
SMOKE="${SMOKE:-0}"
OUT_DIR="${OUT_DIR:-$([ "$SMOKE" = "1" ] && echo outputs_smoke || echo outputs)}"
LOG_DIR="${LOG_DIR:-logs}"
LIMIT="${LIMIT:-$([ "$SMOKE" = "1" ] && echo 5 || echo "")}"
MODEL_TARGET="${MODEL_TARGET:-Miiche/vsp_spatial_planning_direct_sft}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen2.5-VL-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BENCHMARKS="${BENCHMARKS:-vsp_planning blink vstar mmvp}"
BLINK_SUBTASKS="${BLINK_SUBTASKS:-Jigsaw}"
VSTAR_MAX_PIXELS="${VSTAR_MAX_PIXELS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DRY_RUN="${DRY_RUN:-0}"
VSP_ROOT="$MIRAGE_DIR/repo/data/vsp_spatial_planning"

K_SWEEP=(0 1 3 4 5 10 20)          # Axis 3 (4 = training latent size of the released checkpoint)
K_INTERVENE_VSP=(1 3 4 5 10 20)    # Axis 1 x Axis 3 on the in-distribution benchmark
CORE_INTERVENTIONS=(zero noise donor)
EXTRA_INTERVENTIONS=(pause mean)

mkdir -p "$OUT_DIR" "$LOG_DIR"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo; echo "==================== [$(stamp)] $* ===================="; }

COMMON=(--output_dir "$OUT_DIR" --device "$DEVICE" --dtype "$DTYPE" --max_new_tokens "$MAX_NEW_TOKENS"
        --blink_subtasks "$BLINK_SUBTASKS" --vsp_data_root "$VSP_ROOT")
[ -n "$LIMIT" ] && COMMON+=(--limit "$LIMIT")
[ -n "${HF_CACHE:-}" ] && COMMON+=(--hf_cache "$HF_CACHE")

# run_cond <benchmark> <model_mode> <quota> <intervention> <entry> [extra args...]
run_cond() {
  local bench="$1" mode="$2" quota="$3" interv="$4" entry="$5"; shift 5
  local extra=("$@")
  local model="$MODEL_TARGET"; [ "$mode" = "base" ] && model="$MODEL_BASE"
  local name="${bench}__${mode}__K${quota}__${interv}__${entry}"
  local tag=""
  for x in ${extra[@]+"${extra[@]}"}; do [ "$x" = "test_py" ] && tag="__testpy"; done
  name="${name}${tag}"
  local px=()
  [ "$bench" = "vstar" ] && [ -n "$VSTAR_MAX_PIXELS" ] && px=(--max_pixels "$VSTAR_MAX_PIXELS")
  if [ -f "$OUT_DIR/$name.COMPLETE" ]; then
    echo "[$(stamp)] skip (complete): $name"; return 0
  fi
  local cmd=("$PYTHON" -u run_mirage_ablation.py --benchmark "$bench" --model_mode "$mode" --model_path "$model"
             --quota "$quota" --intervention "$interv" --entry "$entry" --run_name "$name" "${COMMON[@]}" ${px[@]+"${px[@]}"} ${extra[@]+"${extra[@]}"})
  echo "[$(stamp)] run: $name"
  if [ "$DRY_RUN" = "1" ]; then printf '   %q' "${cmd[@]}"; echo; return 0; fi
  "${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/$name.log"
}

# ----------------------------------------------------------------------------------------------- Phase 0: preflight
banner "Phase 0: preflight"
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  [ -d "$MIRAGE_DIR/repo/transformers/src/transformers" ] || {
    echo "Mirage repo missing. Run:"; echo "  git clone https://github.com/UMass-Embodied-AGI/Mirage.git $MIRAGE_DIR/repo"
    echo "  git -C $MIRAGE_DIR/repo checkout 53f26de2682025e146781c5b198ec93bdfe4c4d6"; exit 1; }
  head="$(git -C "$MIRAGE_DIR/repo" rev-parse HEAD)"
  [ "$head" = "53f26de2682025e146781c5b198ec93bdfe4c4d6" ] || echo "WARNING: repo at $head, audited commit is 53f26de"
  if [ ! -f "$VSP_ROOT/test_direct.jsonl" ]; then
    echo "[$(stamp)] extracting VSP test set into $VSP_ROOT"
    tar -xzf "$VSP_ROOT/vsp_spatial_planning_test.tar.gz" -C "$VSP_ROOT"
  fi
  [ "$(wc -l < "$VSP_ROOT/test_direct.jsonl")" -eq 400 ] || { echo "VSP test set should have 400 lines"; exit 1; }
  "$PYTHON" -c "import torch, qwen_vl_utils, mathruler, datasets, scipy, PIL; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  if [ "$DEVICE" = "cuda" ]; then
    "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available')"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  fi
  if [ "${SKIP_DIAGNOSTIC:-0}" != "1" ]; then
    echo "[$(stamp)] CPU diagnostic"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -u "$MIRAGE_DIR/diagnostics/verify_mirage_autograd.py" 2>&1 | tee "$LOG_DIR/diagnostic.log" | grep -E "^\[|checks passed"
  fi
fi

for bench in $BENCHMARKS; do
  if [ "$bench" = "vsp_planning" ]; then
    # --------------------------------------------------------------------- Phase 1: in-distribution (VSP planning)
    banner "Phase 1a: VSP reproduction (fork exit rule)"
    run_cond vsp_planning target auto none natural                    # greedy, fork protocol
    run_cond vsp_planning target auto none natural --decoding test_py # src/test.py sampling (paper protocol)
    banner "Phase 1b: VSP latent quota sweep (Axis 3)"
    for K in "${K_SWEEP[@]}"; do run_cond vsp_planning target "$K" none forced; done
    banner "Phase 1c: VSP causal interventions x quota (Axis 1 x Axis 3)"
    for K in "${K_INTERVENE_VSP[@]}"; do
      for I in "${CORE_INTERVENTIONS[@]}"; do run_cond vsp_planning target "$K" "$I" forced; done
    done
    for I in "${EXTRA_INTERVENTIONS[@]}"; do run_cond vsp_planning target 4 "$I" forced; done
    banner "Phase 1d: VSP format controls and base model (H2, Axis 4)"
    run_cond vsp_planning target auto suppress natural
    run_cond vsp_planning base 0 none natural
  else
    # --------------------------------------------------------------------- Phase 2: requested benchmarks
    banner "Phase 2 [$bench]: base model, natural protocol, quota sweep, interventions"
    run_cond "$bench" base 0 none natural
    run_cond "$bench" target auto none natural
    for K in "${K_SWEEP[@]}"; do run_cond "$bench" target "$K" none forced; done
    for I in "${CORE_INTERVENTIONS[@]}" "${EXTRA_INTERVENTIONS[@]}"; do run_cond "$bench" target 4 "$I" forced; done
    run_cond "$bench" target auto suppress natural
  fi
done

# ----------------------------------------------------------------------------------------------- Phase 3: summary
banner "Phase 3: summary"
if [ "$DRY_RUN" != "1" ]; then
  "$PYTHON" -u summarize_ablation_results.py --output_dir "$OUT_DIR" --reference target__K4__none__forced --equiv_margin 0.03 --bootstrap 10000 2>&1 | tee "$LOG_DIR/summary.log" | tail -n 5
  echo "[$(stamp)] summary written to $OUT_DIR/ABLATION_SUMMARY.md"
fi
banner "done"
