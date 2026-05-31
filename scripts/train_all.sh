#!/bin/bash
# train_all.sh - Launch the four E1 convergence jobs in parallel.

set -euo pipefail

CPU_THREADS="${CPU_THREADS:-2}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"
RUN_GROUP="e1_convergence_main"
TB_ROOT="logs/training/${RUN_GROUP}"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

TAG="v1"
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    TAG="$1"
    shift
fi
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CONFIGS=(
    "configs/experiments/main_gnn_ppo_tmc_stable.yaml|mtan"
    "configs/experiments/main_gnn_ppo_tmc_stable_split.yaml|split"
    "configs/experiments/main_gnn_ppo_tmc_stable_dense.yaml|dense"
    "configs/experiments/main_gnn_ppo_tmc_stable_cross.yaml|cross"
)

mkdir -p logs/proc_logs

echo ""
echo "=========================================================="
echo "  E1 convergence launch"
echo "  tensorboard_root=${TB_ROOT}"
echo "  tag=${TAG} cpu_threads=${CPU_THREADS} cpu_interop_threads=${CPU_INTEROP_THREADS}"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "  extra args: ${EXTRA_ARGS[*]}"
echo "=========================================================="
echo ""

PIDS=()
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r cfg model <<< "$entry"
    run_tag="${model}_${TAG}"
    log_file="logs/proc_logs/${run_tag}.log"

    CMD=(python scripts/train.py
        --config "$cfg"
        --device cpu
        --console_mode plain
        --disable_eval
        --disable_save
        --run_group "$RUN_GROUP"
        --run_tag "$run_tag"
        --cpu_threads "$CPU_THREADS"
        --cpu_interop_threads "$CPU_INTEROP_THREADS"
        "${EXTRA_ARGS[@]}"
    )
    if $DRY_RUN; then
        echo "  [DRY] ${run_tag}"
        printf '        %q ' "${CMD[@]}"
        echo ""
    else
        nohup "${CMD[@]}" > "$log_file" 2>&1 &
        pid=$!
        PIDS+=("$pid")
        echo "  [PID ${pid}] ${run_tag}  log -> ${log_file}"
    fi
done

$DRY_RUN && { echo ""; echo "Dry run complete. No processes launched."; exit 0; }

echo ""
echo "----------------------------------------------------------"
echo "  TensorBoard:"
echo "    tensorboard --logdir ${TB_ROOT} --port 6006 --bind_all"
echo ""
echo "  Follow a single run:"
echo "    tail -f logs/proc_logs/split_${TAG}.log"
echo ""
echo "  Check status:"
echo "    ps aux | grep train.py"
echo ""
echo "  Stop all:"
echo "    kill ${PIDS[*]}"
echo "----------------------------------------------------------"
echo ""

while true; do
    alive=0
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done
    [ "$alive" -eq 0 ] && break
    echo "  [$(date +%H:%M:%S)] ${alive} process(es) still running..."
    sleep 30
done

echo ""
failed=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done
if [ "$failed" -gt 0 ]; then
    echo "  WARNING: ${failed} process(es) exited with non-zero code."
    echo "  Check logs in: logs/proc_logs/"
else
    echo "  All experiments finished successfully."
fi
echo ""
