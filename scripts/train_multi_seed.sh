#!/bin/bash
# train_multi_seed.sh - Launch multi-seed E1 convergence runs.

set -euo pipefail

WEIGHTS="0.1 0.9"
SEEDS=(123 456 789 1024)
CPU_THREADS="${CPU_THREADS:-2}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"
TOTAL_CORES="${TOTAL_CORES:-$(nproc 2>/dev/null || echo 32)}"
RESERVED_CORES="${RESERVED_CORES:-0}"
NUM_EPISODES="${NUM_EPISODES:-16000}"
RUN_GROUP="e1_convergence_multiseed"
TB_ROOT="logs/training/${RUN_GROUP}"

AVAILABLE_CORES=$((TOTAL_CORES - RESERVED_CORES))
if [ "$AVAILABLE_CORES" -lt 1 ]; then
    AVAILABLE_CORES=1
fi
MAX_PARALLEL_DEFAULT=$((AVAILABLE_CORES / CPU_THREADS))
if [ "$MAX_PARALLEL_DEFAULT" -lt 1 ]; then
    MAX_PARALLEL_DEFAULT=1
fi
MAX_PARALLEL="${MAX_PARALLEL:-$MAX_PARALLEL_DEFAULT}"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

CONFIGS=(
    "configs/experiments/main_gnn_ppo_tmc_stable_split.yaml|split"
    "configs/experiments/main_gnn_ppo_tmc_stable.yaml|mtan"
    "configs/experiments/main_gnn_ppo_tmc_stable_dense.yaml|dense"
    "configs/experiments/main_gnn_ppo_tmc_stable_cross.yaml|cross"
)

DRY_RUN=false
while [ $# -gt 0 ]; do
    case "$1" in
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p logs/proc_logs

TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
echo ""
echo "=========================================================="
echo "  Multi-seed training: ${#SEEDS[@]} seeds × ${#CONFIGS[@]} models = ${TOTAL} processes"
echo "  tensorboard_root=${TB_ROOT}"
echo "  Weights: ${WEIGHTS}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Num episodes: ${NUM_EPISODES}  (≈ $((NUM_EPISODES / 10)) PPO iterations)"
echo "  cpu_threads=${CPU_THREADS} cpu_interop_threads=${CPU_INTEROP_THREADS}"
echo "  total_cores=${TOTAL_CORES} reserved_cores=${RESERVED_CORES} max_parallel=${MAX_PARALLEL}"
echo "=========================================================="
echo ""

ACTIVE_PIDS=()
FAILED_COUNT=0

reap_finished_jobs() {
    local next_pids=()
    for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            next_pids+=("$pid")
        else
            wait "$pid" || FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done
    ACTIVE_PIDS=("${next_pids[@]}")
}

wait_for_slot() {
    while [ "${#ACTIVE_PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
        reap_finished_jobs
        [ "${#ACTIVE_PIDS[@]}" -lt "$MAX_PARALLEL" ] && break
        sleep 5
    done
}

for seed in "${SEEDS[@]}"; do
    for entry in "${CONFIGS[@]}"; do
        IFS='|' read -r cfg model <<< "$entry"
        run_tag="${model}_w19_s${seed}"
        log_file="logs/proc_logs/${run_tag}.log"
        cmd="python scripts/train.py \
            --config ${cfg} \
            --device cpu \
            --console_mode plain \
            --disable_eval \
            --disable_save \
            --run_group ${RUN_GROUP} \
            --run_tag ${run_tag} \
            --weights ${WEIGHTS} \
            --seed ${seed} \
            --cpu_threads ${CPU_THREADS} \
            --cpu_interop_threads ${CPU_INTEROP_THREADS} \
            --num_episodes ${NUM_EPISODES}"

        if $DRY_RUN; then
            echo "  [DRY] ${run_tag}"
            echo "        ${cmd}"
            echo ""
        else
            wait_for_slot
            nohup bash -lc "$cmd" > "$log_file" 2>&1 &
            pid=$!
            ACTIVE_PIDS+=("$pid")
            echo "  [PID ${pid}] ${run_tag}  log -> ${log_file}"
        fi
    done
done

$DRY_RUN && { echo "Dry run complete. No processes launched."; exit 0; }

echo ""
echo "----------------------------------------------------------"
echo "  TensorBoard:"
echo "    tensorboard --logdir ${TB_ROOT} --port 6006 --bind_all"
echo ""
echo "  Check status:"
echo "    ps aux | grep train.py | grep -v grep | wc -l"
echo "----------------------------------------------------------"
echo ""

while [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; do
    reap_finished_jobs
    alive="${#ACTIVE_PIDS[@]}"
    [ "$alive" -eq 0 ] && break
    echo "  [$(date +%H:%M:%S)] ${alive}/${TOTAL} processes running..."
    sleep 60
done

echo ""
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "  WARNING: ${FAILED_COUNT} process(es) exited with non-zero code."
    echo "  Check logs: logs/proc_logs/"
else
    echo "  All ${TOTAL} experiments finished successfully."
fi
echo ""
