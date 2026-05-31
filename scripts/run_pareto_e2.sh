#!/bin/bash
# run_pareto_e2.sh — E2 Pareto experiment (MTAN representative model)
#
# Learning algorithms:
#   - pure training only
#   - no checkpoint saving
#   - no periodic evaluation
#   - final paper numbers are extracted from metrics_long.csv tail statistics
#
# Deterministic baselines:
#   - quick evaluation only
#   - a few episodes per weight

set -euo pipefail

CPU_THREADS="${CPU_THREADS:-2}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"
TOTAL_CORES="${TOTAL_CORES:-$(nproc 2>/dev/null || echo 32)}"
RESERVED_CORES="${RESERVED_CORES:-2}"
NUM_EPISODES="${NUM_EPISODES:-16000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
TAIL_ITERS="${TAIL_ITERS:-100}"
RUN_GROUP="${RUN_GROUP:-e2_pareto}"
EVAL_OUT="${EVAL_OUT:-results/eval/${RUN_GROUP}}"
WEIGHT_PRESET="${WEIGHT_PRESET:-5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
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

if [ -n "${WEIGHT_ENTRIES:-}" ]; then
    IFS=';' read -r -a WEIGHTS_LIST <<< "${WEIGHT_ENTRIES}"
else
    case "${WEIGHT_PRESET}" in
        5)
            WEIGHTS_LIST=(
                "0.9 0.1|w91"
                "0.7 0.3|w73"
                "0.5 0.5|w55"
                "0.3 0.7|w37"
                "0.1 0.9|w19"
            )
            ;;
        7)
            WEIGHTS_LIST=(
                "0.9 0.1|w91"
                "0.8 0.2|w82"
                "0.7 0.3|w73"
                "0.5 0.5|w55"
                "0.3 0.7|w37"
                "0.2 0.8|w28"
                "0.1 0.9|w19"
            )
            ;;
        *)
            echo "[ERROR] Unsupported WEIGHT_PRESET=${WEIGHT_PRESET}. Use 5 or 7, or pass WEIGHT_ENTRIES."
            exit 1
            ;;
    esac
fi

LEARN_ALGOS=(
    "configs/experiments/main_gnn_ppo_tmc_stable.yaml|proposed"
    "configs/experiments/baseline_pr_no_compression_ppo_tmc_stable.yaml|partition_routing_only"
    "configs/experiments/baseline_mlp_ppo_tmc_stable.yaml|mlp_ppo"
)

DET_ALGOS=(
    "configs/experiments/baseline_local_only_tmc_stable.yaml|local_only"
    "configs/experiments/baseline_single_split_tmc_stable.yaml|single_split"
)

DRY_RUN=false
BASELINES_ONLY=false
TRAIN_ONLY=false
while [ $# -gt 0 ]; do
    case "$1" in
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --baselines_only)
            BASELINES_ONLY=true
            shift
            ;;
        --train_only)
            TRAIN_ONLY=true
            shift
            ;;
        --tail_iters)
            TAIL_ITERS="$2"
            shift 2
            ;;
        --tail_iters=*)
            TAIL_ITERS="${1#*=}"
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p logs/proc_logs "$EVAL_OUT"

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

wait_for_all_jobs() {
    local label="$1"
    while [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; do
        reap_finished_jobs
        local alive="${#ACTIVE_PIDS[@]}"
        [ "$alive" -eq 0 ] && break
        echo "  [$(date +%H:%M:%S)] ${label}: ${alive} process(es) running..."
        sleep 60
    done
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "  WARNING: ${label} had ${FAILED_COUNT} failed process(es)."
    else
        echo "  ${label} completed successfully."
    fi
}

launch_job() {
    local run_name="$1"
    local log_file="$2"
    local cmd="$3"
    if $DRY_RUN; then
        echo "  [DRY] ${run_name}"
        echo "        ${cmd}"
        return
    fi
    wait_for_slot
    nohup bash -lc "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    ACTIVE_PIDS+=("$pid")
    echo "  [PID ${pid}] ${run_name}  log -> ${log_file}"
}

summarize_learning_runs() {
    echo ""
    echo "=========================================================="
    echo "  E2 Pareto — Phase 1b: Summarize converged tail metrics"
    echo "  tail_iters=${TAIL_ITERS}"
    echo "=========================================================="
    echo ""

    for wentry in "${WEIGHTS_LIST[@]}"; do
        IFS='|' read -r weights wtag <<< "$wentry"
        for aentry in "${LEARN_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            run_tag="mtan_${wtag}_${algo}"
            out_file="${EVAL_OUT}/${run_tag}.json"
            cmd="${PYTHON_BIN} scripts/summarize_converged_metrics.py \
                --config ${cfg} \
                --run_tag ${run_tag} \
                --run_group ${RUN_GROUP} \
                --tail_iters ${TAIL_ITERS} \
                --output ${out_file}"
            if $DRY_RUN; then
                echo "  [DRY-SUMMARY] ${run_tag}"
                echo "                ${cmd}"
            else
                bash -lc "$cmd"
            fi
        done
    done
}

echo ""
echo "=========================================================="
echo "  E2 Pareto runtime"
echo "  tensorboard_root=${TB_ROOT}"
echo "  run_group=${RUN_GROUP}"
echo "  eval_out=${EVAL_OUT}"
echo "  weight_preset=${WEIGHT_PRESET}"
echo "  total_cores=${TOTAL_CORES} reserved_cores=${RESERVED_CORES}"
echo "  cpu_threads=${CPU_THREADS} cpu_interop_threads=${CPU_INTEROP_THREADS}"
echo "  max_parallel=${MAX_PARALLEL} tail_iters=${TAIL_ITERS}"
echo "=========================================================="
echo ""
echo "  TensorBoard:"
echo "    tensorboard --logdir ${TB_ROOT} --port 6006 --bind_all"
echo ""

if ! $BASELINES_ONLY; then
    TOTAL=$((${#LEARN_ALGOS[@]} * ${#WEIGHTS_LIST[@]}))
    echo ""
    echo "=========================================================="
    echo "  E2 Pareto — Phase 1: Training (${TOTAL} runs)"
    echo "  No checkpoint saving, no learning-time evaluation"
    echo "=========================================================="
    echo ""

    for wentry in "${WEIGHTS_LIST[@]}"; do
        IFS='|' read -r weights wtag <<< "$wentry"
        for aentry in "${LEARN_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            run_tag="mtan_${wtag}_${algo}"
            log_file="logs/proc_logs/e2_${run_tag}.log"
            cmd="${PYTHON_BIN} scripts/train.py \
                --config ${cfg} \
                --device cpu \
                --console_mode plain \
                --disable_eval \
                --disable_save \
                --run_group ${RUN_GROUP} \
                --run_tag ${run_tag} \
                --weights ${weights} \
                --cpu_threads ${CPU_THREADS} \
                --cpu_interop_threads ${CPU_INTEROP_THREADS} \
                --num_episodes ${NUM_EPISODES}"
            launch_job "${run_tag}" "${log_file}" "${cmd}"
        done
    done

    if ! $DRY_RUN; then
        wait_for_all_jobs "E2 training"
        if [ "$FAILED_COUNT" -gt 0 ]; then
            echo "[ERROR] Aborting E2 because one or more training runs failed. Check logs/proc_logs/."
            exit 1
        fi
        ACTIVE_PIDS=()
        FAILED_COUNT=0
    fi
    summarize_learning_runs
fi

if $TRAIN_ONLY; then
    echo ""
    echo "=========================================================="
    echo "  E2 training and learning-summary completed."
    echo "  Skipping deterministic baselines because --train_only was set."
    echo "=========================================================="
    echo ""
    exit 0
fi

echo ""
echo "=========================================================="
echo "  E2 Pareto — Phase 2: Deterministic baselines"
echo "  ${#DET_ALGOS[@]} algorithms × ${#WEIGHTS_LIST[@]} weights = $((${#DET_ALGOS[@]} * ${#WEIGHTS_LIST[@]})) quick runs"
echo "=========================================================="
echo ""

for wentry in "${WEIGHTS_LIST[@]}"; do
    IFS='|' read -r weights wtag <<< "$wentry"
    for aentry in "${DET_ALGOS[@]}"; do
        IFS='|' read -r cfg algo <<< "$aentry"
        run_name="mtan_${wtag}_${algo}"
        log_file="logs/proc_logs/e2_base_${wtag}_${algo}.log"
        out_file="${EVAL_OUT}/${run_name}.json"
        cmd="${PYTHON_BIN} scripts/evaluate.py \
            --config ${cfg} \
            --weights ${weights} \
            --num_episodes ${EVAL_EPISODES} \
            --cpu_threads ${CPU_THREADS} \
            --output ${out_file}"
        launch_job "${run_name}" "${log_file}" "${cmd}"
    done
done

if ! $DRY_RUN; then
    wait_for_all_jobs "E2 deterministic baselines"
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "[ERROR] E2 deterministic baselines had failures. Check logs/proc_logs/."
        exit 1
    fi
fi

echo ""
echo "=========================================================="
echo "  E2 artifacts ready at: ${EVAL_OUT}/"
echo "  learning algos: tail-summary JSON from metrics_long.csv"
echo "  deterministic baselines: evaluation JSON"
echo "=========================================================="
echo ""
