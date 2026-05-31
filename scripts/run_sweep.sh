#!/bin/bash
# run_sweep.sh — Unified one-parameter sweep driver for E3..E7.
#
# Selects which parameter to sweep through SWEEP (one of: arrival, swarm,
# area, taskdens, maxrange). Scan values are taken from the matching
# environment array, falling back to the default below.
#
# Learning algorithms (proposed / partition_routing_only / mlp_ppo):
#   - retrain from scratch at every scan point
#   - no checkpoint saving, no periodic evaluation
#   - converged tail metrics are summarised from metrics_long.csv
#
# Deterministic baselines (local_only / single_split):
#   - quick evaluation at every scan point
#
# Outputs:
#   results/eval/${RUN_GROUP}/*.json
#   logs/training/${RUN_GROUP}/...
#
# Example (E3 arrival rate sweep on cloud):
#   SWEEP=arrival RUN_GROUP=e3_arrival_w19 WEIGHTS="0.1 0.9" WTAG=w19 \
#     SEED=789 NUM_EPISODES=2500 TAIL_ITERS=30 \
#     MAX_PARALLEL=15 CPU_THREADS=2 bash scripts/run_sweep.sh

set -euo pipefail

SWEEP="${SWEEP:?SWEEP must be one of: arrival, swarm, area, taskdens, maxrange}"

CPU_THREADS="${CPU_THREADS:-2}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"
TOTAL_CORES="${TOTAL_CORES:-$(nproc 2>/dev/null || echo 32)}"
RESERVED_CORES="${RESERVED_CORES:-0}"
NUM_EPISODES="${NUM_EPISODES:-2500}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
TAIL_ITERS="${TAIL_ITERS:-30}"
WEIGHTS="${WEIGHTS:-0.1 0.9}"
WTAG="${WTAG:-w19}"
SEED="${SEED:-789}"
RUN_GROUP="${RUN_GROUP:-e_sweep_${SWEEP}}"
EVAL_OUT="results/eval/${RUN_GROUP}"
TB_ROOT="logs/training/${RUN_GROUP}"

# ---- Default scan values (overridable via env). 9-10 points each. ----
DEFAULT_ARRIVAL=(1.0 2.0 5.0 8.0 10.0 12.0 15.0 20.0 25.0 30.0)
DEFAULT_SWARM=(4 6 8 10 12 14 16 18 20 24)
DEFAULT_AREA=(500 700 900 1000 1200 1400 1600 1800 2000)
DEFAULT_TASKDENS=(1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
DEFAULT_MAXRANGE=(400 500 600 700 800 1000 1200 1400 1600)

case "$SWEEP" in
    arrival)
        SCAN_VALUES=(${ARRIVAL_RATES[@]:-${DEFAULT_ARRIVAL[@]}})
        SCAN_FLAG="--arrival_rate"
        SCAN_TAG="lambda"
        ;;
    swarm)
        SCAN_VALUES=(${SWARM_SIZES[@]:-${DEFAULT_SWARM[@]}})
        SCAN_FLAG="--num_uavs"
        SCAN_TAG="uav"
        ;;
    area)
        SCAN_VALUES=(${AREA_SIZES[@]:-${DEFAULT_AREA[@]}})
        SCAN_FLAG="--area_size"
        SCAN_TAG="area"
        ;;
    taskdens)
        SCAN_VALUES=(${TASK_DENSITIES[@]:-${DEFAULT_TASKDENS[@]}})
        SCAN_FLAG="--avg_tasks_per_request"
        SCAN_TAG="td"
        ;;
    maxrange)
        SCAN_VALUES=(${MAX_RANGES[@]:-${DEFAULT_MAXRANGE[@]}})
        SCAN_FLAG="--max_range"
        SCAN_TAG="mr"
        ;;
    *)
        echo "[ERROR] Unknown SWEEP: $SWEEP" >&2
        exit 1
        ;;
esac

LEARN_ALGOS=(
    "configs/experiments/main_gnn_ppo_tmc_stable.yaml|proposed"
    "configs/experiments/baseline_pr_no_compression_ppo_tmc_stable.yaml|partition_routing_only"
    "configs/experiments/baseline_mlp_ppo_tmc_stable.yaml|mlp_ppo"
)
LEARN_ALGO_ORDER=("proposed" "partition_routing_only" "mlp_ppo")

DET_ALGOS=(
    "configs/experiments/baseline_local_only_tmc_stable.yaml|local_only"
    "configs/experiments/baseline_single_split_tmc_stable.yaml|single_split"
)

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

DRY_RUN=false
BASELINES_ONLY=false
while [ $# -gt 0 ]; do
    case "$1" in
        --dry_run) DRY_RUN=true; shift ;;
        --baselines_only) BASELINES_ONLY=true; shift ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
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

# Format scan value into a filesystem-safe tag.
# area sweep needs "WxH" semantics; others are a single scalar.
format_tag() {
    local v="$1"
    if [ "$SWEEP" = "area" ]; then
        echo "${SCAN_TAG}${v}"
    else
        echo "${SCAN_TAG}${v//./_}"
    fi
}

# Build the parameter override that train.py / evaluate.py expects.
format_flag_args() {
    local v="$1"
    if [ "$SWEEP" = "area" ]; then
        # area is a square in current sweep (W = H = v)
        echo "${SCAN_FLAG} ${v} ${v}"
    else
        echo "${SCAN_FLAG} ${v}"
    fi
}

echo ""
echo "=========================================================="
echo "  Sweep runtime"
echo "  sweep=${SWEEP}  param=${SCAN_FLAG}  values=(${SCAN_VALUES[*]})"
echo "  run_group=${RUN_GROUP}  tensorboard_root=${TB_ROOT}"
echo "  weights=(${WEIGHTS})  wtag=${WTAG}  seed=${SEED:-<unset>}"
echo "  total_cores=${TOTAL_CORES} reserved_cores=${RESERVED_CORES}"
echo "  cpu_threads=${CPU_THREADS} cpu_interop_threads=${CPU_INTEROP_THREADS}"
echo "  max_parallel=${MAX_PARALLEL}"
echo "  num_episodes=${NUM_EPISODES}  tail_iters=${TAIL_ITERS}  eval_episodes=${EVAL_EPISODES}"
echo "=========================================================="
echo ""
echo "  TensorBoard:"
echo "    tensorboard --logdir ${TB_ROOT} --port 6006 --bind_all"
echo ""

# ---- Phase 1: learning algorithms (priority order: proposed > pr-only > mlp) ----
LEARN_TOTAL=$((${#SCAN_VALUES[@]} * ${#LEARN_ALGOS[@]}))
if $BASELINES_ONLY; then
    echo "=========================================================="
    echo "  --baselines_only mode: skipping ${LEARN_TOTAL} learning runs"
    echo "=========================================================="
    echo ""
else
    echo "=========================================================="
    echo "  Phase 1: Learning runs (${LEARN_TOTAL} total)"
    echo "=========================================================="
    echo ""

    for algo_name in "${LEARN_ALGO_ORDER[@]}"; do
        for aentry in "${LEARN_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            [ "$algo" != "$algo_name" ] && continue
            for v in "${SCAN_VALUES[@]}"; do
                tag=$(format_tag "$v")
                flag_args=$(format_flag_args "$v")
                run_tag="mtan_${WTAG}_${algo}_${tag}"
                log_file="logs/proc_logs/${RUN_GROUP}_${run_tag}.log"
                cmd="python scripts/train.py \
                    --config ${cfg} \
                    --device cpu \
                    --console_mode plain \
                    --disable_eval \
                    --disable_save \
                    --run_group ${RUN_GROUP} \
                    --run_tag ${run_tag} \
                    --weights ${WEIGHTS} \
                    ${flag_args} \
                    --cpu_threads ${CPU_THREADS} \
                    --cpu_interop_threads ${CPU_INTEROP_THREADS} \
                    --num_episodes ${NUM_EPISODES}${SEED:+ --seed ${SEED}}"
                launch_job "${run_tag}" "${log_file}" "${cmd}"
            done
        done
    done
fi

# ---- Phase 2: deterministic baselines (run concurrently while learning still finishes) ----
DET_TOTAL=$((${#SCAN_VALUES[@]} * ${#DET_ALGOS[@]}))
echo ""
echo "=========================================================="
echo "  Phase 2: Deterministic baselines (${DET_TOTAL} total)"
echo "=========================================================="
echo ""

for v in "${SCAN_VALUES[@]}"; do
    tag=$(format_tag "$v")
    flag_args=$(format_flag_args "$v")
    for aentry in "${DET_ALGOS[@]}"; do
        IFS='|' read -r cfg algo <<< "$aentry"
        run_name="mtan_${WTAG}_${algo}_${tag}"
        log_file="logs/proc_logs/${RUN_GROUP}_base_${run_name}.log"
        out_file="${EVAL_OUT}/${run_name}.json"
        cmd="python scripts/evaluate.py \
            --config ${cfg} \
            --weights ${WEIGHTS} \
            ${flag_args} \
            --num_episodes ${EVAL_EPISODES} \
            --cpu_threads ${CPU_THREADS} \
            --output ${out_file} \
            ${SEED:+--seed ${SEED}}"
        launch_job "${run_name}" "${log_file}" "${cmd}"
    done
done

if ! $DRY_RUN; then
    wait_for_all_jobs "All jobs (learning + deterministic)"
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "[ERROR] sweep had ${FAILED_COUNT} failed process(es). Check logs/proc_logs/." >&2
        exit 1
    fi
    ACTIVE_PIDS=()
    FAILED_COUNT=0
fi

# ---- Phase 3: summarise learning runs into tail-mean JSON ----
if $BASELINES_ONLY; then
    echo ""
    echo "=========================================================="
    echo "  --baselines_only mode: skipping Phase 3 (learning summaries)"
    echo "=========================================================="
else
    echo ""
    echo "=========================================================="
    echo "  Phase 3: Summarise converged tail metrics (tail_iters=${TAIL_ITERS})"
    echo "=========================================================="
    echo ""

    for v in "${SCAN_VALUES[@]}"; do
        tag=$(format_tag "$v")
        for aentry in "${LEARN_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            run_tag="mtan_${WTAG}_${algo}_${tag}"
            out_file="${EVAL_OUT}/${run_tag}.json"
            cmd="python scripts/summarize_converged_metrics.py \
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
fi

echo ""
echo "=========================================================="
echo "  sweep=${SWEEP} artifacts ready at: ${EVAL_OUT}/"
echo "=========================================================="
echo ""
