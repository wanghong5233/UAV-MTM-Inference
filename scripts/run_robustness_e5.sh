#!/bin/bash
# run_robustness_e5.sh — E5 robustness / sensitivity experiment
#
# Learning algorithms:
#   - retrain from scratch at every parameter point
#   - no checkpoint saving
#   - no periodic evaluation
#   - summarize final paper numbers from metrics_long.csv tail statistics
#
# Deterministic baselines:
#   - quick evaluation at every parameter point
#
# Output:
#   results/eval/e5_robustness/*.json

set -euo pipefail

CPU_THREADS="${CPU_THREADS:-2}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"
TOTAL_CORES="${TOTAL_CORES:-$(nproc 2>/dev/null || echo 32)}"
RESERVED_CORES="${RESERVED_CORES:-0}"
NUM_EPISODES="${NUM_EPISODES:-16000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
TAIL_ITERS="${TAIL_ITERS:-100}"
WEIGHTS="${WEIGHTS:-0.5 0.5}"
WTAG="${WTAG:-w55}"
SEED="${SEED:-}"
EVAL_OUT="results/eval/${RUN_GROUP:-e5_robustness}"
RUN_GROUP="${RUN_GROUP:-e5_robustness}"
TB_ROOT="logs/training/${RUN_GROUP}"

ARRIVAL_RATES=(2.0 5.0 8.0 10.0 15.0 20.0 30.0)
SWARM_SIZES=(4 6 8 10 12 16 20)

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
E5A_ONLY=false
E5B_ONLY=false
while [ $# -gt 0 ]; do
    case "$1" in
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --e5a_only)
            E5A_ONLY=true
            shift
            ;;
        --e5b_only)
            E5B_ONLY=true
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

RUN_E5A=true
RUN_E5B=true
$E5A_ONLY && RUN_E5B=false
$E5B_ONLY && RUN_E5A=false

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
    echo "  E5 — Summarize converged tail metrics"
    echo "  tail_iters=${TAIL_ITERS}"
    echo "=========================================================="
    echo ""

    if $RUN_E5A; then
        for rate in "${ARRIVAL_RATES[@]}"; do
            rate_tag="lambda${rate//./_}"
            for aentry in "${LEARN_ALGOS[@]}"; do
                IFS='|' read -r cfg algo <<< "$aentry"
                run_tag="mtan_${WTAG}_${algo}_${rate_tag}"
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

    if $RUN_E5B; then
        for uavs in "${SWARM_SIZES[@]}"; do
            uav_tag="uav${uavs}"
            for aentry in "${LEARN_ALGOS[@]}"; do
                IFS='|' read -r cfg algo <<< "$aentry"
                run_tag="mtan_${WTAG}_${algo}_${uav_tag}"
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
}

echo ""
echo "=========================================================="
echo "  E5 Robustness runtime"
echo "  tensorboard_root=${TB_ROOT}"
echo "  total_cores=${TOTAL_CORES} reserved_cores=${RESERVED_CORES}"
echo "  cpu_threads=${CPU_THREADS} cpu_interop_threads=${CPU_INTEROP_THREADS}"
echo "  max_parallel=${MAX_PARALLEL} tail_iters=${TAIL_ITERS}"
echo "=========================================================="
echo ""
echo "  TensorBoard:"
echo "    tensorboard --logdir ${TB_ROOT} --port 6006 --bind_all"
echo ""

if $RUN_E5A; then
    COUNT_A=$((${#ARRIVAL_RATES[@]} * ${#LEARN_ALGOS[@]}))
    echo ""
    echo "=========================================================="
    echo "  E5a: Arrival rate sweep — Training (${COUNT_A} runs)"
    echo "=========================================================="
    echo ""
    for algo_name in "${LEARN_ALGO_ORDER[@]}"; do
        for aentry in "${LEARN_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            [ "$algo" != "$algo_name" ] && continue
            for rate in "${ARRIVAL_RATES[@]}"; do
                rate_tag="lambda${rate//./_}"
                run_tag="mtan_${WTAG}_${algo}_${rate_tag}"
                log_file="logs/proc_logs/e5_${run_tag}.log"
                cmd="python scripts/train.py \
                    --config ${cfg} \
                    --device cpu \
                    --console_mode plain \
                    --disable_eval \
                    --disable_save \
                    --run_group ${RUN_GROUP} \
                    --run_tag ${run_tag} \
                    --weights ${WEIGHTS} \
                    --arrival_rate ${rate} \
                    --cpu_threads ${CPU_THREADS} \
                    --cpu_interop_threads ${CPU_INTEROP_THREADS} \
                    --num_episodes ${NUM_EPISODES}${SEED:+ --seed ${SEED}}"
                launch_job "${run_tag}" "${log_file}" "${cmd}"
            done
        done
    done
fi

if $RUN_E5B; then
    COUNT_B=$((${#SWARM_SIZES[@]} * ${#LEARN_ALGOS[@]}))
    echo ""
    echo "=========================================================="
    echo "  E5b: Swarm size sweep — Training (${COUNT_B} runs)"
    echo "=========================================================="
    echo ""
    for algo_name in "${LEARN_ALGO_ORDER[@]}"; do
        for aentry in "${LEARN_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            [ "$algo" != "$algo_name" ] && continue
            for uavs in "${SWARM_SIZES[@]}"; do
                uav_tag="uav${uavs}"
                run_tag="mtan_${WTAG}_${algo}_${uav_tag}"
                log_file="logs/proc_logs/e5_${run_tag}.log"
                cmd="python scripts/train.py \
                    --config ${cfg} \
                    --device cpu \
                    --console_mode plain \
                    --disable_eval \
                    --disable_save \
                    --run_group ${RUN_GROUP} \
                    --run_tag ${run_tag} \
                    --weights ${WEIGHTS} \
                    --num_uavs ${uavs} \
                    --cpu_threads ${CPU_THREADS} \
                    --cpu_interop_threads ${CPU_INTEROP_THREADS} \
                    --num_episodes ${NUM_EPISODES}${SEED:+ --seed ${SEED}}"
                launch_job "${run_tag}" "${log_file}" "${cmd}"
            done
        done
    done
fi

echo ""
echo "=========================================================="
echo "  E5 — Deterministic baselines"
echo "=========================================================="
echo ""

if $RUN_E5A; then
    for rate in "${ARRIVAL_RATES[@]}"; do
        rate_tag="lambda${rate//./_}"
        for aentry in "${DET_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            run_name="mtan_${WTAG}_${algo}_${rate_tag}"
            log_file="logs/proc_logs/e5_base_${run_name}.log"
            out_file="${EVAL_OUT}/${run_name}.json"
            cmd="python scripts/evaluate.py \
                --config ${cfg} \
                --weights ${WEIGHTS} \
                --arrival_rate ${rate} \
                --num_episodes ${EVAL_EPISODES} \
                --cpu_threads ${CPU_THREADS} \
                --output ${out_file} \
                ${SEED:+--seed ${SEED}}"
            launch_job "${run_name}" "${log_file}" "${cmd}"
        done
    done
fi

if $RUN_E5B; then
    for uavs in "${SWARM_SIZES[@]}"; do
        uav_tag="uav${uavs}"
        for aentry in "${DET_ALGOS[@]}"; do
            IFS='|' read -r cfg algo <<< "$aentry"
            run_name="mtan_${WTAG}_${algo}_${uav_tag}"
            log_file="logs/proc_logs/e5_base_${run_name}.log"
            out_file="${EVAL_OUT}/${run_name}.json"
            cmd="python scripts/evaluate.py \
                --config ${cfg} \
                --weights ${WEIGHTS} \
                --num_uavs ${uavs} \
                --num_episodes ${EVAL_EPISODES} \
                --cpu_threads ${CPU_THREADS} \
                --output ${out_file} \
                ${SEED:+--seed ${SEED}}"
            launch_job "${run_name}" "${log_file}" "${cmd}"
        done
    done
fi

if ! $DRY_RUN; then
    wait_for_all_jobs "E5 all jobs (learning + deterministic baselines)"
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "[ERROR] E5 had one or more failed process(es). Check logs/proc_logs/."
        exit 1
    fi
    ACTIVE_PIDS=()
    FAILED_COUNT=0
fi

summarize_learning_runs

echo ""
echo "=========================================================="
echo "  E5 artifacts ready at: ${EVAL_OUT}/"
echo "  learning algos: tail-summary JSON from metrics_long.csv"
echo "  deterministic baselines: evaluation JSON"
echo "=========================================================="
echo ""
