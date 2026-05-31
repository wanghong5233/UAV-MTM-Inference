#!/bin/bash
# run_main_table_e3.sh — Cross-model main-table experiment (E2 in TMC plan).
#
# Runs the 4 models (split/mtan/dense/cross) x 5 algorithms (proposed,
# partition_routing_only, mlp_ppo, local_only, single_split) at the default
# operating point. Produces the data backing the main numerical table
# (tab:main_results) and the 4 per-model bar figures
# (fig:cross_model_{split,mtan,dense,cross}).
#
# Learning algorithms:
#   - pure training only
#   - no checkpoint saving, no periodic evaluation
#   - paper numbers extracted from metrics_long.csv tail statistics
#
# Deterministic baselines:
#   - quick evaluation only

set -euo pipefail

CPU_THREADS="${CPU_THREADS:-2}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"
TOTAL_CORES="${TOTAL_CORES:-$(nproc 2>/dev/null || echo 32)}"
RESERVED_CORES="${RESERVED_CORES:-2}"
NUM_EPISODES="${NUM_EPISODES:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
TAIL_ITERS="${TAIL_ITERS:-100}"
WEIGHTS="${WEIGHTS:-0.1 0.9}"
WTAG="${WTAG:-w19}"
SEED="${SEED:-789}"
RUN_GROUP="${RUN_GROUP:-e2_cross_model_w19}"
EVAL_OUT="results/eval/${RUN_GROUP}"
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

MODELS=(
    "main_gnn_ppo_tmc_stable_split.yaml|baseline_pr_no_compression_ppo_tmc_stable_split.yaml|baseline_mlp_ppo_tmc_stable_split.yaml|baseline_local_only_tmc_stable_split.yaml|baseline_single_split_tmc_stable_split.yaml|split"
    "main_gnn_ppo_tmc_stable.yaml|baseline_pr_no_compression_ppo_tmc_stable.yaml|baseline_mlp_ppo_tmc_stable.yaml|baseline_local_only_tmc_stable.yaml|baseline_single_split_tmc_stable.yaml|mtan"
    "main_gnn_ppo_tmc_stable_dense.yaml|baseline_pr_no_compression_ppo_tmc_stable_dense.yaml|baseline_mlp_ppo_tmc_stable_dense.yaml|baseline_local_only_tmc_stable_dense.yaml|baseline_single_split_tmc_stable_dense.yaml|dense"
    "main_gnn_ppo_tmc_stable_cross.yaml|baseline_pr_no_compression_ppo_tmc_stable_cross.yaml|baseline_mlp_ppo_tmc_stable_cross.yaml|baseline_local_only_tmc_stable_cross.yaml|baseline_single_split_tmc_stable_cross.yaml|cross"
)

DRY_RUN=false
BASELINES_ONLY=false
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
    echo "  E3 Main Table — Phase 1b: Summarize converged tail metrics"
    echo "  tail_iters=${TAIL_ITERS}"
    echo "=========================================================="
    echo ""

    for mentry in "${MODELS[@]}"; do
        IFS='|' read -r cfg_proposed cfg_pr cfg_mlp cfg_lo cfg_ss model <<< "$mentry"
        for pair in "proposed|${cfg_proposed}" "partition_routing_only|${cfg_pr}" "mlp_ppo|${cfg_mlp}"; do
            IFS='|' read -r algo cfg <<< "$pair"
            run_tag="${model}_${WTAG}_${algo}"
            out_file="${EVAL_OUT}/${run_tag}.json"
            cmd="python scripts/summarize_converged_metrics.py \
                --config configs/experiments/${cfg} \
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
echo "  E3 Main Table runtime"
echo "  tensorboard_root=${TB_ROOT}"
echo "  run_group=${RUN_GROUP}"
echo "  weights=${WEIGHTS} (${WTAG})"
echo "  seed=${SEED:-default}"
echo "  total_cores=${TOTAL_CORES} reserved_cores=${RESERVED_CORES}"
echo "  cpu_threads=${CPU_THREADS} cpu_interop_threads=${CPU_INTEROP_THREADS}"
echo "  max_parallel=${MAX_PARALLEL} tail_iters=${TAIL_ITERS}"
echo "=========================================================="
echo ""
echo "  TensorBoard:"
echo "    tensorboard --logdir ${TB_ROOT} --port 6006 --bind_all"
echo ""

# ---- Phase 1: learning runs (queued, do NOT wait) ----
if ! $BASELINES_ONLY; then
    echo ""
    echo "=========================================================="
    echo "  E2 Main Table — Phase 1: Learning runs (12 jobs queued)"
    echo "  No checkpoint saving, no learning-time evaluation"
    echo "=========================================================="
    echo ""

    for mentry in "${MODELS[@]}"; do
        IFS='|' read -r cfg_proposed cfg_pr cfg_mlp cfg_lo cfg_ss model <<< "$mentry"
        for pair in "proposed|${cfg_proposed}" "partition_routing_only|${cfg_pr}" "mlp_ppo|${cfg_mlp}"; do
            IFS='|' read -r algo cfg <<< "$pair"
            run_tag="${model}_${WTAG}_${algo}"
            log_file="logs/proc_logs/e3_${run_tag}.log"
            cmd="python scripts/train.py \
                --config configs/experiments/${cfg} \
                --device cpu \
                --console_mode plain \
                --disable_eval \
                --disable_save \
                --run_group ${RUN_GROUP} \
                --run_tag ${run_tag} \
                --weights ${WEIGHTS} \
                --cpu_threads ${CPU_THREADS} \
                --cpu_interop_threads ${CPU_INTEROP_THREADS} \
                --num_episodes ${NUM_EPISODES}${SEED:+ --seed ${SEED}}"
            launch_job "${run_tag}" "${log_file}" "${cmd}"
        done
    done
fi

# ---- Phase 2: deterministic baselines (queued into same pool as Phase 1) ----
# Single global queue: whenever any learning run finishes early, MAX_PARALLEL
# frees a slot and the next deterministic baseline starts immediately.
# This prevents tail-end CPU idleness while the slowest learning run is still
# converging.
echo ""
echo "=========================================================="
echo "  E2 Main Table — Phase 2: Deterministic baselines (8 jobs queued)"
echo "  Shares one queue with Phase 1; MAX_PARALLEL=${MAX_PARALLEL} controls concurrency."
echo "=========================================================="
echo ""

for mentry in "${MODELS[@]}"; do
    IFS='|' read -r cfg_proposed cfg_pr cfg_mlp cfg_lo cfg_ss model <<< "$mentry"
    for pair in "local_only|${cfg_lo}" "single_split|${cfg_ss}"; do
        IFS='|' read -r algo cfg <<< "$pair"
        run_name="${model}_${WTAG}_${algo}"
        log_file="logs/proc_logs/e3_base_${model}_${algo}.log"
        out_file="${EVAL_OUT}/${run_name}.json"
        cmd="python scripts/evaluate.py \
            --config configs/experiments/${cfg} \
            --weights ${WEIGHTS} \
            --num_episodes ${EVAL_EPISODES} \
            --cpu_threads ${CPU_THREADS} \
            --output ${out_file} \
            ${SEED:+--seed ${SEED}}"
        launch_job "${run_name}" "${log_file}" "${cmd}"
    done
done

# ---- Single global wait: 20 jobs share one queue ----
if ! $DRY_RUN; then
    wait_for_all_jobs "All E2 jobs (learning + deterministic)"
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "[ERROR] E2 had ${FAILED_COUNT} failed process(es). Check logs/proc_logs/."
        exit 1
    fi
    ACTIVE_PIDS=()
    FAILED_COUNT=0
fi

# ---- Phase 3: summarise learning runs from metrics_long.csv ----
if ! $BASELINES_ONLY; then
    summarize_learning_runs
fi

echo ""
echo "=========================================================="
echo "  E2 artifacts ready at: ${EVAL_OUT}/"
echo "  learning algos: tail-summary JSON from metrics_long.csv"
echo "  deterministic baselines: evaluation JSON"
echo "=========================================================="
echo ""
