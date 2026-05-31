"""
Pre-training environment diagnostics for all model variants.

This script performs a short dynamic check of the environment pipeline for
MTAN, Split, Dense, and Cross-Stitch. It validates model-graph costs,
feature sizes, topology/rate ranges, and several action scenarios to catch
obvious anomalies before long training runs.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import UAVEnv  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/experiments/main_gnn_ppo_tmc_stable.yaml",
    "configs/experiments/main_gnn_ppo_tmc_stable_split.yaml",
    "configs/experiments/main_gnn_ppo_tmc_stable_dense.yaml",
    "configs/experiments/main_gnn_ppo_tmc_stable_cross.yaml",
]


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)


def fmt_mbps(x: float) -> float:
    return float(x) / 1e6


def finite_range(arr: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.min(arr)), float(np.max(arr))


def choose_neighbor(env: UAVEnv) -> Optional[int]:
    rates = np.asarray(env.uav_rate[env.source_uav], dtype=np.float64).copy()
    if rates.size == 0:
        return None
    rates[env.source_uav] = 0.0
    idx = int(np.argmax(rates))
    if rates[idx] <= 0.0:
        return None
    return idx


def build_action(env: UAVEnv, scenario: str) -> Optional[Dict[str, np.ndarray]]:
    num_nodes = len(env.model_graph.node_names)
    num_groups = len(env.group_names)
    full_idx = int(np.argmax(np.asarray(env.bit_widths)))
    low_idx = int(np.argmin(np.asarray(env.bit_widths)))

    if scenario == "local_full":
        return {
            "partition": np.zeros(num_nodes - 1, dtype=np.int32),
            "routing": np.full(num_nodes, env.source_uav, dtype=np.int32),
            "compression": np.full(num_groups, full_idx, dtype=np.int32),
        }

    if scenario == "cut_local":
        return {
            "partition": np.ones(num_nodes - 1, dtype=np.int32),
            "routing": np.full(num_nodes, env.source_uav, dtype=np.int32),
            "compression": np.full(num_groups, full_idx, dtype=np.int32),
        }

    if scenario == "remote_lowbw":
        neighbor = choose_neighbor(env)
        if neighbor is None:
            return None
        routing = np.array(
            [env.source_uav if i % 2 == 0 else neighbor for i in range(num_nodes)],
            dtype=np.int32,
        )
        return {
            "partition": np.ones(num_nodes - 1, dtype=np.int32),
            "routing": routing,
            "compression": np.full(num_groups, low_idx, dtype=np.int32),
        }

    if scenario == "random":
        sampled = env.action_space.sample()
        return {
            "partition": np.asarray(sampled["partition"], dtype=np.int32),
            "routing": np.asarray(sampled["routing"], dtype=np.int32),
            "compression": np.asarray(sampled["compression"], dtype=np.int32),
        }

    raise ValueError(f"Unknown scenario: {scenario}")


def add_finite_checks(name: str, value, errors: List[str]) -> None:
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        errors.append(f"{name} contains non-finite values.")


def run_scenario(env: UAVEnv, scenario: str) -> ScenarioResult:
    env.reset(seed=0)
    action = build_action(env, scenario)
    if action is None:
        return ScenarioResult(
            name=scenario,
            ok=False,
            errors=["No one-hop neighbor available from source UAV for remote scenario."],
        )

    parsed = env._parse_action(action)
    metrics, info, final_action, block_graph = env._simulate_epoch(parsed)
    routing = final_action["routing"]
    compression = final_action["compression"]
    theta = env._compute_theta(block_graph, routing)

    errors: List[str] = []
    warnings: List[str] = []

    node_flops = np.asarray(env.model_graph.node_flops, dtype=np.float64)
    node_bytes = np.asarray(env.model_graph.node_out_bytes, dtype=np.float64)
    block_flops = np.asarray(block_graph.block_flops, dtype=np.float64)
    rho = np.asarray(metrics["rho"], dtype=np.float64)
    positive_rates = np.asarray(env.uav_rate[env.uav_rate > 0], dtype=np.float64)
    proc_delay = np.zeros(block_graph.num_blocks, dtype=np.float64)
    for b in range(block_graph.num_blocks):
        u = int(routing[b])
        proc_delay[b] = block_flops[b] / max(float(env.uav_compute[u]), 1e-9) * (1.0 + float(env.chi) * rho[u])

    add_finite_checks("node_flops", node_flops, errors)
    add_finite_checks("node_out_bytes", node_bytes, errors)
    add_finite_checks("block_flops", block_flops, errors)
    add_finite_checks("rho", rho, errors)
    add_finite_checks("proc_delay", proc_delay, errors)
    add_finite_checks("theta", theta, errors)

    if np.min(node_flops) <= 0:
        errors.append("node_flops has non-positive entries.")
    if np.min(node_bytes) <= 0:
        errors.append("node_out_bytes has non-positive entries.")
    if block_graph.num_blocks <= 0:
        errors.append("block_graph has no blocks.")
    if positive_rates.size == 0:
        errors.append("All inter-UAV rates are zero.")
    if float(metrics["delay"]) <= 0:
        errors.append("Delay is non-positive.")
    if float(info["energy"]) <= 0:
        errors.append("Energy objective is non-positive.")
    if float(metrics["accuracy"]) < 0.0 or float(metrics["accuracy"]) > 1.0:
        errors.append("Accuracy is outside [0, 1].")
    if float(metrics["rho_max"]) < 0.0:
        errors.append("rho_max is negative.")

    if scenario == "local_full" and float(metrics["comm_volume_total"]) > 1e-6:
        errors.append("Local full scenario should have zero communication volume.")
    if scenario == "cut_local" and float(metrics["comm_volume_total"]) > 1e-6:
        errors.append("Cut-local scenario should have zero communication volume.")
    if scenario == "remote_lowbw":
        if float(metrics["comm_volume_total"]) <= 0.0:
            errors.append("Remote low-bitwidth scenario did not generate communication volume.")
        if np.max(theta) <= 1e-8:
            errors.append("Remote low-bitwidth scenario generated no active quantization theta.")
        if float(metrics["accuracy"]) >= 0.999999:
            warnings.append("Remote low-bitwidth scenario still shows near-perfect surrogate accuracy.")

    if float(metrics["rho_max"]) > 1.5:
        warnings.append(f"rho_max is high ({float(metrics['rho_max']):.3f}).")
    if positive_rates.size > 0 and fmt_mbps(float(np.max(positive_rates))) > 600.0:
        warnings.append(
            f"Peak one-hop rate is very high ({fmt_mbps(float(np.max(positive_rates))):.2f} Mbps)."
        )
    if float(metrics["delay"]) > 120.0:
        warnings.append(f"Delay is large ({float(metrics['delay']):.3f} s).")

    used_uavs = len(set(int(x) for x in routing[: block_graph.num_blocks]))
    scenario_summary = {
        "blocks": float(block_graph.num_blocks),
        "used_uavs": float(used_uavs),
        "delay_s": float(metrics["delay"]),
        "energy_obj_j": float(info["energy"]),
        "accuracy": float(metrics["accuracy"]),
        "rho_max": float(metrics["rho_max"]),
        "comm_bytes": float(metrics["comm_volume_total"]),
        "theta_max": float(np.max(theta)) if theta.size > 0 else 0.0,
        "bitwidth_min": float(min(env.bit_widths[int(i)] for i in compression)) if compression.size > 0 else 0.0,
        "proc_delay_max_s": float(np.max(proc_delay)) if proc_delay.size > 0 else 0.0,
    }
    return ScenarioResult(
        name=scenario,
        ok=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        summary=scenario_summary,
    )


def inspect_model_graph(env: UAVEnv) -> Tuple[List[str], Dict[str, float]]:
    errors: List[str] = []
    node_flops = np.asarray(env.model_graph.node_flops, dtype=np.float64)
    node_bytes = np.asarray(env.model_graph.node_out_bytes, dtype=np.float64)
    if not np.all(np.isfinite(node_flops)):
        errors.append("node_flops contains non-finite values.")
    if not np.all(np.isfinite(node_bytes)):
        errors.append("node_out_bytes contains non-finite values.")
    if np.min(node_flops) <= 0:
        errors.append("node_flops has non-positive entries.")
    if np.min(node_bytes) <= 0:
        errors.append("node_out_bytes has non-positive entries.")

    graph_summary = {
        "nodes": float(len(env.model_graph.node_names)),
        "groups": float(len(env.group_names)),
        "node_flops_min": float(np.min(node_flops)),
        "node_flops_max": float(np.max(node_flops)),
        "node_bytes_min": float(np.min(node_bytes)),
        "node_bytes_max": float(np.max(node_bytes)),
    }
    return errors, graph_summary


def inspect_topology(env: UAVEnv) -> Tuple[List[str], Dict[str, float]]:
    errors: List[str] = []
    rates = np.asarray(env.uav_rate, dtype=np.float64)
    positive = rates[rates > 0]
    if positive.size == 0:
        errors.append("No positive one-hop rates in topology.")

    topo_summary = {
        "uavs": float(env.num_uavs),
        "positive_links": float(np.count_nonzero(rates > 0)),
        "rate_min_mbps": fmt_mbps(float(np.min(positive))) if positive.size > 0 else 0.0,
        "rate_max_mbps": fmt_mbps(float(np.max(positive))) if positive.size > 0 else 0.0,
        "rate_mean_mbps": fmt_mbps(float(np.mean(positive))) if positive.size > 0 else 0.0,
    }
    return errors, topo_summary


def inspect_capabilities(env: UAVEnv, config: Dict) -> Tuple[List[str], List[str], Dict[str, float]]:
    errors: List[str] = []
    warnings: List[str] = []
    compute = np.asarray(env.uav_compute, dtype=np.float64)
    epsilon = np.asarray(env.uav_epsilon_cpu, dtype=np.float64)
    hover = np.asarray(env.uav_hover_power, dtype=np.float64)
    battery = np.asarray(env.uav_battery_init, dtype=np.float64)
    reserve = float(config.get("energy", {}).get("min_battery", 0.0))
    episode_time = float(config.get("training", {}).get("max_steps_per_episode", 100)) * float(env.delta_t)
    usable = battery - reserve
    hover_need = hover * episode_time
    worker_compute = np.delete(compute, int(env.source_uav)) if compute.size > 1 else compute.copy()
    worker_epsilon = np.delete(epsilon, int(env.source_uav)) if epsilon.size > 1 else epsilon.copy()
    source_compute = float(compute[int(env.source_uav)])
    source_epsilon = float(epsilon[int(env.source_uav)])
    worker_median = float(np.median(worker_compute)) if worker_compute.size > 0 else source_compute
    worker_epsilon_median = float(np.median(worker_epsilon)) if worker_epsilon.size > 0 else source_epsilon
    margin = usable - hover_need
    corr = 0.0
    if compute.size >= 2 and np.std(compute) > 1e-9 and np.std(epsilon) > 1e-15:
        corr = float(np.corrcoef(compute, epsilon)[0, 1])

    if np.any(usable <= 0):
        errors.append("Some UAVs have non-positive usable battery after reserve.")
    if np.any(margin <= 0):
        errors.append("Some UAVs cannot survive one episode under hover-only baseline.")
    if np.any(epsilon <= 0):
        errors.append("Some UAVs have non-positive epsilon_cpu.")
    if worker_compute.size > 0 and source_compute > 1.05 * worker_median:
        warnings.append(
            "Source UAV compute is not lower than worker median; local execution may become overly dominant."
        )
    if corr < -0.15:
        warnings.append(
            "compute vs. epsilon_cpu correlation is negative; faster UAVs may also be more energy-efficient per FLOP."
        )
    if float(np.min(margin)) < 2.0e4:
        warnings.append(
            f"Hover-only battery margin is tight ({float(np.min(margin)):.1f} J)."
        )

    summary = {
        "source_compute_gflops": float(source_compute / 1e9),
        "source_epsilon_pj_per_flop": float(source_epsilon * 1e12),
        "worker_compute_median_gflops": float(worker_median / 1e9),
        "worker_epsilon_median_pj_per_flop": float(worker_epsilon_median * 1e12),
        "compute_min_gflops": float(np.min(compute) / 1e9),
        "compute_max_gflops": float(np.max(compute) / 1e9),
        "epsilon_min_pj_per_flop": float(np.min(epsilon) * 1e12),
        "epsilon_max_pj_per_flop": float(np.max(epsilon) * 1e12),
        "compute_epsilon_corr": float(corr),
        "hover_margin_min_j": float(np.min(margin)),
        "hover_margin_mean_j": float(np.mean(margin)),
    }
    return errors, warnings, summary


def print_section(title: str) -> None:
    print("=" * 88)
    print(title)
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description="Pre-training diagnostics for UAV-MTM environment.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config paths to check.",
    )
    args = parser.parse_args()

    all_ok = True
    scenarios = ["local_full", "cut_local", "remote_lowbw", "random"]

    for cfg_path in args.configs:
        config = load_config(cfg_path)
        env = UAVEnv(config)
        env.reset(seed=0)
        model_name = str(config.get("model", {}).get("name", "unknown"))

        print_section(f"{cfg_path} | model={model_name}")

        graph_errors, graph_summary = inspect_model_graph(env)
        topo_errors, topo_summary = inspect_topology(env)
        cap_errors, cap_warnings, cap_summary = inspect_capabilities(env, config)

        print("[graph]", graph_summary)
        print("[topology]", topo_summary)
        print("[capability]", cap_summary)
        for msg in graph_errors + topo_errors + cap_errors:
            print("ERROR:", msg)
        for msg in cap_warnings:
            print("WARN:", msg)
        if graph_errors or topo_errors or cap_errors:
            all_ok = False

        for scenario in scenarios:
            result = run_scenario(env, scenario)
            status = "PASS" if result.ok else "FAIL"
            print(f"[{status}] scenario={scenario} summary={result.summary}")
            for msg in result.warnings:
                print("WARN:", msg)
            for msg in result.errors:
                print("ERROR:", msg)
            if not result.ok:
                all_ok = False

    if all_ok:
        print_section("PRE-FLIGHT CHECK PASSED")
        return 0

    print_section("PRE-FLIGHT CHECK FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
