"""
Analyze GNN-PPO gradient flow, graph connectivity, and environment metric ranges.

This script provides a math/engineering sanity check before long training:
1) Verify PPO update changes GNN parameters.
2) Check model DAG validity across supported models.
3) Evaluate environment outputs under corner-case actions.
4) Decompose reward terms and inspect magnitude balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.env import UAVEnv
from src.core import AgentRegistry
import src.agents  # noqa: F401


@dataclass
class GraphStats:
    model_name: str
    num_nodes: int
    num_edges: int
    is_dag: bool
    weakly_connected: bool
    num_groups: int
    mandatory_cuts: int
    roots: int
    leaves: int


def is_dag(num_nodes: int, edges: List[Tuple[int, int]]) -> bool:
    indeg = [0] * num_nodes
    outs: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        outs[u].append(v)
        indeg[v] += 1
    q = [i for i in range(num_nodes) if indeg[i] == 0]
    visited = 0
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        visited += 1
        for v in outs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return visited == num_nodes


def weakly_connected(num_nodes: int, edges: List[Tuple[int, int]]) -> bool:
    if num_nodes <= 1:
        return True
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * num_nodes
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return all(seen)


def analyze_graphs(base_cfg_path: str) -> List[GraphStats]:
    model_cfgs = [
        ("mtan", "configs/experiments/main_gnn_ppo_tmc_stable.yaml"),
        ("split", "configs/experiments/main_gnn_ppo_tmc_stable_split.yaml"),
        ("dense", "configs/experiments/main_gnn_ppo_tmc_stable_dense.yaml"),
        ("cross", "configs/experiments/main_gnn_ppo_tmc_stable_cross.yaml"),
    ]
    out = []
    for model_name, cfg_path in model_cfgs:
        cfg = load_config(cfg_path)
        cfg["device"] = "cpu"
        env = UAVEnv(cfg)
        mg = env.model_graph
        n = len(mg.node_names)
        m = len(mg.edges)
        indeg = mg.in_deg
        out.append(
            GraphStats(
                model_name=model_name,
                num_nodes=n,
                num_edges=m,
                is_dag=is_dag(n, mg.edges),
                weakly_connected=weakly_connected(n, mg.edges),
                num_groups=len(mg.compression_groups),
                mandatory_cuts=int(np.sum(mg.mandatory_cuts)),
                roots=int(np.sum(indeg == 0)),
                leaves=int(np.sum(mg.out_deg == 0)),
            )
        )
    return out


def collect_rollout(agent, env, steps: int = 16):
    s, _ = env.reset(seed=0)
    for _ in range(steps):
        a = agent.select_action(s, deterministic=False)
        ns, r, d, t, info = env.step(a)
        agent.store_transition(s, a, r, ns, d or t, info)
        s = ns
        if d or t:
            s, _ = env.reset(seed=0)


def l2_param_delta(before: Dict[str, torch.Tensor], after_module: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for n, p in after_module.named_parameters():
        k = f"{prefix}.{n}"
        if k not in before:
            continue
        d = (p.detach().cpu() - before[k]).float()
        total += float(torch.sum(d * d).item())
    return float(np.sqrt(total))


def analyze_gradient_flow(cfg_path: str) -> Dict[str, float]:
    cfg = load_config(cfg_path)
    cfg["device"] = "cpu"
    cfg["algorithm"]["ppo"]["n_steps"] = 16
    cfg["algorithm"]["ppo"]["n_epochs"] = 2
    cfg["algorithm"]["ppo"]["batch_size"] = 8
    cfg["training"]["max_steps_per_episode"] = 8

    env = UAVEnv(cfg)
    agent = AgentRegistry.create(cfg["algorithm"]["name"], env, cfg)
    collect_rollout(agent, env, steps=16)

    before = {f"model_encoder.{n}": p.detach().cpu().clone() for n, p in agent.model_encoder.named_parameters()}
    before.update({f"uav_encoder.{n}": p.detach().cpu().clone() for n, p in agent.uav_encoder.named_parameters()})
    before.update({f"partition_head.{n}": p.detach().cpu().clone() for n, p in agent.partition_head.named_parameters()})
    before.update({f"routing_q.{n}": p.detach().cpu().clone() for n, p in agent.routing_q.named_parameters()})
    before.update({f"routing_k.{n}": p.detach().cpu().clone() for n, p in agent.routing_k.named_parameters()})
    before.update({f"group_head.{n}": p.detach().cpu().clone() for n, p in agent.group_head.named_parameters()})
    before.update({f"value_head.{n}": p.detach().cpu().clone() for n, p in agent.value_head.named_parameters()})

    metrics = agent.update()
    deltas = {
        "delta_model_encoder": l2_param_delta(before, agent.model_encoder, "model_encoder"),
        "delta_uav_encoder": l2_param_delta(before, agent.uav_encoder, "uav_encoder"),
        "delta_partition_head": l2_param_delta(before, agent.partition_head, "partition_head"),
        "delta_routing_q": l2_param_delta(before, agent.routing_q, "routing_q"),
        "delta_routing_k": l2_param_delta(before, agent.routing_k, "routing_k"),
        "delta_group_head": l2_param_delta(before, agent.group_head, "group_head"),
        "delta_value_head": l2_param_delta(before, agent.value_head, "value_head"),
    }
    deltas.update({f"update_{k}": float(v) for k, v in metrics.items()})
    return deltas


def build_corner_actions(env: UAVEnv) -> Dict[str, Dict[str, np.ndarray]]:
    num_nodes = len(env.model_graph.node_names)
    num_groups = len(env.group_names)
    num_uavs = env.num_uavs
    # Partition: no extra cuts (mandatory cuts still enforced)
    partition_none = np.zeros(num_nodes - 1, dtype=np.int32)
    partition_all = np.ones(num_nodes - 1, dtype=np.int32)

    # Routing by node index placeholder, env will truncate by block count.
    routing_source = np.zeros(num_nodes, dtype=np.int32)
    routing_roundrobin = np.array([i % num_uavs for i in range(num_nodes)], dtype=np.int32)

    # Compression index: 0->32bit, last->lowest bit
    comp_high = np.zeros(num_groups, dtype=np.int32)
    comp_low = np.full(num_groups, fill_value=len(env.bit_widths) - 1, dtype=np.int32)

    return {
        "high_precision_localish": {
            "partition": partition_none,
            "routing": routing_source,
            "compression": comp_high,
        },
        "low_precision_localish": {
            "partition": partition_none,
            "routing": routing_source,
            "compression": comp_low,
        },
        "high_precision_more_split_rr": {
            "partition": partition_all,
            "routing": routing_roundrobin,
            "compression": comp_high,
        },
        "low_precision_more_split_rr": {
            "partition": partition_all,
            "routing": routing_roundrobin,
            "compression": comp_low,
        },
    }


def evaluate_env_corners(cfg_path: str) -> Dict[str, Dict[str, float]]:
    cfg = load_config(cfg_path)
    cfg["device"] = "cpu"
    cfg["simulation"]["repair_routing"] = True
    cfg["simulation"]["repair_accuracy"] = True
    env = UAVEnv(cfg)
    env.reset(seed=0)

    out: Dict[str, Dict[str, float]] = {}
    actions = build_corner_actions(env)
    for name, action in actions.items():
        parsed = env._parse_action(action)
        metrics, info, final_action, block_graph = env._simulate_epoch(parsed)
        reward, reward_info = env.reward_calculator.compute_reward(
            delay=metrics["delay"],
            energy=metrics["energy_obj"],
            accuracy=metrics["accuracy"],
            preference=env.preference,
            rho=metrics["rho"],
            battery_violation=0.0,
            num_blocks=metrics["num_blocks"],
            a_min=env.a_min,
            delta_a=env.delta_a,
            deadline=env.deadline,
            energy_budget=None,
        )
        out[name] = {
            "num_blocks": float(metrics["num_blocks"]),
            "delay": float(metrics["delay"]),
            "energy_obj": float(metrics["energy_obj"]),
            "accuracy": float(metrics["accuracy"]),
            "rho_max": float(np.max(metrics["rho"])),
            "comm_volume_total": float(metrics.get("comm_volume_total", 0.0)),
            "reward": float(reward),
            "delay_norm": float(reward_info["delay_norm"]),
            "energy_norm": float(reward_info["energy_norm"]),
            "penalty_q": float(reward_info["penalty_q"]),
            "accuracy_reward": float(reward_info["accuracy_reward"]),
            "routing_blocks_used": float(len(final_action["routing"])),
            "bitwidth_min": float(np.min([env.bit_widths[int(i)] for i in final_action["compression"]])),
            "bitwidth_max": float(np.max([env.bit_widths[int(i)] for i in final_action["compression"]])),
        }
    return out


def random_rollout_stats(cfg_path: str, episodes: int = 20, steps_per_ep: int = 50):
    cfg = load_config(cfg_path)
    cfg["device"] = "cpu"
    cfg["training"]["max_steps_per_episode"] = steps_per_ep
    env = UAVEnv(cfg)

    all_delay = []
    all_energy = []
    all_acc = []
    all_reward = []
    all_rho = []
    all_blocks = []
    for ep in range(episodes):
        s, _ = env.reset(seed=ep)
        for _ in range(steps_per_ep):
            a = env.action_space.sample()
            s, r, d, t, info = env.step(a)
            all_delay.append(float(info["delay"]))
            all_energy.append(float(info["energy"]))
            all_acc.append(float(info["accuracy"]))
            all_reward.append(float(r))
            all_rho.append(float(info.get("rho_max", 0.0)))
            all_blocks.append(float(info.get("num_blocks", 0.0)))
            if d or t:
                break

    def stat(x):
        x = np.asarray(x, dtype=np.float64)
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p05": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    return {
        "delay": stat(all_delay),
        "energy": stat(all_energy),
        "accuracy": stat(all_acc),
        "reward": stat(all_reward),
        "rho_max": stat(all_rho),
        "num_blocks": stat(all_blocks),
    }


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    cfg_path = "configs/experiments/main_gnn_ppo_tmc_stable.yaml"

    print_section("A) Graph Structure Check")
    stats = analyze_graphs(cfg_path)
    for s in stats:
        print(
            f"{s.model_name:>5} | nodes={s.num_nodes:>3} edges={s.num_edges:>4} groups={s.num_groups:>2} "
            f"mandatory_cuts={s.mandatory_cuts:>2} roots={s.roots:>2} leaves={s.leaves:>2} "
            f"DAG={s.is_dag} connected={s.weakly_connected}"
        )

    print_section("B) Gradient / Parameter Update Check (MTAN)")
    deltas = analyze_gradient_flow(cfg_path)
    for k, v in deltas.items():
        print(f"{k:>24}: {v:.6e}")

    print_section("C) Environment Corner Action Check (MTAN)")
    corners = evaluate_env_corners(cfg_path)
    for name, m in corners.items():
        print(
            f"{name:>28} | blocks={m['num_blocks']:.0f} delay={m['delay']:.4f} "
            f"energy={m['energy_obj']:.2f} acc={m['accuracy']:.4f} "
            f"rho_max={m['rho_max']:.4f} reward={m['reward']:.4f} "
            f"bw[{m['bitwidth_min']:.0f},{m['bitwidth_max']:.0f}]"
        )

    print_section("D) Random Rollout Distribution Check (MTAN)")
    rollout = random_rollout_stats(cfg_path, episodes=20, steps_per_ep=50)
    for key, st in rollout.items():
        print(
            f"{key:>10} | mean={st['mean']:.4f} std={st['std']:.4f} "
            f"p05={st['p05']:.4f} p50={st['p50']:.4f} p95={st['p95']:.4f} "
            f"min={st['min']:.4f} max={st['max']:.4f}"
        )


if __name__ == "__main__":
    main()
