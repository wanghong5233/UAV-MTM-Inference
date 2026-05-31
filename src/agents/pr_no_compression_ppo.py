"""
Partition-routing PPO baseline with full-precision features.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from src.core import AgentRegistry
from .gnn_ppo import GNNPPO


@AgentRegistry.register("pr_no_compression_ppo")
class PRNoCompressionPPO(GNNPPO):
    """Train PPO over partition and routing while freezing compression."""

    def __init__(self, env: gym.Env, config: Dict):
        super().__init__(env, config)
        algo_cfg = config.get("algorithm", {})
        ablation_cfg = algo_cfg.get("ablation", {})
        bit_widths = list(getattr(env, "bit_widths", [32, 16, 8, 4]))
        fixed_bw = ablation_cfg.get("fixed_bitwidth", None)
        if fixed_bw is None:
            self.fixed_compression_index = int(np.argmax(np.asarray(bit_widths, dtype=np.int32))) if bit_widths else 0
        else:
            if int(fixed_bw) not in bit_widths:
                raise ValueError(f"Unknown fixed_bitwidth={fixed_bw}. Available: {bit_widths}")
            self.fixed_compression_index = int(bit_widths.index(int(fixed_bw)))

    def _fixed_compression_action(self) -> torch.Tensor:
        return torch.full(
            (self.num_groups,),
            int(self.fixed_compression_index),
            dtype=torch.long,
            device=self.device,
        )

    def _sample_actions(self, state: Dict, deterministic: bool = False) -> Tuple[Dict, Dict]:
        actions, extra = super()._sample_actions(state, deterministic=deterministic)
        actions["compression"] = self._fixed_compression_action()
        extra["compression_log_prob"] = torch.tensor(0.0, device=self.device)
        return actions, extra

    def _evaluate_actions(
        self,
        state: Dict,
        action: Dict,
        static_eval: Optional[Dict] = None,
        encoded: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoded is None:
            node_emb, uav_emb, graph_emb, swarm_emb, ctx = self._encode_graphs(state)
        else:
            node_emb = encoded["node_emb"]
            uav_emb = encoded["uav_emb"]
            graph_emb = encoded["graph_emb"]
            swarm_emb = encoded["swarm_emb"]
            ctx = encoded["ctx"]

        mandatory = self.mandatory_cuts_t
        prev_partition = state["prev_partition"]
        prev_pad = torch.cat([torch.zeros(1, device=self.device), prev_partition], dim=0)
        part_log_prob = torch.tensor(0.0, device=self.device)
        for i in range(1, self.num_nodes):
            if mandatory[i - 1] > 0:
                continue
            logits = self.partition_head(torch.cat([node_emb[i], ctx, prev_pad[i].unsqueeze(0)], dim=-1)).squeeze(-1)
            part_log_prob = part_log_prob + (
                -F.binary_cross_entropy_with_logits(logits, action["partition"][i - 1], reduction="sum")
            )

        if static_eval is not None:
            block_graph = static_eval["block_graph"]
            block_preds = static_eval["block_preds"]
        else:
            part_np = action["partition"].detach().cpu().numpy()
            task_pop = state["task_popularity"].detach().cpu().numpy()
            block_graph = self.env.model_graph.build_blocks(part_np, task_pop)
            block_preds = {b: [] for b in range(block_graph.num_blocks)}
            for bu, bv in block_graph.block_edges:
                block_preds[bv].append(bu)

        routing_log_prob = torch.tensor(0.0, device=self.device)
        routing_entropy = torch.tensor(0.0, device=self.device)
        uav_adj = state["uav_adj"]
        if static_eval is not None:
            arrival_rate = float(static_eval["arrival_rate"])
            compute_cap_t = static_eval["compute_cap_t"]
            battery_t = static_eval["battery_t"]
        else:
            arrival_rate_scale = float(getattr(self.env.state_space, "arrival_rate_max", 1.0))
            arrival_rate = float(state["arrival_rate"].item()) * max(arrival_rate_scale, 1e-6)
            compute_cap_t = state["uav_nodes"][:, 0] * float(self.env.compute_max)
            battery_t = state["uav_nodes"][:, 1] * float(self.env.battery_max)
        cap_limit_t = compute_cap_t * float(self.env.delta_t)
        bat_mask_const = battery_t > float(self.env.energy_min)
        load_u_t = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)

        assigned_uavs = {}
        for b in range(block_graph.num_blocks):
            block_nodes = block_graph.block_nodes[b]
            block_emb = node_emb[block_nodes].mean(dim=0)
            prev_routing = state["prev_routing"][block_nodes].float().mean()
            q = self.routing_q(torch.cat([block_emb, ctx, prev_routing.unsqueeze(0)], dim=-1))
            k = self.routing_k(uav_emb)
            scores = torch.matmul(k, q) / np.sqrt(q.shape[-1])

            mask = torch.ones(self.num_uavs, device=self.device, dtype=torch.bool)
            for p in block_preds[b]:
                u_prev = assigned_uavs.get(p, None)
                if u_prev is None:
                    continue
                link_mask = (uav_adj[u_prev] > 0) | (self.uav_index_t == int(u_prev))
                mask = mask & link_mask
            inc = float(block_graph.block_flops[b] * block_graph.block_pi_act[b] * arrival_rate * self.env.delta_t)
            cap_mask = (load_u_t + inc) <= cap_limit_t
            mask = mask & cap_mask & bat_mask_const
            if not mask.any():
                mask = torch.ones_like(mask)
            scores = scores.masked_fill(~mask, -1e9)
            log_probs = torch.log_softmax(scores, dim=-1)
            probs = torch.exp(log_probs)
            act = action["routing"][b]
            routing_log_prob = routing_log_prob + log_probs.gather(0, act.long().view(1)).squeeze(0)
            routing_entropy = routing_entropy + (-(probs * log_probs).sum())
            u_sel = int(act.item())
            assigned_uavs[b] = u_sel
            load_u_t[u_sel] = load_u_t[u_sel] + inc

        log_prob = part_log_prob + routing_log_prob
        entropy = routing_entropy
        value = self.value_head(torch.cat([graph_emb, swarm_emb, ctx], dim=-1)).squeeze(-1)
        return log_prob, value, entropy
