"""
GNN-PPO algorithm implementation.

This module provides a lightweight PPO implementation with dual-graph
encoders for model DAG and UAV communication graph.
"""

from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from src.core import BaseAgent, AgentRegistry


class GraphEncoder(nn.Module):
    """Simple message-passing encoder without external dependencies."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.self_layers = nn.ModuleList()
        self.nei_layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.self_layers.append(nn.Linear(in_ch, hidden_dim))
            self.nei_layers.append(nn.Linear(in_ch, hidden_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(self.num_layers):
            # Use incoming-neighbor aggregation for directed graphs:
            # if adj[u, v] = 1 (u -> v), node v aggregates from its predecessors via adj^T.
            adj_in = adj.transpose(-1, -2)
            deg = adj_in.sum(dim=-1, keepdim=True).clamp_min(1.0)
            agg = torch.matmul(adj_in, h) / deg
            h = self.self_layers[i](h) + self.nei_layers[i](agg)
            h = self.activation(h)
            h = self.dropout(h)
        return h


class MLP(nn.Module):
    """Simple MLP builder."""

    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: str = "relu"):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RolloutBuffer:
    """On-policy rollout buffer for PPO."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.states = {}
        self.actions = {"partition": [], "routing": [], "compression": []}
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, state: Dict, action: Dict, log_prob: torch.Tensor, value: torch.Tensor, reward: float, done: bool):
        for key, val in state.items():
            self.states.setdefault(key, []).append(val.detach())
        self.actions["partition"].append(action["partition"])
        self.actions["routing"].append(action["routing"])
        self.actions["compression"].append(action["compression"])
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))

    def get(self) -> Dict:
        batch = {k: torch.stack(v, dim=0) for k, v in self.states.items()}
        batch["actions"] = {
            "partition": torch.stack(self.actions["partition"], dim=0),
            "routing": torch.stack(self.actions["routing"], dim=0),
            "compression": torch.stack(self.actions["compression"], dim=0),
        }
        batch["log_probs"] = torch.stack(self.log_probs, dim=0)
        batch["values"] = torch.stack(self.values, dim=0).squeeze(-1)
        batch["rewards"] = torch.stack(self.rewards, dim=0)
        batch["dones"] = torch.stack(self.dones, dim=0)
        return batch


@AgentRegistry.register('gnn_ppo')
class GNNPPO(BaseAgent, nn.Module):
    """GNN-PPO algorithm with factorized action heads."""
    
    def __init__(self, env: gym.Env, config: Dict):
        nn.Module.__init__(self)
        BaseAgent.__init__(self, env, config)

        algo_cfg = config.get("algorithm", {})
        net_cfg = algo_cfg.get("network", {})
        gnn_cfg = net_cfg.get("gnn", {})
        actor_cfg = net_cfg.get("actor", {})
        critic_cfg = net_cfg.get("critic", {})
        ppo_cfg = algo_cfg.get("ppo", {})

        self.node_feat_dim = env.observation_space["model_nodes"].shape[-1]
        self.uav_feat_dim = env.observation_space["uav_nodes"].shape[-1]
        self.num_nodes = env.observation_space["model_nodes"].shape[0]
        self.num_uavs = env.observation_space["uav_nodes"].shape[0]
        self.num_groups = env.observation_space["prev_compression"].shape[0]
        self.num_bitwidths = env.action_space["compression"].nvec[0]
        self.task_dim = env.observation_space["task_popularity"].shape[0]
        self.ctx_dim = 2 + 1 + self.task_dim

        hidden_dim = int(gnn_cfg.get("hidden_dim", 128))
        num_layers = int(gnn_cfg.get("num_layers", 3))
        dropout = float(gnn_cfg.get("dropout", 0.1))

        self.model_encoder = GraphEncoder(self.node_feat_dim, hidden_dim, num_layers, dropout)
        self.uav_encoder = GraphEncoder(self.uav_feat_dim, hidden_dim, num_layers, dropout)

        self.partition_head = nn.Linear(hidden_dim + self.ctx_dim + 1, 1)
        self.routing_q = nn.Linear(hidden_dim + self.ctx_dim + 1, hidden_dim)
        self.routing_k = nn.Linear(hidden_dim, hidden_dim)

        self.group_head = MLP(hidden_dim + self.ctx_dim + 1, actor_cfg.get("hidden_dims", [128]), self.num_bitwidths)
        self.value_head = MLP(hidden_dim * 2 + self.ctx_dim, critic_cfg.get("hidden_dims", [128]), 1)

        self.init_lr = float(ppo_cfg.get("learning_rate", 3e-4))
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.init_lr,
            eps=float(ppo_cfg.get("adam_eps", 1e-5)),
            weight_decay=float(ppo_cfg.get("weight_decay", 0.0)),
        )

        self.gamma = float(ppo_cfg.get("gamma", 0.99))
        self.gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(ppo_cfg.get("clip_epsilon", 0.2))
        self.value_coef = float(ppo_cfg.get("value_loss_coef", 0.5))
        self.entropy_coef = float(ppo_cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
        self.n_steps = int(ppo_cfg.get("n_steps", 2048))
        self.n_epochs = int(ppo_cfg.get("n_epochs", 10))
        self.batch_size = int(ppo_cfg.get("batch_size", 64))
        self.normalize_advantage = bool(ppo_cfg.get("normalize_advantage", True))
        self.value_clip_enabled = bool(ppo_cfg.get("value_clip_enabled", True))
        self.value_clip_eps = float(ppo_cfg.get("value_clip_epsilon", 0.2))
        self.target_kl = float(ppo_cfg.get("target_kl", 0.03))
        self.lr_schedule = str(ppo_cfg.get("lr_schedule", "constant")).lower()
        train_cfg = config.get("training", {})
        est_total_steps = int(train_cfg.get("num_episodes", 1)) * int(train_cfg.get("max_steps_per_episode", 1))
        self.total_updates_est = max(est_total_steps // max(self.n_steps, 1), 1)
        self.update_calls = 0

        self.buffer = RolloutBuffer(self.device)
        self._cached_action_eval: Optional[Dict[str, torch.Tensor]] = None
        self._cached_state_t: Optional[Dict[str, torch.Tensor]] = None
        self._cached_state_ref = None
        self._cached_action_ref = None

        # Frequently used constant tensors for mask construction.
        self.register_buffer("mandatory_cuts_t", torch.as_tensor(self.env.model_graph.mandatory_cuts, dtype=torch.bool))
        self.register_buffer("uav_index_t", torch.arange(self.num_uavs, dtype=torch.long))

        self.to(self.device)
        
    def select_action(self, state: Dict, deterministic: bool = False) -> Dict:
        with torch.no_grad():
            state_t = self._state_to_tensor(state)
            actions, extra = self._sample_actions(state_t, deterministic=deterministic)
            total_log_prob = (
                extra["partition_log_prob"] + extra["routing_log_prob"] + extra["compression_log_prob"]
            )
            # Cache policy evaluation from action sampling to avoid recomputing
            # log_prob/value in store_transition for the same (state, action).
            self._cached_action_eval = {
                "partition": actions["partition"].detach().clone(),
                "routing": actions["routing"].detach().clone(),
                "compression": actions["compression"].detach().clone(),
                "log_prob": total_log_prob.detach().clone(),
                "value": extra["value"].detach().clone(),
            }
            actions_np = {k: v.detach().cpu().numpy() for k, v in actions.items()}
            self._cached_state_t = state_t
            self._cached_state_ref = state
            self._cached_action_ref = actions_np
        return actions_np

    def store_transition(self, state, action, reward, next_state, done, info=None):
        cache = self._cached_action_eval
        cached_ref_hit = (
            cache is not None
            and self._cached_state_t is not None
            and self._cached_state_ref is state
            and self._cached_action_ref is action
        )

        if cached_ref_hit:
            state_t = self._cached_state_t
            action_t = {
                "partition": cache["partition"],
                "routing": cache["routing"],
                "compression": cache["compression"],
            }
            log_prob = cache["log_prob"]
            value = cache["value"]
        else:
            state_t = self._state_to_tensor(state)
            action_t = self._action_to_tensor(action)

        if (not cached_ref_hit) and (
            cache is not None
            and cache["partition"].shape == action_t["partition"].shape
            and cache["routing"].shape == action_t["routing"].shape
            and cache["compression"].shape == action_t["compression"].shape
            and torch.equal(cache["partition"], action_t["partition"])
            and torch.equal(cache["routing"], action_t["routing"])
            and torch.equal(cache["compression"], action_t["compression"])
        ):
            log_prob = cache["log_prob"]
            value = cache["value"]
        elif not cached_ref_hit:
            log_prob, value, _ = self._evaluate_actions(state_t, action_t)
        self._cached_action_eval = None
        self._cached_state_t = None
        self._cached_state_ref = None
        self._cached_action_ref = None
        self.buffer.add(state_t, action_t, log_prob, value, reward, done)

    def update(
        self,
        batch: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, float]:
        if len(self.buffer.rewards) < self.n_steps:
            return {}

        data = self.buffer.get()
        static_eval_cache = self._build_update_static_cache(data)
        advantages, returns = self._compute_gae(data["rewards"], data["values"], data["dones"])
        advantages_raw = advantages
        adv_raw_mean = float(advantages_raw.mean().item()) if advantages_raw.numel() > 0 else 0.0
        adv_raw_std = float(advantages_raw.std().item()) if advantages_raw.numel() > 0 else 0.0
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_norm_mean = float(advantages.mean().item()) if advantages.numel() > 0 else 0.0
        adv_norm_std = float(advantages.std().item()) if advantages.numel() > 0 else 0.0

        losses = {
            "policy_loss": 0.0,
            "policy_loss_rawadv": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "update_epochs": 0.0,
            "adv_raw_mean": float(adv_raw_mean),
            "adv_raw_std": float(adv_raw_std),
            "adv_norm_mean": float(adv_norm_mean),
            "adv_norm_std": float(adv_norm_std),
        }
        num_samples = data["rewards"].shape[0]
        update_steps = 0
        stop_early = False

        if progress_callback is not None:
            try:
                progress_callback(0, self.n_epochs)
            except Exception:
                pass

        for epoch_idx in range(self.n_epochs):
            indices_t = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx_t = indices_t[start:end]

                # Encode graphs once per mini-batch (space-for-time optimization).
                state_batch = {k: v[batch_idx_t] for k, v in data.items() if k in self.buffer.states}
                encoded_batch = self._encode_graphs_batch(state_batch)
                action_batch = {
                    "partition": data["actions"]["partition"][batch_idx_t],
                    "routing": data["actions"]["routing"][batch_idx_t],
                    "compression": data["actions"]["compression"][batch_idx_t],
                }

                log_probs = []
                values = []
                entropies = []
                for local_idx in range(int(batch_idx_t.shape[0])):
                    i = int(batch_idx_t[local_idx].item())
                    single_state = {k: v[local_idx] for k, v in state_batch.items()}
                    single_action = {
                        "partition": action_batch["partition"][local_idx],
                        "routing": action_batch["routing"][local_idx],
                        "compression": action_batch["compression"][local_idx],
                    }
                    encoded_single = {
                        "node_emb": encoded_batch["node_emb"][local_idx],
                        "uav_emb": encoded_batch["uav_emb"][local_idx],
                        "graph_emb": encoded_batch["graph_emb"][local_idx],
                        "swarm_emb": encoded_batch["swarm_emb"][local_idx],
                        "ctx": encoded_batch["ctx"][local_idx],
                    }
                    lp, val, ent = self._evaluate_actions(
                        single_state,
                        single_action,
                        static_eval=static_eval_cache[int(i)],
                        encoded=encoded_single,
                    )
                    log_probs.append(lp)
                    values.append(val)
                    entropies.append(ent)

                log_prob = torch.stack(log_probs)
                value = torch.stack(values)
                entropy = torch.stack(entropies)
                value_old = data["values"][batch_idx_t]

                log_ratio = torch.clamp(log_prob - data["log_probs"][batch_idx_t], min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages[batch_idx_t]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[batch_idx_t]
                policy_loss = -torch.min(surr1, surr2).mean()
                surr1_raw = ratio * advantages_raw[batch_idx_t]
                surr2_raw = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_raw[batch_idx_t]
                policy_loss_rawadv = -torch.min(surr1_raw, surr2_raw).mean()
                if self.value_clip_enabled:
                    value_clipped = value_old + torch.clamp(value - value_old, -self.value_clip_eps, self.value_clip_eps)
                    value_loss_unclipped = (returns[batch_idx_t] - value) ** 2
                    value_loss_clipped = (returns[batch_idx_t] - value_clipped) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((returns[batch_idx_t] - value) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses["policy_loss"] += policy_loss.item()
                losses["policy_loss_rawadv"] += policy_loss_rawadv.item()
                losses["value_loss"] += value_loss.item()
                losses["entropy"] += entropy.mean().item()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                losses["approx_kl"] += float(approx_kl.item())
                losses["clip_fraction"] += float(clip_fraction.item())
                losses["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                update_steps += 1

                if self.target_kl > 0 and float(approx_kl.item()) > self.target_kl:
                    stop_early = True
                    break

            if progress_callback is not None:
                try:
                    progress_callback(epoch_idx + 1, self.n_epochs)
                except Exception:
                    pass

            if stop_early:
                break

        losses["update_epochs"] = float(update_steps / max((num_samples + self.batch_size - 1) // self.batch_size, 1))

        if update_steps > 0:
            for k in ("policy_loss", "policy_loss_rawadv", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"):
                losses[k] /= float(update_steps)
        else:
            for k in ("policy_loss", "policy_loss_rawadv", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm", "update_epochs"):
                losses[k] = 0.0

        losses["early_stop_kl"] = 1.0 if stop_early else 0.0
        if update_steps > 0:
            self.update_calls += 1
            self._step_lr_schedule()
        losses["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])

        self.buffer.reset()
        return losses
    
    def save(self, path: str):
        torch.save({"model": self.state_dict()}, path)
    
    def load(self, path: str):
        try:
            data = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # Backward compatibility for older PyTorch versions.
            data = torch.load(path, map_location=self.device)
        self.load_state_dict(data["model"])

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_update_static_cache(self, data: Dict) -> List[Dict]:
        """
        Precompute action/state-dependent static structures for PPO update.

        This is a space-for-time optimization:
        - block graph and predecessor map per sample
        - active compression nodes per sample
        - unnormalized arrival rate / compute / battery arrays per sample
        """
        cache: List[Dict] = []
        num_samples = int(data["rewards"].shape[0])
        arrival_rate_scale = float(getattr(self.env.state_space, "arrival_rate_max", 1.0))
        # Bulk host transfer once per update (space-for-time).
        part_all_np = data["actions"]["partition"].detach().cpu().numpy()
        task_pop_all_np = data["task_popularity"].detach().cpu().numpy()
        routing_all_np = data["actions"]["routing"].detach().cpu().numpy()
        arrival_all_np = data["arrival_rate"].detach().cpu().numpy()

        for i in range(num_samples):
            part_np = part_all_np[i]
            task_pop = task_pop_all_np[i]
            block_graph = self.env.model_graph.build_blocks(part_np, task_pop)

            block_preds = {b: [] for b in range(block_graph.num_blocks)}
            for bu, bv in block_graph.block_edges:
                block_preds[bv].append(bu)

            routing_np = routing_all_np[i][: block_graph.num_blocks]
            active_nodes = self._get_active_group_nodes(block_graph, routing_np)

            arrival_rate = float(arrival_all_np[i]) * max(arrival_rate_scale, 1e-6)
            uav_nodes_i = data["uav_nodes"][i]
            compute_cap_t = (uav_nodes_i[:, 0] * float(self.env.compute_max)).detach()
            battery_t = (uav_nodes_i[:, 1] * float(self.env.battery_max)).detach()

            cache.append(
                {
                    "block_graph": block_graph,
                    "block_preds": block_preds,
                    "active_nodes": active_nodes,
                    "arrival_rate": arrival_rate,
                    "compute_cap_t": compute_cap_t,
                    "battery_t": battery_t,
                }
            )

        return cache

    def _state_to_tensor(self, state: Dict) -> Dict[str, torch.Tensor]:
        state_t = {}
        norm_enabled = bool(getattr(self.env.state_space, "norm_enabled", False))
        norm_keys = {
            "model_nodes",
            "model_adj",
            "uav_nodes",
            "uav_adj",
            "uav_rate",
            "task_popularity",
            "arrival_rate",
            "preference",
            "prev_partition",
            "prev_routing",
            "prev_compression",
        }
        for k, v in state.items():
            x = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            if norm_enabled and k in norm_keys:
                # Keep normalized observations numerically safe and bounded.
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                x = torch.clamp(x, 0.0, 1.0)
            else:
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            state_t[k] = x
        return state_t

    def _action_to_tensor(self, action: Dict) -> Dict[str, torch.Tensor]:
        return {
            "partition": torch.as_tensor(action["partition"], dtype=torch.float32, device=self.device),
            "routing": torch.as_tensor(action["routing"], dtype=torch.long, device=self.device),
            "compression": torch.as_tensor(action["compression"], dtype=torch.long, device=self.device),
        }

    def _encode_graphs(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_emb = self.model_encoder(state["model_nodes"], state["model_adj"])
        uav_emb = self.uav_encoder(state["uav_nodes"], state["uav_rate"])
        graph_emb = node_emb.mean(dim=0)
        swarm_emb = uav_emb.mean(dim=0)
        ctx = torch.cat([state["preference"], state["arrival_rate"], state["task_popularity"]], dim=-1)
        return node_emb, uav_emb, graph_emb, swarm_emb, ctx

    def _encode_graphs_batch(self, state_batch: Dict) -> Dict[str, torch.Tensor]:
        """Encode a mini-batch of states in one forward pass."""
        node_emb = self.model_encoder(state_batch["model_nodes"], state_batch["model_adj"])   # [B, N, H]
        uav_emb = self.uav_encoder(state_batch["uav_nodes"], state_batch["uav_rate"])         # [B, U, H]
        graph_emb = node_emb.mean(dim=1)                                                       # [B, H]
        swarm_emb = uav_emb.mean(dim=1)                                                       # [B, H]
        ctx = torch.cat(
            [state_batch["preference"], state_batch["arrival_rate"], state_batch["task_popularity"]],
            dim=-1,
        )                                                                                     # [B, C]
        return {
            "node_emb": node_emb,
            "uav_emb": uav_emb,
            "graph_emb": graph_emb,
            "swarm_emb": swarm_emb,
            "ctx": ctx,
        }

    def _sample_actions(self, state: Dict, deterministic: bool = False) -> Tuple[Dict, Dict]:
        node_emb, uav_emb, graph_emb, swarm_emb, ctx = self._encode_graphs(state)

        # Partition decisions
        mandatory = self.mandatory_cuts_t
        prev_partition = state["prev_partition"]
        prev_pad = torch.cat([torch.zeros(1, device=self.device), prev_partition], dim=0)
        part_actions = torch.zeros(self.num_nodes - 1, device=self.device)
        part_log_prob_sum = torch.tensor(0.0, device=self.device)
        for i in range(1, self.num_nodes):
            if mandatory[i - 1] > 0:
                part_actions[i - 1] = 1.0
                continue
            logits = self.partition_head(torch.cat([node_emb[i], ctx, prev_pad[i].unsqueeze(0)], dim=-1)).squeeze(-1)
            probs = torch.sigmoid(logits)
            if deterministic:
                act = (probs > 0.5).float()
            else:
                act = torch.bernoulli(probs)
            part_actions[i - 1] = act
            part_log_prob_sum = part_log_prob_sum + (-F.binary_cross_entropy_with_logits(logits, act, reduction="sum"))

        part_np = part_actions.detach().cpu().numpy()
        task_pop = state["task_popularity"].detach().cpu().numpy()
        block_graph = self.env.model_graph.build_blocks(part_np, task_pop)

        # Routing decisions (sequential)
        routing_actions = []
        routing_log_prob_sum = torch.tensor(0.0, device=self.device)
        routing_entropy_sum = torch.tensor(0.0, device=self.device)
        uav_adj = state["uav_adj"]
        arrival_rate_scale = float(getattr(self.env.state_space, "arrival_rate_max", 1.0))
        arrival_rate = float(state["arrival_rate"].item()) * max(arrival_rate_scale, 1e-6)
        compute_cap_t = state["uav_nodes"][:, 0] * float(self.env.compute_max)
        battery_t = state["uav_nodes"][:, 1] * float(self.env.battery_max)
        cap_limit_t = compute_cap_t * float(self.env.delta_t)
        bat_mask_const = battery_t > float(self.env.energy_min)
        load_u_t = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
        block_preds = {b: [] for b in range(block_graph.num_blocks)}
        for bu, bv in block_graph.block_edges:
            block_preds[bv].append(bu)

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
            # Compute capacity mask
            inc = float(block_graph.block_flops[b] * block_graph.block_pi_act[b] * arrival_rate * self.env.delta_t)
            cap_mask = (load_u_t + inc) <= cap_limit_t
            mask = mask & cap_mask & bat_mask_const
            if not mask.any():
                mask = torch.ones_like(mask)
            scores = scores.masked_fill(~mask, -1e9)
            log_probs = torch.log_softmax(scores, dim=-1)
            probs = torch.exp(log_probs)
            if deterministic:
                act = torch.argmax(scores)
            else:
                act = torch.multinomial(probs, num_samples=1).squeeze(-1)
            u_sel = int(act.item())
            assigned_uavs[b] = u_sel
            load_u_t[u_sel] = load_u_t[u_sel] + inc
            routing_actions.append(act)
            routing_log_prob_sum = routing_log_prob_sum + log_probs.gather(0, act.long().view(1)).squeeze(0)
            routing_entropy_sum = routing_entropy_sum + (-(probs * log_probs).sum())

        # Compression decisions
        group_actions = []
        group_log_prob_sum = torch.tensor(0.0, device=self.device)
        group_entropy_sum = torch.tensor(0.0, device=self.device)
        routing_np = np.array([int(x.item()) for x in routing_actions], dtype=np.int32)
        active_nodes = self._get_active_group_nodes(block_graph, routing_np)
        prev_comp = state["prev_compression"]
        for g_idx, (group_name, node_indices) in enumerate(self.env.model_graph.compression_groups.items()):
            nodes = active_nodes.get(group_name, []) or node_indices
            if not nodes:
                group_emb = graph_emb
            else:
                group_emb = node_emb[nodes].mean(dim=0)
            prev_c = prev_comp[g_idx].float()
            logits_g = self.group_head(torch.cat([group_emb, ctx, prev_c.unsqueeze(0)], dim=-1))
            log_probs_g = torch.log_softmax(logits_g, dim=-1)
            probs_g = torch.exp(log_probs_g)
            if deterministic:
                act = torch.argmax(logits_g)
            else:
                act = torch.multinomial(probs_g, num_samples=1).squeeze(-1)
            group_actions.append(act)
            group_log_prob_sum = group_log_prob_sum + log_probs_g.gather(0, act.long().view(1)).squeeze(0)
            group_entropy_sum = group_entropy_sum + (-(probs_g * log_probs_g).sum())

        routing_actions = torch.stack(routing_actions) if routing_actions else torch.zeros(0, device=self.device, dtype=torch.long)
        routing_full = torch.zeros(self.num_nodes, device=self.device, dtype=torch.long)
        if routing_actions.numel() > 0:
            routing_full[: routing_actions.shape[0]] = routing_actions

        actions = {
            "partition": part_actions,
            "routing": routing_full,
            "compression": torch.stack(group_actions) if group_actions else torch.zeros(0, device=self.device, dtype=torch.long),
        }

        extra = {
            "partition_log_prob": part_log_prob_sum,
            "routing_log_prob": routing_log_prob_sum,
            "compression_log_prob": group_log_prob_sum,
            "entropy": routing_entropy_sum + group_entropy_sum,
            "value": self.value_head(torch.cat([graph_emb, swarm_emb, ctx], dim=-1)).squeeze(-1),
        }
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

        # Partition log-prob
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

        # Routing log-prob
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

        # Compression log-prob
        comp_log_prob = torch.tensor(0.0, device=self.device)
        comp_entropy = torch.tensor(0.0, device=self.device)
        if static_eval is not None:
            active_nodes = static_eval["active_nodes"]
        else:
            routing_np = action["routing"].detach().cpu().numpy()[: block_graph.num_blocks]
            active_nodes = self._get_active_group_nodes(block_graph, routing_np)
        prev_comp = state["prev_compression"]
        for g_idx, (group_name, node_indices) in enumerate(self.env.model_graph.compression_groups.items()):
            nodes = active_nodes.get(group_name, []) or node_indices
            if not nodes:
                group_emb = graph_emb
            else:
                group_emb = node_emb[nodes].mean(dim=0)
            prev_c = prev_comp[g_idx].float()
            logits_g = self.group_head(torch.cat([group_emb, ctx, prev_c.unsqueeze(0)], dim=-1))
            log_probs_g = torch.log_softmax(logits_g, dim=-1)
            probs_g = torch.exp(log_probs_g)
            act = action["compression"][g_idx]
            comp_log_prob = comp_log_prob + log_probs_g.gather(0, act.long().view(1)).squeeze(0)
            comp_entropy = comp_entropy + (-(probs_g * log_probs_g).sum())

        log_prob = part_log_prob + routing_log_prob + comp_log_prob
        entropy = routing_entropy + comp_entropy
        value = self.value_head(torch.cat([graph_emb, swarm_emb, ctx], dim=-1)).squeeze(-1)
        return log_prob, value, entropy

    def _compute_gae(self, rewards, values, dones) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def _get_active_group_nodes(self, block_graph, routing: np.ndarray) -> Dict[str, List[int]]:
        """Return active source nodes per compression group under current routing."""
        active: Dict[str, List[int]] = {g: [] for g in self.env.model_graph.compression_groups.keys()}
        for (bu, bv), edge_indices in block_graph.block_edge_to_edges.items():
            if int(routing[bu]) == int(routing[bv]):
                continue
            for e_idx in edge_indices:
                src, _ = self.env.model_graph.edges[e_idx]
                g_name = self.env.node_to_group.get(src, None)
                if g_name is not None:
                    active[g_name].append(src)
        return active

    def _step_lr_schedule(self) -> None:
        """Apply optional learning-rate schedule after each PPO update."""
        if self.lr_schedule != "linear":
            return
        progress = min(float(self.update_calls) / float(max(self.total_updates_est, 1)), 1.0)
        lr = max(self.init_lr * (1.0 - progress), self.init_lr * 0.05)
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(lr)

