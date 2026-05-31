"""
MLP-PPO baseline: replace graph encoders with a flat-state MLP encoder.
"""

from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.core import AgentRegistry, BaseAgent
from .gnn_ppo import GNNPPO, MLP, RolloutBuffer


@AgentRegistry.register("mlp_ppo")
class MLPPPO(GNNPPO):
    """PPO baseline without graph inductive bias."""

    def __init__(self, env: gym.Env, config: Dict):
        nn.Module.__init__(self)
        BaseAgent.__init__(self, env, config)

        algo_cfg = config.get("algorithm", {})
        net_cfg = algo_cfg.get("network", {})
        actor_cfg = net_cfg.get("actor", {})
        critic_cfg = net_cfg.get("critic", {})
        mlp_cfg = net_cfg.get("mlp_encoder", {})
        ppo_cfg = algo_cfg.get("ppo", {})

        self.node_feat_dim = env.observation_space["model_nodes"].shape[-1]
        self.uav_feat_dim = env.observation_space["uav_nodes"].shape[-1]
        self.num_nodes = env.observation_space["model_nodes"].shape[0]
        self.num_uavs = env.observation_space["uav_nodes"].shape[0]
        self.num_groups = env.observation_space["prev_compression"].shape[0]
        self.num_bitwidths = env.action_space["compression"].nvec[0]
        self.task_dim = env.observation_space["task_popularity"].shape[0]
        self.ctx_dim = 2 + 1 + self.task_dim

        self.flat_state_keys = (
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
        )
        self.flat_dim = int(
            sum(int(np.prod(env.observation_space[key].shape)) for key in self.flat_state_keys)
        )

        self.hidden_dim = int(mlp_cfg.get("hidden_dim", net_cfg.get("gnn", {}).get("hidden_dim", 128)))
        encoder_hidden_dims = mlp_cfg.get("hidden_dims", [256, 256])
        encoder_activation = mlp_cfg.get("activation", "relu")

        self.flat_encoder = MLP(self.flat_dim, encoder_hidden_dims, self.hidden_dim, activation=encoder_activation)
        self.node_input_proj = nn.Linear(self.node_feat_dim, self.hidden_dim)
        self.uav_input_proj = nn.Linear(self.uav_feat_dim, self.hidden_dim)
        self.node_global_proj = nn.Linear(self.hidden_dim, self.num_nodes * self.hidden_dim)
        self.uav_global_proj = nn.Linear(self.hidden_dim, self.num_uavs * self.hidden_dim)

        self.partition_head = nn.Linear(self.hidden_dim + self.ctx_dim + 1, 1)
        self.routing_q = nn.Linear(self.hidden_dim + self.ctx_dim + 1, self.hidden_dim)
        self.routing_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.group_head = MLP(
            self.hidden_dim + self.ctx_dim + 1,
            actor_cfg.get("hidden_dims", [128]),
            self.num_bitwidths,
        )
        self.value_head = MLP(
            self.hidden_dim * 2 + self.ctx_dim,
            critic_cfg.get("hidden_dims", [128]),
            1,
        )

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
        self._cached_action_eval = None
        self._cached_state_t = None
        self._cached_state_ref = None
        self._cached_action_ref = None

        self.register_buffer("mandatory_cuts_t", torch.as_tensor(self.env.model_graph.mandatory_cuts, dtype=torch.bool))
        self.register_buffer("uav_index_t", torch.arange(self.num_uavs, dtype=torch.long))

        self.to(self.device)

    def _flatten_state(self, state: Dict) -> torch.Tensor:
        return torch.cat([state[key].reshape(-1) for key in self.flat_state_keys], dim=0)

    def _flatten_state_batch(self, state_batch: Dict) -> torch.Tensor:
        return torch.cat(
            [state_batch[key].reshape(state_batch[key].shape[0], -1) for key in self.flat_state_keys],
            dim=-1,
        )

    def _encode_graphs(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_state = self._flatten_state(state)
        global_emb = self.flat_encoder(flat_state)
        node_global = self.node_global_proj(global_emb).view(self.num_nodes, self.hidden_dim)
        uav_global = self.uav_global_proj(global_emb).view(self.num_uavs, self.hidden_dim)
        node_emb = torch.relu(self.node_input_proj(state["model_nodes"]) + node_global)
        uav_emb = torch.relu(self.uav_input_proj(state["uav_nodes"]) + uav_global)
        graph_emb = node_emb.mean(dim=0)
        swarm_emb = uav_emb.mean(dim=0)
        ctx = torch.cat([state["preference"], state["arrival_rate"], state["task_popularity"]], dim=-1)
        return node_emb, uav_emb, graph_emb, swarm_emb, ctx

    def _encode_graphs_batch(self, state_batch: Dict) -> Dict[str, torch.Tensor]:
        flat_state = self._flatten_state_batch(state_batch)
        global_emb = self.flat_encoder(flat_state)
        node_global = self.node_global_proj(global_emb).view(-1, self.num_nodes, self.hidden_dim)
        uav_global = self.uav_global_proj(global_emb).view(-1, self.num_uavs, self.hidden_dim)
        node_emb = torch.relu(self.node_input_proj(state_batch["model_nodes"]) + node_global)
        uav_emb = torch.relu(self.uav_input_proj(state_batch["uav_nodes"]) + uav_global)
        graph_emb = node_emb.mean(dim=1)
        swarm_emb = uav_emb.mean(dim=1)
        ctx = torch.cat(
            [state_batch["preference"], state_batch["arrival_rate"], state_batch["task_popularity"]],
            dim=-1,
        )
        return {
            "node_emb": node_emb,
            "uav_emb": uav_emb,
            "graph_emb": graph_emb,
            "swarm_emb": swarm_emb,
            "ctx": ctx,
        }
