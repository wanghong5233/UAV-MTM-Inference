"""
Unified trainer.

Provides a generic training loop decoupled from algorithms.
"""

import math
import traceback
from collections import deque
from typing import Dict
from pathlib import Path
import numpy as np
import os
import time
from tqdm import tqdm

from .base_agent import BaseAgent


class Trainer:
    """Generic trainer for DRL agents."""
    
    def __init__(
        self,
        agent: BaseAgent,
        config: Dict,
        logger=None,
        evaluator=None,
    ):
        """Initialize the trainer."""
        self.agent = agent
        self.env = agent.env
        self.config = config
        self.logger = logger
        self.evaluator = evaluator
        
        # Training configuration
        train_config = config.get('training', {})
        self.num_episodes = train_config.get('num_episodes', 10000)
        self.max_steps = train_config.get('max_steps_per_episode', 100)
        self.eval_interval = train_config.get('eval_interval', 100)
        self.save_interval = train_config.get('save_interval', 500)
        log_cfg = config.get("logging", {}) if isinstance(config.get("logging", {}), dict) else {}
        self.log_interval = int(log_cfg.get("log_interval", 10))
        # Console output:
        # - "plain": print stable newline logs (Windows-friendly, good for monitoring).
        # - "tqdm": keep a single progress bar line (minimal console).
        self.console_mode = str(log_cfg.get("console_mode", "tqdm")).lower()
        # Console monitoring: keep a rolling tail of recent lines.
        self.console_trend_enabled = bool(log_cfg.get("console_trend_enabled", True))
        self.console_trend_interval = int(log_cfg.get("console_trend_interval", 1))
        self.console_trend_max_lines = int(log_cfg.get("console_trend_max_lines", 30))
        self.console_trend_history = deque(maxlen=max(self.console_trend_max_lines, 1))

        # Reward signal scaling for optimization (raw reward is still logged).
        reward_cfg = config.get("reward", {})
        scale_cfg = reward_cfg.get("signal_scaling", {}) if isinstance(reward_cfg, dict) else {}
        self.reward_signal_scaling_enabled = bool(scale_cfg.get("enabled", False))
        self.reward_signal_scaling_mode = str(scale_cfg.get("mode", "scale_only")).lower()
        self.reward_signal_target_abs = float(scale_cfg.get("target_abs_max", 50.0))
        self.reward_signal_warmup_enabled = bool(scale_cfg.get("warmup_enabled", False))
        self.reward_signal_warmup_episodes = int(scale_cfg.get("warmup_episodes", 200))
        self.reward_signal_warmup_target_abs = float(scale_cfg.get("warmup_target_abs_max", 10.0))
        self.reward_signal_decay = float(scale_cfg.get("running_abs_decay", 0.995))
        self.reward_signal_init_abs = float(scale_cfg.get("init_abs_max", 5.0))
        self.reward_signal_running_abs = max(self.reward_signal_init_abs, 1e-6)
        self.reward_signal_window_size = int(scale_cfg.get("window_size", 4096))
        self.reward_signal_window_min_samples = int(scale_cfg.get("window_min_samples", 128))
        self.reward_signal_window_eps = float(scale_cfg.get("window_eps", 1e-6))
        self.reward_signal_window_z_clip = float(scale_cfg.get("window_z_clip", 3.0))
        self.reward_signal_window = deque(maxlen=max(self.reward_signal_window_size, 32))
        self.reward_signal_window_sum = 0.0
        self.reward_signal_window_sq_sum = 0.0
        self.reward_signal_window_mean = 0.0
        self.reward_signal_window_std = 1.0
        
        # Checkpoint path
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.best_eval_reward = -float('inf')
    
    def train(self) -> Dict:
        """Main training loop."""
        _, target_steps = self._get_rollout_progress()
        n_steps = target_steps or 0
        eval_enabled = bool(self.eval_interval) and int(self.eval_interval) > 0
        save_enabled = bool(self.save_interval) and int(self.save_interval) > 0
        # Compact startup banner
        print("=" * 70)
        print(
            f"  Training  agent={self.agent.name}  device={self.agent.device}"
            f"  episodes={self.num_episodes}  max_steps/ep={self.max_steps}"
        )
        print(
            f"  PPO n_steps={n_steps}  (1 PPO update every ~{max(n_steps//max(self.max_steps,1),1)} episodes)"
        )
        print(
            f"  eval_interval={self.eval_interval} ({'on' if eval_enabled else 'off'})"
            f"  save_interval={self.save_interval} ({'on' if save_enabled else 'off'})"
            f"  log_interval={self.log_interval}  console_mode={self.console_mode}"
        )
        if self.reward_signal_scaling_enabled:
            scale_desc = (
                f"window_std  window={self.reward_signal_window_size}"
                f"  z_clip={self.reward_signal_window_z_clip}"
                if self.reward_signal_scaling_mode == "window_standardize"
                else f"running_abs  decay={self.reward_signal_decay}"
            )
            warmup_desc = (
                f"  warmup: ep<{self.reward_signal_warmup_episodes} -> +/-{self.reward_signal_warmup_target_abs}"
                if self.reward_signal_warmup_enabled else ""
            )
            print(
                f"  reward_scale: {scale_desc}"
                f"  target=+/-{self.reward_signal_target_abs}{warmup_desc}"
            )
        print("=" * 70)
        print(
            "  Console output:\n"
            "  - tqdm bar: collect uses transitions x/n_steps; update uses epoch x/n_epochs\n"
            "  - [STATUS] : a single status line (phase + buffer) for monitoring\n"
            "  - [ITER #N]: printed when a PPO iteration completes (collect+update)\n"
            "  - [EVAL]/[CKPT]: only when enabled"
        )
        print("=" * 70)

        reward_history = []
        length_history = []
        delay_history = []
        energy_history = []
        acc_history = []
        comm_volume_history = []
        rho_max_history = []
        last_update_metrics = {}
        ppo_update_count = 0
        # Rollout (iteration) aggregation: summarize the entire collect phase until update fires.
        roll_eps = 0
        roll_steps = 0
        roll_reward = 0.0
        roll_reward_signal = 0.0
        roll_delay_sum = 0.0
        roll_energy_sum = 0.0
        roll_energy_epoch_obj_sum = 0.0
        roll_energy_epoch_total_sum = 0.0
        roll_energy_effective_epoch_sum = 0.0
        roll_energy_hover_epoch_sum = 0.0
        roll_energy_compute_epoch_sum = 0.0
        roll_energy_comm_epoch_sum = 0.0
        roll_energy_total_per_request_sum = 0.0
        roll_energy_obj_per_request_sum = 0.0
        roll_energy_hover_per_request_sum = 0.0
        roll_energy_compute_per_request_sum = 0.0
        roll_energy_comm_per_request_sum = 0.0
        roll_requests_per_epoch_sum = 0.0
        roll_acc_sum = 0.0
        roll_clip_sum = 0.0
        roll_infeasible_sum = 0.0
        roll_battery_vio_sum = 0.0
        roll_penalty_q_sum = 0.0
        roll_blocks_sum = 0.0
        roll_comm_volume_sum = 0.0
        roll_max_rho = 0.0
        roll_delay_reward_sum = 0.0
        roll_energy_reward_sum = 0.0
        roll_collect_wall_s = 0.0

        self.agent.train_mode()

        use_tqdm = (self.console_mode != "plain")
        pbar = tqdm(
            total=max(int(n_steps), 1),
            desc=f"[{self.agent.device}] collect",
            ascii=True,
            dynamic_ncols=True,
            # Leave printed lines above intact; tqdm only occupies the bottom line.
            leave=True,
            disable=(not use_tqdm),
        )

        def write_line(msg: str) -> None:
            if use_tqdm:
                pbar.write(msg)
            else:
                print(msg, flush=True)

        def redraw_iteration_window(status: str, lines: deque) -> None:
            """
            Redraw a fixed-size rolling window for monitoring.
            In plain console mode, this keeps the visible console stable (no infinite scroll).
            """
            if use_tqdm:
                # tqdm mode: do not clear/redraw the terminal.
                return
            # Windows-friendly clear.
            try:
                os.system("cls" if os.name == "nt" else "clear")
            except Exception:
                pass
            print(status, flush=True)
            for ln in lines:
                print(ln, flush=True)

        # Rolling window (iteration unit, NOT episode unit).
        iter_window: deque = deque(maxlen=max(int(self.console_trend_max_lines), 1))
        status_line = (
            f"[STATUS] phase=collect steps={self.total_steps:08d} "
            f"buf=0/{n_steps} iter#=0000"
        )
        if self.console_mode == "plain" and self.console_trend_enabled:
            redraw_iteration_window(status_line, iter_window)

        for episode in range(self.num_episodes):
            self.episode_count = episode

            # 1) Collect one episode (env.reset -> up to max_steps env.step calls)
            t_collect_start = time.perf_counter()
            try:
                episode_info = self._collect_episode()
            except Exception as exc:
                self._log_event(
                    "collect_episode_exception",
                    {
                        "episode": episode,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                    step=self.total_steps,
                    level="ERROR",
                )
                raise
            roll_collect_wall_s += float(time.perf_counter() - t_collect_start)

            # --- per-episode metrics (available right after collect) ---
            ep_reward = float(episode_info.get("reward", 0.0))
            ep_reward_signal = float(episode_info.get("reward_signal", 0.0))
            ep_len = float(episode_info.get("length", 0.0))
            ep_delay = float(episode_info.get("mean_delay", 0.0))
            ep_energy = float(episode_info.get("mean_energy", 0.0))
            ep_energy_epoch_obj = float(episode_info.get("mean_energy_epoch_obj", 0.0))
            ep_energy_epoch_total = float(episode_info.get("mean_energy_epoch_total", 0.0))
            ep_energy_effective_epoch = float(episode_info.get("mean_energy_effective_epoch", 0.0))
            ep_energy_hover_epoch = float(episode_info.get("mean_energy_hover_epoch", 0.0))
            ep_energy_compute_epoch = float(episode_info.get("mean_energy_compute_epoch", 0.0))
            ep_energy_comm_epoch = float(episode_info.get("mean_energy_comm_epoch", 0.0))
            ep_energy_total_per_request = float(episode_info.get("mean_energy_total_per_request", 0.0))
            ep_energy_obj_per_request = float(episode_info.get("mean_energy_obj_per_request", 0.0))
            ep_energy_hover_per_request = float(episode_info.get("mean_energy_hover_per_request", 0.0))
            ep_energy_compute_per_request = float(episode_info.get("mean_energy_compute_per_request", 0.0))
            ep_energy_comm_per_request = float(episode_info.get("mean_energy_comm_per_request", 0.0))
            ep_requests_per_epoch = float(episode_info.get("mean_requests_per_epoch", 0.0))
            ep_acc = float(episode_info.get("mean_accuracy", 0.0))
            ep_clip = float(episode_info.get("reward_signal_clip_ratio", 0.0))
            ep_infeasible = float(episode_info.get("mean_infeasible_links", 0.0))
            ep_battery_vio = float(episode_info.get("mean_battery_violation", 0.0))
            ep_penalty_q = float(episode_info.get("mean_penalty_q", 0.0))
            ep_blocks = float(episode_info.get("mean_num_blocks", 0.0))
            ep_comm_volume = float(episode_info.get("mean_comm_volume", 0.0))
            ep_rho_max = float(episode_info.get("max_rho", 0.0))
            ep_delay_reward = float(episode_info.get("mean_delay_reward", 0.0))
            ep_energy_reward = float(episode_info.get("mean_energy_reward", 0.0))

            reward_history.append(ep_reward)
            length_history.append(ep_len)
            delay_history.append(ep_delay)
            energy_history.append(ep_energy)
            acc_history.append(ep_acc)
            comm_volume_history.append(ep_comm_volume)
            rho_max_history.append(ep_rho_max)

            # Rollout aggregation (step-weighted)
            roll_eps += 1
            roll_steps += int(ep_len)
            roll_reward += ep_reward
            roll_reward_signal += ep_reward_signal
            roll_delay_sum += ep_delay * ep_len
            roll_energy_sum += ep_energy * ep_len
            roll_energy_epoch_obj_sum += ep_energy_epoch_obj * ep_len
            roll_energy_epoch_total_sum += ep_energy_epoch_total * ep_len
            roll_energy_effective_epoch_sum += ep_energy_effective_epoch * ep_len
            roll_energy_hover_epoch_sum += ep_energy_hover_epoch * ep_len
            roll_energy_compute_epoch_sum += ep_energy_compute_epoch * ep_len
            roll_energy_comm_epoch_sum += ep_energy_comm_epoch * ep_len
            roll_energy_total_per_request_sum += ep_energy_total_per_request * ep_len
            roll_energy_obj_per_request_sum += ep_energy_obj_per_request * ep_len
            roll_energy_hover_per_request_sum += ep_energy_hover_per_request * ep_len
            roll_energy_compute_per_request_sum += ep_energy_compute_per_request * ep_len
            roll_energy_comm_per_request_sum += ep_energy_comm_per_request * ep_len
            roll_requests_per_epoch_sum += ep_requests_per_epoch * ep_len
            roll_acc_sum += ep_acc * ep_len
            roll_clip_sum += ep_clip * ep_len
            roll_infeasible_sum += ep_infeasible * ep_len
            roll_battery_vio_sum += ep_battery_vio * ep_len
            roll_penalty_q_sum += ep_penalty_q * ep_len
            roll_blocks_sum += ep_blocks * ep_len
            roll_comm_volume_sum += ep_comm_volume * ep_len
            roll_max_rho = max(roll_max_rho, ep_rho_max)
            roll_delay_reward_sum += ep_delay_reward * ep_len
            roll_energy_reward_sum += ep_energy_reward * ep_len

            # 2) PPO update (fires only when buffer >= n_steps)
            buf_before, buf_target = self._get_rollout_progress()
            ready_for_update = (
                buf_before is not None
                and buf_target is not None
                and int(buf_before) >= int(buf_target)
            )
            # Running collect-window averages (step-weighted, denominator = actual collected steps).
            run_denom = max(int(roll_steps), 1)
            run_r_step = roll_reward / float(run_denom)
            run_d = roll_delay_sum / float(run_denom)
            run_e = roll_energy_sum / float(run_denom)
            run_a = roll_acc_sum / float(run_denom)
            # Show phase transition clearly for monitoring.
            if use_tqdm:
                if ready_for_update:
                    # Switch progress unit: PPO epoch progress x/10.
                    total_epochs = int(getattr(self.agent, "n_epochs", 10))
                    if pbar.total != max(total_epochs, 1):
                        pbar.reset(total=max(total_epochs, 1))
                    pbar.n = 0
                    pbar.set_description(f"[{self.agent.device}] update")
                    pbar.set_postfix(
                        {
                            "phase": "update",
                            "epoch": f"0/{max(total_epochs, 1)}",
                            "iter#": ppo_update_count + 1,
                            "steps": self.total_steps,
                            "Rbar": f"{run_r_step:+6.2f}",
                            "Dbar": f"{run_d:.2f}",
                            "Ereq": f"{run_e:.1f}",
                            "Abar": f"{run_a:.3f}",
                        },
                        refresh=False,
                    )
                    pbar.refresh()
                else:
                    # Collect progress unit: transitions x/n_steps.
                    collect_target = int(buf_target) if buf_target is not None else int(n_steps)
                    collect_target = max(collect_target, 1)
                    if pbar.total != collect_target:
                        pbar.reset(total=collect_target)
                    collect_curr = int(buf_before) if buf_before is not None else 0
                    pbar.n = max(0, min(collect_curr, collect_target))
                    pbar.set_description(f"[{self.agent.device}] collect")
                    pbar.set_postfix(
                        {
                            "phase": "collect",
                            "buf": self._format_buffer_progress(buf_before, buf_target),
                            "iter#": ppo_update_count,
                            "steps": self.total_steps,
                            "Rbar": f"{run_r_step:+6.2f}",
                            "Dbar": f"{run_d:.2f}",
                            "Ereq": f"{run_e:.1f}",
                            "Abar": f"{run_a:.3f}",
                        },
                        refresh=False,
                    )
                    pbar.refresh()
            # Plain mode does not print extra lines here. The phase transition is shown
            # in the top [STATUS] line when we redraw the iteration rolling window.

            t_update_start = time.perf_counter()
            try:
                if ready_for_update and use_tqdm:
                    def _on_update_progress(epoch_idx: int, total_epochs: int):
                        total = max(int(total_epochs), 1)
                        curr = max(0, min(int(epoch_idx), total))
                        if pbar.total != total:
                            pbar.reset(total=total)
                        pbar.n = curr
                        pbar.set_description(f"[{self.agent.device}] update")
                        pbar.set_postfix(
                            {
                                "phase": "update",
                                "epoch": f"{curr}/{total}",
                                "iter#": ppo_update_count + 1,
                                "steps": self.total_steps,
                                "Rbar": f"{run_r_step:+6.2f}",
                                "Dbar": f"{run_d:.2f}",
                                "Ereq": f"{run_e:.1f}",
                                "Abar": f"{run_a:.3f}",
                            },
                            refresh=False,
                        )
                        pbar.refresh()

                    try:
                        update_metrics = self.agent.update(progress_callback=_on_update_progress)
                    except TypeError:
                        update_metrics = self.agent.update()
                else:
                    update_metrics = self.agent.update()
            except Exception as exc:
                self._log_event(
                    "agent_update_exception",
                    {
                        "episode": episode,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                    step=self.total_steps,
                    level="ERROR",
                )
                raise
            update_wall_s = float(time.perf_counter() - t_update_start)

            did_update = bool(update_metrics)
            if did_update:
                last_update_metrics = dict(update_metrics)
                ppo_update_count += 1
            self._warn_non_finite_dict("update_metrics", update_metrics or {}, episode)

            # --- Console status (NOT logged as history; history is iteration-only) ---
            if self.console_mode == "plain" and self.console_trend_enabled:
                status_line = (
                    f"[STATUS] phase={'update' if ready_for_update else 'collect':<6} "
                    f"steps={self.total_steps:08d} "
                    f"buf={self._format_buffer_progress(buf_before, buf_target):>9} "
                    f"iter#={ppo_update_count:04d}"
                )
                # In plain mode, always refresh status so users can confirm training is alive
                # before the first iteration finishes.
                if not did_update:
                    redraw_iteration_window(status_line, iter_window)

            # --- [ITER #N] one line per iteration (collect + update) ---
            if did_update:
                denom = max(int(roll_steps), 1)
                collect_r_step = roll_reward / float(denom)
                collect_rs_step = roll_reward_signal / float(denom)
                collect_d = roll_delay_sum / float(denom)
                collect_e = roll_energy_sum / float(denom)
                collect_energy_epoch_obj = roll_energy_epoch_obj_sum / float(denom)
                collect_energy_epoch_total = roll_energy_epoch_total_sum / float(denom)
                collect_energy_effective_epoch = roll_energy_effective_epoch_sum / float(denom)
                collect_energy_hover_epoch = roll_energy_hover_epoch_sum / float(denom)
                collect_energy_compute_epoch = roll_energy_compute_epoch_sum / float(denom)
                collect_energy_comm_epoch = roll_energy_comm_epoch_sum / float(denom)
                collect_energy_total_per_request = roll_energy_total_per_request_sum / float(denom)
                collect_energy_obj_per_request = roll_energy_obj_per_request_sum / float(denom)
                collect_energy_hover_per_request = roll_energy_hover_per_request_sum / float(denom)
                collect_energy_compute_per_request = roll_energy_compute_per_request_sum / float(denom)
                collect_energy_comm_per_request = roll_energy_comm_per_request_sum / float(denom)
                collect_requests_per_epoch = roll_requests_per_epoch_sum / float(denom)
                collect_a = roll_acc_sum / float(denom)
                collect_clip = roll_clip_sum / float(denom)
                collect_infeasible = roll_infeasible_sum / float(denom)
                collect_battery_vio = roll_battery_vio_sum / float(denom)
                collect_penalty_q = roll_penalty_q_sum / float(denom)
                collect_blocks = roll_blocks_sum / float(denom)
                collect_comm_volume = roll_comm_volume_sum / float(denom)
                collect_delay_reward = roll_delay_reward_sum / float(denom)
                collect_energy_reward = roll_energy_reward_sum / float(denom)
                m = last_update_metrics
                iter_metrics = {
                    "iteration": float(ppo_update_count),
                    "collect_episodes": float(roll_eps),
                    "collect_steps": float(denom),
                    "collect_wall_s": float(roll_collect_wall_s),
                    "update_wall_s": float(update_wall_s),
                    "mean_reward_raw_step": float(collect_r_step),
                    "mean_reward_signal_step": float(collect_rs_step),
                    "mean_reward_signal_clip_ratio": float(collect_clip),
                    "mean_delay": float(collect_d),
                    "mean_energy": float(collect_e),
                    "mean_energy_epoch_obj": float(collect_energy_epoch_obj),
                    "mean_energy_epoch_total": float(collect_energy_epoch_total),
                    "mean_energy_effective_epoch": float(collect_energy_effective_epoch),
                    "mean_energy_hover_epoch": float(collect_energy_hover_epoch),
                    "mean_energy_compute_epoch": float(collect_energy_compute_epoch),
                    "mean_energy_comm_epoch": float(collect_energy_comm_epoch),
                    "mean_energy_total_per_request": float(collect_energy_total_per_request),
                    "mean_energy_obj_per_request": float(collect_energy_obj_per_request),
                    "mean_energy_hover_per_request": float(collect_energy_hover_per_request),
                    "mean_energy_compute_per_request": float(collect_energy_compute_per_request),
                    "mean_energy_comm_per_request": float(collect_energy_comm_per_request),
                    "mean_requests_per_epoch": float(collect_requests_per_epoch),
                    "mean_accuracy": float(collect_a),
                    "mean_infeasible_links": float(collect_infeasible),
                    "mean_battery_violation": float(collect_battery_vio),
                    "mean_penalty_q": float(collect_penalty_q),
                    "mean_num_blocks": float(collect_blocks),
                    "mean_comm_volume": float(collect_comm_volume),
                    "max_rho": float(roll_max_rho),
                    "mean_delay_reward": float(collect_delay_reward),
                    "mean_energy_reward": float(collect_energy_reward),
                }
                iter_line = (
                    f"[ITER #{ppo_update_count:04d}] steps={self.total_steps:08d} "
                    f"| collect: eps={roll_eps:02d} steps={roll_steps:04d} "
                    f"R/step={collect_r_step:+7.2f} D={collect_d:.2f} Ereq={collect_e:.1f} "
                    f"A={collect_a:.3f} clip={collect_clip:.2f} "
                    f"tC={roll_collect_wall_s:.1f}s "
                    f"| update: pi={self._format_metric_str(m,'policy_loss','.3f')} "
                    f"pi_raw={self._format_metric_str(m,'policy_loss_rawadv','.3f')} "
                    f"v={self._format_metric_str(m,'value_loss','.1f')} "
                    f"ent={self._format_metric_str(m,'entropy','.1f')} "
                    f"kl={self._format_metric_str(m,'approx_kl','.4f')} "
                    f"cf={self._format_metric_str(m,'clip_fraction','.2f')} "
                    f"lr={self._format_metric_str(m,'learning_rate','.2e')} "
                    f"tU={update_wall_s:.1f}s"
                )
                if self.console_mode == "plain" and self.console_trend_enabled:
                    iter_window.append(iter_line)
                    redraw_iteration_window(status_line, iter_window)
                else:
                    write_line(iter_line)
                if self.logger:
                    self._log_iteration_metrics(ppo_update_count, iter_metrics, m)

                # reset rollout accumulators
                roll_eps = 0
                roll_steps = 0
                roll_reward = 0.0
                roll_reward_signal = 0.0
                roll_delay_sum = 0.0
                roll_energy_sum = 0.0
                roll_energy_epoch_obj_sum = 0.0
                roll_energy_epoch_total_sum = 0.0
                roll_energy_effective_epoch_sum = 0.0
                roll_energy_hover_epoch_sum = 0.0
                roll_energy_compute_epoch_sum = 0.0
                roll_energy_comm_epoch_sum = 0.0
                roll_energy_total_per_request_sum = 0.0
                roll_energy_obj_per_request_sum = 0.0
                roll_energy_hover_per_request_sum = 0.0
                roll_energy_compute_per_request_sum = 0.0
                roll_energy_comm_per_request_sum = 0.0
                roll_requests_per_epoch_sum = 0.0
                roll_acc_sum = 0.0
                roll_clip_sum = 0.0
                roll_infeasible_sum = 0.0
                roll_battery_vio_sum = 0.0
                roll_penalty_q_sum = 0.0
                roll_blocks_sum = 0.0
                roll_comm_volume_sum = 0.0
                roll_max_rho = 0.0
                roll_delay_reward_sum = 0.0
                roll_energy_reward_sum = 0.0
                roll_collect_wall_s = 0.0

            # tqdm progress is managed by phase-specific logic above:
            # collect => transitions x/n_steps, update => epoch x/n_epochs.

            # 3) Periodic evaluation
            if eval_enabled and episode % int(self.eval_interval) == 0 and episode > 0:
                write_line(
                    f"[EVAL] steps={self.total_steps:08d} iter#={ppo_update_count} ..."
                )
                eval_results = self._evaluate()
                self._log_eval_metrics(ppo_update_count, eval_results)
                write_line(
                    f"[EVAL] steps={self.total_steps:08d}"
                    f" reward={self._safe_float(eval_results.get('mean_reward')):.3f}"
                    f" delay={self._safe_float(eval_results.get('mean_delay')):.3f}"
                    f" ereq={self._safe_float(eval_results.get('mean_energy')):.1f}"
                    f" acc={self._safe_float(eval_results.get('mean_accuracy')):.3f}"
                )
                if eval_results.get('mean_reward', -float('inf')) > self.best_eval_reward:
                    self.best_eval_reward = eval_results['mean_reward']
                    self._save_checkpoint(episode, is_best=True)

            # 4) Periodic checkpoint
            if save_enabled and episode % int(self.save_interval) == 0 and episode > 0:
                self._save_checkpoint(episode, is_best=False)
                write_line(f"[CKPT] ep={episode:06d}  saved")

        pbar.close()
        best_eval_reward = (
            float(self.best_eval_reward) if math.isfinite(float(self.best_eval_reward)) else None
        )
        safe_last_update = {
            k: (float(v) if math.isfinite(self._safe_float(v, default=float("nan"))) else None)
            for k, v in last_update_metrics.items()
        }
        summary = {
            "num_episodes": int(self.num_episodes),
            "total_steps": int(self.total_steps),
            "best_eval_reward": best_eval_reward,
            "final_episode_reward": float(reward_history[-1]) if reward_history else 0.0,
            "mean_reward": float(np.mean(reward_history)) if reward_history else 0.0,
            "mean_last_100_reward": float(np.mean(reward_history[-100:])) if reward_history else 0.0,
            "reward_signal_running_abs": float(self.reward_signal_running_abs),
            "reward_signal_mode": self.reward_signal_scaling_mode,
            "reward_signal_window_mean": float(self.reward_signal_window_mean),
            "reward_signal_window_std": float(self.reward_signal_window_std),
            "mean_episode_length": float(np.mean(length_history)) if length_history else 0.0,
            "mean_delay": float(np.mean(delay_history)) if delay_history else 0.0,
            "mean_energy": float(np.mean(energy_history)) if energy_history else 0.0,
            "mean_accuracy": float(np.mean(acc_history)) if acc_history else 0.0,
            "mean_comm_volume": float(np.mean(comm_volume_history)) if comm_volume_history else 0.0,
            "mean_peak_rho": float(np.mean(rho_max_history)) if rho_max_history else 0.0,
            "mean_last_100_comm_volume": float(np.mean(comm_volume_history[-100:])) if comm_volume_history else 0.0,
            "mean_last_100_peak_rho": float(np.mean(rho_max_history[-100:])) if rho_max_history else 0.0,
            "mean_energy_semantics": "expected_compute_plus_communication_energy_per_request",
            "last_update_metrics": safe_last_update,
        }
        self._log_event("training_summary", summary, step=self.total_steps, level="INFO")
        print("Training completed.")
        return summary
    
    def _collect_episode(self) -> Dict:
        """Collect one episode of data."""
        state, _ = self.env.reset()
        done = False
        truncated = False
        step = 0
        episode_reward_raw = 0.0
        episode_reward_signal = 0.0
        episode_length = 0
        sum_reward_raw = 0.0
        sum_reward_raw_sq = 0.0
        sum_reward_signal = 0.0
        sum_reward_signal_sq = 0.0
        signal_clip_count = 0.0

        sum_delay = 0.0
        sum_energy = 0.0
        sum_accuracy = 0.0
        sum_energy_epoch_obj = 0.0
        sum_energy_epoch_total = 0.0
        sum_energy_effective_epoch = 0.0
        sum_energy_hover_epoch = 0.0
        sum_energy_compute_epoch = 0.0
        sum_energy_comm_epoch = 0.0
        sum_energy_total_per_request = 0.0
        sum_energy_obj_per_request = 0.0
        sum_energy_effective_per_request = 0.0
        sum_energy_hover_per_request = 0.0
        sum_energy_compute_per_request = 0.0
        sum_energy_comm_per_request = 0.0
        sum_requests_per_epoch = 0.0
        sum_infeasible_links = 0.0
        sum_battery_violation = 0.0
        sum_penalty_q = 0.0
        sum_num_blocks = 0.0
        sum_comm_volume = 0.0
        max_rho = 0.0
        sum_delay_reward = 0.0
        sum_energy_reward = 0.0
        sum_accuracy_reward = 0.0

        while not (done or truncated) and step < self.max_steps:
            # Select action
            action = self.agent.select_action(state, deterministic=False)

            # Step environment
            next_state, reward_raw, done, truncated, info = self.env.step(action)

            if not math.isfinite(float(reward_raw)):
                self._log_event(
                    "non_finite_reward",
                    {
                        "episode": self.episode_count,
                        "episode_step": step,
                        "raw_reward": reward_raw,
                    },
                    step=self.total_steps,
                    level="ERROR",
                )
                reward_raw = -1e6

            reward_signal, is_signal_clipped = self._scale_reward_for_update(float(reward_raw))

            # Store transition if needed
            self.agent.store_transition(state, action, reward_signal, next_state, done or truncated, info)

            # Update statistics
            episode_reward_raw += reward_raw
            episode_reward_signal += reward_signal
            episode_length += 1
            self.total_steps += 1
            sum_reward_raw += float(reward_raw)
            sum_reward_raw_sq += float(reward_raw) * float(reward_raw)
            sum_reward_signal += float(reward_signal)
            sum_reward_signal_sq += float(reward_signal) * float(reward_signal)
            signal_clip_count += float(1.0 if is_signal_clipped else 0.0)

            sum_delay += self._safe_float(info.get("delay", 0.0))
            sum_energy += self._safe_float(
                info.get("energy_record", info.get("energy_action_per_request", info.get("energy", 0.0)))
            )
            sum_accuracy += self._safe_float(info.get("accuracy", 0.0))
            sum_energy_epoch_obj += self._safe_float(info.get("energy_epoch_obj", info.get("energy", 0.0)))
            sum_energy_epoch_total += self._safe_float(info.get("energy_epoch_total", info.get("energy", 0.0)))
            sum_energy_effective_epoch += self._safe_float(info.get("energy_effective_epoch", 0.0))
            sum_energy_hover_epoch += self._safe_float(info.get("energy_hover", 0.0))
            sum_energy_compute_epoch += self._safe_float(info.get("energy_compute", 0.0))
            sum_energy_comm_epoch += self._safe_float(info.get("energy_comm", 0.0))
            sum_energy_total_per_request += self._safe_float(info.get("energy_total_per_request", 0.0))
            sum_energy_obj_per_request += self._safe_float(info.get("energy_obj_per_request", 0.0))
            sum_energy_effective_per_request += self._safe_float(
                info.get("energy_effective_per_request", 0.0)
            )
            sum_energy_hover_per_request += self._safe_float(info.get("energy_hover_per_request", 0.0))
            sum_energy_compute_per_request += self._safe_float(info.get("energy_compute_per_request", 0.0))
            sum_energy_comm_per_request += self._safe_float(info.get("energy_comm_per_request", 0.0))
            sum_requests_per_epoch += self._safe_float(info.get("requests_per_epoch", 0.0))
            sum_infeasible_links += self._safe_float(info.get("infeasible_links", 0.0))
            sum_battery_violation += self._safe_float(info.get("battery_violation", 0.0))
            sum_penalty_q += self._safe_float(info.get("penalty_q", 0.0))
            sum_num_blocks += self._safe_float(info.get("num_blocks", 0.0))
            sum_comm_volume += self._safe_float(info.get("comm_volume_total", 0.0))
            sum_delay_reward += self._safe_float(info.get("delay_reward", 0.0))
            sum_energy_reward += self._safe_float(info.get("energy_reward", 0.0))
            sum_accuracy_reward += self._safe_float(info.get("accuracy_reward", 0.0))
            max_rho = max(max_rho, self._safe_float(info.get("rho_max", 0.0)))

            state = next_state
            step += 1

        denom = max(episode_length, 1)
        mean_reward_raw_step = float(sum_reward_raw / denom)
        var_reward_raw_step = float(max(sum_reward_raw_sq / denom - mean_reward_raw_step * mean_reward_raw_step, 0.0))
        std_reward_raw_step = float(np.sqrt(var_reward_raw_step))
        mean_reward_signal = float(sum_reward_signal / denom)
        var_reward_signal = float(max(sum_reward_signal_sq / denom - mean_reward_signal * mean_reward_signal, 0.0))
        std_reward_signal = float(np.sqrt(var_reward_signal))
        return {
            "reward": float(episode_reward_raw),
            "reward_signal": float(episode_reward_signal),
            "mean_reward_raw_step": mean_reward_raw_step,
            "std_reward_raw_step": std_reward_raw_step,
            "mean_reward_signal": mean_reward_signal,
            "std_reward_signal_step": std_reward_signal,
            "reward_signal_clip_ratio": float(signal_clip_count / denom),
            "length": int(episode_length),
            "total_steps": int(self.total_steps),
            "mean_delay": float(sum_delay / denom),
            "mean_energy": float(sum_energy / denom),
            "mean_energy_epoch_obj": float(sum_energy_epoch_obj / denom),
            "mean_energy_epoch_total": float(sum_energy_epoch_total / denom),
            "mean_energy_effective_epoch": float(sum_energy_effective_epoch / denom),
            "mean_energy_hover_epoch": float(sum_energy_hover_epoch / denom),
            "mean_energy_compute_epoch": float(sum_energy_compute_epoch / denom),
            "mean_energy_comm_epoch": float(sum_energy_comm_epoch / denom),
            "mean_energy_total_per_request": float(sum_energy_total_per_request / denom),
            "mean_energy_obj_per_request": float(sum_energy_obj_per_request / denom),
            "mean_energy_effective_per_request": float(sum_energy_effective_per_request / denom),
            "mean_energy_hover_per_request": float(sum_energy_hover_per_request / denom),
            "mean_energy_compute_per_request": float(sum_energy_compute_per_request / denom),
            "mean_energy_comm_per_request": float(sum_energy_comm_per_request / denom),
            "mean_requests_per_epoch": float(sum_requests_per_epoch / denom),
            "mean_accuracy": float(sum_accuracy / denom),
            "mean_infeasible_links": float(sum_infeasible_links / denom),
            "mean_battery_violation": float(sum_battery_violation / denom),
            "mean_penalty_q": float(sum_penalty_q / denom),
            "mean_num_blocks": float(sum_num_blocks / denom),
            "mean_comm_volume": float(sum_comm_volume / denom),
            "max_rho": float(max_rho),
            "mean_delay_reward": float(sum_delay_reward / denom),
            "mean_energy_reward": float(sum_energy_reward / denom),
            "mean_accuracy_reward": float(sum_accuracy_reward / denom),
        }
    
    def _evaluate(self) -> Dict:
        """Evaluate the current policy."""
        if self.evaluator is None:
            return {}
        
        eval_results = self.evaluator.evaluate(self.agent)
        return eval_results

    def _log_iteration_metrics(self, iteration: int, iter_info: Dict, update_metrics: Dict) -> None:
        """Log compact iteration-level metrics (x-axis is PPO iteration)."""
        if self.logger is None:
            return

        # --- Paper-facing metrics (shown in convergence plots) ---
        metrics = {
            "metrics/reward":   iter_info.get("mean_reward_raw_step", 0.0),
            "metrics/delay":    iter_info.get("mean_delay", 0.0),
            "metrics/energy":   iter_info.get("mean_energy", 0.0),   # Ereq: per-request compute+comm energy
            "metrics/accuracy": iter_info.get("mean_accuracy", 0.0),
            "metrics/comm_volume": iter_info.get("mean_comm_volume", 0.0),
            "metrics/peak_rho": iter_info.get("max_rho", 0.0),
            "training/policy_loss_raw": (update_metrics or {}).get("policy_loss_rawadv", 0.0),
            "training/value_loss":      (update_metrics or {}).get("value_loss", 0.0),
        }

        # --- Debug metrics (energy decomposition + reward decomposition) ---
        debug = {
            "metrics_debug/energy_compute": iter_info.get("mean_energy_compute_per_request", 0.0),
            "metrics_debug/energy_comm":    iter_info.get("mean_energy_comm_per_request", 0.0),
            "metrics_debug/penalty_q":      iter_info.get("mean_penalty_q", 0.0),
            "metrics_debug/delay_reward":   iter_info.get("mean_delay_reward", 0.0),
            "metrics_debug/energy_reward":  iter_info.get("mean_energy_reward", 0.0),
        }

        self.logger.log_metrics(metrics, iteration)
        self.logger.log_metrics(debug, iteration)

    def _log_eval_metrics(self, iteration: int, eval_results: Dict):
        """Log compact eval metrics; x-axis is PPO iteration."""
        if self.logger is None:
            return

        metrics = {}
        if "mean_delay" in eval_results:
            metrics["metrics/eval_delay"] = eval_results.get("mean_delay")
        if "mean_energy" in eval_results:
            metrics["metrics/eval_energy"] = eval_results.get("mean_energy")
        if "mean_accuracy" in eval_results:
            metrics["metrics/eval_accuracy"] = eval_results.get("mean_accuracy")
        if metrics:
            self.logger.log_metrics(metrics, iteration)
    
    def _save_checkpoint(self, episode: int, is_best: bool = False):
        """Save checkpoint."""
        if is_best:
            save_path = self.checkpoint_dir / f"{self.agent.name}_best.pth"
            print(f"[INFO] Saved best model to {save_path}")
        else:
            save_path = self.checkpoint_dir / f"{self.agent.name}_ep{episode}.pth"

        self.agent.save(str(save_path))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert value to finite float; return default on failure."""
        try:
            x = float(value)
            if math.isfinite(x):
                return x
            return float(default)
        except Exception:
            return float(default)

    def _warn_non_finite_dict(self, name: str, values: Dict, episode: int) -> None:
        """Log warning when update metrics contain invalid values."""
        if not values:
            return
        bad = {}
        for k, v in values.items():
            fv = self._safe_float(v, default=float("nan"))
            if not math.isfinite(fv):
                bad[k] = str(v)
        if bad:
            self._log_event(
                "non_finite_metrics",
                {"name": name, "episode": episode, "metrics": bad},
                step=self.total_steps,
                level="WARN",
            )

    def _log_event(self, event: str, payload: Dict, step: int = None, level: str = "INFO") -> None:
        """Write structured event through logger if available."""
        if self.logger is not None and hasattr(self.logger, "log_event"):
            self.logger.log_event(event=event, payload=payload, step=step, level=level)

    def _format_metric_str(self, metrics: Dict, key: str, fmt: str, default: str = "NA") -> str:
        """Format metric from dict if finite; otherwise return default string."""
        if not metrics or key not in metrics:
            return default
        x = self._safe_float(metrics.get(key), default=float("nan"))
        if not math.isfinite(x):
            return default
        return format(x, fmt)

    def _scale_reward_for_update(self, reward_raw: float):
        """
        Scale raw reward into a bounded training signal.

        Returns:
            (scaled_reward, is_clipped)
        """
        r = self._safe_float(reward_raw, default=-1e6)
        if not self.reward_signal_scaling_enabled:
            return r, False

        target_abs = self._current_reward_signal_target_abs()

        if self.reward_signal_scaling_mode == "window_standardize":
            # O(1) sliding-window stats (space-for-time):
            # avoid O(window_size) numpy conversion per environment step.
            if len(self.reward_signal_window) == self.reward_signal_window.maxlen:
                old = float(self.reward_signal_window[0])
                self.reward_signal_window_sum -= old
                self.reward_signal_window_sq_sum -= old * old
            self.reward_signal_window.append(r)
            self.reward_signal_window_sum += r
            self.reward_signal_window_sq_sum += r * r
            if len(self.reward_signal_window) >= self.reward_signal_window_min_samples:
                n = float(len(self.reward_signal_window))
                mean = float(self.reward_signal_window_sum / n)
                var = float(max(self.reward_signal_window_sq_sum / n - mean * mean, 0.0))
                std = float(math.sqrt(var))
                std = max(std, self.reward_signal_window_eps)
                self.reward_signal_window_mean = mean
                self.reward_signal_window_std = std
                z = (r - mean) / std
                z_clip = max(self.reward_signal_window_z_clip, self.reward_signal_window_eps)
                zc = float(np.clip(z, -z_clip, z_clip))
                is_clipped = bool(abs(z) > z_clip + 1e-12)
                r_scaled = (zc / z_clip) * target_abs
                return float(r_scaled), is_clipped
            # Fallback in the early phase before window has enough samples.
            # Keep scale-only behavior to avoid unstable std estimates.

        abs_r = abs(r)
        self.reward_signal_running_abs = max(
            abs_r,
            self.reward_signal_running_abs * self.reward_signal_decay,
        )
        denom = max(self.reward_signal_running_abs, 1e-6)
        scale = target_abs / denom
        r_scaled = r * scale
        r_clipped = float(np.clip(r_scaled, -target_abs, target_abs))
        is_clipped = bool(abs(r_scaled) > target_abs + 1e-12)
        return r_clipped, is_clipped

    def _current_reward_signal_target_abs(self) -> float:
        """Return current target bound for reward signal scaling."""
        if (
            self.reward_signal_scaling_enabled
            and self.reward_signal_warmup_enabled
            and self.episode_count < self.reward_signal_warmup_episodes
        ):
            return float(self.reward_signal_warmup_target_abs)
        return float(self.reward_signal_target_abs)

    def _get_rollout_progress(self):
        """Return current buffer length and update target steps if available."""
        target = getattr(self.agent, "n_steps", None)
        buffer_obj = getattr(self.agent, "buffer", None)
        if buffer_obj is None or not hasattr(buffer_obj, "rewards"):
            return None, target
        try:
            current = len(buffer_obj.rewards)
        except Exception:
            current = None
        return current, target

    @staticmethod
    def _format_buffer_progress(current, target) -> str:
        """Format rollout buffer progress for progress bar display."""
        if current is None and target is None:
            return "-"
        if current is None:
            return f"?/{target}"
        if target is None:
            return str(current)
        return f"{current}/{target}"

