"""
Unified training script (config-driven).

Usage:
    python scripts/train.py --config configs/experiments/main_gnn_ppo.yaml
"""

import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch

# Add project root to import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config, apply_overrides
from src.core import AgentRegistry, Trainer, Evaluator
from src.env import UAVEnv
from src.utils import Logger, ExperimentTracker
import src.agents  # noqa: F401  # Ensure agent classes are registered


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Train DRL algorithm')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    parser.add_argument('--device', type=str, default=None, help='Device override (cuda/cpu)')
    parser.add_argument('--num_episodes', type=int, default=None, help='Override training.num_episodes')
    parser.add_argument('--max_steps', type=int, default=None, help='Override training.max_steps_per_episode')
    parser.add_argument('--ppo_n_steps', type=int, default=None, help='Override algorithm.ppo.n_steps')
    parser.add_argument('--ppo_n_epochs', type=int, default=None, help='Override algorithm.ppo.n_epochs')
    parser.add_argument('--ppo_batch_size', type=int, default=None, help='Override algorithm.ppo.batch_size')
    parser.add_argument('--console_mode', type=str, choices=['tqdm', 'plain'], default=None, help='Console output mode override')
    parser.add_argument('--disable_eval', action='store_true', help='Disable periodic evaluation (fast iteration)')
    parser.add_argument('--disable_save', action='store_true', help='Disable periodic checkpoint saving (fast iteration)')
    parser.add_argument('--run_tag', type=str, default=None, help='Optional run tag for output folder')
    parser.add_argument('--run_group', type=str, default=None,
                        help='Optional experiment group under logs/training/, e.g. e2_pareto')
    parser.add_argument('--cpu_threads', type=int, default=None, help='Override Torch CPU intra-op threads (CPU runs only)')
    parser.add_argument('--cpu_interop_threads', type=int, default=None, help='Override Torch CPU inter-op threads (CPU runs only)')
    parser.add_argument('--weights', type=float, nargs=2, default=None, metavar=('W_DELAY', 'W_ENERGY'),
                        help='Override preference.weights, e.g. --weights 0.5 0.5')
    parser.add_argument('--arrival_rate', type=float, default=None, help='Override task.arrival_rate (req/s)')
    parser.add_argument('--num_uavs', type=int, default=None, help='Override uav.num_uavs')
    parser.add_argument('--area_size', type=float, nargs=2, default=None, metavar=('W', 'H'),
                        help='Override uav.area_size (meters), e.g. --area_size 1000 1000')
    parser.add_argument('--avg_tasks_per_request', type=float, default=None,
                        help='Override task.avg_tasks_per_request (average number of active tasks per request)')
    parser.add_argument('--max_range', type=float, default=None,
                        help='Override network.max_range (meters); links beyond this distance are unreachable')
    parser.add_argument(
        '--keep_run_dir',
        action='store_true',
        help='Keep existing run directory when run_tag already exists (default: overwrite for clean rerun)',
    )
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='KEY=VALUE', help='Generic config override (dot-path), e.g. --set task.deadline=3.0')
    args = parser.parse_args()
    
    # Load configuration.
    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    
    # Apply CLI overrides.
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    if args.num_episodes is not None:
        config.setdefault('training', {})['num_episodes'] = int(args.num_episodes)
    if args.max_steps is not None:
        config.setdefault('training', {})['max_steps_per_episode'] = int(args.max_steps)
    if args.ppo_n_steps is not None:
        config.setdefault('algorithm', {}).setdefault('ppo', {})['n_steps'] = int(args.ppo_n_steps)
    if args.ppo_n_epochs is not None:
        config.setdefault('algorithm', {}).setdefault('ppo', {})['n_epochs'] = int(args.ppo_n_epochs)
    if args.ppo_batch_size is not None:
        config.setdefault('algorithm', {}).setdefault('ppo', {})['batch_size'] = int(args.ppo_batch_size)
    if args.console_mode is not None:
        config.setdefault('logging', {})['console_mode'] = str(args.console_mode)
    if args.disable_eval:
        config.setdefault('training', {})['eval_interval'] = 0
    if args.disable_save:
        config.setdefault('training', {})['save_interval'] = 0
    if args.cpu_threads is not None:
        config.setdefault('runtime', {})['cpu_threads'] = int(args.cpu_threads)
    if args.cpu_interop_threads is not None:
        config.setdefault('runtime', {})['cpu_interop_threads'] = int(args.cpu_interop_threads)
    if args.weights is not None:
        config.setdefault('preference', {})['weights'] = list(args.weights)
        config['preference']['mode'] = 'fixed'
    if args.arrival_rate is not None:
        config.setdefault('task', {})['arrival_rate'] = float(args.arrival_rate)
    if args.num_uavs is not None:
        uav_cfg = config.setdefault('uav', {})
        uav_cfg['num_uavs'] = int(args.num_uavs)
        # Keep source/worker counts consistent with the new total.
        # Source UAV count is assumed to remain 1 (the sensing node).
        uav_cfg['num_task_uavs'] = 1
        uav_cfg['num_worker_uavs'] = max(int(args.num_uavs) - 1, 0)
    if args.area_size is not None:
        config.setdefault('uav', {})['area_size'] = [float(args.area_size[0]), float(args.area_size[1])]
    if args.avg_tasks_per_request is not None:
        config.setdefault('task', {})['avg_tasks_per_request'] = float(args.avg_tasks_per_request)
    if args.max_range is not None:
        config.setdefault('network', {})['max_range'] = float(args.max_range)
    if args.overrides:
        apply_overrides(config, args.overrides)
    
    # Create experiment tracker.
    experiment_name = config.get('experiment', {}).get('name', 'default')
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("logs") / "training"
    if args.run_group:
        run_root = run_root / str(args.run_group)
    run_dir = run_root / experiment_name / run_tag
    if args.run_group:
        config.setdefault("runtime", {})["run_group"] = str(args.run_group)
    if run_dir.exists() and not args.keep_run_dir:
        print(f"[INFO] Existing run directory found, removing for clean rerun: {run_dir}")
        shutil.rmtree(run_dir)
    config.setdefault("runtime", {})["run_dir"] = str(run_dir)
    config["checkpoint_dir"] = str(run_dir / "checkpoints")

    # Optional CPU threading tuning (effective only for CPU runs).
    runtime_cfg = config.setdefault("runtime", {})
    requested_device_cfg = str(config.get("device", "auto")).lower()
    use_cpu_backend = requested_device_cfg == "cpu" or (
        requested_device_cfg == "auto" and (not torch.cuda.is_available())
    )
    if use_cpu_backend:
        cpu_threads = int(runtime_cfg.get("cpu_threads", 0) or 0)
        cpu_interop_threads = int(runtime_cfg.get("cpu_interop_threads", 0) or 0)
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
        if cpu_interop_threads > 0:
            try:
                torch.set_num_interop_threads(cpu_interop_threads)
            except RuntimeError as exc:
                print(f"[WARN] Failed to set cpu_interop_threads={cpu_interop_threads}: {exc}")
        # Persist effective values for reproducibility.
        runtime_cfg["cpu_threads_effective"] = int(torch.get_num_threads())
        runtime_cfg["cpu_interop_threads_effective"] = int(torch.get_num_interop_threads())
        runtime_cfg["cpu_logical_cores"] = int(os.cpu_count() or 1)
        print(
            "[INFO] CPU threading: "
            f"logical_cores={runtime_cfg['cpu_logical_cores']} "
            f"intra_op={runtime_cfg['cpu_threads_effective']} "
            f"interop={runtime_cfg['cpu_interop_threads_effective']}"
        )

    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        config=config,
        output_dir=str(run_dir)
    )
    print(f"[INFO] Run directory: {run_dir}")

    # Global seed for full reproducibility.
    global_seed = int(config.get("seed", 42))
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global seed: {global_seed}")
    
    # Create environment.
    print("Creating environment...")
    env = UAVEnv(config)
    
    # Create agent.
    algo_name = config.get('algorithm', {}).get('name', 'gnn_ppo')
    print(f"Creating agent: {algo_name}")
    agent = AgentRegistry.create(algo_name, env, config)
    requested_device = str(config.get("device", "auto"))
    actual_device = str(agent.device)
    cuda_available = torch.cuda.is_available()
    device_msg = (
        f"[INFO] Device requested={requested_device} | actual={actual_device} | cuda_available={cuda_available}"
    )
    if actual_device.startswith("cuda") and cuda_available:
        device_index = agent.device.index if agent.device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        device_msg += f" | gpu={gpu_name}"
    print(device_msg)
    print(
        "[INFO] Env summary: "
        f"uavs={env.num_uavs}, model_nodes={len(env.model_graph.node_names)}, "
        f"groups={len(env.model_graph.compression_groups)}"
    )
    
    # Create logger.
    logger = Logger(config, log_dir=str(run_dir))
    
    # Create evaluator.
    evaluator = Evaluator(env, config)
    
    # Create trainer.
    trainer = Trainer(agent, config, logger, evaluator)
    
    # Start training.
    print("="*50)
    print("Start training")
    print("="*50)
    try:
        summary = trainer.train()
        tracker.log_result(summary)
        print("[INFO] Training finished.")
    except Exception as exc:
        if logger is not None:
            logger.log_event(
                event="training_exception",
                payload={"error": repr(exc), "traceback": traceback.format_exc()},
                level="ERROR",
            )
        raise
    finally:
        # Finalize.
        if logger is not None:
            logger.close()
        tracker.finish()


if __name__ == '__main__':
    main()

