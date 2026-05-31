"""
Unified evaluation script.

Usage:
    python scripts/evaluate.py --config configs/experiments/main_gnn_ppo.yaml --checkpoint checkpoints/gnn_ppo_best.pth
    python scripts/evaluate.py --config configs/experiments/baseline_local_only_tmc_stable_split.yaml
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config, apply_overrides
from src.core import AgentRegistry, Evaluator
from src.env import UAVEnv
import src.agents  # noqa: F401  # Ensure agent classes are registered


def main():
    parser = argparse.ArgumentParser(description='Evaluate DRL algorithm')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Optional path to save metrics JSON')
    parser.add_argument('--device', type=str, default=None, help='Device override (cuda/cpu)')
    parser.add_argument('--num_episodes', type=int, default=None, help='Override evaluation.num_episodes')
    parser.add_argument('--seed', type=int, default=None, help='Optional environment seed override')
    parser.add_argument('--weights', type=float, nargs=2, default=None, metavar=('W_DELAY', 'W_ENERGY'),
                        help='Override preference.weights, e.g. --weights 0.5 0.5')
    parser.add_argument('--arrival_rate', type=float, default=None, help='Override task.arrival_rate (req/s)')
    parser.add_argument('--num_uavs', type=int, default=None, help='Override uav.num_uavs')
    parser.add_argument('--area_size', type=float, nargs=2, default=None, metavar=('W', 'H'),
                        help='Override uav.area_size (meters)')
    parser.add_argument('--avg_tasks_per_request', type=float, default=None,
                        help='Override task.avg_tasks_per_request')
    parser.add_argument('--max_range', type=float, default=None,
                        help='Override network.max_range (meters)')
    parser.add_argument('--cpu_threads', type=int, default=None, help='Override Torch CPU intra-op threads')
    parser.add_argument('--run_tag', type=str, default=None, help='Optional tag appended to output path')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='KEY=VALUE', help='Generic config override (dot-path), e.g. --set task.deadline=3.0')
    args = parser.parse_args()
    
    # Load configuration.
    config = load_config(args.config)
    
    # Apply CLI overrides.
    if args.device is not None:
        config['device'] = args.device
    if args.num_episodes is not None:
        config.setdefault('evaluation', {})['num_episodes'] = int(args.num_episodes)
    if args.seed is not None:
        config['seed'] = int(args.seed)
    if args.weights is not None:
        config.setdefault('preference', {})['weights'] = list(args.weights)
        config['preference']['mode'] = 'fixed'
    if args.arrival_rate is not None:
        config.setdefault('task', {})['arrival_rate'] = float(args.arrival_rate)
    if args.num_uavs is not None:
        uav_cfg = config.setdefault('uav', {})
        uav_cfg['num_uavs'] = int(args.num_uavs)
        uav_cfg['num_task_uavs'] = 1
        uav_cfg['num_worker_uavs'] = max(int(args.num_uavs) - 1, 0)
    if args.area_size is not None:
        config.setdefault('uav', {})['area_size'] = [float(args.area_size[0]), float(args.area_size[1])]
    if args.avg_tasks_per_request is not None:
        config.setdefault('task', {})['avg_tasks_per_request'] = float(args.avg_tasks_per_request)
    if args.max_range is not None:
        config.setdefault('network', {})['max_range'] = float(args.max_range)
    if args.cpu_threads is not None:
        import torch
        torch.set_num_threads(int(args.cpu_threads))
    if args.overrides:
        apply_overrides(config, args.overrides)

    # Create environment.
    env = UAVEnv(config)
    
    # Create agent.
    algo_name = config.get('algorithm', {}).get('name', 'gnn_ppo')
    agent = AgentRegistry.create(algo_name, env, config)
    
    # Load checkpoint if provided.
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    else:
        print("[INFO] No checkpoint provided; evaluating the policy as initialized/configured.")
    
    # Create evaluator.
    evaluator = Evaluator(env, config)
    
    # Evaluate.
    print("Starting evaluation...")
    results = evaluator.evaluate(agent, verbose=True)
    
    # Print results.
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for key, value in results.items():
        if not key == 'trajectories':
            print(f"{key}: {value:.4f}")
    
    # Save results.
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove trajectories because they can be too large.
        results_to_save = {}
        for k, v in results.items():
            if k == 'trajectories':
                continue
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            results_to_save[k] = v

        # Normalize deterministic-evaluation outputs to paper-facing semantics:
        # use per-step delay/energy/reward as the primary fields, while preserving the
        # raw episode-total fields for debugging/traceability.
        mean_length = results_to_save.get('mean_length', 50.0) or 50.0
        if 'mean_delay_per_step' in results_to_save:
            results_to_save['mean_delay_episode_total'] = results_to_save.get('mean_delay')
            results_to_save['mean_delay'] = results_to_save['mean_delay_per_step']
        if 'std_delay_per_step' in results_to_save:
            results_to_save['std_delay_episode_total'] = results_to_save.get('std_delay')
            results_to_save['std_delay'] = results_to_save['std_delay_per_step']
        if 'mean_energy_per_step' in results_to_save:
            results_to_save['mean_energy_episode_total'] = results_to_save.get('mean_energy')
            results_to_save['mean_energy'] = results_to_save['mean_energy_per_step']
        if 'std_energy_per_step' in results_to_save:
            results_to_save['std_energy_episode_total'] = results_to_save.get('std_energy')
            results_to_save['std_energy'] = results_to_save['std_energy_per_step']
        if 'mean_reward' in results_to_save and mean_length > 1:
            raw_reward = results_to_save['mean_reward']
            # If reward magnitude >> 10, it's clearly episode-total, normalize it
            if abs(raw_reward) > 10:
                results_to_save['mean_reward_episode_total'] = raw_reward
                results_to_save['mean_reward'] = raw_reward / mean_length
                if 'std_reward' in results_to_save:
                    results_to_save['std_reward_episode_total'] = results_to_save['std_reward']
                    results_to_save['std_reward'] = results_to_save['std_reward'] / mean_length
        if 'mean_comm_volume_per_step' in results_to_save:
            results_to_save['mean_comm_volume'] = results_to_save['mean_comm_volume_per_step']
        if 'mean_rho_max_per_step' in results_to_save:
            results_to_save['mean_rho_max'] = results_to_save['mean_rho_max_per_step']
        if 'mean_energy_compute_per_step' in results_to_save:
            results_to_save['mean_energy_compute'] = results_to_save['mean_energy_compute_per_step']
        if 'mean_energy_comm_per_step' in results_to_save:
            results_to_save['mean_energy_comm'] = results_to_save['mean_energy_comm_per_step']
        results_to_save['source'] = 'deterministic_evaluation_per_step'

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n[INFO] Results saved to {output_path}")


if __name__ == '__main__':
    main()

