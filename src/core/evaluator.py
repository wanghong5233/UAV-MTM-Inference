"""
Unified evaluator.

Provides evaluation logic decoupled from algorithms.
"""

import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from .base_agent import BaseAgent


class Evaluator:
    """Generic evaluator for DRL agents."""
    
    def __init__(self, env, config: Dict):
        """Initialize the evaluator."""
        self.env = env
        self.config = config
        
        # Evaluation configuration
        eval_config = config.get('evaluation', {})
        self.num_episodes = eval_config.get('num_episodes', 100)
        self.deterministic = eval_config.get('deterministic', True)
        self.save_trajectories = eval_config.get('save_trajectories', False)
        
        # Max steps per episode
        self.max_steps = config.get('training', {}).get('max_steps_per_episode', 100)
    
    def evaluate(self, agent: BaseAgent, verbose: bool = False) -> Dict:
        """Evaluate the current agent."""
        agent.eval_mode()
        
        episode_rewards = []
        episode_lengths = []
        episode_delays = []
        episode_energies = []
        episode_accuracies = []
        episode_delay_per_step = []
        episode_energy_per_step = []
        episode_comm_volumes = []
        episode_peak_rhos = []
        episode_num_blocks = []
        episode_infeasible_links = []
        episode_energy_compute = []
        episode_energy_comm = []
        
        trajectories = [] if self.save_trajectories else None
        
        iterator = tqdm(range(self.num_episodes), desc="Evaluating", ascii=True) if verbose else range(self.num_episodes)
        
        for _ in iterator:
            episode_data = self._evaluate_episode(agent)
            
            episode_rewards.append(episode_data['reward'])
            episode_lengths.append(episode_data['length'])
            episode_delays.append(episode_data.get('delay', 0.0))
            episode_energies.append(episode_data.get('energy', 0.0))
            episode_accuracies.append(episode_data.get('accuracy', 0.0))
            episode_delay_per_step.append(episode_data.get('delay_per_step', 0.0))
            episode_energy_per_step.append(episode_data.get('energy_per_step', 0.0))
            episode_comm_volumes.append(episode_data.get('comm_volume_per_step', 0.0))
            episode_peak_rhos.append(episode_data.get('rho_max_per_step', 0.0))
            episode_num_blocks.append(episode_data.get('num_blocks_per_step', 0.0))
            episode_infeasible_links.append(episode_data.get('infeasible_links_per_step', 0.0))
            episode_energy_compute.append(episode_data.get('energy_compute_per_step', 0.0))
            episode_energy_comm.append(episode_data.get('energy_comm_per_step', 0.0))
            
            if self.save_trajectories:
                trajectories.append(episode_data.get('trajectory', []))
        
        # Aggregate statistics. Sample std across episodes is included for every
        # paper-facing metric so that bar/line plots can render error bars/bands
        # for both learning-based and deterministic agents.
        results = {
            'mean_reward':                    np.mean(episode_rewards),
            'std_reward':                     np.std(episode_rewards),
            'mean_length':                    np.mean(episode_lengths),
            'mean_delay':                     np.mean(episode_delays),
            'std_delay':                      np.std(episode_delays),
            'mean_energy':                    np.mean(episode_energies),
            'std_energy':                     np.std(episode_energies),
            'mean_accuracy':                  np.mean(episode_accuracies),
            'std_accuracy':                   np.std(episode_accuracies),
            'mean_delay_per_step':            np.mean(episode_delay_per_step),
            'std_delay_per_step':             np.std(episode_delay_per_step),
            'mean_energy_per_step':           np.mean(episode_energy_per_step),
            'std_energy_per_step':            np.std(episode_energy_per_step),
            'mean_comm_volume_per_step':      np.mean(episode_comm_volumes),
            'mean_rho_max_per_step':          np.mean(episode_peak_rhos),
            'mean_num_blocks_per_step':       np.mean(episode_num_blocks),
            'mean_infeasible_links_per_step': np.mean(episode_infeasible_links),
            'mean_energy_compute_per_step':   np.mean(episode_energy_compute),
            'mean_energy_comm_per_step':      np.mean(episode_energy_comm),
            'num_episodes':                   self.num_episodes,
        }
        
        if self.save_trajectories:
            results['trajectories'] = trajectories
        
        return results
    
    def _evaluate_episode(self, agent: BaseAgent) -> Dict:
        """Evaluate a single episode."""
        state, _ = self.env.reset()
        done = False
        truncated = False
        step = 0
        
        episode_reward = 0.0
        episode_delay = 0.0
        episode_energy = 0.0
        episode_accuracy = 0.0
        episode_comm_volume = 0.0
        episode_rho_max = 0.0
        episode_num_blocks = 0.0
        episode_infeasible_links = 0.0
        episode_energy_compute = 0.0
        episode_energy_comm = 0.0
        
        trajectory = [] if self.save_trajectories else None
        
        while not (done or truncated) and step < self.max_steps:
            # Select action (deterministic)
            action = agent.select_action(state, deterministic=self.deterministic)
            
            # Step environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Accumulate metrics
            episode_reward += reward
            episode_delay += info.get('delay', 0.0)
            episode_energy += info.get('energy', 0.0)
            episode_accuracy += info.get('accuracy', 0.0)
            episode_comm_volume += info.get('comm_volume_total', 0.0)
            episode_rho_max += info.get('rho_max', 0.0)
            episode_num_blocks += info.get('num_blocks', 0.0)
            episode_infeasible_links += info.get('infeasible_links', 0.0)
            episode_energy_compute += info.get('energy_compute_per_request', 0.0)
            episode_energy_comm += info.get('energy_comm_per_request', 0.0)
            
            # Save trajectory
            if self.save_trajectories:
                trajectory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'info': info,
                })
            
            state = next_state
            step += 1
        
        return {
            'reward': episode_reward,
            'length': step,
            'delay': episode_delay,
            'energy': episode_energy,
            'accuracy': episode_accuracy / max(step, 1),  # mean accuracy
            'delay_per_step': episode_delay / max(step, 1),
            'energy_per_step': episode_energy / max(step, 1),
            'comm_volume_per_step': episode_comm_volume / max(step, 1),
            'rho_max_per_step': episode_rho_max / max(step, 1),
            'num_blocks_per_step': episode_num_blocks / max(step, 1),
            'infeasible_links_per_step': episode_infeasible_links / max(step, 1),
            'energy_compute_per_step': episode_energy_compute / max(step, 1),
            'energy_comm_per_step': episode_energy_comm / max(step, 1),
            'trajectory': trajectory,
        }

