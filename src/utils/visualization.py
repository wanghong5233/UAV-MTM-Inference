"""
Visualization utilities.

Generate training and analysis figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from pathlib import Path


# Plot style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_curves(
    results: Dict[str, List],
    save_path: str = None
):
    """
    Plot common training curves from a result dictionary.

    Args:
        results: Dict of scalar series, e.g., {"reward": [...], "loss": [...]}
        save_path: Optional output path.
    """
    if not results:
        return

    keys = [k for k, v in results.items() if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0]
    if not keys:
        return

    fig, ax = plt.subplots()
    for k in keys:
        y = np.asarray(results[k], dtype=np.float32)
        x = np.arange(len(y))
        ax.plot(x, y, label=k)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_comparison(
    algorithm_results: Dict[str, Dict],
    metrics: List[str],
    save_path: str = None
):
    """
    Plot a grouped bar chart for algorithm comparison.
    """
    if not algorithm_results or not metrics:
        return

    algos = list(algorithm_results.keys())
    x = np.arange(len(algos))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots()
    for i, metric in enumerate(metrics):
        vals = [algorithm_results[a].get(metric, 0.0) for a in algos]
        ax.bar(x + i * width, vals, width=width, label=metric)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(algos, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_accuracy_fitting(
    distortions: np.ndarray,
    accuracies: np.ndarray,
    fitted_curve: np.ndarray = None,
    save_path: str = None
):
    """Plot accuracy fitting scatter/curve."""
    fig, ax = plt.subplots()
    
    # Scatter points
    ax.scatter(distortions, accuracies, alpha=0.6, label='Sampling Points')
    
    # Fitted curve
    if fitted_curve is not None:
        ax.plot(distortions, fitted_curve, 'r-', linewidth=2, label='Fitted Curve')
    
    ax.set_xlabel('Distortion (NMSE)')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pareto_frontier(
    algorithm_results: Dict[str, Dict],
    x_metric: str = 'delay',
    y_metric: str = 'energy',
    save_path: str = None
):
    """Plot Pareto-style scatter on two metrics."""
    if not algorithm_results:
        return

    fig, ax = plt.subplots()
    for name, res in algorithm_results.items():
        x = res.get(x_metric, None)
        y = res.get(y_metric, None)
        if x is None or y is None:
            continue
        ax.scatter([x], [y], label=name)
        ax.annotate(name, (x, y))

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_results(results: Dict[str, List], save_path: str = None):
    """Backward-compatible alias for training curve plotting."""
    return plot_training_curves(results, save_path=save_path)

