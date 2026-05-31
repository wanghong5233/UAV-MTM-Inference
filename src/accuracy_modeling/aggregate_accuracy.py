"""
Aggregate accuracy construction for multi-task scene understanding.

We define a dimensionless aggregate accuracy score A ∈ [0,1] by normalizing each
task metric w.r.t. the full-precision baseline (reference) and then averaging.

This implements Scheme A ("relative degradation") discussed in the paper notes:
  - Baseline (full precision) yields A_full = 1.
  - Quantization degrades metrics -> A decreases.

Notation:
  seg metric: mIoU (higher is better)
  depth metric: absolute error (lower is better)
  normal metric: mean angular error (lower is better)
"""

from __future__ import annotations

from typing import Dict, Tuple


def compute_aggregate_accuracy_relative(
    metrics: Dict,
    reference_metrics: Dict,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute aggregate accuracy A ∈ [0,1] using relative degradation w.r.t. a reference.

    For segmentation (higher better): relative degradation = max(0, (m_ref - m) / m_ref)
    For depth/normal (lower better): relative degradation = max(0, (e - e_ref) / e_ref)

    The per-task scores are:
      A_task = clamp(1 - degradation, 0, 1)
    and the aggregate is the weighted average.

    Args:
        metrics: dict with keys 'seg_miou', 'depth_abs_err', 'normal_mean'
        reference_metrics: dict with the same keys for full precision baseline
        weights: (w_seg, w_depth, w_normal)
        eps: numerical stability for denominators

    Returns:
        dict:
          - A: aggregate accuracy in [0,1]
          - A_seg, A_depth, A_normal: per-task scores in [0,1]
          - delta_seg, delta_depth, delta_normal: relative degradations (≥0)
    """
    w_seg, w_depth, w_normal = weights

    m = float(metrics["seg_miou"])
    d = float(metrics["depth_abs_err"])
    n = float(metrics["normal_mean"])

    m_ref = float(reference_metrics["seg_miou"])
    d_ref = float(reference_metrics["depth_abs_err"])
    n_ref = float(reference_metrics["normal_mean"])

    # Relative degradations (non-negative)
    delta_seg = max(0.0, (m_ref - m) / max(eps, m_ref))
    delta_depth = max(0.0, (d - d_ref) / max(eps, d_ref))
    delta_normal = max(0.0, (n - n_ref) / max(eps, n_ref))

    # Per-task scores in [0,1]
    A_seg = min(1.0, max(0.0, 1.0 - delta_seg))
    A_depth = min(1.0, max(0.0, 1.0 - delta_depth))
    A_normal = min(1.0, max(0.0, 1.0 - delta_normal))

    denom = max(eps, (w_seg + w_depth + w_normal))
    A = (w_seg * A_seg + w_depth * A_depth + w_normal * A_normal) / denom

    return {
        "A": A,
        "A_seg": A_seg,
        "A_depth": A_depth,
        "A_normal": A_normal,
        "delta_seg": delta_seg,
        "delta_depth": delta_depth,
        "delta_normal": delta_normal,
    }

