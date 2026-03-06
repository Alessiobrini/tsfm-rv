"""
evaluation/mcs.py — Model Confidence Set (Hansen, Lunde & Nason 2011).

The MCS procedure identifies the set of models that contains the best model
with a given confidence level. It sequentially eliminates the worst-performing
model until no model can be rejected.

Reference: Hansen, Lunde & Nason (2011, Econometrica).

Uses the T_max statistic with block bootstrap for p-value computation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MCSResult:
    """Result of Model Confidence Set procedure."""
    surviving_models: List[str]
    eliminated_models: List[str]
    p_values: Dict[str, float]
    alpha: float


def block_bootstrap_indices(
    T: int,
    block_length: int,
    n_bootstrap: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate block bootstrap index arrays.

    Returns shape (n_bootstrap, T).
    """
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(T / block_length))
    indices = np.zeros((n_bootstrap, T), dtype=int)

    for b in range(n_bootstrap):
        starts = rng.randint(0, T - block_length + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
        indices[b] = idx[:T]

    return indices


def model_confidence_set(
    losses: Dict[str, np.ndarray],
    alpha: float = 0.10,
    n_bootstrap: int = 10000,
    block_length: int = 22,
    seed: int = 42,
) -> MCSResult:
    """Compute the Model Confidence Set.

    Parameters
    ----------
    losses : Dict[str, np.ndarray]
        Keys: model names. Values: T-length arrays of loss values.
    alpha : float
        Significance level.
    n_bootstrap : int
        Number of block bootstrap replications.
    block_length : int
        Block length for block bootstrap.
    seed : int
        Random seed.

    Returns
    -------
    MCSResult
    """
    model_names = list(losses.keys())
    loss_matrix = np.column_stack([losses[m] for m in model_names])
    T, M = loss_matrix.shape

    boot_indices = block_bootstrap_indices(T, block_length, n_bootstrap, seed)

    surviving = list(range(M))
    eliminated = []
    p_values = {}

    while len(surviving) > 1:
        n_surv = len(surviving)

        # Compute pairwise loss differentials and their variances
        d_bar = np.zeros((n_surv, n_surv))
        var_d = np.zeros((n_surv, n_surv))

        for i in range(n_surv):
            for j in range(i + 1, n_surv):
                d_ij = loss_matrix[:, surviving[i]] - loss_matrix[:, surviving[j]]
                d_bar[i, j] = np.mean(d_ij)
                d_bar[j, i] = -d_bar[i, j]
                # Bootstrap variance (computed ONCE per pair)
                boot_means = np.mean(d_ij[boot_indices], axis=1)
                v = float(np.var(boot_means))
                var_d[i, j] = v
                var_d[j, i] = v

        # T_max statistic: max |t_ij| over pairs
        t_stats = np.zeros((n_surv, n_surv))
        for i in range(n_surv):
            for j in range(i + 1, n_surv):
                if var_d[i, j] > 0:
                    t_stats[i, j] = d_bar[i, j] / np.sqrt(var_d[i, j])
                    t_stats[j, i] = -t_stats[i, j]

        T_max = np.max(np.abs(t_stats))

        # Bootstrap distribution of T_max under H0
        T_max_boot = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            boot_loss = loss_matrix[boot_indices[b]]
            t_boot = np.zeros((n_surv, n_surv))
            for i in range(n_surv):
                for j in range(i + 1, n_surv):
                    d_ij_boot = boot_loss[:, surviving[i]] - boot_loss[:, surviving[j]]
                    d_bar_boot = np.mean(d_ij_boot)
                    if var_d[i, j] > 0:
                        t_boot[i, j] = (d_bar_boot - d_bar[i, j]) / np.sqrt(var_d[i, j])
            T_max_boot[b] = np.max(np.abs(t_boot))

        # p-value
        p_val = np.mean(T_max_boot >= T_max)

        if p_val < alpha:
            # Eliminate worst model
            avg_losses = np.mean(loss_matrix[:, surviving], axis=0)
            worst_idx = np.argmax(avg_losses)
            worst_model = surviving[worst_idx]
            eliminated.append(model_names[worst_model])
            p_values[model_names[worst_model]] = p_val
            surviving.pop(worst_idx)
        else:
            break

    for idx in surviving:
        p_values[model_names[idx]] = 1.0

    return MCSResult(
        surviving_models=[model_names[i] for i in surviving],
        eliminated_models=eliminated,
        p_values=p_values,
        alpha=alpha,
    )
