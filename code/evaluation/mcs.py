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

    # Precompute bootstrap means for ALL models once (avoids repeated
    # 2.6GB allocations inside the while loop).
    # all_boot_means[b, m] = mean of loss_matrix[boot_indices[b], m]
    all_boot_means = np.zeros((n_bootstrap, M))
    for m_idx in range(M):
        col = loss_matrix[:, m_idx]
        all_boot_means[:, m_idx] = np.mean(col[boot_indices], axis=1)

    surviving = list(range(M))
    eliminated = []
    p_values = {}

    while len(surviving) > 1:
        n_surv = len(surviving)

        # Sample means and bootstrap variances for surviving pairs
        surv_means = np.mean(loss_matrix[:, surviving], axis=0)  # (n_surv,)
        boot_m = all_boot_means[:, surviving]  # (n_bootstrap, n_surv)

        # Pairwise differences: d_bar[i,j] = mean(L_i - L_j)
        d_bar = surv_means[:, None] - surv_means[None, :]  # (n_surv, n_surv)

        # Bootstrap variance of pairwise mean differences
        # boot_diff[b, i, j] = boot_m[b, i] - boot_m[b, j]
        # var_d[i, j] = Var_b(boot_diff[:, i, j])
        ii, jj = np.triu_indices(n_surv, k=1)
        boot_diff_pairs = boot_m[:, ii] - boot_m[:, jj]  # (B, n_pairs)
        var_pairs = np.var(boot_diff_pairs, axis=0)        # (n_pairs,)
        sd_pairs = np.sqrt(np.maximum(var_pairs, 1e-30))

        # T-statistics for observed data
        d_bar_pairs = d_bar[ii, jj]
        t_pairs = d_bar_pairs / sd_pairs
        T_max = np.max(np.abs(t_pairs))

        # Bootstrap T_max distribution
        t_boot = (boot_diff_pairs - d_bar_pairs[None, :]) / sd_pairs[None, :]
        T_max_boot = np.max(np.abs(t_boot), axis=1)

        # p-value
        p_val = np.mean(T_max_boot >= T_max)

        if p_val < alpha:
            # Eliminate worst model (highest average loss)
            worst_local = np.argmax(surv_means)
            worst_global = surviving[worst_local]
            eliminated.append(model_names[worst_global])
            p_values[model_names[worst_global]] = p_val
            surviving.pop(worst_local)
        else:
            break

    # Surviving models get p-value = 1.0 (or the last p_val)
    for idx in surviving:
        p_values[model_names[idx]] = max(p_val if 'p_val' in dir() else 1.0, alpha)

    return MCSResult(
        surviving_models=[model_names[i] for i in surviving],
        eliminated_models=eliminated,
        p_values=p_values,
        alpha=alpha,
    )
