"""
evaluation/mcs.py — Model Confidence Set (Hansen, Lunde & Nason 2011).

The MCS procedure identifies the set of models that contains the best model
with a given confidence level. It tests equal predictive ability on the models
still in the set and, while the test rejects, eliminates one model.

Reference: Hansen, Lunde & Nason (2011, Econometrica), Sec. 3.1.2.

Test statistic: the range statistic T_R = max_{i,j} |t_ij|, with
t_ij = dbar_ij / sqrt(var(dbar_ij)) and var(dbar_ij) from a moving-block
bootstrap. Elimination rule: e_R = argmax_i max_j t_ij, the model whose
standardized loss difference against some other model in the set equals T_R
(the model with the higher mean loss in the pair that attains T_R).
MCS p-values follow their Definition 4: the p-value of an eliminated model is
the largest test p-value up to its elimination, and the models left in the set
get the p-value of the last test, or 1 if a single model is left.
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
    p_run = 0.0  # largest test p-value so far

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
        T_R = np.max(np.abs(t_pairs))

        # Bootstrap distribution of T_R
        t_boot = (boot_diff_pairs - d_bar_pairs[None, :]) / sd_pairs[None, :]
        T_R_boot = np.max(np.abs(t_boot), axis=1)

        # p-value
        p_val = np.mean(T_R_boot >= T_R)

        if p_val < alpha:
            # e_R: in the pair that attains T_R, the model with the higher mean loss
            k = np.argmax(np.abs(t_pairs))
            worst_local = ii[k] if t_pairs[k] > 0 else jj[k]
            worst_global = surviving[worst_local]
            p_run = max(p_run, p_val)
            eliminated.append(model_names[worst_global])
            p_values[model_names[worst_global]] = p_run
            surviving.pop(worst_local)
        else:
            break

    # Models left in the set: the p-value of the test that kept them (Definition 4),
    # or 1 when one model is left
    last = 1.0 if len(surviving) == 1 else max(p_run, p_val)
    for idx in surviving:
        p_values[model_names[idx]] = last

    return MCSResult(
        surviving_models=[model_names[i] for i in surviving],
        eliminated_models=eliminated,
        p_values=p_values,
        alpha=alpha,
    )
