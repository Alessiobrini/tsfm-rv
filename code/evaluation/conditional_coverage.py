"""
evaluation/conditional_coverage.py — Conditional coverage of interval forecasts.

Marginal coverage can be right while the misses cluster in time or concentrate
in high-volatility states; split-conformal CQR delivers marginal coverage by
construction, so these are the tests that matter for it.

    christoffersen_test   unconditional coverage, independence and joint LR
                          tests on the miss sequence (Christoffersen 1998)
    dq_test               regression of misses on a lagged miss and the
                          volatility state, joint Wald test with a Monte
                          Carlo p-value (Engle and Manganelli 2004 dynamic
                          quantile test; Dufour 2006)
    coverage_by_regime    coverage within calm / normal / turbulent states

At h > 1, h-step-ahead misses are serially correlated even under a correct
forecast, because consecutive forecasts share h-1 days of unobserved shocks;
pass ``subsample=h`` to test every h-th observation.

References
----------
Christoffersen, P. F. (1998). Evaluating Interval Forecasts. International
    Economic Review 39(4), 841-862.
Dufour, J.-M. (2006). Monte Carlo tests with nuisance parameters. Journal of
    Econometrics 133(2), 443-477.
Engle, R. F. & Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value
    at Risk by Regression Quantiles. JBES 22(4), 367-381.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

REGIMES = ("calm", "normal", "turbulent")


def _bernoulli_loglik(n0: float, n1: float, p: float) -> float:
    return float(xlogy(n0, 1.0 - p) + xlogy(n1, p))


def christoffersen_test(misses: np.ndarray, nominal: float, subsample: int = 1) -> Dict[str, float]:
    """Christoffersen (1998) LR tests on a 0/1 miss sequence.

    ``nominal`` is the interval's nominal coverage, so the expected miss rate
    is 1 - nominal. Returns LR statistics and p-values for unconditional
    coverage (chi2(1)), independence against a first-order Markov chain
    (chi2(1)) and the joint test (chi2(2)).
    """
    hit = np.asarray(misses, dtype=float)[::subsample]
    alpha = 1.0 - nominal
    n1 = hit.sum()
    n0 = len(hit) - n1
    pi = n1 / len(hit)
    lr_uc = -2.0 * (_bernoulli_loglik(n0, n1, alpha) - _bernoulli_loglik(n0, n1, pi))

    prev, cur = hit[:-1], hit[1:]
    n00 = np.sum((prev == 0) & (cur == 0)); n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0)); n11 = np.sum((prev == 1) & (cur == 1))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi_t = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr_ind = -2.0 * (_bernoulli_loglik(n00 + n10, n01 + n11, pi_t)
                     - _bernoulli_loglik(n00, n01, pi01) - _bernoulli_loglik(n10, n11, pi11))
    lr_cc = lr_uc + lr_ind
    return {
        "n": int(len(hit)), "miss_rate": float(pi),
        "lr_uc": float(lr_uc), "p_uc": float(stats.chi2.sf(lr_uc, 1)),
        "lr_ind": float(lr_ind), "p_ind": float(stats.chi2.sf(lr_ind, 1)),
        "lr_cc": float(lr_cc), "p_cc": float(stats.chi2.sf(lr_cc, 2)),
    }


def _dq_stats(hits: np.ndarray, z: np.ndarray, alpha: float):
    """Wald statistic and state t-ratio for each row of ``hits`` (B, n).

    Regression of hit_t - alpha on [1, hit_{t-1}, z_t] with HC1 covariance,
    vectorized over the B rows. Pseudo-inverses handle rows whose lagged
    hit has no variation.
    """
    y = hits[:, 1:] - alpha                                          # (B, m)
    B, m = y.shape
    X = np.stack([np.ones((B, m)), hits[:, :-1], np.broadcast_to(z[1:], (B, m))], axis=2)
    xtx_inv = np.linalg.pinv(np.einsum("bmi,bmj->bij", X, X))
    beta = np.einsum("bij,bmj,bm->bi", xtx_inv, X, y)
    e = y - np.einsum("bmi,bi->bm", X, beta)
    meat = np.einsum("bmi,bm,bmj->bij", X, e ** 2, X)
    cov = xtx_inv @ meat @ xtx_inv * m / (m - 3)
    wald = np.einsum("bi,bij,bj->b", beta, np.linalg.pinv(cov), beta)
    t_state = beta[:, 2] / np.sqrt(np.maximum(cov[:, 2, 2], 1e-300))
    return wald, beta[:, 2], t_state


def dq_test(misses: np.ndarray, state: np.ndarray, nominal: float, subsample: int = 1,
            n_sim: int = 999, seed: int = 0) -> Dict[str, float]:
    """Regression of (miss - expected miss rate) on [1, lagged miss, state].

    ``state`` is standardized inside. Under correct conditional coverage all
    coefficients are zero. The p-value is a Monte Carlo p-value (Dufour 2006):
    the Wald statistic is recomputed on ``n_sim`` i.i.d. Bernoulli miss
    sequences at the nominal rate with the same state series, because the
    asymptotic chi2(3) rejects a correct interval about 31% of the time at
    n = 41 (h = 22 after subsampling). ``beta_state`` > 0 means misses become
    more likely as volatility rises.
    """
    hit = np.asarray(misses, dtype=float)[::subsample]
    z = np.asarray(state, dtype=float)[::subsample]
    if hit[1:].std() == 0:            # no variation in misses: nothing to regress
        return {"beta_state": np.nan, "t_state": np.nan, "dq_stat": np.nan, "p_dq": np.nan}
    z = (z - z.mean()) / z.std()
    alpha = 1.0 - nominal
    wald, b_state, t_state = (v[0] for v in _dq_stats(hit[None, :], z, alpha))
    sims = (np.random.default_rng(seed).uniform(size=(n_sim, len(hit))) < alpha).astype(float)
    wald_null, _, _ = _dq_stats(sims, z, alpha)
    p = (1 + np.sum(wald_null >= wald)) / (n_sim + 1)
    return {"beta_state": float(b_state), "t_state": float(t_state),
            "dq_stat": float(wald), "p_dq": float(p)}


def volatility_regimes(state: pd.Series) -> pd.Series:
    """Label each date calm / normal / turbulent by terciles of the state."""
    return pd.qcut(state, 3, labels=list(REGIMES))


def coverage_by_regime(covered: np.ndarray, regimes: Sequence[str]) -> Dict[str, float]:
    """Empirical coverage within each volatility regime."""
    s = pd.Series(np.asarray(covered, dtype=float)).groupby(np.asarray(regimes)).mean()
    return {f"cov_{r}": float(s.get(r, np.nan)) for r in REGIMES}
