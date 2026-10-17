"""
evaluation/combination.py — recursive forecast-combination weights.

Rows are consecutive forecast origins, and the target of row r is realized h - 1 rows after its
origin. The weight used at origin t is therefore estimated on the errors of rows 0 to t - h, the
rows whose targets are observed by origin t. With h = 1 these are all earlier rows.

Reference: Bates & Granger (1969), Timmermann (2006).
"""

import numpy as np

WARMUP = 100          # observed errors required before the estimated weight replaces equal weights


def bates_granger_weights(actual, f1, f2, horizon=1, warmup=WARMUP):
    """Recursive Bates--Granger weight on f1, from the errors observed by each origin.

    w_t = (s22 - s12) / (s11 + s22 - 2 s12), with the error second moments taken over rows
    0 to t - horizon (expanding window), clipped to [0, 1]. Equal weight while fewer than
    `warmup` errors are observed, and when the denominator is degenerate.
    """
    a = np.asarray(actual, float)
    e1, e2 = a - np.asarray(f1, float), a - np.asarray(f2, float)
    n = len(a)
    w = np.full(n, 0.5)
    for t in range(n):
        end = t - horizon + 1          # rows q <= t - horizon
        if end < warmup:
            continue
        x1, x2 = e1[:end], e2[:end]
        s11, s22, s12 = np.mean(x1 * x1), np.mean(x2 * x2), np.mean(x1 * x2)
        denom = s11 + s22 - 2.0 * s12
        if abs(denom) >= 1e-18:
            w[t] = min(1.0, max(0.0, (s22 - s12) / denom))
    return w


def bates_granger_recursive(actual, f1, f2, horizon=1, warmup=WARMUP):
    """Combined forecast w_t f1_t + (1 - w_t) f2_t with the recursive Bates--Granger weight."""
    w = bates_granger_weights(actual, f1, f2, horizon, warmup)
    return w * np.asarray(f1, float) + (1.0 - w) * np.asarray(f2, float)


def min_variance_recursive(actual, members, horizon=1, warmup=WARMUP):
    """Recursive minimum-variance combination of K forecasts (generalized Bates--Granger).

    At origin t the weights minimize the error variance under the K x K error covariance of
    rows 0 to t - horizon: w = Sigma^{-1} 1 / (1' Sigma^{-1} 1), negative weights clipped to 0
    and renormalized. Equal weights while fewer than `warmup` errors are observed, or when the
    covariance is singular.
    """
    a = np.asarray(actual, float)
    F = np.column_stack([np.asarray(f, float) for f in members])   # n x K
    n, K = F.shape
    E = a[:, None] - F
    ones = np.ones(K)
    comb = np.empty(n)
    for t in range(n):
        end = t - horizon + 1
        w = ones / K
        if end >= warmup:
            try:
                Sig = np.atleast_2d(np.cov(E[:end].T, bias=True))
                raw = np.linalg.solve(Sig, ones)
                cand = raw / raw.sum()
                if not np.all(np.isfinite(cand)):
                    raise np.linalg.LinAlgError
                cand = np.clip(cand, 0.0, None)
                s = cand.sum()
                w = cand / s if s > 0 else ones / K
            except np.linalg.LinAlgError:
                w = ones / K
        comb[t] = float(F[t] @ w)
    return comb
