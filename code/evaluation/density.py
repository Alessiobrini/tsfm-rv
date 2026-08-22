"""
evaluation/density.py — Density forecast evaluation for the TSFM RV study.

Implements:
    - CRPS from a quantile grid (Gneiting & Raftery 2007)
    - CRPS from samples (closed form, for cross-checks)
    - PIT values and KS uniformity test (Diebold-Gunther-Tay 1998)
    - Interval coverage and mean width at nominal levels
    - density_summary: one-shot per-asset metrics

Design choices (set by Brini, May 2026 reply):
    - Single common quantile grid across all models. Sample-based forecasts
      are reduced to the same grid via empirical quantiles upstream so that
      CRPS rankings reflect calibration, not the number of quantiles emitted.
    - Log-RV is the primary scale for CRPS (level-RV is dominated by COVID
      spikes; log is balanced). PIT and coverage are invariant under
      monotone transforms applied consistently to forecast and target.

References:
    Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
        Prediction, and Estimation. JASA 102(477), 359-378.
    Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic
        forecasts, calibration and sharpness. JRSS-B 69(2), 243-268.
    Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). Evaluating Density
        Forecasts. International Economic Review 39, 863-883.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Small positive floor consistent with the QLIKE floor used by
# evaluation/loss_functions.py and the runners.
RV_FLOOR = 1e-12

ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]

# Common quantile grid for the density phase.
# Includes the symmetric pairs needed for 50%, 80%, and 95% intervals,
# plus enough resolution between for a usable PIT histogram.
DEFAULT_QUANTILE_LEVELS: np.ndarray = np.array(
    [0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975],
    dtype=float,
)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_quantile_inputs(
    actuals: np.ndarray,
    q_grid: np.ndarray,
    levels: np.ndarray,
) -> None:
    if q_grid.ndim != 2:
        raise ValueError(f"q_grid must be 2D (T, K), got shape {q_grid.shape}")
    T, K = q_grid.shape
    if levels.ndim != 1 or len(levels) != K:
        raise ValueError(
            f"levels must be 1D with length K={K}, got shape {levels.shape}"
        )
    if actuals.shape != (T,):
        raise ValueError(f"actuals must be 1D with length T={T}, got {actuals.shape}")
    if not np.all(np.diff(levels) > 0):
        raise ValueError("levels must be strictly increasing")
    if np.any(levels <= 0) or np.any(levels >= 1):
        raise ValueError("levels must lie in the open interval (0, 1)")


def _enforce_monotone_quantiles(q_grid: np.ndarray) -> np.ndarray:
    """Per-row isotonic (cumulative-max) projection to remove crossings."""
    return np.maximum.accumulate(q_grid, axis=1)


# ----------------------------------------------------------------------
# CRPS
# ----------------------------------------------------------------------

def crps_from_quantiles(
    actuals: ArrayLike,
    q_grid: ArrayLike,
    levels: ArrayLike = DEFAULT_QUANTILE_LEVELS,
    enforce_monotone: bool = True,
) -> np.ndarray:
    """Per-observation CRPS from a quantile-grid predictive distribution.

    Uses the weighted-pinball trapezoidal approximation

        CRPS(F, y) = 2 * integral_{0}^{1} PL_tau(F^{-1}(tau), y) d tau

    where PL_tau(q, y) = (1{y <= q} - tau) * (q - y). With K quantile
    levels tau_1 < ... < tau_K and matching predicted quantiles
    q_1 < ... < q_K, the integral is approximated by the trapezoidal rule
    over the K-1 intervals.

    Parameters
    ----------
    actuals : (T,) array
        Realized target values (e.g., RV or log RV).
    q_grid : (T, K) array
        Predicted quantiles, row t corresponds to observation t.
    levels : (K,) array
        Quantile levels in (0, 1), strictly increasing. Defaults to
        DEFAULT_QUANTILE_LEVELS.
    enforce_monotone : bool
        Apply cumulative-max to remove quantile crossings before scoring.

    Returns
    -------
    np.ndarray of shape (T,)
        Per-observation CRPS values. Lower is better.
    """
    actuals = np.asarray(actuals, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    _validate_quantile_inputs(actuals, q_grid, levels)
    if enforce_monotone:
        q_grid = _enforce_monotone_quantiles(q_grid)

    # Pinball loss matrix PL[t, k] for every (t, k).
    diff = q_grid - actuals[:, None]                       # (T, K)
    indicator = (actuals[:, None] <= q_grid).astype(float)  # (T, K)
    pinball = (indicator - levels[None, :]) * diff          # (T, K)

    # Trapezoidal integration in tau across axis=1, then double.
    integral = np.trapz(pinball, x=levels, axis=1)          # (T,)
    return 2.0 * integral


def crps_from_samples(
    actuals: ArrayLike,
    samples: ArrayLike,
) -> np.ndarray:
    """Per-observation CRPS from Monte Carlo samples.

    Closed-form unbiased estimator (Gneiting & Raftery 2007, eq. 20):

        CRPS = (1/m) * sum_i |x_i - y|
               - (1/(2 m^2)) * sum_{i, j} |x_i - x_j|

    Use this for cross-checking the quantile estimator, or directly when
    you have far more samples than the common quantile grid resolves.

    Parameters
    ----------
    actuals : (T,) array
    samples : (T, M) array
        M samples per observation.

    Returns
    -------
    np.ndarray of shape (T,)
    """
    actuals = np.asarray(actuals, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (T, M), got {samples.shape}")
    T, M = samples.shape
    if actuals.shape != (T,):
        raise ValueError(f"actuals must have length T={T}, got {actuals.shape}")

    abs_err = np.mean(np.abs(samples - actuals[:, None]), axis=1)
    # Sort along axis=1 to compute the spread term efficiently:
    # E|X - X'| = (2/m^2) * sum_i (2i - m - 1) * x_(i)
    s = np.sort(samples, axis=1)
    idx = np.arange(1, M + 1)
    spread = (2.0 / (M ** 2)) * np.sum((2 * idx - M - 1) * s, axis=1)
    return abs_err - 0.5 * spread


# ----------------------------------------------------------------------
# PIT and uniformity test
# ----------------------------------------------------------------------

def pit_values(
    actuals: ArrayLike,
    q_grid: ArrayLike,
    levels: ArrayLike = DEFAULT_QUANTILE_LEVELS,
    enforce_monotone: bool = True,
    extrapolate: str = "clip",
) -> np.ndarray:
    """Probability-integral-transform (PIT) values from a quantile grid.

    PIT_t = F_t(y_t), where F_t is the predictive CDF. We linearly
    interpolate the (quantile, level) pairs to estimate F_t.

    Parameters
    ----------
    actuals : (T,) array
    q_grid : (T, K) array
    levels : (K,) array
    enforce_monotone : bool
    extrapolate : {"clip", "linear"}
        How to handle y outside [q_min, q_max]:
        - "clip": return 0 if y < q_min, 1 if y > q_max
        - "linear": linearly extrapolate using the two endpoint quantiles

    Returns
    -------
    np.ndarray of shape (T,)
        PIT values in [0, 1] (or possibly outside if extrapolate="linear").
    """
    actuals = np.asarray(actuals, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    _validate_quantile_inputs(actuals, q_grid, levels)
    if enforce_monotone:
        q_grid = _enforce_monotone_quantiles(q_grid)

    T = len(actuals)
    pit = np.empty(T)
    for t in range(T):
        q = q_grid[t]
        y = actuals[t]
        if y <= q[0]:
            if extrapolate == "clip":
                pit[t] = 0.0
            else:
                slope = (levels[1] - levels[0]) / (q[1] - q[0] + 1e-30)
                pit[t] = max(0.0, levels[0] + slope * (y - q[0]))
        elif y >= q[-1]:
            if extrapolate == "clip":
                pit[t] = 1.0
            else:
                slope = (levels[-1] - levels[-2]) / (q[-1] - q[-2] + 1e-30)
                pit[t] = min(1.0, levels[-1] + slope * (y - q[-1]))
        else:
            pit[t] = np.interp(y, q, levels)
    return pit


def pit_ks_test(pit: ArrayLike) -> Dict[str, float]:
    """Kolmogorov-Smirnov test of PIT values against Uniform(0, 1).

    Returns the KS statistic and p-value. Low p-value rejects uniformity
    (i.e., the model is probabilistically miscalibrated).
    """
    pit = np.asarray(pit, dtype=float)
    statistic, pvalue = stats.kstest(pit, "uniform")
    return {"ks_stat": float(statistic), "ks_pvalue": float(pvalue), "n": int(len(pit))}


# ----------------------------------------------------------------------
# Coverage and sharpness
# ----------------------------------------------------------------------

def _symmetric_quantile_indices(
    levels: np.ndarray,
    nominal_level: float,
    atol: float = 1e-6,
) -> tuple[int, int]:
    """Return (i_low, i_high) for the symmetric (1-nominal)/2 quantile pair."""
    alpha = (1.0 - nominal_level) / 2.0
    i_low = int(np.argmin(np.abs(levels - alpha)))
    i_high = int(np.argmin(np.abs(levels - (1.0 - alpha))))
    if abs(levels[i_low] - alpha) > atol or abs(levels[i_high] - (1.0 - alpha)) > atol:
        raise ValueError(
            f"Quantile grid does not contain the {nominal_level:.0%} interval "
            f"endpoints ({alpha:.4f}, {1 - alpha:.4f}). Got nearest "
            f"({levels[i_low]:.4f}, {levels[i_high]:.4f})."
        )
    return i_low, i_high


def interval_coverage(
    actuals: ArrayLike,
    q_grid: ArrayLike,
    nominal_level: float,
    levels: ArrayLike = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Empirical coverage of the central (nominal_level * 100)% PI."""
    actuals = np.asarray(actuals, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    i_low, i_high = _symmetric_quantile_indices(levels, nominal_level)
    inside = (actuals >= q_grid[:, i_low]) & (actuals <= q_grid[:, i_high])
    return float(np.mean(inside))


def interval_width(
    q_grid: ArrayLike,
    nominal_level: float,
    levels: ArrayLike = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Mean width of the central (nominal_level * 100)% PI."""
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    i_low, i_high = _symmetric_quantile_indices(levels, nominal_level)
    return float(np.mean(q_grid[:, i_high] - q_grid[:, i_low]))


def tail_exceedance(
    actuals: ArrayLike,
    q_grid: ArrayLike,
    nominal_level: float,
    levels: ArrayLike = DEFAULT_QUANTILE_LEVELS,
) -> Tuple[float, float]:
    """Empirical left/right breach rates for the central (nominal_level)% PI.

    For miscoverage alpha = 1 - nominal_level, the expected breach rate
    in each tail is alpha / 2. Central coverage hides asymmetric
    miscalibration: a model whose 80% PI covers 80% on average could be
    20% below Q10 and 0% above Q90. Separating the two rates pinpoints
    which tail is broken (Q4 in the design log).

    Returns
    -------
    (left_rate, right_rate)
        left_rate  = P(actual < Q_{alpha/2})
        right_rate = P(actual > Q_{1 - alpha/2})
        left_rate + right_rate + central_coverage == 1 (modulo ties).
    """
    actuals = np.asarray(actuals, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    i_low, i_high = _symmetric_quantile_indices(levels, nominal_level)
    left = float(np.mean(actuals < q_grid[:, i_low]))
    right = float(np.mean(actuals > q_grid[:, i_high]))
    return left, right


# ----------------------------------------------------------------------
# Stronger PIT diagnostics (Q5 in the design log)
#
# KS against uniform is sensitive to median shift but partially blind to
# symmetric over/underconfidence and completely blind to autocorrelation
# in the PIT series. The functions below add:
#   - Berkowitz LR test against iid N(0, 1) on z_t = Phi^{-1}(PIT_t)
#   - Anderson-Darling against uniform (weights the tails)
#   - Ljung-Box on z_t for serial correlation
#   - Histogram shape label {uniform, U, hump, left-skewed, right-skewed}
# ----------------------------------------------------------------------

_PIT_CLIP_EPS = 1e-6


def _pit_to_z(pit: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.clip(pit, _PIT_CLIP_EPS, 1.0 - _PIT_CLIP_EPS))


def berkowitz_test(pit: ArrayLike) -> Dict[str, float]:
    """Berkowitz (2001) likelihood-ratio test of PIT calibration + independence.

    Transform z_t = Phi^{-1}(PIT_t). Under the null of a correctly
    specified predictive density, z_t ~ iid N(0, 1). The alternative is
    z_t = mu + rho * z_{t-1} + eps_t, eps_t ~ N(0, sigma^2), with three
    free parameters. LR statistic ~ chi^2(3) under the null.

    Standard in finance density evaluation: jointly catches mean bias,
    variance mis-scaling, and lag-1 serial correlation that KS misses.
    """
    pit = np.asarray(pit, dtype=float)
    if pit.size < 5:
        return {"berkowitz_lr": float("nan"), "berkowitz_pvalue": float("nan")}
    z = _pit_to_z(pit)
    T = len(z)

    # Restricted (null): z_t ~ iid N(0, 1), conditioned on z_1 to match
    # the unrestricted likelihood's conditioning.
    ll_r = -0.5 * (T - 1) * np.log(2 * np.pi) - 0.5 * float(np.sum(z[1:] ** 2))

    # Unrestricted: AR(1) Normal MLE conditioned on z_1.
    y = z[1:]
    x = z[:-1]
    x_design = np.column_stack([np.ones_like(x), x])
    try:
        beta_hat, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"berkowitz_lr": float("nan"), "berkowitz_pvalue": float("nan")}
    resid = y - x_design @ beta_hat
    sigma2_hat = float(np.mean(resid ** 2))
    if sigma2_hat <= 0:
        return {"berkowitz_lr": float("nan"), "berkowitz_pvalue": float("nan")}

    ll_u = (
        -0.5 * (T - 1) * np.log(2 * np.pi * sigma2_hat)
        - 0.5 * float(np.sum(resid ** 2)) / sigma2_hat
    )
    lr = float(2.0 * (ll_u - ll_r))
    pvalue = float(stats.chi2.sf(max(lr, 0.0), df=3))
    return {"berkowitz_lr": lr, "berkowitz_pvalue": pvalue}


def anderson_darling_uniform(pit: ArrayLike) -> Dict[str, float]:
    """Anderson-Darling against Uniform(0, 1), tail-weighted.

        A^2 = -n - (1/n) * sum_{i=1}^n (2i - 1) * [ln(u_(i)) + ln(1 - u_(n+1-i))]

    Reject uniformity when A^2 exceeds the critical value (5%: ~2.492 for
    fully specified F; we use the standard fixed critical-value approx).
    """
    pit = np.asarray(pit, dtype=float)
    if pit.size < 2:
        return {"ad_stat": float("nan"), "ad_pvalue": float("nan")}
    u = np.sort(np.clip(pit, _PIT_CLIP_EPS, 1.0 - _PIT_CLIP_EPS))
    n = len(u)
    i = np.arange(1, n + 1)
    a2 = -n - (1.0 / n) * float(np.sum((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))))
    # Marsaglia & Marsaglia (2004) approximation for AD against fully
    # specified F. Polynomial extrapolation diverges for a2 >> 10, so we
    # clip the inner expression to keep p <= 1 and pin p == 0 for extreme
    # values where the approximation is not valid.
    if a2 < 0.2:
        p = 1.0 - np.exp(-13.436 + 101.14 * a2 - 223.73 * a2 ** 2)
    elif a2 < 0.34:
        p = 1.0 - np.exp(-8.318 + 42.796 * a2 - 59.938 * a2 ** 2)
    elif a2 < 0.6:
        p = np.exp(0.9177 - 4.279 * a2 - 1.38 * a2 ** 2)
    elif a2 < 10.0:
        p = np.exp(min(1.2937 - 5.709 * a2 + 0.0186 * a2 ** 2, 0.0))
    else:
        p = 0.0
    return {"ad_stat": float(a2), "ad_pvalue": float(np.clip(p, 0.0, 1.0))}


def ljung_box_pit(pit: ArrayLike, lags: int = 10) -> Dict[str, float]:
    """Ljung-Box test on z_t = Phi^{-1}(PIT_t) for serial correlation.

    Significant Q indicates the conditional density is missing
    time-varying structure (a vol-clustering tell that KS would miss).
    Returns the lag-`lags` portmanteau statistic and p-value.
    """
    pit = np.asarray(pit, dtype=float)
    if pit.size < lags + 2:
        return {"lb_stat": float("nan"), "lb_pvalue": float("nan")}
    z = _pit_to_z(pit)
    n = len(z)
    z_centered = z - z.mean()
    denom = float(np.sum(z_centered ** 2))
    if denom <= 0:
        return {"lb_stat": float("nan"), "lb_pvalue": float("nan")}

    q = 0.0
    for k in range(1, lags + 1):
        num = float(np.sum(z_centered[:-k] * z_centered[k:]))
        rho_k = num / denom
        q += rho_k ** 2 / (n - k)
    q *= n * (n + 2)
    return {
        "lb_stat": float(q),
        "lb_pvalue": float(stats.chi2.sf(q, df=lags)),
    }


def pit_histogram_shape(
    pit: ArrayLike,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Classify PIT histogram shape relative to uniform.

    Returns a dict with the bin counts (as a JSON-serialisable list) and
    a string `shape` label in {uniform, U, hump, left-skewed,
    right-skewed} derived from the first two moments of the PIT
    distribution and the U-statistic (excess tail mass over centre).
    """
    pit = np.asarray(np.clip(pit, 0.0, 1.0), dtype=float)
    n = len(pit)
    counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    expected = n / n_bins

    # Center moments of PIT around 0.5; under uniform mean=0.5, var=1/12.
    mean = float(np.mean(pit))
    var = float(np.var(pit))
    skew = float(stats.skew(pit)) if n > 2 else 0.0

    # U-statistic: excess mass in the two outer bins vs the two centre bins.
    outer = int(counts[0] + counts[-1])
    centre = int(counts[n_bins // 2 - 1] + counts[n_bins // 2])
    u_stat = (outer - centre) / max(expected, 1e-9)  # scaled deviation

    if abs(skew) > 0.25:
        shape = "right-skewed" if skew > 0 else "left-skewed"
    elif u_stat > 0.5:
        shape = "U"
    elif u_stat < -0.5:
        shape = "hump"
    elif abs(mean - 0.5) < 0.03 and abs(var - 1.0 / 12.0) < 0.01:
        shape = "uniform"
    else:
        shape = "uniform-ish"

    return {
        "shape": shape,
        "u_stat": float(u_stat),
        "skewness": float(skew),
        "bin_counts": [int(c) for c in counts],
    }


# ----------------------------------------------------------------------
# Affine recalibration applied at the distribution level (Phase 3)
# ----------------------------------------------------------------------

def apply_affine_to_quantiles(
    q_grid: ArrayLike,
    alpha: Union[float, np.ndarray],
    beta: Union[float, np.ndarray],
    floor: float = RV_FLOOR,
    on_nonpositive_beta: str = "warn",
) -> np.ndarray:
    """Apply the median's MZ (alpha, beta) uniformly to every quantile.

    The (alpha, beta) inputs are the coefficients from the recursive MZ
    regression of RV on the model's median forecast. We apply that single
    pair to every quantile of the predictive distribution.

    This is the headline Phase-3 test: does the affine correction that
    fixes the median's bias also fix the rest of the predictive
    distribution, or does it leave interval coverage broken?

    Guards (Q3 in the design log):
      * If beta_t <= 0 the linear transform reverses quantile order, so we
        emit a warning (or raise) per ``on_nonpositive_beta``.
      * After the affine transform we apply per-row cumulative-max to
        enforce monotonicity even when the transform compressed adjacent
        quantiles past each other in finite precision.
      * We clip below ``floor`` so the corrected quantiles stay strictly
        positive (RV is positive; consistent with the QLIKE floor).

    Parameters
    ----------
    q_grid : (T, K) array
    alpha, beta : scalar, or (T,) array for recursive correction
    floor : float
        Lower clip applied after the affine transform. Default RV_FLOOR.
    on_nonpositive_beta : {"warn", "raise", "ignore"}
        What to do when any beta is <= 0.

    Returns
    -------
    np.ndarray of shape (T, K)
        Corrected, monotone, positive quantiles.
    """
    q_grid = np.asarray(q_grid, dtype=float)
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta_arr = np.atleast_1d(np.asarray(beta, dtype=float))

    bad_beta_mask = beta_arr <= 0
    if bad_beta_mask.any():
        bad_count = int(bad_beta_mask.sum())
        msg = (
            f"apply_affine_to_quantiles: encountered beta <= 0 for "
            f"{bad_count} of {len(beta_arr)} block(s); the affine transform "
            "reverses quantile order in those rows. The cumulative-max guard "
            "will mask the reversal but the resulting density is meaningless. "
            "Inspect the upstream recursive_mz_correction."
        )
        if on_nonpositive_beta == "raise":
            raise ValueError(msg)
        if on_nonpositive_beta == "warn":
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    alpha_b = float(alpha_arr[0]) if alpha_arr.shape == (1,) else alpha_arr[:, None]
    beta_b = float(beta_arr[0]) if beta_arr.shape == (1,) else beta_arr[:, None]
    corrected = alpha_b + beta_b * q_grid
    corrected = np.maximum.accumulate(corrected, axis=1)
    return np.clip(corrected, floor, None)


# ----------------------------------------------------------------------
# Distributional recalibration (Phase 4)
#
# Two complementary tools that target coverage directly, beyond what the
# affine MZ correction can do:
#   - isotonic_quantile_recalibration: Kuleshov, Fenner & Ermon (2018).
#     Learn a monotone map on PIT values so that calibration PIT becomes
#     uniform; apply the inverse map to relabel the quantile levels.
#   - split_conformal_quantile_recalibration: Romano, Patterson & Candes
#     (2019), CQR. Calibrate an additive offset so that the central PI
#     achieves nominal coverage on the calibration block.
#
# Per Brini's reply: exchangeability is violated by volatility clustering
# and regime drift, so split-conformal is fit *per refit block* and the
# coverage guarantee is treated as an empirical question, not assumed.
# ----------------------------------------------------------------------

def fit_isotonic_recalibration(
    cal_pit: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Kuleshov-style isotonic recalibration on calibration PIT values.

    Returns the pair (sorted_calibration_pit, empirical_cdf_at_those_points)
    that defines the empirical CDF H_hat used as the recalibration map.
    Apply via :func:`apply_isotonic_recalibration_to_quantiles`.
    """
    cal_pit = np.asarray(cal_pit, dtype=float)
    if cal_pit.size == 0:
        raise ValueError("calibration PIT is empty")
    u_sorted = np.sort(np.clip(cal_pit, 0.0, 1.0))
    h_vals = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
    return u_sorted, h_vals


def apply_isotonic_recalibration_to_quantiles(
    q_grid: ArrayLike,
    levels: ArrayLike,
    cal_pit_sorted: np.ndarray,
    cal_h_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel a quantile grid under an isotonic recalibration map.

    The recalibrated CDF is F'_t(y) = H_hat(F_t(y)). Equivalently, the
    recalibrated tau-quantile of F'_t equals the H_hat^{-1}(tau)-quantile
    of F_t. We invert H_hat at each requested level and resample the
    quantile grid by linear interpolation.

    Parameters
    ----------
    q_grid : (T, K) array
        Original quantile grid.
    levels : (K,) array
        Original quantile levels.
    cal_pit_sorted, cal_h_values
        Outputs of :func:`fit_isotonic_recalibration`.

    Returns
    -------
    (q_grid_new, levels) : the relabelled quantile grid at the same
        nominal levels (so downstream coverage/sharpness use the
        unchanged target levels).
    """
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)

    # H_hat^{-1}(tau): the tau-th quantile of the calibration PIT.
    adjusted_levels = np.interp(levels, cal_h_values, cal_pit_sorted)
    adjusted_levels = np.clip(adjusted_levels, 1e-6, 1 - 1e-6)

    T, _ = q_grid.shape
    q_new = np.empty((T, len(levels)))
    for t in range(T):
        q_new[t] = np.interp(adjusted_levels, levels, q_grid[t])
    return q_new, levels


def split_conformal_offset(
    cal_actuals: ArrayLike,
    cal_lower: ArrayLike,
    cal_upper: ArrayLike,
    nominal_level: float,
) -> float:
    """Split-conformal CQR offset (Romano, Patterson & Candes 2019).

    For miscoverage alpha = 1 - nominal_level, the conformity score is
    E_i = max(cal_lower_i - cal_actual_i, cal_actual_i - cal_upper_i).
    The CQR offset is the ceil((1-alpha)(n+1))/n empirical quantile of
    {E_i}. Add this offset to the upper bound and subtract from the
    lower bound to obtain a recalibrated PI.
    """
    cal_actuals = np.asarray(cal_actuals, dtype=float)
    cal_lower = np.asarray(cal_lower, dtype=float)
    cal_upper = np.asarray(cal_upper, dtype=float)
    n = len(cal_actuals)
    if n == 0:
        raise ValueError("calibration set is empty")
    if not (0.0 < nominal_level < 1.0):
        raise ValueError("nominal_level must lie in (0, 1)")

    scores = np.maximum(cal_lower - cal_actuals, cal_actuals - cal_upper)
    # Finite-sample-adjusted quantile per the CQR paper.
    k = int(np.ceil((nominal_level) * (n + 1)))
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


def apply_conformal_to_quantiles(
    q_grid: ArrayLike,
    levels: ArrayLike,
    cal_actuals: ArrayLike,
    cal_q_grid: ArrayLike,
    nominal_levels: Sequence[float] = (0.50, 0.80, 0.95),
) -> Dict[float, np.ndarray]:
    """Per-nominal-level conformalised PI offsets applied to test quantiles.

    Returns a dict mapping each nominal level to a (T, 2) array of
    [lower, upper] conformalised PI endpoints on the test set.

    This is per-level CQR, not a full recalibration of every quantile.
    For full distributional recalibration use the isotonic map above; use
    this when the headline question is whether the 50/80/95 PIs achieve
    nominal coverage out of sample.
    """
    levels = np.asarray(levels, dtype=float)
    cal_actuals = np.asarray(cal_actuals, dtype=float)
    cal_q_grid = np.asarray(cal_q_grid, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)

    out: Dict[float, np.ndarray] = {}
    for nl in nominal_levels:
        i_low, i_high = _symmetric_quantile_indices(levels, nl)
        offset = split_conformal_offset(
            cal_actuals=cal_actuals,
            cal_lower=cal_q_grid[:, i_low],
            cal_upper=cal_q_grid[:, i_high],
            nominal_level=nl,
        )
        lower = q_grid[:, i_low] - offset
        upper = q_grid[:, i_high] + offset
        out[nl] = np.column_stack([lower, upper])
    return out


# ----------------------------------------------------------------------
# Summary container and one-shot evaluator
# ----------------------------------------------------------------------

@dataclass
class DensityScores:
    """Per-asset density-evaluation summary."""

    n_obs: int
    crps_mean: float
    crps_median: float
    pit_ks_stat: float
    pit_ks_pvalue: float
    coverage: Dict[float, float]            # nominal -> empirical central coverage
    mean_width: Dict[float, float]          # nominal -> mean PI width
    tail_left: Dict[float, float] = field(default_factory=dict)   # nominal -> P(act < Q_low)
    tail_right: Dict[float, float] = field(default_factory=dict)  # nominal -> P(act > Q_high)
    pit_berkowitz_lr: float = float("nan")
    pit_berkowitz_pvalue: float = float("nan")
    pit_ad_stat: float = float("nan")
    pit_ad_pvalue: float = float("nan")
    pit_lb_stat: float = float("nan")
    pit_lb_pvalue: float = float("nan")
    pit_shape: str = ""
    pit_u_stat: float = float("nan")
    pit_skewness: float = float("nan")
    pit_bin_counts: Sequence[int] = field(default_factory=list)
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "n_obs": self.n_obs,
            "crps_mean": self.crps_mean,
            "crps_median": self.crps_median,
            "pit_ks_stat": self.pit_ks_stat,
            "pit_ks_pvalue": self.pit_ks_pvalue,
            "pit_berkowitz_lr": self.pit_berkowitz_lr,
            "pit_berkowitz_pvalue": self.pit_berkowitz_pvalue,
            "pit_ad_stat": self.pit_ad_stat,
            "pit_ad_pvalue": self.pit_ad_pvalue,
            "pit_lb_stat": self.pit_lb_stat,
            "pit_lb_pvalue": self.pit_lb_pvalue,
            "pit_shape": self.pit_shape,
            "pit_u_stat": self.pit_u_stat,
            "pit_skewness": self.pit_skewness,
            "pit_bin_counts": ",".join(str(c) for c in self.pit_bin_counts),
        }
        for nom, cov in self.coverage.items():
            tag = int(round(nom * 100))
            out[f"coverage_{tag}"] = cov
            if nom in self.tail_left:
                out[f"tail_left_{tag}"] = self.tail_left[nom]
            if nom in self.tail_right:
                out[f"tail_right_{tag}"] = self.tail_right[nom]
        for nom, w in self.mean_width.items():
            out[f"width_{int(round(nom * 100))}"] = w
        out.update(self.extras)
        return out


def density_summary(
    actuals: ArrayLike,
    q_grid: ArrayLike,
    levels: ArrayLike = DEFAULT_QUANTILE_LEVELS,
    nominal_levels: Sequence[float] = (0.50, 0.80, 0.95),
    enforce_monotone: bool = True,
    pit_lags: int = 10,
    histogram_bins: int = 10,
) -> DensityScores:
    """All density metrics in one call (Q3-Q5 diagnostics included)."""
    actuals = np.asarray(actuals, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    _validate_quantile_inputs(actuals, q_grid, levels)
    if enforce_monotone:
        q_grid = _enforce_monotone_quantiles(q_grid)

    crps = crps_from_quantiles(actuals, q_grid, levels, enforce_monotone=False)
    pit = pit_values(actuals, q_grid, levels, enforce_monotone=False)
    ks = pit_ks_test(pit)
    berk = berkowitz_test(pit)
    ad = anderson_darling_uniform(pit)
    lb = ljung_box_pit(pit, lags=pit_lags)
    shape = pit_histogram_shape(pit, n_bins=histogram_bins)

    coverage: Dict[float, float] = {}
    tail_left: Dict[float, float] = {}
    tail_right: Dict[float, float] = {}
    widths: Dict[float, float] = {}
    for nl in nominal_levels:
        coverage[nl] = interval_coverage(actuals, q_grid, nl, levels)
        widths[nl] = interval_width(q_grid, nl, levels)
        left, right = tail_exceedance(actuals, q_grid, nl, levels)
        tail_left[nl] = left
        tail_right[nl] = right

    return DensityScores(
        n_obs=int(len(actuals)),
        crps_mean=float(np.mean(crps)),
        crps_median=float(np.median(crps)),
        pit_ks_stat=ks["ks_stat"],
        pit_ks_pvalue=ks["ks_pvalue"],
        coverage=coverage,
        mean_width=widths,
        tail_left=tail_left,
        tail_right=tail_right,
        pit_berkowitz_lr=berk["berkowitz_lr"],
        pit_berkowitz_pvalue=berk["berkowitz_pvalue"],
        pit_ad_stat=ad["ad_stat"],
        pit_ad_pvalue=ad["ad_pvalue"],
        pit_lb_stat=lb["lb_stat"],
        pit_lb_pvalue=lb["lb_pvalue"],
        pit_shape=shape["shape"],
        pit_u_stat=shape["u_stat"],
        pit_skewness=shape["skewness"],
        pit_bin_counts=shape["bin_counts"],
    )


# ----------------------------------------------------------------------
# Self-test: a well-calibrated Gaussian should produce uniform PIT, and
# its quantile-CRPS should agree with its sample-CRPS within MC noise.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 2000
    M = 1000
    mu = rng.normal(size=T)
    sigma = 1.0

    # True predictive distribution: N(mu_t, 1). Draw the realization from it.
    y = rng.normal(loc=mu, scale=sigma)

    # Build the common quantile grid from the true CDF.
    levels = DEFAULT_QUANTILE_LEVELS
    z = stats.norm.ppf(levels)
    q_grid = mu[:, None] + sigma * z[None, :]

    # Also draw samples for the closed-form CRPS estimator.
    samples = rng.normal(loc=mu[:, None], scale=sigma, size=(T, M))

    summary = density_summary(y, q_grid, levels)
    crps_samp = float(np.mean(crps_from_samples(y, samples)))

    print("Self-test: well-calibrated N(mu_t, 1) predictive distribution")
    print(f"  CRPS (quantile grid):  {summary.crps_mean:.4f}")
    print(f"  CRPS (samples, M={M}): {crps_samp:.4f}")
    print(f"  Theoretical (sigma/sqrt(pi)): {sigma / np.sqrt(np.pi):.4f}")
    print(f"  PIT KS stat:    {summary.pit_ks_stat:.4f}  (p={summary.pit_ks_pvalue:.3f})")
    print(f"  Coverage:  50={summary.coverage[0.50]:.3f}  "
          f"80={summary.coverage[0.80]:.3f}  95={summary.coverage[0.95]:.3f}")
    print(f"  Widths:    50={summary.mean_width[0.50]:.3f}  "
          f"80={summary.mean_width[0.80]:.3f}  95={summary.mean_width[0.95]:.3f}")
    print(f"  Theoretical widths:  50={2 * stats.norm.ppf(0.75):.3f}  "
          f"80={2 * stats.norm.ppf(0.90):.3f}  95={2 * stats.norm.ppf(0.975):.3f}")

    # -- Recalibration self-test ----------------------------------------
    # Build a deliberately overconfident forecast: report sigma_hat = 0.5
    # when the truth is sigma = 1. PIT should be U-shaped, coverage too low.
    print("\nRecalibration self-test: overconfident predictive sigma_hat = 0.5")
    sigma_hat = 0.5
    q_grid_bad = mu[:, None] + sigma_hat * z[None, :]
    bad = density_summary(y, q_grid_bad, levels)
    print(f"  Raw: CRPS={bad.crps_mean:.4f}  KS={bad.pit_ks_stat:.3f}  "
          f"cov80={bad.coverage[0.80]:.3f}  width80={bad.mean_width[0.80]:.3f}")

    # Split: first half = calibration, second half = test.
    half = T // 2
    cal_pit = pit_values(y[:half], q_grid_bad[:half], levels)
    fit = fit_isotonic_recalibration(cal_pit)
    q_grid_iso, _ = apply_isotonic_recalibration_to_quantiles(
        q_grid_bad[half:], levels, *fit
    )
    iso = density_summary(y[half:], q_grid_iso, levels)
    print(f"  Isotonic recalib (test): CRPS={iso.crps_mean:.4f}  "
          f"KS={iso.pit_ks_stat:.3f}  cov80={iso.coverage[0.80]:.3f}  "
          f"width80={iso.mean_width[0.80]:.3f}")

    cqr = apply_conformal_to_quantiles(
        q_grid_bad[half:], levels, y[:half], q_grid_bad[:half],
        nominal_levels=(0.80,),
    )
    lower, upper = cqr[0.80][:, 0], cqr[0.80][:, 1]
    cqr_cov = float(np.mean((y[half:] >= lower) & (y[half:] <= upper)))
    cqr_width = float(np.mean(upper - lower))
    print(f"  Split-conformal (test):  cov80={cqr_cov:.3f}  width80={cqr_width:.3f}")

    # -- Q4 / Q5 diagnostics on the overconfident forecast --------------
    print("\nQ4 / Q5 diagnostics on overconfident forecast:")
    print(f"  PIT shape: {bad.pit_shape}  (U={bad.pit_u_stat:+.2f}, skew={bad.pit_skewness:+.2f})")
    print(f"  Berkowitz LR={bad.pit_berkowitz_lr:.1f}  p={bad.pit_berkowitz_pvalue:.3g}")
    print(f"  Anderson-Darling A^2={bad.pit_ad_stat:.2f}  p={bad.pit_ad_pvalue:.3g}")
    print(f"  Ljung-Box Q(10)={bad.pit_lb_stat:.1f}  p={bad.pit_lb_pvalue:.3g}")
    for nl in (0.50, 0.80, 0.95):
        print(
            f"  PI {int(nl * 100):>2}%: cov={bad.coverage[nl]:.3f}  "
            f"left breach={bad.tail_left[nl]:.3f}  right breach={bad.tail_right[nl]:.3f}  "
            f"(nominal each={ (1 - nl) / 2:.3f})"
        )

    # -- Q3 MZ guard test ------------------------------------------------
    print("\nQ3: MZ guard on apply_affine_to_quantiles")
    q_for_mz = np.exp(mu[:, None] + sigma * z[None, :])  # positive log-normal grid
    out_ok = apply_affine_to_quantiles(q_for_mz, alpha=0.0, beta=1.0)
    print(f"  beta=1.0 identity:  monotone={bool((np.diff(out_ok, axis=1) >= 0).all())}  "
          f"positive={bool((out_ok > 0).all())}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_bad = apply_affine_to_quantiles(q_for_mz, alpha=0.0, beta=-1.0)
        warned = any(issubclass(w.category, RuntimeWarning) for w in caught)
    print(f"  beta=-1.0 fired warning={warned}  "
          f"monotone after guard={bool((np.diff(out_bad, axis=1) >= 0).all())}  "
          f"positive={bool((out_bad > 0).all())}")
