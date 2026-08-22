"""
models/har_density.py — Density forecast wrapper around (Log-)HAR.

Provides an econometric density baseline for the TSFM density-evaluation
phase. The wrapper pairs a fitted (Log-)HAR point model with one of two
residual distributions:

    - "gaussian":  log RV_{t+h} | F_t ~ N(log_pred_t, sigma^2),
                   where sigma^2 is the OLS residual variance.
                   The level-space density is log-normal with mean
                   exp(log_pred_t + sigma^2 / 2), matching the
                   bias-corrected point forecast used in the paper.
    - "empirical": uses the empirical CDF of in-sample residuals as the
                   conditional distribution of log RV_{t+h} - log_pred_t.

Output is a (T, K) quantile grid on the common DEFAULT_QUANTILE_LEVELS
grid, in either log or level space, drop-in compatible with everything
in evaluation/density.py.

References
----------
Corsi, F. (2009). A simple approximate long-memory model of realized
    volatility. Journal of Financial Econometrics 7(2), 174-196.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.density import DEFAULT_QUANTILE_LEVELS
from models.har import HARModel


ResidualMode = Literal["gaussian", "empirical"]


@dataclass
class HARDensityForecast:
    """Container for HAR predictive distribution at a set of forecast dates."""

    dates: pd.DatetimeIndex
    point: pd.Series                 # bias-corrected level forecast
    quantile_levels: np.ndarray      # (K,)
    log_quantiles: np.ndarray        # (T, K), log space
    level_quantiles: np.ndarray      # (T, K), level space (always positive)
    sigma: float                     # residual scale (log space)
    mode: ResidualMode

    def to_frame(self, space: Literal["log", "level"] = "level") -> pd.DataFrame:
        """Quantile grid as a DataFrame indexed by date."""
        q = self.log_quantiles if space == "log" else self.level_quantiles
        cols = [f"q{int(round(t * 1000)) / 10:g}" for t in self.quantile_levels]
        return pd.DataFrame(q, index=self.dates, columns=cols)


class HARDensityModel:
    """Density wrapper around (Log-)HAR.

    Always fits on log RV (the only configuration in the paper where the
    density baseline is well posed). To match the paper's HAR point
    benchmark, set residual_mode="gaussian"; the level-space mean of the
    resulting log-normal equals the bias-corrected forecast exp(mu +
    sigma^2 / 2).

    Parameters
    ----------
    residual_mode : {"gaussian", "empirical"}
    use_hac : bool
        Passed to the underlying HARModel for Newey-West HAC SEs.
    levels : (K,) array
        Quantile grid to emit. Defaults to DEFAULT_QUANTILE_LEVELS.
    """

    def __init__(
        self,
        residual_mode: ResidualMode = "gaussian",
        use_hac: bool = True,
        levels: Optional[np.ndarray] = None,
    ):
        self.residual_mode = residual_mode
        self.use_hac = use_hac
        self.levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        self._har = HARModel(use_hac=use_hac, use_log=True)
        self._fitted = False
        self._sigma: Optional[float] = None
        self._empirical_resid: Optional[np.ndarray] = None
        self._train_X: Optional[pd.DataFrame] = None
        self._train_y_log: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARDensityModel":
        """Fit Log-HAR on the training window and capture residual scale."""
        result = self._har.fit(X, y)
        # Residuals from the Log-HAR fit are already in log space.
        resid = result.residuals.to_numpy()
        self._sigma = float(np.sqrt(np.var(resid, ddof=len(result.params))))
        self._empirical_resid = resid - resid.mean()
        self._train_X = X
        self._train_y_log = np.log(y.clip(lower=1e-10))
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_density(self, X: pd.DataFrame) -> HARDensityForecast:
        """Return predictive distribution for each row of X."""
        if not self._fitted:
            raise RuntimeError("call fit() before predict_density()")
        assert self._sigma is not None

        # Underlying HAR predict() returns level-space bias-corrected forecasts.
        point_level = self._har.predict(X).astype(float)
        # Recover the log-space conditional mean used as the location.
        # point_level = exp(mu + sigma^2 / 2)  =>  mu = log(point) - sigma^2 / 2
        mu = np.log(point_level.clip(lower=1e-30).to_numpy()) - 0.5 * self._sigma ** 2

        if self.residual_mode == "gaussian":
            z = stats.norm.ppf(self.levels)
            log_q = mu[:, None] + self._sigma * z[None, :]
        else:
            # Empirical residual quantiles.
            r_q = np.quantile(self._empirical_resid, self.levels)
            log_q = mu[:, None] + r_q[None, :]

        level_q = np.exp(log_q)

        return HARDensityForecast(
            dates=X.index,
            point=point_level,
            quantile_levels=self.levels,
            log_quantiles=log_q,
            level_quantiles=level_q,
            sigma=self._sigma,
            mode=self.residual_mode,
        )


# ----------------------------------------------------------------------
# Self-test on a real VOLARE asset slice.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import load_data
    from features import build_har_features, build_target, align_features_target
    from evaluation.density import density_summary

    data = load_data(dataset="volare", tickers=["AAPL"])
    rv = data.rv["AAPL"].dropna()
    horizon = 1

    X = build_har_features(rv)
    y = build_target(rv, horizon=horizon)
    X, y = align_features_target(X, y)

    # 75/25 walk-forward split for the self-test.
    cut = int(len(X) * 0.75)
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]

    for mode in ("gaussian", "empirical"):
        model = HARDensityModel(residual_mode=mode).fit(X_tr, y_tr)
        fc = model.predict_density(X_te)

        # Score in log space (primary per Brini reply).
        log_actual = np.log(y_te.to_numpy())
        log_summary = density_summary(log_actual, fc.log_quantiles, fc.quantile_levels)
        lvl_summary = density_summary(y_te.to_numpy(), fc.level_quantiles, fc.quantile_levels)

        print(f"\nAAPL h=1  Log-HAR + {mode} residuals  (n_test={len(y_te)})")
        print(f"  sigma_hat = {fc.sigma:.4f}")
        print(f"  Log-space:   CRPS={log_summary.crps_mean:.4f}  "
              f"KS={log_summary.pit_ks_stat:.3f}  "
              f"cov50/80/95={log_summary.coverage[0.50]:.2f}/"
              f"{log_summary.coverage[0.80]:.2f}/"
              f"{log_summary.coverage[0.95]:.2f}")
        print(f"  Level-space: CRPS={lvl_summary.crps_mean:.3e}  "
              f"KS={lvl_summary.pit_ks_stat:.3f}  "
              f"cov50/80/95={lvl_summary.coverage[0.50]:.2f}/"
              f"{lvl_summary.coverage[0.80]:.2f}/"
              f"{lvl_summary.coverage[0.95]:.2f}")
