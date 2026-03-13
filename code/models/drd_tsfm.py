"""
models/drd_tsfm.py — DRD-TSFM hybrid covariance forecasting model.

Applies the DRD decomposition (Bollerslev, Patton & Quaedvlieg 2018)
with TSFM-based forecasting of each component:
    D (volatilities): forecast each diagonal variance via TSFM, take sqrt
    R (correlations): forecast Fisher-z transformed correlations via TSFM,
                      inverse-transform and project to valid correlation matrix
    Sigma_hat = D_hat R_hat D_hat
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from covariance_utils import drd_to_cov, ensure_correlation, ensure_psd
from models.foundation import BaseTSFM


class DRDTSFMModel:
    """DRD-TSFM hybrid covariance forecasting model.

    Uses a TSFM to forecast each component of the DRD decomposition:
    - Variances: TSFM on log(variance) series, exponentiate, take sqrt
    - Correlations: TSFM on Fisher-z(correlation) series, tanh to recover r

    Parameters
    ----------
    tsfm : BaseTSFM
        A loaded TSFM model instance (e.g., Chronos-Bolt or Moirai).
    context_length : int
        Number of past observations to feed the TSFM.
    """

    def __init__(self, tsfm: BaseTSFM, context_length: int = 512):
        self.tsfm = tsfm
        self.context_length = context_length

    def forecast_variance(
        self,
        var_series: np.ndarray,
        horizon: int,
    ) -> float:
        """Forecast a single variance element using the TSFM.

        Operates in log space for positivity, then exponentiates.

        Parameters
        ----------
        var_series : np.ndarray
            Historical variance values (at least context_length observations).
        horizon : int
            Forecast horizon.

        Returns
        -------
        float
            Forecasted variance (positive).
        """
        # Work in log space for positivity
        log_var = np.log(np.clip(var_series, 1e-20, None))
        ctx = log_var[-self.context_length:]

        result = self.tsfm.predict(ctx, horizon)
        if horizon == 1:
            log_pred = float(result.point[0])
        else:
            log_pred = float(np.mean(result.point[:horizon]))

        return max(np.exp(log_pred), 1e-20)

    def forecast_correlation(
        self,
        corr_series: np.ndarray,
        horizon: int,
    ) -> float:
        """Forecast a single correlation element using the TSFM.

        Operates in Fisher-z space, then applies tanh to recover [-1, 1].

        Parameters
        ----------
        corr_series : np.ndarray
            Historical correlation values, clipped to (-0.999, 0.999).
        horizon : int
            Forecast horizon.

        Returns
        -------
        float
            Forecasted correlation in [-0.999, 0.999].
        """
        z = np.arctanh(np.clip(corr_series, -0.999, 0.999))
        ctx = z[-self.context_length:]

        result = self.tsfm.predict(ctx, horizon)
        if horizon == 1:
            z_pred = float(result.point[0])
        else:
            z_pred = float(np.mean(result.point[:horizon]))

        return float(np.clip(np.tanh(z_pred), -0.999, 0.999))

    def predict_matrix(
        self,
        pair_series: Dict[Tuple[str, str], pd.Series],
        assets: List[str],
        forecast_date: pd.Timestamp,
        horizon: int = 1,
    ) -> np.ndarray:
        """Predict the full covariance matrix for a single date.

        Parameters
        ----------
        pair_series : dict
            Maps (asset1, asset2) -> covariance series.
        assets : list
            Sorted asset list.
        forecast_date : Timestamp
            Date to forecast (context window ends just before this date).
        horizon : int
            Forecast horizon.

        Returns
        -------
        np.ndarray
            Forecasted covariance matrix (N x N), PSD-projected.
        """
        n = len(assets)

        # Step 1: Forecast variances
        d = np.zeros(n)
        for idx, asset in enumerate(assets):
            var_series = pair_series.get((asset, asset))
            if var_series is None:
                d[idx] = 1e-5
                continue

            # Get history up to (but not including) forecast_date
            hist = var_series[var_series.index < forecast_date].dropna()
            if len(hist) < self.context_length:
                # Fallback: last observed value
                d[idx] = np.sqrt(max(float(hist.iloc[-1]), 1e-20)) if len(hist) > 0 else 1e-5
                continue

            var_pred = self.forecast_variance(hist.values, horizon)
            d[idx] = np.sqrt(var_pred)

        # Step 2: Forecast correlations
        R = np.eye(n)
        for i, a1 in enumerate(assets):
            for j, a2 in enumerate(assets):
                if j <= i:
                    continue

                cov_ij = pair_series.get((a1, a2))
                var_i = pair_series.get((a1, a1))
                var_j = pair_series.get((a2, a2))

                if cov_ij is None or var_i is None or var_j is None:
                    continue

                # Compute historical correlation series
                common_idx = (
                    cov_ij.index
                    .intersection(var_i.index)
                    .intersection(var_j.index)
                )
                common_idx = common_idx[common_idx < forecast_date]
                if len(common_idx) < self.context_length:
                    # Fallback: last observed correlation
                    if len(common_idx) > 0:
                        last = common_idx[-1]
                        denom = np.sqrt(
                            max(var_i.get(last, 1e-10), 1e-20)
                            * max(var_j.get(last, 1e-10), 1e-20)
                        )
                        r_val = np.clip(cov_ij.get(last, 0) / denom, -0.999, 0.999)
                        R[i, j] = R[j, i] = r_val
                    continue

                cov_vals = cov_ij.loc[common_idx]
                var_i_vals = var_i.loc[common_idx]
                var_j_vals = var_j.loc[common_idx]

                denom = np.sqrt(var_i_vals * var_j_vals).clip(lower=1e-20)
                corr = (cov_vals / denom).clip(-0.999, 0.999).values

                r_pred = self.forecast_correlation(corr, horizon)
                R[i, j] = R[j, i] = r_pred

        # Ensure valid correlation matrix
        R = ensure_correlation(R)

        # Step 3: Recombine
        cov_hat = drd_to_cov(d, R)
        cov_hat = ensure_psd(cov_hat)

        return cov_hat
