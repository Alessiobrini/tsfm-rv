"""
models/har_drd.py — HAR-DRD covariance forecasting model.

Implements the Bollerslev, Patton & Quaedvlieg (2018) decomposition:
    Sigma = D R D
where:
    - D: diagonal matrix of standard deviations, each forecast via Log-HAR
    - R: correlation matrix, each element forecast via HAR on Fisher-z transforms
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Tuple

from covariance_utils import cov_to_drd, drd_to_cov, ensure_correlation, ensure_psd


class HARDRDModel:
    """HAR-DRD covariance forecasting model.

    Step 1 (D): Forecast each diagonal variance using Log-HAR, take sqrt for std dev.
    Step 2 (R): Forecast each off-diagonal correlation using HAR on Fisher-z(r_ij).
    Step 3: Recombine Sigma_hat = D_hat R_hat D_hat.

    Parameters
    ----------
    daily_lag : int
        Lag for daily component.
    weekly_lag : int
        Window for weekly average.
    monthly_lag : int
        Window for monthly average.
    """

    def __init__(self, daily_lag: int = 1, weekly_lag: int = 5, monthly_lag: int = 22):
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        self._var_models = {}    # asset -> OLS result (Log-HAR for variance)
        self._corr_models = {}   # (a1, a2) -> OLS result (HAR for Fisher-z)
        self._var_sigma2 = {}    # asset -> residual variance for bias correction

    def _build_har_features(self, series: pd.Series) -> pd.DataFrame:
        rv_d = series.shift(self.daily_lag)
        rv_w = series.rolling(self.weekly_lag, min_periods=self.weekly_lag).mean().shift(1)
        rv_m = series.rolling(self.monthly_lag, min_periods=self.monthly_lag).mean().shift(1)
        return pd.DataFrame({'d': rv_d, 'w': rv_w, 'm': rv_m}, index=series.index)

    def _fit_single_ols(self, features: pd.DataFrame, target: pd.Series, train_dates):
        common = features.index.intersection(train_dates).intersection(target.index)
        combined = features.loc[common].copy()
        combined['_target'] = target.loc[common]
        combined = combined.dropna()
        X = combined[['d', 'w', 'm']]
        y = combined['_target']
        if len(X) < 50:
            return None, 0.0
        X_const = sm.add_constant(X, has_constant='add')
        result = sm.OLS(y, X_const).fit()
        return result, result.mse_resid

    def fit(
        self,
        pair_series: Dict[Tuple[str, str], pd.Series],
        assets: List[str],
        train_dates: pd.DatetimeIndex,
    ) -> None:
        """Fit Log-HAR for variances and HAR for Fisher-z correlations.

        Parameters
        ----------
        pair_series : dict
            Maps (asset1, asset2) -> covariance series.
        assets : list
            Sorted asset list.
        train_dates : DatetimeIndex
            Training period.
        """
        # Step 1: Fit Log-HAR for each diagonal element (variance)
        self._var_models = {}
        self._var_sigma2 = {}
        for asset in assets:
            var_series = pair_series.get((asset, asset))
            if var_series is None:
                continue
            log_var = np.log(var_series.clip(lower=1e-20))
            features = self._build_har_features(log_var)
            model, sigma2 = self._fit_single_ols(features, log_var, train_dates)
            self._var_models[asset] = model
            self._var_sigma2[asset] = sigma2

        # Step 2: Fit HAR for each off-diagonal correlation (Fisher-z)
        # First compute correlation series for each off-diagonal pair
        self._corr_models = {}
        for i, a1 in enumerate(assets):
            for j, a2 in enumerate(assets):
                if j <= i:
                    continue
                cov_ij = pair_series.get((a1, a2))
                var_i = pair_series.get((a1, a1))
                var_j = pair_series.get((a2, a2))
                if cov_ij is None or var_i is None or var_j is None:
                    continue

                # Align dates
                common_idx = cov_ij.index.intersection(var_i.index).intersection(var_j.index)
                cov_ij_a = cov_ij.loc[common_idx]
                var_i_a = var_i.loc[common_idx]
                var_j_a = var_j.loc[common_idx]

                # Compute correlation
                denom = np.sqrt(var_i_a * var_j_a)
                denom = denom.clip(lower=1e-20)
                corr = (cov_ij_a / denom).clip(-0.999, 0.999)

                # Fisher-z transform
                z = np.arctanh(corr)

                features = self._build_har_features(z)
                model, _ = self._fit_single_ols(features, z, train_dates)
                self._corr_models[(a1, a2)] = model

    def predict_matrix(
        self,
        pair_series: Dict[Tuple[str, str], pd.Series],
        assets: List[str],
        forecast_date: pd.Timestamp,
    ) -> np.ndarray:
        """Predict the full covariance matrix for a single date.

        Parameters
        ----------
        pair_series : dict
            Covariance series.
        assets : list
            Sorted asset list.
        forecast_date : Timestamp
            Date to forecast.

        Returns
        -------
        np.ndarray
            Forecasted covariance matrix (N x N).
        """
        n = len(assets)

        # Step 1: Predict variances via Log-HAR
        d = np.zeros(n)
        for idx, asset in enumerate(assets):
            model = self._var_models.get(asset)
            var_series = pair_series.get((asset, asset))
            if model is None or var_series is None:
                # Fallback: last observed variance
                if var_series is not None:
                    prev = var_series[var_series.index < forecast_date]
                    d[idx] = np.sqrt(max(float(prev.iloc[-1]), 1e-20)) if len(prev) > 0 else 1e-5
                continue

            log_var = np.log(var_series.clip(lower=1e-20))
            features = self._build_har_features(log_var)
            if forecast_date in features.index:
                x = features.loc[[forecast_date]].dropna()
            else:
                prior = features.index[features.index <= forecast_date]
                if len(prior) == 0:
                    d[idx] = 1e-5
                    continue
                x = features.loc[[prior[-1]]].dropna()

            if len(x) == 0:
                prev = var_series[var_series.index < forecast_date]
                d[idx] = np.sqrt(max(float(prev.iloc[-1]), 1e-20)) if len(prev) > 0 else 1e-5
                continue

            x_const = sm.add_constant(x, has_constant='add')
            log_pred = float(model.predict(x_const).iloc[0])
            # Bias correction
            sigma2 = self._var_sigma2.get(asset, 0.0)
            var_pred = np.exp(log_pred + 0.5 * sigma2)
            d[idx] = np.sqrt(max(var_pred, 1e-20))

        # Step 2: Predict correlations via HAR on Fisher-z
        R = np.eye(n)
        for i, a1 in enumerate(assets):
            for j, a2 in enumerate(assets):
                if j <= i:
                    continue
                model = self._corr_models.get((a1, a2))
                if model is None:
                    # Fallback: last observed correlation
                    cov_ij = pair_series.get((a1, a2))
                    var_i = pair_series.get((a1, a1))
                    var_j = pair_series.get((a2, a2))
                    if cov_ij is not None and var_i is not None and var_j is not None:
                        prev_dates = cov_ij.index[cov_ij.index < forecast_date]
                        if len(prev_dates) > 0:
                            last = prev_dates[-1]
                            denom = np.sqrt(max(var_i.get(last, 1e-10), 1e-20) *
                                            max(var_j.get(last, 1e-10), 1e-20))
                            r_val = np.clip(cov_ij.get(last, 0) / denom, -0.999, 0.999)
                            R[i, j] = R[j, i] = r_val
                    continue

                # Build Fisher-z series for prediction
                cov_ij = pair_series.get((a1, a2))
                var_i = pair_series.get((a1, a1))
                var_j = pair_series.get((a2, a2))
                if cov_ij is None or var_i is None or var_j is None:
                    continue

                common_idx = cov_ij.index.intersection(var_i.index).intersection(var_j.index)
                denom = np.sqrt(var_i.loc[common_idx] * var_j.loc[common_idx]).clip(lower=1e-20)
                corr = (cov_ij.loc[common_idx] / denom).clip(-0.999, 0.999)
                z = np.arctanh(corr)

                features = self._build_har_features(z)
                if forecast_date in features.index:
                    x = features.loc[[forecast_date]].dropna()
                else:
                    prior = features.index[features.index <= forecast_date]
                    if len(prior) == 0:
                        continue
                    x = features.loc[[prior[-1]]].dropna()

                if len(x) == 0:
                    continue

                x_const = sm.add_constant(x, has_constant='add')
                z_pred = float(model.predict(x_const).iloc[0])
                r_pred = np.clip(np.tanh(z_pred), -0.999, 0.999)
                R[i, j] = R[j, i] = r_pred

        # Ensure valid correlation matrix
        R = ensure_correlation(R)

        # Step 3: Recombine
        cov_hat = drd_to_cov(d, R)
        cov_hat = ensure_psd(cov_hat)

        return cov_hat
