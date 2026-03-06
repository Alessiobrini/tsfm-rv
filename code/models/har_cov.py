"""
models/har_cov.py — Element-wise HAR for covariance matrix forecasting.

Applies the standard HAR model (Corsi 2009) independently to each element
of the realized covariance matrix. Reconstructs the full matrix from
element-wise forecasts, with PSD projection if needed.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Tuple

from covariance_utils import ivech, ensure_psd


class ElementwiseHARCov:
    """Element-wise HAR model for covariance matrix forecasting.

    Forecasts each unique element (i,j) of the N x N covariance matrix
    independently using a univariate HAR model. Reconstructs the full
    matrix and projects to PSD if necessary.

    Parameters
    ----------
    daily_lag : int
        Lag for daily component (default 1).
    weekly_lag : int
        Window for weekly average (default 5).
    monthly_lag : int
        Window for monthly average (default 22).
    """

    def __init__(self, daily_lag: int = 1, weekly_lag: int = 5, monthly_lag: int = 22):
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        self._models = {}  # (i, j) -> OLS result

    def _build_har_features(self, series: pd.Series) -> pd.DataFrame:
        """Build HAR regressors for a single element series."""
        rv_d = series.shift(self.daily_lag)
        rv_w = series.rolling(self.weekly_lag, min_periods=self.weekly_lag).mean().shift(1)
        rv_m = series.rolling(self.monthly_lag, min_periods=self.monthly_lag).mean().shift(1)
        features = pd.DataFrame({'d': rv_d, 'w': rv_w, 'm': rv_m}, index=series.index)
        return features

    def fit(
        self,
        pair_series: Dict[Tuple[str, str], pd.Series],
        pairs: List[Tuple[str, str]],
        train_dates: pd.DatetimeIndex,
    ) -> None:
        """Fit HAR model for each covariance pair on training data.

        Parameters
        ----------
        pair_series : dict
            Maps (asset1, asset2) -> pd.Series of covariance values.
        pairs : list
            List of (asset1, asset2) pairs to forecast (upper triangle + diagonal).
        train_dates : DatetimeIndex
            Training period dates.
        """
        self._models = {}
        for pair in pairs:
            series = pair_series[pair]
            features = self._build_har_features(series)
            target = series

            # Align to training dates
            common = features.index.intersection(train_dates).intersection(target.index)
            X = features.loc[common].dropna()
            y = target.loc[X.index]
            mask = ~y.isna()
            X, y = X[mask], y[mask]

            if len(X) < 50:
                self._models[pair] = None
                continue

            X_const = sm.add_constant(X, has_constant='add')
            self._models[pair] = sm.OLS(y, X_const).fit()

    def predict_single(
        self,
        pair: Tuple[str, str],
        pair_series: Dict[Tuple[str, str], pd.Series],
        forecast_date: pd.Timestamp,
    ) -> float:
        """Predict a single element for a given date."""
        model = self._models.get(pair)
        if model is None:
            # Fallback: use last observed value
            s = pair_series[pair]
            prev = s[s.index < forecast_date]
            return float(prev.iloc[-1]) if len(prev) > 0 else 0.0

        series = pair_series[pair]
        features = self._build_har_features(series)
        if forecast_date not in features.index:
            # Use nearest prior date
            prior = features.index[features.index <= forecast_date]
            if len(prior) == 0:
                return 0.0
            forecast_date = prior[-1]

        x = features.loc[[forecast_date]].dropna()
        if len(x) == 0:
            prev = series[series.index < forecast_date]
            return float(prev.iloc[-1]) if len(prev) > 0 else 0.0

        x_const = sm.add_constant(x, has_constant='add')
        return float(model.predict(x_const).iloc[0])

    def predict_matrix(
        self,
        pairs: List[Tuple[str, str]],
        pair_series: Dict[Tuple[str, str], pd.Series],
        assets: List[str],
        forecast_date: pd.Timestamp,
        apply_psd: bool = True,
    ) -> np.ndarray:
        """Predict the full covariance matrix for a single date.

        Parameters
        ----------
        pairs : list
            Upper-triangular pairs.
        pair_series : dict
            Covariance series for each pair.
        assets : list
            Sorted asset list.
        forecast_date : Timestamp
            Date to forecast.
        apply_psd : bool
            Whether to project to PSD.

        Returns
        -------
        np.ndarray
            Forecasted covariance matrix (N x N).
        """
        n = len(assets)
        asset_idx = {a: i for i, a in enumerate(assets)}
        mat = np.zeros((n, n))

        for pair in pairs:
            val = self.predict_single(pair, pair_series, forecast_date)
            i, j = asset_idx[pair[0]], asset_idx[pair[1]]
            mat[i, j] = val
            mat[j, i] = val

        if apply_psd:
            mat = ensure_psd(mat)

        return mat
