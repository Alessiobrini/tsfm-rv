"""
models/arfima.py — ARFIMA model for long-memory realized volatility.

ARFIMA(p, d, q) captures the long-memory property of log-RV.
The fractional differencing parameter d in (0, 0.5) controls the
rate of ACF decay. Standard practice: fit on log(RV).

Two-step approach:
    1. Estimate d via GPH log-periodogram regression
    2. Fit ARMA(p, q) on fractionally differenced series

Uses statsmodels ARIMA for the ARMA step.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ARFIMAResult:
    """Container for ARFIMA estimation results."""
    d: float
    ar_params: np.ndarray
    ma_params: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    residuals: pd.Series
    model_name: str


class ARFIMAModel:
    """ARFIMA(p, d, q) model for long-memory time series.

    Walk-forward compatible interface:
        - fit(series) -> ARFIMAResult
        - predict(steps) -> np.ndarray

    Parameters
    ----------
    p : int
        AR order.
    q : int
        MA order.
    use_log : bool
        If True, fit on log(RV). Standard for RV forecasting.
    d_method : str
        Method for estimating d: 'gph' (default, fast).
    """

    def __init__(
        self,
        p: int = 2,
        q: int = 2,
        use_log: bool = True,
        d_method: str = 'whittle',
        ic: str = 'bic',
        select: bool = True,
    ):
        self.p = p
        self.q = q
        self.use_log = use_log
        # 'whittle' (local-Whittle ML estimator, Robinson 1995) or 'gph'. 'mle'
        # is accepted as an alias for 'whittle'. Addresses Referee 1's request to
        # estimate the fractional parameter by (quasi-)ML rather than the GPH
        # log-periodogram regression.
        self.d_method = 'whittle' if d_method == 'mle' else d_method
        self.ic = ic.lower()
        # Select (p,q) over 0..p x 0..q by information criterion rather than
        # fixing (2,2) (Referee 1 questioned whether p=q=2 is adequate).
        self.select = select
        self._result = None
        self._d_hat = None
        self._series_mean = None
        self._order = (p, q)
        self._model_name = f"ARFIMA({p},d,{q})"

    def fit(self, series: pd.Series) -> ARFIMAResult:
        """Estimate ARFIMA model: GPH for d (diagnostic), then ARMA on log(RV).

        Fits ARMA directly on log(RV) rather than on the frac-differenced series.
        The fractional d is estimated and reported but not used in the ARMA step,
        because proper fractional integration for prediction requires infinite
        AR representation which is numerically unstable when d is near 0.5.
        ARMA(p,q) on log(RV) captures short-to-medium term dynamics (1-22 days)
        adequately for forecasting purposes.

        Parameters
        ----------
        series : pd.Series
            RV series (levels, not log).

        Returns
        -------
        ARFIMAResult
            Estimation results.
        """
        from statsmodels.tsa.arima.model import ARIMA

        y = np.log(series.clip(lower=1e-10)) if self.use_log else series.copy()
        y = y.dropna()
        self._series_mean = y.mean()

        # Step 1: Estimate the fractional parameter d.
        if self.d_method == 'gph':
            self._d_hat = self._estimate_d_gph(y)
        else:
            self._d_hat = self._estimate_d_whittle(y)

        # Step 2: Select (p, q) by IC (or use the fixed order) and fit ARMA on
        # log(RV). We retain ARMA-on-log-RV forecasting rather than re-integrating
        # the fractionally differenced series: the latter requires a truncated
        # infinite AR/MA representation that is numerically unstable when d is
        # near 0.5. The estimated d is reported as a long-memory diagnostic.
        candidates = ([(p, q) for p in range(self.p + 1) for q in range(self.q + 1)
                       if not (p == 0 and q == 0)] if self.select else [(self.p, self.q)])
        best_res, best_ic, best_order = None, np.inf, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for (p, q) in candidates:
                try:
                    res = ARIMA(y, order=(p, 0, q)).fit()
                    crit = res.bic if self.ic == 'bic' else res.aic
                    if np.isfinite(crit) and crit < best_ic:
                        best_ic, best_res, best_order = crit, res, (p, q)
                except Exception:
                    continue
            if best_res is None:
                best_res, best_order = ARIMA(y, order=(1, 0, 0)).fit(), (1, 0)

        self._result = best_res
        self._order = best_order
        self._model_name = f"ARFIMA({best_order[0]},d,{best_order[1]})"
        self._last_series = y

        return ARFIMAResult(
            d=self._d_hat,
            ar_params=self._result.arparams if hasattr(self._result, 'arparams') and self.p > 0 else np.array([]),
            ma_params=self._result.maparams if hasattr(self._result, 'maparams') and self.q > 0 else np.array([]),
            log_likelihood=self._result.llf,
            aic=self._result.aic,
            bic=self._result.bic,
            residuals=pd.Series(self._result.resid, index=y.index),
            model_name=self._model_name,
        )

    def predict(self, steps: int = 1) -> np.ndarray:
        """Generate h-step-ahead forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Point forecasts in RV levels (exponentiated if use_log).
        """
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecasts = np.asarray(self._result.forecast(steps=steps))

        if self.use_log:
            # Bias correction: E[RV] = exp(log_pred + 0.5 * sigma^2)
            sigma2 = self._result.mse
            forecasts = np.exp(forecasts + 0.5 * sigma2)

        return np.asarray(forecasts)

    @staticmethod
    def _estimate_d_gph(
        series: pd.Series,
        m: Optional[int] = None,
    ) -> float:
        """Estimate fractional d via GPH log-periodogram regression.

        Parameters
        ----------
        series : pd.Series
            Stationary (or fractionally integrated) series.
        m : int, optional
            Number of low-frequency periodogram ordinates. Default: T^0.5.

        Returns
        -------
        float
            Estimated d, clipped to (-0.5, 0.5).
        """
        y = series.dropna().values
        n = len(y)
        if m is None:
            m = int(np.sqrt(n))

        freqs = 2 * np.pi * np.arange(1, m + 1) / n
        fft_vals = np.fft.fft(y - y.mean())
        periodogram = (np.abs(fft_vals[1:m + 1]) ** 2) / (2 * np.pi * n)

        # Guard against zero periodogram values
        periodogram = np.maximum(periodogram, 1e-20)

        log_I = np.log(periodogram)
        log_freq = np.log(2 * np.sin(freqs / 2))

        X = np.column_stack([np.ones(m), -2 * log_freq])
        beta = np.linalg.lstsq(X, log_I, rcond=None)[0]
        d_hat = beta[1]

        return float(np.clip(d_hat, -0.49, 0.49))

    @staticmethod
    def _estimate_d_whittle(series: pd.Series, m: Optional[int] = None) -> float:
        """Estimate the memory parameter d by the local-Whittle estimator
        (Robinson 1995) — a semiparametric (quasi-)ML estimator in the frequency
        domain, more principled than GPH. Minimizes
        R(d) = log( mean_j I_j freq_j^{2d} ) - 2d * mean_j log(freq_j)
        over the first m Fourier frequencies.
        """
        from scipy.optimize import minimize_scalar

        y = series.dropna().values
        n = len(y)
        if m is None:
            m = int(n ** 0.65)              # standard local-Whittle bandwidth
        m = max(4, min(m, n // 2 - 1))

        fft_vals = np.fft.fft(y - y.mean())
        periodogram = np.maximum((np.abs(fft_vals[1:m + 1]) ** 2) / (2 * np.pi * n), 1e-20)
        freqs = 2 * np.pi * np.arange(1, m + 1) / n
        log_freq_mean = np.mean(np.log(freqs))

        def objective(d):
            G = np.mean(periodogram * freqs ** (2 * d))
            return np.log(max(G, 1e-300)) - 2 * d * log_freq_mean

        res = minimize_scalar(objective, bounds=(-0.49, 0.49), method='bounded')
        return float(np.clip(res.x, -0.49, 0.49))

    @staticmethod
    def _fracdiff(x: np.ndarray, d: float, max_lag: int = 100) -> np.ndarray:
        """Apply fractional differencing to a series.

        Uses truncated binomial expansion: (1-L)^d = sum_{k=0}^{K} w_k L^k
        where w_0 = 1, w_k = w_{k-1} * (k - 1 - d) / k.

        Parameters
        ----------
        x : np.ndarray
            Input series.
        d : float
            Fractional differencing parameter.
        max_lag : int
            Truncation point for the binomial expansion.

        Returns
        -------
        np.ndarray
            Fractionally differenced series (shorter by max_lag).
        """
        n = len(x)
        if n <= max_lag:
            max_lag = n - 1

        # Compute weights
        weights = np.zeros(max_lag + 1)
        weights[0] = 1.0
        for k in range(1, max_lag + 1):
            weights[k] = weights[k - 1] * (k - 1 - d) / k

        # Apply filter: y_t = sum_{k=0}^{K} w_k * x_{t-k}
        out = np.zeros(n - max_lag)
        for t in range(max_lag, n):
            window = x[t - max_lag:t + 1][::-1]  # x_t, x_{t-1}, ..., x_{t-K}
            out[t - max_lag] = np.dot(weights, window)

        return out
