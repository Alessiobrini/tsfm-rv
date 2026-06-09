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
        frac_trunc: int = 200,
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
        self.frac_trunc = frac_trunc       # truncation lag for the (1-L)^d filter
        self._result = None
        self._d_hat = None
        self._series_mean = None
        self._w = None                     # fractionally differenced series
        self._psi = None                   # (1-L)^{-d} re-integration weights
        self._sigma2 = 0.0
        self._order = (p, q)
        self._model_name = f"ARFIMA({p},d,{q})"

    def fit(self, series: pd.Series) -> ARFIMAResult:
        """Estimate ARFIMA(p, d, q) on (log) RV.

        Step 1: estimate the memory parameter d (local-Whittle ML by default, or
        GPH). Step 2: fractionally difference the de-meaned series, w = (1-L)^d
        (y - mu), and fit an IC-selected ARMA(p,q) on w. Forecasting (see predict)
        ARMA-forecasts w and re-integrates (1-L)^{-d}, so the long-memory parameter
        genuinely enters the forecast (distinct from the plain ARMA benchmark).

        Parameters
        ----------
        series : pd.Series
            RV series (levels, not log).

        Returns
        -------
        ARFIMAResult
            Estimation results (d, ARMA params, IC, residuals).
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

        # Step 2: genuine ARFIMA. Fractionally difference the de-meaned series with
        # the estimated d, fit ARMA(p,q) by IC on the short-memory residual w_t =
        # (1-L)^d (y - mu), and forecast by re-integrating (1-L)^{-d} (see predict).
        # The slowly-decaying re-integration weights inject long-memory persistence
        # into the multi-step forecast -- this is what distinguishes ARFIMA from the
        # plain ARMA-on-log-RV benchmark (which would otherwise be identical).
        yc = (y - self._series_mean).values.astype(float)
        K = min(self.frac_trunc, len(yc) - 1)
        self._w = np.asarray(self._fracdiff(yc, self._d_hat, max_lag=K), dtype=float)

        candidates = ([(pp, qq) for pp in range(self.p + 1) for qq in range(self.q + 1)
                       if not (pp == 0 and qq == 0)] if self.select else [(self.p, self.q)])
        best_res, best_ic, best_order = None, np.inf, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for (pp, qq) in candidates:
                try:
                    res = ARIMA(self._w, order=(pp, 0, qq), trend='n').fit()
                    crit = res.bic if self.ic == 'bic' else res.aic
                    if np.isfinite(crit) and crit < best_ic:
                        best_ic, best_res, best_order = crit, res, (pp, qq)
                except Exception:
                    continue
            if best_res is None:
                best_res, best_order = ARIMA(self._w, order=(1, 0, 0), trend='n').fit(), (1, 0)

        self._result = best_res
        self._order = best_order
        self._sigma2 = float(getattr(best_res, "mse", np.var(self._w)))

        # Inverse fractional-difference weights psi_j of (1-L)^{-d}:
        #   psi_0 = 1, psi_j = psi_{j-1} * (j - 1 + d) / j  (decay ~ j^{d-1}: long memory)
        J = len(self._w) + 64
        psi = np.empty(J + 1)
        psi[0] = 1.0
        for j in range(1, J + 1):
            psi[j] = psi[j - 1] * (j - 1 + self._d_hat) / j
        self._psi = psi
        self._model_name = f"ARFIMA({best_order[0]},d,{best_order[1]})"
        self._last_series = y

        return ARFIMAResult(
            d=self._d_hat,
            ar_params=best_res.arparams if (hasattr(best_res, 'arparams') and best_order[0] > 0) else np.array([]),
            ma_params=best_res.maparams if (hasattr(best_res, 'maparams') and best_order[1] > 0) else np.array([]),
            log_likelihood=best_res.llf,
            aic=best_res.aic,
            bic=best_res.bic,
            residuals=pd.Series(best_res.resid),
            model_name=self._model_name,
        )

    def predict(self, steps: int = 1) -> np.ndarray:
        """h-step ARFIMA forecast.

        Forecast the fractionally differenced series w with the fitted ARMA, then
        re-integrate y_{T+k} - mu = sum_{j>=0} psi_j w_{T+k-j} (using observed w
        for past lags and ARMA forecasts for future lags), and exponentiate with
        the Jensen bias correction when use_log.
        """
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_hat = np.asarray(self._result.forecast(steps=steps), dtype=float)

        w_seq = np.concatenate([self._w, w_hat])   # observed w, then forecasts
        n_obs = len(self._w)                        # time T sits at index n_obs-1
        psi = self._psi
        out = np.empty(steps)
        for k in range(1, steps + 1):
            idx0 = n_obs - 1 + k                    # index in w_seq of time T+k
            jmax = min(len(psi) - 1, idx0)
            # sum_{j=0}^{jmax} psi_j * w_{T+k-j}
            out[k - 1] = self._series_mean + float(
                np.dot(psi[:jmax + 1], w_seq[idx0 - jmax: idx0 + 1][::-1])
            )

        if self.use_log:
            out = np.exp(out + 0.5 * self._sigma2)
        return np.asarray(out, dtype=float)

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
