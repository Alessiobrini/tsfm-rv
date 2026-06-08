"""models/arma.py — ARMA(p,q) on log-RV with information-criterion order selection.

Referee 1 (1.4.2) recommended adding an ARMA-on-log-RV benchmark, noting that in
their experience ARMA models on log-RV often forecast volatility as well as or
better than HAR without needing fractional integration, and that the order should
be selected rather than fixed at (2,2).

The model is fit on log(RV) (so forecasts are positive after exponentiating) with
the standard ``exp(. + 0.5 sigma^2)`` Jensen bias correction, and selects (p, q)
over a small grid by AIC or BIC. It exposes the same ``fit(series)/predict(steps)``
interface as ARFIMAModel, so it plugs into the series walk-forward / iterated
engines and forecasts multi-step natively (iterated).
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ARMAResult:
    """Container for ARMA estimation results."""
    order: tuple
    aic: float
    bic: float
    log_likelihood: float
    model_name: str


class ARMAModel:
    """ARMA(p, q) on (log) RV with AIC/BIC order selection.

    Parameters
    ----------
    max_p, max_q : int
        Upper bounds for the (p, q) grid search (inclusive). Default 2 each
        (Referee 1 notes (2,2) is "probably adequate"; we select within the grid).
    use_log : bool
        Fit on log(RV) with bias-corrected exponentiation (standard for RV).
    ic : str
        Selection criterion: 'bic' (default, parsimonious) or 'aic'.
    """

    def __init__(self, max_p: int = 2, max_q: int = 2, use_log: bool = True, ic: str = "bic"):
        self.max_p = max_p
        self.max_q = max_q
        self.use_log = use_log
        self.ic = ic.lower()
        if self.ic not in ("aic", "bic"):
            raise ValueError(f"ic must be 'aic' or 'bic', got {ic!r}")
        self._result = None
        self._order = None
        self._model_name = "ARMA"

    def fit(self, series: pd.Series) -> ARMAResult:
        """Select (p, q) by IC over the grid and fit on (log) RV."""
        from statsmodels.tsa.arima.model import ARIMA

        y = np.log(series.clip(lower=1e-10)) if self.use_log else series.copy()
        y = y.dropna()

        best_res, best_ic, best_order = None, np.inf, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue  # need at least one dynamic term
                    try:
                        res = ARIMA(y, order=(p, 0, q)).fit()
                        crit = res.bic if self.ic == "bic" else res.aic
                        if np.isfinite(crit) and crit < best_ic:
                            best_ic, best_res, best_order = crit, res, (p, q)
                    except Exception:
                        continue

        if best_res is None:
            # Fallback: AR(1).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_res, best_order = ARIMA(y, order=(1, 0, 0)).fit(), (1, 0)

        self._result = best_res
        self._order = best_order
        self._model_name = f"ARMA({best_order[0]},{best_order[1]})"

        return ARMAResult(
            order=best_order,
            aic=best_res.aic,
            bic=best_res.bic,
            log_likelihood=best_res.llf,
            model_name=self._model_name,
        )

    def predict(self, steps: int = 1) -> np.ndarray:
        """h-step-ahead forecasts (iterated; in RV levels if use_log)."""
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(self._result.forecast(steps=steps), dtype=float)
        if self.use_log:
            sigma2 = float(self._result.mse)
            fc = np.exp(fc + 0.5 * sigma2)
        return np.asarray(fc, dtype=float)
