"""models/mem.py — Multiplicative Error Model (Engle 2002) for realized measures.

Referee 1 (1.4.3) noted the MEM is a standard econometric model for volatility /
non-negative financial activity variables, is presented in the VOLARE paper
(Cipollini et al. 2026), and should either join the horse race or be explained.
We add it.

MEM(1,1):
    RV_t = mu_t * eps_t,   eps_t i.i.d. with E[eps_t] = 1,
    mu_t = omega + alpha * RV_{t-1} + beta * mu_{t-1}.

The conditional mean mu_t follows a GARCH-type recursion on RV *levels* (not
squared returns). Estimated by exponential QMLE (consistent for the conditional
mean regardless of the true error density; Engle 2002). With omega > 0 and
alpha, beta >= 0 the forecasts are strictly positive by construction, so the MEM
sidesteps the negative-forecast problem entirely (no flooring needed).

Multi-step forecast: mu_{T+1} = omega + alpha RV_T + beta mu_T, and for k >= 2,
mu_{T+k} = omega + (alpha + beta) mu_{T+k-1} (since E[RV_{T+j}] = mu_{T+j}).

Interface matches ARFIMAModel / ARMAModel: fit(series) / predict(steps).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MEMResult:
    """Container for MEM estimation results."""
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    model_name: str


class MEMModel:
    """MEM(1,1) estimated by exponential QMLE. Forecasts are strictly positive."""

    def __init__(self, max_persistence: float = 0.9999):
        self.max_persistence = max_persistence
        self._params = None       # (omega, alpha, beta)
        self._last_rv = None
        self._last_mu = None
        self._model_name = "MEM"

    @staticmethod
    def _filter(rv: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
        """Run the conditional-mean recursion; seed mu_0 at the sample mean."""
        n = len(rv)
        mu = np.empty(n)
        mu[0] = rv.mean()
        for t in range(1, n):
            mu[t] = omega + alpha * rv[t - 1] + beta * mu[t - 1]
        return mu

    def _negloglik(self, theta: np.ndarray, rv: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= self.max_persistence:
            return 1e12
        mu = np.maximum(self._filter(rv, omega, alpha, beta), 1e-12)
        # Exponential QMLE negative log-likelihood (up to a constant): sum log(mu) + RV/mu.
        return float(np.sum(np.log(mu) + rv / mu))

    def fit(self, series: pd.Series) -> MEMResult:
        from scipy.optimize import minimize

        rv = np.maximum(np.asarray(series.dropna(), dtype=float), 1e-12)
        m = rv.mean()

        # Sensible GARCH-like start: moderate alpha, high beta, omega = m*(1-persistence).
        x0 = np.array([m * 0.05, 0.10, 0.85])
        bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]
        res = minimize(self._negloglik, x0, args=(rv,), method="L-BFGS-B", bounds=bounds)

        omega, alpha, beta = res.x
        self._params = (float(omega), float(alpha), float(beta))
        mu = self._filter(rv, omega, alpha, beta)
        self._last_rv = float(rv[-1])
        self._last_mu = float(mu[-1])

        return MEMResult(
            omega=float(omega), alpha=float(alpha), beta=float(beta),
            log_likelihood=float(-res.fun), model_name=self._model_name,
        )

    def predict(self, steps: int = 1) -> np.ndarray:
        if self._params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        omega, alpha, beta = self._params
        persist = alpha + beta

        out = [omega + alpha * self._last_rv + beta * self._last_mu]  # mu_{T+1}
        for _ in range(1, steps):
            out.append(omega + persist * out[-1])
        return np.asarray(out[:steps], dtype=float)
