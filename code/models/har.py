"""
models/har.py — HAR family of realized volatility models.

Implements:
    - HAR       (Corsi 2009)
    - HAR-J     (Andersen, Bollerslev & Diebold 2007)
    - HAR-RS    (Patton & Sheppard 2015)
    - HARQ      (Bollerslev, Patton & Quaedvlieg 2016)
    - Log-HAR   (HAR on log(RV) with bias correction)

All models are estimated via OLS (statsmodels) with optional Newey-West HAC
standard errors. Each model class follows a fit/predict interface.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class HARResult:
    """Container for HAR estimation results.

    Attributes:
        params: Estimated coefficients.
        std_errors: Standard errors (HAC or OLS).
        t_stats: t-statistics.
        r_squared: In-sample R².
        residuals: In-sample residuals.
        model_name: String identifier.
    """
    params: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    r_squared: float
    residuals: pd.Series
    model_name: str


class HARModel:
    """Standard HAR model (Corsi 2009).

    RV_{t+h} = β₀ + β₁·RV_d + β₂·RV_w + β₃·RV_m + ε_t

    Parameters
    ----------
    use_hac : bool
        Use Newey-West HAC standard errors.
    hac_max_lags : int
        Maximum lags for HAC.
    use_log : bool
        If True, fit on log(RV) — Log-HAR variant.
    """

    def __init__(
        self,
        use_hac: bool = True,
        hac_max_lags: int = 22,
        use_log: bool = False,
        constrained: bool = False,
    ):
        self.use_hac = use_hac
        self.hac_max_lags = hac_max_lags
        self.use_log = use_log
        # Nelson & Cao (1992)-style non-negativity: constrain the intercept and
        # lag coefficients to be >= 0 so that, with non-negative regressors, the
        # level HAR forecast is guaranteed non-negative (Referee 1 1.4.1 — the
        # 1e-10 flooring was "unacceptable"; the fix is in the application, not a
        # floor). Ignored when use_log (Log-HAR is positive by construction).
        self.constrained = constrained
        self._ols_result = None        # statsmodels result (unconstrained fits)
        self._params = None            # pd.Series of coefficients (both paths)
        self._resid_var = None         # residual variance (for log bias correction)
        self._constrained_fit = False  # True if this fit used the NNLS path
        self._model_name = "Log-HAR" if use_log else "HAR"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HARResult:
        """Estimate HAR model via OLS.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix from features.build_har_features().
        y : pd.Series
            Target: RV_{t+h} or log(RV_{t+h}).

        Returns
        -------
        HARResult
            Estimation results.
        """
        if self.use_log:
            y = np.log(y)
            X = np.log(X.clip(lower=1e-10))

        X_const = sm.add_constant(X, has_constant='add')

        if self.constrained and not self.use_log:
            # Non-negative least squares: intercept + all lag coefficients >= 0,
            # guaranteeing a non-negative forecast for non-negative regressors.
            from scipy.optimize import nnls
            A = X_const.values.astype(float)
            b = np.asarray(y, dtype=float)
            coef, _ = nnls(A, b)
            self._params = pd.Series(coef, index=X_const.columns)
            resid = b - A @ coef
            n, k = A.shape
            self._resid_var = float(np.sum(resid ** 2) / max(n - k, 1))
            ss_tot = float(np.sum((b - b.mean()) ** 2))
            r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
            self._ols_result = None
            self._constrained_fit = True
            nan = pd.Series(np.nan, index=self._params.index)  # HAC SEs N/A under NNLS
            return HARResult(
                params=self._params, std_errors=nan, t_stats=nan,
                r_squared=r2, residuals=pd.Series(resid, index=X_const.index),
                model_name=self._model_name,
            )

        if self.use_hac:
            self._ols_result = sm.OLS(y, X_const).fit(
                cov_type='HAC', cov_kwds={'maxlags': self.hac_max_lags}
            )
        else:
            self._ols_result = sm.OLS(y, X_const).fit()

        self._constrained_fit = False
        self._params = self._ols_result.params
        self._resid_var = float(self._ols_result.mse_resid)

        return HARResult(
            params=self._ols_result.params,
            std_errors=self._ols_result.bse,
            t_stats=self._ols_result.tvalues,
            r_squared=self._ols_result.rsquared,
            residuals=self._ols_result.resid,
            model_name=self._model_name,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate point forecasts from fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for forecast dates.

        Returns
        -------
        pd.Series
            Point forecasts of RV (or log-RV if use_log=True).
        """
        if self._params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.use_log:
            X = np.log(X.clip(lower=1e-10))

        X_const = sm.add_constant(X, has_constant='add')

        if self._constrained_fit:
            # Manual linear prediction from the NNLS coefficients.
            cols = [c for c in self._params.index]
            pred = X_const[cols].values @ self._params.values
            pred = pd.Series(pred, index=X.index)
        else:
            pred = self._ols_result.predict(X_const)

        if self.use_log:
            # Bias correction: E[RV] = exp(log_pred + 0.5 * sigma²)
            sigma2 = self._resid_var
            pred = np.exp(pred + 0.5 * sigma2)

        return pd.Series(pred, index=X.index, name=f'{self._model_name}_forecast')

    def linear_coef(self) -> Tuple[float, float, float, float, bool, float]:
        """Return (intercept, b_d, b_w, b_m, use_log, resid_var) for the fitted
        pure-HAR model. Used by the iterated multi-step forecaster, which needs
        the coefficients in closed form regardless of OLS vs. NNLS fit. Valid
        only for the 3-regressor HAR / Log-HAR (RV_d, RV_w, RV_m)."""
        if self._params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        p = self._params
        return (
            float(p.get('const', 0.0)),
            float(p['RV_d']), float(p['RV_w']), float(p['RV_m']),
            bool(self.use_log),
            float(self._resid_var) if self.use_log else 0.0,
        )


class HARJModel(HARModel):
    """HAR-J model: HAR + jump component.

    RV_{t+h} = β₀ + β₁·RV_d + β₂·RV_w + β₃·RV_m + β₄·J_d + ε_t
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = "HAR-J"


class HARRSModel(HARModel):
    """HAR-RS model: semivariance decomposition.

    RV_{t+h} = β₀ + β₁⁺·RS⁺_d + β₁⁻·RS⁻_d + β₂⁺·RS⁺_w + β₂⁻·RS⁻_w
                   + β₃⁺·RS⁺_m + β₃⁻·RS⁻_m + ε_t
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = "HAR-RS"


class HARQModel(HARModel):
    """HARQ model: HAR + realized quarticity interaction.

    RV_{t+h} = β₀ + β₁·RV_d + β₁Q·(RV_d·√RQ_d) + β₂·RV_w + β₃·RV_m + ε_t
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = "HARQ"


def get_har_model(model_name: str, **kwargs) -> HARModel:
    """Factory function to get HAR model variant by name.

    Parameters
    ----------
    model_name : str
        One of: 'HAR', 'HAR-J', 'HAR-RS', 'HARQ', 'Log-HAR'.

    Returns
    -------
    HARModel
        Instantiated model.
    """
    models = {
        'HAR': HARModel,
        'HAR-J': HARJModel,
        'HAR-RS': HARRSModel,
        'HARQ': HARQModel,
        'Log-HAR': lambda **kw: HARModel(use_log=True, **kw),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    constructor = models[model_name]
    return constructor(**kwargs)
