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
    ):
        self.use_hac = use_hac
        self.hac_max_lags = hac_max_lags
        self.use_log = use_log
        self._ols_result = None
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

        if self.use_hac:
            self._ols_result = sm.OLS(y, X_const).fit(
                cov_type='HAC', cov_kwds={'maxlags': self.hac_max_lags}
            )
        else:
            self._ols_result = sm.OLS(y, X_const).fit()

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
        if self._ols_result is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.use_log:
            X = np.log(X.clip(lower=1e-10))

        X_const = sm.add_constant(X, has_constant='add')
        pred = self._ols_result.predict(X_const)

        if self.use_log:
            # Bias correction: E[RV] = exp(log_pred + 0.5 * sigma²)
            sigma2 = self._ols_result.mse_resid
            pred = np.exp(pred + 0.5 * sigma2)

        return pd.Series(pred, index=X.index, name=f'{self._model_name}_forecast')


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
