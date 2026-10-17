"""Unit tests for the econometric benchmarks added/changed in Workstream B:
constrained-OLS HAR (positivity), ARMA-on-log-RV (selection), MEM (Engle 2002).

Run: pytest code/tests/test_models.py -q
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from features import build_har_features, build_target, align_features_target
from models.har import HARModel
from models.arma import ARMAModel
from models.mem import MEMModel
from models.arfima import ARFIMAModel


def _rv_series(n=400, seed=0):
    """Persistent, strictly positive RV-like series (AR(1) in logs)."""
    rng = np.random.default_rng(seed)
    log_rv = np.empty(n)
    log_rv[0] = -8.0
    for t in range(1, n):
        log_rv[t] = -8.0 * 0.05 + 0.95 * log_rv[t - 1] + rng.normal(0, 0.3)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(np.exp(log_rv), index=idx, name="rv")


# --------------------------------------------------------------------------
# Constrained-OLS HAR (Nelson-Cao positivity)
# --------------------------------------------------------------------------
def _har_Xy(rv, horizon=1):
    feats = build_har_features(rv)
    tgt = build_target(rv, horizon=horizon, target_kind="point")
    return align_features_target(feats, tgt)


def test_constrained_har_params_nonnegative_and_positive_forecasts():
    rv = _rv_series()
    X, y = _har_Xy(rv)
    m = HARModel(constrained=True)
    res = m.fit(X, y)
    # All coefficients (intercept + lags) >= 0.
    assert (res.params.values >= -1e-9).all()
    # With non-negative regressors, forecasts are non-negative.
    preds = m.predict(X)
    assert (preds.values >= 0).all()


def test_constrained_har_linear_coef_roundtrip():
    rv = _rv_series()
    X, y = _har_Xy(rv)
    m = HARModel(constrained=True)
    m.fit(X, y)
    b0, bd, bw, bm, use_log, sigma2 = m.linear_coef()
    assert use_log is False
    # linear_coef must reproduce the model's own prediction on a row.
    row = X.iloc[[10]]
    manual = b0 + bd * row["RV_d"].iloc[0] + bw * row["RV_w"].iloc[0] + bm * row["RV_m"].iloc[0]
    assert m.predict(row).iloc[0] == pytest.approx(manual, rel=1e-9)


def test_unconstrained_har_still_works_and_log_har_positive():
    rv = _rv_series()
    X, y = _har_Xy(rv)
    HARModel(constrained=False).fit(X, y)  # OLS path still fine
    log_har = HARModel(use_log=True)
    log_har.fit(X, y)
    assert (log_har.predict(X).values > 0).all()  # exp() => strictly positive
    b0, bd, bw, bm, use_log, sigma2 = log_har.linear_coef()
    assert use_log is True and sigma2 >= 0


# --------------------------------------------------------------------------
# ARMA on log-RV
# --------------------------------------------------------------------------
def test_arma_fits_selects_order_and_forecasts_positive():
    rv = _rv_series()
    m = ARMAModel(max_p=2, max_q=2, use_log=True, ic="bic")
    res = m.fit(rv)
    p, q = res.order
    assert 0 <= p <= 2 and 0 <= q <= 2 and (p, q) != (0, 0)
    fc = m.predict(steps=5)
    assert fc.shape == (5,)
    assert (fc > 0).all()  # exponentiated => positive


def test_arma_rejects_bad_ic():
    with pytest.raises(ValueError):
        ARMAModel(ic="xic")


# --------------------------------------------------------------------------
# MEM (Engle 2002)
# --------------------------------------------------------------------------
def test_mem_fits_valid_params_and_positive_forecasts():
    rv = _rv_series()
    m = MEMModel()
    res = m.fit(rv)
    assert res.omega > 0 and res.alpha >= 0 and res.beta >= 0
    assert res.alpha + res.beta < 1.0  # stationarity
    fc = m.predict(steps=10)
    assert fc.shape == (10,)
    assert (fc > 0).all()


def test_arfima_whittle_d_and_forecast():
    rv = _rv_series()
    m = ARFIMAModel(p=2, q=2, use_log=True, d_method="whittle", select=True)
    res = m.fit(rv)
    assert -0.5 < res.d < 0.5                      # valid memory parameter
    assert 0 <= m._order[0] <= 2 and 0 <= m._order[1] <= 2
    fc = m.predict(steps=5)
    assert fc.shape == (5,) and (fc > 0).all()     # exponentiated => positive


def test_arfima_mle_alias_maps_to_whittle():
    m = ARFIMAModel(d_method="mle")
    assert m.d_method == "whittle"


def test_arfima_distinct_from_arma():
    """Genuine fractional forecasting must make ARFIMA differ from plain ARMA on
    log-RV (the earlier bug had them byte-identical because d was unused)."""
    rv = _rv_series(seed=7)
    af = ARFIMAModel(d_method="whittle")
    af.fit(rv)
    fc_af = af.predict(steps=22)
    am = ARMAModel()
    am.fit(rv)
    fc_am = am.predict(steps=22)
    assert np.isfinite(fc_af).all() and (fc_af > 0).all()
    assert af._w is not None and af._psi is not None        # fractional machinery populated
    assert np.max(np.abs(fc_af - fc_am)) > 1e-9             # not identical to ARMA


def test_mem_multistep_recursion():
    rv = _rv_series()
    m = MEMModel()
    m.fit(rv)
    fc = m.predict(steps=4)
    omega, alpha, beta = m._params
    persist = alpha + beta
    # From step 2 on, mu_{T+k} = omega + persist * mu_{T+k-1}.
    for k in range(1, 4):
        assert fc[k] == pytest.approx(omega + persist * fc[k - 1], rel=1e-9)
