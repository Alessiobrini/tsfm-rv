"""Unit tests for the point-in-time vs. average forecast alignment in the
zero-shot (TSFM) and series (ARFIMA/ARMA/MEM) walk-forward engines.

These pin down that:
  * "point" records the h-th rollout step against RV_{i+h-1},
  * "avg"   records the mean of the first h steps against the h-day average,
  * the origin-date convention is identical across engines (so models align
    by date for DM / MCS).

Run: pytest code/tests/test_rolling_forecast.py -q
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from forecasting.rolling_forecast import (
    zero_shot_forecast,
    walk_forward_series_forecast,
    iterated_har_forecast,
)
from models.har import HARModel


# --------------------------------------------------------------------------
# zero_shot_forecast (TSFM path)
# --------------------------------------------------------------------------
class _Forecast:
    def __init__(self, point):
        self.point = point


class _RampTSFM:
    """Returns point[k] = BASE + k, so the h-th step (k=h-1) is identifiable
    and distinct from the 1-step (k=0)."""
    BASE = 1000.0

    def predict(self, context, horizon):
        return _Forecast(np.array([self.BASE + k for k in range(horizon)], dtype=float))


def _ramp_series(n=40):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.arange(n, dtype=float), index=idx, name="rv")  # value == position


def test_zero_shot_point_takes_hth_step_and_future_actual():
    rv = _ramp_series(40)
    ctx = 10
    h = 5
    actual, forecast = zero_shot_forecast(rv, _RampTSFM(), horizon=h, context_length=ctx,
                                          target_kind="point")
    # Forecast is constant at the h-th rollout step (BASE + h - 1).
    assert np.allclose(forecast.values, _RampTSFM.BASE + h - 1)
    # Actual at origin date i is RV_{i+h-1}; since value == position, it equals i+h-1.
    for date, a in actual.items():
        i = rv.index.get_loc(date)
        assert a == i + h - 1
    # Origins run from ctx up to the last with a realized target (n-1).
    assert rv.index.get_loc(actual.index[0]) == ctx
    assert rv.index.get_loc(actual.index[-1]) == len(rv) - h  # i+h-1 == n-1


def test_zero_shot_h1_reduces_to_one_step():
    rv = _ramp_series(20)
    actual, forecast = zero_shot_forecast(rv, _RampTSFM(), horizon=1, context_length=5,
                                          target_kind="point")
    assert np.allclose(forecast.values, _RampTSFM.BASE)        # point[0]
    for date, a in actual.items():
        assert a == rv.index.get_loc(date)                     # RV_i


def test_zero_shot_avg_averages_steps_and_window():
    rv = _ramp_series(40)
    h = 5
    actual, forecast = zero_shot_forecast(rv, _RampTSFM(), horizon=h, context_length=10,
                                          target_kind="avg")
    # Forecast = mean(BASE .. BASE+h-1) = BASE + (h-1)/2.
    assert np.allclose(forecast.values, _RampTSFM.BASE + (h - 1) / 2)
    # Actual = mean(RV_i .. RV_{i+h-1}) = i + (h-1)/2.
    for date, a in actual.items():
        i = rv.index.get_loc(date)
        assert a == pytest.approx(i + (h - 1) / 2)


def test_zero_shot_rejects_bad_kind():
    with pytest.raises(ValueError):
        zero_shot_forecast(_ramp_series(20), _RampTSFM(), 1, 5, target_kind="x")


# --------------------------------------------------------------------------
# walk_forward_series_forecast (ARFIMA/ARMA/MEM path)
# --------------------------------------------------------------------------
class _RampSeriesModel:
    """fit(series) is a no-op; predict(steps) returns BASE + k ramp."""
    BASE = 2000.0

    def fit(self, series):
        self._n = len(series)

    def predict(self, steps):
        return np.array([self.BASE + k for k in range(steps)], dtype=float)


# train_window >= 60 so the engine's internal `len(fit_series) < 50` guard
# (intended for real ARFIMA estimation) never trips in these tests.
def test_series_point_records_hth_step_against_future_actual():
    rv = _ramp_series(160)  # value == position
    h = 3
    actual, forecast = walk_forward_series_forecast(
        rv, _RampSeriesModel, train_window=60, test_window=20, step_size=20,
        horizon=h, reestimate_every=5, target_kind="point",
    )
    assert len(actual) > 0
    # Forecast constant at h-th step; actual is RV_{i+h-1} == position+h-1.
    assert np.allclose(forecast.values, _RampSeriesModel.BASE + h - 1)
    for date, a in actual.items():
        i = rv.index.get_loc(date)
        assert a == i + h - 1


def test_series_avg_records_window_mean():
    rv = _ramp_series(160)
    h = 5
    actual, forecast = walk_forward_series_forecast(
        rv, _RampSeriesModel, train_window=60, test_window=20, step_size=20,
        horizon=h, reestimate_every=5, target_kind="avg",
    )
    assert len(actual) > 0
    assert np.allclose(forecast.values, _RampSeriesModel.BASE + (h - 1) / 2)
    for date, a in actual.items():
        i = rv.index.get_loc(date)
        assert a == pytest.approx(i + (h - 1) / 2)


def test_series_h1_matches_origin_value():
    rv = _ramp_series(160)
    actual, forecast = walk_forward_series_forecast(
        rv, _RampSeriesModel, train_window=60, test_window=20, step_size=20,
        horizon=1, reestimate_every=5, target_kind="point",
    )
    assert len(actual) > 0
    assert np.allclose(forecast.values, _RampSeriesModel.BASE)
    for date, a in actual.items():
        assert a == rv.index.get_loc(date)


# --------------------------------------------------------------------------
# iterated_har_forecast (pure HAR / Log-HAR recursive plug-in)
# --------------------------------------------------------------------------
class _FakeOLS:
    def __init__(self, params, mse_resid=0.0):
        self.params = params
        self.mse_resid = mse_resid


class _FakeHAR:
    """1-step HAR with KNOWN coefficients: RV_t = b0 + bd*RV_d + bw*RV_w + bm*RV_m."""
    B0, BD, BW, BM = 0.001, 0.5, 0.3, 0.15
    use_log = False

    def fit(self, X, y):
        self._ols_result = _FakeOLS(
            pd.Series({"const": self.B0, "RV_d": self.BD, "RV_w": self.BW, "RV_m": self.BM}),
            mse_resid=0.0,
        )
        return self


def _expected_rollout(vals, i, h, weekly=5, monthly=22):
    """Independent reimplementation of the recursion to cross-check alignment."""
    work = list(vals[i - monthly:i])
    preds = []
    for _ in range(h):
        rv_d = work[-1]
        rv_w = float(np.mean(work[-weekly:]))
        rv_m = float(np.mean(work[-monthly:]))
        nxt = _FakeHAR.B0 + _FakeHAR.BD * rv_d + _FakeHAR.BW * rv_w + _FakeHAR.BM * rv_m
        work.append(nxt)
        preds.append(nxt)
    return preds


def _positive_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(rng.uniform(0.05, 1.0, n), index=idx, name="rv")


@pytest.mark.parametrize("h", [1, 5, 22])
def test_iterated_har_recursion_matches_hand_rollout(h):
    rv = _positive_series(320)
    vals = rv.values.astype(float)
    actual, forecast = iterated_har_forecast(
        rv, _FakeHAR, horizon=h, train_window=120, test_window=40, step_size=40,
        reestimate_every=10, target_kind="point",
    )
    assert len(actual) > 0
    for date in forecast.index:
        i = rv.index.get_loc(date)
        exp = _expected_rollout(vals, i, h)
        assert forecast.loc[date] == pytest.approx(exp[h - 1])   # h-th rollout step
        assert actual.loc[date] == pytest.approx(vals[i + h - 1])  # RV_{i+h-1}


def test_iterated_har_h1_is_one_step_linear():
    rv = _positive_series(300)
    vals = rv.values.astype(float)
    actual, forecast = iterated_har_forecast(
        rv, _FakeHAR, horizon=1, train_window=120, test_window=40, step_size=40,
        reestimate_every=10, target_kind="point",
    )
    for date in forecast.index:
        i = rv.index.get_loc(date)
        exp = (_FakeHAR.B0 + _FakeHAR.BD * vals[i - 1]
               + _FakeHAR.BW * np.mean(vals[i - 5:i])
               + _FakeHAR.BM * np.mean(vals[i - 22:i]))
        assert forecast.loc[date] == pytest.approx(exp)
        assert actual.loc[date] == pytest.approx(vals[i])  # RV_i at h=1


def test_iterated_log_har_forecasts_are_positive():
    """Real Log-HAR rollout: exp(.) guarantees strictly positive forecasts —
    the property that fixes Referee 1's negative-prediction / flooring concern."""
    rv = _positive_series(320, seed=3)
    log_har_factory = lambda: HARModel(use_log=True)
    actual, forecast = iterated_har_forecast(
        rv, log_har_factory, horizon=5, train_window=150, test_window=40, step_size=40,
        reestimate_every=22, target_kind="point",
    )
    assert len(forecast) > 0
    assert (forecast.values > 0).all()
    # Dates are a subset of the series and align with actuals.
    assert forecast.index.equals(actual.index)
    assert set(forecast.index).issubset(set(rv.index))


def test_iterated_har_rejects_bad_kind():
    with pytest.raises(ValueError):
        iterated_har_forecast(_positive_series(80), _FakeHAR, 1, target_kind="x")
