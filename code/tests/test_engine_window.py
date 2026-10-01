"""Tests for the estimation window of the series engine (ARFIMA/ARMA/MEM) and the
iterated HAR engine (HAR/Log-HAR).

Each fit at origin i uses exactly the train_window most recent observations, rows
i - train_window to i - 1, so the window slides with the origin, and no value dated
at or after the origin changes the forecast made at that origin.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from forecasting.rolling_forecast import (  # noqa: E402
    walk_forward_series_forecast, iterated_har_forecast,
)
from models.har import HARModel  # noqa: E402
from models.mem import MEMModel  # noqa: E402


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    x = np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.2 + rng.normal(0, 0.2, n))
    return pd.Series(0.01 * x, index=idx, name="sigma")


class _SeriesRecorder:
    fits = []

    def fit(self, series):
        _SeriesRecorder.fits.append((series.index[0], series.index[-1], len(series)))

    def predict(self, steps):
        return np.ones(steps)


class _HARRecorder:
    fits = []

    def fit(self, X, y):
        _HARRecorder.fits.append((X.index[0], X.index[-1], len(X), y.index[-1], float(y.iloc[-1])))

    def linear_coef(self):
        return (0.0, 1.0, 0.0, 0.0, False, 0.0)


@pytest.fixture(autouse=True)
def _reset():
    _SeriesRecorder.fits = []
    _HARRecorder.fits = []


@pytest.mark.parametrize("h", [1, 5, 22])
def test_series_engine_window_slides_and_ends_before_origin(h):
    s = _series(400)
    W = 100
    actual, _ = walk_forward_series_forecast(
        s, _SeriesRecorder, train_window=W, test_window=40, step_size=40,
        horizon=h, reestimate_every=1, target_kind="point")
    pos = {d: k for k, d in enumerate(s.index)}
    origins = [pos[d] for d in actual.index]
    fits = _SeriesRecorder.fits
    # one fit per recorded origin, plus fits at origins whose target is beyond the sample
    assert len(fits) >= len(origins)
    for first, last, n in fits:
        assert n == W
        assert pos[first] == pos[last] - W + 1
    fit_by_end = {pos[last] + 1: (pos[first], n) for first, last, n in fits}
    for i in origins:
        assert i in fit_by_end, "no fit ending the day before this origin"
        assert fit_by_end[i] == (i - W, W)


def test_series_engine_no_longer_grows_within_a_fold():
    s = _series(400)
    walk_forward_series_forecast(s, _SeriesRecorder, train_window=100, test_window=40,
                                 step_size=40, horizon=1, reestimate_every=1)
    firsts = [f[0] for f in _SeriesRecorder.fits]
    assert len(set(firsts)) == len(firsts), "window start must move at every refit"
    assert {f[2] for f in _SeriesRecorder.fits} == {100}


@pytest.mark.parametrize("h", [1, 5, 22])
def test_iterated_har_window_slides_and_targets_are_observed(h):
    s = _series(400)
    W = 120
    actual, _ = iterated_har_forecast(
        s, _HARRecorder, horizon=h, train_window=W, test_window=40, step_size=40,
        reestimate_every=1, target_kind="point")
    pos = {d: k for k, d in enumerate(s.index)}
    fit_by_end = {pos[f[1]] + 1: f for f in _HARRecorder.fits}
    for d in actual.index:
        i = pos[d]
        first, last, n, y_last_date, y_last = fit_by_end[i]
        # 22 of the W rows seed the monthly lag, so W - 22 regression rows
        assert n == W - 22
        assert pos[first] == i - W + 22
        assert pos[y_last_date] == i - 1
        assert y_last == pytest.approx(s.iloc[i - 1])


def _corrupt_from(s, i):
    c = s.copy()
    c.iloc[i:] = c.iloc[i:] * 7.0 + 0.3
    return c


@pytest.mark.parametrize("h", [1, 5, 22])
def test_log_har_forecast_at_origin_ignores_data_from_origin_on(h):
    s = _series(360, seed=1)
    kw = dict(horizon=h, train_window=150, test_window=50, step_size=50,
              reestimate_every=1, target_kind="point")
    _, base = iterated_har_forecast(s, lambda: HARModel(use_log=True), **kw)
    for d in [base.index[0], base.index[len(base) // 2], base.index[-1]]:
        i = s.index.get_loc(d)
        _, alt = iterated_har_forecast(_corrupt_from(s, i), lambda: HARModel(use_log=True), **kw)
        assert alt.loc[d] == pytest.approx(base.loc[d], rel=1e-12)


@pytest.mark.parametrize("h", [1, 5])
def test_mem_forecast_at_origin_ignores_data_from_origin_on(h):
    s = _series(300, seed=2)
    kw = dict(train_window=120, test_window=40, step_size=40, horizon=h,
              reestimate_every=1, target_kind="point")
    _, base = walk_forward_series_forecast(s, MEMModel, **kw)
    for d in [base.index[0], base.index[-1]]:
        i = s.index.get_loc(d)
        _, alt = walk_forward_series_forecast(_corrupt_from(s, i), MEMModel, **kw)
        assert alt.loc[d] == pytest.approx(base.loc[d], rel=1e-10)


def test_output_dates_unchanged_by_the_window_fix():
    """Origins depend only on the fold layout and the horizon, not on the window."""
    s = _series(400)
    a, _ = walk_forward_series_forecast(s, _SeriesRecorder, train_window=100, test_window=40,
                                        step_size=40, horizon=5, reestimate_every=1)
    pos = [s.index.get_loc(d) for d in a.index]
    assert pos[0] == 100
    assert pos == list(range(100, 400 - 5 + 1))
