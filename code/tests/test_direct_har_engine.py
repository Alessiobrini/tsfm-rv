"""Tests for the direct-model engine (walk_forward_forecast) and the HAR-RS features.

Covers a rolling estimation window with no look-ahead, re-estimation at every origin, the
Bollerslev-Patton-Quaedvlieg insanity filter, and HAR-RS with only the daily lag split into
semivariances.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from features import build_har_rs_features, build_target, align_features_target  # noqa: E402
from forecasting.rolling_forecast import walk_forward_forecast  # noqa: E402


class _Recorder:
    """Model stub: records the rows it was fit on, predicts a fixed or scripted value."""
    fits = []

    def __init__(self, pred=None):
        self.pred = pred

    def fit(self, X, y):
        _Recorder.fits.append((X.index[0], X.index[-1], len(X), float(y.mean())))
        self._y = y
        return self

    def predict(self, X):
        val = self._y.iloc[-1] if self.pred is None else self.pred
        return pd.Series([val], index=X.index)


def _panel(n=60, horizon=1):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rv = pd.Series(np.arange(1, n + 1, dtype=float), index=idx)
    X = pd.DataFrame({"RV_d": rv.shift(1)}, index=idx)
    y = build_target(rv, horizon=horizon, target_kind="point")
    return align_features_target(X, y)


@pytest.fixture(autouse=True)
def _reset():
    _Recorder.fits = []


@pytest.mark.parametrize("h", [1, 5, 22])
def test_no_training_target_after_the_origin(h):
    X, y = _panel(n=100, horizon=h)
    walk_forward_forecast(X, y, lambda: _Recorder(), train_window=30, test_window=10,
                          step_size=10, horizon=h)
    pos = {d: k for k, d in enumerate(X.index)}
    # One fit per forecast origin, in order: fit j belongs to the j-th forecast row.
    first_test = 30
    for j, (_, last_row, _, _) in enumerate(_Recorder.fits):
        p = first_test + j
        assert pos[last_row] <= p - h, "a training row's target lies after the origin"
        assert pos[last_row] == p - h


def test_refit_every_origin_and_window_rolls():
    X, y = _panel(n=60, horizon=1)
    walk_forward_forecast(X, y, lambda: _Recorder(), train_window=20, test_window=10,
                          step_size=10, horizon=1, reestimate_every=1)
    firsts = [f[0] for f in _Recorder.fits]
    lengths = [f[2] for f in _Recorder.fits]
    assert len(_Recorder.fits) == len(X) - 20
    assert len(set(firsts)) == len(firsts), "window start must move at every origin"
    assert set(lengths) == {20}


def test_window_is_shorter_only_at_the_start_of_the_sample():
    X, y = _panel(n=60, horizon=5)
    walk_forward_forecast(X, y, lambda: _Recorder(), train_window=20, test_window=10,
                          step_size=10, horizon=5)
    lengths = [f[2] for f in _Recorder.fits]
    assert lengths[:4] == [16, 17, 18, 19]
    assert set(lengths[4:]) == {20}


def test_horizon_beyond_the_history_raises_instead_of_wrapping():
    X, y = _panel(n=60, horizon=22)
    with pytest.raises(ValueError, match="observed target"):
        walk_forward_forecast(X, y, lambda: _Recorder(), train_window=20, test_window=10,
                              step_size=10, horizon=22)


def test_first_window_is_clamped_at_the_start_of_the_sample():
    X, y = _panel(n=40, horizon=3)
    walk_forward_forecast(X, y, lambda: _Recorder(), train_window=20, test_window=5,
                          step_size=5, horizon=3)
    first_start, _, first_len, _ = _Recorder.fits[0]
    assert first_start == X.index[0]
    assert first_len == 18


def test_insanity_filter_replaces_out_of_range_with_window_mean():
    X, y = _panel(n=40, horizon=1)
    _, fc = walk_forward_forecast(X, y, lambda: _Recorder(pred=-5.0), train_window=20,
                                  test_window=10, step_size=10, horizon=1,
                                  insanity_filter=True)
    means = [f[3] for f in _Recorder.fits]
    np.testing.assert_allclose(fc.values, means)


def test_insanity_filter_keeps_in_range_forecasts_and_is_off_by_default():
    X, y = _panel(n=40, horizon=1)
    _, kept = walk_forward_forecast(X, y, lambda: _Recorder(), train_window=20,
                                    test_window=10, step_size=10, insanity_filter=True)
    _Recorder.fits = []
    _, raw = walk_forward_forecast(X, y, lambda: _Recorder(pred=-5.0), train_window=20,
                                   test_window=10, step_size=10)
    assert (raw.values == -5.0).all()
    # the stub predicts the last training target (row p - 1 at h = 1), inside the window's range
    lasts = [y.iloc[p - 1] for p in range(20, len(X))]
    np.testing.assert_allclose(kept.values, lasts)


def test_insanity_filter_boundary_values_are_kept():
    X, y = _panel(n=40, horizon=1)
    lo = float(y.iloc[0:20].min())
    _, fc = walk_forward_forecast(X, y, lambda: _Recorder(pred=lo), train_window=20,
                                  test_window=1, step_size=1, horizon=1,
                                  insanity_filter=True)
    assert fc.iloc[0] == lo


def test_output_dates_unchanged_by_horizon_argument():
    X, y = _panel(n=60, horizon=5)
    a1, _ = walk_forward_forecast(X, y, lambda: _Recorder(), train_window=20,
                                  test_window=10, step_size=10, horizon=1)
    a5, _ = walk_forward_forecast(X, y, lambda: _Recorder(), train_window=20,
                                  test_window=10, step_size=10, horizon=5)
    assert a1.index.equals(a5.index)


def test_har_rs_splits_only_the_daily_lag():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    good = pd.Series(np.linspace(1, 2, 40), index=idx)
    bad = pd.Series(np.linspace(3, 4, 40), index=idx)
    rv = good + bad
    f = build_har_rs_features(good, bad, rv)
    assert list(f.columns) == ["RS_pos_d", "RS_neg_d", "RV_w", "RV_m"]
    t = idx[30]
    assert f.loc[t, "RS_pos_d"] == good.shift(1).loc[t]
    assert f.loc[t, "RS_neg_d"] == bad.shift(1).loc[t]
    assert f.loc[t, "RV_w"] == pytest.approx(rv.iloc[25:30].mean())
    assert f.loc[t, "RV_m"] == pytest.approx(rv.iloc[8:30].mean())
    assert f.iloc[:22].isna().any(axis=1).all()


def test_har_rs_with_nan_input_propagates_nan():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    good = pd.Series(1.0, index=idx)
    bad = pd.Series(1.0, index=idx)
    bad.iloc[25] = np.nan
    f = build_har_rs_features(good, bad, good + bad)
    assert np.isnan(f["RS_neg_d"].iloc[26])
