"""Tests for the origin-date winsorization bounds and the recursive MZ correction.

Both must use only information observed by the forecast origin.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from forecasting.bounds import window_bounds, clip_to_window  # noqa: E402
from evaluation.mz_regression import recursive_mz_correction  # noqa: E402


def _sigma(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-02", periods=n, freq="B")
    return pd.Series(np.exp(rng.normal(-4.5, 0.4, n)), index=idx)


@pytest.mark.parametrize("w", [512, 1000])
def test_bounds_are_the_range_of_the_window_before_each_date(w):
    s = _sigma()
    lo, hi = window_bounds(s, w)
    for p in [1, 10, w - 1, w, w + 7, len(s) - 1]:
        past = s.iloc[max(0, p - w):p]
        assert lo.iloc[p] == past.min() and hi.iloc[p] == past.max()
    assert np.isnan(lo.iloc[0]) and np.isnan(hi.iloc[0])


def test_bounds_at_a_date_ignore_data_from_that_date_on():
    s = _sigma()
    p = 1200
    c = s.copy()
    c.iloc[p:] = 5.0
    c.iloc[p + 3] = 1e-12
    a, b = window_bounds(s, 1000), window_bounds(c, 1000)
    assert a[0].iloc[p] == b[0].iloc[p] and a[1].iloc[p] == b[1].iloc[p]


def test_zero_values_are_left_out_of_the_lower_bound():
    s = _sigma(50)
    s.iloc[5] = 0.0
    lo, _ = window_bounds(s, 1000)
    assert lo.iloc[6] == s.iloc[:6][s.iloc[:6] > 0].min()
    assert (lo.dropna() > 0).all()


def test_clip_to_window_clips_each_forecast_to_its_own_bounds():
    s = _sigma()
    dates = s.index[1000:1010]
    f = pd.Series(np.r_[np.full(5, 10.0), np.full(5, 1e-9)], index=dates)
    lo, hi = window_bounds(s, 1000)
    out = clip_to_window(f, s, 1000)
    np.testing.assert_array_equal(out.values[:5], hi.loc[dates[:5]].values)
    np.testing.assert_array_equal(out.values[5:], lo.loc[dates[5:]].values)


def test_clip_to_window_rejects_dates_without_history():
    s = _sigma(100)
    with pytest.raises(ValueError):
        clip_to_window(pd.Series([0.01], index=[s.index[0]]), s)
    with pytest.raises(ValueError):
        clip_to_window(pd.Series([0.01], index=[pd.Timestamp("1990-01-01")]), s)


def test_window_clip_of_a_full_sample_clipped_forecast_equals_clip_of_the_raw_one():
    s = _sigma()
    rng = np.random.default_rng(3)
    dates = s.index[1000:]
    raw = pd.Series(np.exp(rng.normal(-4.5, 1.5, len(dates))), index=dates)
    full = raw.clip(lower=s.min(), upper=s.max())
    pd.testing.assert_series_equal(clip_to_window(full, s), clip_to_window(raw, s))


def _mz_data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    f = np.exp(rng.normal(-4.5, 0.3, n))
    y = 0.001 + 0.9 * f + rng.normal(0, 0.002, n)
    return y, f


@pytest.mark.parametrize("h", [1, 5, 22])
def test_mz_correction_at_t_ignores_targets_realized_after_t(h):
    y, f = _mz_data()
    w = 252
    base = recursive_mz_correction(y, f, min_window=w, horizon=h)
    for t in [w, w + 50, len(y) - 1]:
        c = y.copy()
        c[t - h + 1:] = 99.0           # targets not observed by origin t
        alt = recursive_mz_correction(c, f, min_window=w, horizon=h)
        assert alt[t - w] == pytest.approx(base[t - w], rel=1e-12)
        if h > 1:
            c2 = y.copy()
            c2[t - h] = 99.0           # the last target observed by origin t does matter
            alt2 = recursive_mz_correction(c2, f, min_window=w, horizon=h)
            assert alt2[t - w] != pytest.approx(base[t - w], rel=1e-9)


def test_mz_correction_at_h1_uses_every_earlier_row():
    y, f = _mz_data()
    out = recursive_mz_correction(y, f, min_window=252, horizon=1)
    t = 300
    X = np.c_[np.ones(t), f[:t]]
    a, b = np.linalg.lstsq(X, y[:t], rcond=None)[0]
    assert out[t - 252] == pytest.approx(a + b * f[t], rel=1e-9)
