"""Helpers behind the appendix and robustness tables: real-time rescaling, the equal-weight panel
series, the log retransformation, the realized utility and the AR(22) weights of Log-HAR."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compute_economic_value as ev  # noqa: E402
from compute_log_input import back_to_vol, clip_log, past_error_variance  # noqa: E402
from evaluation.forecast_sets import common_index, equal_weight_series, rescale_level  # noqa: E402
from refit_lognormal_moments import ar22_psi  # noqa: E402


def _file(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"actual": rng.lognormal(-4, 0.3, n), "forecast": rng.lognormal(-4, 0.3, n)},
                        index=idx)


@pytest.mark.parametrize("h", [1, 5, 22])
def test_rescale_level_uses_only_targets_observed_by_the_origin(h):
    d = _file()
    base = rescale_level(d, h, window=10, burn=3)
    p = 40
    late = d.copy()
    late.iloc[p - h + 1:, late.columns.get_loc("actual")] *= 7.0     # targets not yet observed at p
    assert base.iloc[p] == pytest.approx(rescale_level(late, h, window=10, burn=3).iloc[p], rel=1e-14)
    early = d.copy()
    early.iloc[p - h, early.columns.get_loc("actual")] *= 7.0        # the last observed target
    assert base.iloc[p] != pytest.approx(rescale_level(early, h, window=10, burn=3).iloc[p], rel=1e-6)


def test_rescale_level_leaves_the_burn_in_unchanged():
    d = _file()
    out = rescale_level(d, 5, window=10, burn=3)
    F = d["forecast"].values ** 2
    assert np.allclose(out.values[:7], F[:7], rtol=0, atol=0)        # fewer than 3 observed rows
    assert not np.isclose(out.values[7], F[7])


def test_equal_weight_series_sums_to_the_mean_of_asset_means():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series(np.arange(10, dtype=float), index=idx)
    b = pd.Series([5.0, 7.0, 9.0], index=idx[-3:])
    s = equal_weight_series(pd.DataFrame({"a": a, "b": b}))
    assert s.sum() == pytest.approx(0.5 * (a.mean() + b.mean()))
    assert len(s) == 10


def test_common_index_by_column_and_by_row():
    idx = pd.bdate_range("2020-01-01", periods=5)
    x = pd.DataFrame({"actual": [1.0, np.nan, 1, 1, 1], "forecast": [1.0, 1, 1, np.nan, 1]}, index=idx)
    y = pd.DataFrame({"actual": [1.0] * 5, "forecast": [np.nan, 1, 1, 1, 1]}, index=idx)
    assert list(common_index({"x": x, "y": y}, "forecast")) == list(idx[[1, 2, 4]])
    assert list(common_index({"x": x, "y": y})) == list(idx[[2, 4]])


def test_past_error_variance_excludes_the_current_error():
    d = _file(n=40)
    v = past_error_variance(d, window=10, min_periods=5)
    assert v.iloc[:5].isna().all()
    bumped = d.copy()
    bumped.iloc[20, bumped.columns.get_loc("actual")] += 1.0
    v2 = past_error_variance(bumped, window=10, min_periods=5)
    assert v.iloc[20] == v2.iloc[20]
    assert v.iloc[21] != v2.iloc[21]
    e = d["actual"] - d["forecast"]
    assert v.iloc[20] == pytest.approx(e.iloc[10:20].var(), rel=1e-12)


def test_back_to_vol_applies_the_variance_term_after_the_burn_in():
    d = _file(n=5)
    v = pd.Series([np.nan, np.nan, 0.2, 0.4, 0.6], index=d.index)
    out = back_to_vol(d, v)
    assert np.allclose(out.values, np.exp(d["forecast"].values + 0.5 * np.array([0, 0, 0.2, 0.4, 0.6])))
    with pytest.raises(ValueError):
        back_to_vol(d, pd.Series([0.1, 0.2], index=d.index[[0, 0]]))     # duplicate origin dates


def test_clip_log_uses_the_log_of_the_past_range():
    idx = pd.bdate_range("2020-01-01", periods=6)
    sig = pd.Series([1.0, 2.0, 4.0, 3.0, 2.0, 1.0], index=idx)
    d = pd.DataFrame({"actual": np.log(sig.values), "forecast": [0.0, 5.0, -5.0, 0.5, 5.0, -5.0]},
                     index=idx).iloc[1:]
    out = clip_log(d, sig)
    assert np.allclose(out["forecast"].values, [np.log(1.0), np.log(1.0), 0.5, np.log(4.0), np.log(1.0)])
    with pytest.raises(ValueError):
        clip_log(pd.DataFrame({"forecast": [0.0]}, index=idx[:1]), sig)


def test_utility_peaks_at_four_percent_for_a_perfect_forecast():
    r = np.sort(np.append(np.linspace(0.2, 5.0, 2001), 1.0))
    u = ev.utility(r, np.ones_like(r))
    assert u.max() == pytest.approx(0.04, abs=1e-12)
    assert r[np.argmax(u)] == pytest.approx(1.0, abs=5e-3)
    assert ev.utility(np.exp(0.5), 1.0) < ev.utility(np.exp(-0.5), 1.0)


def test_utility_of_the_targeted_position_matches_the_closed_form():
    rng = np.random.default_rng(1)
    F, rv = np.exp(rng.normal(-9, 1, 200)), np.exp(rng.normal(-9, 1, 200))
    assert np.allclose(ev.utility_of_position(ev.positions(F), rv), ev.utility(rv, F), rtol=1e-12)


def test_gradual_moves_fifteen_percent_toward_the_target():
    assert np.allclose(ev.gradual(np.array([1.0, 2.0, 2.0, 0.0])), [1.0, 1.15, 1.2775, 1.2775 * 0.85])


def test_costs_lower_utility_by_the_turnover_times_the_spread():
    F = np.array([1e-4, 4e-4, 1e-4])
    ns = ev.net_series(F, F.copy(), 1e-3)
    dx = np.abs(np.diff(ev.positions(F), prepend=ev.positions(F)[0]))
    assert np.allclose(ns["zero"] - ns["full"], 252 * 1e-3 * dx)
    assert ns["zero"][0] == ns["full"][0]


def test_positions_use_only_the_forecast():
    rng = np.random.default_rng(3)
    F, rv = np.exp(rng.normal(-9, 1, 50)), np.exp(rng.normal(-9, 1, 50))
    a = ev.net_series(F, rv, 1e-4)
    rv2 = rv.copy()
    rv2[30:] *= 5.0
    b = ev.net_series(F, rv2, 1e-4)
    for case in a:
        assert np.allclose(a[case][:30], b[case][:30])
        assert not np.allclose(a[case][30:], b[case][30:])


def test_ar22_weights_reduce_to_an_ar1_without_weekly_and_monthly_terms():
    psi = ar22_psi(0.6, 0.0, 0.0, 10)
    assert np.allclose(psi, 0.6 ** np.arange(10))


def test_ar22_weights_of_a_monthly_term_spread_over_22_lags():
    psi = ar22_psi(0.0, 0.0, 0.44, 23)
    assert psi[0] == 1.0
    assert np.allclose(psi[1], 0.02)              # 0.44 / 22 on the first lag
    assert np.allclose(psi[2], 0.02 + 0.02 ** 2)  # second lag plus the square of the first
