"""Scoring target, common sample and model filter shared by the evaluation scripts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PAPER_MODELS  # noqa: E402
from evaluation.targets import common_dates, holding_rv  # noqa: E402


def _rv(n=10):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.arange(1.0, n + 1.0), index=idx)


def test_holding_rv_h1_is_rv_on_the_date():
    rv = _rv()
    d = rv.index[[2, 5, 7]]
    assert np.allclose(holding_rv(rv, d, 1), rv.loc[d].values)


def test_holding_rv_averages_the_h_rows_from_the_date():
    rv = _rv()
    # rows 3, 4, 5 hold 4, 5, 6
    assert np.allclose(holding_rv(rv, rv.index[[3]], 3), [5.0])
    # last full window ends on the final row
    assert np.allclose(holding_rv(rv, rv.index[[7]], 3), [9.0])


def test_holding_rv_counts_rows_not_calendar_days():
    rv = _rv().drop(_rv().index[4])          # a missing day inside the window
    out = holding_rv(rv, rv.index[[3]], 3)   # rows hold 4, 6, 7
    assert np.allclose(out, [17.0 / 3.0])


def test_holding_rv_rejects_unknown_dates_and_short_tails():
    rv = _rv()
    with pytest.raises(ValueError):
        holding_rv(rv, [pd.Timestamp("2030-01-01")], 1)
    with pytest.raises(ValueError):
        holding_rv(rv, rv.index[[8]], 3)


def test_common_dates_intersects_and_ignores_missing_forecasts():
    idx = pd.bdate_range("2020-01-01", periods=6)
    a = pd.DataFrame({"forecast": [1, 1, 1, 1, 1, 1.0]}, index=idx)
    b = pd.DataFrame({"forecast": [np.nan, 1, 1, 1, 1, 1.0]}, index=idx)
    c = pd.DataFrame({"forecast": [1, 1, 1, 1.0]}, index=idx[2:])
    out = common_dates({"a": a, "b": b, "c": c})
    assert list(out) == list(idx[2:])


def _write(folder, name, n=5):
    idx = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    pd.DataFrame({"date": idx, "actual": 0.01, "forecast": 0.01}).to_csv(folder / name, index=False)


def test_paper_models_are_the_17_of_the_comparison():
    assert len(PAPER_MODELS) == 17 and len(set(PAPER_MODELS)) == 17
    assert PAPER_MODELS[0] == "Log_HAR" and "ttm" in PAPER_MODELS


def test_main_loader_skips_models_outside_the_comparison(tmp_path, monkeypatch):
    import run_evaluation_volare as rev
    for name in ["ttm_AAPL_h1.csv", "Log_HAR_AAPL_h1.csv", "ttm_ft_AAPL_h1.csv", "xgb_har_AAPL_h1.csv"]:
        _write(tmp_path, name)
    monkeypatch.setattr(rev, "FORECAST_DIR", tmp_path)
    groups = rev.load_all_forecasts()
    assert sorted(groups[("AAPL", 1)]) == ["Log_HAR", "ttm"]


def test_advanced_loader_skips_models_outside_the_comparison(tmp_path):
    import run_advanced_evaluation as rae
    for name in ["sundial_AAPL_h5.csv", "sundial_ft_AAPL_h5.csv", "Log_HAR_refit22_AAPL_h5.csv"]:
        _write(tmp_path, name)
    groups = rae.load_forecasts(tmp_path)
    assert sorted(groups[("AAPL", 5)]) == ["sundial"]
