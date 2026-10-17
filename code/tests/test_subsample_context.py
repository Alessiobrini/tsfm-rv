"""Subsample and context tables use the dates common to the paper's models."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gen_context_sensitivity as gcs  # noqa: E402
from compute_subsample_metrics import per_asset_subsample  # noqa: E402


def _write(folder, name, idx, actual, forecast):
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "actual": actual, "forecast": forecast}
                 ).to_csv(folder / name, index=False)


def _panel(n=60):
    idx = pd.bdate_range("2020-01-01", periods=n)   # spans 1 March 2020
    a = np.linspace(0.01, 0.02, n)
    return idx, a


def test_subsample_scores_every_model_on_common_dates(tmp_path):
    idx, a = _panel()
    _write(tmp_path, "Log_HAR_AAPL_h1.csv", idx[5:], a[5:], 1.1 * a[5:])
    _write(tmp_path, "ttm_AAPL_h1.csv", idx, a, a)          # five extra early dates
    res = per_asset_subsample(tmp_path, tickers=["AAPL"], horizons=[1], models=["Log_HAR", "ttm"],
                              series_index={"AAPL": idx})
    n = res.groupby(["period", "model"])["n_obs"].first().unstack()
    assert (n["Log_HAR"] == n["ttm"]).all()
    assert n.sum(axis=0)["ttm"] == len(idx) - 5
    assert res[res.model == "ttm"]["QLIKE"].abs().max() < 1e-12


def test_subsample_rejects_files_with_different_actuals(tmp_path):
    idx, a = _panel()
    _write(tmp_path, "Log_HAR_AAPL_h1.csv", idx, a, a)
    _write(tmp_path, "ttm_AAPL_h1.csv", idx, a * 1.01, a)
    with pytest.raises(ValueError):
        per_asset_subsample(tmp_path, tickers=["AAPL"], horizons=[1], models=["Log_HAR", "ttm"],
                            series_index={"AAPL": idx})


def test_context_table_uses_the_dates_common_to_all_main_files(tmp_path, monkeypatch):
    idx, a = _panel()
    monkeypatch.setattr(gcs, "FC", tmp_path)
    monkeypatch.setattr(gcs, "PAPER_MODELS", ["Log_HAR", "ttm"])
    monkeypatch.setattr(gcs, "ALL", ["AAPL"])
    gcs.ref_dates.cache_clear()
    _write(tmp_path, "Log_HAR_AAPL_h1.csv", idx[10:], a[10:], a[10:])
    _write(tmp_path, "ttm_AAPL_h1.csv", idx, a, a)
    # a context file that is wrong only on the ten dates outside the common sample
    f = a.copy()
    f[:10] = 5 * a[:10]
    _write(tmp_path, "ttm_AAPL_h1_ctx128.csv", idx, a, f)
    rd = gcs.ref_dates(1, "AAPL")
    assert list(rd) == list(idx[10:])
    m, n = gcs.mean_qlike("ttm", 1, 128)
    assert n == 1 and m == pytest.approx(0.0, abs=1e-12)
    gcs.ref_dates.cache_clear()


def test_subsample_splits_by_the_date_of_the_target(tmp_path):
    idx = pd.bdate_range("2020-02-03", periods=40)          # crosses 1 March 2020
    a = np.linspace(0.01, 0.02, len(idx))
    h = 5
    rows = idx[: len(idx) - h + 1]
    _write(tmp_path, f"Log_HAR_AAPL_h{h}.csv", rows, a[h - 1:], a[h - 1:])
    _write(tmp_path, f"ttm_AAPL_h{h}.csv", rows, a[h - 1:], a[h - 1:])
    res = per_asset_subsample(tmp_path, tickers=["AAPL"], horizons=[h], models=["Log_HAR", "ttm"],
                              series_index={"AAPL": idx})
    n_pre = res[(res.model == "ttm") & (res.period == "pre-COVID")]["n_obs"].iloc[0]
    # targets before 1 March 2020 are the rows whose date + 4 trading days precede it
    expected = int((idx[np.arange(len(rows)) + h - 1] < "2020-03-01").sum())
    assert n_pre == expected
    assert n_pre < int((rows < "2020-03-01").sum())


def test_target_dates_move_h_minus_one_rows():
    from evaluation.targets import target_dates
    idx = pd.bdate_range("2020-01-01", periods=10)
    assert list(target_dates(idx, idx[:3], 1)) == list(idx[:3])
    assert list(target_dates(idx, idx[:3], 4)) == list(idx[3:6])
    with pytest.raises(ValueError):
        target_dates(idx, idx[8:], 4)
