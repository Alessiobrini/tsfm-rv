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
    res = per_asset_subsample(tmp_path, tickers=["AAPL"], horizons=[1], models=["Log_HAR", "ttm"])
    n = res.groupby(["period", "model"])["n_obs"].first().unstack()
    assert (n["Log_HAR"] == n["ttm"]).all()
    assert n.sum(axis=0)["ttm"] == len(idx) - 5
    assert res[res.model == "ttm"]["QLIKE"].abs().max() < 1e-12


def test_subsample_rejects_files_with_different_actuals(tmp_path):
    idx, a = _panel()
    _write(tmp_path, "Log_HAR_AAPL_h1.csv", idx, a, a)
    _write(tmp_path, "ttm_AAPL_h1.csv", idx, a * 1.01, a)
    with pytest.raises(ValueError):
        per_asset_subsample(tmp_path, tickers=["AAPL"], horizons=[1], models=["Log_HAR", "ttm"])


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
