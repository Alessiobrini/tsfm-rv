"""Averaged-target table: one target for every model, on common dates."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.targets import holding_rv  # noqa: E402
from gen_avg_target_table import loss_ratios  # noqa: E402

H = 3
MODELS = ["Log_HAR", "HAR_J", "ttm"]


def _rv(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.lognormal(-9, 0.5, n), index=idx)


def _write(folder, model, idx, actual, forecast):
    d = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "actual": actual, "forecast": forecast})
    d.to_csv(folder / f"{model}_AAPL_h{H}.csv", index=False)


def _files(folder, rv, har_j_shift=0, ttm_extra=0):
    idx = rv.index[5:len(rv) - H + 1]
    target = holding_rv(rv, idx, H)
    vol = np.sqrt(rv)
    mean_vol = np.array([vol.iloc[i:i + H].mean() for i in range(5, len(rv) - H + 1)])
    _write(folder, "Log_HAR", idx, mean_vol, 1.1 * np.sqrt(target))
    _write(folder, "HAR_J", idx, np.sqrt(np.roll(target, har_j_shift)), 0.9 * np.sqrt(target))
    tidx = rv.index[5 - ttm_extra:len(rv) - H + 1]
    ttarget = holding_rv(rv, tidx, H)
    _write(folder, "ttm", tidx, np.sqrt(ttarget), np.sqrt(ttarget))
    return target


def test_perfect_forecast_of_the_mean_rv_has_zero_loss(tmp_path):
    rv = _rv()
    _files(tmp_path, rv, ttm_extra=3)
    res = loss_ratios(tmp_path, {"AAPL": rv}, tickers=["AAPL"], horizons=[H], models=MODELS)
    q = res.set_index("model")["qlike"]
    assert q["ttm"] == pytest.approx(0.0, abs=1e-12)
    assert q["Log_HAR"] > 0 and q["HAR_J"] > 0
    # the three extra early dates of TTM are not scored
    assert (res["n"] == len(rv) - 5 - H + 1).all()


def test_every_model_is_scored_against_the_mean_rv(tmp_path):
    rv = _rv(seed=1)
    target = _files(tmp_path, rv)
    res = loss_ratios(tmp_path, {"AAPL": rv}, tickers=["AAPL"], horizons=[H], models=MODELS)
    q = res.set_index("model")["qlike"]
    for m, k in [("Log_HAR", 1.1), ("HAR_J", 0.9)]:
        r = 1.0 / k ** 2
        assert q[m] == pytest.approx(r - np.log(r) - 1, rel=1e-10)
    assert res.set_index("model").loc["Log_HAR", "ratio"] == 1.0


def test_misdated_stored_target_is_caught(tmp_path):
    rv = _rv(seed=2)
    _files(tmp_path, rv, har_j_shift=1)
    with pytest.raises(ValueError, match="differs from the target"):
        loss_ratios(tmp_path, {"AAPL": rv}, tickers=["AAPL"], horizons=[H], models=MODELS)


def test_missing_file_stops_the_table(tmp_path):
    rv = _rv(seed=3)
    _files(tmp_path, rv)
    (tmp_path / f"ttm_AAPL_h{H}.csv").unlink()
    with pytest.raises(FileNotFoundError):
        loss_ratios(tmp_path, {"AAPL": rv}, tickers=["AAPL"], horizons=[H], models=MODELS)
