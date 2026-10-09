"""Tests for the Giacomini-Rossi helpers: the loss scale and the cross-asset average by date."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from evaluation.gr_fluctuation import cross_asset_average, gr_fluctuation_multiple  # noqa: E402
from evaluation.loss_functions import compute_loss_series  # noqa: E402


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sig = np.exp(-4.5 + 0.3 * rng.standard_normal(n))
    return sig, {"bench": sig * np.exp(0.2 * rng.standard_normal(n)), "m": sig * np.exp(0.1 * rng.standard_normal(n))}


def test_qlike_on_volatility_is_squared_before_the_loss():
    sig, f = _data()
    res = gr_fluctuation_multiple(sig, f, "bench", window_fraction=0.3, hac_lags=0, scale="vol")["m"]
    d = (compute_loss_series(sig ** 2, f["bench"] ** 2, "QLIKE")
         - compute_loss_series(sig ** 2, f["m"] ** 2, "QLIKE"))
    m = res.window_size
    first = d[:m].mean() / np.sqrt(np.var(d[:m]) / m)
    assert res.rolling_dm.iloc[0] == pytest.approx(first)
    wrong = gr_fluctuation_multiple(sig, f, "bench", window_fraction=0.3, hac_lags=0, scale="var")["m"]
    assert not np.allclose(res.rolling_dm.values, wrong.rolling_dm.values)


def test_cross_asset_average_aligns_by_date_on_common_dates():
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.bdate_range("2021-01-04", periods=4))
    b = pd.Series([10.0, 20.0, 30.0], index=pd.bdate_range("2021-01-06", periods=3))
    avg = cross_asset_average([a, b])
    assert list(avg.index) == list(pd.bdate_range("2021-01-06", periods=2))
    assert list(avg.values) == [(3 + 10) / 2, (4 + 20) / 2]


def test_cross_asset_average_rejects_positional_index():
    with pytest.raises(ValueError):
        cross_asset_average([pd.Series([1.0, 2.0])])
    with pytest.raises(ValueError):
        cross_asset_average([])
