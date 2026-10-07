"""Leakage and algebra checks for the supervised ML baselines (models/ml.py)."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models import ml  # noqa: E402


def _vol(n=700, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.97 * x[i - 1] + rng.normal(0, 0.2)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    return pd.Series(np.exp(-4.5 + x), index=idx)


class _LinearLearner:
    """Least squares on the HAR inputs, with the tuning and fitting interface of the learners. It
    records the training targets and the origin indices it is handed."""
    lookback = 22

    def __init__(self):
        self.base = ml.XGBHAR()
        self.seen = []

    def features(self, vol):
        return self.base.features(vol)

    def _ls(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        return np.linalg.lstsq(A, y, rcond=None)[0]

    def tune(self, Xtr, ytr, Xva, yva):
        b = self._ls(Xtr, ytr)
        pred = np.column_stack([np.ones(len(Xva)), Xva]) @ b
        return dict(cfg={}, n=1, v=float(np.var(yva - pred)), score=0.0)

    def fit(self, X, y, tuned):
        self.seen.append(len(y))
        return self._ls(X, y)

    def predict(self, b, X):
        return np.column_stack([np.ones(len(X)), X]) @ b


@pytest.mark.parametrize("kind", ["point", "avg"])
@pytest.mark.parametrize("h", [1, 5, 22])
def test_log_targets(kind, h):
    v = np.arange(1.0, 41.0)
    y = ml.log_targets(v, h, kind)
    q = 7
    expected = v[q + h - 1] if kind == "point" else v[q:q + h].mean()
    assert y[q] == pytest.approx(np.log(expected))
    assert np.isnan(y[len(v) - h + 1:]).all() and np.isfinite(y[: len(v) - h + 1]).all()


def test_har_features_use_rows_before_the_origin():
    v = np.arange(1.0, 61.0)
    X = ml.XGBHAR().features(v)
    q = 30
    assert X[q, 0] == pytest.approx(np.log(v[q - 1]))
    assert X[q, 1] == pytest.approx(np.log(v[q - 5:q].mean()))
    assert X[q, 2] == pytest.approx(np.log(v[q - 22:q].mean()))
    assert np.isnan(X[:22]).all()


def test_lstm_features_are_the_last_lookback_logs():
    v = np.arange(1.0, 61.0)
    X = ml.LSTMSeq(lookback=10).features(v)
    assert np.allclose(X[25], np.log(v[15:25]))
    assert np.isnan(X[:10]).all()


@pytest.mark.parametrize("kind", ["point", "avg"])
@pytest.mark.parametrize("h", [1, 5, 22])
def test_forecasts_do_not_use_data_after_the_origin(kind, h):
    vol = _vol()
    kw = dict(window=400, refit_every=22, retune_every=100, val_size=100)
    base = ml.rolling_ml(vol, _LinearLearner(), h, kind, **kw)
    p = 520
    cut = vol.index[p]
    vol2 = vol.copy()
    vol2.iloc[p:] = vol2.iloc[p:] * 3.0               # rows from the origin on are not observed at p
    alt = ml.rolling_ml(vol2, _LinearLearner(), h, kind, **kw)
    upto = base.index <= cut
    assert np.allclose(base.loc[upto, "forecast"], alt.loc[upto, "forecast"], rtol=1e-12)
    later = base.index > cut
    assert not np.allclose(base.loc[later, "forecast"], alt.loc[later, "forecast"])


def test_training_window_size_and_actuals():
    vol = _vol()
    h = 5
    lr = _LinearLearner()
    out = ml.rolling_ml(vol, lr, h, "point", window=400, refit_every=22, retune_every=100, val_size=100)
    assert set(lr.seen) == {400 - 22 - h + 1}
    i = vol.index.get_loc(out.index[3])
    assert out["actual"].iloc[3] == pytest.approx(vol.iloc[i + h - 1])
    assert out.index[0] == vol.index[400] and out.index[-1] == vol.index[len(vol) - h]


def test_retransformation():
    assert ml.to_vol(np.log(0.02), 0.08)[()] == pytest.approx(0.02 * np.exp(0.04))


def test_xgb_and_lstm_run_and_give_finite_forecasts():
    vol = _vol(n=560)
    kw = dict(window=400, refit_every=60, retune_every=200, val_size=100)
    pytest.importorskip("xgboost")
    out = ml.rolling_ml(vol, ml.XGBHAR(), 5, "point", **kw)
    assert np.isfinite(out["forecast"]).all() and (out["forecast"] > 0).all()
    pytest.importorskip("torch")
    lstm = ml.LSTMSeq(lookback=22, seeds=2, max_epochs=5, patience=2)
    out = ml.rolling_ml(vol, lstm, 1, "point", **kw)
    assert np.isfinite(out["forecast"]).all() and (out["forecast"] > 0).all()
