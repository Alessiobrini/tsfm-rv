"""Unit tests for the volatility-scale QLIKE and the min-RV floor (Workstream C)."""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evaluation.loss_functions import qlike, compute_loss_series, compute_all_losses


def test_qlike_vol_equals_var_on_squares():
    rng = np.random.default_rng(0)
    vol_a = rng.uniform(0.01, 0.05, 200)
    vol_f = rng.uniform(0.01, 0.05, 200)
    # QLIKE on the vol scale must equal QLIKE on variance fed the squared inputs.
    assert qlike(vol_a, vol_f, scale="vol") == pytest.approx(
        qlike(vol_a ** 2, vol_f ** 2, scale="var")
    )


def test_qlike_var_floor_clamps_denominator():
    actual = np.array([1e-4, 1e-4])
    forecast = np.array([1e-12, 1e-12])  # variance forecast far below floor
    floored = qlike(actual, forecast, scale="var", var_floor=1e-6)
    unfloored = qlike(actual, forecast, scale="var")
    # Flooring the tiny denominator reduces the (otherwise huge) QLIKE.
    assert floored < unfloored


def test_qlike_rejects_bad_scale():
    with pytest.raises(ValueError):
        qlike(np.array([1.0]), np.array([1.0]), scale="bogus")


def test_compute_all_losses_scale_vol():
    vol_a = np.array([0.02, 0.03, 0.04])
    vol_f = np.array([0.025, 0.028, 0.05])
    d = compute_all_losses(vol_a, vol_f, scale="vol", var_floor=1e-8)
    # MSE/MAE are on the volatility scale (small numbers ~1e-4 / 1e-2).
    assert d["MSE"] == pytest.approx(np.mean((vol_a - vol_f) ** 2))
    assert d["MAE"] == pytest.approx(np.mean(np.abs(vol_a - vol_f)))
    # QLIKE matches the variance-scale computation on squared inputs.
    assert d["QLIKE"] == pytest.approx(qlike(vol_a ** 2, vol_f ** 2, scale="var", var_floor=1e-8))


def test_compute_loss_series_qlike_vol_matches_pointwise():
    vol_a = np.array([0.02, 0.03])
    vol_f = np.array([0.025, 0.02])
    s = compute_loss_series(vol_a, vol_f, "QLIKE", scale="vol")
    a, f = vol_a ** 2, vol_f ** 2
    expected = a / f - np.log(a / f) - 1
    assert np.allclose(s, expected)
