"""Tests for the recursive combination weights: each weight uses only errors observed by its origin."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evaluation.combination import (  # noqa: E402
    bates_granger_weights, bates_granger_recursive, min_variance_recursive, WARMUP,
)


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(-4.5, 0.4, n))
    f1 = a * np.exp(rng.normal(0, 0.2, n))
    f2 = a * np.exp(rng.normal(0.05, 0.35, n))
    return a, f1, f2


@pytest.mark.parametrize("h", [1, 5, 22])
def test_weight_at_t_ignores_targets_not_yet_observed(h):
    a, f1, f2 = _data()
    t = 400
    w = bates_granger_weights(a, f1, f2, horizon=h)
    b = a.copy()
    b[t - h + 1:] = 10.0                     # rows t - h + 1 onward are not observed at origin t
    w2 = bates_granger_weights(b, f1, f2, horizon=h)
    assert np.allclose(w[:t + 1], w2[:t + 1], rtol=0, atol=0)
    f3 = a * np.exp(np.random.default_rng(1).normal(0, 0.3, len(a)))
    m = min_variance_recursive(a, [f1, f2, f3], horizon=h)
    m2 = min_variance_recursive(b, [f1, f2, f3], horizon=h)
    assert np.array_equal(m[:t + 1], m2[:t + 1])


@pytest.mark.parametrize("h", [1, 5, 22])
def test_weight_uses_the_last_observed_row(h):
    a, f1, f2 = _data()
    t = 400
    b = a.copy()
    b[t - h] = 10.0                          # row t - h is observed at origin t
    assert bates_granger_weights(a, f1, f2, horizon=h)[t] != bates_granger_weights(b, f1, f2, horizon=h)[t]


@pytest.mark.parametrize("h", [1, 5, 22])
def test_equal_weights_until_warmup_errors_are_observed(h):
    a, f1, f2 = _data()
    w = bates_granger_weights(a, f1, f2, horizon=h)
    first = WARMUP + h - 1                   # first origin with WARMUP observed errors
    assert np.all(w[:first] == 0.5)
    assert w[first] != 0.5


def test_h1_matches_expanding_window_formula():
    a, f1, f2 = _data()
    w = bates_granger_weights(a, f1, f2, horizon=1)
    for t in [WARMUP, 250, 599]:
        e1, e2 = (a - f1)[:t], (a - f2)[:t]
        s11, s22, s12 = (e1 ** 2).mean(), (e2 ** 2).mean(), (e1 * e2).mean()
        assert w[t] == pytest.approx(min(1, max(0, (s22 - s12) / (s11 + s22 - 2 * s12))))


def test_weights_bounded_and_identical_members():
    a, f1, f2 = _data()
    w = bates_granger_weights(a, f1, f2, horizon=5)
    assert np.all((w >= 0) & (w <= 1))
    assert np.allclose(bates_granger_recursive(a, f1, f1, horizon=5), f1)   # degenerate denominator
    assert np.allclose(min_variance_recursive(a, [f1, f1], horizon=5), f1)


def test_short_series_stays_equal_weight():
    a, f1, f2 = _data(n=50)
    assert np.allclose(bates_granger_recursive(a, f1, f2, horizon=22), 0.5 * (f1 + f2))
