"""Unit tests for features.build_target — the point-in-time vs. average target.

The index alignment of the forecast target is the single highest-risk change in
the revision (Referee 1's main objection was the averaged target). These tests
pin down the alignment on a hand-checked toy series before any forecast is run.

Run: pytest code/tests/test_features.py -q
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

# Make the `code/` package importable when run from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from features import build_target


def _toy_rv(n: int = 10) -> pd.Series:
    """RV_t = t, so values are 0,1,...,n-1 — trivial to check alignment by eye."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.arange(n, dtype=float), index=idx, name="rv")


def test_h1_is_identity_for_both_kinds():
    """At h=1 both conventions equal RV_t (features already shifted)."""
    rv = _toy_rv()
    pd.testing.assert_series_equal(build_target(rv, 1, "point"), rv)
    pd.testing.assert_series_equal(build_target(rv, 1, "avg"), rv)


def test_point_h5_is_h1_shifted_four_rows():
    """Point-in-time h=5 target at row i must equal the h=1 target at row i+4."""
    rv = _toy_rv()
    t1 = build_target(rv, 1, "point")
    t5 = build_target(rv, 5, "point")
    for i in range(len(rv) - 4):
        assert t5.iloc[i] == t1.iloc[i + 4]
    # The last (h-1)=4 rows have no future observation -> NaN.
    assert t5.iloc[-4:].isna().all()
    assert t5.iloc[: len(rv) - 4].notna().all()


def test_point_values_are_single_future_day():
    """Point target reads RV on exactly one future day, not an average."""
    rv = _toy_rv()  # values 0..9
    t5 = build_target(rv, 5, "point")
    assert t5.iloc[0] == 4.0  # RV at index 4 (= RV_{0+5-1})
    assert t5.iloc[5] == 9.0  # RV at index 9


def test_avg_values_are_forward_mean():
    """Average target reads the mean of RV over days i..i+h-1."""
    rv = _toy_rv()  # values 0..9
    t5 = build_target(rv, 5, "avg")
    assert t5.iloc[0] == np.mean([0, 1, 2, 3, 4])   # 2.0
    assert t5.iloc[3] == np.mean([3, 4, 5, 6, 7])   # 5.0
    assert t5.iloc[-4:].isna().all()


def test_point_and_avg_differ_for_h_gt_1():
    rv = _toy_rv()
    assert not build_target(rv, 5, "point").equals(build_target(rv, 5, "avg"))


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        build_target(_toy_rv(), 5, "bogus")
