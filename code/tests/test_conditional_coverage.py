"""Tests for the conditional-coverage tests: Christoffersen (1998) LR tests,
the dynamic-quantile breach regression, and coverage by volatility regime."""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from evaluation.conditional_coverage import (  # noqa: E402
    christoffersen_test,
    coverage_by_regime,
    dq_test,
    volatility_regimes,
)


def test_christoffersen_matches_hand_computation():
    misses = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    r = christoffersen_test(misses, nominal=0.8)
    # n1 = 3, n = 10, pi = 0.3; LR_uc = -2 [7 ln .8 + 3 ln .2 - 7 ln .7 - 3 ln .3]
    lr_uc = -2 * (7 * np.log(.8) + 3 * np.log(.2) - 7 * np.log(.7) - 3 * np.log(.3))
    assert np.isclose(r["lr_uc"], lr_uc)
    assert np.isclose(r["lr_cc"], r["lr_uc"] + r["lr_ind"])


def test_correct_iid_misses_reject_at_about_nominal_rate():
    rng = np.random.default_rng(0)
    rej = {"uc": 0, "ind": 0, "cc": 0, "dq": 0}
    reps = 300
    for _ in range(reps):
        misses = rng.uniform(size=1500) < 0.2
        state = rng.standard_normal(1500)
        r = christoffersen_test(misses, nominal=0.8)
        d = dq_test(misses, state, nominal=0.8)
        rej["uc"] += r["p_uc"] < 0.05
        rej["ind"] += r["p_ind"] < 0.05
        rej["cc"] += r["p_cc"] < 0.05
        rej["dq"] += d["p_dq"] < 0.05
    for k, v in rej.items():
        assert v / reps < 0.10, (k, v / reps)


def test_dq_monte_carlo_p_value_has_correct_size_in_small_samples():
    # n = 41 is the h = 22 test length after subsampling; the asymptotic
    # chi2 rejects a correct interval about 31% of the time there.
    rng = np.random.default_rng(2)
    reps, rej = 200, 0
    for k in range(reps):
        misses = rng.uniform(size=41) < 0.2
        state = np.cumsum(rng.standard_normal(41)) * 0.1 + rng.standard_normal(41)
        rej += dq_test(misses, state, nominal=0.8, seed=k)["p_dq"] < 0.05
    assert rej / reps < 0.10


def test_state_dependent_misses_are_detected():
    rng = np.random.default_rng(1)
    state = rng.standard_normal(1500)
    p_miss = np.where(state > 0.5, 0.40, 0.10)         # misses concentrate in high states
    misses = rng.uniform(size=1500) < p_miss
    d = dq_test(misses, state, nominal=0.8)
    assert d["p_dq"] < 0.01 and d["beta_state"] > 0


def test_clustered_misses_reject_independence():
    misses = np.repeat(np.arange(300) % 5 == 0, 5)   # runs of 5 misses, 20% miss rate
    r = christoffersen_test(misses, nominal=0.8)
    assert r["p_ind"] < 1e-6 and r["p_uc"] > 0.5


def test_coverage_by_regime_labels_terciles():
    state = pd.Series(np.arange(9.0))
    regimes = volatility_regimes(state)
    covered = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0])
    out = coverage_by_regime(covered, regimes)
    assert out == {"cov_calm": 1.0, "cov_normal": 2 / 3, "cov_turbulent": 0.0}
