"""Tests for the Model Confidence Set: the T_R elimination rule and the MCS p-values of
Hansen, Lunde & Nason (2011)."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from evaluation.mcs import model_confidence_set  # noqa: E402

B = 2000


def _losses(seed=0, n=2000):
    """A is best and precise. C is worse than A by 0.05 with tiny noise, so t_CA is large.
    B has the highest mean loss but so much noise that t_BA is small."""
    rng = np.random.default_rng(seed)
    return {
        "A": 1.00 + 0.01 * rng.standard_normal(n),
        "B": 1.30 + 20.0 * rng.standard_normal(n),
        "C": 1.05 + 0.01 * rng.standard_normal(n),
    }


def test_eliminates_the_model_that_attains_t_r_not_the_highest_mean():
    res = model_confidence_set(_losses(), alpha=0.10, n_bootstrap=B, block_length=5)
    # the old rule (highest mean loss) would eliminate B first
    assert res.eliminated_models[0] == "C"
    assert set(res.surviving_models) == {"A", "B"}


def test_pvalues_follow_definition_4():
    rng = np.random.default_rng(1)
    n = 1500
    losses = {f"m{k}": 1.0 + 0.02 * k + 0.05 * rng.standard_normal(n) for k in range(5)}
    res = model_confidence_set(losses, alpha=0.10, n_bootstrap=B, block_length=5)
    seq = [res.p_values[m] for m in res.eliminated_models]
    assert all(b >= a for a, b in zip(seq, seq[1:]))  # cumulative max over elimination steps
    assert all(p < 0.10 for p in seq)
    surv = [res.p_values[m] for m in res.surviving_models]
    if len(res.surviving_models) == 1:
        assert surv == [1.0]
    else:
        assert all(p >= 0.10 for p in surv) and len(set(surv)) == 1
    assert set(res.p_values) == set(losses)


def test_identical_losses_all_survive():
    base = 1.0 + np.random.default_rng(2).standard_normal(1000)
    res = model_confidence_set({m: base.copy() for m in "XYZ"}, alpha=0.10, n_bootstrap=B, block_length=5)
    assert res.eliminated_models == []
    assert set(res.surviving_models) == set("XYZ")


def test_rejection_rate_under_equal_expected_losses_is_near_alpha():
    # three models with the same expected loss: the first test should reject in about 10% of samples
    rng = np.random.default_rng(5)
    rejected = 0
    reps = 200
    for r in range(reps):
        common = rng.standard_normal(500)
        losses = {m: 1.0 + common + 0.5 * rng.standard_normal(500) for m in "XYZ"}
        res = model_confidence_set(losses, alpha=0.10, n_bootstrap=400, block_length=5, seed=r)
        rejected += len(res.eliminated_models) > 0
    assert 0.03 <= rejected / reps <= 0.20


def test_single_model():
    res = model_confidence_set({"only": np.ones(50)}, n_bootstrap=100, block_length=5)
    assert res.surviving_models == ["only"] and res.p_values["only"] == 1.0


def test_clear_case_matches_arch():
    arch = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(4)
    n = 1500
    losses = {"A": 1.00 + 0.05 * rng.standard_normal(n),
              "B": 1.01 + 0.05 * rng.standard_normal(n),
              "C": 1.20 + 0.05 * rng.standard_normal(n),
              "D": 1.25 + 3.00 * rng.standard_normal(n)}
    ours = model_confidence_set(losses, alpha=0.10, n_bootstrap=B, block_length=10)
    import pandas as pd
    mcs = arch.MCS(pd.DataFrame(losses), size=0.10, reps=B, block_size=10, method="R",
                   bootstrap="moving block", seed=0)
    mcs.compute()
    assert set(ours.surviving_models) == set(mcs.included)
    assert ours.eliminated_models[0] == "C"
