"""Tests for the fine-tuning windows, the Sundial loss layout and the training helpers."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
torch = pytest.importorskip("torch")

from finetune.common import EarlyStopper, lr_lambda  # noqa: E402
from finetune.windows import EMBARGO, TRAIN_END, RVWindowDataset, purge_cutoff  # noqa: E402
from finetune.train import sundial_loss_inputs  # noqa: E402


def _panel(n_names=2, start="2017-01-02", end="2022-06-30", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    rows = []
    for k in range(n_names):
        rv = np.exp(-9 + np.cumsum(rng.normal(0, 0.05, len(dates))))
        rows.append(pd.DataFrame({"permno": 100 + k, "date": dates, "rv5": rv}))
    return pd.concat(rows, ignore_index=True)


def test_train_windows_end_by_train_end_and_hold_volatility():
    p = _panel()
    ds = RVWindowDataset(p, "train", 64, 8, stride=3)
    assert len(ds) > 0
    for permno, t in ds.index:
        d = p[p.permno == permno].sort_values("date").date.values
        assert pd.Timestamp(d[t + 8 - 1]) <= TRAIN_END
    permno, t = ds.index[5]
    g = p[p.permno == permno].sort_values("date")
    x = ds[5]
    assert np.allclose(x["past_values"].squeeze(-1).numpy(), np.sqrt(g.rv5.values[t - 64:t]), rtol=1e-6)
    assert np.allclose(x["future_values"].squeeze(-1).numpy(), np.sqrt(g.rv5.values[t:t + 8]), rtol=1e-6)


def test_val_windows_end_in_2021_before_the_purge():
    p = _panel()
    ds = RVWindowDataset(p, "val", 64, 8, stride=1)
    cut = purge_cutoff(p.date.values)
    assert len(ds) > 0 and cut < EMBARGO
    ends = []
    for permno, t in ds.index:
        d = p[p.permno == permno].sort_values("date").date.values
        ends.append(pd.Timestamp(d[t + 8 - 1]))
    assert min(ends) > TRAIN_END and max(ends) <= cut
    cal = pd.bdate_range(cut, EMBARGO - pd.Timedelta(days=1))
    assert len(cal) - 1 >= 21                      # at least 22 trading days between cut and embargo


def test_split_rejects_unknown_and_drops_nonpositive_rv():
    p = _panel()
    with pytest.raises(ValueError):
        RVWindowDataset(p, "test", 64, 8)
    p.loc[10, "rv5"] = 0.0
    ds = RVWindowDataset(p, "train", 64, 8)
    assert all(np.isfinite(v).all() for v in ds.series.values())


def _last_patch_target(labels, n_patches, it, ot, H):
    """The model's own slicing: labels cut to seq_len - it + ot and unfolded with step it."""
    seq_len = n_patches * it
    lab = labels[:, : seq_len - it + ot].unfold(-1, ot, it)
    assert lab.shape[1] == n_patches
    return lab[:, -1, :H]


@pytest.mark.parametrize("L", [512, 1000, 17])
def test_sundial_last_patch_targets_the_days_after_the_context(L):
    it, ot, H, B = 16, 720, 96, 3
    past = torch.randn(B, L, 1)
    future = torch.randn(B, H, 1)
    ids, labels, lm, my = sundial_loss_inputs(past, future, it, ot)
    n_patches = -(-L // it)
    assert ids.shape == (B, L) and lm.shape == (B, n_patches) and lm[:, -1].eq(1).all() and lm[:, :-1].eq(0).all()
    assert my[:, :H].eq(1).all() and my[:, H:].eq(0).all()
    assert torch.allclose(_last_patch_target(labels, n_patches, it, ot, H), future.squeeze(-1))


def test_sundial_layout_unchanged_for_whole_patches():
    it, ot, H, L = 16, 720, 96, 512
    past, future = torch.randn(2, L, 1), torch.randn(2, H, 1)
    _, labels, _, _ = sundial_loss_inputs(past, future, it, ot)
    series = torch.cat([past, future], dim=1).squeeze(-1)
    old = torch.zeros(2, L - it + ot)
    old[:, : L + H - it] = series[:, it:]
    assert torch.equal(labels, old)


def test_sundial_rejects_long_horizon():
    with pytest.raises(ValueError):
        sundial_loss_inputs(torch.randn(1, 32, 1), torch.randn(1, 721, 1), 16, 720)


def test_warmup_and_early_stopping():
    assert lr_lambda(0, 500) == pytest.approx(1 / 500) and lr_lambda(499, 500) == 1.0 and lr_lambda(3, 0) == 1.0
    s = EarlyStopper(patience=2)
    assert s.update(1.0) and not s.update(1.0) and not s.should_stop
    assert not s.update(2.0) and s.should_stop
    assert not EarlyStopper(patience=0).should_stop


def test_factory_requires_a_checkpoint_for_fine_tuned_models():
    from models.foundation import get_foundation_model
    with pytest.raises(ValueError):
        get_foundation_model("ttm-ft", context_length=1000)
    m = get_foundation_model("ttm-ft", context_length=1000, checkpoint_path="/x")
    assert m.checkpoint_path == "/x" and m._model_name == "TTM-FT"
    s = get_foundation_model("sundial-ft", context_length=1000, checkpoint_path="/y")
    assert s.model_id == "/y" and s._model_name == "Sundial-FT"
    z = get_foundation_model("ttm", context_length=1000, checkpoint_path=None)
    assert z.checkpoint_path is None and z._model_name == "TTM"
