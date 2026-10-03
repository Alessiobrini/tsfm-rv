"""Tests for the TTM wrappers that need no model download."""
import sys
import pathlib
import types

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.foundation import TTMModel, TTMR2Model  # noqa: E402


def test_ttm_r2_accepts_only_released_context_lengths():
    for c in (512, 1024, 1536):
        assert TTMR2Model(context_length=c).context_length == c
    with pytest.raises(ValueError):
        TTMR2Model(context_length=1000)


class _FakeOut:
    def __init__(self, n):
        import torch
        self.prediction_outputs = torch.arange(n, dtype=torch.float32).reshape(1, n, 1)


class _FakeTTM:
    def __init__(self, prefix):
        self.config = types.SimpleNamespace(resolution_prefix_tuning=prefix)
        self.calls = []

    def __call__(self, past_values, **kwargs):
        self.calls.append((tuple(past_values.shape), sorted(kwargs)))
        return _FakeOut(96)


@pytest.mark.parametrize("prefix,expected", [(True, ["freq_token"]), (False, [])])
def test_frequency_token_only_for_checkpoints_trained_with_it(prefix, expected):
    m = TTMModel(context_length=512)
    m.model, m._eff_ctx = _FakeTTM(prefix), 512
    out = m.predict(np.ones(1000, dtype=float), 22)
    shape, kw = m.model.calls[0]
    assert kw == expected
    assert shape == (1, 512, 1)            # only the last 512 observations are passed
    assert len(out.point) == 22


class _FakeForecast:
    seen = []

    def __init__(self, module, prediction_length, context_length, **kwargs):
        _FakeForecast.seen.append(context_length)
        self.h = prediction_length

    def forward(self, past_target, past_observed_target, past_is_pad, num_samples):
        import torch
        _FakeForecast.shapes = (tuple(past_target.shape), int(past_is_pad.sum()))
        return torch.ones(1, num_samples, self.h)


@pytest.mark.parametrize("ctx,expected_len,expected_pad", [(1000, 1000, 0), (512, 512, 0), (128, 512, 384)])
def test_moirai_moe_forecaster_gets_the_length_of_the_data_it_receives(monkeypatch, ctx, expected_len, expected_pad):
    import sys as _sys
    import types as _types
    fake = _types.ModuleType("uni2ts.model.moirai_moe")
    fake.MoiraiMoEForecast = _FakeForecast
    monkeypatch.setitem(_sys.modules, "uni2ts.model.moirai_moe", fake)
    from models.foundation import MoiraiMoEModel
    m = MoiraiMoEModel(context_length=ctx)
    m.module = object()
    _FakeForecast.seen = []
    m.predict(np.arange(1, 1501, dtype=float), 22)
    assert _FakeForecast.seen == [expected_len]
    assert _FakeForecast.shapes == ((1, expected_len, 1), expected_pad)
