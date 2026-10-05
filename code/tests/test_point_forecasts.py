"""Point forecasts of the TSFM wrappers that need no model download: Chronos-Bolt
averages its nine trained quantiles, TimesFM 2.5 takes its mean output, and TTM
reports no predictive quantiles."""
import sys
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.foundation import ChronosModel, TimesFMModel  # noqa: E402


class _FakeBolt:
    """predict_quantiles returns right-skewed quantiles and, as the library does,
    the 0.5 quantile as its second value."""

    def __init__(self):
        self.levels = None

    def predict_quantiles(self, ctx, prediction_length, quantile_levels):
        self.levels = list(quantile_levels)
        lv = np.asarray(quantile_levels)
        steps = np.arange(1, prediction_length + 1)[:, None]
        q = np.exp(2.0 * lv)[None, :] * steps                       # (h, n_levels)
        median = q[:, list(quantile_levels).index(0.5)]
        return torch.tensor(q[None], dtype=torch.float32), torch.tensor(median[None], dtype=torch.float32)


@pytest.mark.parametrize("h", [1, 5, 22])
def test_chronos_bolt_point_is_average_of_nine_quantiles(h):
    m = ChronosModel(model_id="amazon/chronos-bolt-small")
    m.pipeline = _FakeBolt()
    out = m.predict(np.linspace(0.01, 0.02, 100), h)
    assert m.pipeline.levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lv = np.array(m.pipeline.levels)
    steps = np.arange(1, h + 1)
    expected = np.exp(2.0 * lv).mean() * steps
    assert np.allclose(out.point, expected, rtol=1e-6)
    assert np.all(out.point > np.exp(2.0 * 0.5) * steps)           # above the median
    assert np.allclose(out.lower, np.exp(0.2) * steps, rtol=1e-6)
    assert np.allclose(out.upper, np.exp(1.8) * steps, rtol=1e-6)


class _FakeTimesFM:
    """forecast returns the 0.5 quantile as point_forecast and a quantile array
    whose column 0 is the mean output."""

    def forecast(self, horizon, inputs):
        steps = np.arange(1, horizon + 1, dtype=float)
        qf = np.zeros((1, horizon, 10))
        qf[0, :, 0] = 7.0 * steps                                    # mean output
        for j in range(1, 10):
            qf[0, :, j] = j * steps                                  # quantiles 0.1 to 0.9
        return qf[:, :, 5], qf


@pytest.mark.parametrize("h", [1, 5, 22])
def test_timesfm_point_is_mean_output(h):
    m = TimesFMModel()
    m.model = _FakeTimesFM()
    out = m.predict(np.linspace(0.01, 0.02, 100), h)
    steps = np.arange(1, h + 1, dtype=float)
    assert np.allclose(out.point, 7.0 * steps)
    assert np.allclose(out.lower, 1.0 * steps)
    assert np.allclose(out.upper, 9.0 * steps)


class _FakeLagPredictor:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, dataset):
        import types
        samples = np.tile(np.arange(1.0, self.horizon + 1.0), (4, 1))
        return iter([types.SimpleNamespace(samples=samples)])


@pytest.mark.parametrize("ctx,expected", [(np.linspace(0.01, 0.02, 50), True),
                                          (np.linspace(-5.0, -4.0, 50), False)])
def test_lag_llama_clips_samples_at_zero_only_for_a_nonnegative_series(ctx, expected):
    from models.foundation import LagLlamaModel
    m = LagLlamaModel(context_length=50)
    m.ckpt_path = "unused"
    seen = []
    m._make_dataset = lambda c: None
    m._get_predictor = lambda h, nonnegative=True: seen.append(nonnegative) or _FakeLagPredictor(h)
    out = m.predict(ctx, 5)
    assert seen == [expected]
    assert np.allclose(out.point, np.arange(1.0, 6.0))
