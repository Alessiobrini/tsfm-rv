"""
models/foundation.py — Wrappers for time series foundation models.

Provides a unified interface for:
    - Chronos-2 / Chronos-Bolt (Amazon)
    - TimesFM 2.5 (Google)
    - Moirai 2.0 (Salesforce)

Each wrapper follows the same predict() interface:
    input:  np.ndarray of historical RV (context window)
    output: TSFMForecast with point forecasts + optional prediction intervals

Models are imported with try/except so the codebase runs even if a package
is not installed — unavailable models raise ImportError at load time.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


# Default number of MC samples for predict_density() when a wrapper falls
# back to empirical quantiles from samples. The Brini reply flags that the
# zero-shot studies' default of 20 is too few for tail quantiles; we use
# 500 so that Q2.5 / Q97.5 are estimated from ~12 samples in each tail
# bin rather than ~5 (Q2 in the design log).
DENSITY_NUM_SAMPLES = 500


def _interp_quantiles_to_grid(
    native_levels: np.ndarray,
    native_quantiles: np.ndarray,
    target_levels: np.ndarray,
) -> np.ndarray:
    """Resample a quantile vector at new levels.

    Interior levels use monotone linear interpolation in level space.
    Tail levels are extrapolated in log space when all native quantiles
    are positive (RV is log-normal-ish, so log-space extrapolation
    captures the heavy upper tail far better than linear). When any
    native quantile is non-positive we fall back to linear extrapolation
    and clip at zero.

    Limitation: when only 9 deciles are available (TimesFM, Moirai 2.0),
    the Q2.5 / Q97.5 extrapolation can carry ~20-40% relative error on a
    true log-normal because the deciles do not constrain the very
    extreme tails. This applies equally to both wrappers, so cross-model
    comparison remains apples-to-apples; tail-CRPS interpretation is the
    only place where the limitation matters.

    Parameters
    ----------
    native_levels : (J,) array of quantile levels emitted by the model
    native_quantiles : (horizon, J) array of corresponding quantile values
    target_levels : (K,) array of levels to resample at
    """
    horizon = native_quantiles.shape[0]
    out = np.empty((horizon, len(target_levels)), dtype=float)
    inside = (target_levels >= native_levels[0]) & (target_levels <= native_levels[-1])
    below = target_levels < native_levels[0]
    above = target_levels > native_levels[-1]

    for t in range(horizon):
        q = native_quantiles[t]
        out[t, inside] = np.interp(target_levels[inside], native_levels, q)

        if q.min() > 0:
            # Log-space extrapolation for positive, heavy-tailed series.
            log_q = np.log(q)
            if below.any():
                slope = (log_q[1] - log_q[0]) / (native_levels[1] - native_levels[0] + 1e-30)
                out[t, below] = np.exp(log_q[0] + slope * (target_levels[below] - native_levels[0]))
            if above.any():
                slope = (log_q[-1] - log_q[-2]) / (native_levels[-1] - native_levels[-2] + 1e-30)
                out[t, above] = np.exp(log_q[-1] + slope * (target_levels[above] - native_levels[-1]))
        else:
            if below.any():
                slope = (q[1] - q[0]) / (native_levels[1] - native_levels[0] + 1e-30)
                out[t, below] = q[0] + slope * (target_levels[below] - native_levels[0])
            if above.any():
                slope = (q[-1] - q[-2]) / (native_levels[-1] - native_levels[-2] + 1e-30)
                out[t, above] = q[-1] + slope * (target_levels[above] - native_levels[-1])

    out = np.maximum.accumulate(out, axis=1)
    return np.clip(out, 0.0, None)


@dataclass
class TSFMForecast:
    """Container for foundation model forecast output.

    Fields added for the density phase (Brini, May 2026 reply):
        quantile_levels : (K,) array of levels in (0, 1)
        quantile_grid   : (horizon, K) array of predicted quantiles
    Both are optional; legacy point-only callers can ignore them.
    """
    point: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None
    quantile_grid: Optional[np.ndarray] = None
    quantile_levels: Optional[np.ndarray] = None
    model_name: str = ""


class BaseTSFM(ABC):
    """Abstract base class for time series foundation models."""

    @abstractmethod
    def load_model(self) -> None:
        """Load pretrained model weights."""
        pass

    @abstractmethod
    def predict(
        self,
        context: np.ndarray,
        horizon: int,
    ) -> TSFMForecast:
        """Generate forecasts given a context window.

        Parameters
        ----------
        context : np.ndarray
            Historical values (e.g., past RV observations).
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        TSFMForecast
            Point forecast and optional intervals.
        """
        pass

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Return a forecast populated with `quantile_grid` and `quantile_levels`.

        Default implementation: call :meth:`predict` and, if it returns
        samples, derive empirical quantiles. Subclasses with a native
        quantile API should override this (Chronos-Bolt does so).

        Models that emit neither samples nor native quantiles (currently
        only TTM) raise NotImplementedError — they have no density and
        must be flagged "point-only" in the harness.
        """
        from evaluation.density import DEFAULT_QUANTILE_LEVELS  # local import: optional dep
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        fc = self.predict(context, horizon)
        if fc.samples is None:
            raise NotImplementedError(
                f"{type(self).__name__}.predict() returns no samples; "
                "override predict_density() with a native quantile path."
            )
        samples = np.asarray(fc.samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError(
                f"expected samples of shape (num_samples, horizon); "
                f"got {samples.shape}"
            )
        # samples: (num_samples, horizon) -> quantiles: (horizon, K)
        q_grid = np.quantile(samples, levels, axis=0).T
        fc.quantile_grid = q_grid
        fc.quantile_levels = levels
        return fc


class ChronosModel(BaseTSFM):
    """Wrapper for Amazon Chronos-2 / Chronos-Bolt models.

    Uses the `chronos-forecasting` package.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier:
        - "amazon/chronos-bolt-small"   (fastest)
        - "amazon/chronos-bolt-base"    (larger)
    device : str
        "cuda" or "cpu".
    num_samples : int
        Number of forecast samples for probabilistic output.
    context_length : int
        Maximum context window length.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        device: str = "cpu",
        num_samples: int = 20,
        context_length: int = 512,
    ):
        self.model_id = model_id
        self.device = device
        self.num_samples = num_samples
        self.context_length = context_length
        self.pipeline = None
        self._model_name = model_id.split("/")[-1]

    def load_model(self) -> None:
        """Load Chronos pipeline from HuggingFace."""
        import torch

        if "bolt" in self.model_id:
            from chronos.chronos_bolt import ChronosBoltPipeline
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            try:
                self.pipeline = ChronosBoltPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=dtype,
                )
            except TypeError:
                self.pipeline = ChronosBoltPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    dtype=dtype,
                )
        else:
            from chronos import ChronosPipeline
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            try:
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=dtype,
                )
            except TypeError:
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    dtype=dtype,
                )

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using Chronos."""
        if self.pipeline is None:
            self.load_model()

        import torch

        ctx = context[-self.context_length:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        # Chronos-Bolt uses quantile prediction (no num_samples needed)
        # Chronos-T5 uses sampling. The API handles both transparently.
        if "bolt" in self.model_id:
            # Bolt returns quantile forecasts directly
            quantiles, mean = self.pipeline.predict_quantiles(
                ctx_tensor,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            # quantiles shape: (1, horizon, 3), mean shape: (1, horizon)
            point = mean.numpy().squeeze(0)  # (horizon,)
            q = quantiles.numpy().squeeze(0)  # (horizon, 3)
            lower = q[:, 0]
            median = q[:, 1]
            upper = q[:, 2]
            # Use median as point forecast (more robust for RV)
            point = median
            return TSFMForecast(
                point=point,
                lower=lower,
                upper=upper,
                model_name=self._model_name,
            )

        # Original Chronos: sample-based
        samples = self.pipeline.predict(
            ctx_tensor,
            prediction_length=horizon,
            num_samples=self.num_samples,
        )  # (1, num_samples, horizon)
        samples_np = samples.numpy().squeeze(0)  # (num_samples, horizon)
        point = np.median(samples_np, axis=0)
        lower = np.percentile(samples_np, 10, axis=0)
        upper = np.percentile(samples_np, 90, axis=0)
        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            samples=samples_np,
            model_name=self._model_name,
        )

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Chronos-Bolt native 9 deciles + log-space tail extrapolation.

        Chronos-Bolt's `predict_quantiles` advertises arbitrary quantile
        levels, but the underlying model is trained on a fixed grid of
        9 deciles {0.1, ..., 0.9}. Levels outside [0.1, 0.9] are clamped
        to the boundary with a warning. We query the native deciles and
        extrapolate to {Q2.5, Q5, Q95, Q97.5} the same way as TimesFM
        and Moirai 2.0 (Q1 flag in the design log).

        Chronos-T5 has no native quantile API; the default sample-based
        implementation in :class:`BaseTSFM` handles it.
        """
        from evaluation.density import DEFAULT_QUANTILE_LEVELS  # local: optional dep
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )

        if "bolt" not in self.model_id:
            return super().predict_density(context, horizon, levels)

        if self.pipeline is None:
            self.load_model()

        import torch

        ctx = context[-self.context_length:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        native_levels = np.arange(0.1, 0.91, 0.1)               # (9,)
        quantiles, _ = self.pipeline.predict_quantiles(
            ctx_tensor,
            prediction_length=horizon,
            quantile_levels=native_levels.tolist(),
        )
        native_q = quantiles.numpy().squeeze(0)                  # (horizon, 9)
        q_grid = _interp_quantiles_to_grid(native_levels, native_q, levels)

        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        i_lo = int(np.argmin(np.abs(levels - 0.10)))
        i_hi = int(np.argmin(np.abs(levels - 0.90)))
        return TSFMForecast(
            point=q_grid[:, median_idx],
            lower=q_grid[:, i_lo],
            upper=q_grid[:, i_hi],
            quantile_grid=q_grid,
            quantile_levels=levels,
            model_name=self._model_name,
        )


class TimesFMModel(BaseTSFM):
    """Wrapper for Google TimesFM 2.5.

    Requires timesfm >= 2.0.0 installed from GitHub:
        pip install "timesfm @ git+https://github.com/google-research/timesfm.git"

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    context_length : int
        Context window size.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        context_length: int = 512,
        device: str = "cpu",
        **kwargs,
    ):
        self.model_id = model_id
        self.context_length = context_length
        self.device = device
        self.model = None
        self._model_name = "TimesFM-2.5"

    def load_model(self) -> None:
        """Load TimesFM 2.5 model using the v2.5 API."""
        import timesfm
        import torch
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")

        # Workaround: huggingface_hub >= 0.36 passes extra hub kwargs
        # (proxies, resume_download, etc.) through _from_pretrained into
        # __init__, which only accepts (torch_compile, config). Patch
        # __init__ to absorb unexpected kwargs.
        _orig_init = TimesFM_2p5_200M_torch.__init__

        def _patched_init(self_inner, torch_compile=True, config=None, **_extra):
            _orig_init(self_inner, torch_compile=torch_compile, config=config)

        TimesFM_2p5_200M_torch.__init__ = _patched_init
        try:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_id,
                torch_compile=False,
            )
        finally:
            TimesFM_2p5_200M_torch.__init__ = _orig_init

        self.model.compile(timesfm.ForecastConfig(
            max_context=self.context_length,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        ))

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using TimesFM 2.5."""
        if self.model is None:
            self.load_model()

        ctx = context[-self.context_length:].astype(np.float64)

        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=[ctx],
        )

        point = point_forecast[0, :horizon]
        # quantile_forecast: (1, horizon, 11) — index 0=mean, 1=q10, ..., 5=q50, ..., 9=q90
        lower = quantile_forecast[0, :horizon, 1]  # q10
        upper = quantile_forecast[0, :horizon, 9]  # q90

        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            model_name=self._model_name,
        )

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Use TimesFM's 9 native deciles + monotone tail extrapolation.

        TimesFM 2.5's quantile head emits {Q10, Q20, ..., Q90}; we
        extrapolate linearly to the common grid's tails {Q2.5, Q5, Q95,
        Q97.5}. If a future TimesFM release accepts arbitrary quantile
        levels in `forecast()`, replace the extrapolation with a direct
        call.
        """
        from evaluation.density import DEFAULT_QUANTILE_LEVELS
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        if self.model is None:
            self.load_model()

        ctx = context[-self.context_length:].astype(np.float64)
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=[ctx],
        )
        # quantile_forecast: (1, horizon, 11) — col 0=mean, cols 1..9=Q10..Q90
        native_levels = np.arange(0.1, 0.91, 0.1)             # (9,)
        native_q = quantile_forecast[0, :horizon, 1:10]        # (horizon, 9)
        q_grid = _interp_quantiles_to_grid(native_levels, native_q, levels)

        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        i_lo = int(np.argmin(np.abs(levels - 0.10)))
        i_hi = int(np.argmin(np.abs(levels - 0.90)))
        return TSFMForecast(
            point=q_grid[:, median_idx],
            lower=q_grid[:, i_lo],
            upper=q_grid[:, i_hi],
            quantile_grid=q_grid,
            quantile_levels=levels,
            model_name=self._model_name,
        )


class MoiraiModel(BaseTSFM):
    """Wrapper for Salesforce Moirai 2.0.

    Uses uni2ts package.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    context_length : int
        Context window size.
    num_samples : int
        Number of forecast samples.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 512,
        num_samples: int = 20,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.device = device
        self.module = None
        self._model_name = model_id.split("/")[-1]

    def load_model(self) -> None:
        """Load Moirai 2.0 model via uni2ts."""
        import torch
        from uni2ts.model.moirai2 import Moirai2Module

        self.module = Moirai2Module.from_pretrained(self.model_id)
        if self.device == "cpu":
            self.module = self.module.float()
        self.module.eval()

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using Moirai 2.0."""
        if self.module is None:
            self.load_model()

        from uni2ts.model.moirai2 import Moirai2Forecast

        ctx = context[-self.context_length:].astype(np.float32)

        predictor = Moirai2Forecast(
            module=self.module,
            prediction_length=horizon,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            module_kwargs=dict(
                num_samples=self.num_samples,
            ),
        )

        # Moirai2Forecast.predict expects List[np.ndarray]
        # Each array has shape (past_time, target_dim=1)
        past_target = [ctx.reshape(-1, 1)]
        result = predictor.predict(past_target=past_target)

        # result shape: (batch=1, quantiles=9, horizon)
        # Quantile levels: 0.1, 0.2, ..., 0.9
        quantiles = result[0]  # (9, horizon)
        point = quantiles[4]   # median (0.5 quantile)
        lower = quantiles[0]   # 0.1 quantile
        upper = quantiles[8]   # 0.9 quantile

        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            model_name=self._model_name,
        )

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Native 9 quantiles {Q10..Q90} + monotone tail extrapolation.

        Internally Moirai 2.0 computes quantiles from MC samples; we
        elevate num_samples to DENSITY_NUM_SAMPLES for the duration of
        the call so the native deciles are less noisy.
        """
        from evaluation.density import DEFAULT_QUANTILE_LEVELS
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        if self.module is None:
            self.load_model()

        from uni2ts.model.moirai2 import Moirai2Forecast

        ctx = context[-self.context_length:].astype(np.float32)
        predictor = Moirai2Forecast(
            module=self.module,
            prediction_length=horizon,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            module_kwargs=dict(num_samples=DENSITY_NUM_SAMPLES),
        )
        past_target = [ctx.reshape(-1, 1)]
        result = predictor.predict(past_target=past_target)
        # result[0] shape: (9, horizon), levels 0.1..0.9
        native_q = result[0].T                                   # (horizon, 9)
        native_levels = np.arange(0.1, 0.91, 0.1)
        q_grid = _interp_quantiles_to_grid(native_levels, native_q, levels)

        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        i_lo = int(np.argmin(np.abs(levels - 0.10)))
        i_hi = int(np.argmin(np.abs(levels - 0.90)))
        return TSFMForecast(
            point=q_grid[:, median_idx],
            lower=q_grid[:, i_lo],
            upper=q_grid[:, i_hi],
            quantile_grid=q_grid,
            quantile_levels=levels,
            model_name=self._model_name,
        )


class LagLlamaModel(BaseTSFM):
    """Wrapper for Lag-Llama (probabilistic, decoder-only).

    Uses the lag-llama package with GluonTS integration.

    Parameters
    ----------
    context_length : int
        Context window size.
    num_samples : int
        Number of forecast samples for probabilistic output.
    n_layer : int
        Number of transformer layers.
    n_head : int
        Number of attention heads.
    n_embd_per_head : int
        Embedding dimension per head.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        context_length: int = 512,
        num_samples: int = 100,
        n_layer: int = 8,
        n_head: int = 4,
        n_embd_per_head: int = 36,
        device: str = "cpu",
        max_epochs: int = 0,
    ):
        self.context_length = context_length
        self.num_samples = num_samples
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd_per_head = n_embd_per_head
        self.device = device
        self.max_epochs = max_epochs
        self.ckpt_path = None
        self._predictors = {}  # cache by horizon
        self._model_name = "Lag-Llama"

    def load_model(self) -> None:
        """Download Lag-Llama checkpoint from HuggingFace."""
        from huggingface_hub import hf_hub_download
        self.ckpt_path = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
        )
        self._shim_gluonts_loss_module()

    @staticmethod
    def _shim_gluonts_loss_module() -> None:
        """Restore gluonts.torch.modules.loss for Lag-Llama's estimator import.

        Lag-Llama's estimator.py imports DistributionLoss and
        NegativeLogLikelihood from gluonts.torch.modules.loss, a module
        removed in gluonts >= 0.15. Both symbols are only used as
        default values for a training-time `loss` argument that is
        never invoked during inference, so we install a stub module
        with no-op classes so the import succeeds.
        """
        import sys
        import types

        mod_name = "gluonts.torch.modules.loss"
        if mod_name in sys.modules:
            return
        loss_mod = types.ModuleType(mod_name)

        class DistributionLoss:  # pragma: no cover - stub
            def __init__(self, *args, **kwargs):
                pass

        class NegativeLogLikelihood(DistributionLoss):  # pragma: no cover - stub
            pass

        loss_mod.DistributionLoss = DistributionLoss
        loss_mod.NegativeLogLikelihood = NegativeLogLikelihood
        sys.modules[mod_name] = loss_mod
        # Also attach to the parent module so `from gluonts.torch.modules import loss` works.
        import gluonts.torch.modules as _parent
        setattr(_parent, "loss", loss_mod)

    def _get_predictor_impl(self, horizon: int):
        """Build a fresh predictor for a given horizon (no caching)."""
        import torch
        self._shim_gluonts_loss_module()
        from lag_llama.gluon.estimator import LagLlamaEstimator

        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

        try:
            estimator = LagLlamaEstimator(
                prediction_length=horizon,
                context_length=self.context_length,
                input_size=1,
                n_layer=self.n_layer,
                n_head=self.n_head,
                n_embd_per_head=self.n_embd_per_head,
                rope_scaling=None,
                scaling="mean",
                time_feat=True,
                nonnegative_pred_samples=True,
                num_parallel_samples=self.num_samples,
                ckpt_path=self.ckpt_path,
                trainer_kwargs={"max_epochs": self.max_epochs},
                device=torch.device(self.device),
            )
            import lightning.pytorch as pl
            pl.seed_everything(42, workers=True)
            lightning_module = estimator.create_lightning_module()
            transformation = estimator.create_transformation()
            return estimator.create_predictor(transformation, lightning_module)
        finally:
            torch.load = _orig_load

    def _get_predictor(self, horizon: int):
        """Get or create a cached predictor for a given horizon."""
        if horizon not in self._predictors:
            self._predictors[horizon] = self._get_predictor_impl(horizon)
        return self._predictors[horizon]

    def _make_dataset(self, series: np.ndarray):
        """Create a GluonTS PandasDataset from a 1-D numpy array."""
        from gluonts.dataset.pandas import PandasDataset
        arr = series.astype(np.float32)
        dates = pd.date_range(end="2025-01-01", periods=len(arr), freq="B")
        return PandasDataset.from_long_dataframe(
            pd.DataFrame({
                "target": arr,
                "item_id": "item",
                "timestamp": dates,
            }),
            target="target",
            item_id="item_id",
            timestamp="timestamp",
        )

    def fine_tune_predictor(self, train_data: np.ndarray, horizon: int):
        """Fine-tune Lag-Llama on train_data and return a predictor.

        Parameters
        ----------
        train_data : np.ndarray
            Training time series (full rolling window).
        horizon : int
            Forecast horizon.

        Returns
        -------
        predictor
            A fine-tuned GluonTS predictor.
        """
        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator

        if self.ckpt_path is None:
            self.load_model()

        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

        try:
            estimator = LagLlamaEstimator(
                prediction_length=horizon,
                context_length=self.context_length,
                input_size=1,
                n_layer=self.n_layer,
                n_head=self.n_head,
                n_embd_per_head=self.n_embd_per_head,
                rope_scaling=None,
                scaling="mean",
                time_feat=True,
                nonnegative_pred_samples=True,
                num_parallel_samples=self.num_samples,
                ckpt_path=self.ckpt_path,
                trainer_kwargs={
                    "max_epochs": self.max_epochs,
                    "enable_progress_bar": False,
                },
                device=torch.device(self.device),
            )
            import lightning.pytorch as pl
            pl.seed_everything(42, workers=True)

            dataset = self._make_dataset(train_data)
            predictor = estimator.train(dataset, cache_data=True, shuffle_buffer_length=1000)
        finally:
            torch.load = _orig_load

        return predictor

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using Lag-Llama."""
        if self.ckpt_path is None:
            self.load_model()

        ctx = context[-self.context_length:].astype(np.float32)
        dataset = self._make_dataset(ctx)

        predictor = self._get_predictor(horizon)
        forecasts = list(predictor.predict(dataset))
        fc = forecasts[0]

        # fc.samples: (num_samples, horizon)
        samples = fc.samples
        point = np.median(samples, axis=0)
        lower = np.percentile(samples, 10, axis=0)
        upper = np.percentile(samples, 90, axis=0)

        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            samples=samples,
            model_name=self._model_name,
        )

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Empirical quantiles from DENSITY_NUM_SAMPLES MC samples.

        Lag-Llama has no native quantile API; we draw extra samples
        through its standard sampling path. The cached predictor is
        invalidated and rebuilt with the elevated sample count so we do
        not leak the higher num_samples back into legacy `predict()`.
        """
        from evaluation.density import DEFAULT_QUANTILE_LEVELS
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        if self.ckpt_path is None:
            self.load_model()

        # Build a density-specific predictor keyed separately so we don't
        # rebuild the estimator on every call (the original code popped the
        # cache each time, causing a full checkpoint reload per forecast date).
        density_key = (horizon, DENSITY_NUM_SAMPLES)
        if density_key not in self._predictors:
            orig_num_samples = self.num_samples
            self.num_samples = DENSITY_NUM_SAMPLES
            try:
                self._predictors[density_key] = self._get_predictor_impl(horizon)
            finally:
                self.num_samples = orig_num_samples

        ctx = context[-self.context_length:].astype(np.float32)
        dataset = self._make_dataset(ctx)
        predictor = self._predictors[density_key]
        forecasts = list(predictor.predict(dataset))
        samples = forecasts[0].samples         # (num_samples, horizon)

        q_grid = np.quantile(samples, levels, axis=0).T   # (horizon, K)
        q_grid = np.maximum.accumulate(q_grid, axis=1)
        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        i_lo = int(np.argmin(np.abs(levels - 0.10)))
        i_hi = int(np.argmin(np.abs(levels - 0.90)))
        return TSFMForecast(
            point=q_grid[:, median_idx],
            lower=q_grid[:, i_lo],
            upper=q_grid[:, i_hi],
            samples=samples,
            quantile_grid=q_grid,
            quantile_levels=levels,
            model_name=self._model_name,
        )

    def predict_with_predictor(self, predictor, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using a pre-trained/fine-tuned predictor."""
        ctx = context[-self.context_length:].astype(np.float32)
        dataset = self._make_dataset(ctx)

        forecasts = list(predictor.predict(dataset))
        fc = forecasts[0]

        samples = fc.samples
        point = np.median(samples, axis=0)
        lower = np.percentile(samples, 10, axis=0)
        upper = np.percentile(samples, 90, axis=0)

        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            samples=samples,
            model_name=f"{self._model_name}-FT",
        )


class KronosModel(BaseTSFM):
    """Wrapper for Kronos (finance-specific, K-line foundation model).

    Kronos is trained on OHLCV candlestick data, not univariate volatility.
    We create synthetic OHLCV from the RV series as an adaptation:
        open = previous close, high = max(open, close) * (1 + noise),
        low = min(open, close) * (1 - noise), close = RV_t.
    Results should be interpreted with this caveat.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for Kronos.
    tokenizer_id : str
        HuggingFace model identifier for the Kronos tokenizer.
    context_length : int
        Max context window.
    sample_count : int
        Number of forecast paths to average.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_id: str = "NeoQuasar/Kronos-base",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        context_length: int = 512,
        sample_count: int = 5,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.context_length = context_length
        self.sample_count = sample_count
        self.device = device
        self.predictor = None
        self._model_name = "Kronos"

    def load_model(self) -> None:
        """Load Kronos tokenizer + model from HuggingFace."""
        import sys
        from pathlib import Path
        kronos_path = str(Path(__file__).resolve().parent.parent.parent / "vendor" / "Kronos")
        if kronos_path not in sys.path:
            sys.path.insert(0, kronos_path)

        from model import Kronos as KronosNet, KronosTokenizer, KronosPredictor

        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        model = KronosNet.from_pretrained(self.model_id)

        self.predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            max_context=self.context_length,
        )

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using Kronos with synthetic OHLCV adaptation."""
        if self.predictor is None:
            self.load_model()

        ctx = context[-self.context_length:]

        # Build synthetic OHLCV from univariate RV
        close = ctx.copy()
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        high = np.maximum(open_prices, close) * 1.001
        low = np.minimum(open_prices, close) * 0.999

        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
        })

        # Create timestamps for context and forecast periods
        # Kronos's calc_time_stamps expects pd.Series (with .dt accessor), not DatetimeIndex
        x_timestamp = pd.Series(pd.date_range(end="2025-01-01", periods=len(ctx), freq="B"))
        y_timestamp = pd.Series(pd.date_range(
            start=x_timestamp.iloc[-1] + pd.tseries.offsets.BDay(1),
            periods=horizon, freq="B",
        ))

        pred_df = self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=1.0,
            top_p=0.9,
            sample_count=self.sample_count,
            verbose=False,
        )

        # Extract close column as point forecast
        point = pred_df['close'].values

        return TSFMForecast(
            point=point,
            model_name=self._model_name,
        )


class TotoModel(BaseTSFM):
    """Wrapper for Datadog Toto (Student-T mixture output, decoder-only).

    Uses the `toto-ts` package.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    context_length : int
        Maximum context window length.
    num_samples : int
        Number of forecast samples.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        context_length: int = 512,
        num_samples: int = 20,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.device = device
        self.forecaster = None
        self._model_name = "Toto"

    def load_model(self) -> None:
        """Load Toto model from HuggingFace."""
        from toto.model.toto import Toto
        from toto.inference.forecaster import TotoForecaster

        toto = Toto.from_pretrained(self.model_id).to(self.device)
        self.forecaster = TotoForecaster(toto.model)

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using Toto."""
        if self.forecaster is None:
            self.load_model()

        import torch
        from toto.data.util.dataset import MaskedTimeseries

        ctx = context[-self.context_length:].astype(np.float32)
        T = len(ctx)

        # Toto expects (batch, n_variables, time_steps) for all time-indexed fields.
        device = self.device
        series = torch.tensor(ctx, dtype=torch.float32).reshape(1, 1, T).to(device)
        timestamp_seconds = torch.zeros(1, 1, T, dtype=torch.float32).to(device)
        time_interval_seconds = torch.full((1, 1), 86400.0, dtype=torch.float32).to(device)

        masked_ts = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),
            id_mask=torch.zeros_like(series, dtype=torch.long),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        forecast = self.forecaster.forecast(
            masked_ts,
            prediction_length=horizon,
            num_samples=self.num_samples,
            samples_per_batch=self.num_samples,
        )

        # forecast.median: (batch=1, n_variables=1, horizon)
        # forecast.samples: (batch=1, n_variables=1, horizon, num_samples)
        point = forecast.median.cpu().numpy()[0, 0, :]  # (horizon,)
        lower = forecast.quantile(0.1).cpu().numpy()[0, 0, :]
        upper = forecast.quantile(0.9).cpu().numpy()[0, 0, :]

        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            model_name=self._model_name,
        )

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Toto's parametric mixture supports arbitrary quantiles via `forecast.quantile()`.

        We elevate num_samples to DENSITY_NUM_SAMPLES so the mixture is
        evaluated with enough Monte Carlo precision for the tail levels.
        """
        from evaluation.density import DEFAULT_QUANTILE_LEVELS
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        if self.forecaster is None:
            self.load_model()

        import torch
        from toto.data.util.dataset import MaskedTimeseries

        ctx = context[-self.context_length:].astype(np.float32)
        T = len(ctx)
        device = self.device
        series = torch.tensor(ctx, dtype=torch.float32).reshape(1, 1, T).to(device)
        timestamp_seconds = torch.zeros(1, 1, T, dtype=torch.float32).to(device)
        time_interval_seconds = torch.full((1, 1), 86400.0, dtype=torch.float32).to(device)
        masked_ts = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),
            id_mask=torch.zeros_like(series, dtype=torch.long),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )
        forecast = self.forecaster.forecast(
            masked_ts,
            prediction_length=horizon,
            num_samples=DENSITY_NUM_SAMPLES,
            samples_per_batch=DENSITY_NUM_SAMPLES,
        )
        # forecast.quantile(q) -> (batch=1, n_variables=1, horizon)
        q_grid = np.stack(
            [forecast.quantile(float(level)).cpu().numpy()[0, 0, :] for level in levels],
            axis=1,
        )  # (horizon, K)
        q_grid = np.maximum.accumulate(q_grid, axis=1)

        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        i_lo = int(np.argmin(np.abs(levels - 0.10)))
        i_hi = int(np.argmin(np.abs(levels - 0.90)))
        return TSFMForecast(
            point=q_grid[:, median_idx],
            lower=q_grid[:, i_lo],
            upper=q_grid[:, i_hi],
            quantile_grid=q_grid,
            quantile_levels=levels,
            model_name=self._model_name,
        )


class SundialModel(BaseTSFM):
    """Wrapper for Sundial (flow-matching generative, ICML 2025 Oral).

    Uses HuggingFace transformers with trust_remote_code=True.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    context_length : int
        Maximum context window length.
    num_samples : int
        Number of forecast samples.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_id: str = "thuml/sundial-base-128m",
        context_length: int = 512,
        num_samples: int = 20,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.device = device
        self.model = None
        self._model_name = "Sundial"

    def load_model(self) -> None:
        """Load Sundial model from HuggingFace."""
        from transformers import AutoModelForCausalLM
        from transformers import DynamicCache
        import torch

        # Sundial's modeling code calls DynamicCache.get_max_length() which was
        # removed in newer transformers versions.  Add it back as an alias.
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = (
                DynamicCache.get_max_cache_shape
                if hasattr(DynamicCache, "get_max_cache_shape")
                else lambda self: None
            )
        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length=0, layer_idx=0):
                if hasattr(self, "get_seq_length"):
                    return self.get_seq_length(layer_idx=layer_idx)
                return 0
            DynamicCache.get_usable_length = _get_usable_length

        # Always use float32 — bfloat16 causes dtype mismatches between
        # RevIN normalization (float32) and model weights (bfloat16).
        # Model is only 128M params so float32 fits easily on GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

        # Sundial was written for transformers <4.50 where do_sample=False routed
        # through _greedy_search() (which Sundial overrides with attention-mask and
        # multi-sample flow-matching logic).  In transformers >=4.50, _greedy_search
        # is removed and both modes use _sample().  We redirect _sample back to
        # Sundial's _greedy_search with the correct argument bridging.
        import types

        # Restore _extract_past_from_model_output (removed in transformers >=4.50).
        if not hasattr(self.model, "_extract_past_from_model_output"):
            def _extract_past(outputs, standardize_cache_format=False):
                return getattr(outputs, "past_key_values", None)
            self.model._extract_past_from_model_output = _extract_past

        # Bridge _sample → _greedy_search
        _greedy = self.model._greedy_search

        def _patched_sample(
            self_inner, input_ids, logits_processor=None, stopping_criteria=None,
            generation_config=None, synced_gpus=False, streamer=None, **model_kwargs
        ):
            # _greedy_search expects past_key_values=None on first call
            # (it manages the cache internally). Remove any pre-initialized cache.
            model_kwargs.pop("past_key_values", None)
            model_kwargs.pop("cache_position", None)
            gc = generation_config
            return _greedy(
                input_ids=input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=gc.max_length if gc else None,
                pad_token_id=gc.pad_token_id if gc else None,
                eos_token_id=gc.eos_token_id if gc else None,
                output_attentions=gc.output_attentions if gc else False,
                output_hidden_states=gc.output_hidden_states if gc else False,
                output_scores=gc.output_scores if gc else False,
                output_logits=gc.output_logits if gc else False,
                return_dict_in_generate=False,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        self.model._sample = types.MethodType(_patched_sample, self.model)

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using Sundial."""
        if self.model is None:
            self.load_model()

        import torch

        ctx = context[-self.context_length:].astype(np.float32)
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        if self.device == "cuda":
            ctx_tensor = ctx_tensor.cuda()

        with torch.no_grad():
            samples = self.model.generate(
                ctx_tensor,
                max_new_tokens=horizon,
                num_samples=self.num_samples,
            )  # (1, num_samples, horizon)

        samples_np = samples.cpu().numpy().squeeze(0)  # (num_samples, horizon)
        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(1, -1)

        point = np.median(samples_np, axis=0)
        lower = np.percentile(samples_np, 10, axis=0)
        upper = np.percentile(samples_np, 90, axis=0)

        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            samples=samples_np,
            model_name=self._model_name,
        )

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Empirical quantiles from Sundial's flow-matching samples.

        Sundial has no native quantile API. We temporarily elevate
        num_samples to DENSITY_NUM_SAMPLES so the tail quantiles are
        sampled with enough precision.
        """
        from evaluation.density import DEFAULT_QUANTILE_LEVELS
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        if self.model is None:
            self.load_model()

        import torch
        ctx = context[-self.context_length:].astype(np.float32)
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        if self.device == "cuda":
            ctx_tensor = ctx_tensor.cuda()
        with torch.no_grad():
            samples = self.model.generate(
                ctx_tensor,
                max_new_tokens=horizon,
                num_samples=DENSITY_NUM_SAMPLES,
            )
        samples_np = samples.cpu().numpy().squeeze(0)            # (num_samples, horizon)
        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(1, -1)
        q_grid = np.quantile(samples_np, levels, axis=0).T       # (horizon, K)
        q_grid = np.maximum.accumulate(q_grid, axis=1)

        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        i_lo = int(np.argmin(np.abs(levels - 0.10)))
        i_hi = int(np.argmin(np.abs(levels - 0.90)))
        return TSFMForecast(
            point=q_grid[:, median_idx],
            lower=q_grid[:, i_lo],
            upper=q_grid[:, i_hi],
            samples=samples_np,
            quantile_grid=q_grid,
            quantile_levels=levels,
            model_name=self._model_name,
        )


class MoiraiMoEModel(BaseTSFM):
    """Wrapper for Salesforce Moirai-MoE (sparse Mixture of Experts).

    Uses the same uni2ts package as Moirai 2.0 but with MoE-specific modules.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    context_length : int
        Context window size.
    num_samples : int
        Number of forecast samples.
    patch_size : int
        Patch size for MoE architecture.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-moe-1.0-R-small",
        context_length: int = 512,
        num_samples: int = 20,
        patch_size: int = 16,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.device = device
        self.module = None
        self._model_name = model_id.split("/")[-1]

    def load_model(self) -> None:
        """Load Moirai-MoE model via uni2ts."""
        import torch
        from uni2ts.model.moirai_moe import MoiraiMoEModule

        self.module = MoiraiMoEModule.from_pretrained(self.model_id)
        if self.device == "cpu":
            self.module = self.module.float()
        self.module.eval()

    # Moirai-MoE's architecture requires a fixed 512-token context
    # (its MoE routing and positional encodings are trained at that size).
    # For shorter context_length, we left-pad to 512 and mark padding via
    # past_is_pad so the model ignores padded positions.
    _FIXED_CTX = 512

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using Moirai-MoE."""
        if self.module is None:
            self.load_model()

        import torch
        from uni2ts.model.moirai_moe import MoiraiMoEForecast

        ctx = context[-self.context_length:].astype(np.float32)
        T = len(ctx)

        # Pad to fixed 512 if context is shorter
        if T < self._FIXED_CTX:
            pad_len = self._FIXED_CTX - T
            ctx_padded = np.concatenate([np.zeros(pad_len, dtype=np.float32), ctx])
            is_pad = np.concatenate([np.ones(pad_len, dtype=bool), np.zeros(T, dtype=bool)])
            obs = np.concatenate([np.zeros(pad_len, dtype=bool), np.ones(T, dtype=bool)])
        else:
            ctx_padded = ctx
            is_pad = np.zeros(T, dtype=bool)
            obs = np.ones(T, dtype=bool)

        L = len(ctx_padded)
        past_target = torch.tensor(ctx_padded.reshape(1, L, 1), dtype=torch.float32)
        past_observed = torch.tensor(obs.reshape(1, L, 1), dtype=torch.bool)
        past_is_pad = torch.tensor(is_pad.reshape(1, L), dtype=torch.bool)

        forecast_module = MoiraiMoEForecast(
            module=self.module,
            prediction_length=horizon,
            context_length=self._FIXED_CTX,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
        )

        with torch.no_grad():
            # Output shape: (batch=1, num_samples, horizon)
            samples = forecast_module.forward(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
                num_samples=self.num_samples,
            )

        samples_np = samples.numpy()[0]  # (num_samples, horizon)
        point = np.median(samples_np, axis=0)
        lower = np.percentile(samples_np, 10, axis=0)
        upper = np.percentile(samples_np, 90, axis=0)

        return TSFMForecast(
            point=point,
            lower=lower,
            upper=upper,
            samples=samples_np,
            model_name=self._model_name,
        )

    def predict_density(
        self,
        context: np.ndarray,
        horizon: int,
        levels: Optional[np.ndarray] = None,
    ) -> TSFMForecast:
        """Empirical quantiles from Moirai-MoE's MC samples (elevated count)."""
        from evaluation.density import DEFAULT_QUANTILE_LEVELS
        levels = np.asarray(
            DEFAULT_QUANTILE_LEVELS if levels is None else levels, dtype=float
        )
        if self.module is None:
            self.load_model()

        import torch
        from uni2ts.model.moirai_moe import MoiraiMoEForecast

        ctx = context[-self.context_length:].astype(np.float32)
        T = len(ctx)
        if T < self._FIXED_CTX:
            pad_len = self._FIXED_CTX - T
            ctx_padded = np.concatenate([np.zeros(pad_len, dtype=np.float32), ctx])
            is_pad = np.concatenate([np.ones(pad_len, dtype=bool), np.zeros(T, dtype=bool)])
            obs = np.concatenate([np.zeros(pad_len, dtype=bool), np.ones(T, dtype=bool)])
        else:
            ctx_padded = ctx
            is_pad = np.zeros(T, dtype=bool)
            obs = np.ones(T, dtype=bool)

        L = len(ctx_padded)
        past_target = torch.tensor(ctx_padded.reshape(1, L, 1), dtype=torch.float32)
        past_observed = torch.tensor(obs.reshape(1, L, 1), dtype=torch.bool)
        past_is_pad = torch.tensor(is_pad.reshape(1, L), dtype=torch.bool)

        forecast_module = MoiraiMoEForecast(
            module=self.module,
            prediction_length=horizon,
            context_length=self._FIXED_CTX,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            patch_size=self.patch_size,
            num_samples=DENSITY_NUM_SAMPLES,
        )
        with torch.no_grad():
            samples = forecast_module.forward(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
                num_samples=DENSITY_NUM_SAMPLES,
            )
        samples_np = samples.numpy()[0]                          # (num_samples, horizon)
        q_grid = np.quantile(samples_np, levels, axis=0).T       # (horizon, K)
        q_grid = np.maximum.accumulate(q_grid, axis=1)

        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        i_lo = int(np.argmin(np.abs(levels - 0.10)))
        i_hi = int(np.argmin(np.abs(levels - 0.90)))
        return TSFMForecast(
            point=q_grid[:, median_idx],
            lower=q_grid[:, i_lo],
            upper=q_grid[:, i_hi],
            samples=samples_np,
            quantile_grid=q_grid,
            quantile_levels=levels,
            model_name=self._model_name,
        )


class TTMModel(BaseTSFM):
    """Wrapper for IBM Granite TTM (Tiny Time Mixers) r2.1.

    Uses frequency prefix tuning for daily-frequency zero-shot forecasting.

    Density note: TTM is point-only. Its TSMixer architecture emits a
    single point forecast per horizon step, with no quantile head or
    sampling mechanism. predict_density() inherits the BaseTSFM default,
    which will raise NotImplementedError. In the density-evaluation
    harness TTM should be excluded with the comment: "TTM is excluded
    from density evaluation because it does not emit a predictive
    distribution."

    Parameters
    ----------
    model_path : str
        HuggingFace model identifier.
    context_length : int
        Context window size.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
        context_length: int = 512,
        device: str = "cpu",
        **kwargs,
    ):
        self.model_path = model_path
        self.context_length = context_length
        self.device = device
        self.model = None
        self._model_name = "TTM"
        # Daily frequency token from DEFAULT_FREQUENCY_MAPPING
        self._freq_token_id = 8  # 'd' / 'D' -> 8

    def load_model(self) -> None:
        """Load TTM r2.1 model with daily frequency support."""
        from tsfm_public import get_model

        # get_model selects the best matching branch automatically.
        # Available r2.1 branches have specific context/prediction combos:
        #   512-96, 512-48, 360-60, 180-60, 90-30, 52-16
        # We request prediction_length=96 for ctx=512, but smaller for
        # shorter contexts to match available branches.
        pred_len_map = {512: 96, 360: 60, 256: 48, 180: 60, 128: 30, 90: 30, 52: 16}
        pred_len = pred_len_map.get(self.context_length, 48)

        self.model = get_model(
            model_path=self.model_path,
            context_length=self.context_length,
            prediction_length=pred_len,
            freq="D",
        )
        if self.device == "cuda":
            import torch
            self.model = self.model.cuda()
        self.model.eval()

    def predict(self, context: np.ndarray, horizon: int) -> TSFMForecast:
        """Generate forecast using TTM."""
        if self.model is None:
            self.load_model()

        import torch

        ctx = context[-self.context_length:].astype(np.float32)
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        freq_token = torch.tensor([[self._freq_token_id]])

        if self.device == "cuda":
            ctx_tensor = ctx_tensor.cuda()
            freq_token = freq_token.cuda()

        with torch.no_grad():
            out = self.model(past_values=ctx_tensor, freq_token=freq_token)

        pred_all = out.prediction_outputs[0, :, 0].cpu().numpy()
        # Trim to requested horizon
        point = pred_all[:horizon]

        return TSFMForecast(
            point=point,
            lower=point * 0.8,  # rough CI placeholder
            upper=point * 1.2,
            model_name=self._model_name,
        )


def get_foundation_model(model_name: str, **kwargs) -> BaseTSFM:
    """Factory function to get a TSFM by name.

    Parameters
    ----------
    model_name : str
        One of: 'chronos-bolt-small', 'chronos-bolt-base',
                'timesfm-2.5', 'moirai-2.0-small'.

    Returns
    -------
    BaseTSFM
        Instantiated model wrapper.
    """
    models = {
        'chronos-bolt-small': lambda: ChronosModel(
            "amazon/chronos-bolt-small", **kwargs
        ),
        'chronos-bolt-base': lambda: ChronosModel(
            "amazon/chronos-bolt-base", **kwargs
        ),
        'timesfm-2.5': lambda: TimesFMModel(
            "google/timesfm-2.5-200m-pytorch", **kwargs
        ),
        'moirai-2.0-small': lambda: MoiraiModel(
            "Salesforce/moirai-2.0-R-small", **kwargs
        ),
        'lag-llama': lambda: LagLlamaModel(**kwargs),
        'kronos': lambda: KronosModel(**kwargs),
        'toto': lambda: TotoModel(**kwargs),
        'sundial': lambda: SundialModel(**kwargs),
        'moirai-moe-small': lambda: MoiraiMoEModel(
            "Salesforce/moirai-moe-1.0-R-small", **kwargs
        ),
        'ttm': lambda: TTMModel(**kwargs),
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(models.keys())}"
        )
    return models[model_name]()
