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


@dataclass
class TSFMForecast:
    """Container for foundation model forecast output."""
    point: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None
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
            self.pipeline = ChronosBoltPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            )
        else:
            from chronos import ChronosPipeline
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
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
        else:
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

    def _get_predictor(self, horizon: int):
        """Get or create a cached predictor for a given horizon."""
        if horizon not in self._predictors:
            import torch
            from lag_llama.gluon.estimator import LagLlamaEstimator

            # Lag-Llama checkpoint contains non-weight objects (GluonTS distributions);
            # PyTorch 2.6+ defaults weights_only=True which rejects them.
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
                predictor = estimator.create_predictor(transformation, lightning_module)
                self._predictors[horizon] = predictor
            finally:
                torch.load = _orig_load
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

        # Toto expects (n_variables, time_steps) for the series
        device = self.device
        series = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)
        timestamp_seconds = torch.zeros(1, T, dtype=torch.float32).to(device)
        time_interval_seconds = torch.full((1,), 86400.0).to(device)

        masked_ts = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),
            id_mask=torch.zeros_like(series),
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
            model_name=self._model_name,
        )


class TTMModel(BaseTSFM):
    """Wrapper for IBM Granite TTM (Tiny Time Mixers) r2.1.

    Uses frequency prefix tuning for daily-frequency zero-shot forecasting.

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
