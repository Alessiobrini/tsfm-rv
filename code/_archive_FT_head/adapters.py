"""
adapters.py — Head-only adapter interfaces for the aligned TSFM study.

This file defines the experiment-facing contract for head-only fine-tuning.
For Chronos-Bolt-S, we implement a concrete "frozen backbone + trainable
forecast head" approach by:

1. keeping the Chronos pipeline fully frozen,
2. extracting deterministic features from its quantile forecasts, and
3. training a small regression head on log-targets.

This is intentionally conservative: it avoids undocumented model-internal
training hooks while preserving the aligned FT-Head protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

import numpy as np

from fine_tuning.data import WindowedDataset
from fine_tuning.protocol import FineTuneRuntimeConfig


class HeadOnlyAdapterError(RuntimeError):
    """Raised when a head-only adapter cannot be executed."""


@dataclass
class ValidationSelection:
    """Validation diagnostics returned by a fitted adapter."""

    best_epoch: int | None = None
    best_val_qlike: float | None = None
    notes: str = ""
    extra: dict | None = None


def _validate_selection_metric(metric: str) -> str:
    metric_u = metric.upper()
    if metric_u not in {"QLIKE", "MSE", "MAE"}:
        raise ValueError(f"Unsupported selection metric '{metric}'. Choose from QLIKE, MSE, MAE.")
    return metric_u


def _validate_target_transform(transform: str) -> str:
    transform_l = transform.lower()
    if transform_l not in {"log", "level"}:
        raise ValueError(f"Unsupported target transform '{transform}'. Choose from log or level.")
    return transform_l


def _metric_value(metric: str, actual: np.ndarray, forecast: np.ndarray) -> float:
    metric_u = _validate_selection_metric(metric)
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if metric_u == "QLIKE":
        ratio = actual / np.maximum(forecast, 1e-10)
        return float(np.mean(ratio - np.log(np.maximum(ratio, 1e-10)) - 1))
    if metric_u == "MSE":
        return float(np.mean((actual - forecast) ** 2))
    return float(np.mean(np.abs(actual - forecast)))


def _fit_linear_log_head(
    features: np.ndarray, y_log: np.ndarray, ridge_alpha: float = 0.0
) -> tuple[np.ndarray, float]:
    """Fit a log-space linear head. Returns (coef, bias_correction).

    bias_correction = exp(0.5 * residual_variance) corrects for the Jensen gap:
    OLS in log space estimates E[log y | x], but we need exp(E[log y | x]) * correction
    to approximate E[y | x] in levels.
    """
    X = features.astype(np.float64)
    y = y_log.astype(np.float64)
    n = len(X)
    if ridge_alpha > 0.0:
        feat_mean = X.mean(axis=0)
        feat_std = np.maximum(X.std(axis=0), 1e-8)
        X_s = (X - feat_mean) / feat_std
        A = np.column_stack([np.ones(n), X_s])
        reg = ridge_alpha * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        c_s = np.linalg.solve(A.T @ A + reg, A.T @ y)
        intercept = c_s[0] - np.dot(c_s[1:], feat_mean / feat_std)
        coef = np.concatenate([[intercept], c_s[1:] / feat_std])
    else:
        A = np.column_stack([np.ones(n), X])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    A_orig = np.column_stack([np.ones(n), X])
    bias_correction = float(np.exp(0.5 * np.var(y - A_orig @ coef)))
    return coef.astype(np.float32), bias_correction


def _predict_linear_log_head(
    features: np.ndarray, coef: np.ndarray, bias_correction: float = 1.0
) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(features), dtype=np.float32), features]).astype(np.float32)
    pred_log = X_design @ coef
    return (np.exp(pred_log) * bias_correction).astype(np.float32)


class BaseHeadOnlyAdapter(ABC):
    """Common interface for frozen-backbone / trainable-head fine-tuning."""

    model_name: str = "unknown"

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.runtime_cfg = runtime_cfg
        self.selection_metric = _validate_selection_metric(selection_metric)
        self.train_target_transform = _validate_target_transform(train_target_transform)

    def training_targets(self, ds: WindowedDataset) -> np.ndarray:
        return ds.targets_log.astype(np.float32) if self.train_target_transform == "log" else ds.targets_level.astype(np.float32)

    @abstractmethod
    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        """Fit the trainable head using log targets and select by validation QLIKE."""

    @abstractmethod
    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        """Return level forecasts for the provided context windows."""


class PlaceholderHeadOnlyAdapter(BaseHeadOnlyAdapter):
    """Temporary adapter that makes missing model wiring explicit."""

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        raise HeadOnlyAdapterError(
            f"Head-only fine-tuning is not yet implemented for '{self.model_name}'. "
            "The rolling protocol is in place, but this model still needs a "
            "backbone-specific trainable-head adapter."
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        raise HeadOnlyAdapterError(
            f"Cannot predict because '{self.model_name}' head-only adapter is not implemented."
        )


class ChronosBoltHeadAdapter(PlaceholderHeadOnlyAdapter):
    model_name = "chronos-bolt-small"
    _HF_REPO = "amazon/chronos-bolt-small"
    _PIPELINE_CACHE: dict = {}
    _RAW_CACHE: dict = {}

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self.pipeline = None
        self.head = None
        self._head_state = None
        self._selection = ValidationSelection(notes="Frozen Chronos quantile features + trainable regression head.")
        self._raw_val_scale = None
        self._cache_key = (self.context_length, self.horizon, self.runtime_cfg.device)

    def _load_pipeline(self) -> None:
        import torch
        from chronos.chronos_bolt import ChronosBoltPipeline

        if self.pipeline is not None:
            return
        if self._cache_key in self._PIPELINE_CACHE:
            self.pipeline = self._PIPELINE_CACHE[self._cache_key]
            return

        dtype = torch.float32 if self.runtime_cfg.device == "cpu" else torch.bfloat16
        try:
            self.pipeline = ChronosBoltPipeline.from_pretrained(
                self._HF_REPO,
                device_map=self.runtime_cfg.device,
                torch_dtype=dtype,
            )
        except TypeError:
            # Older Chronos releases accept `dtype` instead of `torch_dtype`.
            self.pipeline = ChronosBoltPipeline.from_pretrained(
                self._HF_REPO,
                device_map=self.runtime_cfg.device,
                dtype=dtype,
            )
        self._PIPELINE_CACHE[self._cache_key] = self.pipeline

    def _raw_outputs(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import torch

        self._load_pipeline()
        cache = self._RAW_CACHE.setdefault(self._cache_key, {})
        q10_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        q50_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        q90_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        mean_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                q10_out[i], q50_out[i], q90_out[i], mean_out[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx)

        if not missing_contexts:
            return q10_out, q50_out, q90_out, mean_out

        batch_size = max(1, self.runtime_cfg.batch_size)

        for start in range(0, len(missing_contexts), batch_size):
            batch = np.asarray(missing_contexts[start:start + batch_size], dtype=np.float32)
            ctx_tensor = torch.tensor(batch, dtype=torch.float32)
            quantiles, mean = self.pipeline.predict_quantiles(
                ctx_tensor,
                prediction_length=self.horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )

            q = quantiles.detach().cpu().numpy()  # (B, H, 3)
            m = mean.detach().cpu().numpy()       # (B, H)

            q10 = q[:, :, 0].astype(np.float32)
            q50 = q[:, :, 1].astype(np.float32)
            q90 = q[:, :, 2].astype(np.float32)
            m = m.astype(np.float32)

            for j in range(len(batch)):
                pos = missing_positions[start + j]
                key = np.asarray(missing_contexts[start + j], dtype=np.float32).tobytes()
                cache[key] = (q10[j], q50[j], q90[j], m[j])
                q10_out[pos] = q10[j]
                q50_out[pos] = q50[j]
                q90_out[pos] = q90[j]
                mean_out[pos] = m[j]

        return q10_out, q50_out, q90_out, mean_out

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q10, q50, q90, mean = self._raw_outputs(contexts)
        if len(mean) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Aggregate along the direct horizon so the head always maps into
        # the same scalar target definition used by the econometric study.
        agg_mean = np.mean(mean, axis=1)
        agg_q10 = np.mean(q10, axis=1)
        agg_q50 = np.mean(q50, axis=1)
        agg_q90 = np.mean(q90, axis=1)

        raw_level = np.clip(agg_q50, 1e-10, None).astype(np.float32)
        features = np.column_stack([
            np.log(np.clip(agg_q50, 1e-10, None)),
            np.log(np.clip(agg_mean, 1e-10, None)),
            np.log(np.clip(agg_q10, 1e-10, None)),
            np.log(np.clip(agg_q90, 1e-10, None)),
        ]).astype(np.float32)
        return features, raw_level

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("Chronos adapter received empty train/validation data.")

        X_train, _ = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        # Diagnose raw Chronos scale first using the frozen median forecast.
        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))
        self._raw_val_scale = raw_val_scale

        candidates = {
            "raw_q50": raw_val,
        }
        try:
            candidate_heads = {
                "linear_q50": _fit_linear_log_head(X_train[:, [0]], y_train),
                "linear_q50_mean": _fit_linear_log_head(X_train[:, :2], y_train),
                "linear_all": _fit_linear_log_head(X_train, y_train, ridge_alpha=1.0),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"Chronos linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_q50": raw_val,
            "linear_q50": _predict_linear_log_head(X_val[:, [0]], *candidate_heads["linear_q50"]),
            "linear_q50_mean": _predict_linear_log_head(X_val[:, :2], *candidate_heads["linear_q50_mean"]),
            "linear_all": _predict_linear_log_head(X_val, *candidate_heads["linear_all"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)

        if best_name == "raw_q50":
            self.head = None
            self._head_state = {"variant": best_name, "coef": None, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "coef": coef.copy(), "bias_correction": bc}

        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)
        self._selection = ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=(
                "Frozen Chronos-Bolt quantile features with a linear log-space "
                f"calibration head selected by validation {self.selection_metric}."
            ),
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )
        return self._selection

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        if self._head_state and self._head_state.get("variant") == "linear_q50":
            return _predict_linear_log_head(X[:, [0]], self.head, bc)
        if self._head_state and self._head_state.get("variant") == "linear_q50_mean":
            return _predict_linear_log_head(X[:, :2], self.head, bc)
        return _predict_linear_log_head(X, self.head, bc)


class TimesFMHeadAdapter(PlaceholderHeadOnlyAdapter):
    model_name = "timesfm-2.5"
    _MODEL_CACHE: dict = {}
    _RAW_CACHE: dict = {}

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self.model = None
        self.head = None
        self._head_state = None
        self._cache_key = (self.context_length, self.runtime_cfg.device)

    def _load_model(self) -> None:
        try:
            import timesfm
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
        except ModuleNotFoundError as exc:
            raise HeadOnlyAdapterError(
                "TimesFM is not installed in the current Python environment. "
                "Activate the environment that has TimesFM, or install it with "
                "`pip install \"timesfm @ git+https://github.com/google-research/timesfm.git\"`."
            ) from exc

        import torch

        if self.model is not None:
            return
        if self._cache_key in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[self._cache_key]
            return

        if self.runtime_cfg.device == "cuda":
            torch.set_float32_matmul_precision("high")

        _orig_init = TimesFM_2p5_200M_torch.__init__

        def _patched_init(self_inner, torch_compile=True, config=None, **_extra):
            _orig_init(self_inner, torch_compile=torch_compile, config=config)

        TimesFM_2p5_200M_torch.__init__ = _patched_init
        try:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch",
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

        self._MODEL_CACHE[self._cache_key] = self.model

    def _raw_outputs(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._load_model()
        cache = self._RAW_CACHE.setdefault((self.context_length, self.horizon, self.runtime_cfg.device), {})
        point_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        q10_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        q50_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        q90_out = np.empty((len(contexts), self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                point_out[i], q10_out[i], q50_out[i], q90_out[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx.astype(np.float64, copy=False))

        if not missing_contexts:
            return point_out, q10_out, q50_out, q90_out

        batch_size = max(1, self.runtime_cfg.batch_size)
        for start in range(0, len(missing_contexts), batch_size):
            batch = missing_contexts[start:start + batch_size]
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=self.horizon,
                inputs=batch,
            )
            point_batch = point_forecast[:, :self.horizon].astype(np.float32)
            q10_batch = quantile_forecast[:, :self.horizon, 1].astype(np.float32)
            q50_batch = quantile_forecast[:, :self.horizon, 5].astype(np.float32)
            q90_batch = quantile_forecast[:, :self.horizon, 9].astype(np.float32)

            for j in range(len(batch)):
                pos = missing_positions[start + j]
                key = np.asarray(missing_contexts[start + j], dtype=np.float32).tobytes()
                cache[key] = (point_batch[j], q10_batch[j], q50_batch[j], q90_batch[j])
                point_out[pos] = point_batch[j]
                q10_out[pos] = q10_batch[j]
                q50_out[pos] = q50_batch[j]
                q90_out[pos] = q90_batch[j]

        return point_out, q10_out, q50_out, q90_out

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point, q10, q50, q90 = self._raw_outputs(contexts)
        if len(point) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        agg_point = np.mean(point, axis=1)
        agg_q10 = np.mean(q10, axis=1)
        agg_q50 = np.mean(q50, axis=1)
        agg_q90 = np.mean(q90, axis=1)

        raw_level = np.clip(agg_point, 1e-10, None).astype(np.float32)
        features = np.column_stack([
            np.log(np.clip(agg_point, 1e-10, None)),
            np.log(np.clip(agg_q10, 1e-10, None)),
            np.log(np.clip(agg_q50, 1e-10, None)),
            np.log(np.clip(agg_q90, 1e-10, None)),
        ]).astype(np.float32)
        return features, raw_level

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("TimesFM adapter received empty train/validation data.")

        X_train, _ = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))

        try:
            candidate_heads = {
                "linear_point": _fit_linear_log_head(X_train[:, [0]], y_train),
                "linear_point_q50": _fit_linear_log_head(X_train[:, [0, 2]], y_train),
                "linear_all": _fit_linear_log_head(X_train, y_train, ridge_alpha=1.0),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"TimesFM linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_point": raw_val,
            "linear_point": _predict_linear_log_head(X_val[:, [0]], *candidate_heads["linear_point"]),
            "linear_point_q50": _predict_linear_log_head(X_val[:, [0, 2]], *candidate_heads["linear_point_q50"]),
            "linear_all": _predict_linear_log_head(X_val, *candidate_heads["linear_all"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)
        if best_name == "raw_point":
            self.head = None
            self._head_state = {"variant": best_name, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "bias_correction": bc}
        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)

        return ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=f"Frozen TimesFM forecast summaries with a linear log-space calibration head selected by validation {self.selection_metric}.",
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        if self._head_state and self._head_state.get("variant") == "linear_point":
            return _predict_linear_log_head(X[:, [0]], self.head, bc)
        if self._head_state and self._head_state.get("variant") == "linear_point_q50":
            return _predict_linear_log_head(X[:, [0, 2]], self.head, bc)
        return _predict_linear_log_head(X, self.head, bc)


class MoiraiMoEHeadAdapter(BaseHeadOnlyAdapter):
    model_name = "moirai-moe-small"
    _MODULE_CACHE: dict = {}
    _RAW_CACHE: dict = {}
    _FIXED_CTX = 512
    _PATCH_SIZE = 16

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self.module = None
        self.head = None
        self._head_state = None
        self.num_samples = 20
        self._cache_key = (
            self.context_length,
            self.horizon,
            self.runtime_cfg.device,
            self.num_samples,
        )

    def _load_module(self) -> None:
        try:
            from uni2ts.model.moirai_moe import MoiraiMoEModule
        except ModuleNotFoundError as exc:
            raise HeadOnlyAdapterError(
                "uni2ts is not installed. Install it with "
                "`pip install uni2ts`."
            ) from exc

        if self.module is not None:
            return
        if self._cache_key in self._MODULE_CACHE:
            self.module = self._MODULE_CACHE[self._cache_key]
            return

        self.module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small")
        if self.runtime_cfg.device == "cpu":
            self.module = self.module.float()
        self.module.eval()
        self._MODULE_CACHE[self._cache_key] = self.module

    def _raw_samples(self, contexts: np.ndarray) -> np.ndarray:
        import torch
        from uni2ts.model.moirai_moe import MoiraiMoEForecast

        self._load_module()
        cache = self._RAW_CACHE.setdefault(self._cache_key, {})
        outputs = np.empty((len(contexts), self.num_samples, self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                outputs[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx)

        if not missing_contexts:
            return outputs

        batch_size = max(1, self.runtime_cfg.batch_size)

        for start in range(0, len(missing_contexts), batch_size):
            batch = np.asarray(missing_contexts[start:start + batch_size], dtype=np.float32)
            B = len(batch)
            T_actual = batch.shape[1]

            if T_actual < self._FIXED_CTX:
                pad_len = self._FIXED_CTX - T_actual
                ctx_padded = np.concatenate(
                    [np.zeros((B, pad_len), dtype=np.float32), batch], axis=1
                )
            else:
                ctx_padded = batch[:, -self._FIXED_CTX:]

            L = ctx_padded.shape[1]
            past_target = torch.tensor(ctx_padded.reshape(B, L, 1), dtype=torch.float32)
            past_observed = torch.tensor(obs.reshape(B, L, 1), dtype=torch.bool)
            past_is_pad = torch.tensor(is_pad.reshape(B, L), dtype=torch.bool)

            forecast_module = MoiraiMoEForecast(
                module=self.module,
                prediction_length=self.horizon,
                context_length=self._FIXED_CTX,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                patch_size=self._PATCH_SIZE,
                num_samples=self.num_samples,
            )

            with torch.no_grad():
                samples = forecast_module.forward(
                    past_target=past_target,
                    past_observed_target=past_observed,
                    past_is_pad=past_is_pad,
                    num_samples=self.num_samples,
                )  # (B, num_samples, horizon)

            samples_np = samples.numpy().astype(np.float32)
            for j in range(B):
                pos = missing_positions[start + j]
                key = np.asarray(missing_contexts[start + j], dtype=np.float32).tobytes()
                cache[key] = samples_np[j]
                outputs[pos] = samples_np[j]

        return outputs

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = self._raw_samples(contexts)
        if len(samples) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        median_path = np.median(samples, axis=1)
        mean_path = np.mean(samples, axis=1)
        q10_path = np.percentile(samples, 10, axis=1)
        q90_path = np.percentile(samples, 90, axis=1)

        agg_median = np.mean(median_path, axis=1)
        agg_mean = np.mean(mean_path, axis=1)
        agg_q10 = np.mean(q10_path, axis=1)
        agg_q90 = np.mean(q90_path, axis=1)

        raw_level = np.clip(agg_median, 1e-10, None).astype(np.float32)
        features = np.column_stack([
            np.log(np.clip(agg_median, 1e-10, None)),
            np.log(np.clip(agg_mean, 1e-10, None)),
            np.log(np.clip(agg_q10, 1e-10, None)),
            np.log(np.clip(agg_q90, 1e-10, None)),
        ]).astype(np.float32)
        return features, raw_level

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("Moirai-MoE adapter received empty train/validation data.")

        X_train, _ = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))

        try:
            candidate_heads = {
                "linear_median": _fit_linear_log_head(X_train[:, [0]], y_train),
                "linear_median_mean": _fit_linear_log_head(X_train[:, :2], y_train),
                "linear_all": _fit_linear_log_head(X_train, y_train, ridge_alpha=1.0),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"Moirai-MoE linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_median": raw_val,
            "linear_median": _predict_linear_log_head(X_val[:, [0]], *candidate_heads["linear_median"]),
            "linear_median_mean": _predict_linear_log_head(X_val[:, :2], *candidate_heads["linear_median_mean"]),
            "linear_all": _predict_linear_log_head(X_val, *candidate_heads["linear_all"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)
        if best_name == "raw_median":
            self.head = None
            self._head_state = {"variant": best_name, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "bias_correction": bc}
        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)

        return ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=f"Frozen Moirai-MoE sample summaries with a linear log-space calibration head selected by validation {self.selection_metric}.",
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        if self._head_state and self._head_state.get("variant") == "linear_median":
            return _predict_linear_log_head(X[:, [0]], self.head, bc)
        if self._head_state and self._head_state.get("variant") == "linear_median_mean":
            return _predict_linear_log_head(X[:, :2], self.head, bc)
        return _predict_linear_log_head(X, self.head, bc)


class LagLlamaHeadAdapter(BaseHeadOnlyAdapter):
    model_name = "lag-llama"
    _PREDICTOR_CACHE: dict = {}
    _CKPT_CACHE: dict = {}
    _RAW_CACHE: dict = {}
    _N_LAYER = 8
    _N_HEAD = 4
    _N_EMBD_PER_HEAD = 36

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self._predictor = None
        self.head = None
        self._head_state = None
        self.num_samples = 20
        self._cache_key = (
            self.context_length,
            self.horizon,
            self.runtime_cfg.device,
            self.num_samples,
        )

    def _load_predictor(self) -> None:
        if self._predictor is not None:
            return
        if self._cache_key in self._PREDICTOR_CACHE:
            self._predictor = self._PREDICTOR_CACHE[self._cache_key]
            return

        try:
            import torch
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ModuleNotFoundError as exc:
            raise HeadOnlyAdapterError(
                "lag_llama is not installed. Install it from the Lag-Llama repository."
            ) from exc

        from huggingface_hub import hf_hub_download
        import lightning.pytorch as pl

        if "ckpt" not in self._CKPT_CACHE:
            self._CKPT_CACHE["ckpt"] = hf_hub_download(
                repo_id="time-series-foundation-models/Lag-Llama",
                filename="lag-llama.ckpt",
            )
        ckpt_path = self._CKPT_CACHE["ckpt"]

        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            estimator = LagLlamaEstimator(
                prediction_length=self.horizon,
                context_length=self.context_length,
                input_size=1,
                n_layer=self._N_LAYER,
                n_head=self._N_HEAD,
                n_embd_per_head=self._N_EMBD_PER_HEAD,
                rope_scaling=None,
                scaling="mean",
                time_feat=True,
                nonnegative_pred_samples=True,
                num_parallel_samples=self.num_samples,
                ckpt_path=ckpt_path,
                trainer_kwargs={"max_epochs": 0, "enable_progress_bar": False},
                device=torch.device(self.runtime_cfg.device),
            )
            pl.seed_everything(42, workers=True)
            lightning_module = estimator.create_lightning_module()
            transformation = estimator.create_transformation()
            self._predictor = estimator.create_predictor(transformation, lightning_module)
        finally:
            torch.load = _orig_load

        self._PREDICTOR_CACHE[self._cache_key] = self._predictor

    def _raw_samples(self, contexts: np.ndarray) -> np.ndarray:
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        self._load_predictor()
        cache = self._RAW_CACHE.setdefault(self._cache_key, {})
        outputs = np.empty((len(contexts), self.num_samples, self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                outputs[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx)

        if not missing_contexts:
            return outputs

        # Build a multi-item GluonTS dataset: one item per missing context.
        # All contexts share a synthetic end date to match LagLlamaModel._make_dataset().
        items = []
        for ctx in missing_contexts:
            arr = ctx.astype(np.float32)
            start_ts = pd.date_range(end="2025-01-01", periods=len(arr), freq="B")[0]
            items.append({"start": pd.Period(start_ts, freq="B"), "target": arr})

        dataset = ListDataset(items, freq="B")
        forecasts = list(self._predictor.predict(dataset))

        for j, fc in enumerate(forecasts):
            pos = missing_positions[j]
            samples_np = fc.samples[:self.num_samples, :self.horizon].astype(np.float32)
            key = np.asarray(missing_contexts[j], dtype=np.float32).tobytes()
            cache[key] = samples_np
            outputs[pos] = samples_np

        return outputs

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = self._raw_samples(contexts)
        if len(samples) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        median_path = np.median(samples, axis=1)
        mean_path = np.mean(samples, axis=1)
        q10_path = np.percentile(samples, 10, axis=1)
        q90_path = np.percentile(samples, 90, axis=1)

        agg_median = np.mean(median_path, axis=1)
        agg_mean = np.mean(mean_path, axis=1)
        agg_q10 = np.mean(q10_path, axis=1)
        agg_q90 = np.mean(q90_path, axis=1)

        raw_level = np.clip(agg_median, 1e-10, None).astype(np.float32)
        features = np.column_stack([
            np.log(np.clip(agg_median, 1e-10, None)),
            np.log(np.clip(agg_mean, 1e-10, None)),
            np.log(np.clip(agg_q10, 1e-10, None)),
            np.log(np.clip(agg_q90, 1e-10, None)),
        ]).astype(np.float32)
        return features, raw_level

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("Lag-Llama adapter received empty train/validation data.")

        X_train, _ = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))

        try:
            candidate_heads = {
                "linear_median": _fit_linear_log_head(X_train[:, [0]], y_train),
                "linear_median_mean": _fit_linear_log_head(X_train[:, :2], y_train),
                "linear_all": _fit_linear_log_head(X_train, y_train, ridge_alpha=1.0),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"Lag-Llama linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_median": raw_val,
            "linear_median": _predict_linear_log_head(X_val[:, [0]], *candidate_heads["linear_median"]),
            "linear_median_mean": _predict_linear_log_head(X_val[:, :2], *candidate_heads["linear_median_mean"]),
            "linear_all": _predict_linear_log_head(X_val, *candidate_heads["linear_all"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)
        if best_name == "raw_median":
            self.head = None
            self._head_state = {"variant": best_name, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "bias_correction": bc}
        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)

        return ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=f"Frozen Lag-Llama sample summaries with a linear log-space calibration head selected by validation {self.selection_metric}.",
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        if self._head_state and self._head_state.get("variant") == "linear_median":
            return _predict_linear_log_head(X[:, [0]], self.head, bc)
        if self._head_state and self._head_state.get("variant") == "linear_median_mean":
            return _predict_linear_log_head(X[:, :2], self.head, bc)
        return _predict_linear_log_head(X, self.head, bc)


class SundialHeadAdapter(PlaceholderHeadOnlyAdapter):
    model_name = "sundial"
    _MODEL_CACHE: dict = {}
    _RAW_CACHE: dict = {}

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self.model = None
        self.num_samples = 20
        self._cache_key = (
            self.context_length,
            self.horizon,
            self.runtime_cfg.device,
            self.num_samples,
        )

    def _load_model(self) -> None:
        import types
        import torch
        from transformers import AutoModelForCausalLM, DynamicCache

        if self.model is not None:
            return
        if self._cache_key in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[self._cache_key]
            return

        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = (
                DynamicCache.get_max_cache_shape
                if hasattr(DynamicCache, "get_max_cache_shape")
                else lambda self: None
            )
        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length=0, layer_idx=0):
                # Sundial's older generation mixin expects a cache helper that
                # newer transformers removed. For inference here, using the
                # current cache length is the intended compatibility behavior.
                if hasattr(self, "get_seq_length"):
                    return self.get_seq_length(layer_idx=layer_idx)
                return 0
            DynamicCache.get_usable_length = _get_usable_length

        self.model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        if self.runtime_cfg.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

        if not hasattr(self.model, "_extract_past_from_model_output"):
            def _extract_past(outputs, standardize_cache_format=False):
                return getattr(outputs, "past_key_values", None)
            self.model._extract_past_from_model_output = _extract_past

        _greedy = self.model._greedy_search

        def _patched_sample(
            self_inner, input_ids, logits_processor=None, stopping_criteria=None,
            generation_config=None, synced_gpus=False, streamer=None, **model_kwargs
        ):
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
        self._MODEL_CACHE[self._cache_key] = self.model

    def _raw_samples(self, contexts: np.ndarray) -> np.ndarray:
        import torch

        self._load_model()
        cache = self._RAW_CACHE.setdefault(self._cache_key, {})
        outputs = np.empty((len(contexts), self.num_samples, self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                outputs[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx)

        if not missing_contexts:
            return outputs

        batch_size = max(1, self.runtime_cfg.batch_size)

        for start in range(0, len(missing_contexts), batch_size):
            batch = np.asarray(missing_contexts[start:start + batch_size], dtype=np.float32)
            ctx_tensor = torch.tensor(batch, dtype=torch.float32)
            if self.runtime_cfg.device == "cuda":
                ctx_tensor = ctx_tensor.cuda()

            with torch.no_grad():
                samples = self.model.generate(
                    ctx_tensor,
                    max_new_tokens=self.horizon,
                    num_samples=self.num_samples,
                )

            samples_np = samples.detach().cpu().numpy().astype(np.float32)
            if samples_np.ndim == 2:
                samples_np = samples_np[:, np.newaxis, :]
            for j, sample in enumerate(samples_np):
                pos = missing_positions[start + j]
                key = np.asarray(missing_contexts[start + j], dtype=np.float32).tobytes()
                cache[key] = sample
                outputs[pos] = sample

        return outputs

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = self._raw_samples(contexts)
        if len(samples) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        median_path = np.median(samples, axis=1)
        mean_path = np.mean(samples, axis=1)
        q10_path = np.percentile(samples, 10, axis=1)
        q90_path = np.percentile(samples, 90, axis=1)

        agg_median = np.mean(median_path, axis=1)
        agg_mean = np.mean(mean_path, axis=1)
        agg_q10 = np.mean(q10_path, axis=1)
        agg_q90 = np.mean(q90_path, axis=1)

        raw_level = np.clip(agg_median, 1e-10, None).astype(np.float32)
        features = np.column_stack([
            np.log(np.clip(agg_median, 1e-10, None)),
            np.log(np.clip(agg_mean, 1e-10, None)),
            np.log(np.clip(agg_q10, 1e-10, None)),
            np.log(np.clip(agg_q90, 1e-10, None)),
        ]).astype(np.float32)
        return features, raw_level

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("Sundial adapter received empty train/validation data.")

        X_train, _ = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))

        try:
            candidate_heads = {
                "linear_median": _fit_linear_log_head(X_train[:, [0]], y_train),
                "linear_median_mean": _fit_linear_log_head(X_train[:, :2], y_train),
                "linear_all": _fit_linear_log_head(X_train, y_train, ridge_alpha=1.0),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"Sundial linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_median": raw_val,
            "linear_median": _predict_linear_log_head(X_val[:, [0]], *candidate_heads["linear_median"]),
            "linear_median_mean": _predict_linear_log_head(X_val[:, :2], *candidate_heads["linear_median_mean"]),
            "linear_all": _predict_linear_log_head(X_val, *candidate_heads["linear_all"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)
        if best_name == "raw_median":
            self.head = None
            self._head_state = {"variant": best_name, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "bias_correction": bc}
        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)

        return ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=f"Frozen Sundial sample summaries with a linear log-space calibration head selected by validation {self.selection_metric}.",
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        if self._head_state and self._head_state.get("variant") == "linear_median":
            return _predict_linear_log_head(X[:, [0]], self.head, bc)
        if self._head_state and self._head_state.get("variant") == "linear_median_mean":
            return _predict_linear_log_head(X[:, :2], self.head, bc)
        return _predict_linear_log_head(X, self.head, bc)


class TTMHeadAdapter(PlaceholderHeadOnlyAdapter):
    model_name = "ttm"
    _MODEL_CACHE: dict = {}
    _RAW_CACHE: dict = {}

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self.model = None
        self.head = None
        self._model_name = "TTM"
        self._freq_token_id = 8
        self._cache_key = (self.context_length, self.horizon, self.runtime_cfg.device)

    def _load_model(self) -> None:
        from tsfm_public import get_model

        if self.model is not None:
            return
        if self._cache_key in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[self._cache_key]
            return

        pred_len_map = {512: 96, 360: 60, 256: 48, 180: 60, 128: 30, 90: 30, 52: 16}
        pred_len = pred_len_map.get(self.context_length, 48)

        self.model = get_model(
            model_path="ibm-granite/granite-timeseries-ttm-r2",
            context_length=self.context_length,
            prediction_length=pred_len,
            freq="D",
        )
        if self.runtime_cfg.device == "cuda":
            import torch
            self.model = self.model.cuda()
        self.model.eval()
        self._MODEL_CACHE[self._cache_key] = self.model

    def _raw_point_forecasts(self, contexts: np.ndarray) -> np.ndarray:
        import torch

        self._load_model()
        cache = self._RAW_CACHE.setdefault(self._cache_key, {})
        preds = np.empty((len(contexts), self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                preds[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx)

        if not missing_contexts:
            return preds

        batch_size = max(1, self.runtime_cfg.batch_size)

        for start in range(0, len(missing_contexts), batch_size):
            batch = np.asarray(missing_contexts[start:start + batch_size], dtype=np.float32)
            ctx_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(-1)
            freq_token = torch.full((len(batch), 1), self._freq_token_id, dtype=torch.long)
            if self.runtime_cfg.device == "cuda":
                ctx_tensor = ctx_tensor.cuda()
                freq_token = freq_token.cuda()

            with torch.no_grad():
                out = self.model(past_values=ctx_tensor, freq_token=freq_token)
            pred_all = out.prediction_outputs[:, :, 0].detach().cpu().numpy()
            pred_batch = pred_all[:, :self.horizon].astype(np.float32)

            for j, pred in enumerate(pred_batch):
                pos = missing_positions[start + j]
                key = np.asarray(missing_contexts[start + j], dtype=np.float32).tobytes()
                cache[key] = pred
                preds[pos] = pred

        return preds

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_path = self._raw_point_forecasts(contexts)
        if len(raw_path) == 0:
            return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)

        agg_raw = np.mean(raw_path, axis=1).astype(np.float32)
        features = np.log(np.clip(agg_raw.reshape(-1, 1), 1e-10, None)).astype(np.float32)
        return features, np.clip(agg_raw, 1e-10, None)

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("TTM adapter received empty train/validation data.")

        X_train, raw_train = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))

        try:
            candidate_heads = {
                "linear_raw": _fit_linear_log_head(X_train, y_train),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"TTM linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_point": raw_val,
            "linear_raw": _predict_linear_log_head(X_val, *candidate_heads["linear_raw"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)
        if best_name == "raw_point":
            self.head = None
            self._head_state = {"variant": best_name, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "bias_correction": bc}
        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)

        return ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=f"Frozen TTM point forecasts with a linear log-space calibration head selected by validation {self.selection_metric}.",
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        return _predict_linear_log_head(X, self.head, bc)


class TotoHeadAdapter(BaseHeadOnlyAdapter):
    model_name = "toto"
    _MODEL_ID = "Datadog/Toto-Open-Base-1.0"
    _FORECASTER_CACHE: dict = {}
    _RAW_CACHE: dict = {}

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self.forecaster = None
        self.head = None
        self._head_state = None
        self.num_samples = 20
        self._cache_key = (
            self.context_length,
            self.horizon,
            self.runtime_cfg.device,
            self.num_samples,
        )

    def _load_forecaster(self) -> None:
        try:
            from toto.model.toto import Toto
            from toto.inference.forecaster import TotoForecaster
        except ModuleNotFoundError as exc:
            raise HeadOnlyAdapterError(
                "toto-ts is not installed. Install it from the Datadog Toto repository."
            ) from exc

        if self.forecaster is not None:
            return
        if self._cache_key in self._FORECASTER_CACHE:
            self.forecaster = self._FORECASTER_CACHE[self._cache_key]
            return

        toto = Toto.from_pretrained(self._MODEL_ID).to(self.runtime_cfg.device)
        self.forecaster = TotoForecaster(toto.model)
        self._FORECASTER_CACHE[self._cache_key] = self.forecaster

    def _raw_samples(self, contexts: np.ndarray) -> np.ndarray:
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        self._load_forecaster()
        cache = self._RAW_CACHE.setdefault(self._cache_key, {})
        outputs = np.empty((len(contexts), self.num_samples, self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                outputs[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx)

        if not missing_contexts:
            return outputs

        device = self.runtime_cfg.device
        batch_size = max(1, self.runtime_cfg.batch_size)

        for start in range(0, len(missing_contexts), batch_size):
            batch = np.asarray(missing_contexts[start:start + batch_size], dtype=np.float32)
            B, T = batch.shape

            # Toto expects (B, n_variables=1, time_steps)
            series = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
            masked_ts = MaskedTimeseries(
                series=series,
                padding_mask=torch.ones_like(series, dtype=torch.bool),
                id_mask=torch.zeros_like(series, dtype=torch.long),
                timestamp_seconds=torch.zeros(B, 1, T, dtype=torch.float32).to(device),
                time_interval_seconds=torch.full((B, 1), 86400.0, dtype=torch.float32).to(device),
            )

            forecast = self.forecaster.forecast(
                masked_ts,
                prediction_length=self.horizon,
                num_samples=self.num_samples,
                samples_per_batch=self.num_samples,
            )

            # forecast.samples: (B, 1, horizon, num_samples) → (B, num_samples, horizon)
            samples_np = forecast.samples.cpu().numpy()[:, 0, :, :].transpose(0, 2, 1).astype(np.float32)

            for j in range(B):
                pos = missing_positions[start + j]
                key = np.asarray(missing_contexts[start + j], dtype=np.float32).tobytes()
                cache[key] = samples_np[j]
                outputs[pos] = samples_np[j]

        return outputs

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = self._raw_samples(contexts)
        if len(samples) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        agg_median = np.mean(np.median(samples, axis=1), axis=1)
        agg_mean = np.mean(np.mean(samples, axis=1), axis=1)
        agg_q10 = np.mean(np.percentile(samples, 10, axis=1), axis=1)
        agg_q90 = np.mean(np.percentile(samples, 90, axis=1), axis=1)

        raw_level = np.clip(agg_median, 1e-10, None).astype(np.float32)
        features = np.column_stack([
            np.log(np.clip(agg_median, 1e-10, None)),
            np.log(np.clip(agg_mean, 1e-10, None)),
            np.log(np.clip(agg_q10, 1e-10, None)),
            np.log(np.clip(agg_q90, 1e-10, None)),
        ]).astype(np.float32)
        return features, raw_level

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("Toto adapter received empty train/validation data.")

        X_train, _ = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))

        try:
            candidate_heads = {
                "linear_median": _fit_linear_log_head(X_train[:, [0]], y_train),
                "linear_median_mean": _fit_linear_log_head(X_train[:, :2], y_train),
                "linear_all": _fit_linear_log_head(X_train, y_train, ridge_alpha=1.0),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"Toto linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_median": raw_val,
            "linear_median": _predict_linear_log_head(X_val[:, [0]], *candidate_heads["linear_median"]),
            "linear_median_mean": _predict_linear_log_head(X_val[:, :2], *candidate_heads["linear_median_mean"]),
            "linear_all": _predict_linear_log_head(X_val, *candidate_heads["linear_all"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)
        if best_name == "raw_median":
            self.head = None
            self._head_state = {"variant": best_name, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "bias_correction": bc}
        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)

        return ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=f"Frozen Toto sample summaries with a linear log-space calibration head selected by validation {self.selection_metric}.",
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        if self._head_state and self._head_state.get("variant") == "linear_median":
            return _predict_linear_log_head(X[:, [0]], self.head, bc)
        if self._head_state and self._head_state.get("variant") == "linear_median_mean":
            return _predict_linear_log_head(X[:, :2], self.head, bc)
        return _predict_linear_log_head(X, self.head, bc)


class ChronosBoltBaseHeadAdapter(ChronosBoltHeadAdapter):
    model_name = "chronos-bolt-base"
    _HF_REPO = "amazon/chronos-bolt-base"
    _PIPELINE_CACHE: dict = {}
    _RAW_CACHE: dict = {}


class Moirai20SmallHeadAdapter(BaseHeadOnlyAdapter):
    model_name = "moirai-2.0-small"
    _MODULE_CACHE: dict = {}
    _RAW_CACHE: dict = {}
    _FIXED_CTX = 512
    _PATCH_SIZE = 16
    _N_QUANTILES = 9

    def __init__(
        self,
        context_length: int,
        horizon: int,
        runtime_cfg: FineTuneRuntimeConfig,
        selection_metric: str = "QLIKE",
        train_target_transform: str = "log",
    ):
        super().__init__(
            context_length,
            horizon,
            runtime_cfg,
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        self.module = None
        self.head = None
        self._head_state = None
        self.num_samples = 20
        self._cache_key = (
            self.context_length,
            self.horizon,
            self.runtime_cfg.device,
            self.num_samples,
        )

    def _load_module(self) -> None:
        try:
            from uni2ts.model.moirai2 import Moirai2Module
        except ModuleNotFoundError as exc:
            raise HeadOnlyAdapterError(
                "uni2ts is not installed. Install it with `pip install uni2ts`."
            ) from exc

        if self.module is not None:
            return
        if self._cache_key in self._MODULE_CACHE:
            self.module = self._MODULE_CACHE[self._cache_key]
            return

        self.module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        if self.runtime_cfg.device == "cpu":
            self.module = self.module.float()
        self.module.eval()
        self._MODULE_CACHE[self._cache_key] = self.module

    def _raw_samples(self, contexts: np.ndarray) -> np.ndarray:
        from uni2ts.model.moirai2 import Moirai2Forecast

        self._load_module()
        cache = self._RAW_CACHE.setdefault(self._cache_key, {})
        outputs = np.empty((len(contexts), self._N_QUANTILES, self.horizon), dtype=np.float32)
        missing_positions = []
        missing_contexts = []

        for i, ctx in enumerate(contexts):
            key = ctx.astype(np.float32, copy=False).tobytes()
            cached = cache.get(key)
            if cached is not None:
                outputs[i] = cached
            else:
                missing_positions.append(i)
                missing_contexts.append(ctx)

        if not missing_contexts:
            return outputs

        batch_size = max(1, self.runtime_cfg.batch_size)

        for start in range(0, len(missing_contexts), batch_size):
            batch = np.asarray(missing_contexts[start:start + batch_size], dtype=np.float32)
            B = len(batch)
            T_actual = batch.shape[1]

            if T_actual < self._FIXED_CTX:
                pad_len = self._FIXED_CTX - T_actual
                ctx_padded = np.concatenate(
                    [np.zeros((B, pad_len), dtype=np.float32), batch], axis=1
                )
                is_pad = np.concatenate(
                    [np.ones((B, pad_len), dtype=bool), np.zeros((B, T_actual), dtype=bool)], axis=1
                )
                obs = np.concatenate(
                    [np.zeros((B, pad_len), dtype=bool), np.ones((B, T_actual), dtype=bool)], axis=1
                )
            else:
                ctx_padded = batch[:, -self._FIXED_CTX:]
                is_pad = np.zeros((B, self._FIXED_CTX), dtype=bool)
                obs = np.ones((B, self._FIXED_CTX), dtype=bool)

            L = ctx_padded.shape[1]
            predictor = Moirai2Forecast(
                module=self.module,
                prediction_length=self.horizon,
                context_length=L,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                module_kwargs=dict(num_samples=self.num_samples),
            )
            past_target = [ctx_padded[j].reshape(-1, 1) for j in range(B)]
            samples_np = np.asarray(predictor.predict(past_target=past_target), dtype=np.float32)
            for j in range(B):
                pos = missing_positions[start + j]
                key = np.asarray(missing_contexts[start + j], dtype=np.float32).tobytes()
                cache[key] = samples_np[j]
                outputs[pos] = samples_np[j]

        return outputs

    def _featurize(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = self._raw_samples(contexts)
        if len(samples) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        agg_median = np.mean(samples[:, 4, :], axis=1)
        agg_mean = np.mean(np.mean(samples, axis=1), axis=1)
        agg_q10 = np.mean(samples[:, 0, :], axis=1)
        agg_q90 = np.mean(samples[:, 8, :], axis=1)

        raw_level = np.clip(agg_median, 1e-10, None).astype(np.float32)
        features = np.column_stack([
            np.log(np.clip(agg_median, 1e-10, None)),
            np.log(np.clip(agg_mean, 1e-10, None)),
            np.log(np.clip(agg_q10, 1e-10, None)),
            np.log(np.clip(agg_q90, 1e-10, None)),
        ]).astype(np.float32)
        return features, raw_level

    def fit(
        self,
        train_data: WindowedDataset,
        val_data: WindowedDataset,
    ) -> ValidationSelection:
        if train_data.is_empty or val_data.is_empty:
            raise HeadOnlyAdapterError("Moirai-2.0-S adapter received empty train/validation data.")

        X_train, _ = self._featurize(train_data.contexts)
        X_val, raw_val = self._featurize(val_data.contexts)
        y_train = self.training_targets(train_data)
        y_val_level = val_data.targets_level.astype(np.float32)

        raw_val_qlike = _metric_value("QLIKE", y_val_level, raw_val)
        raw_val_scale = float(np.mean(raw_val / np.maximum(y_val_level, 1e-10)))

        try:
            candidate_heads = {
                "linear_median": _fit_linear_log_head(X_train[:, [0]], y_train),
                "linear_median_mean": _fit_linear_log_head(X_train[:, :2], y_train),
                "linear_all": _fit_linear_log_head(X_train, y_train, ridge_alpha=1.0),
            }
        except np.linalg.LinAlgError as exc:
            raise HeadOnlyAdapterError(f"Moirai-2.0-S linear calibration failed: {exc}") from exc

        val_candidates = {
            "raw_median": raw_val,
            "linear_median": _predict_linear_log_head(X_val[:, [0]], *candidate_heads["linear_median"]),
            "linear_median_mean": _predict_linear_log_head(X_val[:, :2], *candidate_heads["linear_median_mean"]),
            "linear_all": _predict_linear_log_head(X_val, *candidate_heads["linear_all"]),
        }
        scores = {name: _metric_value(self.selection_metric, y_val_level, pred) for name, pred in val_candidates.items()}
        best_name = min(scores, key=scores.get)
        if best_name == "raw_median":
            self.head = None
            self._head_state = {"variant": best_name, "bias_correction": 1.0}
        else:
            coef, bc = candidate_heads[best_name]
            self.head = coef
            self._head_state = {"variant": best_name, "bias_correction": bc}
        val_level_pred = self.predict_levels(val_data.contexts)
        val_qlike = _metric_value("QLIKE", y_val_level, val_level_pred)

        return ValidationSelection(
            best_epoch=1,
            best_val_qlike=scores[best_name],
            notes=f"Frozen Moirai-2.0-S sample summaries with a linear log-space calibration head selected by validation {self.selection_metric}.",
            extra={
                "raw_val_qlike": raw_val_qlike,
                "raw_val_scale_mean": raw_val_scale,
                "selected_variant": best_name,
                "selection_metric_value": scores[best_name],
                "selection_metric_name": self.selection_metric,
                "train_target_transform": self.train_target_transform,
                "selected_variant_qlike": val_qlike,
            },
        )

    def predict_levels(self, contexts: np.ndarray) -> np.ndarray:
        X, raw = self._featurize(contexts)
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.head is None:
            return raw.astype(np.float32)
        bc = float(self._head_state.get("bias_correction", 1.0)) if self._head_state else 1.0
        if self._head_state and self._head_state.get("variant") == "linear_median":
            return _predict_linear_log_head(X[:, [0]], self.head, bc)
        if self._head_state and self._head_state.get("variant") == "linear_median_mean":
            return _predict_linear_log_head(X[:, :2], self.head, bc)
        return _predict_linear_log_head(X, self.head, bc)


ADAPTER_REGISTRY: Dict[str, Type[BaseHeadOnlyAdapter]] = {
    "chronos-bolt-small": ChronosBoltHeadAdapter,
    "chronos-bolt-base": ChronosBoltBaseHeadAdapter,
    "timesfm-2.5": TimesFMHeadAdapter,
    "moirai-moe-small": MoiraiMoEHeadAdapter,
    "moirai-2.0-small": Moirai20SmallHeadAdapter,
    "lag-llama": LagLlamaHeadAdapter,
    "toto": TotoHeadAdapter,
    "sundial": SundialHeadAdapter,
    "ttm": TTMHeadAdapter,
}


def build_head_only_adapter(
    model_name: str,
    context_length: int,
    horizon: int,
    runtime_cfg: FineTuneRuntimeConfig,
    selection_metric: str = "QLIKE",
    train_target_transform: str = "log",
) -> BaseHeadOnlyAdapter:
    """Factory for model-specific head-only adapters."""

    if model_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown head-only fine-tuning model '{model_name}'. "
            f"Choose from {sorted(ADAPTER_REGISTRY)}."
        )
    return ADAPTER_REGISTRY[model_name](
        context_length=context_length,
        horizon=horizon,
        runtime_cfg=runtime_cfg,
        selection_metric=selection_metric,
        train_target_transform=train_target_transform,
    )
