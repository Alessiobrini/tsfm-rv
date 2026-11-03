"""
data.py — Shared window construction and rolling block generation.
"""

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from fine_tuning.protocol import RefitBlock, RollingRefitConfig


@dataclass
class WindowedDataset:
    """Context/target windows for one segment of one asset."""

    contexts: np.ndarray
    targets_log: np.ndarray
    targets_level: np.ndarray
    forecast_dates: pd.DatetimeIndex
    context_length: int
    horizon: int

    def __len__(self) -> int:
        return int(len(self.targets_level))

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


def generate_refit_blocks(
    n_obs: int,
    cfg: RollingRefitConfig,
) -> List[RefitBlock]:
    """Create expanding-window train/validation/test blocks."""

    blocks: List[RefitBlock] = []
    train_start = 0
    train_end = cfg.min_train_size
    refit_id = 0

    while True:
        val_start = train_end
        val_end = val_start + cfg.validation_size
        test_start = val_end
        test_end = test_start + cfg.test_size

        if test_end > n_obs:
            break

        blocks.append(
            RefitBlock(
                refit_id=refit_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        refit_id += 1
        if cfg.expanding_train:
            train_end += cfg.step_size
        else:
            train_start += cfg.step_size
            train_end += cfg.step_size

    return blocks


def direct_level_target(values: np.ndarray, forecast_idx: int, horizon: int) -> float:
    """Direct target in levels: next-day RV or future average RV."""

    if horizon == 1:
        return float(values[forecast_idx])
    return float(np.mean(values[forecast_idx:forecast_idx + horizon]))


def build_windowed_dataset(
    series: pd.Series,
    forecast_indices: Iterable[int],
    context_length: int,
    horizon: int,
) -> WindowedDataset:
    """Build aligned context windows with log targets."""

    values = series.values.astype(np.float64)
    dates = series.index

    contexts = []
    targets_level = []
    forecast_dates = []

    for idx in forecast_indices:
        if idx < context_length:
            continue
        if idx + horizon > len(values):
            continue

        ctx = values[idx - context_length:idx]
        if np.any(~np.isfinite(ctx)):
            continue

        target_level = direct_level_target(values, idx, horizon)
        if not np.isfinite(target_level) or target_level <= 0:
            continue

        contexts.append(ctx.astype(np.float32))
        targets_level.append(target_level)
        forecast_dates.append(dates[idx])

    if not contexts:
        return WindowedDataset(
            contexts=np.empty((0, context_length), dtype=np.float32),
            targets_log=np.empty((0,), dtype=np.float32),
            targets_level=np.empty((0,), dtype=np.float32),
            forecast_dates=pd.DatetimeIndex([]),
            context_length=context_length,
            horizon=horizon,
        )

    targets_level_arr = np.asarray(targets_level, dtype=np.float32)
    return WindowedDataset(
        contexts=np.stack(contexts).astype(np.float32),
        targets_log=np.log(targets_level_arr).astype(np.float32),
        targets_level=targets_level_arr,
        forecast_dates=pd.DatetimeIndex(forecast_dates),
        context_length=context_length,
        horizon=horizon,
    )


def block_train_indices(block: RefitBlock, horizon: int) -> range:
    """Forecast indices whose direct targets stay inside the training segment."""

    stop = max(block.train_start, block.train_end - horizon + 1)
    return range(block.train_start, stop)


def block_val_indices(block: RefitBlock, horizon: int) -> range:
    """Forecast indices whose direct targets stay inside the validation segment."""

    stop = max(block.val_start, block.val_end - horizon + 1)
    return range(block.val_start, stop)


def block_test_indices(block: RefitBlock, horizon: int) -> range:
    """Forecast indices whose direct targets stay inside the test segment."""

    stop = max(block.test_start, block.test_end - horizon + 1)
    return range(block.test_start, stop)

