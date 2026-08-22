"""
protocol.py — Dataclasses for the aligned TSFM fine-tuning study.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config import VOLARE_RESULTS_DIR


@dataclass(frozen=True)
class RollingRefitConfig:
    """Rolling refit schedule shared across all TSFMs."""

    min_train_size: int = 756
    validation_size: int = 126
    test_size: int = 126
    step_size: int = 126
    expanding_train: bool = True


@dataclass(frozen=True)
class FineTuneGrid:
    """Common grid for the first aligned head-only experiment."""

    models: List[str] = field(default_factory=lambda: [
        "chronos-bolt-small",
        "timesfm-2.5",
        "moirai-moe-small",
        "lag-llama",
        "sundial",
        "ttm",
    ])
    contexts: List[int] = field(default_factory=lambda: [128, 256, 512])
    horizons: List[int] = field(default_factory=lambda: [1, 22])
    asset_class: str = "stocks"
    selection_metric: str = "QLIKE"
    train_target_transform: str = "log"


@dataclass(frozen=True)
class FineTuneRuntimeConfig:
    """Runtime knobs that are model-agnostic."""

    device: str = "cpu"
    random_seed: int = 42
    max_epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 0.0
    patience: int = 3
    num_workers: int = 0


@dataclass(frozen=True)
class FineTuneExperimentConfig:
    """Top-level experiment configuration."""

    grid: FineTuneGrid = field(default_factory=FineTuneGrid)
    rolling: RollingRefitConfig = field(default_factory=RollingRefitConfig)
    runtime: FineTuneRuntimeConfig = field(default_factory=FineTuneRuntimeConfig)
    output_root: Path = VOLARE_RESULTS_DIR / "fine_tune_head"
    tickers: Optional[List[str]] = None


@dataclass(frozen=True)
class RefitBlock:
    """One rolling train/validation/test block."""

    refit_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

