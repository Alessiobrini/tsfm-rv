"""
config.py — Central configuration for realized volatility forecasting project.

All hyperparameters, file paths, model settings, and constants.
Modify this file to change experimental setup without touching model code.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# VOLARE dataset paths (long-format CSV, one file per asset class)
VOLARE_DATA_DIR = RAW_DIR / "volare"
VOLARE_STOCKS_FILE = VOLARE_DATA_DIR / "realized_variance_stocks.csv"
VOLARE_FOREX_FILE = VOLARE_DATA_DIR / "realized_variance_forex.csv"
VOLARE_FUTURES_FILE = VOLARE_DATA_DIR / "realized_variance_futures.csv"

VOLARE_RESULTS_DIR = RESULTS_DIR / "volare"

# ============================================================
# Random seed
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Asset selection (VOLARE)
# ============================================================
VOLARE_STOCK_TICKERS = [
    'AAPL', 'ADBE', 'AMD', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO',
    'CVX', 'DIS', 'GE', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM',
    'KO', 'MCD', 'META', 'MMM', 'MRK', 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL',
    'PG', 'PM', 'SHW', 'TRV', 'TSLA', 'UNH', 'V', 'VZ', 'WMT', 'XOM',
]
VOLARE_FX_TICKERS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']
VOLARE_FUTURES_TICKERS = ['C', 'CL', 'ES', 'GC', 'NG']
VOLARE_ALL_TICKERS = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS

# Quick-test subset (used as default when no --tickers argument is supplied)
REPRESENTATIVE_TICKERS = ['AAPL', 'JPM', 'AMZN', 'CAT']


# ============================================================
# Data settings
# ============================================================
@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # VOLARE column mappings (long-format CSV)
    volare_rv_col: str = "rv5"      # 5-min realized variance
    volare_bpv_col: str = "bv5"     # 5-min bipower variation
    volare_good_col: str = "rsp5"   # 5-min positive semivariance
    volare_bad_col: str = "rsn5"    # 5-min negative semivariance
    volare_rq_col: str = "rq5"      # 5-min realized quarticity

    # Outlier treatment
    winsorize: bool = False
    winsorize_pctile: float = 0.995


# ============================================================
# HAR model settings
# ============================================================
@dataclass
class HARConfig:
    """Configuration for HAR family models."""
    daily_lag: int = 1
    weekly_lag: int = 5
    monthly_lag: int = 22

    use_log_rv: bool = False
    hac_std_errors: bool = True
    hac_max_lags: int = 22

    models: List[str] = field(default_factory=lambda: [
        'HAR', 'HAR-J', 'HAR-RS', 'HARQ',
    ])


# ============================================================
# ARFIMA settings
# ============================================================
@dataclass
class ARFIMAConfig:
    """Configuration for ARFIMA model."""
    use_log_rv: bool = True
    max_ar: int = 2
    max_ma: int = 2
    d_method: str = "mle"


# ============================================================
# Foundation model settings
# ============================================================
@dataclass
class FoundationModelConfig:
    """Configuration for time series foundation models."""
    # Chronos-Bolt
    chronos_model_ids: List[str] = field(default_factory=lambda: [
        "amazon/chronos-bolt-small",
        "amazon/chronos-bolt-base",
    ])
    chronos_context_length: int = 512
    chronos_num_samples: int = 20

    # TimesFM 2.5 (requires timesfm>=2.0.0 from GitHub)
    timesfm_model_id: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_context_length: int = 512

    # Moirai 2.0
    moirai_model_ids: List[str] = field(default_factory=lambda: [
        "Salesforce/moirai-2.0-R-small",
    ])
    moirai_context_length: int = 512
    moirai_num_samples: int = 20

    # Lag-Llama
    lagllama_context_length: int = 512
    lagllama_num_samples: int = 20
    lagllama_n_layer: int = 8
    lagllama_n_head: int = 4
    lagllama_n_embd_per_head: int = 36

    # Toto
    toto_model_id: str = "Datadog/Toto-Open-Base-1.0"
    toto_context_length: int = 512
    toto_num_samples: int = 20

    # Sundial
    sundial_model_id: str = "thuml/sundial-base-128m"
    sundial_context_length: int = 512
    sundial_num_samples: int = 20

    # Moirai-MoE
    moirai_moe_model_ids: List[str] = field(default_factory=lambda: [
        "Salesforce/moirai-moe-1.0-R-small",
    ])
    moirai_moe_context_length: int = 512
    moirai_moe_num_samples: int = 20

    # General TSFM settings
    device: str = "cpu"                  # "cuda" or "cpu"
    batch_size: int = 32
    zero_shot: bool = True
    fine_tune: bool = False


# ============================================================
# Forecasting settings
# ============================================================
@dataclass
class ForecastConfig:
    """Configuration for the rolling/expanding forecast engine."""
    horizons: List[int] = field(default_factory=lambda: [1, 5, 22])

    # Walk-forward design for econometric baselines
    train_window: int = 252              # 1 year
    test_window: int = 126               # 6 months
    step_size: int = 126

    # TSFM context window (zero-shot evaluation)
    tsfm_context_length: int = 512

    multistep_method: str = "direct"     # "direct" or "iterated"
    reestimate_every: int = 1


# ============================================================
# Evaluation settings
# ============================================================
@dataclass
class EvalConfig:
    """Configuration for forecast evaluation."""
    loss_functions: List[str] = field(default_factory=lambda: [
        'MSE', 'MAE', 'QLIKE', 'R2OOS',
    ])
    primary_loss: str = "QLIKE"

    dm_alternative: str = "two-sided"
    dm_hac_lags: Optional[int] = None

    mcs_alpha: float = 0.10
    mcs_n_bootstrap: int = 10000
    mcs_block_length: int = 22


# ============================================================
# Default configs (instantiate once)
# ============================================================
data_cfg = DataConfig()
har_cfg = HARConfig()
arfima_cfg = ARFIMAConfig()
fm_cfg = FoundationModelConfig()
forecast_cfg = ForecastConfig()
eval_cfg = EvalConfig()
