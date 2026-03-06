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
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
PLOTS_DIR = DATA_DIR / "plots"

# Dataset paths
DATA_FILE = RAW_DIR / "RV_March2024.xlsx"          # CAPIRe proxy dataset
VOLARE_DATA_DIR = RAW_DIR / "volare"               # VOLARE dataset directory
VOLARE_STOCKS_FILE = VOLARE_DATA_DIR / "realized_variance_stocks.csv"
VOLARE_FOREX_FILE = VOLARE_DATA_DIR / "realized_variance_forex.csv"
VOLARE_FUTURES_FILE = VOLARE_DATA_DIR / "realized_variance_futures.csv"

# VOLARE results go in a separate subdirectory
VOLARE_RESULTS_DIR = RESULTS_DIR / "volare"

# Covariance data paths (already extracted CSVs)
VOLARE_COV_STOCKS_FILE = RAW_DIR / "realized_covariance_stocks.csv"
VOLARE_COV_FOREX_FILE = RAW_DIR / "realized_covariance_forex.csv"
VOLARE_COV_FUTURES_FILE = RAW_DIR / "realized_covariance_futures.csv"

# Covariance results
COV_RESULTS_DIR = RESULTS_DIR / "covariance"

# ============================================================
# Random seed
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Asset selection
# ============================================================
# All 30 DJIA stocks in the proxy dataset
ALL_TICKERS = [
    'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
    'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD',
    'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT',
]

# Exclude assets with short history (<3000 obs) from main analysis
EXCLUDED_TICKERS = ['DOW']  # Only 1,266 obs — too short

# Primary analysis tickers (full sample, 26 assets)
MAIN_TICKERS = [t for t in ALL_TICKERS if t not in EXCLUDED_TICKERS]

# Representative subset for plots / quick experiments
REPRESENTATIVE_TICKERS = ['AAPL', 'JPM', 'AMZN', 'CAT']

# All VOLARE tickers by asset class
VOLARE_STOCK_TICKERS = [
    'AAPL', 'ADBE', 'AMD', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO',
    'CVX', 'DIS', 'GE', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM',
    'KO', 'MCD', 'META', 'MMM', 'MRK', 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL',
    'PG', 'PM', 'SHW', 'TRV', 'TSLA', 'UNH', 'V', 'VZ', 'WMT', 'XOM',
]
VOLARE_FX_TICKERS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']
VOLARE_FUTURES_TICKERS = ['C', 'CL', 'ES', 'GC', 'NG']
VOLARE_ALL_TICKERS = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS

# ============================================================
# Data settings
# ============================================================
@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Which RV measure to use as primary target
    rv_measure: str = "RV"          # "RV" (1-min) or "RV_5" (5-min)
    bpv_measure: str = "BPV"        # Bipower variation sheet
    good_measure: str = "Good"      # Positive semivariance sheet
    bad_measure: str = "Bad"        # Negative semivariance sheet
    rq_measure: str = "RQ"          # Realized quarticity sheet

    # Missing data treatment
    zero_is_missing: bool = True    # Zeros represent missing data
    min_obs_per_asset: int = 500    # Minimum observations for inclusion

    # VOLARE column mappings (long-format CSV)
    volare_rv_col: str = "rv5"      # 5-min realized variance
    volare_bpv_col: str = "bv5"     # 5-min bipower variation
    volare_good_col: str = "rsp5"   # 5-min positive semivariance
    volare_bad_col: str = "rsn5"    # 5-min negative semivariance
    volare_rq_col: str = "rq5"     # 5-min realized quarticity

    # Outlier treatment
    winsorize: bool = False         # Whether to winsorize extreme values
    winsorize_pctile: float = 0.995 # Upper percentile for winsorization


# ============================================================
# HAR model settings
# ============================================================
@dataclass
class HARConfig:
    """Configuration for HAR family models."""
    # Lag structure (in trading days)
    daily_lag: int = 1              # RV_{t-1}
    weekly_lag: int = 5             # Average of RV_{t-1} to RV_{t-5}
    monthly_lag: int = 22           # Average of RV_{t-1} to RV_{t-22}

    # Estimation
    use_log_rv: bool = False        # Estimate HAR on log(RV) instead of RV
    hac_std_errors: bool = True     # Newey-West HAC standard errors
    hac_max_lags: int = 22          # Max lags for HAC

    # Variants to run
    models: List[str] = field(default_factory=lambda: [
        'HAR',      # Standard HAR
        'HAR-J',    # + jump component
        'HAR-RS',   # + semivariance decomposition
        'HARQ',     # + realized quarticity
    ])


# ============================================================
# Realized GARCH settings
# ============================================================
@dataclass
class RealizedGARCHConfig:
    """Configuration for Realized GARCH model."""
    p: int = 1                      # GARCH order
    q: int = 1                      # ARCH order
    dist: str = "normal"            # Error distribution: "normal", "t", "skewt"
    returns_source: str = "yahoo"   # Where to get daily returns: "yahoo" or "file"


# ============================================================
# ARFIMA settings
# ============================================================
@dataclass
class ARFIMAConfig:
    """Configuration for ARFIMA model."""
    use_log_rv: bool = True         # Fit on log(RV) — standard practice
    max_ar: int = 2                 # Max AR order for selection
    max_ma: int = 2                 # Max MA order for selection
    d_method: str = "mle"           # Fractional diff estimation: "mle", "gph", "whittle"


# ============================================================
# Foundation model settings
# ============================================================
@dataclass
class FoundationModelConfig:
    """Configuration for time series foundation models."""
    # Chronos-2 / Chronos-Bolt
    chronos_model_ids: List[str] = field(default_factory=lambda: [
        "amazon/chronos-bolt-small",
        "amazon/chronos-bolt-base",
    ])
    chronos_context_length: int = 512    # Max context window
    chronos_num_samples: int = 20        # Number of forecast samples

    # TimesFM 2.5
    timesfm_model_id: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_context_length: int = 512
    timesfm_freq: str = "D"             # Daily frequency token

    # Moirai 2.0
    moirai_model_ids: List[str] = field(default_factory=lambda: [
        "Salesforce/moirai-2.0-R-small",
    ])
    moirai_context_length: int = 512
    moirai_num_samples: int = 20

    # Lag-Llama
    lagllama_context_length: int = 512
    lagllama_num_samples: int = 100
    lagllama_n_layer: int = 8
    lagllama_n_head: int = 4
    lagllama_n_embd_per_head: int = 36

    # Kronos
    kronos_model_id: str = "NeoQuasar/Kronos-base"
    kronos_tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_context_length: int = 512
    kronos_sample_count: int = 5

    # General TSFM settings
    device: str = "cpu"                  # "cuda" or "cpu"
    batch_size: int = 32
    zero_shot: bool = True
    fine_tune: bool = False              # Enable fine-tuning pass


# ============================================================
# Forecasting settings
# ============================================================
@dataclass
class ForecastConfig:
    """Configuration for the rolling/expanding forecast engine."""
    # Forecast horizons (in trading days)
    horizons: List[int] = field(default_factory=lambda: [1, 5, 22])

    # Walk-forward design for econometric baselines
    train_window: int = 252              # Training window in trading days (1 year)
    test_window: int = 126               # Test window per fold (6 months)
    step_size: int = 126                 # Slide by test window length

    # TSFM context window (zero-shot evaluation)
    tsfm_context_length: int = 512       # Context window for foundation models

    # Multi-step forecast method
    multistep_method: str = "direct"     # "direct" or "iterated"

    # Re-estimation frequency for econometric models
    reestimate_every: int = 1            # Re-estimate every N steps (1 = every day)


# ============================================================
# Evaluation settings
# ============================================================
@dataclass
class EvalConfig:
    """Configuration for forecast evaluation."""
    # Loss functions
    loss_functions: List[str] = field(default_factory=lambda: [
        'MSE', 'MAE', 'QLIKE', 'R2OOS',
    ])
    primary_loss: str = "QLIKE"          # Primary metric for ranking

    # Diebold-Mariano test
    dm_alternative: str = "two-sided"    # "two-sided", "less", "greater"
    dm_hac_lags: Optional[int] = None    # None = auto (h-1 for h-step)

    # Model Confidence Set
    mcs_alpha: float = 0.10              # Significance level
    mcs_n_bootstrap: int = 10000         # Bootstrap replications
    mcs_block_length: int = 22           # Block bootstrap block length


# ============================================================
# Default configs (instantiate once)
# ============================================================
data_cfg = DataConfig()
har_cfg = HARConfig()
rgarch_cfg = RealizedGARCHConfig()
arfima_cfg = ARFIMAConfig()
fm_cfg = FoundationModelConfig()
forecast_cfg = ForecastConfig()
eval_cfg = EvalConfig()
