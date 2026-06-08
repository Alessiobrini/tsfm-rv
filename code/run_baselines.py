"""
run_baselines.py — Helper functions for econometric baseline runs.

Imported by run_baselines_volare.py. Not a standalone entry point;
run the pipeline through run_baselines_volare.py.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import arfima_cfg
from features import (
    build_har_features, build_har_j_features, build_har_rs_features,
    build_harq_features, build_target, align_features_target,
)
from models.har import HARModel, HARJModel, HARRSModel, HARQModel
from models.arfima import ARFIMAModel
from models.arma import ARMAModel
from models.mem import MEMModel


# Pure-RV series models (forecast multi-step natively / iterated; no auxiliary
# regressors). HAR/Log-HAR are also pure-RV but use the dedicated iterated-HAR
# engine; the three below run through the series walk-forward engine.
SERIES_MODELS = ['ARFIMA', 'ARMA', 'MEM']
# Pure HAR variants forecast via the iterated recursive plug-in.
ITERATED_HAR_MODELS = ['HAR', 'Log-HAR']
# Augmented HAR variants carry auxiliary regressors (jumps, semivariances,
# quarticity) that cannot be projected forward, so they are forecast directly.
DIRECT_HAR_MODELS = ['HAR-J', 'HAR-RS', 'HARQ']

AVAILABLE_MODELS = ['HAR', 'Log-HAR', 'HAR-J', 'HAR-RS', 'HARQ', 'ARFIMA', 'ARMA', 'MEM']


def build_features_and_target(data, ticker, horizon, model_name,
                              target_kind="point", target_scale="vol"):
    """Build features and aligned target for a given model/ticker/horizon.

    Returns (X, y) for feature-based HAR models or (series, None) for the
    series models (ARFIMA/ARMA/MEM) and the iterated-HAR models (which take the
    raw series and build features internally).

    Scale convention (target_scale="vol", the headline): pure-RV models forecast
    volatility directly and are fed sqrt(RV). The augmented HAR variants are
    variance-decomposition models, so they are always built on the variance
    scale here; the *caller* maps their forecasts to volatility (sqrt).
    """
    rv = data.rv[ticker].dropna()

    if model_name in SERIES_MODELS or model_name in ITERATED_HAR_MODELS:
        series = np.sqrt(rv) if target_scale == "vol" else rv
        return series, None

    elif model_name == 'HAR-J':
        jump = data.jump[ticker].dropna()
        common_idx = rv.index.intersection(jump.index)
        rv_a, jump_a = rv.loc[common_idx], jump.loc[common_idx]
        features = build_har_j_features(rv_a, jump_a)
        target = build_target(rv_a, horizon=horizon, target_kind=target_kind)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HAR-RS':
        good = data.good[ticker].dropna()
        bad = data.bad[ticker].dropna()
        common_idx = rv.index.intersection(good.index).intersection(bad.index)
        good_a, bad_a = good.loc[common_idx], bad.loc[common_idx]
        rv_a = rv.loc[common_idx]
        features = build_har_rs_features(good_a, bad_a)
        target = build_target(rv_a, horizon=horizon, target_kind=target_kind)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HARQ':
        rq = data.rq[ticker].dropna()
        common_idx = rv.index.intersection(rq.index)
        rv_a, rq_a = rv.loc[common_idx], rq.loc[common_idx]
        features = build_harq_features(rv_a, rq_a)
        target = build_target(rv_a, horizon=horizon, target_kind=target_kind)
        X, y = align_features_target(features, target)
        return X, y

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_factory(model_name):
    """Return a callable that creates a fresh model instance."""
    if model_name == 'HAR':
        # Level HAR with Nelson-Cao non-negativity (Referee 1 1.4.1): positive
        # forecasts without the 1e-10 floor.
        return lambda: HARModel(constrained=True)
    elif model_name == 'HAR-J':
        return lambda: HARJModel()
    elif model_name == 'HAR-RS':
        return lambda: HARRSModel()
    elif model_name == 'HARQ':
        return lambda: HARQModel()
    elif model_name == 'Log-HAR':
        return lambda: HARModel(use_log=True)
    elif model_name == 'ARFIMA':
        return lambda: ARFIMAModel(
            p=arfima_cfg.max_ar,
            q=arfima_cfg.max_ma,
            use_log=arfima_cfg.use_log_rv,
            d_method=arfima_cfg.d_method,
        )
    elif model_name == 'ARMA':
        return lambda: ARMAModel(max_p=2, max_q=2, use_log=True, ic='bic')
    elif model_name == 'MEM':
        return lambda: MEMModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")
