"""
run_baselines.py — Helper functions for econometric baseline runs.

Imported by run_baselines_volare.py. Not a standalone entry point;
run the pipeline through run_baselines_volare.py.
"""

import sys
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


AVAILABLE_MODELS = ['HAR', 'HAR-J', 'HAR-RS', 'HARQ', 'Log-HAR', 'ARFIMA']


def build_features_and_target(data, ticker, horizon, model_name):
    """Build features and aligned target for a given model/ticker/horizon.

    Returns (X, y) for feature-based models or (series, None) for ARFIMA.
    """
    rv = data.rv[ticker].dropna()
    target = build_target(rv, horizon=horizon)

    if model_name == 'HAR':
        features = build_har_features(rv)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HAR-J':
        jump = data.jump[ticker].dropna()
        common_idx = rv.index.intersection(jump.index)
        rv_a, jump_a = rv.loc[common_idx], jump.loc[common_idx]
        features = build_har_j_features(rv_a, jump_a)
        target = build_target(rv_a, horizon=horizon)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HAR-RS':
        good = data.good[ticker].dropna()
        bad = data.bad[ticker].dropna()
        common_idx = rv.index.intersection(good.index).intersection(bad.index)
        good_a, bad_a = good.loc[common_idx], bad.loc[common_idx]
        rv_a = rv.loc[common_idx]
        features = build_har_rs_features(good_a, bad_a)
        target = build_target(rv_a, horizon=horizon)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HARQ':
        rq = data.rq[ticker].dropna()
        common_idx = rv.index.intersection(rq.index)
        rv_a, rq_a = rv.loc[common_idx], rq.loc[common_idx]
        features = build_harq_features(rv_a, rq_a)
        target = build_target(rv_a, horizon=horizon)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'Log-HAR':
        features = build_har_features(rv)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'ARFIMA':
        return rv, None

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_factory(model_name):
    """Return a callable that creates a fresh model instance."""
    if model_name == 'HAR':
        return lambda: HARModel()
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
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
