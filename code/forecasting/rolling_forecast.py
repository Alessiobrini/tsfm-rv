"""
forecasting/rolling_forecast.py — Walk-forward and zero-shot forecast engines.

Supports:
    1. walk_forward_forecast(): Sliding window for econometric baselines
    2. zero_shot_forecast(): Rolling context window for TSFMs
    3. expanding_window_forecast(): Legacy expanding window (kept for validation)
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ForecastOutput:
    """Container for forecast results."""
    actual: pd.Series
    forecasts: Dict[str, pd.Series]
    dates: pd.DatetimeIndex
    ticker: str
    horizon: int


def generate_walk_forward_folds(
    n_obs: int,
    train_window: int = 252,
    test_window: int = 126,
    step_size: int = 126,
) -> List[Tuple[int, int, int, int]]:
    """Generate (train_start, train_end, test_start, test_end) index tuples.

    Parameters
    ----------
    n_obs : int
        Total number of observations.
    train_window : int
        Number of training observations per fold.
    test_window : int
        Number of test observations per fold.
    step_size : int
        How far to slide forward each fold.

    Returns
    -------
    List of (train_start, train_end, test_start, test_end) tuples.
        train spans [train_start, train_end), test spans [test_start, test_end).
    """
    folds = []
    fold_start = 0
    while fold_start + train_window < n_obs:
        train_start = fold_start
        train_end = fold_start + train_window
        test_start = train_end
        test_end = min(test_start + test_window, n_obs)
        if test_start >= n_obs:
            break
        folds.append((train_start, train_end, test_start, test_end))
        fold_start += step_size
    return folds


def walk_forward_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    train_window: int = 252,
    test_window: int = 126,
    step_size: int = 126,
    reestimate_every: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """Run walk-forward (sliding window) forecasts for a single model.

    For each fold:
        - Train on X[train_start:train_end], y[train_start:train_end]
        - Predict on X[test_start:test_end], re-estimating every N steps

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (aligned, no NaN).
    y : pd.Series
        Full target series (aligned, no NaN).
    model_factory : Callable
        Returns a fresh model with .fit(X, y) and .predict(X).
    train_window : int
        Training window size.
    test_window : int
        Test window size per fold.
    step_size : int
        Slide distance between folds.
    reestimate_every : int
        Re-estimate within test window every N steps (1 = every day).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (actuals, forecasts) aligned over all test folds.
    """
    n = len(X)
    folds = generate_walk_forward_folds(n, train_window, test_window, step_size)

    if len(folds) == 0:
        raise ValueError(
            f"No folds generated: n_obs={n}, train_window={train_window}, "
            f"test_window={test_window}"
        )

    all_actuals = []
    all_forecasts = []
    all_dates = []
    seen_dates = set()

    for train_start, train_end, test_start, test_end in folds:
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        model = None
        last_fit = -reestimate_every

        for i in range(len(X_test)):
            date = X_test.index[i]
            if date in seen_dates:
                continue  # Avoid duplicates from overlapping folds

            # Re-estimate if needed
            if i - last_fit >= reestimate_every or model is None:
                # Expanding within the fold: train on original train + test seen so far
                if i > 0 and reestimate_every > 1:
                    X_fit = pd.concat([X_train, X_test.iloc[:i]])
                    y_fit = pd.concat([y_train, y_test.iloc[:i]])
                else:
                    X_fit = X_train
                    y_fit = y_train
                model = model_factory()
                model.fit(X_fit, y_fit)
                last_fit = i

            # Predict single observation
            pred = model.predict(X_test.iloc[[i]])
            pred_val = pred.values[0] if hasattr(pred, 'values') else float(pred)

            all_actuals.append(y_test.iloc[i])
            all_forecasts.append(pred_val)
            all_dates.append(date)
            seen_dates.add(date)

    actual_series = pd.Series(all_actuals, index=all_dates, name='actual')
    forecast_series = pd.Series(all_forecasts, index=all_dates, name='forecast')

    return actual_series, forecast_series


def walk_forward_series_forecast(
    series: pd.Series,
    model_factory: Callable,
    train_window: int = 252,
    test_window: int = 126,
    step_size: int = 126,
    horizon: int = 1,
    reestimate_every: int = 22,
) -> Tuple[pd.Series, pd.Series]:
    """Walk-forward for series-based models (ARFIMA) that take raw series, not features.

    Parameters
    ----------
    series : pd.Series
        Full time series (e.g., RV or log-RV).
    model_factory : Callable
        Returns a model with .fit(series) and .predict(steps) interface.
    train_window : int
        Training window size.
    test_window : int
        Test window size per fold.
    step_size : int
        Slide distance.
    horizon : int
        Forecast horizon.
    reestimate_every : int
        Re-estimate the model every N steps (default 22 = monthly).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (actuals, forecasts) aligned over all test folds.
    """
    n = len(series)
    folds = generate_walk_forward_folds(n, train_window, test_window, step_size)

    if len(folds) == 0:
        raise ValueError(f"No folds: n_obs={n}, train_window={train_window}")

    all_actuals = []
    all_forecasts = []
    all_dates = []
    seen_dates = set()

    for train_start, train_end, test_start, test_end in folds:
        model = None
        last_fit = -reestimate_every

        for j, i in enumerate(range(test_start, test_end)):
            date = series.index[i]
            if date in seen_dates:
                continue

            # Re-estimate periodically
            if j - last_fit >= reestimate_every or model is None:
                fit_series = series.iloc[train_start:i]
                if len(fit_series) < 50:
                    continue
                model = model_factory()
                model.fit(fit_series)
                last_fit = j

            pred = model.predict(steps=horizon)

            # For h=1, take first forecast; for h>1, take the h-th value
            pred_val = float(pred[0]) if hasattr(pred, '__len__') else float(pred)

            # Target: actual RV at forecast date
            if i + horizon - 1 < n:
                if horizon == 1:
                    actual_val = series.iloc[i]
                else:
                    end_idx = min(i + horizon, n)
                    actual_val = series.iloc[i:end_idx].mean()

                all_actuals.append(actual_val)
                all_forecasts.append(pred_val)
                all_dates.append(date)
                seen_dates.add(date)

    actual_series = pd.Series(all_actuals, index=all_dates, name='actual')
    forecast_series = pd.Series(all_forecasts, index=all_dates, name='forecast')

    return actual_series, forecast_series


def zero_shot_forecast(
    rv_series: pd.Series,
    model,
    horizon: int,
    context_length: int = 512,
) -> Tuple[pd.Series, pd.Series]:
    """Run zero-shot TSFM evaluation over the full series.

    For each date from context_length onward:
        - Extract context = rv[date - context_length : date]
        - Call model.predict(context, horizon) -> point forecast
        - Store actual and forecast

    Parameters
    ----------
    rv_series : pd.Series
        Full RV series (no NaN).
    model : BaseTSFM
        Foundation model with .predict(context, horizon) -> TSFMForecast.
    horizon : int
        Forecast horizon.
    context_length : int
        Context window size.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (actuals, forecasts) aligned over the evaluation period.
    """
    values = rv_series.values
    dates = rv_series.index
    n = len(values)

    actuals = []
    forecasts = []
    forecast_dates = []

    start_idx = context_length

    for i in range(start_idx, n):
        # Context: last context_length observations before date i
        ctx = values[i - context_length:i]

        # Predict
        result = model.predict(ctx, horizon)
        point = result.point

        # For h=1 take first value, for h>1 take mean of horizon
        if horizon == 1:
            pred_val = float(point[0])
            actual_val = values[i] if i < n else np.nan
        else:
            pred_val = float(np.mean(point[:horizon]))
            end_idx = min(i + horizon, n)
            if end_idx > i:
                actual_val = float(np.mean(values[i:end_idx]))
            else:
                actual_val = np.nan

        if not np.isnan(actual_val):
            actuals.append(actual_val)
            forecasts.append(pred_val)
            forecast_dates.append(dates[i])

    actual_series = pd.Series(actuals, index=forecast_dates, name='actual')
    forecast_series = pd.Series(forecasts, index=forecast_dates, name='forecast')

    return actual_series, forecast_series


def expanding_window_forecast(
    rv_series: pd.Series,
    feature_builder: Callable,
    model_factory: Callable,
    oos_start: str,
    horizon: int = 1,
    min_train: int = 500,
    reestimate_every: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """Run expanding-window one-step-ahead forecasts (legacy, kept for validation).

    Parameters
    ----------
    rv_series : pd.Series
        Full RV series for one asset (DatetimeIndex).
    feature_builder : Callable
        Function: rv -> (features_df, target_series).
    model_factory : Callable
        Function: () -> model with .fit(X, y) and .predict(X).
    oos_start : str
        Start date of out-of-sample period.
    horizon : int
        Forecast horizon.
    min_train : int
        Minimum training observations.
    reestimate_every : int
        Re-estimate every N steps.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (actual, forecasts) aligned over the OOS period.
    """
    X_full, y_full = feature_builder(rv_series)

    oos_mask = X_full.index >= oos_start
    oos_dates = X_full.index[oos_mask]

    if len(oos_dates) == 0:
        raise ValueError(f"No OOS dates after {oos_start}")

    actuals = []
    forecasts = []
    forecast_dates = []

    model = None
    last_fit = -reestimate_every

    for i, date in enumerate(oos_dates):
        train_mask = X_full.index < date
        X_train = X_full[train_mask]
        y_train = y_full[train_mask]

        if len(X_train) < min_train:
            continue

        if i - last_fit >= reestimate_every or model is None:
            model = model_factory()
            model.fit(X_train, y_train)
            last_fit = i

        X_oos = X_full.loc[[date]]
        pred = model.predict(X_oos)

        if date in y_full.index and not np.isnan(y_full[date]):
            actuals.append(y_full[date])
            forecasts.append(pred.values[0] if hasattr(pred, 'values') else float(pred))
            forecast_dates.append(date)

    actual_series = pd.Series(actuals, index=forecast_dates, name='actual')
    forecast_series = pd.Series(forecasts, index=forecast_dates, name='forecast')

    return actual_series, forecast_series
