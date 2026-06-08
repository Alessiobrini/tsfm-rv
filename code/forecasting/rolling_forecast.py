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
    train_window: int = 1000,
    test_window: int = 126,
    step_size: int = 126,
    horizon: int = 1,
    reestimate_every: int = 22,
    target_kind: str = "point",
) -> Tuple[pd.Series, pd.Series]:
    """Walk-forward for series-based models (ARFIMA/ARMA/MEM) that take a raw series.

    These models forecast ``horizon`` steps natively (iterated multi-step). With
    ``target_kind="point"`` we record the ``h``-th step against the realized
    ``RV_{i+h-1}``; with ``"avg"`` we record the mean of the first ``h`` steps
    against the ``h``-day average. Origin-date and target conventions match
    ``zero_shot_forecast`` and ``features.build_target`` so models align by date.

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
    if target_kind not in ("point", "avg"):
        raise ValueError(f"target_kind must be 'point' or 'avg', got {target_kind!r}")

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

            # Native iterated multi-step forecast: horizon values ahead.
            pred = np.atleast_1d(np.asarray(model.predict(steps=horizon), dtype=float))

            # Need a realized target at i+h-1.
            if i + horizon - 1 < n:
                if target_kind == "point":
                    # h-th step vs. RV_{i+h-1}.
                    pred_val = float(pred[horizon - 1]) if len(pred) >= horizon else float(pred[-1])
                    actual_val = float(series.iloc[i + horizon - 1])
                else:  # "avg"
                    end_idx = min(i + horizon, n)
                    pred_val = float(np.mean(pred[:horizon]))
                    actual_val = float(series.iloc[i:end_idx].mean())

                all_actuals.append(actual_val)
                all_forecasts.append(pred_val)
                all_dates.append(date)
                seen_dates.add(date)

    actual_series = pd.Series(all_actuals, index=all_dates, name='actual')
    forecast_series = pd.Series(all_forecasts, index=all_dates, name='forecast')

    return actual_series, forecast_series


def iterated_har_forecast(
    rv_series: pd.Series,
    model_factory: Callable,
    horizon: int,
    train_window: int = 1000,
    test_window: int = 126,
    step_size: int = 126,
    reestimate_every: int = 22,
    weekly_lag: int = 5,
    monthly_lag: int = 22,
    target_kind: str = "point",
    min_train: int = 50,
) -> Tuple[pd.Series, pd.Series]:
    """Iterated multi-step forecast for pure HAR / Log-HAR (Corsi 2009).

    A *one-step* HAR is estimated — ``RV_t`` on ``(RV_{t-1}, RV^{(w)}_{t-1},
    RV^{(m)}_{t-1})`` — and its forecast is rolled forward ``horizon`` days,
    rebuilding the daily/weekly/monthly aggregates from the predicted path at
    each step. This is the iterated multi-step forecast Referee 1 asks for: the
    well-estimated 1-step model is kept and errors are allowed to compound.

    Valid ONLY for the pure-RV HAR variants (``HAR``, ``Log-HAR``). The augmented
    variants (HAR-J, HAR-RS, HARQ) carry auxiliary regressors (jumps,
    semivariances, realized quarticity) that cannot be projected forward without
    an extra model, so those are forecast directly (their literature-standard
    mode; e.g. Bollerslev, Patton & Quaedvlieg 2016 estimate HARQ-h directly).

    The rollout is computed from the fitted OLS coefficients in closed form
    (``b0 + b1*RV_d + b2*RV_w + b3*RV_m``; for Log-HAR on logs with the
    ``exp(. + 0.5 sigma^2)`` bias correction), so it is exact and avoids calling
    statsmodels h times per origin.

    Origin-date and target conventions match ``walk_forward_series_forecast`` /
    ``zero_shot_forecast`` so all models align by date for DM / MCS.

    Parameters
    ----------
    rv_series : pd.Series
        Full RV series on the modeling scale (no NaN).
    model_factory : Callable
        Returns a fresh ``HARModel`` (``use_log`` either) exposing ``.fit(X, y)``
        and, after fitting, ``._ols_result.params`` plus ``.use_log``.
    horizon : int
        Forecast horizon h.
    target_kind : str
        ``"point"`` (record h-th rollout step vs RV_{i+h-1}) or ``"avg"``.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (actuals, forecasts) aligned over the evaluation period.
    """
    if target_kind not in ("point", "avg"):
        raise ValueError(f"target_kind must be 'point' or 'avg', got {target_kind!r}")

    # Lazy import: keep this engine's HAR dependency out of the generic module top.
    from features import build_har_features, build_target, align_features_target

    n = len(rv_series)
    folds = generate_walk_forward_folds(n, train_window, test_window, step_size)
    if len(folds) == 0:
        raise ValueError(f"No folds: n_obs={n}, train_window={train_window}")

    vals = rv_series.values.astype(float)
    idx = rv_series.index

    all_actuals: List[float] = []
    all_forecasts: List[float] = []
    all_dates: List = []
    seen_dates = set()

    for train_start, train_end, test_start, test_end in folds:
        coefs = None          # (b0, b_d, b_w, b_m)
        use_log = False
        sigma2 = 0.0
        last_fit = -reestimate_every

        for j, i in enumerate(range(test_start, test_end)):
            date = idx[i]
            if date in seen_dates:
                continue
            if i + horizon - 1 >= n or i < monthly_lag:
                continue  # no realized target, or not enough history to seed

            # (Re)estimate the 1-step HAR on the expanding in-fold window.
            if j - last_fit >= reestimate_every or coefs is None:
                fit_series = rv_series.iloc[train_start:i]
                if len(fit_series) < min_train:
                    continue
                feats = build_har_features(fit_series, monthly_lag=monthly_lag,
                                           weekly_lag=weekly_lag)
                tgt = build_target(fit_series, horizon=1, target_kind="point")
                X, y = align_features_target(feats, tgt)
                if len(X) < min_train:
                    continue
                model = model_factory()
                model.fit(X, y)
                res = model._ols_result
                p = res.params
                coefs = (
                    float(p.get('const', 0.0)),
                    float(p['RV_d']), float(p['RV_w']), float(p['RV_m']),
                )
                use_log = bool(getattr(model, 'use_log', False))
                sigma2 = float(getattr(res, 'mse_resid', 0.0)) if use_log else 0.0
                last_fit = j

            if coefs is None:
                continue

            b0, b_d, b_w, b_m = coefs
            # Seed with the last `monthly_lag` actuals (info through i-1).
            work = list(vals[i - monthly_lag:i])
            preds: List[float] = []
            for _ in range(horizon):
                rv_d = work[-1]
                rv_w = float(np.mean(work[-weekly_lag:]))
                rv_m = float(np.mean(work[-monthly_lag:]))
                if use_log:
                    lp = (b0
                          + b_d * np.log(max(rv_d, 1e-10))
                          + b_w * np.log(max(rv_w, 1e-10))
                          + b_m * np.log(max(rv_m, 1e-10)))
                    nxt = float(np.exp(lp + 0.5 * sigma2))
                else:
                    nxt = b0 + b_d * rv_d + b_w * rv_w + b_m * rv_m
                work.append(nxt)
                preds.append(nxt)

            if target_kind == "point":
                fc = preds[horizon - 1]
                act = float(vals[i + horizon - 1])
            else:  # "avg"
                fc = float(np.mean(preds))
                act = float(np.mean(vals[i:i + horizon]))

            all_actuals.append(act)
            all_forecasts.append(fc)
            all_dates.append(date)
            seen_dates.add(date)

    actual_series = pd.Series(all_actuals, index=all_dates, name='actual')
    forecast_series = pd.Series(all_forecasts, index=all_dates, name='forecast')
    return actual_series, forecast_series


def zero_shot_forecast(
    rv_series: pd.Series,
    model,
    horizon: int,
    context_length: int = 1000,
    target_kind: str = "point",
) -> Tuple[pd.Series, pd.Series]:
    """Run zero-shot TSFM evaluation over the full series.

    At each origin date ``i`` (information available through ``i-1``):
        - Extract context = series[i - context_length : i]
        - Call model.predict(context, horizon) -> point forecast (the model's
          conditional mean; see ``models.foundation``)
        - Store the actual and the forecast under the origin date ``i``.

    Two target conventions, matching ``features.build_target``:

    ``"point"`` (main paper)
        Forecast the value ``horizon`` steps ahead, ``RV_{i+h-1}``, using the
        ``h``-th step of the rollout (``point[horizon-1]``). Reduces to ``RV_i``
        at ``h=1``. This is the genuine iterated multi-step forecast.

    ``"avg"`` (appendix)
        Forecast the ``h``-day average ``mean(RV_i .. RV_{i+h-1})`` with the
        average of the first ``h`` rollout steps.

    The actual and the origin-date convention are identical to the econometric
    engines (``walk_forward_series_forecast``), so forecasts align by date for
    DM / MCS comparison.

    Parameters
    ----------
    rv_series : pd.Series
        Full series on the modeling scale (no NaN).
    model : BaseTSFM
        Foundation model with .predict(context, horizon) -> TSFMForecast.
    horizon : int
        Forecast horizon ``h``.
    context_length : int
        Context window size.
    target_kind : str
        ``"point"`` or ``"avg"``.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (actuals, forecasts) aligned over the evaluation period.
    """
    if target_kind not in ("point", "avg"):
        raise ValueError(f"target_kind must be 'point' or 'avg', got {target_kind!r}")

    values = rv_series.values
    dates = rv_series.index
    n = len(values)

    actuals = []
    forecasts = []
    forecast_dates = []

    start_idx = context_length

    for i in range(start_idx, n):
        # The target day is i+h-1; skip origins with no realized target yet.
        if i + horizon - 1 >= n:
            break

        # Context: last context_length observations before date i.
        ctx = values[i - context_length:i]

        result = model.predict(ctx, horizon)
        point = np.atleast_1d(np.asarray(result.point, dtype=float))

        if target_kind == "point":
            # h-th rollout step -> value h days ahead of the information set.
            pred_val = float(point[horizon - 1]) if len(point) >= horizon else float(point[-1])
            actual_val = float(values[i + horizon - 1])
        else:  # "avg"
            pred_val = float(np.mean(point[:horizon]))
            actual_val = float(np.mean(values[i:i + horizon]))

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
