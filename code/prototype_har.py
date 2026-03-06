"""
prototype_har.py — End-to-end HAR model proof of concept.

Loads the proxy dataset, constructs HAR regressors, runs expanding-window
1-step-ahead forecasts, computes OOS evaluation metrics, saves forecasts
to CSV, and generates actual-vs-predicted plots.

This validates the entire pipeline before scaling to all models.
"""

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
DATA_PATH = Path("G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/data/raw/RV_March2024.xlsx")
OUTPUT_DIR = Path("G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/results")
PLOT_DIR = Path("G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/data/plots")

# Assets to prototype with
TICKERS = ['AAPL', 'JPM', 'AMZN', 'CAT']

# Forecast settings
OOS_START = '2022-01-03'  # ~2 years OOS
MIN_TRAIN = 500           # Min training window
HORIZON = 1               # 1-step-ahead

np.random.seed(42)

# ============================================================
# 1. Load data
# ============================================================
print("=" * 70)
print("HAR PROTOTYPE — End-to-End Test")
print("=" * 70)

print("\n1. Loading data...")
dates_df = pd.read_excel(DATA_PATH, sheet_name='Dates', header=None)
companies_df = pd.read_excel(DATA_PATH, sheet_name='Companies', header=None)
dates = pd.to_datetime(dates_df.iloc[:, 0])
tickers_all = companies_df.iloc[:, 0].tolist()

rv_df = pd.read_excel(DATA_PATH, sheet_name='RV', header=None)
rv_df.index = dates
rv_df.columns = tickers_all
rv_df = rv_df.apply(pd.to_numeric, errors='coerce')
rv_df = rv_df.replace(0.0, np.nan)

print(f"   Loaded {rv_df.shape[0]} dates, {rv_df.shape[1]} assets")
print(f"   Date range: {dates.min().date()} to {dates.max().date()}")

# ============================================================
# 2. Build HAR features for each asset
# ============================================================
print("\n2. Building HAR features...")

def build_har(rv: pd.Series):
    """Build HAR regressors and target for a single asset."""
    rv_d = rv.shift(1)                                  # RV_{t-1}
    rv_w = rv.rolling(5, min_periods=5).mean().shift(1)  # RV^{(w)}_{t-1}
    rv_m = rv.rolling(22, min_periods=22).mean().shift(1) # RV^{(m)}_{t-1}

    X = pd.DataFrame({'RV_d': rv_d, 'RV_w': rv_w, 'RV_m': rv_m}, index=rv.index)
    y = rv  # Target: RV_t (features are already lagged)

    # Drop NaN rows
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


# ============================================================
# 3. Expanding-window forecast
# ============================================================
print("\n3. Running expanding-window forecasts...")

results = {}
for ticker in TICKERS:
    print(f"\n   Processing {ticker}...")
    rv = rv_df[ticker].dropna()
    X, y = build_har(rv)

    # OOS dates
    oos_mask = X.index >= OOS_START
    oos_dates = X.index[oos_mask]
    print(f"   Total obs: {len(X)}, OOS: {len(oos_dates)} (from {oos_dates[0].date()})")

    actuals = []
    forecasts = []
    fcast_dates = []

    for date in oos_dates:
        # Training data: everything before this date
        train_mask = X.index < date
        X_train = X[train_mask]
        y_train = y[train_mask]

        if len(X_train) < MIN_TRAIN:
            continue

        # Fit OLS with constant
        X_train_c = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_c).fit()

        # Forecast
        X_oos = sm.add_constant(X.loc[[date]], has_constant='add')
        pred = model.predict(X_oos).values[0]

        # Ensure non-negative forecast
        pred = max(pred, 0.001)

        actuals.append(y[date])
        forecasts.append(pred)
        fcast_dates.append(date)

    results[ticker] = {
        'actual': pd.Series(actuals, index=fcast_dates, name='actual'),
        'forecast': pd.Series(forecasts, index=fcast_dates, name='HAR'),
    }
    print(f"   Forecasts generated: {len(fcast_dates)}")

# ============================================================
# 4. Compute evaluation metrics
# ============================================================
print("\n4. Computing evaluation metrics...")
print(f"\n{'Ticker':8s} {'MSE':>10s} {'MAE':>10s} {'QLIKE':>10s} {'R2_OOS':>10s}")
print("-" * 55)

metrics_list = []
for ticker in TICKERS:
    actual = results[ticker]['actual'].values
    fcast = results[ticker]['forecast'].values

    # MSE
    mse = np.mean((actual - fcast) ** 2)

    # MAE
    mae = np.mean(np.abs(actual - fcast))

    # QLIKE: (actual/forecast) - log(actual/forecast) - 1
    ratio = actual / fcast
    qlike = np.mean(ratio - np.log(ratio) - 1)

    # R2 OOS
    ss_model = np.sum((actual - fcast) ** 2)
    ss_mean = np.sum((actual - np.mean(actual)) ** 2)
    r2_oos = 1 - ss_model / ss_mean

    print(f"{ticker:8s} {mse:10.4f} {mae:10.4f} {qlike:10.6f} {r2_oos:10.4f}")

    metrics_list.append({
        'ticker': ticker,
        'model': 'HAR',
        'horizon': HORIZON,
        'MSE': mse,
        'MAE': mae,
        'QLIKE': qlike,
        'R2_OOS': r2_oos,
        'n_forecasts': len(actual),
    })

metrics_df = pd.DataFrame(metrics_list)

# ============================================================
# 5. Save forecasts to CSV
# ============================================================
print("\n5. Saving results...")
(OUTPUT_DIR / 'forecasts').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'metrics').mkdir(parents=True, exist_ok=True)

for ticker in TICKERS:
    df = pd.DataFrame({
        'date': results[ticker]['actual'].index,
        'actual': results[ticker]['actual'].values,
        'HAR_forecast': results[ticker]['forecast'].values,
    })
    df.to_csv(OUTPUT_DIR / 'forecasts' / f'har_forecast_{ticker}_h1.csv', index=False)

metrics_df.to_csv(OUTPUT_DIR / 'metrics' / 'har_prototype_metrics.csv', index=False)
print(f"   Forecasts saved to {OUTPUT_DIR / 'forecasts'}")
print(f"   Metrics saved to {OUTPUT_DIR / 'metrics'}")

# ============================================================
# 6. Plot actual vs predicted
# ============================================================
print("\n6. Generating plots...")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 4 * len(TICKERS)), sharex=False)
if len(TICKERS) == 1:
    axes = [axes]

for i, ticker in enumerate(TICKERS):
    ax = axes[i]
    actual = results[ticker]['actual']
    fcast = results[ticker]['forecast']

    ax.plot(actual.index, actual.values, 'k-', linewidth=0.6, label='Actual RV', alpha=0.8)
    ax.plot(fcast.index, fcast.values, 'r-', linewidth=0.6, label='HAR Forecast', alpha=0.7)
    ax.set_title(f'{ticker} — HAR 1-Step-Ahead Forecast (OOS: {OOS_START} onwards)', fontsize=11)
    ax.set_ylabel('Realized Variance')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add metrics annotation
    mse = metrics_df[metrics_df.ticker == ticker]['MSE'].values[0]
    qlike = metrics_df[metrics_df.ticker == ticker]['QLIKE'].values[0]
    r2 = metrics_df[metrics_df.ticker == ticker]['R2_OOS'].values[0]
    ax.text(0.02, 0.95, f'MSE={mse:.3f}  QLIKE={qlike:.5f}  R²_OOS={r2:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'har_prototype_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Plot saved to {PLOT_DIR / 'har_prototype_forecast.png'}")

# ============================================================
# 7. Summary statistics of forecast errors
# ============================================================
print("\n7. Forecast error diagnostics...")
for ticker in TICKERS:
    actual = results[ticker]['actual']
    fcast = results[ticker]['forecast']
    errors = actual - fcast

    print(f"\n   {ticker}:")
    print(f"     Mean error: {errors.mean():.4f}")
    print(f"     Std error:  {errors.std():.4f}")
    print(f"     Min error:  {errors.min():.4f}")
    print(f"     Max error:  {errors.max():.4f}")
    print(f"     AC(1) of errors: {errors.autocorr(lag=1):.4f}")

    # Check if forecast tracks actual direction
    actual_up = (actual.diff() > 0).iloc[1:]
    fcast_up = (fcast.diff() > 0).iloc[1:]
    directional_accuracy = (actual_up == fcast_up).mean()
    print(f"     Directional accuracy: {directional_accuracy:.3f}")

# ============================================================
# 8. Last model coefficients (for sanity check)
# ============================================================
print("\n8. Last-window HAR coefficients (sanity check):")
for ticker in TICKERS:
    rv = rv_df[ticker].dropna()
    X, y = build_har(rv)
    X_c = sm.add_constant(X)
    model = sm.OLS(y, X_c).fit(cov_type='HAC', cov_kwds={'maxlags': 22})

    print(f"\n   {ticker}:")
    for param, val, se, t in zip(model.params.index, model.params, model.bse, model.tvalues):
        sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.960 else '*' if abs(t) > 1.645 else ''
        print(f"     {param:8s}: {val:8.4f} (SE={se:.4f}, t={t:.2f}) {sig}")
    print(f"     R² = {model.rsquared:.4f}, Adj-R² = {model.rsquared_adj:.4f}")

print("\n" + "=" * 70)
print("HAR PROTOTYPE COMPLETE")
print("=" * 70)
