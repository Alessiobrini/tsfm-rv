"""Data exploration script for RV_March2024.xlsx proxy dataset.

The Excel file has sheets: Read.me, RV, BPV, Good, Bad, RQ, Dates, Companies,
RV_5, BPV_5, Good_5, Bad_5, RQ_5.

- Dates sheet: contains the date index
- Companies sheet: contains ticker/company names
- Data sheets (RV, BPV, etc.): rows=dates, columns=assets, NO header row
- Missing data encoded as zeros (per Read.me)
- 1-min measures: RV, BPV, Good, Bad, RQ
- 5-min measures: RV_5, BPV_5, Good_5, Bad_5, RQ_5
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)

DATA_PATH = "G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/data/raw/RV_March2024.xlsx"
OUT_DIR = "G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/data"

# ============================================================
# 1. Load metadata sheets
# ============================================================
print("=" * 80)
print("1. LOADING METADATA")
print("=" * 80)

# Read Read.me
readme = pd.read_excel(DATA_PATH, sheet_name='Read.me', header=None)
print("Read.me contents:")
print(readme.to_string())

# Read Dates
dates_df = pd.read_excel(DATA_PATH, sheet_name='Dates', header=None)
print(f"\nDates sheet shape: {dates_df.shape}")
print(f"First 5 dates: {dates_df.iloc[:5, 0].tolist()}")
print(f"Last 5 dates: {dates_df.iloc[-5:, 0].tolist()}")

# Read Companies
companies_df = pd.read_excel(DATA_PATH, sheet_name='Companies', header=None)
print(f"\nCompanies sheet shape: {companies_df.shape}")
print(f"Companies: {companies_df.iloc[:, 0].tolist()}")

# Parse dates
dates = pd.to_datetime(dates_df.iloc[:, 0])
tickers = companies_df.iloc[:, 0].tolist()
n_dates = len(dates)
n_tickers = len(tickers)
print(f"\nNumber of dates: {n_dates}")
print(f"Number of tickers: {n_tickers}")
print(f"Date range: {dates.min()} to {dates.max()}")

# ============================================================
# 2. Load data sheets and attach proper labels
# ============================================================
print("\n" + "=" * 80)
print("2. LOADING DATA SHEETS")
print("=" * 80)

data_sheets = ['RV', 'BPV', 'Good', 'Bad', 'RQ', 'RV_5', 'BPV_5', 'Good_5', 'Bad_5', 'RQ_5']
data = {}

for sheet in data_sheets:
    df = pd.read_excel(DATA_PATH, sheet_name=sheet, header=None)
    print(f"Sheet '{sheet}': raw shape = {df.shape}")

    # The shape should be (n_dates, n_tickers)
    # But first row might have been consumed as header when n_dates+1 rows exist
    if df.shape[0] == n_dates:
        df.index = dates
        df.columns = tickers
    elif df.shape[0] == n_dates - 1:
        # First row was consumed as header — need to re-read
        df_full = pd.read_excel(DATA_PATH, sheet_name=sheet, header=None)
        # This should already give us all rows without header consumption
        # Check if pandas is eating first row
        print(f"  WARNING: row count mismatch. Raw has {df.shape[0]} rows, expected {n_dates}")
        df.index = dates[1:]
        df.columns = tickers
    elif df.shape[0] == n_dates + 1:
        # Extra header row
        df = df.iloc[1:].reset_index(drop=True)
        df.index = dates
        df.columns = tickers

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    data[sheet] = df
    print(f"  Final shape: {df.shape}, dtype check: all float = {all(df.dtypes == 'float64')}")

# ============================================================
# 3. SCHEMA AND MEASURES
# ============================================================
print("\n" + "=" * 80)
print("3. REALIZED MEASURES AVAILABLE")
print("=" * 80)

measures_1min = {'RV': 'Realized Variance (1-min)', 'BPV': 'Bipower Variation (1-min)',
                 'Good': 'Good (Positive) Semivariance (1-min)', 'Bad': 'Bad (Negative) Semivariance (1-min)',
                 'RQ': 'Realized Quarticity (1-min)'}
measures_5min = {'RV_5': 'Realized Variance (5-min)', 'BPV_5': 'Bipower Variation (5-min)',
                 'Good_5': 'Good (Positive) Semivariance (5-min)', 'Bad_5': 'Bad (Negative) Semivariance (5-min)',
                 'RQ_5': 'Realized Quarticity (5-min)'}

for k, v in {**measures_1min, **measures_5min}.items():
    if k in data:
        print(f"  {k:8s} -> {v} [shape={data[k].shape}]")

# ============================================================
# 4. ASSET COVERAGE
# ============================================================
print("\n" + "=" * 80)
print("4. ASSET COVERAGE")
print("=" * 80)

rv = data['RV']
print(f"Total assets: {len(tickers)}")
print(f"Tickers: {tickers}")

# Count non-zero (non-missing) observations per asset
# Missing = 0 per readme
for sheet_name in ['RV', 'RV_5']:
    df = data[sheet_name]
    print(f"\nNon-zero observations per asset ({sheet_name}):")
    nonzero_counts = (df != 0).sum()
    for t in tickers:
        pct = nonzero_counts[t] / len(df) * 100
        print(f"  {t:8s}: {nonzero_counts[t]:5d} / {len(df)} ({pct:.1f}%)")

# ============================================================
# 5. MISSING DATA ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("5. MISSING DATA ANALYSIS")
print("=" * 80)

rv = data['RV']
# Zeros = missing
zero_mask = (rv == 0)
print(f"Total zeros (=missing) in RV: {zero_mask.sum().sum()} out of {rv.size} ({zero_mask.sum().sum()/rv.size*100:.2f}%)")

# Per-asset zero counts
print("\nZero counts per asset (RV):")
zero_per_asset = zero_mask.sum()
for t in tickers:
    print(f"  {t:8s}: {zero_per_asset[t]:5d} zeros ({zero_per_asset[t]/len(rv)*100:.1f}%)")

# Per-date zero counts (how many assets missing per day)
zero_per_date = zero_mask.sum(axis=1)
print(f"\nDays with at least one missing asset: {(zero_per_date > 0).sum()}")
print(f"Days with all assets present: {(zero_per_date == 0).sum()}")
print(f"Max assets missing on a single day: {zero_per_date.max()}")

# Check for weekends/holidays in dates
if hasattr(dates, 'dt'):
    dow = dates.dt.dayofweek
else:
    dow = dates.dayofweek
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print(f"\nDay-of-week distribution:")
for i, day in enumerate(weekdays):
    count = (dow == i).sum()
    if count > 0:
        print(f"  {day}: {count}")

# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 80)
print("6. DESCRIPTIVE STATISTICS (RV, 1-min)")
print("=" * 80)

# Replace zeros with NaN for stats
rv_clean = rv.replace(0, np.nan)

stats_list = []
for t in tickers:
    s = rv_clean[t].dropna()
    if len(s) < 50:
        continue
    log_s = np.log(s)

    # Autocorrelations
    ac1 = s.autocorr(lag=1) if len(s) > 1 else np.nan
    ac5 = s.autocorr(lag=5) if len(s) > 5 else np.nan
    ac22 = s.autocorr(lag=22) if len(s) > 22 else np.nan

    stats_list.append({
        'ticker': t,
        'n_obs': len(s),
        'mean': s.mean(),
        'std': s.std(),
        'min': s.min(),
        'p25': s.quantile(0.25),
        'median': s.median(),
        'p75': s.quantile(0.75),
        'max': s.max(),
        'skew': s.skew(),
        'kurtosis': s.kurtosis(),
        'ac1': ac1,
        'ac5': ac5,
        'ac22': ac22,
        'log_mean': log_s.mean(),
        'log_std': log_s.std(),
        'log_skew': log_s.skew(),
        'log_kurtosis': log_s.kurtosis(),
    })

stats_df = pd.DataFrame(stats_list)
print(stats_df.to_string(index=False))

# Save stats
stats_df.to_csv(os.path.join(OUT_DIR, 'descriptive_stats_rv.csv'), index=False)
print(f"\nSaved descriptive stats to {OUT_DIR}/descriptive_stats_rv.csv")

# ============================================================
# 7. CROSS-ASSET CORRELATION
# ============================================================
print("\n" + "=" * 80)
print("7. CROSS-ASSET CORRELATION (RV)")
print("=" * 80)

# Only use assets with sufficient data
good_tickers = [t for t in tickers if (rv_clean[t].notna().sum() > 500)]
corr_matrix = rv_clean[good_tickers].corr()
print(f"Correlation matrix ({len(good_tickers)} assets):")
print(corr_matrix.round(3).to_string())

# ============================================================
# 8. RV vs BPV comparison (jump detection feasibility)
# ============================================================
print("\n" + "=" * 80)
print("8. RV vs BPV (JUMP COMPONENT)")
print("=" * 80)

bpv_clean = data['BPV'].replace(0, np.nan)
for t in good_tickers[:5]:
    rv_s = rv_clean[t].dropna()
    bpv_s = bpv_clean[t].reindex(rv_s.index).dropna()
    common = rv_s.index.intersection(bpv_s.index)
    if len(common) > 0:
        jump = np.maximum(rv_s[common].values - bpv_s[common].values, 0)
        pct_jump = (jump > 0).mean() * 100
        mean_jump = jump.mean()
        print(f"  {t}: {pct_jump:.1f}% days with positive jump, mean jump = {mean_jump:.4f}")

# ============================================================
# 9. Semivariance decomposition check
# ============================================================
print("\n" + "=" * 80)
print("9. SEMIVARIANCE DECOMPOSITION CHECK")
print("=" * 80)

good_clean = data['Good'].replace(0, np.nan)
bad_clean = data['Bad'].replace(0, np.nan)

for t in good_tickers[:5]:
    rv_s = rv_clean[t].dropna()
    g_s = good_clean[t].reindex(rv_s.index)
    b_s = bad_clean[t].reindex(rv_s.index)
    both_ok = g_s.notna() & b_s.notna()
    if both_ok.sum() > 0:
        total = g_s[both_ok] + b_s[both_ok]
        ratio = (total / rv_s[both_ok])
        print(f"  {t}: Good+Bad vs RV ratio: mean={ratio.mean():.4f}, std={ratio.std():.4f} (should be ~1.0)")

# ============================================================
# 10. Quarticity availability
# ============================================================
print("\n" + "=" * 80)
print("10. REALIZED QUARTICITY CHECK")
print("=" * 80)

rq_clean = data['RQ'].replace(0, np.nan)
for t in good_tickers[:5]:
    rq_s = rq_clean[t].dropna()
    print(f"  {t}: {len(rq_s)} non-zero RQ obs, mean={rq_s.mean():.4f}, std={rq_s.std():.4f}")

print("\n\nDONE - Data exploration complete")
