"""Generate all data exploration plots and save to data/plots/."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
from statsmodels.graphics.tsaplots import plot_acf
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/data/raw/RV_March2024.xlsx"
PLOT_DIR = "G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/data/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Load data ---
dates_df = pd.read_excel(DATA_PATH, sheet_name='Dates', header=None)
companies_df = pd.read_excel(DATA_PATH, sheet_name='Companies', header=None)
dates = pd.to_datetime(dates_df.iloc[:, 0])
tickers = companies_df.iloc[:, 0].tolist()

sheets = {}
for sheet in ['RV', 'BPV', 'Good', 'Bad', 'RQ', 'RV_5']:
    df = pd.read_excel(DATA_PATH, sheet_name=sheet, header=None)
    df.index = dates
    df.columns = tickers
    df = df.apply(pd.to_numeric, errors='coerce')
    sheets[sheet] = df

rv = sheets['RV'].replace(0, np.nan)
rv5 = sheets['RV_5'].replace(0, np.nan)

# Representative assets: tech, finance, consumer, industrial
rep_tickers = ['AAPL', 'JPM', 'AMZN', 'CAT']

# ============================================================
# Plot 1: Time series of RV for representative assets
# ============================================================
print("Plotting time series of RV...")
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for i, t in enumerate(rep_tickers):
    s = rv[t].dropna()
    axes[i].plot(s.index, s.values, linewidth=0.5, color='steelblue')
    axes[i].set_ylabel(f'RV ({t})', fontsize=11)
    axes[i].set_title(f'{t} — Daily Realized Variance (1-min)', fontsize=12)
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('Date', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rv_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved rv_timeseries.png")

# ============================================================
# Plot 2: Distribution of RV vs log-RV (histogram + QQ plot)
# ============================================================
print("Plotting RV vs log-RV distributions...")
fig, axes = plt.subplots(4, 4, figsize=(18, 16))
for i, t in enumerate(rep_tickers):
    s = rv[t].dropna()
    log_s = np.log(s)

    # Histogram RV
    axes[i, 0].hist(s.values, bins=100, density=True, color='steelblue', alpha=0.7, edgecolor='none')
    axes[i, 0].set_title(f'{t} — RV Histogram', fontsize=10)
    axes[i, 0].set_xlabel('RV')
    axes[i, 0].set_ylabel('Density')

    # QQ plot RV
    sp_stats.probplot(s.values, dist="norm", plot=axes[i, 1])
    axes[i, 1].set_title(f'{t} — RV QQ Plot', fontsize=10)
    axes[i, 1].get_lines()[0].set_markersize(2)

    # Histogram log-RV
    axes[i, 2].hist(log_s.values, bins=100, density=True, color='coral', alpha=0.7, edgecolor='none')
    axes[i, 2].set_title(f'{t} — log(RV) Histogram', fontsize=10)
    axes[i, 2].set_xlabel('log(RV)')
    axes[i, 2].set_ylabel('Density')

    # QQ plot log-RV
    sp_stats.probplot(log_s.values, dist="norm", plot=axes[i, 3])
    axes[i, 3].set_title(f'{t} — log(RV) QQ Plot', fontsize=10)
    axes[i, 3].get_lines()[0].set_markersize(2)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rv_vs_logrv_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved rv_vs_logrv_distributions.png")

# ============================================================
# Plot 3: ACF of RV and log-RV (out to 100+ lags)
# ============================================================
print("Plotting ACF of RV and log-RV...")
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
n_lags = 120
for i, t in enumerate(rep_tickers):
    s = rv[t].dropna()
    log_s = np.log(s)

    plot_acf(s.values, lags=n_lags, ax=axes[i, 0], alpha=0.05,
             title=f'{t} — ACF of RV', zero=False, markersize=2)
    axes[i, 0].set_xlabel('Lag (days)')

    plot_acf(log_s.values, lags=n_lags, ax=axes[i, 1], alpha=0.05,
             title=f'{t} — ACF of log(RV)', zero=False, markersize=2)
    axes[i, 1].set_xlabel('Lag (days)')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'acf_rv_logrv.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved acf_rv_logrv.png")

# ============================================================
# Plot 4: Cross-asset correlation heatmap
# ============================================================
print("Plotting cross-asset correlation heatmap...")
# Use all assets with >1000 obs
good_tickers = [t for t in tickers if rv[t].notna().sum() > 1000]
corr = rv[good_tickers].corr()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(good_tickers)))
ax.set_yticks(range(len(good_tickers)))
ax.set_xticklabels(good_tickers, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(good_tickers, fontsize=9)
ax.set_title('Cross-Asset Correlation of Daily Realized Variance (1-min)', fontsize=13)
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')

# Add correlation values
for i in range(len(good_tickers)):
    for j in range(len(good_tickers)):
        val = corr.values[i, j]
        color = 'white' if val > 0.7 or val < 0.3 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5.5, color=color)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved correlation_heatmap.png")

# ============================================================
# Plot 5: Volatility clustering — zoomed window
# ============================================================
print("Plotting volatility clustering...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

t = 'AAPL'
s = rv[t].dropna()

# Full series
axes[0].plot(s.index, s.values, linewidth=0.5, color='steelblue')
axes[0].set_title(f'{t} — Full RV Series (2003-2024)', fontsize=12)
axes[0].set_ylabel('RV')
axes[0].grid(True, alpha=0.3)

# 2020 COVID period zoom
mask = (s.index >= '2019-01-01') & (s.index <= '2021-12-31')
s_zoom = s[mask]
axes[1].plot(s_zoom.index, s_zoom.values, linewidth=0.8, color='steelblue')
axes[1].fill_between(s_zoom.index, 0, s_zoom.values, alpha=0.3, color='steelblue')
axes[1].set_title(f'{t} — RV Zoom: 2019-2021 (COVID period)', fontsize=12)
axes[1].set_ylabel('RV')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'volatility_clustering.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved volatility_clustering.png")

# ============================================================
# Plot 6: 1-min vs 5-min RV comparison
# ============================================================
print("Plotting 1-min vs 5-min RV comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, t in enumerate(rep_tickers):
    ax = axes[i // 2, i % 2]
    r1 = rv[t].dropna()
    r5 = rv5[t].reindex(r1.index).dropna()
    common = r1.index.intersection(r5.index)

    ax.scatter(r1[common].values, r5[common].values, s=1, alpha=0.3, color='steelblue')
    max_val = max(r1[common].max(), r5[common].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.5)
    ax.set_xlabel('RV (1-min)')
    ax.set_ylabel('RV (5-min)')
    ax.set_title(f'{t}: 1-min vs 5-min RV', fontsize=11)
    ax.grid(True, alpha=0.3)
    corr_val = np.corrcoef(r1[common].values, r5[common].values)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rv_1min_vs_5min.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved rv_1min_vs_5min.png")

# ============================================================
# Plot 7: Jump component time series
# ============================================================
print("Plotting jump component time series...")
bpv = sheets['BPV'].replace(0, np.nan)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, t in enumerate(rep_tickers):
    ax = axes[i // 2, i % 2]
    r = rv[t].dropna()
    b = bpv[t].reindex(r.index).dropna()
    common = r.index.intersection(b.index)
    jump = np.maximum(r[common] - b[common], 0)

    ax.bar(common, jump.values, width=1, color='coral', alpha=0.6, label='Jump (max(RV-BPV, 0))')
    ax.plot(common, r[common].values, linewidth=0.3, color='steelblue', alpha=0.5, label='RV')
    ax.set_title(f'{t} — Jump Component', fontsize=11)
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'jump_component.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved jump_component.png")

print("\nAll plots saved to:", PLOT_DIR)
print("DONE")
