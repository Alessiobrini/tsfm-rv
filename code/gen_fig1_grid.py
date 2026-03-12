"""Generate Figure 1: 2x2 grid of forecast vs actual RV at h=1 for AAPL, JPM, TSLA, EURUSD."""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

base = r"G:\Other computers\Dell Duke\Workfiles\Postdoc_file\human_x_AI_finance\results\volare\forecasts"
out = r"G:\Other computers\Dell Duke\Workfiles\Postdoc_file\human_x_AI_finance\paper\figures\fig1_forecast_vs_actual.pdf"

tickers = ['AAPL', 'JPM', 'TSLA', 'EURUSD']
models = {
    'Log-HAR': 'Log_HAR',
    'Moirai 2.0': 'moirai_2_0_small',
    'Chronos-Bolt-Small': 'chronos_bolt_small',
}
colors = {
    'Actual': '#888888',
    'Log-HAR': '#1f77b4',
    'Moirai 2.0': '#d62728',
    'Chronos-Bolt-Small': '#2ca02c',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, ticker in enumerate(tickers):
    ax = axes[idx]

    # Load all model forecasts and align on common dates
    dfs = {}
    for label, prefix in models.items():
        fpath = os.path.join(base, f"{prefix}_{ticker}_h1.csv")
        df = pd.read_csv(fpath, parse_dates=['date'])
        df = df.set_index('date').sort_index()
        dfs[label] = df

    # Find common dates across all models
    common_idx = dfs['Log-HAR'].index
    for label in models:
        common_idx = common_idx.intersection(dfs[label].index)

    # Take last 500 observations
    common_idx = common_idx[-500:]

    # Plot actual (from any model -- they should be the same)
    actual = dfs['Log-HAR'].loc[common_idx, 'actual']
    ax.plot(common_idx, actual, color=colors['Actual'], linewidth=0.7, alpha=0.8, label='Actual')

    # Plot model forecasts
    for label in models:
        forecast = dfs[label].loc[common_idx, 'forecast']
        ax.plot(common_idx, forecast, color=colors[label], linewidth=0.7, alpha=0.75, label=label)

    ax.set_title(ticker, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel('Realized Variance', fontsize=14)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    # Rotate x-axis dates
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha('right')

# Single legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12,
           frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(out, bbox_inches='tight', dpi=300)
print(f"Saved to {out}")
