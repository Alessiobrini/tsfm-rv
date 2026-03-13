"""Generate bootstrap CI figure for cross-sectional mean QLIKE."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N_BOOT = 10_000

MODEL_DISPLAY = {
    'HAR': 'HAR', 'HAR_J': 'HAR-J', 'HAR_RS': 'HAR-RS', 'HARQ': 'HARQ',
    'Log_HAR': 'Log-HAR', 'ARFIMA': 'ARFIMA',
    'Chronos_Bolt_Small': 'Chronos-Bolt-S', 'Chronos_Bolt_Base': 'Chronos-Bolt-B',
    'Moirai_Small': 'Moirai-2.0-S', 'Lag_Llama': 'Lag-Llama',
    'toto': 'Toto', 'sundial': 'Sundial', 'moirai_moe_small': 'Moirai-MoE-S',
}
ECON_MODELS = {'HAR', 'HAR-J', 'HAR-RS', 'HARQ', 'Log-HAR', 'ARFIMA'}

# Only equity tickers (40 stocks)
STOCK_TICKERS = [
    'AAPL','ADBE','AMD','AMGN','AMZN','AXP','BA','C','CAT','CRM',
    'CSCO','CVX','DIS','GE','GOOGL','GS','HD','HON','IBM','JNJ',
    'JPM','KO','MCD','META','MMM','MRK','MSFT','NFLX','NKE','NVDA',
    'ORCL','PG','PM','SHW','TRV','TSLA','UNH','V','VZ','WMT','XOM'
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax_idx, h in enumerate([1, 5, 22]):
    df = pd.read_csv(f'results/volare/metrics/metrics_by_asset_h{h}.csv')
    df = df[df['ticker'].isin(STOCK_TICKERS)]
    df['model_display'] = df['model'].map(MODEL_DISPLAY)
    df = df.dropna(subset=['model_display'])

    # Pivot to get QLIKE per asset per model
    pivot = df.pivot_table(index='ticker', columns='model_display', values='QLIKE')

    # Filter models with mean QLIKE > 2 for readability
    means = pivot.mean()
    keep = means[means <= 2].index.tolist()
    pivot = pivot[keep]

    # Sort by mean QLIKE
    order = pivot.mean().sort_values().index.tolist()

    # Bootstrap
    results = []
    for model in order:
        vals = pivot[model].dropna().values
        boot_means = np.array([
            np.mean(np.random.choice(vals, size=len(vals), replace=True))
            for _ in range(N_BOOT)
        ])
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        results.append({
            'model': model,
            'mean': np.mean(vals),
            'lo': lo,
            'hi': hi,
            'is_econ': model in ECON_MODELS
        })

    y_pos = np.arange(len(results))
    for i, r in enumerate(results):
        color = '#1f77b4' if r['is_econ'] else '#d62728'
        ax = axes[ax_idx]
        ax.errorbar(r['mean'], i,
                     xerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]],
                     fmt='o', color=color, capsize=3, markersize=5, linewidth=1.5)

    axes[ax_idx].set_yticks(y_pos)
    axes[ax_idx].set_yticklabels([r['model'] for r in results], fontsize=11)
    axes[ax_idx].set_xlabel('Mean QLIKE', fontsize=12)
    axes[ax_idx].set_title(f'$h = {h}$', fontsize=13)
    axes[ax_idx].grid(axis='x', alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='#1f77b4', label='Econometric', linestyle='None', markersize=6),
    Line2D([0], [0], marker='o', color='#d62728', label='TSFM', linestyle='None', markersize=6)
]
axes[2].legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('paper/figures/fig_bootstrap_ci.pdf', bbox_inches='tight', dpi=300)
print("Saved paper/figures/fig_bootstrap_ci.pdf")
