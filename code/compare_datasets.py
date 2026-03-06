"""
compare_datasets.py — Diagnostic comparison of CAPIRe vs VOLARE realized volatility.

Compares the two datasets on overlapping tickers and date ranges.
Produces descriptive stats, correlation analysis, time series plots,
and distributional comparisons.

Usage:
    python compare_datasets.py [--tickers AAPL JPM AMZN CAT]
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from config import REPRESENTATIVE_TICKERS, DATA_DIR
from data_loader import load_data


OUTPUT_DIR = DATA_DIR / "dataset_comparison"


def compute_descriptive_stats(rv_series, name):
    """Compute descriptive statistics for an RV series."""
    s = rv_series.dropna()
    acf_vals = []
    for lag in [1, 5, 22]:
        if len(s) > lag:
            acf_vals.append(s.autocorr(lag=lag))
        else:
            acf_vals.append(np.nan)

    return {
        'dataset': name,
        'n_obs': len(s),
        'mean': s.mean(),
        'std': s.std(),
        'median': s.median(),
        'skewness': s.skew(),
        'kurtosis': s.kurt(),
        'min': s.min(),
        'max': s.max(),
        'ACF(1)': acf_vals[0],
        'ACF(5)': acf_vals[1],
        'ACF(22)': acf_vals[2],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare CAPIRe vs VOLARE datasets")
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Tickers to compare (default: AAPL JPM AMZN CAT)')
    args = parser.parse_args()

    tickers = args.tickers or REPRESENTATIVE_TICKERS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CAPIRe data for {tickers}...")
    capire = load_data(dataset="capire", tickers=tickers)

    print(f"Loading VOLARE data for {tickers}...")
    volare = load_data(dataset="volare", tickers=tickers)

    # --- 1. Descriptive stats ---
    all_stats = []
    for ticker in tickers:
        if ticker in capire.tickers:
            all_stats.append({
                'ticker': ticker,
                **compute_descriptive_stats(capire.rv[ticker], 'CAPIRe'),
            })
        if ticker in volare.tickers:
            all_stats.append({
                'ticker': ticker,
                **compute_descriptive_stats(volare.rv[ticker], 'VOLARE'),
            })

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(OUTPUT_DIR / "descriptive_stats.csv", index=False)
    print(f"\nDescriptive statistics saved to {OUTPUT_DIR / 'descriptive_stats.csv'}")
    print(stats_df.to_string(index=False))

    # --- 2. Correlation on matched dates ---
    corr_results = []
    for ticker in tickers:
        if ticker not in capire.tickers or ticker not in volare.tickers:
            continue

        c = capire.rv[ticker].dropna()
        v = volare.rv[ticker].dropna()
        common = c.index.intersection(v.index)

        if len(common) < 50:
            print(f"  {ticker}: only {len(common)} overlapping dates, skipping")
            continue

        c_aligned = c.loc[common]
        v_aligned = v.loc[common]

        pearson_r, pearson_p = stats.pearsonr(c_aligned, v_aligned)
        spearman_r, spearman_p = stats.spearmanr(c_aligned, v_aligned)
        ks_stat, ks_p = stats.ks_2samp(c_aligned, v_aligned)

        # Mean absolute difference
        mad = (c_aligned - v_aligned).abs().mean()
        # Relative difference
        rel_diff = ((c_aligned - v_aligned) / c_aligned).mean()

        corr_results.append({
            'ticker': ticker,
            'n_common_dates': len(common),
            'overlap_start': str(common.min().date()),
            'overlap_end': str(common.max().date()),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'mean_abs_diff': mad,
            'mean_rel_diff': rel_diff,
        })

    if corr_results:
        corr_df = pd.DataFrame(corr_results)
        corr_df.to_csv(OUTPUT_DIR / "correlation_analysis.csv", index=False)
        print(f"\nCorrelation analysis:")
        print(corr_df[['ticker', 'n_common_dates', 'pearson_r', 'spearman_r',
                        'ks_statistic', 'ks_p_value', 'mean_abs_diff']].to_string(index=False))

    # --- 3. Time series overlay plots ---
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 3.5 * len(tickers)),
                             squeeze=False)

    for i, ticker in enumerate(tickers):
        ax = axes[i, 0]

        if ticker in capire.tickers:
            c = capire.rv[ticker].dropna()
            ax.plot(c.index, c.values, alpha=0.7, linewidth=0.5,
                    label='CAPIRe', color='steelblue')

        if ticker in volare.tickers:
            v = volare.rv[ticker].dropna()
            ax.plot(v.index, v.values, alpha=0.7, linewidth=0.5,
                    label='VOLARE', color='coral')

        ax.set_title(f'{ticker} — Daily Realized Variance', fontsize=11)
        ax.set_ylabel('RV')
        ax.legend(loc='upper right', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rv_overlay_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nTime series plots saved to {OUTPUT_DIR / 'rv_overlay_timeseries.png'}")

    # --- 4. Overlapping period zoom + difference ---
    fig, axes = plt.subplots(len(tickers), 2, figsize=(14, 3.5 * len(tickers)),
                             squeeze=False)

    for i, ticker in enumerate(tickers):
        if ticker not in capire.tickers or ticker not in volare.tickers:
            continue

        c = capire.rv[ticker].dropna()
        v = volare.rv[ticker].dropna()
        common = c.index.intersection(v.index)
        if len(common) < 50:
            continue

        c_a = c.loc[common]
        v_a = v.loc[common]

        # Overlay in overlap
        ax1 = axes[i, 0]
        ax1.plot(common, c_a.values, alpha=0.7, linewidth=0.5,
                 label='CAPIRe', color='steelblue')
        ax1.plot(common, v_a.values, alpha=0.7, linewidth=0.5,
                 label='VOLARE', color='coral')
        ax1.set_title(f'{ticker} — Overlap Period', fontsize=10)
        ax1.set_ylabel('RV')
        ax1.legend(fontsize=8)

        # QQ plot
        ax2 = axes[i, 1]
        q_c = np.quantile(c_a, np.linspace(0.01, 0.99, 100))
        q_v = np.quantile(v_a, np.linspace(0.01, 0.99, 100))
        ax2.scatter(q_c, q_v, s=8, alpha=0.7, color='purple')
        lims = [min(q_c.min(), q_v.min()), max(q_c.max(), q_v.max())]
        ax2.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('CAPIRe quantiles')
        ax2.set_ylabel('VOLARE quantiles')
        ax2.set_title(f'{ticker} — QQ Plot', fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rv_overlap_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overlap comparison plots saved to {OUTPUT_DIR / 'rv_overlap_comparison.png'}")

    # --- 5. Summary ---
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"CAPIRe: {len(capire.dates)} dates, {capire.dates[0].date()} to {capire.dates[-1].date()}")
    print(f"VOLARE: {len(volare.dates)} dates, {volare.dates[0].date()} to {volare.dates[-1].date()}")

    if corr_results:
        avg_pearson = np.mean([r['pearson_r'] for r in corr_results])
        avg_ks_p = np.mean([r['ks_p_value'] for r in corr_results])
        print(f"\nAvg Pearson r (overlap): {avg_pearson:.4f}")
        print(f"Avg KS p-value: {avg_ks_p:.4f}")

        if avg_pearson > 0.95:
            print("  => Series are very highly correlated — may be near-identical estimators")
        elif avg_pearson > 0.80:
            print("  => Series are strongly correlated but meaningfully different")
        else:
            print("  => Series show notable differences — dual presentation justified")

        if avg_ks_p < 0.05:
            print("  => KS test: distributions are significantly different")
        else:
            print("  => KS test: cannot reject identical distributions")

    print("\nDiagnostic complete. Review plots in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
