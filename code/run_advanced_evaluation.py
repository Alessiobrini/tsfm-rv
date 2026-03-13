"""
run_advanced_evaluation.py — Mincer-Zarnowitz regressions & Giacomini-Rossi Fluctuation Tests.

Reads existing forecast CSVs (both CAPIRe and VOLARE) and produces:
    1. MZ regression tables (per horizon, averaged across assets)
    2. GR Fluctuation Test plots (rolling DM vs benchmark)
    3. LaTeX tables for the paper

Usage:
    python run_advanced_evaluation.py [--dataset volare] [--horizons 1 5 22]
                                      [--benchmark HAR] [--gr-plot]
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RESULTS_DIR, VOLARE_RESULTS_DIR, FIGURES_DIR,
    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
)
from evaluation.mz_regression import mz_regression, mz_table
from evaluation.gr_fluctuation import gr_fluctuation_test, gr_fluctuation_multiple
from evaluation.loss_functions import compute_loss_series
from run_evaluation import parse_forecast_filename, align_forecasts
from utils import setup_logger


# Model display name mapping (filename prefix -> paper name)
MODEL_DISPLAY = {
    "HAR": "HAR",
    "HAR_J": "HAR-J",
    "HAR_RS": "HAR-RS",
    "HARQ": "HARQ",
    "Log_HAR": "Log-HAR",
    "ARFIMA": "ARFIMA",
    "chronos_bolt_small": "Chronos-Bolt-S",
    "chronos_bolt_base": "Chronos-Bolt-B",
    "chronos2_small": "Chronos-2",
    "moirai_2_0_small": "Moirai-2.0-S",
    "lag_llama": "Lag-Llama",
    "toto": "Toto",
    "sundial": "Sundial",
    "moirai_moe_small": "Moirai-MoE-S",
}

# Models to exclude from analysis (duplicates or irrelevant variants)
EXCLUDE_MODELS = set()  # Add model prefixes here to skip them


def load_forecasts(forecast_dir: Path):
    """Load all forecast CSVs from a directory, grouped by (ticker, horizon)."""
    csv_files = list(forecast_dir.glob("*.csv"))
    groups = defaultdict(dict)
    for fpath in csv_files:
        model_name, ticker, horizon = parse_forecast_filename(fpath)
        if model_name is None:
            continue
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if 'actual' not in df.columns or 'forecast' not in df.columns:
            continue
        df = df.dropna(subset=['actual', 'forecast'])
        groups[(ticker, horizon)][model_name] = df
    return groups


def run_mz_analysis(groups, horizons, logger):
    """Run Mincer-Zarnowitz regressions across all assets and horizons.

    Returns a dict: {horizon: DataFrame with avg MZ stats across assets}.
    """
    mz_results_by_h = {}

    for h in horizons:
        h_groups = {k: v for k, v in groups.items() if k[1] == h}
        if not h_groups:
            continue

        asset_tables = []
        for (ticker, horizon), model_dfs in sorted(h_groups.items()):
            actual, forecasts = align_forecasts(model_dfs)
            if actual is None or len(actual) < 50:
                continue

            # Convert to dict of arrays
            fcast_dict = {m: f.values for m, f in forecasts.items()}
            tbl = mz_table(actual.values, fcast_dict, hac_lags=max(1, h - 1))
            tbl['ticker'] = ticker
            asset_tables.append(tbl)

        if not asset_tables:
            continue

        all_tbl = pd.concat(asset_tables)

        # Average across assets
        numeric_cols = ['alpha', 'alpha_se', 'alpha_pval',
                        'beta', 'beta_se', 'beta_pval',
                        'R2', 'F_stat', 'F_pval']
        avg = all_tbl.groupby(all_tbl.index)[numeric_cols].mean()

        # Count rejections of joint H0 at 5%
        reject_counts = all_tbl.groupby(all_tbl.index)['F_pval'].apply(
            lambda x: (x < 0.05).mean()
        ).rename('pct_reject_05')
        avg = avg.join(reject_counts)

        avg['horizon'] = h
        mz_results_by_h[h] = avg

        logger.info(f"\nMZ Regression — h={h} (avg across {len(asset_tables)} assets):")
        display = avg[['alpha', 'beta', 'R2', 'F_pval', 'pct_reject_05']].round(4)
        # Rename for display
        display_names = {m: MODEL_DISPLAY.get(m, m) for m in display.index}
        display = display.rename(index=display_names)
        logger.info(f"\n{display.to_string()}")

    return mz_results_by_h


def run_gr_analysis(groups, horizons, benchmark, loss_type, logger,
                    window_fraction=0.3):
    """Run Giacomini-Rossi Fluctuation Tests against a benchmark.

    Returns a dict: {horizon: {model: GRFluctuationResult averaged info}}.
    """
    gr_results_by_h = {}

    for h in horizons:
        h_groups = {k: v for k, v in groups.items() if k[1] == h}
        if not h_groups:
            continue

        # Collect rolling DM series per model (across assets)
        model_rolling = defaultdict(list)
        model_sup_stats = defaultdict(list)
        model_reject_05 = defaultdict(list)

        for (ticker, horizon), model_dfs in sorted(h_groups.items()):
            if benchmark not in model_dfs:
                continue

            actual, forecasts = align_forecasts(model_dfs)
            if actual is None or len(actual) < 100:
                continue

            dates = actual.index
            fcast_dict = {m: f.values for m, f in forecasts.items()}
            if benchmark not in fcast_dict:
                continue

            results = gr_fluctuation_multiple(
                actual.values, fcast_dict, benchmark,
                loss_type=loss_type,
                window_fraction=window_fraction,
                hac_lags=max(1, h - 1),
                dates=dates,
            )

            for model_name, res in results.items():
                model_rolling[model_name].append(res.rolling_dm)
                model_sup_stats[model_name].append(res.sup_stat)
                model_reject_05[model_name].append(res.reject_05)

        if not model_rolling:
            continue

        # Summarize
        summary_rows = []
        gr_rolling_avg = {}

        for model_name in sorted(model_rolling.keys()):
            sup_stats = model_sup_stats[model_name]
            rejects = model_reject_05[model_name]
            display_name = MODEL_DISPLAY.get(model_name, model_name)
            summary_rows.append({
                'model': display_name,
                'avg_sup_stat': np.mean(sup_stats),
                'max_sup_stat': np.max(sup_stats),
                'pct_reject_05': np.mean(rejects),
                'n_assets': len(sup_stats),
            })

            # Average rolling DM across assets (align by relative position)
            all_series = model_rolling[model_name]
            max_len = max(len(s) for s in all_series)
            # Pad shorter series with NaN and average
            padded = np.full((len(all_series), max_len), np.nan)
            for i, s in enumerate(all_series):
                padded[i, :len(s)] = s.values
            avg_rolling = np.nanmean(padded, axis=0)
            # Use dates from the longest series
            longest = max(all_series, key=len)
            gr_rolling_avg[model_name] = pd.Series(
                avg_rolling[:len(longest)], index=longest.index,
                name=display_name,
            )

        summary_df = pd.DataFrame(summary_rows).set_index('model')
        gr_results_by_h[h] = {
            'summary': summary_df,
            'rolling': gr_rolling_avg,
        }

        logger.info(f"\nGR Fluctuation Test vs {MODEL_DISPLAY.get(benchmark, benchmark)} — h={h}:")
        logger.info(f"\n{summary_df.round(3).to_string()}")

    return gr_results_by_h


def generate_mz_latex(mz_results_by_h, output_dir):
    """Generate LaTeX table for MZ regression results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for h, df in mz_results_by_h.items():
        display_names = {m: MODEL_DISPLAY.get(m, m) for m in df.index}
        df = df.rename(index=display_names)

        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append(f"\\caption{{Mincer-Zarnowitz forecast efficiency regression, $h = {h}$}}")
        lines.append(f"\\label{{tab:mz_h{h}}}")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\toprule")
        lines.append("Model & $\\hat{\\alpha}$ & $\\hat{\\beta}$ & "
                      "$R^2$ & $F$-stat & $p$(joint) & \\% Reject \\\\")
        lines.append("\\midrule")

        for model in df.index:
            row = df.loc[model]
            alpha_str = f"{row['alpha']:.4f}"
            beta_str = f"{row['beta']:.3f}"
            r2_str = f"{row['R2']:.3f}"
            f_str = f"{row['F_stat']:.2f}"
            fp_str = f"{row['F_pval']:.3f}"
            rej_str = f"{row['pct_reject_05']:.1%}"
            lines.append(f"{model} & {alpha_str} & {beta_str} & "
                          f"{r2_str} & {f_str} & {fp_str} & {rej_str} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\begin{tablenotes}\\small")
        lines.append("\\item \\textit{Notes:} Mincer-Zarnowitz regression: "
                      "$\\text{RV}_t = \\alpha + \\beta \\hat{f}_t + \\varepsilon_t$. "
                      "Under forecast efficiency, $\\alpha = 0$ and $\\beta = 1$. "
                      "$F$-stat tests the joint null. HAC standard errors (Newey-West). "
                      "\\% Reject shows the fraction of assets rejecting at 5\\%.")
        lines.append("\\end{tablenotes}")
        lines.append("\\end{table}")

        tex_path = output_dir / f"mz_regression_h{h}.tex"
        with open(tex_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"  Saved: {tex_path}")


def generate_gr_plots(gr_results_by_h, benchmark, output_dir):
    """Generate GR Fluctuation Test plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    output_dir.mkdir(parents=True, exist_ok=True)
    bench_display = MODEL_DISPLAY.get(benchmark, benchmark)

    for h, data in gr_results_by_h.items():
        rolling = data['rolling']
        summary = data['summary']
        if not rolling:
            continue

        # Select top models to plot (avoid clutter)
        models_to_plot = list(rolling.keys())
        if len(models_to_plot) > 8:
            # Pick the ones with highest sup_stat
            top = summary.nlargest(8, 'avg_sup_stat')
            models_to_plot = [m for m in rolling.keys()
                              if MODEL_DISPLAY.get(m, m) in top.index]

        fig, ax = plt.subplots(figsize=(12, 5))

        for model_name in sorted(models_to_plot):
            series = rolling[model_name]
            display_name = MODEL_DISPLAY.get(model_name, model_name)
            ax.plot(series.index, series.values, label=display_name,
                    linewidth=0.8, alpha=0.85)

        # Critical value bands (approximate for mu=0.3)
        cv_05 = 2.80
        ax.axhline(y=cv_05, color='red', linestyle='--', linewidth=0.7,
                    alpha=0.6, label=f'5% CV ($\\pm${cv_05:.2f})')
        ax.axhline(y=-cv_05, color='red', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        ax.set_ylabel('Rolling DM statistic')
        ax.set_title(f'Giacomini-Rossi Fluctuation Test vs {bench_display} ($h = {h}$)')
        ax.legend(fontsize=7, ncol=3, loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        fig.autofmt_xdate()
        plt.tight_layout()

        fig_path = output_dir / f"gr_fluctuation_h{h}.pdf"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced evaluation: MZ regression & GR Fluctuation Test"
    )
    parser.add_argument('--dataset', default='volare',
                        choices=['capire', 'volare'],
                        help='Which forecast set to evaluate')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Horizons to evaluate (default: all found)')
    parser.add_argument('--benchmark', default='HAR',
                        help='Benchmark model for GR test (default: HAR)')
    parser.add_argument('--loss', default='QLIKE',
                        choices=['MSE', 'MAE', 'QLIKE'],
                        help='Loss function for GR test')
    parser.add_argument('--gr-plot', action='store_true',
                        help='Generate GR fluctuation plots')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX tables')
    parser.add_argument('--window-fraction', type=float, default=0.3,
                        help='GR window fraction (default: 0.3)')
    args = parser.parse_args()

    logger = setup_logger("advanced_eval")

    # Determine paths
    if args.dataset == 'volare':
        forecast_dir = VOLARE_RESULTS_DIR / "forecasts"
        metrics_dir = VOLARE_RESULTS_DIR / "metrics"
        tables_dir = VOLARE_RESULTS_DIR / "tables"
    else:
        forecast_dir = RESULTS_DIR / "forecasts"
        metrics_dir = RESULTS_DIR / "metrics"
        tables_dir = RESULTS_DIR / "tables"

    logger.info(f"Loading forecasts from {forecast_dir}")
    groups = load_forecasts(forecast_dir)
    logger.info(f"Loaded {len(groups)} (ticker, horizon) groups")

    all_horizons = sorted(set(h for _, h in groups.keys()))
    horizons = args.horizons or all_horizons
    logger.info(f"Horizons: {horizons}")

    # --- Mincer-Zarnowitz ---
    logger.info("\n" + "=" * 60)
    logger.info("MINCER-ZARNOWITZ REGRESSIONS")
    logger.info("=" * 60)
    mz_results = run_mz_analysis(groups, horizons, logger)

    # Save MZ results
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for h, df in mz_results.items():
        df.to_csv(metrics_dir / f"mz_regression_h{h}.csv")
    logger.info(f"MZ results saved to {metrics_dir}")

    if args.latex:
        generate_mz_latex(mz_results, tables_dir)

    # --- Giacomini-Rossi ---
    logger.info("\n" + "=" * 60)
    logger.info(f"GIACOMINI-ROSSI FLUCTUATION TEST (benchmark: {args.benchmark})")
    logger.info("=" * 60)
    gr_results = run_gr_analysis(
        groups, horizons, args.benchmark, args.loss, logger,
        window_fraction=args.window_fraction,
    )

    # Save GR summary
    for h, data in gr_results.items():
        data['summary'].to_csv(metrics_dir / f"gr_fluctuation_h{h}.csv")
    logger.info(f"GR results saved to {metrics_dir}")

    # GR plots
    if args.gr_plot:
        generate_gr_plots(gr_results, args.benchmark, FIGURES_DIR)

    logger.info("\nAdvanced evaluation complete.")


if __name__ == "__main__":
    main()
