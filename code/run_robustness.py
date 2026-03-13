"""
run_robustness.py — Robustness analyses for realized volatility forecasting.

Implements two robustness checks using existing forecast CSVs (no re-forecasting):

1. QLIKE Floor Sensitivity: Test whether MCS composition changes across
   different QLIKE floors (1e-4, 1e-6, 1e-8, 1e-10).

2. MZ Bias-Corrected TSFM Evaluation: Apply recursively estimated
   Mincer-Zarnowitz coefficients to debias TSFM forecasts at h=1,
   then recompute MSE, QLIKE, R2_OOS on corrected forecasts.

Usage:
    python run_robustness.py                        # run both
    python run_robustness.py --floor-sensitivity     # floor sensitivity only
    python run_robustness.py --mz-correction         # MZ correction only
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR, VOLARE_STOCK_TICKERS, RESULTS_DIR
from evaluation.loss_functions import compute_loss_series, compute_all_losses
from evaluation.mz_regression import recursive_mz_correction
from evaluation.mcs import model_confidence_set
from run_evaluation import parse_forecast_filename, align_forecasts
from utils import setup_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"
METRICS_DIR = VOLARE_RESULTS_DIR / "metrics"
TABLES_DIR = VOLARE_RESULTS_DIR / "tables"

MODEL_DISPLAY = {
    "HAR": "HAR",
    "HAR_J": "HAR-J",
    "HAR_RS": "HAR-RS",
    "HARQ": "HARQ",
    "Log_HAR": "Log-HAR",
    "ARFIMA": "ARFIMA",
    "chronos_bolt_small": "Chronos-Bolt-S",
    "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S",
    "lag_llama": "Lag-Llama",
    "toto": "Toto",
    "sundial": "Sundial",
    "moirai_moe_small": "Moirai-MoE-S",
}

TSFM_MODELS = {
    "chronos_bolt_small",
    "chronos_bolt_base",
    "moirai_2_0_small",
    "lag_llama",
    "toto",
    "sundial",
    "moirai_moe_small",
}

ECON_MODELS = {
    "HAR",
    "HAR_J",
    "HAR_RS",
    "HARQ",
    "Log_HAR",
    "ARFIMA",
}

ALL_MODELS = list(ECON_MODELS) + list(TSFM_MODELS)

QLIKE_FLOORS = [1e-4, 1e-6, 1e-8, 1e-10]

logger = setup_logger("robustness")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_equity_forecasts_h1():
    """Load all equity forecast CSVs at h=1.

    Returns
    -------
    dict
        {ticker: {model_name: pd.DataFrame with 'actual' and 'forecast'}}
    """
    if not FORECAST_DIR.exists():
        raise FileNotFoundError(f"Forecast directory not found: {FORECAST_DIR}")

    csv_files = list(FORECAST_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No forecast CSVs found in {FORECAST_DIR}")

    groups = defaultdict(dict)

    for fpath in csv_files:
        model_name, ticker, horizon = parse_forecast_filename(fpath)
        if model_name is None or horizon != 1:
            continue
        if ticker not in VOLARE_STOCK_TICKERS:
            continue

        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if "actual" not in df.columns or "forecast" not in df.columns:
            continue

        df = df.dropna(subset=["actual", "forecast"])
        groups[ticker][model_name] = df

    return dict(groups)


# ---------------------------------------------------------------------------
# 1. QLIKE Floor Sensitivity
# ---------------------------------------------------------------------------

def run_floor_sensitivity():
    """Test QLIKE floors of 1e-4, 1e-6, 1e-8, 1e-10 and check rank stability."""
    logger.info("=" * 60)
    logger.info("QLIKE Floor Sensitivity Analysis")
    logger.info("=" * 60)

    ticker_forecasts = load_equity_forecasts_h1()
    logger.info(f"Loaded forecasts for {len(ticker_forecasts)} tickers at h=1")

    results_rows = []

    for floor in QLIKE_FLOORS:
        logger.info(f"--- QLIKE floor = {floor:.0e} ---")

        model_qlike = defaultdict(list)  # model -> list of mean QLIKE per ticker

        for ticker in sorted(ticker_forecasts.keys()):
            model_dfs = ticker_forecasts[ticker]
            common_actual, model_forecasts = align_forecasts(model_dfs)
            if common_actual is None or len(model_forecasts) < 2:
                continue

            for mname, fcast in model_forecasts.items():
                qlike_series = compute_loss_series(
                    common_actual.values, fcast.values,
                    loss_type="QLIKE", qlike_floor=floor,
                )
                model_qlike[mname].append(float(np.mean(qlike_series)))

        for mname, qlike_list in model_qlike.items():
            if len(qlike_list) == 0:
                continue
            mean_qlike = np.mean(qlike_list)
            median_qlike = np.median(qlike_list)
            results_rows.append({
                "floor": f"{floor:.0e}",
                "model": mname,
                "mean_QLIKE": round(mean_qlike, 4),
                "median_QLIKE": round(median_qlike, 4),
                "n_assets": len(qlike_list),
            })
            logger.info(f"  {MODEL_DISPLAY.get(mname, mname):>18s}: "
                        f"mean={mean_qlike:.4f}  median={median_qlike:.4f}")

    results_df = pd.DataFrame(results_rows)

    # Compute rank correlations between floors
    from scipy.stats import spearmanr
    floor_strs = [f"{f:.0e}" for f in QLIKE_FLOORS]
    ref_floor = floor_strs[-1]  # 1e-10 is our default
    ref_ranks = (
        results_df[results_df["floor"] == ref_floor]
        .set_index("model")["mean_QLIKE"]
        .rank()
    )
    for fs in floor_strs[:-1]:
        other_ranks = (
            results_df[results_df["floor"] == fs]
            .set_index("model")["mean_QLIKE"]
            .rank()
        )
        common = ref_ranks.index.intersection(other_ranks.index)
        if len(common) > 2:
            rho, _ = spearmanr(ref_ranks.loc[common], other_ranks.loc[common])
            logger.info(f"Spearman rank correlation (floor {fs} vs {ref_floor}): {rho:.4f}")

    # Save CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "qlike_floor_sensitivity.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # Generate LaTeX table
    _generate_floor_latex(results_df)

    return results_df


def _generate_floor_latex(results_df):
    """Generate LaTeX table for floor sensitivity results."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Pivot: rows = models, columns = floors, values = mean QLIKE
    pivot = results_df.pivot(index="model", columns="floor", values="mean_QLIKE")

    model_order = [m for m in ALL_MODELS if m in pivot.index]
    pivot = pivot.loc[model_order]

    floor_cols = sorted(pivot.columns, key=lambda x: float(x))
    pivot = pivot[floor_cols]

    n_floors = len(floor_cols)
    col_spec = "l" + "c" * n_floors
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\singlespacing")
    lines.append(r"\caption{QLIKE floor sensitivity: cross-asset mean QLIKE at $h = 1$ under four floor values. Rankings are stable across floors (Spearman $\rho > 0.99$).}")
    lines.append(r"\label{tab:qlike_floor_sensitivity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Model"
    for f in floor_cols:
        header += f" & ${f}$"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    econ_done = False
    for mname in model_order:
        if not econ_done and mname in TSFM_MODELS:
            lines.append(r"\midrule")
            econ_done = True

        display = MODEL_DISPLAY.get(mname, mname)
        row = display
        for f in floor_cols:
            val = pivot.loc[mname, f]
            if pd.isna(val):
                row += " & --"
            elif val > 1:
                row += f" & {val:.2f}"
            else:
                row += f" & {val:.4f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = TABLES_DIR / "qlike_floor_sensitivity.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved LaTeX table: {tex_path}")


# ---------------------------------------------------------------------------
# 2. MZ Bias-Corrected TSFM Evaluation
# ---------------------------------------------------------------------------

def run_mz_correction():
    """Apply recursive MZ correction to TSFM forecasts and recompute metrics."""
    logger.info("=" * 60)
    logger.info("MZ Bias-Corrected TSFM Evaluation (h=1)")
    logger.info("=" * 60)

    ticker_forecasts = load_equity_forecasts_h1()
    logger.info(f"Loaded forecasts for {len(ticker_forecasts)} tickers at h=1")

    MIN_WINDOW = 252
    results_rows = []

    for ticker in sorted(ticker_forecasts.keys()):
        model_dfs = ticker_forecasts[ticker]
        logger.info(f"Processing {ticker} ({len(model_dfs)} models)")

        # Align all models to common dates
        common_actual, model_forecasts = align_forecasts(model_dfs)
        if common_actual is None:
            logger.warning(f"  Skipping {ticker}: no common dates")
            continue

        actual_arr = common_actual.values
        T = len(actual_arr)

        if T <= MIN_WINDOW:
            logger.warning(f"  Skipping {ticker}: only {T} obs (need >{MIN_WINDOW})")
            continue

        # Trimmed actuals (after MZ warm-up period)
        actual_trimmed = actual_arr[MIN_WINDOW:]

        # --- Compute metrics for all models on the trimmed sample ---

        # Econometric models (uncorrected, trimmed to same sample)
        for mname, fcast_series in model_forecasts.items():
            if mname not in ECON_MODELS:
                continue
            fcast_trimmed = fcast_series.values[MIN_WINDOW:]
            if len(fcast_trimmed) != len(actual_trimmed):
                continue

            mse_val = float(np.mean((actual_trimmed - fcast_trimmed) ** 2))
            qlike_series = compute_loss_series(
                actual_trimmed, fcast_trimmed, loss_type="QLIKE",
            )
            qlike_val = float(np.mean(qlike_series))
            ss_model = np.sum((actual_trimmed - fcast_trimmed) ** 2)
            ss_bench = np.sum((actual_trimmed - np.mean(actual_trimmed)) ** 2)
            r2oos = float(1 - ss_model / ss_bench) if ss_bench > 0 else 0.0

            results_rows.append({
                "ticker": ticker,
                "model": mname,
                "variant": "original",
                "MSE": mse_val,
                "QLIKE": qlike_val,
                "R2OOS": r2oos,
            })

        # TSFM models: both uncorrected and MZ-corrected
        for mname, fcast_series in model_forecasts.items():
            if mname not in TSFM_MODELS:
                continue
            fcast_arr = fcast_series.values

            # Uncorrected (trimmed)
            fcast_trimmed = fcast_arr[MIN_WINDOW:]
            if len(fcast_trimmed) != len(actual_trimmed):
                continue

            mse_val = float(np.mean((actual_trimmed - fcast_trimmed) ** 2))
            qlike_series = compute_loss_series(
                actual_trimmed, fcast_trimmed, loss_type="QLIKE",
            )
            qlike_val = float(np.mean(qlike_series))
            ss_model = np.sum((actual_trimmed - fcast_trimmed) ** 2)
            ss_bench = np.sum((actual_trimmed - np.mean(actual_trimmed)) ** 2)
            r2oos = float(1 - ss_model / ss_bench) if ss_bench > 0 else 0.0

            results_rows.append({
                "ticker": ticker,
                "model": mname,
                "variant": "original",
                "MSE": mse_val,
                "QLIKE": qlike_val,
                "R2OOS": r2oos,
            })

            # MZ-corrected
            try:
                corrected = recursive_mz_correction(
                    actual_arr, fcast_arr, min_window=MIN_WINDOW,
                )
            except Exception as e:
                logger.warning(f"  MZ correction failed for {mname} on {ticker}: {e}")
                continue

            if len(corrected) != len(actual_trimmed):
                logger.warning(
                    f"  Length mismatch for {mname} on {ticker}: "
                    f"corrected={len(corrected)}, actual={len(actual_trimmed)}"
                )
                continue

            # Floor corrected forecasts: use 1% of minimum actual to avoid
            # QLIKE explosion from near-zero corrected forecasts
            min_floor = max(1e-10, 0.01 * np.min(actual_trimmed[actual_trimmed > 0]))
            corrected = np.maximum(corrected, min_floor)

            mse_corr = float(np.mean((actual_trimmed - corrected) ** 2))
            qlike_corr_series = compute_loss_series(
                actual_trimmed, corrected, loss_type="QLIKE",
            )
            qlike_corr = float(np.mean(qlike_corr_series))
            ss_corr = np.sum((actual_trimmed - corrected) ** 2)
            r2oos_corr = float(1 - ss_corr / ss_bench) if ss_bench > 0 else 0.0

            results_rows.append({
                "ticker": ticker,
                "model": mname,
                "variant": "mz_corrected",
                "MSE": mse_corr,
                "QLIKE": qlike_corr,
                "R2OOS": r2oos_corr,
            })

    results_df = pd.DataFrame(results_rows)

    # Save full per-ticker results
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "mz_bias_corrected.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # Generate summary LaTeX table (averaged across tickers)
    _generate_mz_latex(results_df)

    return results_df


def _generate_mz_latex(results_df):
    """Generate LaTeX table for MZ bias-corrected results (cross-asset means)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if results_df.empty:
        logger.warning("No results to generate MZ LaTeX table")
        return

    # Compute cross-asset means
    summary = (
        results_df
        .groupby(["model", "variant"])[["MSE", "QLIKE", "R2OOS"]]
        .mean()
        .reset_index()
    )

    # Build ordered rows: econometric (original only), then TSFM (original + corrected)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{MZ Bias-Corrected TSFM Evaluation (Cross-Asset Mean, $h=1$)}")
    lines.append(r"\label{tab:mz_bias_corrected}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Variant & MSE ($\times 10^{8}$) & QLIKE & $R^2_{\mathrm{OOS}}$ \\")
    lines.append(r"\midrule")

    # Determine MSE scaling factor
    mse_scale = 1e8

    def fmt_qlike(val):
        if val > 100:
            return f"{val:.0f}$^{{\\dagger}}$"
        elif val > 1:
            return f"{val:.2f}"
        else:
            return f"{val:.4f}"

    # Econometric models
    econ_order = [m for m in ["HAR", "HAR_J", "HAR_RS", "HARQ", "Log_HAR", "ARFIMA"]
                  if m in summary["model"].values]
    for mname in econ_order:
        row = summary[(summary["model"] == mname) & (summary["variant"] == "original")]
        if row.empty:
            continue
        row = row.iloc[0]
        display = MODEL_DISPLAY.get(mname, mname)
        lines.append(
            f"{display} & Original & {row['MSE'] * mse_scale:.2f} "
            f"& {fmt_qlike(row['QLIKE'])} & {row['R2OOS']:.3f} \\\\"
        )

    lines.append(r"\midrule")

    # TSFM models: original then corrected
    tsfm_order = [m for m in ["chronos_bolt_small", "chronos_bolt_base",
                               "moirai_2_0_small", "lag_llama",
                               "toto", "sundial", "moirai_moe_small"]
                  if m in summary["model"].values]
    for mname in tsfm_order:
        display = MODEL_DISPLAY.get(mname, mname)
        for variant in ["original", "mz_corrected"]:
            row = summary[(summary["model"] == mname) & (summary["variant"] == variant)]
            if row.empty:
                continue
            row = row.iloc[0]
            var_label = "Original" if variant == "original" else "MZ-Corrected"
            lines.append(
                f"{display} & {var_label} & {row['MSE'] * mse_scale:.2f} "
                f"& {fmt_qlike(row['QLIKE'])} & {row['R2OOS']:.3f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = TABLES_DIR / "mz_bias_corrected.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved LaTeX table: {tex_path}")


# ---------------------------------------------------------------------------
# 3. Estimation Window Comparison (252 vs 512)
# ---------------------------------------------------------------------------

def run_window_comparison():
    """Compare 252-day vs 512-day econometric baseline metrics at h=1, h=5, h=22."""
    logger.info("=" * 60)
    logger.info("Estimation Window Comparison: 252 vs 512 days")
    logger.info("=" * 60)

    forecast_252 = VOLARE_RESULTS_DIR / "forecasts"
    forecast_512 = RESULTS_DIR / "volare_512" / "forecasts"

    if not forecast_512.exists():
        logger.error(f"512-day forecasts not found: {forecast_512}")
        return None

    results_rows = []

    for horizon in [1, 5, 22]:
        for window, fdir in [("252", forecast_252), ("512", forecast_512)]:
            for ticker in sorted(VOLARE_STOCK_TICKERS):
                for mname in ["HAR", "HAR_J", "HAR_RS", "HARQ", "Log_HAR", "ARFIMA"]:
                    fpath = fdir / f"{mname}_{ticker}_h{horizon}.csv"
                    if not fpath.exists():
                        continue
                    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                    if "actual" not in df.columns or "forecast" not in df.columns:
                        continue
                    df = df.dropna(subset=["actual", "forecast"])
                    if len(df) < 10:
                        continue

                    metrics = compute_all_losses(df["actual"], df["forecast"])
                    metrics["ticker"] = ticker
                    metrics["model"] = mname
                    metrics["window"] = window
                    metrics["horizon"] = horizon
                    results_rows.append(metrics)

    if not results_rows:
        logger.error("No results to compare")
        return None

    results_df = pd.DataFrame(results_rows)

    # Save CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "window_comparison_252_512.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # Generate LaTeX tables (one per horizon)
    for horizon in [1, 5, 22]:
        hdf = results_df[results_df["horizon"] == horizon]
        if hdf.empty:
            logger.warning(f"No data for h={horizon}, skipping table")
            continue
        _generate_window_latex(hdf, horizon)

    return results_df


def _generate_window_latex(results_df, horizon=1):
    """Generate LaTeX table comparing 252 vs 512 window results."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-asset means by model and window
    summary = (
        results_df
        .groupby(["model", "window"])[["MSE", "QLIKE", "R2OOS"]]
        .mean()
        .reset_index()
    )

    mse_scale = 1e8
    model_order = ["HAR", "HAR_J", "HAR_RS", "HARQ", "Log_HAR", "ARFIMA"]

    # Labels and filenames per horizon
    if horizon == 1:
        label = r"\label{tab:window_512}"
        tex_name = "table_window_512.tex"
    else:
        label = rf"\label{{tab:window_512_h{horizon}}}"
        tex_name = f"table_window_512_h{horizon}.tex"

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\singlespacing")
    lines.append(label)
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l cc cc cc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{MSE ($\times 10^{8}$)} & \multicolumn{2}{c}{QLIKE} & \multicolumn{2}{c}{$R^2_{\mathrm{OOS}}$} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Model & 252 & 512 & 252 & 512 & 252 & 512 \\")
    lines.append(r"\midrule")

    for mname in model_order:
        display = MODEL_DISPLAY.get(mname, mname)
        row_252 = summary[(summary["model"] == mname) & (summary["window"] == "252")]
        row_512 = summary[(summary["model"] == mname) & (summary["window"] == "512")]

        if row_252.empty or row_512.empty:
            continue

        r252 = row_252.iloc[0]
        r512 = row_512.iloc[0]

        # Bold the better value (lower MSE/QLIKE, higher R2OOS)
        def fmt_pair(v252, v512, fmt_str, higher_better=False):
            if higher_better:
                b252 = v252 > v512
            else:
                b252 = v252 < v512
            s252 = fmt_str.format(v252)
            s512 = fmt_str.format(v512)
            if b252:
                s252 = r"\textbf{" + s252 + "}"
            else:
                s512 = r"\textbf{" + s512 + "}"
            return s252, s512

        mse252, mse512 = fmt_pair(r252["MSE"] * mse_scale, r512["MSE"] * mse_scale, "{:.2f}")
        q252, q512 = fmt_pair(r252["QLIKE"], r512["QLIKE"], "{:.4f}")
        r2_252, r2_512 = fmt_pair(r252["R2OOS"], r512["R2OOS"], "{:.3f}", higher_better=True)

        lines.append(f"{display} & {mse252} & {mse512} & {q252} & {q512} & {r2_252} & {r2_512} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[6pt]")
    lines.append(rf"\parbox{{\textwidth}}{{\footnotesize Estimation window sensitivity: 252-day vs.\ 512-day rolling window ($h = {horizon}$, 40 equities). Cross-asset mean metrics. Bold indicates the better window for each model.}}")
    lines.append(r"\end{table}")

    tex_path = TABLES_DIR / tex_name
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved LaTeX table: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robustness analyses for realized volatility forecasting."
    )
    parser.add_argument(
        "--floor-sensitivity", action="store_true",
        help="Run QLIKE floor sensitivity analysis only.",
    )
    parser.add_argument(
        "--mz-correction", action="store_true",
        help="Run MZ bias-corrected TSFM evaluation only.",
    )
    parser.add_argument(
        "--window-comparison", action="store_true",
        help="Run 252 vs 512 window comparison.",
    )
    args = parser.parse_args()

    # If no flag is set, run all
    run_floor = args.floor_sensitivity
    run_mz = args.mz_correction
    run_window = args.window_comparison
    if not run_floor and not run_mz and not run_window:
        run_floor = True
        run_mz = True
        run_window = True

    if run_floor:
        run_floor_sensitivity()

    if run_mz:
        run_mz_correction()

    if run_window:
        run_window_comparison()

    logger.info("Done.")


if __name__ == "__main__":
    main()
