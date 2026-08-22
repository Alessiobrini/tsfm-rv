"""
run_density_forecasts.py — Run zero-shot TSFM DENSITY forecasts on VOLARE.

Mirrors run_foundation_volare.py but calls model.predict_density() and
persists the full quantile grid per asset/horizon, in the same CSV
schema as run_har_density_baseline.py (columns: actual, q_0025, q_0050,
..., q_0975).

Output layout:
    results/volare/density/<model>/<ticker>_h<h>.csv
    results/volare/density/<model>/_summary.csv  (CRPS / KS / coverage)

Notes:
    - TTM has no predictive distribution and is excluded by default with
      an explicit warning. Override with --include-ttm at your peril.
    - Sample-based wrappers internally bump num_samples to
      DENSITY_NUM_SAMPLES (200), so wall-time per asset is ~5-10x the
      point-forecast runner. Plan SLURM time accordingly.
    - PIT and coverage are scale-invariant under log; CRPS is reported
      in both log and level space, with log as primary per Brini's note.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    REPRESENTATIVE_TICKERS,
    VOLARE_FUTURES_TICKERS,
    VOLARE_FX_TICKERS,
    VOLARE_RESULTS_DIR,
    VOLARE_STOCK_TICKERS,
    fm_cfg,
    forecast_cfg,
)
from data_loader import load_data
from evaluation.density import DEFAULT_QUANTILE_LEVELS, density_summary
from forecasting.rolling_forecast import zero_shot_density_forecast
from models.foundation import get_foundation_model
from utils import setup_logger

DENSITY_DIR = VOLARE_RESULTS_DIR / "density"
DATASET_KEY = {"stocks": "volare", "fx": "volare_fx", "futures": "volare_futures"}

# TSFM model identifiers that emit a predictive distribution.
DENSITY_MODELS = [
    "chronos-bolt-small",
    "chronos-bolt-base",
    "timesfm-2.5",
    "moirai-2.0-small",
    "moirai-moe-small",
    "lag-llama",
    "sundial",
    "toto",
]
POINT_ONLY_MODELS = {"ttm"}


def _safe(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _q_columns(levels: np.ndarray) -> List[str]:
    return [f"q_{int(round(level * 1000)):04d}" for level in levels]


def _output_dir(model_name: str) -> Path:
    return DENSITY_DIR / _safe(model_name)


def _output_path(model_name: str, ticker: str, horizon: int, context_length: int) -> Path:
    base = _output_dir(model_name)
    suffix = "" if context_length == 512 else f"_ctx{context_length}"
    return base / f"{ticker}_h{horizon}{suffix}.csv"


def _write_density(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def _read_density(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def score_density_frame(df: pd.DataFrame, levels: np.ndarray) -> dict:
    actual = df["actual"].to_numpy()
    q_grid_level = df[_q_columns(levels)].to_numpy()
    q_grid_log = np.log(np.clip(q_grid_level, 1e-30, None))
    log_actual = np.log(np.clip(actual, 1e-30, None))

    log_summary = density_summary(log_actual, q_grid_log, levels)
    lvl_summary = density_summary(actual, q_grid_level, levels)

    out = {"n_obs": int(len(actual))}
    out.update({f"log_{k}": v for k, v in log_summary.to_dict().items() if k != "n_obs"})
    out.update({f"lvl_{k}": v for k, v in lvl_summary.to_dict().items() if k != "n_obs"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TSFM zero-shot density forecasts on VOLARE"
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"TSFM names. Defaults to all density-capable models: {DENSITY_MODELS}",
    )
    parser.add_argument("--device", default=fm_cfg.device)
    parser.add_argument(
        "--context-length", type=int, default=forecast_cfg.tsfm_context_length,
    )
    parser.add_argument(
        "--asset-class", default="stocks",
        choices=["stocks", "fx", "futures"],
    )
    parser.add_argument("--all-tickers", action="store_true")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (model, ticker, horizon) triples whose density file already exists.",
    )
    parser.add_argument(
        "--include-ttm", action="store_true",
        help="Run TTM even though it has no density. Will raise at predict_density() call.",
    )
    args = parser.parse_args()

    if args.all_tickers:
        tickers: Sequence[str] = {
            "stocks": VOLARE_STOCK_TICKERS,
            "fx": VOLARE_FX_TICKERS,
            "futures": VOLARE_FUTURES_TICKERS,
        }[args.asset_class]
    else:
        tickers = args.tickers or REPRESENTATIVE_TICKERS

    horizons = args.horizons or forecast_cfg.horizons
    raw_models = args.models or DENSITY_MODELS
    model_names: List[str] = []
    for name in raw_models:
        if name in POINT_ONLY_MODELS and not args.include_ttm:
            print(f"Skipping {name}: point-only, no predictive distribution.")
            continue
        model_names.append(name)

    levels = DEFAULT_QUANTILE_LEVELS

    logger = setup_logger("density_forecasts")
    logger.info("=== VOLARE Dataset — TSFM Density Forecasts ===")
    logger.info(f"Models: {model_names}")
    logger.info(f"Tickers: {list(tickers)}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Device: {args.device}, ctx_length: {args.context_length}")
    logger.info(f"Quantile grid (K={len(levels)}): {list(np.round(levels, 4))}")

    data = load_data(dataset=DATASET_KEY[args.asset_class], tickers=list(tickers))

    summary_rows: dict[str, list] = {m: [] for m in model_names}

    for model_name in model_names:
        logger.info("\n" + "=" * 60)

        # Build pending list before loading the model.
        pending: List[tuple] = []
        for ticker in tickers:
            for horizon in horizons:
                out_path = _output_path(model_name, ticker, horizon, args.context_length)
                if args.skip_existing and out_path.exists():
                    df_cached = _read_density(out_path)
                    metrics = score_density_frame(df_cached, levels)
                    summary_rows[model_name].append(
                        {"ticker": ticker, "horizon": horizon, **metrics}
                    )
                    continue
                pending.append((ticker, horizon))

        if not pending:
            logger.info(f"All runs already exist for {model_name}; skipping load.")
            continue

        logger.info(f"Loading {model_name} ({len(pending)} runs pending)")
        t_load = time.time()
        try:
            model = get_foundation_model(
                model_name,
                device=args.device,
                context_length=args.context_length,
            )
            model.load_model()
        except Exception as exc:
            logger.error(f"Failed to load {model_name}: {exc}")
            continue
        logger.info(f"  Loaded in {time.time() - t_load:.1f}s")

        first_done = False
        for run_idx, (ticker, horizon) in enumerate(pending):
            label = f"{model_name} | {ticker} | h={horizon}"
            logger.info(f"  Running {label}")
            t0 = time.time()
            try:
                rv = data.rv[ticker].dropna()
                if len(rv) < args.context_length + 10:
                    logger.warning(
                        f"    Skip: only {len(rv)} obs (need {args.context_length}+)"
                    )
                    continue
                df = zero_shot_density_forecast(
                    rv_series=rv,
                    model=model,
                    horizon=horizon,
                    context_length=args.context_length,
                    levels=levels,
                )
                out_path = _output_path(model_name, ticker, horizon, args.context_length)
                written = _write_density(df, out_path)
                metrics = score_density_frame(df, levels)
                summary_rows[model_name].append(
                    {"ticker": ticker, "horizon": horizon, **metrics}
                )
                elapsed = time.time() - t0
                logger.info(
                    f"    Done in {elapsed:.1f}s | n={metrics['n_obs']} | "
                    f"log CRPS={metrics['log_crps_mean']:.4f} | "
                    f"log KS={metrics['log_pit_ks_stat']:.3f} | "
                    f"cov80={metrics['log_coverage_80']:.2f} | "
                    f"written -> {written.name}"
                )
                if not first_done:
                    first_done = True
                    remaining = len(pending) - (run_idx + 1)
                    eta_h = elapsed * remaining / 3600
                    logger.info(
                        f"  >>> TIMING: first run {elapsed:.0f}s | "
                        f"{remaining} runs left | ETA ~{eta_h:.1f}h"
                    )
                    if eta_h > 23:
                        logger.warning(
                            f"  >>> WARNING: ETA {eta_h:.1f}h > 24h. Split the job."
                        )
            except NotImplementedError as exc:
                logger.error(f"    SKIP {label}: {exc}")
            except Exception as exc:
                logger.error(f"    FAILED {label}: {exc}")

        if summary_rows[model_name]:
            out_csv = _output_dir(model_name) / "_summary.csv"
            _output_dir(model_name).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summary_rows[model_name]).to_csv(out_csv, index=False)
            logger.info(f"Wrote summary: {out_csv}")

    # Cross-model headline summary
    logger.info("\n=== Cross-model summary (mean across assets) ===")
    for model_name in model_names:
        df = pd.DataFrame(summary_rows[model_name])
        if df.empty:
            continue
        for h in sorted(df["horizon"].unique()):
            sub = df[df["horizon"] == h]
            logger.info(
                f"  {model_name:>22s} h={int(h):>2d}  "
                f"log CRPS={sub['log_crps_mean'].mean():.4f}  "
                f"log KS={sub['log_pit_ks_stat'].mean():.3f}  "
                f"cov50/80/95={sub['log_coverage_50'].mean():.2f}/"
                f"{sub['log_coverage_80'].mean():.2f}/"
                f"{sub['log_coverage_95'].mean():.2f}"
            )

    logger.info("Density forecasts complete.")


if __name__ == "__main__":
    main()
