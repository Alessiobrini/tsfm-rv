"""
run_ml_volare.py — supervised machine-learning baselines on VOLARE.

XGBoost on HAR-style inputs and an LSTM on 22 or 252 days of past log volatility, fitted on each
asset's trailing 1,000 days (models/ml.py). Each forecast is clipped to the window bounds used for
every model (forecasting/bounds.py). Files hold the origin date, the realized target and the
volatility forecast, as for the other models, with the log forecast, the variance term and the
tuned setting.

Usage:
    python run_ml_volare.py --tickers AAPL --asset-class stocks --models xgb-har lstm-22 \
        --horizons 1 5 22 --target-kind point --results-dir results/ml_point --threads 4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data
from forecasting.bounds import clip_to_window, BOUNDS_WINDOW
from models.ml import LEARNERS, rolling_ml
from utils import setup_logger


def main():
    ap = argparse.ArgumentParser(description="Supervised ML baselines on VOLARE")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--asset-class", default="stocks", choices=["stocks", "fx", "futures"])
    ap.add_argument("--models", nargs="+", default=list(LEARNERS), choices=list(LEARNERS))
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 22])
    ap.add_argument("--target-kind", default="point", choices=["point", "avg"])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--last-origins", type=int, default=None,
                    help="Smoke tests only: forecast only the last N origins.")
    a = ap.parse_args()

    logger = setup_logger("ml_volare")
    out_dir = Path(a.results_dir) / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {"stocks": "volare", "fx": "volare_fx", "futures": "volare_futures"}[a.asset_class]
    data = load_data(dataset=key, tickers=a.tickers)

    for t in a.tickers:
        vol = np.sqrt(data.rv[t].dropna())
        for name in a.models:
            for h in a.horizons:
                if a.target_kind == "avg" and h == 1:
                    continue
                path = out_dir / f"{name.replace('-', '_')}_{t}_h{h}.csv"
                if a.skip_existing and path.exists():
                    continue
                t0 = time.time()
                start = None
                if a.last_origins:
                    start = max(a.window, len(vol) - h + 1 - a.last_origins)
                df = rolling_ml(vol, LEARNERS[name](a.threads), h, a.target_kind, window=a.window,
                                start=start)
                df["forecast"] = clip_to_window(df["forecast"], vol, BOUNDS_WINDOW)
                df.to_csv(path)
                logger.info(f"{name} {t} h={h} {a.target_kind}: {len(df)} origins in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
