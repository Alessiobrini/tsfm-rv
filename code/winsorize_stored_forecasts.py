"""winsorize_stored_forecasts.py -- clip stored forecasts to the window bounds at each origin.

The runners clip every forecast to the range of the asset's volatility series over the
BOUNDS_WINDOW days before the forecast origin (forecasting/bounds.py). This script applies the same
rule to forecast files already on disk, for every model, so files written under an earlier rule
(bounds from the asset's full sample) can be brought to the current one without a rerun. The window
range lies inside the full-sample range, so clipping a file that was clipped to the full-sample
range gives the same values as clipping the raw forecasts.

A file is rewritten only if at least one forecast changes, with the layout that
save_single_forecast writes (index `date`, columns `actual`, `forecast`), and values are read with
round-trip float precision so unchanged rows are written back identically.

Usage (from the project root):
    python code/winsorize_stored_forecasts.py                    # dry run on results/volare*/forecasts
    python code/winsorize_stored_forecasts.py --apply            # rewrites the affected files
    python code/winsorize_stored_forecasts.py --folders DIR ...  # other forecast folders
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import VOLARE_ALL_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS  # noqa: E402
from data_loader import load_data  # noqa: E402
from forecasting.bounds import clip_to_window, BOUNDS_WINDOW  # noqa: E402

FOLDERS = [ROOT / "results" / "volare" / "forecasts", ROOT / "results" / "volare_avg" / "forecasts"]
PAT = re.compile(r"^(?P<model>.+)_(?P<ticker>[A-Z]+)_h(?P<h>\d+)(?P<ctx>_ctx\d+)?$")


def volatility_series():
    """Per-asset volatility series sqrt(RV), as the runners build it (data.rv[ticker].dropna())."""
    out = {}
    for key, tickers in [("volare", [t for t in VOLARE_ALL_TICKERS
                                     if t not in VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS]),
                         ("volare_fx", VOLARE_FX_TICKERS), ("volare_futures", VOLARE_FUTURES_TICKERS)]:
        data = load_data(dataset=key, tickers=tickers)
        for t in tickers:
            out[t] = np.sqrt(data.rv[t].dropna())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    ap.add_argument("--folders", nargs="+", type=Path, default=FOLDERS)
    args = ap.parse_args()

    sigma = volatility_series()
    rows = []
    for folder in args.folders:
        for f in sorted(Path(folder).glob("*.csv")):
            m = PAT.match(f.stem)
            if m is None or m["ticker"] not in sigma:
                continue
            tk = m["ticker"]
            d = pd.read_csv(f, index_col="date", parse_dates=True, float_precision="round_trip")
            clipped = clip_to_window(d["forecast"], sigma[tk], BOUNDS_WINDOW)
            n_up = int((clipped < d["forecast"]).sum())
            n_low = int((clipped > d["forecast"]).sum())
            if n_up + n_low == 0:
                continue
            rows.append(dict(folder=str(folder), model=m["model"] + (m["ctx"] or ""),
                             file=f.name, above_cap=n_up, below_floor=n_low))
            if args.apply:
                d["forecast"] = clipped
                d.index = d.index.strftime("%Y-%m-%d")
                d.index.name = "date"
                d.to_csv(f)
    r = pd.DataFrame(rows)
    if r.empty:
        print("No forecast outside the window bounds. Nothing to change.")
        return
    print(("APPLIED" if args.apply else "DRY RUN") + f": {len(r)} files with values outside the bounds")
    print(r.groupby(["folder", "model"])[["above_cap", "below_floor"]].sum().to_string())


if __name__ == "__main__":
    main()
