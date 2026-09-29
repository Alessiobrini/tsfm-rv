"""winsorize_stored_forecasts.py -- clip stored TSFM forecasts to the winsorization bounds.

The TSFM runner (run_foundation_volare.py) clips every forecast to the asset's realized-volatility
support [sqrt(min RV), sqrt(max RV)] over the full VOLARE series. Runs made on the cluster before
that cap reached it (headline Toto and Moirai-MoE files, all context-sensitivity *_ctx* files, and
the averaged-target TSFM files) were written without the upper cap. This script applies the same
bounds to the stored files after the fact:

  * on the headline files in results/volare/forecasts it reproduces the manual clip of 2026-06-09
    (it should find nothing left to change);
  * on the *_ctx* files and on results/volare_avg/forecasts it applies the cap they never received.

A file is rewritten only if at least one forecast changes, with the same layout that
save_single_forecast writes (index `date`, columns `actual`, `forecast`), and values are read with
round-trip float precision so unchanged rows are written back identically.

Usage (from the project root):
    python code/winsorize_stored_forecasts.py            # dry run, prints what would change
    python code/winsorize_stored_forecasts.py --apply    # rewrites the affected files
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import VOLARE_STOCKS_FILE, VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE  # noqa: E402

TSFM = ["chronos_bolt_small", "chronos_bolt_base", "timesfm_2_5", "moirai_2_0_small",
        "lag_llama", "toto", "sundial", "moirai_moe_small", "ttm"]
FOLDERS = [ROOT / "results" / "volare" / "forecasts", ROOT / "results" / "volare_avg" / "forecasts"]
PAT = re.compile(r"^(?P<model>.+)_(?P<ticker>[A-Z]+)_h(?P<h>\d+)(?P<ctx>_ctx\d+)?$")


def bounds():
    """Per-asset [sqrt(min RV), sqrt(max RV)] over the full series, as the runners compute them."""
    rv = pd.concat([pd.read_csv(f, usecols=["date", "symbol", "rv5"])
                    for f in (VOLARE_STOCKS_FILE, VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE)]).dropna()
    rv = rv[rv.rv5 > 0]
    return np.sqrt(rv.groupby("symbol").rv5.min()), np.sqrt(rv.groupby("symbol").rv5.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    args = ap.parse_args()

    floor, cap = bounds()
    rows = []
    for folder in FOLDERS:
        for f in sorted(folder.glob("*.csv")):
            m = PAT.match(f.stem)
            if m is None or m["model"] not in TSFM:
                continue
            tk = m["ticker"]
            d = pd.read_csv(f, index_col="date", float_precision="round_trip")
            clipped = d["forecast"].clip(lower=floor[tk], upper=cap[tk])
            n_up = int((d["forecast"] > cap[tk]).sum())
            n_low = int((d["forecast"] < floor[tk]).sum())
            if n_up + n_low == 0:
                continue
            rows.append(dict(folder=folder.parent.name, model=m["model"] + (m["ctx"] or ""),
                             file=f.name, above_cap=n_up, below_floor=n_low))
            if args.apply:
                d["forecast"] = clipped
                d.to_csv(f)
    r = pd.DataFrame(rows)
    if r.empty:
        print("No forecast outside the bounds. Nothing to change.")
        return
    print(("APPLIED" if args.apply else "DRY RUN") + f": {len(r)} files with values outside the bounds")
    print(r.groupby(["folder", "model"])[["above_cap", "below_floor"]].sum().to_string())


if __name__ == "__main__":
    main()
