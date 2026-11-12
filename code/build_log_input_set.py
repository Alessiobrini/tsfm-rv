"""The forecast set with every foundation model fed log volatility, for the MCS and DM table of the
log versions.

Copies the files of the eight econometric models from the main point forecasts and replaces each
foundation model's file with its log-input forecast converted back to volatility as in
compute_log_input.py: the log forecast clipped to the log of the window bounds, exp(m + v / 2) with
v the variance of the model's past one-step log errors, then clipped to the window bounds. The
actual column is the volatility of the main files. run_evaluation_volare.py then runs the DM tests
and the MCS on this set, and gen_log_input_tables.py builds the table:

  python code/build_log_input_set.py
  python code/run_evaluation_volare.py --horizons 1 5 22 \\
      --forecast-dir results/volare_log_input/forecasts --metrics-dir results/volare_log_input/metrics

Usage:  python code/build_log_input_set.py [--forecasts DIR] [--log-forecasts DIR] [--out DIR]
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_log_input import LOG_FORECASTS, back_to_vol, clip_log, past_error_variance  # noqa: E402
from evaluation.forecast_sets import ALL, DIRECT, ECON, FORECASTS, ROOT, TSFM, load_sigma_all, read  # noqa: E402
from forecasting.bounds import clip_to_window  # noqa: E402

OUT = ROOT / "results" / "volare_log_input" / "forecasts"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--log-forecasts", type=Path, default=LOG_FORECASTS,
                    help="log-volatility forecasts of the nine foundation models")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists and is not overwritten")
    a.out.mkdir(parents=True)
    sigma = load_sigma_all()
    for t in ALL:
        sig = sigma[t]
        v1 = {m: past_error_variance(clip_log(read(a.log_forecasts / f"{m}_{t}_h1.csv"), sig)) for m in TSFM}
        for h in [1, 5, 22]:
            for m in ECON + DIRECT:
                shutil.copy2(a.forecasts / f"{m}_{t}_h{h}.csv", a.out / f"{m}_{t}_h{h}.csv")
            for m in TSFM:
                main_file = read(a.forecasts / f"{m}_{t}_h{h}.csv")
                log = clip_log(read(a.log_forecasts / f"{m}_{t}_h{h}.csv"), sig)
                f = clip_to_window(back_to_vol(log, v1[m]), sig)
                d = main_file.loc[main_file.index.intersection(f.index)].copy()
                if not np.allclose(np.exp(log.loc[d.index, "actual"]), d["actual"], rtol=1e-8, atol=0):
                    raise ValueError(f"{m} {t} h{h}: log actuals do not match the main file")
                d["forecast"] = f.loc[d.index].values
                d.index = d.index.strftime("%Y-%m-%d")
                d.index.name = "date"
                d[["actual", "forecast"]].to_csv(a.out / f"{m}_{t}_h{h}.csv")
        print("done", t, flush=True)


if __name__ == "__main__":
    main()
