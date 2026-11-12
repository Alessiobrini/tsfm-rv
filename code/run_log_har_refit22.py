"""Log-HAR refitted every 22 origins instead of every origin, for the table of the supervised ML
baselines.

The ML baselines are refitted every 22 origins. This run gives Log-HAR the same cadence, with
everything else as in the main run: the iterated forecast, the 1,000-day window and the clip to the
window bounds. It calls run_baselines_volare.py with reestimate_every_har_iterated set to 22 and
writes Log_HAR_{ticker}_h{h}.csv to results/log_har_refit22/forecasts/ unless --results-dir is given.
Every other option is passed on to run_baselines_volare.py.

Usage:  python code/run_log_har_refit22.py --tickers AAPL --asset-class stocks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

config.forecast_cfg.reestimate_every_har_iterated = 22
import run_baselines_volare  # noqa: E402

if __name__ == "__main__":
    if run_baselines_volare.forecast_cfg.reestimate_every_har_iterated != 22:
        raise RuntimeError("the refit cadence did not reach run_baselines_volare")
    args = sys.argv[1:]
    if "--results-dir" not in args:
        args += ["--results-dir", "results/log_har_refit22"]
    sys.argv = [sys.argv[0], "--models", "Log-HAR", "--horizons", "1", "5", "22", "--target-kind", "point",
                "--scale", "vol", "--skip-existing"] + args
    run_baselines_volare.main()
