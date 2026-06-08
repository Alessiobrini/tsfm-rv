"""run_local.py — local pipeline orchestrator (replaces the SLURM scripts).

Runs the VOLARE pipeline stages sequentially on the local machine: econometric
baselines on CPU, TSFMs on MPS (Apple Silicon), then evaluation and table/figure
generation. Each stage shells out to the existing run scripts, so the run is
restartable (--skip-existing, on by default) and individual stages / single
models can be run in isolation. The old cluster/*.slurm files are kept only for
reference.

Examples
--------
    # Full econometric + TSFM + evaluation sweep (main: point target, vol scale):
    python run_local.py
    # Just one slow TSFM, all asset classes, resuming:
    python run_local.py --stages foundation --models sundial
    # The h-day-average appendix arm:
    python run_local.py --target-kind avg
"""

import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable
CODE = Path(__file__).resolve().parent
ASSET_CLASSES = ["stocks", "fx", "futures"]
TSFMS = [
    "chronos-bolt-small", "chronos-bolt-base", "timesfm-2.5", "moirai-2.0-small",
    "lag-llama", "toto", "sundial", "moirai-moe-small", "ttm",
]


def run(script_args):
    cmd = [PY] + script_args
    print("\n>>>", " ".join(script_args), flush=True)
    rc = subprocess.run(cmd, cwd=str(CODE)).returncode
    if rc != 0:
        print(f"   (stage exited with code {rc} — continuing)", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description="Local VOLARE pipeline orchestrator")
    ap.add_argument("--stages", nargs="+",
                    default=["baselines", "foundation", "evaluation"],
                    choices=["baselines", "foundation", "evaluation", "tables", "figures"])
    ap.add_argument("--target-kind", default=None, choices=["point", "avg"])
    ap.add_argument("--scale", default=None, choices=["vol", "var"])
    ap.add_argument("--device", default=None, help="TSFM device (default: auto = MPS)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="TSFM subset (default: all 9)")
    ap.add_argument("--asset-classes", nargs="+", default=ASSET_CLASSES,
                    choices=ASSET_CLASSES)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Recompute even if output CSVs exist")
    args = ap.parse_args()

    skip = [] if args.no_skip_existing else ["--skip-existing"]
    tk = ["--target-kind", args.target_kind] if args.target_kind else []
    sc = ["--scale", args.scale] if args.scale else []
    dev = ["--device", args.device] if args.device else []

    if "baselines" in args.stages:
        for ac in args.asset_classes:
            run(["run_baselines_volare.py", "--all-tickers", "--asset-class", ac] + skip + tk + sc)

    if "foundation" in args.stages:
        models = args.models or TSFMS
        # Loop model-outer so each model loads once and sweeps all assets.
        for m in models:
            for ac in args.asset_classes:
                run(["run_foundation_volare.py", "--all-tickers", "--asset-class", ac,
                     "--models", m] + skip + tk + sc + dev)

    if "evaluation" in args.stages:
        run(["run_evaluation_volare.py", "--latex"] + sc)

    if "tables" in args.stages:
        run(["process_results.py"])

    if "figures" in args.stages:
        run(["generate_figures.py"])

    print("\nrun_local complete.")


if __name__ == "__main__":
    main()
