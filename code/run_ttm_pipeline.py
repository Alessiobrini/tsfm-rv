"""
run_ttm_pipeline.py — Run TTM forecasts + context sensitivity + regenerate all tables.

This script runs locally (CPU only, ~30 min total) and produces:
1. TTM forecasts at ctx=512 for all 50 assets, h=1,5,22
2. TTM context sensitivity at ctx=128, 256 for all 50 assets
3. Regenerated evaluation metrics, MCS, DM, tables

Usage:
    python run_ttm_pipeline.py
"""

import subprocess
import sys
import os

PYTHON = sys.executable
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)

def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
    else:
        print(f"  DONE")
    return result.returncode

def main():
    # Step 1: Run TTM forecasts at default context (512)
    for ac in ["stocks", "fx", "futures"]:
        run(
            f'"{PYTHON}" code/run_foundation_volare.py '
            f'--models ttm --all-tickers --asset-class {ac} '
            f'--horizons 1 5 22 --device cpu --skip-existing',
            f"TTM forecasts: {ac} (ctx=512)"
        )

    # Step 2: Context sensitivity (ctx=128, 256)
    for ctx in [128, 256]:
        for ac in ["stocks", "fx", "futures"]:
            run(
                f'"{PYTHON}" code/run_foundation_volare.py '
                f'--models ttm --all-tickers --asset-class {ac} '
                f'--horizons 1 5 22 --device cpu '
                f'--context-length {ctx} --skip-existing',
                f"TTM context sensitivity: {ac} (ctx={ctx})"
            )

    # Step 3: Regenerate evaluation (metrics, DM, MCS)
    run(
        f'"{PYTHON}" code/run_evaluation_volare.py --horizons 1 5 22',
        "Regenerate evaluation metrics + MCS + DM"
    )

    # Step 4: Regenerate robustness analyses
    run(
        f'"{PYTHON}" code/run_robustness.py --floor-sensitivity --mz-correction --window-comparison',
        "Regenerate robustness: floor sensitivity + MZ correction + window comparison"
    )

    # Step 5: Regenerate subsample metrics
    run(
        f'"{PYTHON}" code/compute_subsample_metrics.py',
        "Regenerate subsample metrics"
    )

    # Step 6: Regenerate context sensitivity table
    run(
        f'"{PYTHON}" code/compute_context_sensitivity.py',
        "Regenerate context sensitivity table"
    )

    # Step 7: Regenerate main LaTeX tables
    run(
        f'"{PYTHON}" code/process_results.py',
        "Regenerate LaTeX tables"
    )

    print("\n" + "="*60)
    print("  TTM PIPELINE COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy updated tables from results/volare/tables/ to paper/tables/")
    print("2. Update paper text with TTM results")
    print("3. Recompile PDF")


if __name__ == "__main__":
    main()
