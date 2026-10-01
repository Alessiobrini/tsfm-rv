#!/bin/bash
# =============================================================================
# run_pure_econometric.slurm — rerun of HAR, Log-HAR, ARFIMA, ARMA and MEM
# =============================================================================
# Reruns the five pure econometric models with an estimation window of exactly
# the 1,000 most recent observations, re-estimated at every origin. Both targets:
# point (main tables) and the h-day average (appendix). Writes to separate
# folders so the stored forecasts in results/volare and results/volare_avg are
# left untouched until the final evaluation is rerun.
# One CPU array task per (ticker, model, target): task ID = 10 * ticker + 2 * model + target,
# tickers 0-39 stocks, 40-44 FX, 45-49 futures.
# Submit:  sbatch cluster/run_pure_econometric.slurm
# Resubmit failed tasks with --array=<ids>; finished files are skipped.
# Outputs: results/pure_window_point/forecasts/, results/pure_window_avg/forecasts/
# =============================================================================
#SBATCH --job-name=pure-econ
#SBATCH --partition=common
#SBATCH --array=0-499%150
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=16:00:00
#SBATCH --output=logs/pure_econ_%A_%a.out
#SBATCH --error=logs/pure_econ_%A_%a.err

set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
MODELS=(HAR Log-HAR ARFIMA ARMA MEM)
KINDS=(point avg)

ID=$SLURM_ARRAY_TASK_ID
T=$((ID / 10)); M=$(((ID % 10) / 2)); K=$((ID % 2))
if   [ "$T" -lt 40 ]; then TICKER=${STOCK_TICKERS[$T]};          CLASS=stocks
elif [ "$T" -lt 45 ]; then TICKER=${FX_TICKERS[$((T-40))]};      CLASS=fx
else                       TICKER=${FUTURES_TICKERS[$((T-45))]}; CLASS=futures
fi
MODEL=${MODELS[$M]}; KIND=${KINDS[$K]}
echo "Job started: $(date)  task ${ID}: ${TICKER} (${CLASS}) ${MODEL} ${KIND}"

python code/run_baselines_volare.py \
    --tickers "${TICKER}" \
    --asset-class "${CLASS}" \
    --horizons 1 5 22 \
    --models "${MODEL}" \
    --target-kind "${KIND}" \
    --scale vol \
    --skip-existing \
    --results-dir "results/pure_window_${KIND}"

echo "Job finished: $(date)"
