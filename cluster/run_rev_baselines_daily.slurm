#!/bin/bash
# =============================================================================
# run_rev_baselines_daily.slurm — econometric baselines, DAILY re-estimation
# =============================================================================
# Re-fit all eight econometric models for every VOLARE asset with the rolling
# 1000-day window re-estimated EVERY day (reestimate_every=1 for the iterated
# HAR/Log-HAR and the series ARFIMA/ARMA/MEM; HAR-J/RS/Q were already daily).
# Conforms to standard HAR practice. TSFMs are NOT re-run (zero-shot, unchanged).
#
# No --skip-existing: this OVERWRITES the prior every-22-day econometric
# forecasts. One CPU array task per ticker (0-39 stocks, 40-44 FX, 45-49 fut).
# Daily cadence ~ 20-40 min/ticker; 6h is ample.
#
# Submit:  sbatch cluster/run_rev_baselines_daily.slurm
# Outputs: results/volare/forecasts/<model>_<ticker>_h<h>.csv  (volatility scale)
# =============================================================================
#SBATCH --job-name=rev-base-daily
#SBATCH --partition=common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rev_baselines_daily_%A_%a.out
#SBATCH --error=logs/rev_baselines_daily_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"

echo "Job started:  $(date)"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)

ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};          CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};      CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi

echo "Ticker: ${TICKER}  (${CLASS})"

python code/run_baselines_volare.py \
    --tickers "${TICKER}" \
    --asset-class "${CLASS}" \
    --horizons 1 5 22 \
    --target-kind point \
    --scale vol

echo "Job finished: $(date)"
