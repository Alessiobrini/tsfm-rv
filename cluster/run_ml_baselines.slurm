#!/bin/bash
# =============================================================================
# run_ml_baselines.slurm — supervised ML baselines (XGBoost, LSTM)
# =============================================================================
# XGBoost on HAR-style inputs and an LSTM on 22 or 252 days of past log
# volatility, fitted on each asset's trailing 1,000 days and refitted every 22
# days (code/models/ml.py). One array task per ticker (0-39 stocks, 40-44 FX,
# 45-49 futures). Choose the models and the target at submission:
#   sbatch --export=ALL,MODELS="xgb-har lstm-22",TARGET=point,HORIZONS="1 5 22" cluster/run_ml_baselines.slurm
#   sbatch --export=ALL,MODELS="lstm-252",TARGET=avg,HORIZONS="5 22" cluster/run_ml_baselines.slurm
# Outputs: results/ml_point/forecasts/ or results/ml_avg/forecasts/
# =============================================================================
#SBATCH --job-name=ml-base
#SBATCH --partition=common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ml_base_%A_%a.out
#SBATCH --error=logs/ml_base_%A_%a.err

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
: "${MODELS:?set MODELS}" "${TARGET:?set TARGET}" "${HORIZONS:=1 5 22}"

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};          CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};      CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi
echo "Job started: $(date)  ${TICKER} (${CLASS}) on $(hostname)  models: ${MODELS}  target: ${TARGET}"

# shellcheck disable=SC2086
python code/run_ml_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" --models ${MODELS} \
    --horizons ${HORIZONS} --target-kind "${TARGET}" --results-dir "results/ml_${TARGET}" \
    --threads 4 --skip-existing

echo "Job finished: $(date)"
