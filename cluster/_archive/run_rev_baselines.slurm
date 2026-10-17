#!/bin/bash
# =============================================================================
# run_rev_baselines.slurm — REVISION pipeline, econometric baselines (CPU array)
# =============================================================================
# Post-IJF revision. Econometric forecasts for all 50 VOLARE assets at
# horizons h = 1, 5, 22. (These also run locally on a laptop; this script is for
# a clean cluster reproduction alongside the TSFM jobs.)
#
# Eight models (run_baselines.AVAILABLE_MODELS), each in its defensible
# multi-step mode (see code/run_baselines.py):
#     HAR        — level HAR, Nelson-Cao non-negativity-constrained OLS, ITERATED
#     Log-HAR    — HAR on log-RV (positive by construction), ITERATED
#     HAR-J      — jump component,        DIRECT (auxiliary regressor)
#     HAR-RS     — semivariances,         DIRECT (auxiliary regressors)
#     HARQ       — realized quarticity,   DIRECT (auxiliary regressor)
#     ARFIMA     — local-Whittle d + IC (p,q), ITERATED
#     ARMA       — ARMA on log-RV, IC-selected,  ITERATED
#     MEM        — Engle (2002), positive by construction, ITERATED
#
# Revised settings (defaults; explicit for auditability):
#     --target-kind point   point-in-time RV_{t+h}
#     --scale vol           forecast volatility = sqrt(RV)
#   1000-day training window (config), min-RV floor in code.
#
# One array task per ticker. 50 tasks: 0-39 stocks, 40-44 FX, 45-49 futures.
# CPU only (no GPU). ARMA/ARFIMA do a small (p,q) grid search and the augmented
# HAR variants re-estimate every step, so budget up to ~30 min/ticker; 8h is
# generous headroom. --skip-existing makes the job restartable.
#
# Outputs: results/volare/forecasts/<model>_<ticker>_h<h>.csv  (volatility scale)
#
# Submit:
#     sbatch cluster/run_rev_baselines.slurm
# =============================================================================
#SBATCH --job-name=rev-baselines
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rev_baselines_%A_%a.out
#SBATCH --error=logs/rev_baselines_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
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
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};        CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};    CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi

echo "Ticker: ${TICKER}  (${CLASS})"

python code/run_baselines_volare.py \
    --tickers "${TICKER}" \
    --asset-class "${CLASS}" \
    --horizons 1 5 22 \
    --target-kind point \
    --scale vol \
    --skip-existing

echo "Job finished: $(date)"
