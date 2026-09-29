#!/bin/bash
# =============================================================================
# run_c6_direct_har.slurm — rerun of the direct HAR models (HAR-J, HAR-RS, HARQ)
# =============================================================================
# Reruns the three direct HAR models with the current engine: a rolling 1,000-row
# window with no look-ahead, re-estimated at every origin, the Bollerslev,
# Patton and Quaedvlieg (2016) insanity filter, and HAR-RS with only the daily
# lag split into semivariances. Both targets: point (main tables) and the
# h-day average (appendix). Writes to separate folders so the stored forecasts
# in results/volare and results/volare_avg are left untouched until the final
# evaluation is rerun. One CPU array task per ticker (0-39 stocks, 40-44 FX, 45-49 fut).
# Submit:  sbatch cluster/run_c6_direct_har.slurm
# Outputs: results/c6_direct_point/forecasts/, results/c6_direct_avg/forecasts/
# =============================================================================
#SBATCH --job-name=c6-direct-har
#SBATCH --partition=common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/c6_direct_har_%A_%a.out
#SBATCH --error=logs/c6_direct_har_%A_%a.err

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

for KIND in point avg; do
    python code/run_baselines_volare.py \
        --tickers "${TICKER}" \
        --asset-class "${CLASS}" \
        --horizons 1 5 22 \
        --models HAR-J HAR-RS HARQ \
        --target-kind "${KIND}" \
        --scale vol \
        --results-dir "results/c6_direct_${KIND}"
done

echo "Job finished: $(date)"
