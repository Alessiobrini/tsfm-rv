#!/bin/bash
# =============================================================================
# run_rev_avg_target_daily.slurm — averaged-target arm, DAILY re-estimation
# =============================================================================
# Re-fit the eight econometric models under the h-day-AVERAGE target with daily
# re-estimation (reestimate_every=1). --target-kind avg routes all outputs to
# results/volare_avg/ so it never collides with the point-in-time run. TSFM
# averaged-target forecasts are unchanged (zero-shot) and are NOT re-run.
# No --skip-existing: overwrites the prior every-22-day averaged-target forecasts.
#
# Submit:  sbatch cluster/run_rev_avg_target_daily.slurm
# After:   python code/run_evaluation_volare.py --results-subdir volare_avg --scale vol
# =============================================================================
#SBATCH --job-name=avg-daily
#SBATCH --partition=common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rev_avg_daily_%A_%a.out
#SBATCH --error=logs/rev_avg_daily_%A_%a.err
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

echo "Ticker: ${TICKER}  (${CLASS})  [averaged target, daily]"

python code/run_baselines_volare.py \
    --tickers "${TICKER}" \
    --asset-class "${CLASS}" \
    --horizons 1 5 22 \
    --target-kind avg \
    --scale vol

echo "Job finished: $(date)"
