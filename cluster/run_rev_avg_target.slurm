#!/bin/bash
# =============================================================================
# run_rev_avg_target.slurm — REVISION: h-day-AVERAGE target robustness arm (B9)
# =============================================================================
# Re-runs the full forecasting pipeline under the legacy h-day-AVERAGE realized
# volatility target (Patton & Sheppard 2015) so the appendix can report both
# targets. Everything else matches the headline pipeline: volatility scale, mean
# point forecast, 1000-day window / context, iterated / direct multistep as in
# the point-target run. `--target-kind avg` routes all outputs to
# results/volare_avg/ (forecasts + metrics), so it never collides with the
# headline results/volare/ files.
#
# One array task per ticker (0-49: 0-39 stocks, 40-44 FX, 45-49 futures). Each
# task runs the eight econometric baselines (CPU) AND the nine TSFMs (GPU) for
# its ticker. --skip-existing makes it restartable. Budget 12h (TSFM samplers
# dominate); GPU partition because the foundation models need CUDA.
#
# Submit:
#     sbatch cluster/run_rev_avg_target.slurm
#
# AFTER it finishes and results/volare_avg/forecasts is populated, build the
# appendix tables locally (point the metrics + table step at volare_avg):
#     python code/run_evaluation_volare.py --results-subdir volare_avg --scale vol
#     # then a process_results variant pointed at results/volare_avg/metrics
#     # (see code/gen_pooled50.py / process_results.py for the table builders).
# =============================================================================
#SBATCH --job-name=rev-avg-target
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/rev_avg_%A_%a.out
#SBATCH --error=logs/rev_avg_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface
source cluster/setup_models.sh 2>/dev/null || true
source cluster/setup_new_models.sh 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};        CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};    CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi
echo "=== avg-target  $TICKER ($CLASS)  $(date) ==="

# Econometric baselines (CPU), averaged target
python code/run_baselines_volare.py \
    --tickers "${TICKER}" --asset-class "${CLASS}" --horizons 1 5 22 \
    --target-kind avg --scale vol --skip-existing

# Foundation models (GPU), averaged target
MODELS="chronos-bolt-small chronos-bolt-base timesfm-2.5 moirai-2.0-small moirai-moe-small lag-llama toto sundial ttm"
python code/run_foundation_volare.py \
    --tickers "${TICKER}" --asset-class "${CLASS}" --horizons 1 5 22 \
    --models ${MODELS} --device cuda \
    --target-kind avg --scale vol --skip-existing

echo "Job finished: $(date)"
