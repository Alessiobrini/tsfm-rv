#!/bin/bash
# =============================================================================
# run_rev_context_sensitivity.slurm — REVISION: TSFM context-length sensitivity
# =============================================================================
# Re-runs the nine TSFMs at context lengths 128, 256, and 512 days (the default
# headline run uses 1000) so the appendix context-sensitivity table can be
# refreshed on the revised pipeline (point target, vol scale, mean point forecast).
#
# Forecasts save with a _ctx<N> filename suffix (run_foundation_volare appends it
# whenever the context differs from the 1000-day default), so they do NOT collide
# with the main results. After this finishes, build the table locally with
# code/gen_context_sensitivity.py (TODO: aggregates the _ctx* CSVs into
# paper/tables/table_context_sensitivity.tex).
#
# One array task per ticker (0-49: 0-39 stocks, 40-44 FX, 45-49 futures); each
# task sweeps all nine models at the three context lengths. --skip-existing makes
# it restartable. Expect this to be the heaviest job (9 models x 3 contexts x 3
# horizons per ticker); budget 24h.
#
# Submit:  sbatch cluster/run_rev_context_sensitivity.slurm
# =============================================================================
#SBATCH --job-name=rev-ctx-sens
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/rev_ctx_sens_%A_%a.out
#SBATCH --error=logs/rev_ctx_sens_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface
# Best-effort setup; `|| true` suspends `set -e` through the sourced scripts so a
# non-zero import check (e.g. the timesfm version probe) cannot abort the job
# before the timesfm-2.0 upgrade inside setup_new_models.sh runs.
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

MODELS="chronos-bolt-small chronos-bolt-base timesfm-2.5 moirai-2.0-small moirai-moe-small lag-llama toto sundial ttm"
for CTX in 128 256 512; do
  echo "=== $TICKER ($CLASS) ctx=$CTX  $(date) ==="
  python code/run_foundation_volare.py \
      --tickers "${TICKER}" --asset-class "${CLASS}" --horizons 1 5 22 \
      --models ${MODELS} --device cuda \
      --context-length ${CTX} --target-kind point --scale vol --skip-existing
done
echo "Job finished: $(date)"
