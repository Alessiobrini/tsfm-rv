#!/bin/bash
# =============================================================================
# run_rev_lagsundial_fix.slurm — rerun ONLY lag-llama + sundial for the avg and
# context-sensitivity arms, after the dependency fix.
# =============================================================================
# lag-llama and sundial failed in the first avg/ctx submissions because
# setup_new_models.sh's unpinned `pip install` had pulled pandas 3.0.3 /
# transformers 4.57.6 into the shared env: gluonts 0.14.4 (lag-llama) needs
# pandas <2.2 and sundial's HF remote code needs transformers with
# DynamicCache.seen_tokens (<~4.46). The env has been pinned back to
# pandas==2.1.4 + transformers==4.44.2 (the set that ran all nine models in the
# original run). This job therefore does NOT source setup_new_models.sh, so it
# cannot re-upgrade those libraries. --skip-existing protects the 15/7 models
# already produced; only lag-llama and sundial are computed here.
#
# Submit:  sbatch cluster/run_rev_lagsundial_fix.slurm
# =============================================================================
#SBATCH --job-name=rev-lagsun-fix
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/rev_lagsun_%A_%a.out
#SBATCH --error=logs/rev_lagsun_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -uo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface
# NOTE: deliberately NOT sourcing cluster/setup_*_models.sh — the env is already
# pinned to pandas==2.1.4 / transformers==4.44.2 and must not be upgraded.
cd "$SLURM_SUBMIT_DIR"

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};        CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};    CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi
MODELS="lag-llama sundial"
echo "=== lag/sundial fix  $TICKER ($CLASS)  $(date) ==="

# Averaged-target arm (B9)
python code/run_foundation_volare.py \
    --tickers "${TICKER}" --asset-class "${CLASS}" --horizons 1 5 22 \
    --models ${MODELS} --device cuda \
    --target-kind avg --scale vol --skip-existing

# Context-length sweep (B6)
for CTX in 128 256 512; do
  echo "--- ctx=$CTX  $(date) ---"
  python code/run_foundation_volare.py \
      --tickers "${TICKER}" --asset-class "${CLASS}" --horizons 1 5 22 \
      --models ${MODELS} --device cuda \
      --context-length ${CTX} --target-kind point --scale vol --skip-existing
done
echo "Job finished: $(date)"
