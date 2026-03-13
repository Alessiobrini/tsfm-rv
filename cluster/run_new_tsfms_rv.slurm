#!/bin/bash
#SBATCH --job-name=new-tsfm-rv
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/new_tsfm_rv_%A_%a.out
#SBATCH --error=logs/new_tsfm_rv_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
source cluster/setup_models.sh
source cluster/setup_new_models.sh
cd "$SLURM_SUBMIT_DIR"

# Map array task -> ticker + asset class
STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)

ID=$SLURM_ARRAY_TASK_ID
if [ $ID -lt 40 ]; then
    TICKER=${STOCK_TICKERS[$ID]}
    CLASS=stocks
elif [ $ID -lt 45 ]; then
    TICKER=${FX_TICKERS[$((ID-40))]}
    CLASS=fx
else
    TICKER=${FUTURES_TICKERS[$((ID-45))]}
    CLASS=futures
fi

echo "Task $ID: $TICKER ($CLASS)"

python code/run_foundation_volare.py \
    --tickers $TICKER \
    --asset-class $CLASS \
    --horizons 1 5 22 \
    --models toto sundial moirai-moe-small \
    --device cuda \
    --skip-existing
