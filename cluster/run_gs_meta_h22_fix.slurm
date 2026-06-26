#!/bin/bash
# Finish the two tasks that hit the 6h limit in run_rev_baselines_daily.slurm:
# only ARMA and MEM at h=22 for GS and META remained (daily ARMA BIC search).
#SBATCH --job-name=gsmeta-h22
#SBATCH --partition=common
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gsmeta_h22_%A_%a.out
#SBATCH --error=logs/gsmeta_h22_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"

TICKERS=(GS META)
T=${TICKERS[$SLURM_ARRAY_TASK_ID]}
echo "Job started: $(date) — fixing ${T} h=22 (ARMA, MEM)"

python code/run_baselines_volare.py \
    --tickers "${T}" \
    --asset-class stocks \
    --horizons 22 \
    --models ARMA MEM \
    --target-kind point \
    --scale vol

echo "Job finished: $(date)"
