#!/bin/bash
#SBATCH --job-name=rv-tsfm-vol-small
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/rv_tsfm_vol_small_%j.out
#SBATCH --error=logs/rv_tsfm_vol_small_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

# Ensure Lag-Llama is available
source cluster/setup_models.sh

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

echo "Running VOLARE foundation models for FX (5 tickers)"
python code/run_foundation_volare.py \
    --all-tickers \
    --asset-class fx \
    --horizons 1 5 22 \
    --models chronos-bolt-small chronos-bolt-base moirai-2.0-small lag-llama \
    --device cuda \
    --skip-existing

echo "Running VOLARE foundation models for futures (5 tickers)"
python code/run_foundation_volare.py \
    --all-tickers \
    --asset-class futures \
    --horizons 1 5 22 \
    --models chronos-bolt-small chronos-bolt-base moirai-2.0-small lag-llama \
    --device cuda \
    --skip-existing

echo "Job finished: $(date)"
