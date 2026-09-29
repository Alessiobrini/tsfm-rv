#!/bin/bash
#SBATCH --job-name=rv-base-vol-small
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rv_base_vol_small_%j.out
#SBATCH --error=logs/rv_base_vol_small_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

echo "Running VOLARE baselines for FX (5 tickers)"
python code/run_baselines_volare.py \
    --all-tickers \
    --asset-class fx \
    --horizons 1 5 22 \
    --skip-existing

echo "Running VOLARE baselines for futures (5 tickers)"
python code/run_baselines_volare.py \
    --all-tickers \
    --asset-class futures \
    --horizons 1 5 22 \
    --skip-existing

echo "Job finished: $(date)"
