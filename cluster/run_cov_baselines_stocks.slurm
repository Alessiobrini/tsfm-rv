#!/bin/bash
#SBATCH --job-name=cov-base-stocks
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cov_baselines_stocks_%j.out
#SBATCH --error=logs/cov_baselines_stocks_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
echo "Running covariance baselines on stocks (820 pairs)"

cd "$SLURM_SUBMIT_DIR"

python code/run_cov_baselines.py \
    --asset-class stocks \
    --horizons 1 5 22 \
    --models element-har har-drd

echo "Job finished: $(date)"
