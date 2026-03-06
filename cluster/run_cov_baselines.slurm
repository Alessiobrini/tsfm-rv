#!/bin/bash
#SBATCH --job-name=cov-baselines
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cov_baselines_%j.out
#SBATCH --error=logs/cov_baselines_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

# Conda setup
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
echo "Running covariance baselines on forex (quick test)"

cd "$SLURM_SUBMIT_DIR"

python code/run_cov_baselines.py \
    --asset-class forex \
    --horizons 1 5 22 \
    --models element-har har-drd

echo "Job finished: $(date)"
