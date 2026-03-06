#!/bin/bash
#SBATCH --job-name=portfolio-eval
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/portfolio_eval_%j.out
#SBATCH --error=logs/portfolio_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

for ASSET_CLASS in forex futures stocks; do
    echo "=== Portfolio eval: ${ASSET_CLASS} ==="
    python code/run_portfolio_eval.py \
        --asset-class ${ASSET_CLASS} \
        --horizons 1 5 22 \
        --tc-bps 10
done

echo "Job finished: $(date)"
