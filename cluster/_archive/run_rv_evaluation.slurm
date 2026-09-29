#!/bin/bash
#SBATCH --job-name=rv-evaluation
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rv_evaluation_%j.out
#SBATCH --error=logs/rv_evaluation_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

echo "=== CAPIRe evaluation ==="
python code/run_evaluation.py \
    --horizons 1 5 22 \
    --latex \
    --mcs-bootstrap 10000

echo "=== VOLARE evaluation ==="
python code/run_evaluation_volare.py \
    --horizons 1 5 22 \
    --latex \
    --mcs-bootstrap 10000

echo "Job finished: $(date)"
