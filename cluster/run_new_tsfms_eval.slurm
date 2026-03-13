#!/bin/bash
#SBATCH --job-name=new-tsfm-eval
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/new_tsfm_eval_%j.out
#SBATCH --error=logs/new_tsfm_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
source cluster/setup_models.sh
source cluster/setup_new_models.sh
cd "$SLURM_SUBMIT_DIR"

NEW_MODELS="toto sundial moirai-moe-small"

# Stage 1: Covariance forecasting (forex + futures)
for MODEL in $NEW_MODELS; do
    for CLASS in forex futures; do
        echo "=== RCov: $MODEL on $CLASS ==="
        python code/run_cov_foundation.py \
            --asset-class $CLASS \
            --horizons 1 5 22 \
            --models $MODEL \
            --device cuda \
            --skip-existing
    done
done

# Stage 2: Portfolio evaluation (auto-discovers npz files)
for CLASS in forex futures; do
    echo "=== Portfolio eval: $CLASS ==="
    python code/run_portfolio_eval.py --asset-class $CLASS
done

# Stage 3: Full evaluation -- DM tests + MCS (auto-discovers CSVs)
echo "=== Evaluation: DM + MCS ==="
python code/run_evaluation_volare.py --horizons 1 5 22 --latex

# Stage 4: Advanced evaluation -- MZ regressions + GR fluctuation
echo "=== Advanced eval: MZ + GR ==="
python code/run_advanced_evaluation.py --dataset volare --horizons 1 5 22 --gr-plot

# Stage 5: Regenerate all LaTeX tables
echo "=== Regenerating tables ==="
python code/process_results.py
