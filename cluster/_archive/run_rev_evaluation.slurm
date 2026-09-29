#!/bin/bash
# =============================================================================
# run_rev_evaluation.slurm — REVISION pipeline, forecast evaluation (CPU, single)
# =============================================================================
# Post-IJF revision. Reads every forecast CSV in results/volare/forecasts/ and
# computes, per (ticker, horizon): MSE/MAE/QLIKE/R2_OOS, pairwise Diebold-Mariano
# tests, and the Model Confidence Set; writes metric CSVs + LaTeX tables.
#
#   --scale vol : stored forecasts are on the volatility scale, so QLIKE squares
#                 them back to variance internally (Patton-robust). MSE/MAE stay
#                 on the volatility scale. (Forecasts are pre-floored at the
#                 min-RV-in-window, so no extra floor is applied here.)
#
# MUST run AFTER the baseline + TSFM forecast jobs finish. Single CPU job; the
# MCS block-bootstrap (10,000 reps over many asset/horizon groups) is the cost,
# hence the generous wall time and memory.
#
# Outputs: results/volare/metrics/{metrics_by_asset_h*, aggregate_metrics,
#          mcs_results_h*, mcs_all_results, dm_pvalues_*}.csv
#          results/volare/tables/metrics_h*.tex
#
# Submit after the forecast jobs (capture their job IDs from sbatch):
#     FAST=$(sbatch --parsable cluster/run_rev_tsfm_fast.slurm)
#     HEAVY=$(sbatch --parsable cluster/run_rev_tsfm_heavy.slurm)
#     BASE=$(sbatch --parsable cluster/run_rev_baselines.slurm)
#     sbatch --dependency=afterok:${FAST}:${HEAVY}:${BASE} cluster/run_rev_evaluation.slurm
# Or simply submit by hand once you see all forecast jobs COMPLETED.
# =============================================================================
#SBATCH --job-name=rev-eval
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rev_eval_%j.out
#SBATCH --error=logs/rev_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"

echo "Job started: $(date)"

python code/run_evaluation_volare.py \
    --horizons 1 5 22 \
    --scale vol \
    --latex

echo "Job finished: $(date)"
