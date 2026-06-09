#!/bin/bash
#
# Pull the revised VOLARE results from Duke DCC down to this LOCAL repo.
# Run this ON THE MAC (not the cluster), after the SLURM jobs finish.
#
# Adopts the cluster-sync convention from the mixed-frequency-attention project:
# the `dcc` SSH alias (see ~/.ssh/config) uses ControlMaster multiplexing, so you
# authenticate ONCE per session (NetID password + Duo push) and every rsync below
# reuses that connection -- no repeated MFA. Open it once with `ssh dcc` in a real
# Terminal at the start of a work session.
#
# Usage:
#   bash code/pull_results.sh           # metrics + LaTeX tables (fast)
#   bash code/pull_results.sh full      # also ALL forecast CSVs (for figures / per-asset)
#   bash code/pull_results.sh logs      # SLURM rev_* logs + a remote state summary
#
# Override defaults via env vars, e.g.:
#   REMOTE=ab978@dcc-login.oit.duke.edu bash code/pull_results.sh full
#
set -euo pipefail

REMOTE="${REMOTE:-dcc}"
REMOTE_ROOT="${REMOTE_ROOT:-/hpc/group/darec/ab978/human-x-ai-finance}"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODE="${1:-tables}"   # tables (default) | full | logs

echo ">> Opening SSH connection to '$REMOTE' (reuses ControlMaster if live; else NetID + Duo push '1')"
ssh "$REMOTE" "echo connected: \$(hostname)" || {
  echo "SSH failed. Run in a real Terminal (not via Claude) so Duo works, and ensure the 'dcc'"
  echo "alias is in ~/.ssh/config (or pass REMOTE=ab978@dcc-login.oit.duke.edu)."; exit 1; }

# SAFETY: archive any existing local results/volare before pulling, so a re-pull
# never silently overwrites a prior copy. (Old IJF results already live under
# results/_archive/volare_ijf.) rsync would otherwise merge/overwrite by filename.
if [ -d "$LOCAL_ROOT/results/volare" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  mkdir -p "$LOCAL_ROOT/results/_archive"
  echo ">> Archiving existing local results/volare -> results/_archive/volare_local_$STAMP"
  mv "$LOCAL_ROOT/results/volare" "$LOCAL_ROOT/results/_archive/volare_local_$STAMP"
fi
mkdir -p "$LOCAL_ROOT/results/volare/metrics" "$LOCAL_ROOT/results/volare/tables"

echo ">> Pulling metrics + LaTeX tables for the revised run"
rsync -avz "$REMOTE:$REMOTE_ROOT/results/volare/metrics/" \
           "$LOCAL_ROOT/results/volare/metrics/" || echo "   (no metrics yet)"
rsync -avz "$REMOTE:$REMOTE_ROOT/results/volare/tables/" \
           "$LOCAL_ROOT/results/volare/tables/" || echo "   (no tables yet)"

if [ "$MODE" = "full" ]; then
  echo ">> Pulling all forecast CSVs (volatility scale)"
  mkdir -p "$LOCAL_ROOT/results/volare/forecasts"
  rsync -avz "$REMOTE:$REMOTE_ROOT/results/volare/forecasts/" \
             "$LOCAL_ROOT/results/volare/forecasts/"
fi

if [ "$MODE" = "logs" ]; then
  echo ">> Pulling SLURM rev_* logs + remote state summary"
  mkdir -p "$LOCAL_ROOT/logs"
  rsync -avz "$REMOTE:$REMOTE_ROOT/logs/rev_"* "$LOCAL_ROOT/logs/" 2>/dev/null || echo "   (no rev_ logs)"
  ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
    echo '--- forecast CSV count (expect up to 2550) ---'; ls results/volare/forecasts 2>/dev/null | wc -l; \
    echo '--- metrics ---'; ls results/volare/metrics 2>/dev/null; \
    echo '--- recent SLURM jobs (24h) ---'; sacct --starttime now-24hours --format=JobID,JobName%16,State,ExitCode,Elapsed 2>/dev/null | head -40 || echo 'sacct unavailable'"
fi

echo ">> Done. Local: results/volare/{metrics,tables$([ "$MODE" = full ] && echo ,forecasts)}"
