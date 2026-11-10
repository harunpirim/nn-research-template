#!/usr/bin/env bash
# Run CCAST jobs from your laptop (macOS, Linux, or WSL on Windows) over SSH.
# Requires the "ccast" SSH host alias described in README.md.
#
#   scripts/ccast/ccast.sh push                          copy code, configs and data to CCAST
#   scripts/ccast/ccast.sh submit configs/x.yaml [args]  push, then submit a training job
#   scripts/ccast/ccast.sh status                        list your queued/running jobs
#   scripts/ccast/ccast.sh log                           show the end of the newest job log
#   scripts/ccast/ccast.sh pull                          copy runs/ back to your laptop
#
# Settings (environment variables):
#   CCAST_HOST  SSH alias or user@host       (default: ccast)
#   CCAST_DIR   project folder on CCAST, relative to your home there
#               (default: projects/<this folder's name>)
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root
HOST=${CCAST_HOST:-ccast}
REMOTE_DIR=${CCAST_DIR:-projects/$(basename "$PWD")}

push() {
  ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
  # runs/ is never pushed: results only flow back from CCAST, so nothing gets overwritten.
  rsync -az \
    --exclude '.git/' --exclude 'runs/' --exclude '.venv/' --exclude '__pycache__/' \
    --exclude '.ipynb_checkpoints/' --exclude '.DS_Store' \
    ./ "$HOST:$REMOTE_DIR/"
}

case "${1:-}" in
  push)
    push ;;
  submit)
    shift
    [[ $# -ge 1 ]] || { echo "usage: $0 submit configs/<name>.yaml [extra args]"; exit 1; }
    push
    ssh "$HOST" "cd '$REMOTE_DIR' && mkdir -p runs/_slurm && sbatch scripts/ccast/job.slurm $(printf '%q ' "$@")" ;;
  status)
    ssh "$HOST" 'squeue -u "$USER"' ;;
  log)
    ssh "$HOST" "cd '$REMOTE_DIR' && f=\$(ls -t runs/_slurm/*.out 2>/dev/null | head -1) && [ -n \"\$f\" ] && echo \"== \$f\" && tail -n 40 \"\$f\" || echo 'No job logs yet.'" ;;
  pull)
    mkdir -p runs
    rsync -az --exclude '_smoke/' "$HOST:$REMOTE_DIR/runs/" runs/ ;;
  *)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
