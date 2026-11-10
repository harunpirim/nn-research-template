#!/bin/bash
# Slurm job for NDSU CCAST: trains one config (all its seeds and sweep variants).
#
# Submit from your laptop:  scripts/ccast/ccast.sh submit configs/<name>.yaml [extra args]
# Or on CCAST, from the project folder:
#     mkdir -p runs/_slurm && sbatch scripts/ccast/job.slurm configs/<name>.yaml [extra args]
# (Slurm opens the log file before the job starts, so runs/_slurm must already exist.)
#
# CHECK BEFORE FIRST USE (ask your instructor or support@ccast.ndsu.edu):
#   --account     your CCAST project account
#   --partition   the GPU queue name; `freenodes -a` lists nodes and GPU types
#   module/conda  how Python is provided (see the one-time setup in README.md)
#
#SBATCH --job-name=nn-train
#SBATCH --account=x-ccast-prj-CHANGE_ME
#SBATCH --partition=gpus
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/_slurm/%x-%j.out

set -euo pipefail

CONFIG=${1:?usage: sbatch scripts/ccast/job.slurm configs/<name>.yaml [extra args]}
shift

cd "$SLURM_SUBMIT_DIR"

# Python environment (created once; see README.md, "CCAST one-time setup").
# CHANGE_ME if your module or environment name differs.
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate nn

# Large datasets can live in scratch instead of the project folder:
# export DATA_DIR="$SCRATCH/$(basename "$PWD")/data"

echo "Job $SLURM_JOB_ID on $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-none} | config: $CONFIG $*"
python -m src.train --config "$CONFIG" "$@"
