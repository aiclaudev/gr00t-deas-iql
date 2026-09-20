#!/bin/bash
#SBATCH --job-name=deas-after-preflight
#SBATCH --account=sub
#SBATCH --qos=own
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --container=docker.io/library/python:3.11
#SBATCH --export=NONE
#SBATCH --chdir=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T

# Submit with --dependency=afterok:<final-preflight-job> and --kill-on-invalid-dep=yes.
# This closed CPU batch verifies finished jobs and submits the authorized sweep once.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} || $# -lt 2 ]]; then
    echo "Use sbatch with --preflight-root PATH [--seeds 42,43,44 --target-steps 10000]." >&2
    exit 2
fi

REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
TRAIN_ENV=/home/nas_main/dohyunlee/miniconda3/envs/groot-train
cd "$REPO_ROOT"
set +u
source /home/nas_main/dohyunlee/miniconda3/etc/profile.d/conda.sh
conda activate "$TRAIN_ENV"
set -u

export PYTHONPATH=$REPO_ROOT
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=
export USE_TF=0
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export WANDB_ENTITY=aiclaudev
export WANDB_PROJECT='gr00t1.5 finetune'
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
export WANDB_SAVE_CODE=false
export TMPDIR=/home/nas_main/dohyunlee/.tmp/deas-${SLURM_JOB_ID}
[[ $(realpath -m -- "$TMPDIR") == /home/nas_main/dohyunlee/* ]] || exit 2
mkdir -p -- "$TMPDIR"
exec "$TRAIN_ENV/bin/python" "$REPO_ROOT/scripts/submit_deas_after_preflight.py" "$@"
