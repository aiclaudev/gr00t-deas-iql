#!/bin/bash
#SBATCH --job-name=svf-rc-actor
#SBATCH --account=sub
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
# One task evaluation, after the matching SVF training stage succeeds.
set -euo pipefail
[[ $# -eq 16 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || {
    echo 'Expected CHECKPOINT EXPECTED_STEP plus the 14 RoboCasa evaluation arguments in an sbatch worker.' >&2
    exit 2
}
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
CHECKPOINT=$1
EXPECTED_STEP=$2
shift 2
[[ $EXPECTED_STEP =~ ^[1-9][0-9]*$ && ${13} == 0 ]] || {
    echo 'SVF EXPECTED_STEP must be positive; legacy EXPECTED_TRAINING_STEPS must be 0.' >&2
    exit 2
}
# Pure stdlib only: verify atomic completion before the evaluator imports CUDA.
export PYTHONDONTWRITEBYTECODE=1
/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python \
    "$REPO_ROOT/scripts/robocasa/validate_svf_checkpoint.py" \
    --checkpoint "$CHECKPOINT" --expected-step "$EXPECTED_STEP" --actor "$1"
exec bash "$REPO_ROOT/slurm/robocasa_pipeline_eval.sbatch" "$@"
