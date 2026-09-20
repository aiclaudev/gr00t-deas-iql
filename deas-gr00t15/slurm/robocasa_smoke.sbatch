#!/bin/bash
#SBATCH --job-name=deas-rc-smoke
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm-logs/%x-%j.log
set -euo pipefail
PROJECT_ROOT="${DEAS_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the DEAS repository}}"
source "$PROJECT_ROOT/scripts/robocasa/environment.sh"
exec python -u scripts/robocasa/smoke.py \
    --env-name "${1:-CoffeeSetupMug}" --n-envs "${2:-1}" \
    --output "output/robocasa-smoke/${SLURM_JOB_ID}"
