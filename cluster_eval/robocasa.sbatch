#!/usr/bin/env bash
#SBATCH --job-name=parallel-robocasa
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run with sbatch, not on login.' >&2; exit 2; }
# sbatch runs a copied script; pass the repository root explicitly.
ROOT=${1:?repository root}; shift
source "$ROOT/cluster_eval/env.sh"
source "$ROOT/local_eval/env.sh"
set -u
export CUDA_VISIBLE_DEVICES=0
exec "$DEAS_PYTHON" "$ROOT/local_eval/run_suite.py" --python-executable "$DEAS_PYTHON" --gpus 0 --jobs-per-gpu 1 --n-envs 8 --num-samples 10 "$@"
