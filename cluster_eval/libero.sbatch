#!/usr/bin/env bash
#SBATCH --job-name=parallel-libero17
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run with sbatch, not on login.' >&2; exit 2; }
ROOT=${1:?repository root}
source "$ROOT/cluster_eval/env.sh"
export MODEL=${2:?prepared checkpoint} OUTPUT_ROOT=${3:?output directory}
export SEED=${4:-42} N_EPISODES=${5:-50} N_ENVS=${6:-8}
export GPU_DEVICE=0
export SUITES='libero_spatial libero_object libero_goal libero_10'
exec bash "$ROOT/local_libero/run_suites_parallel.sh"
