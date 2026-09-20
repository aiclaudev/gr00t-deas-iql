#!/bin/bash
#SBATCH --job-name=deas-rc-summary
#SBATCH --account=sub
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --container=docker.io/library/python:3.11
#SBATCH --export=NONE
#SBATCH --chdir=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
set -euo pipefail
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
MANIFEST=${1:?Absolute evaluation manifest required}
OUTPUT_DIR=${2:?Absolute aggregate output directory required}
[[ $MANIFEST == /home/nas_main/dohyunlee/* && $OUTPUT_DIR == /home/nas_main/dohyunlee/* ]] || exit 2
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
# Pure stdlib: no simulator imports, credentials, GPU, or worker-side Slurm CLI.
exec /home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python \
    "$REPO_ROOT/scripts/robocasa/aggregate_results.py" --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR"
