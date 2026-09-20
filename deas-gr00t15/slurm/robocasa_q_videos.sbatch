#!/bin/bash
#SBATCH --job-name=bon50-q-videos
#SBATCH --account=sub
#SBATCH --qos=extra
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -euo pipefail
REPO=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
export TMPDIR=/home/nas_main/dohyunlee/.tmp/q-videos-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=''
source "$REPO/scripts/robocasa/environment.sh"
exec python -u "$REPO/scripts/robocasa/render_q_comparisons.py" \
 --results "$REPO/output/robocasa-comparison/seed42-bc2-vs-bon50-eval012-50ep-20260919/bon/results" \
 --output "$REPO/output/robocasa-comparison/seed42-bc2-vs-bon50-eval012-50ep-20260919/q-videos-eval0"
