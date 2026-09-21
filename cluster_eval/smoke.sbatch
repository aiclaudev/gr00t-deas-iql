#!/usr/bin/env bash
#SBATCH --job-name=parallel-eval-smoke
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
ROOT=${1:?repo}; MODE=${2:?robocasa or libero}; OUTPUT=${3:?output}
source "$ROOT/cluster_eval/env.sh"
export CUDA_VISIBLE_DEVICES=0
case "$MODE" in
 robocasa)
  source "$ROOT/local_eval/env.sh"
  exec "$DEAS_PYTHON" "$DEAS_ROOT/scripts/robocasa/smoke.py" --n-envs 2 --output "$OUTPUT";;
 libero)
  source "$LIBERO_VENV/.venv/bin/activate"
  export PYTHONPATH="$GR00T17_ROOT:$LIBERO_REPO"
  exec python "$ROOT/cluster_eval/smoke_libero.py" --output "$OUTPUT";;
 *) exit 2;;
esac
