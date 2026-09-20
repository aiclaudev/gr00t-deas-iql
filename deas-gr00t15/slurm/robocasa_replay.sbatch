#!/bin/bash
#SBATCH --job-name=deas-rc-replay
#SBATCH --account=sub
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -euo pipefail
[[ $# -eq 3 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
USER_ROOT=/home/nas_main/dohyunlee
RUN_ROOT=$(realpath -e -- "$1")
TRACE=$(realpath -e -- "$2")
REPORT=$(realpath -m -- "$3")
[[ $RUN_ROOT == "$USER_ROOT"/* && $TRACE == "$RUN_ROOT"/results/* && $REPORT == "$USER_ROOT"/* ]] || exit 2
CODE_ROOT=$RUN_ROOT/source-snapshot/repo
[[ -f $CODE_ROOT/scripts/robocasa/replay_inference.py ]] || exit 2
export TMPDIR=$USER_ROOT/.tmp/rc-replay-${SLURM_JOB_ID}
mkdir -p -- "$TMPDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE=$TMPDIR/hf-modules TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
source "$REPO_ROOT/scripts/robocasa/environment.sh"
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
exec python -u "$CODE_ROOT/scripts/robocasa/replay_inference.py" --trace "$TRACE" --report "$REPORT" --recompute
