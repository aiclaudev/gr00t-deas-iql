#!/bin/bash
#SBATCH --job-name=deas-rc-preview
#SBATCH --account=sub
#SBATCH --partition=compute
#SBATCH --qos=extra
#SBATCH --array=0-3%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
# Each array task is an independent short preview; at most one GPU at a time.
set -euo pipefail
[[ $# -eq 2 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-3]$ ]] || {
    echo 'Usage via sbatch: robocasa_preview.sbatch ACTOR OUTPUT_ROOT' >&2
    exit 2
}
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
USER_ROOT=/home/nas_main/dohyunlee
ACTOR=$(realpath -e -- "$1")
OUTPUT_ROOT=$(realpath -e -- "$2")
[[ $ACTOR == "$USER_ROOT"/* && $OUTPUT_ROOT == "$USER_ROOT"/* ]] || exit 2
[[ -f $ACTOR/config.json && -f $ACTOR/experiment_cfg/metadata.json ]] || exit 2
TASKS=(CoffeeSetupMug PnPMicrowaveToCounter TurnOffStove PnPCounterToMicrowave)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=$OUTPUT_ROOT/task-runs/$TASK
export TMPDIR=$USER_ROOT/.tmp/rc-preview-${SLURM_JOB_ID}
export HF_MODULES_CACHE=$TMPDIR/hf-modules
export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_LOG_MODEL=false WANDB_WATCH=false WANDB_SAVE_CODE=false
mkdir -p -- "$TMPDIR" "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$REPO_ROOT/scripts/robocasa/environment.sh"
exec python -u "$REPO_ROOT/scripts/robocasa/policy_smoke.py" \
    --actor "$ACTOR" --output "$OUTPUT" --task "$TASK" \
    --inference-steps 5 --video-fps 15 --seed 42
