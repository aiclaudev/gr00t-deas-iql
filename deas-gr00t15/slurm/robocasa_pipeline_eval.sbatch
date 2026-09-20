#!/bin/bash
#SBATCH --job-name=deas-rc-actor
#SBATCH --account=sub
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
# Actor-only direct inference: model actions are passed to RoboCasa env.step.
# Use scripts/robocasa/submit_evaluations.py to prepare paths and dependencies.
set -euo pipefail
[[ ( $# -eq 13 || $# -eq 14 ) && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || {
    echo 'Expected ACTOR TASK OUTPUT EPISODES N_ENVS EVAL_SEED TRAIN_SEED REPORT_TO RUN_NAME WANDB_GROUP ACTION_HORIZON DENOISING_STEPS EXPECTED_TRAINING_STEPS [SAVE_VIDEO=0|1] in an sbatch worker.' >&2
    exit 2
}
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
USER_ROOT=/home/nas_main/dohyunlee
ACTOR=$(realpath -e -- "$1")
TASK=$2
OUTPUT=$(realpath -m -- "$3")
EPISODES=$4
N_ENVS=$5
EVAL_SEED=$6
TRAINING_SEED=$7
REPORT_TO=$8
RUN_NAME=$9
WANDB_GROUP=${10}
ACTION_HORIZON=${11}
DENOISING_STEPS=${12}
EXPECTED_TRAINING_STEPS=${13}
SAVE_VIDEO=${14:-0}
case "$SAVE_VIDEO" in 0|1) ;; *) echo "SAVE_VIDEO must be 0 or 1" >&2; exit 2 ;; esac
VIDEO_ARGS=()
if [[ $SAVE_VIDEO == 1 ]]; then
    VIDEO_ARGS+=(--save_video)
fi
[[ $ACTOR == "$USER_ROOT"/* && $OUTPUT == "$USER_ROOT"/* ]] || exit 2
case "$TASK" in CoffeeSetupMug|PnPMicrowaveToCounter|TurnOffStove|PnPCounterToMicrowave) ;; *) exit 2 ;; esac
for value in "$EPISODES" "$N_ENVS" "$ACTION_HORIZON" "$DENOISING_STEPS"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || exit 2
done
[[ $EVAL_SEED =~ ^[0-9]+$ && $TRAINING_SEED =~ ^[0-9]+$ && $EXPECTED_TRAINING_STEPS =~ ^[0-9]+$ ]] || exit 2
case "$REPORT_TO" in none|wandb) ;; *) exit 2 ;; esac
[[ -f $ACTOR/config.json && -f $ACTOR/experiment_cfg/metadata.json ]] || {
    echo 'Missing actor config or RoboCasa normalization metadata after dependency completion.' >&2
    exit 2
}
# Keep caches and short Unix socket paths inside the user's home, including
# NUMBA_CACHE_DIR created by environment.sh. No package installation in workers.
export TMPDIR=$USER_ROOT/.tmp/rc-${SLURM_JOB_ID}
[[ $(realpath -m -- "$TMPDIR") == "$USER_ROOT"/* ]] || exit 2
mkdir -p -- "$TMPDIR" "$(dirname -- "$OUTPUT")"
mkdir -- "$OUTPUT"  # Existing results must never be overwritten.
export HF_MODULES_CACHE=$OUTPUT/.cache/hf-modules
export TRITON_CACHE_DIR=$OUTPUT/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$OUTPUT/.cache/torchinductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT='gr00t1.5 finetune'
export WANDB_ENTITY=aiclaudev
export WANDB_LOG_MODEL=false WANDB_WATCH=false WANDB_SAVE_CODE=false
export WANDB_DIR=$OUTPUT
if [[ $REPORT_TO == wandb ]]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=disabled
fi
mkdir -p -- "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$REPO_ROOT/scripts/robocasa/environment.sh"
python - "$ACTOR" "$EXPECTED_TRAINING_STEPS" <<'PY'
import json
from pathlib import Path
import sys
import torch

actor = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
if expected_steps:
    state = json.loads((actor / 'trainer_state.json').read_text())
    if state.get('global_step') != expected_steps:
        raise ValueError(f'Expected final actor step {expected_steps}, got {state.get("global_step")}')
if torch.cuda.device_count() != 1:
    raise RuntimeError('This evaluation requires exactly one allocated GPU')
metadata = json.loads((actor / 'experiment_cfg/metadata.json').read_text())
if 'new_embodiment' not in metadata:
    raise ValueError('Actor normalization metadata lacks new_embodiment')
index = actor / 'model.safetensors.index.json'
if index.is_file():
    names = set(json.loads(index.read_text())['weight_map'].values())
    if not names:
        raise ValueError('Empty actor shard index')
    for name in names:
        shard = (actor / name).resolve()
        if not shard.is_relative_to(actor) or not shard.is_file():
            raise ValueError(f'Invalid or missing actor shard: {name}')
elif not (actor / 'model.safetensors').is_file():
    raise FileNotFoundError('Actor safetensors weights are missing')
print('Actor evaluation preflight passed', flush=True)
PY
exec python -u "$REPO_ROOT/scripts/eval_policy_robocasa.py" \
    --actor_model_path "$ACTOR" --model_type gr00tn15 \
    --data_config single_panda_gripper_rl_inference --embodiment_tag new_embodiment \
    --action_horizon "$ACTION_HORIZON" --denoising_steps "$DENOISING_STEPS" \
    --env_name "$TASK" --num_episodes "$EPISODES" --n_envs "$N_ENVS" \
    --seed "$EVAL_SEED" --training_seed "$TRAINING_SEED" --noise 0.0 \
    --output_path "$OUTPUT" --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" --wandb_group "$WANDB_GROUP" "${VIDEO_ARGS[@]}"
