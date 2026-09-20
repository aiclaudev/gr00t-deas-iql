#!/bin/bash
#SBATCH --job-name=deas-rc-bon
#SBATCH --account=sub
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -euo pipefail
[[ $# -eq 12 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
USER_ROOT=/home/nas_main/dohyunlee
ACTOR=$(realpath -e -- "$1")
CRITIC=$(realpath -e -- "$2")
TASK=$3
OUTPUT=$(realpath -m -- "$4")
EPISODES=$5
EVAL_SEED=$6
GROUP=$7
CODE_ROOT=$(realpath -e -- "$8")
EXECUTE_HORIZON=$9
NUM_SAMPLES=${10:-10}
REPORT_TO=${11:-wandb}
EXPECTED_FEATURE_PASSES=${12}
[[ $EXPECTED_FEATURE_PASSES == 1 || $EXPECTED_FEATURE_PASSES == 2 ]] || exit 2
case "$REPORT_TO" in none|wandb) ;; *) exit 2 ;; esac
[[ $NUM_SAMPLES =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $EXECUTE_HORIZON =~ ^([1-9]|1[0-6])$ ]] || exit 2
[[ $CODE_ROOT == "$USER_ROOT"/* ]] || exit 2
[[ $ACTOR == "$USER_ROOT"/* && $CRITIC == "$USER_ROOT"/* && $OUTPUT == "$USER_ROOT"/* ]] || exit 2
[[ $EPISODES =~ ^[1-9][0-9]*$ && $EVAL_SEED =~ ^[0-9]+$ ]] || exit 2
case "$TASK" in CoffeeSetupMug|PnPMicrowaveToCounter|TurnOffStove|PnPCounterToMicrowave) ;; *) exit 2 ;; esac
export TMPDIR=$USER_ROOT/.tmp/rc-bon-${SLURM_JOB_ID}
mkdir -p -- "$TMPDIR" "$(dirname -- "$OUTPUT")"
mkdir -- "$OUTPUT"
export HF_MODULES_CACHE=$OUTPUT/.cache/hf-modules
export TRITON_CACHE_DIR=$OUTPUT/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$OUTPUT/.cache/torchinductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT='gr00t1.5 finetune' WANDB_ENTITY=aiclaudev
if [[ $REPORT_TO == wandb ]]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=disabled
fi
export WANDB_LOG_MODEL=false WANDB_WATCH=false WANDB_SAVE_CODE=false
export WANDB_DIR=$OUTPUT
mkdir -p -- "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$REPO_ROOT/scripts/robocasa/environment.sh"
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
python - "$ACTOR" "$CRITIC" "$EXPECTED_FEATURE_PASSES" <<'PY'
import json,sys
from pathlib import Path
import torch
if torch.cuda.device_count()!=1:
    raise RuntimeError('Expected exactly one allocated GPU')
for name in sys.argv[1:3]:
    root=Path(name)
    state=json.loads((root/'trainer_state.json').read_text())
    if state['global_step']!=10000:
        raise ValueError('Expected a completed 10000-step checkpoint')
    metadata=json.loads((root/'experiment_cfg/metadata.json').read_text())
    if 'new_embodiment' not in metadata:
        raise ValueError('Missing normalization metadata')
config=json.loads((Path(sys.argv[2])/'config.json').read_text())
actual=config['critic_cfg'].get('online_q_feature_passes',2)
if actual != int(sys.argv[3]):
    raise ValueError(f'Critic feature path mismatch: expected {sys.argv[3]}, found {actual}')
print('Actor and critic final-checkpoint and feature-path checks passed',flush=True)
PY
python -u "$CODE_ROOT/scripts/eval_policy_robocasa.py" \
    --actor_model_path "$ACTOR" --critic_model_path "$CRITIC" \
    --model_type deas --deas_backend checkpoint \
    --data_config single_panda_gripper_rl_inference --embodiment_tag new_embodiment \
    --action_horizon 16 --execute_horizon "$EXECUTE_HORIZON" --denoising_steps 4 --num_samples "$NUM_SAMPLES" --temperature 0 \
    --env_name "$TASK" --num_episodes "$EPISODES" --n_envs 1 \
    --seed "$EVAL_SEED" --training_seed 42 --noise 0.0 \
    --output_path "$OUTPUT" --report_to "$REPORT_TO" --save_video --save_inference_inputs \
    --run_name "deas-bon${NUM_SAMPLES}-train42-eval${EVAL_SEED}-${TASK}-exec${EXECUTE_HORIZON}" --wandb_group "$GROUP"

# One-episode validation also checks actual input/RNG replay on this allocated GPU.
if [[ $EPISODES == 1 ]]; then
    python -u "$CODE_ROOT/scripts/robocasa/replay_inference.py" \
        --trace "$OUTPUT/inference/call-000000.npz" --recompute \
        --report "$OUTPUT/replay-verification.json"
fi
