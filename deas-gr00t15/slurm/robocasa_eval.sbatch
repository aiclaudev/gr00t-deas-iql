#!/bin/bash
#SBATCH --job-name=deas-rc-eval
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm-logs/%x-%j.log
set -euo pipefail
PROJECT_ROOT="${DEAS_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the DEAS repository}}"
source "$PROJECT_ROOT/scripts/robocasa/environment.sh"
# Usage: sbatch ... ACTOR TASK [CRITIC] [EPISODES] [N_ENVS] [SEED]
ACTOR=$(realpath -e "${1:?Actor checkpoint directory required}")
TASK="${2:-CoffeeSetupMug}"
CRITIC="${3:-}"
EPISODES="${4:-50}"
N_ENVS="${5:-1}"
SEED="${6:-42}"
[[ -f "$ACTOR/config.json" && -f "$ACTOR/experiment_cfg/metadata.json" ]] || {
    echo 'Actor must contain config.json and experiment_cfg/metadata.json from RoboCasa fine-tuning.' >&2; exit 2;
}
MODEL_ARGS=(--model_type gr00tn15)
if [[ -n "$CRITIC" ]]; then
    CRITIC=$(realpath -e "$CRITIC")
    [[ -f "$CRITIC/config.json" ]] || { echo 'Missing critic config.json' >&2; exit 2; }
    MODEL_ARGS=(--model_type deas --critic_model_path "$CRITIC" --num_samples "${NUM_SAMPLES:-10}" --temperature "${TEMPERATURE:-0.0}")
fi
OUTPUT="output/robocasa-eval/${TASK}/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT"
exec python -u scripts/eval_policy_robocasa.py \
    --actor_model_path "$ACTOR" "${MODEL_ARGS[@]}" \
    --data_config single_panda_gripper_rl_inference --embodiment_tag new_embodiment \
    --action_horizon "${ACTION_HORIZON:-16}" --env_name "$TASK" \
    --num_episodes "$EPISODES" --n_envs "$N_ENVS" --seed "$SEED" \
    --noise 0.0 --output_path "$OUTPUT"
