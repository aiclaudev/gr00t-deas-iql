#!/usr/bin/env bash
# Run one parallel RoboCasa evaluation of a deas-gr00t15 checkpoint.
#
# Parallelism comes from the repository's own vector environment:
# load_robocasa_gym_env builds an AsyncVectorEnv of N_ENVS spawned simulator
# processes, and gr00t/eval/rollout.py::evaluate_vector_policy steps them
# together, so every policy call is a single batched inference over all live
# environments.
#
#   ACTOR=/path/to/checkpoint ENV_NAME=CoffeeSetupMug N_ENVS=8 N_EPISODES=50 \
#     local_eval/run_eval.sh
#
# Best-of-N instead of plain BC: also set CRITIC and MODEL_TYPE=deas.
set -euo pipefail

LOCAL_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${LOCAL_EVAL_DIR}/env.sh"

ACTOR="${ACTOR:?set ACTOR to a GR00T checkpoint directory (see local_eval/fetch_weights.py)}"
CRITIC="${CRITIC:-}"
MODEL_TYPE="${MODEL_TYPE:-gr00tn15}"      # gr00tn15 = BC, deas = best-of-N with a critic
DEAS_BACKEND="${DEAS_BACKEND:-iql}"       # legacy | checkpoint | iql
ENV_NAME="${ENV_NAME:-CoffeeSetupMug}"
N_ENVS="${N_ENVS:-8}"
N_EPISODES="${N_EPISODES:-50}"
SEED="${SEED:-42}"
TRAINING_SEED="${TRAINING_SEED:-}"
GPU_DEVICE="${GPU_DEVICE:-0}"
DATA_CONFIG="${DATA_CONFIG:-single_panda_gripper_rl_inference}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SAVE_INFERENCE_INPUTS="${SAVE_INFERENCE_INPUTS:-0}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
OUTPUT_PATH="${OUTPUT_PATH:-${LOCAL_EVAL_DIR}/../local_outputs/robocasa_eval/$(date +%Y%m%d_%H%M%S)_${ENV_NAME}}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export MUJOCO_EGL_DEVICE_ID="${GPU_DEVICE}"

EXTRA=()
[[ -n "${CRITIC}" ]] && EXTRA+=(--critic_model_path "${CRITIC}" --deas_backend "${DEAS_BACKEND}")
[[ -n "${EXECUTE_HORIZON}" ]] && EXTRA+=(--execute_horizon "${EXECUTE_HORIZON}")
[[ -n "${TRAINING_SEED}" ]] && EXTRA+=(--training_seed "${TRAINING_SEED}")
[[ -n "${RUN_NAME}" ]] && EXTRA+=(--run_name "${RUN_NAME}")
[[ -n "${WANDB_GROUP}" ]] && EXTRA+=(--wandb_group "${WANDB_GROUP}")
[[ "${SAVE_VIDEO}" == "1" ]] && EXTRA+=(--save_video)
[[ "${SAVE_INFERENCE_INPUTS}" == "1" ]] && EXTRA+=(--save_inference_inputs)

mkdir -p "${OUTPUT_PATH}"
echo "Task ${ENV_NAME} | ${N_EPISODES} episodes | n_envs=${N_ENVS} | GPU ${GPU_DEVICE} | ${MODEL_TYPE}"
echo "Actor  ${ACTOR}"
[[ -n "${CRITIC}" ]] && echo "Critic ${CRITIC} (${DEAS_BACKEND}, N=${NUM_SAMPLES}, T=${TEMPERATURE})"
echo "Output ${OUTPUT_PATH}"

cd "${DEAS_ROOT}"
conda run --no-capture-output -n "${DEAS_CONDA_ENV}" python scripts/eval_policy_robocasa.py \
    --actor_model_path "${ACTOR}" \
    --model_type "${MODEL_TYPE}" \
    --env_name "${ENV_NAME}" \
    --num_episodes "${N_EPISODES}" \
    --n_envs "${N_ENVS}" \
    --seed "${SEED}" \
    --data_config "${DATA_CONFIG}" \
    --embodiment_tag "${EMBODIMENT_TAG}" \
    --action_horizon "${ACTION_HORIZON}" \
    --denoising_steps "${DENOISING_STEPS}" \
    --num_samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --output_path "${OUTPUT_PATH}" \
    --report_to "${REPORT_TO}" \
    "${EXTRA[@]}" \
    2>&1 | tee "${OUTPUT_PATH}/eval.log"
