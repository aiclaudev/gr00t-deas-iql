#!/usr/bin/env bash
# Re-collect training data from a real RoboCasa evaluation run.
#
# Two stages, each resumable on its own:
#
#   1. Evaluate with data collection and video recording on. The eval writes
#      per-episode videos, and DataCollectionWrapper records simulator states,
#      actions, rewards and dones into demo.hdf5. Images are not stored here.
#   2. Replay those states to regenerate the camera views and stream them
#      straight into per-camera mp4s, producing a GR00T LeRobot v2.1 dataset
#      that loads with single_panda_gripper_rl, so the rollouts can join the
#      human demos in DEAS critic / IQL training.
#
# Images are only ever stored as video. robocasa's own dataset_states_to_obs.py
# would write the replayed frames back into HDF5 as raw uint8 first, which at
# 128x128x3 across three cameras is roughly 44 MB per 300-step episode before
# they are re-encoded anyway.
#
#   ACTOR=/path/to/ckpt ENV_NAME=CoffeeSetupMug N_EPISODES=50 N_ENVS=8 \
#     local_collect/collect_robocasa.sh
#
# Successful and failed episodes are both kept; IQL needs the failures.
set -euo pipefail

LOCAL_COLLECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../local_eval/env.sh
source "${LOCAL_COLLECT_DIR}/../local_eval/env.sh"

ACTOR="${ACTOR:?set ACTOR to a GR00T checkpoint directory}"
CRITIC="${CRITIC:-}"
MODEL_TYPE="${MODEL_TYPE:-gr00tn15}"
DEAS_BACKEND="${DEAS_BACKEND:-iql}"
ENV_NAME="${ENV_NAME:-CoffeeSetupMug}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-8}"
SEED="${SEED:-42}"
GPU_DEVICE="${GPU_DEVICE:-0}"
DATA_CONFIG="${DATA_CONFIG:-single_panda_gripper_rl_inference}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CAMERA_SIZE="${CAMERA_SIZE:-128}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SAVE_TRACE="${SAVE_TRACE:-0}"
STAGE="${STAGE:-all}"          # all | eval | replay
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_${ENV_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT:-${LOCAL_COLLECT_DIR}/..}/local_outputs/recollect/${RUN_NAME}}"

EVAL_DIR="${OUTPUT_ROOT}/eval"
RAW_DIR="${OUTPUT_ROOT}/raw"
RAW_HDF5="${RAW_DIR}/demo.hdf5"
DATASET_ROOT="${OUTPUT_ROOT}/dataset"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export MUJOCO_EGL_DEVICE_ID="${GPU_DEVICE}"
mkdir -p "${EVAL_DIR}" "${RAW_DIR}" "${DATASET_ROOT}"

echo "Task    ${ENV_NAME}"
echo "Actor   ${ACTOR}"
echo "Output  ${OUTPUT_ROOT}"
echo "${N_EPISODES} episodes | n_envs=${N_ENVS} | GPU ${GPU_DEVICE} | stage ${STAGE}"

run_eval() {
    echo "==> [1/2] Evaluating with collection and video"
    local extra=()
    [[ -n "${CRITIC}" ]] && extra+=(--critic_model_path "${CRITIC}" --deas_backend "${DEAS_BACKEND}"
                                    --num_samples "${NUM_SAMPLES}" --temperature "${TEMPERATURE}")
    [[ "${SAVE_VIDEO}" == "1" ]] && extra+=(--save_video)
    [[ "${SAVE_TRACE}" == "1" ]] && extra+=(--save_inference_inputs)
    cd "${DEAS_ROOT}"
    conda run --no-capture-output -n "${DEAS_CONDA_ENV}" python scripts/eval_policy_robocasa.py \
        --actor_model_path "${ACTOR}" \
        --model_type "${MODEL_TYPE}" \
        --env_name "${ENV_NAME}" \
        --num_episodes "${N_EPISODES}" \
        --n_envs "${N_ENVS}" \
        --seed "${SEED}" \
        --data_config "${DATA_CONFIG}" \
        --action_horizon "${ACTION_HORIZON}" \
        --denoising_steps "${DENOISING_STEPS}" \
        --output_path "${EVAL_DIR}" \
        --collect_data \
        --data_collection_path "${RAW_DIR}" \
        --report_to none \
        "${extra[@]}" 2>&1 | tee "${OUTPUT_ROOT}/eval.log"
}

run_replay() {
    echo "==> [2/2] Replaying states straight into a LeRobot dataset"
    [[ -f "${RAW_HDF5}" ]] || { echo "Missing ${RAW_HDF5}; run the eval stage first" >&2; exit 1; }
    conda run --no-capture-output -n "${DEAS_CONDA_ENV}" python \
        "${LOCAL_COLLECT_DIR}/robocasa_replay_to_lerobot.py" \
        --hdf5 "${RAW_HDF5}" \
        --output-root "${DATASET_ROOT}" \
        --camera-size "${CAMERA_SIZE}" \
        --overwrite 2>&1 | tee "${OUTPUT_ROOT}/replay.log"
}

case "${STAGE}" in
    all)    run_eval; run_replay ;;
    eval)   run_eval ;;
    replay) run_replay ;;
    *)      echo "Unknown STAGE ${STAGE}; use all, eval or replay" >&2; exit 2 ;;
esac

echo
echo "Videos   ${EVAL_DIR}/videos"
echo "Dataset  ${DATASET_ROOT}/${ENV_NAME}"
