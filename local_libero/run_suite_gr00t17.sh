#!/usr/bin/env bash
# Evaluate a GR00T N1.7 checkpoint on LIBERO inside the island environment.
#
#   MODEL=/path/to/checkpoint OUTPUT_ROOT=~/runs/libero-n17 local_libero/run_suite_gr00t17.sh
#
# Any extra arguments go straight to run_suite_gr00t17.py.
set -euo pipefail

LOCAL_LIBERO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${LOCAL_LIBERO_DIR}/env.sh"

MODEL="${MODEL:?set MODEL to a GR00T N1.7 checkpoint directory or Hub id}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/local_outputs/libero_n17/$(date +%Y%m%d_%H%M%S)}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-8}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
SEED="${SEED:-0}"
GPU_DEVICE="${GPU_DEVICE:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"

[[ -d "${LIBERO_VENV}/.venv" ]] || {
    echo "No LIBERO environment at ${LIBERO_VENV}; run local_libero/setup_env.sh" >&2
    exit 1
}

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export MUJOCO_EGL_DEVICE_ID="${GPU_DEVICE}"
mkdir -p "${OUTPUT_ROOT}"

EXTRA=()
[[ "${SAVE_VIDEO}" == "1" ]] && EXTRA+=(--save-video)

echo "Model   ${MODEL}"
echo "Suites  ${SUITES}"
echo "Output  ${OUTPUT_ROOT}"
echo "${N_EPISODES} episodes/task | n_envs=${N_ENVS} | GPU ${GPU_DEVICE}"

# shellcheck source=/dev/null
source "${LIBERO_VENV}/.venv/bin/activate"
exec python "${LOCAL_LIBERO_DIR}/run_suite_gr00t17.py" \
    --model-path "${MODEL}" \
    --output-root "${OUTPUT_ROOT}" \
    --suites ${SUITES} \
    --n-episodes "${N_EPISODES}" \
    --n-envs "${N_ENVS}" \
    --n-action-steps "${N_ACTION_STEPS}" \
    --seed "${SEED}" \
    "${EXTRA[@]}" "$@" \
    2>&1 | tee -a "${OUTPUT_ROOT}/eval.log"
