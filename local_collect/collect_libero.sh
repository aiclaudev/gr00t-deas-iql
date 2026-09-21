#!/usr/bin/env bash
# Re-collect training data from a real LIBERO evaluation run.
#
#   MODEL=/path/to/n17 SUITES=libero_spatial N_EPISODES=20 local_collect/collect_libero.sh
#
# Two stages, both resumable:
#   1. Collect  - run the policy in parallel LIBERO envs; each worker streams its
#                 episodes to shards, camera views straight into mp4.
#   2. Assemble - renumber the shards into GR00T LeRobot v2.1 datasets, one per
#                 suite. Videos are moved, never re-encoded.
set -euo pipefail

LOCAL_COLLECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../local_libero/env.sh
source "${LOCAL_COLLECT_DIR}/../local_libero/env.sh"

MODEL="${MODEL:?set MODEL to a GR00T N1.7 checkpoint directory or Hub id}"
SUITES="${SUITES:-libero_spatial}"
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-8}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
GPU_DEVICE="${GPU_DEVICE:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
GROUP_BY="${GROUP_BY:-suite}"
MOVE_SHARDS="${MOVE_SHARDS:-1}"
STAGE="${STAGE:-all}"          # all | collect | assemble
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/local_outputs/recollect_libero/${RUN_NAME}}"

[[ -d "${LIBERO_VENV}/.venv" ]] || {
    echo "No LIBERO environment at ${LIBERO_VENV}; run local_libero/setup_env.sh" >&2
    exit 1
}

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export MUJOCO_EGL_DEVICE_ID="${GPU_DEVICE}"
mkdir -p "${OUTPUT_ROOT}"

echo "Model   ${MODEL}"
echo "Suites  ${SUITES}"
echo "Output  ${OUTPUT_ROOT}"
echo "${N_EPISODES} episodes/task | n_envs=${N_ENVS} | GPU ${GPU_DEVICE} | stage ${STAGE}"

# shellcheck source=/dev/null
source "${LIBERO_VENV}/.venv/bin/activate"

run_collect() {
    echo "==> [1/2] Collecting rollouts"
    local extra=()
    [[ "${SAVE_VIDEO}" == "1" ]] && extra+=(--save-video)
    python "${LOCAL_COLLECT_DIR}/collect_libero.py" \
        --model-path "${MODEL}" \
        --output-root "${OUTPUT_ROOT}" \
        --suites ${SUITES} \
        --n-episodes "${N_EPISODES}" \
        --n-envs "${N_ENVS}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --seed "${SEED}" \
        --fps "${FPS}" \
        "${extra[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/collect.log"
}

run_assemble() {
    echo "==> [2/2] Assembling shards into LeRobot datasets"
    [[ -d "${OUTPUT_ROOT}/shards" ]] || {
        echo "No shards under ${OUTPUT_ROOT}; run the collect stage first" >&2; exit 1; }
    local extra=()
    [[ "${MOVE_SHARDS}" == "1" ]] && extra+=(--move)
    python "${LOCAL_COLLECT_DIR}/libero_shards_to_lerobot.py" \
        --shards "${OUTPUT_ROOT}/shards" \
        --output-root "${OUTPUT_ROOT}/dataset" \
        --group-by "${GROUP_BY}" \
        --overwrite "${extra[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/assemble.log"
}

case "${STAGE}" in
    all)      run_collect; run_assemble ;;
    collect)  run_collect ;;
    assemble) run_assemble ;;
    *)        echo "Unknown STAGE ${STAGE}; use all, collect or assemble" >&2; exit 2 ;;
esac

echo
echo "Dataset  ${OUTPUT_ROOT}/dataset"
