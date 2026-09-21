#!/usr/bin/env bash
# Evaluate the four LIBERO suites concurrently, all on one GPU.
#
# A single-suite run leaves the GPU idle while its simulators step: measured
# 44.5% average utilisation at n_envs=16, swinging between 0% and 74%, with CPU
# load at 14 of 64 cores. Running the suites side by side fills those gaps —
# one suite infers while another steps — without needing a second GPU.
#
# Each process loads its own copy of the policy (about 17 GB resident), so four
# fit in an 80 GB card.
#
#   MODEL=~/data/ckpt_... SEED=0 local_libero/run_suites_parallel.sh
set -uo pipefail

LOCAL_LIBERO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${LOCAL_LIBERO_DIR}/env.sh"

MODEL="${MODEL:?set MODEL to a prepared GR00T N1.7 checkpoint}"
SEED="${SEED:-0}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
GPU_DEVICE="${GPU_DEVICE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/local_outputs/libero_n17_fewshot1k}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export MUJOCO_EGL_DEVICE_ID="${GPU_DEVICE}"

ROOT="${OUTPUT_ROOT}/seed${SEED}"
mkdir -p "${ROOT}"

EXTRA=()
[[ "${SKIP_COMPLETED}" == "1" ]] && EXTRA+=(--skip-completed)

# shellcheck source=/dev/null
source "${LIBERO_VENV}/.venv/bin/activate"

echo "seed ${SEED} | GPU ${GPU_DEVICE} | n_envs=${N_ENVS} | ${N_EPISODES} ep/task"
echo "suites: ${SUITES}"
echo "output: ${ROOT}"
echo "started $(date -Is)"

PIDS=()
for SUITE in ${SUITES}; do
    LOG="${ROOT}/${SUITE}.log"
    python "${LOCAL_LIBERO_DIR}/run_suite_gr00t17.py" \
        --model-path "${MODEL}" \
        --output-root "${ROOT}" \
        --suites "${SUITE}" \
        --n-episodes "${N_EPISODES}" \
        --n-envs "${N_ENVS}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --seed "${SEED}" \
        "${EXTRA[@]}" > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "  ${SUITE} -> pid $! (${LOG})"
    # Stagger the starts so four checkpoint loads do not collide on disk and GPU.
    sleep 20
done

STATUS=0
for PID in "${PIDS[@]}"; do
    wait "${PID}" || STATUS=1
done
# Each process only knew its own suite, so its summary covered only that suite.
# Rebuild one that spans all four.
python "${LOCAL_LIBERO_DIR}/run_suite_gr00t17.py" \
    --output-root "${ROOT}" --suites ${SUITES} \
    --n-episodes "${N_EPISODES}" --n-envs "${N_ENVS}" \
    --n-action-steps "${N_ACTION_STEPS}" --seed "${SEED}" \
    --summarise-only 2>&1 | tail -20

echo "finished $(date -Is) (status ${STATUS})"
