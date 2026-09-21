#!/usr/bin/env bash
# Build the conda environment used for RoboCasa evaluation of deas-gr00t15.
#
# The environment is cloned from an existing GR00T training environment because
# that one already carries a compiled flash-attn wheel for this exact
# python/torch ABI; gr00t/model/backbone/eagle2_hg_model/radio_model.py imports
# flash_attn at module level, so it is not optional, and building it from source
# takes about an hour.
#
# Idempotent: re-running only repairs what is missing.
set -euo pipefail

SOURCE_ENV="${SOURCE_ENV:-gr00t-train}"
TARGET_ENV="${TARGET_ENV:-deas-rc}"
LOCAL_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEAS_ROOT="$(cd "${LOCAL_EVAL_DIR}/../deas-gr00t15" && pwd)"
ROBOCASA_DIR="${ROBOCASA_DIR:-/home/junhyeong/workspace/robocasa}"
ROBOSUITE_DIR="${ROBOSUITE_DIR:-/home/junhyeong/workspace/robosuite}"

for directory in "${ROBOCASA_DIR}" "${ROBOSUITE_DIR}"; do
    [[ -d "${directory}" ]] || { echo "Missing simulator source: ${directory}" >&2; exit 1; }
done

if ! conda env list | awk '{print $1}' | grep -qx "${TARGET_ENV}"; then
    echo "==> Cloning ${SOURCE_ENV} into ${TARGET_ENV}"
    conda create -y -n "${TARGET_ENV}" --clone "${SOURCE_ENV}"
else
    echo "==> Conda environment ${TARGET_ENV} already exists"
fi

run() { conda run --no-capture-output -n "${TARGET_ENV}" "$@"; }

# The cloned environment's gr00t points at a different checkout. Drop that link
# and register this repository's snapshot instead. --no-deps keeps the cloned,
# already-consistent dependency set intact.
echo "==> Pointing gr00t at ${DEAS_ROOT}"
run python -m pip uninstall -y gr00t >/dev/null 2>&1 || true
run python -m pip install -q --no-deps -e "${DEAS_ROOT}"

# Simulator runtime. These are the versions the existing robocasa-eval
# environment runs, except numba/llvmlite which the clone already provides at a
# NumPy 1.26-compatible version. pynput is deliberately left out: it only backs
# robosuite's teleoperation devices and demo scripts, which headless evaluation
# never imports, and the evdev extension it depends on does not compile against
# this kernel's input-event headers.
echo "==> Installing the simulator runtime"
run python -m pip install -q \
    mujoco==3.2.6 \
    mink==0.0.5 \
    qpsolvers==4.8.1 \
    quadprog==0.1.13 \
    pygame==2.6.1 \
    hidapi==0.14.0.post4 \
    glfw==2.10.0 \
    PyOpenGL==3.1.10 \
    imageio-ffmpeg==0.6.0 \
    lxml==5.3.0

# robomimic backs robocasa's state replay, which regenerates camera
# observations from recorded simulator states during data re-collection.
# --no-deps keeps its stale pins away from the environment above.
run python -m pip install -q --no-deps "robomimic==0.2.0"

# pip may resolve a different NumPy while installing the above; RoboCasa's
# version assert and this repository's patch both target exactly 1.26.4.
echo "==> Pinning numpy==1.26.4"
run python -m pip install -q "numpy==1.26.4"

# RoboCasa 0.2.0 hard-asserts NumPy 1.23.x. The patch adds an opt-in for 1.26.4
# behind ROBOCASA_ALLOW_NUMPY_126 and leaves the 1.23.x path untouched, so the
# existing robocasa-eval environment keeps working from the same source tree.
if grep -q ROBOCASA_ALLOW_NUMPY_126 "${ROBOCASA_DIR}/robocasa/__init__.py"; then
    echo "==> RoboCasa NumPy 1.26 patch already applied"
else
    echo "==> Applying the RoboCasa NumPy 1.26 patch to ${ROBOCASA_DIR}"
    patch -p1 -d "${ROBOCASA_DIR}" < "${DEAS_ROOT}/scripts/robocasa/robocasa-numpy126.patch"
fi

# The clone inherited an activation hook that prepends another GR00T checkout to
# PYTHONPATH, which would shadow this repository's gr00t package. Keep the CUDA
# library paths it sets and drop the PYTHONPATH line.
ENV_PREFIX="$(conda run -n "${TARGET_ENV}" python -c 'import sys; print(sys.prefix)')"
STALE_HOOK="${ENV_PREFIX}/etc/conda/activate.d/${SOURCE_ENV}.sh"
if [[ -f "${STALE_HOOK}" ]]; then
    echo "==> Removing the inherited PYTHONPATH from ${STALE_HOOK}"
    grep -v '^export PYTHONPATH=' "${STALE_HOOK}" \
        | grep -v '^export _GROOT_SMOKE_OLD_PYTHONPATH=' > "${STALE_HOOK}.new"
    mv "${STALE_HOOK}.new" "${STALE_HOOK}"
    STALE_UNDO="${ENV_PREFIX}/etc/conda/deactivate.d/${SOURCE_ENV}.sh"
    if [[ -f "${STALE_UNDO}" ]]; then
        grep -v 'PYTHONPATH' "${STALE_UNDO}" > "${STALE_UNDO}.new"
        mv "${STALE_UNDO}.new" "${STALE_UNDO}"
    fi
fi

echo "==> Verifying"
# shellcheck source=/dev/null
source "${LOCAL_EVAL_DIR}/env.sh"
DEAS_CONDA_ENV="${TARGET_ENV}" run python - <<'PY'
import numpy, torch, robosuite, robocasa, mujoco, gymnasium
import flash_attn
import gr00t
from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env  # noqa: F401
print(f"gr00t      {gr00t.__file__}")
print(f"robocasa   {robocasa.__version__} ({robocasa.__file__})")
print(f"robosuite  {robosuite.__version__}")
print(f"mujoco     {mujoco.__version__}")
print(f"gymnasium  {gymnasium.__version__}")
print(f"numpy      {numpy.__version__}")
print(f"torch      {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"flash_attn {flash_attn.__version__}")
PY

echo
echo "Ready. Evaluate with local_eval/run_eval.sh or local_eval/run_suite.py."
