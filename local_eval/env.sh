#!/usr/bin/env bash
# Source this before running any RoboCasa evaluation in this repo.
# It only exports variables; it never installs anything.

LOCAL_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEAS_ROOT="${DEAS_ROOT:-$(cd "${LOCAL_EVAL_DIR}/../deas-gr00t15" && pwd)}"
export DEAS_CONDA_ENV="${DEAS_CONDA_ENV:-deas-rc}"
export ROBOSUITE_DIR="${ROBOSUITE_DIR:-/home/junhyeong/workspace/robosuite}"
export ROBOCASA_DIR="${ROBOCASA_DIR:-/home/junhyeong/workspace/robocasa}"

# robosuite and robocasa are used from source, not pip-installed, so that the
# existing robocasa-eval environment keeps working untouched.
export PYTHONPATH="${DEAS_ROOT}:${ROBOSUITE_DIR}:${ROBOCASA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# RoboCasa 0.2.0 asserts NumPy 1.23.x; the vendored patch opts 1.26.4 in explicitly.
export ROBOCASA_ALLOW_NUMPY_126=1

# Offscreen rendering. MUJOCO_EGL_DEVICE_ID follows CUDA_VISIBLE_DEVICES so that
# simulator rendering and policy inference land on the same GPU.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
  else
    export MUJOCO_EGL_DEVICE_ID=0
  fi
fi

# Each AsyncVectorEnv worker is a separate process; keep BLAS single-threaded so
# n_envs workers do not oversubscribe the machine.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/deas-numba-${USER}}"
mkdir -p "${NUMBA_CACHE_DIR}"
