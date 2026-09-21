#!/usr/bin/env bash
# Source before running LIBERO evaluation. Only exports; installs nothing.

LOCAL_LIBERO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${LOCAL_LIBERO_DIR}/.." && pwd)}"
export GR00T17_ROOT="${GR00T17_ROOT:-${REPO_ROOT}/gr00t17}"
export LIBERO_VENV="${LIBERO_VENV:-${HOME}/envs/libero_island}"

# Offscreen rendering; MUJOCO_EGL_DEVICE_ID follows CUDA_VISIBLE_DEVICES so the
# simulator renders on the same GPU the policy runs on.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
  else
    export MUJOCO_EGL_DEVICE_ID=0
  fi
fi

# Each vector-env worker is its own process; keep BLAS single-threaded so
# n_envs workers do not oversubscribe the machine.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# numba defaults to one thread per core (64 here). Left alone, every one of the
# n_envs worker processes would claim all of them.
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
# Share one JIT cache so workers do not each recompile on startup.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/libero-numba-${USER}}"
mkdir -p "${NUMBA_CACHE_DIR}"
export TOKENIZERS_PARALLELISM=false
