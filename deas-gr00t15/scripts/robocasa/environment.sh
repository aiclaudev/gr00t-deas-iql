#!/bin/bash
# Source from login or a batch worker. Never installs packages on a worker.
DEAS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "${DEAS_VENV:-$DEAS_ROOT/.venv-robocasa}/bin/activate"
export PYTHONPATH="$DEAS_ROOT:$DEAS_ROOT/../bench/robosuite:$DEAS_ROOT/../bench/robocasa${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export ROBOCASA_ALLOW_NUMPY_126=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/deas-numba-${SLURM_JOB_ID:-$USER}"
mkdir -p "$NUMBA_CACHE_DIR"
cd "$DEAS_ROOT"
