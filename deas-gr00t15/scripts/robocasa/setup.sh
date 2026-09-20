#!/bin/bash
# Run on login. Reuse the existing personal GR00T dependencies without modifying them.
set -euo pipefail
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo 'Run package installation on login.' >&2; exit 2; }
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
BASE_PYTHON="${BASE_PYTHON:-/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python}"
VENV="${DEAS_VENV:-$PROJECT_ROOT/.venv-robocasa}"
"$BASE_PYTHON" - <<'PY'
import importlib.metadata as m
import sys
assert sys.version_info[:2] == (3, 11), 'This environment was validated with Python 3.11'
for name, version in {'numpy': '1.26.4', 'torch': '2.8.0+cu128', 'gymnasium': '1.0.0',
                      'transformers': '4.51.3', 'diffusers': '0.30.2'}.items():
    assert m.version(name) == version, (name, m.version(name), version)
PY
if [[ ! -e "$VENV/pyvenv.cfg" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi
# Legacy simulator metadata pins conflict with the DEAS preprocessing stack.
# Install the explicitly tested simulator runtime, then local sources without resolution.
"$VENV/bin/python" -m pip install \
    mujoco==3.2.6 qpsolvers==4.13.0 quadprog==0.1.13 pygame==2.6.1 \
    pynput==1.8.2 hidapi==0.15.0 lxml==6.1.3 glfw==2.10.2 PyOpenGL==3.1.10
"$VENV/bin/python" -m pip install --no-deps -e . -e ../bench/robosuite -e ../bench/robocasa
if ! rg -q ROBOCASA_ALLOW_NUMPY_126 ../bench/robocasa/robocasa/__init__.py; then
    git -C ../bench/robocasa apply --check "$PROJECT_ROOT/scripts/robocasa/robocasa-numpy126.patch"
    git -C ../bench/robocasa apply "$PROJECT_ROOT/scripts/robocasa/robocasa-numpy126.patch"
fi
mkdir -p slurm-logs
printf 'Ready: source %s/scripts/robocasa/environment.sh\n' "$PROJECT_ROOT"
