#!/usr/bin/env bash
# Build the LIBERO "island" environment for gr00t17 (N1.7) evaluation.
#
# LIBERO pins robosuite 1.4.0, which cannot coexist with the RoboCasa side's
# robosuite 1.5.2, so LIBERO gets its own interpreter. gr00t17 is exposed into
# it through a .pth file rather than being installed, so the island supplies
# gr00t's runtime dependencies itself and nothing is re-resolved.
#
# This follows gr00t17/gr00t/eval/sim/LIBERO/setup_libero.sh, with two changes:
# the LIBERO checkout already on this machine is used instead of a git
# submodule, and the virtualenv lives outside the vendored snapshot.
set -euo pipefail

LOCAL_LIBERO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LOCAL_LIBERO_DIR}/.." && pwd)"
GR00T17_ROOT="${GR00T17_ROOT:-${REPO_ROOT}/gr00t17}"
LIBERO_REPO="${LIBERO_REPO:-/home/junhyeong/workspace/Isaac-GR00T/external_dependencies/LIBERO}"
VENV="${LIBERO_VENV:-${HOME}/envs/libero_island}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

[[ -d "${LIBERO_REPO}" ]] || { echo "No LIBERO checkout at ${LIBERO_REPO}" >&2; exit 1; }
[[ -d "${GR00T17_ROOT}/gr00t" ]] || { echo "No gr00t17 at ${GR00T17_ROOT}" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

echo "==> Creating ${VENV} (python ${PYTHON_VERSION})"
mkdir -p "${VENV}"
if [[ ! -x "${VENV}/.venv/bin/python" ]]; then
    uv venv "${VENV}/.venv" --python "${PYTHON_VERSION}"
fi
# shellcheck source=/dev/null
source "${VENV}/.venv/bin/activate"

# LIBERO's requirements are pinned for python 3.7-era wheels; lift the ones that
# have no 3.12 build. robosuite stays at LIBERO's 1.4.0 pin on purpose.
echo "==> Patching LIBERO requirements for python ${PYTHON_VERSION}"
PATCHED="${VENV}/requirements-py${PYTHON_VERSION}.txt"
LIBERO_REPO="${LIBERO_REPO}" PATCHED="${PATCHED}" python - <<'PY'
import os
from pathlib import Path

replacements = {
    "torch": "torch==2.9.0",
    "torchvision": "torchvision==0.24.0",
    "hydra-core": "hydra-core==1.3.2",
    "numpy": "numpy==1.26.4",
    "transformers": "transformers==4.57.3",
    "opencv-python": "opencv-python==4.10.0.84",
    "matplotlib": "matplotlib==3.9.4",
    "wandb": "wandb==0.18.7",  # 0.13.1 pulls pathtools, which uses the removed `imp`
}
source = Path(os.environ["LIBERO_REPO"]) / "requirements.txt"
lines = []
for raw in source.read_text().splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw)
        continue
    name = stripped.split("==", 1)[0].strip().lower()
    lines.append(replacements.get(name, raw))
lines += ["torch==2.9.0", "torchvision==0.24.0"]
Path(os.environ["PATCHED"]).write_text("\n".join(lines) + "\n")
PY

echo "==> Installing LIBERO and its dependencies"
uv pip install "cmake<4" setuptools wheel
uv pip install --no-build-isolation-package egl-probe --requirements "${PATCHED}"
uv pip install -e "${LIBERO_REPO}" --config-settings editable_mode=compat

# gr00t17's runtime stack. Explicit pins stop the resolver backtracking
# numba/llvmlite onto builds that have no python 3.12 wheel.
echo "==> Installing the gr00t17 runtime stack"
uv pip install torch==2.9.0 torchvision==0.24.0 pydantic av tianshou==0.5.1 \
    numba==0.65.1 llvmlite==0.47.0 tyro pandas dm_tree einops==0.8.1 \
    albumentations==1.4.18 zmq
uv pip install transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
# pyarrow backs the parquet writes that data re-collection produces.
uv pip install pyarrow

# The rest of gr00t17's model runtime, from its pyproject. Loading Gr00tN1d7
# pulls in diffusers (gr00t/model/modules/dit.py) and peft; the others are
# imported along the policy path. torchcodec is deliberately left out: it only
# decodes dataset videos, needs a system FFmpeg 4-7 that this host does not
# have, and simulator evaluation renders its frames directly.
uv pip install diffusers==0.35.1 peft==0.17.1 lmdb==1.7.5 jsonlines==4.0.0 datasets==3.6.0

# conf.yaml sets use_flash_attention. Take the official prebuilt cp312 wheel
# that gr00t17 pins in [tool.uv.sources]; building it from source takes about
# an hour.
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# robosuite 1.4.0 calls mj_fullM(model, dst, M); mujoco 3.10.0 changed that
# signature to mj_fullM(model, data, dst). mujoco is otherwise unpinned and
# would float to the latest release and break env creation.
echo "==> Pinning numpy and mujoco"
uv pip install numpy==1.26.4 mujoco==3.3.1

echo "==> Exposing gr00t17 at ${GR00T17_ROOT}"
GR00T17_ROOT="${GR00T17_ROOT}" python - <<'PY'
import os
import pathlib
import sysconfig

target = pathlib.Path(os.environ["GR00T17_ROOT"]).resolve().as_posix()
pathlib.Path(sysconfig.get_path("purelib"), "gr00t.pth").write_text(target + "\n")
print(f"gr00t.pth -> {target}")
PY

# robosuite 1.4.0 attaches a log handler to the hardcoded path /tmp/robosuite.log,
# which on a shared host usually belongs to another user and makes importing
# robosuite raise PermissionError. Disable it through macros_private.py.
echo "==> Disabling robosuite's hardcoded /tmp/robosuite.log handler"
python "${LOCAL_LIBERO_DIR}/_disable_robosuite_file_log.py"

# LIBERO asks on first import whether to download its datasets. Evaluation only
# needs the bddl task files, which ship with the checkout, so decline.
if [[ ! -d "${HOME}/.libero" ]]; then
    echo "==> Initialising ~/.libero (declining the dataset download)"
    printf 'n\n' | python -c "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs" || true
fi

echo "==> Verifying"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
python - <<'PY'
import gymnasium as gym
import mujoco, numpy, robosuite, torch

from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

register_libero_envs()
suites = sorted({key.split("/")[0] for key in gym.registry if key.startswith("libero_sim/")})
tasks = [key for key in gym.registry if key.startswith("libero_sim/")]
print(f"registered libero_sim tasks: {len(tasks)}")
print(f"numpy {numpy.__version__}  torch {torch.__version__}  "
      f"robosuite {robosuite.__version__}  mujoco {mujoco.__version__}  "
      f"gymnasium {gym.__version__}")

import os
if os.environ.get("VERIFY_RENDER", "1") == "0":
    print("Renderer check deferred to an allocated worker")
    raise SystemExit(0)
env = gym.make(tasks[0])
observation, info = env.reset(seed=0)
print(f"reset ok: {tasks[0]}")
print(f"  observation keys: {sorted(observation)}")
env.close()
PY

echo
echo "Ready. Activate with: source ${VENV}/.venv/bin/activate"
