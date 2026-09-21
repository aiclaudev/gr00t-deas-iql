#!/usr/bin/env bash
# Run installation on login; renderer verification belongs to smoke.sbatch.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cluster_eval/env.sh"
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os
from pathlib import Path
root=Path(os.environ['LIBERO_REPO'])/'libero/libero'
p=Path(os.environ['LIBERO_CONFIG_PATH'])/'config.yaml'
if not p.exists():
    paths={'benchmark_root':root,'bddl_files':root/'bddl_files',
           'init_states':root/'init_files','assets':root/'assets',
           'datasets':Path(os.environ['LIBERO_VENV']).parents[1]/'libero_datasets'}
    p.write_text('\n'.join(f'{k}: {v}' for k,v in paths.items())+'\n')
PY
VERIFY_RENDER=0 bash "$ROOT/local_libero/setup_env.sh"
