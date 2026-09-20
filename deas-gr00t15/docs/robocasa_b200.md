# RoboCasa evaluation on B200

## Scope and installed environment

This checkout evaluates **PandaOmron kitchen tasks**, not the GR1 tabletop tasks in
some upstream GR00T examples. The DEAS actor is GR00T N1.5 and the critic reranks
sampled action sequences. The four supplied evaluation tasks are CoffeeSetupMug,
PnPMicrowaveToCounter, TurnOffStove, and PnPCounterToMicrowave.

The existing `../bench/robocasa` is RoboCasa 0.2.0 (commit 756598a), and
`../bench/robosuite` is robosuite 1.5.2 (commit 232ce7d4). Kitchen fixture,
texture, scene and Objaverse directories are already present. The optional
AI-generated object collection is absent; do not redownload all assets blindly.

`.venv-robocasa` inherits packages **read-only** from the existing personal
`/home/nas_main/dohyunlee/miniconda3/envs/groot-train` environment. It is not a
standalone environment: that parent must remain available and unchanged.
The project and both bench checkouts take import precedence through
`scripts/robocasa/environment.sh`. The separate `../Isaac-GR00T/.venv` is a newer
GR00T stack and is not used here.

Validated target versions: Python 3.11, torch 2.8.0+cu128, transformers 4.51.3,
diffusers 0.30.2, Gymnasium 1.0.0, NumPy 1.26.4, MuJoCo 3.2.6.
A complete installed-package snapshot is saved in
`output/robocasa-environment-freeze.txt`.

### Legacy dependency conflicts

RoboCasa 0.2.0 hard-asserts NumPy 1.23.x, while DEAS's albumentations and
numpydantic need newer NumPy. A small patch in
`../bench/robocasa/robocasa/__init__.py` allows **only NumPy 1.26.4** when
`ROBOCASA_ALLOW_NUMPY_126=1`; the environment script sets this opt-in. The original
check remains in force otherwise. The patch is saved as
`scripts/robocasa/robocasa-numpy126.patch`; `setup.sh` applies it if missing.

RoboCasa requires MuJoCo 3.2.6, while robosuite 1.5.2's package metadata requests
MuJoCo >=3.3.0. The evaluation environment deliberately uses 3.2.6 and installs
the local sources with `--no-deps`. Legacy NumPy/Numba/Tianshou pins also differ
from DEAS. Consequently `pip check` is **not clean**; actual task smoke tests
are the compatibility evidence, not a claim of general compatibility with every
robot/controller or task. Do not run an unconstrained dependency upgrade.

To reconstruct the overlay on the same parent, on **login**:

```bash
cd /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
bash scripts/robocasa/setup.sh
source scripts/robocasa/environment.sh
python -m pytest -q tests/test_robocasa_rollout.py
```

## Batch smoke check

Run from the DEAS repository so Slurm resolves log paths and the project root.
The NGC image provides runtime libraries and compilers; packages are installed
on login, never inside the worker. Each submission is one independent task.

```bash
mkdir -p slurm-logs
snode --json
sbatch --export=NONE slurm/robocasa_smoke.sbatch CoffeeSetupMug 1
# To validate spawn-based vector environments:
sbatch --export=NONE slurm/robocasa_smoke.sbatch PnPCounterToMicrowave 2
```

Resources: one GPU, eight CPUs, 48 GiB RAM, own QOS, 20-minute limit.
The smoke checks the actual DEAS wrapper, three 256x256 RGB cameras, language,
action chunks, reset and close. It imports actor and critic code but does **not**
load weights or measure task success. Images and `report.json` are written to
`output/robocasa-smoke/JOB_ID/`; stdout is in `slurm-logs/deas-rc-smoke-JOB_ID.log`.
Use `sjob JOB_ID` for worker diagnostics.

## Evaluate a trained actor or actor + critic

Required inputs:

- A RoboCasa-finetuned N1.5 actor checkpoint, including model weights,
  `config.json`, and `experiment_cfg/metadata.json` with `new_embodiment` statistics.
- For DEAS, a trained critic checkpoint with its model weights and config.
  The action horizon must match `critic_action_horizon` (default evaluation: 16).

`../models/GR00T-N1.5-3B` is a base model; its presence alone does not establish
that a RoboCasa actor or critic has been trained. No trained checkpoint was
selected for this setup task, and no task success rate is claimed.

```bash
# Arguments: ACTOR TASK [CRITIC] [EPISODES] [N_ENVS] [SEED]
sbatch --export=NONE slurm/robocasa_eval.sbatch \
  /absolute/path/to/actor/checkpoint-30000 CoffeeSetupMug '' 50 1 42

sbatch --export=NONE slurm/robocasa_eval.sbatch \
  /absolute/path/to/actor/checkpoint-30000 CoffeeSetupMug \
  /absolute/path/to/critic/checkpoint-30000 50 1 42
```

Resources: one GPU, eight CPUs, 96 GiB RAM, own QOS, four-hour limit.
The four hours is an initial conservative limit, **not a measured duration**;
adjust `sbatch --time=...` after a small worker evaluation, leaving 20–30% headroom.
Use `--export=NONE,ACTION_HORIZON=16,NUM_SAMPLES=10,TEMPERATURE=0.0`
when explicitly overriding model settings. Start with one environment and one
episode for the first real checkpoint. Each task / seed / method is a separate
submission. No inference server or live worker attachment is needed.

Outputs: `output/robocasa-eval/TASK/JOB_ID/eval.csv` and `success.txt`.
The existing held-out protocol is preserved: object split B and layout/style
pairs `(1,1), (2,2), (4,4), (6,9), (7,10)`; generative textures remain disabled
unless explicitly enabled. Environment seeds are offset by vector index.

Corrections made to evaluation: Text language observations compatible with
Gymnasium, unique vector seeds/video folders, temperature forwarding, action
noise application, success events within a chunk, exact executed-step counts,
time-limit truncation, next-step autoreset accounting and exact episode count.

## Training prerequisites

The repository's training entrypoints are `scripts/gr00t_finetune.py` and
`scripts/gr00t_deas_critic_finetune.py`. Their existing edits were preserved.
The example `bash_scripts/` use paths under the author's home/debug directories;
do not launch those unchanged on login. Actual actor/critic training needs
chosen dataset paths, output paths and a separate appropriately sized sbatch.
This setup does not start long training or select training hyperparameters.

## Verified results — 2026-09-18

All four worker jobs completed successfully using the NGC image and overlay venv.

| Task | Job | Vector environments | Result |
|---|---:|---:|---|
| CoffeeSetupMug | 136129 | 1 | passed |
| PnPMicrowaveToCounter | 136141 | 2 | passed |
| TurnOffStove | 136142 | 2 | passed |
| PnPCounterToMicrowave | 136143 | 2 | passed |

Each check executed eight simulator steps per environment, with object split B,
layout/style (1,1), three camera image checks and a second reset. This samples
assets; it does not exhaustively cover all layouts, styles or object instances.
The same CoffeeSetupMug check also passed on login under a five-minute timeout.
The two CPU regression tests for success/length/autoreset accounting passed.
Actor/critic module imports and the evaluation CLI `--help` passed.

Evidence: `output/robocasa-smoke/validation-summary.json`, per-job `report.json`,
three PNG files per job, and the corresponding `slurm-logs/` files.
**Not yet tested:** trained actor/critic weight loading, real policy actions,
full episodes / success rate, video encoding and demonstration collection.
