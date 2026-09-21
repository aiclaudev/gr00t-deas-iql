# Parallel LIBERO evaluation

Evaluates GR00T checkpoints on LIBERO with several simulators stepping in
parallel, across whole task suites.

| File | Purpose |
| --- | --- |
| `setup_env.sh` | Build the LIBERO island environment |
| `env.sh` | Runtime exports; sourced by everything else |
| `list_tasks.py` | Task names per suite, as gymnasium ids |
| `run_suite_gr00t17.py` | Evaluate an N1.7 checkpoint over whole suites |
| `run_suite_gr00t17.sh` | Shell wrapper that activates the island |
| `_disable_robosuite_file_log.py` | Setup helper; see below |

## Why LIBERO needs its own environment

LIBERO pins `robosuite==1.4.0`. The RoboCasa side (`local_eval`, `deas-rc`) runs
robosuite 1.5.2, and the two cannot share an interpreter. So LIBERO gets an
"island": its own python 3.12 virtualenv holding LIBERO, robosuite 1.4.0,
mujoco 3.3.1 and numpy 1.26.4, with gr00t17 exposed through a `.pth` file rather
than installed, so the island supplies gr00t's runtime dependencies itself and
nothing is re-resolved. This follows
`gr00t17/gr00t/eval/sim/LIBERO/setup_libero.sh`, with the LIBERO checkout
already on this machine used instead of a git submodule, and the virtualenv
placed outside the vendored snapshot.

mujoco is pinned to 3.3.1 because robosuite 1.4.0 calls
`mj_fullM(model, dst, M)`, whose signature changed in mujoco 3.10.0.

## Setup

```bash
local_libero/setup_env.sh        # builds ~/envs/libero_island
```

Verified after setup: 130 registered `libero_sim/` tasks, numpy 1.26.4,
torch 2.9.0+cu128, robosuite 1.4.0, mujoco 3.3.1, gymnasium 0.29.1, and EGL
offscreen rendering producing 256x256 frames with real image variance.

Two things the setup handles that are easy to trip over on a shared host:

- robosuite 1.4.0 ships `FILE_LOGGING_LEVEL = "DEBUG"`, and its logger attaches
  a handler to the literal path `/tmp/robosuite.log`. On this machine that file
  belongs to another user, so merely importing robosuite raised
  `PermissionError`. `_disable_robosuite_file_log.py` writes the
  `macros_private.py` override robosuite imports first, with file logging off.
- LIBERO prompts on first import about downloading its datasets. Evaluation only
  needs the bddl task files, which ship with the checkout, so the setup answers
  no and creates `~/.libero` itself.

## Suites

130 tasks: `libero_spatial`, `libero_object`, `libero_goal` and `libero_10` with
10 each, and `libero_90` with 90.

```bash
source ~/envs/libero_island/.venv/bin/activate
local_libero/list_tasks.py                    # all suites
local_libero/list_tasks.py --suites libero_10 --json
```

## Evaluating an N1.7 checkpoint

```bash
MODEL=/path/to/checkpoint OUTPUT_ROOT=~/runs/libero-n17 \
  local_libero/run_suite_gr00t17.sh

# one suite, fewer episodes
MODEL=/path/to/checkpoint SUITES=libero_spatial N_EPISODES=10 N_ENVS=8 \
  local_libero/run_suite_gr00t17.sh
```

Environment overrides: `MODEL` `OUTPUT_ROOT` `SUITES` `N_EPISODES` `N_ENVS`
`N_ACTION_STEPS` `SEED` `GPU_DEVICE` `SAVE_VIDEO`. Extra arguments pass through
to `run_suite_gr00t17.py`, which also takes `--policy-client-host/--policy-client-port`
to evaluate against a running policy server, `--tasks` to pick individual tasks,
`--skip-completed` to resume, and `--dry-run`.

Parallelism is gr00t17's own: `run_rollout_gymnasium_policy` builds an
`AsyncVectorEnv` of `--n-envs` spawned LIBERO simulators and steps them
together, so each policy call is one batched inference over every live
environment. Verified directly: 4 spawned simulators returning batched
`(4, 256, 256, 3)` observations and stepping in lockstep.

**The policy is loaded once and reused across tasks.** gr00t17's own
`rollout_policy.py` entrypoint evaluates one task per process, which for the
full benchmark would reload a multi-billion parameter checkpoint 130 times.

Output under `--output-root`:

```
results/<suite>/<task>.json     per task, written as each finishes
summary.json, summary.md        refreshed after every task
videos/<suite>/<task>/          with --save-video
```

Per-suite success rate is reported as the mean over tasks, which is how LIBERO
is normally reported, with the pooled rate over all episodes alongside. A task
that raises is recorded as failed and the sweep continues.

`--max-episode-steps` defaults to 520 for `libero_10` and `libero_90` and 280
for the others; pass it explicitly to override.

## Cost

The full benchmark at the usual 50 episodes per task is 130 x 50 = 6500
episodes. Start with one suite, or a lower `--n-episodes`, before committing to
a full sweep, and use `--skip-completed` so an interrupted run resumes.

## Not done yet: LIBERO on deas-gr00t15 (N1.5)

Only the N1.7 path exists so far. `deas-gr00t15` has no LIBERO environment at
all — just `LiberoDataConfig` in `gr00t/experiment/data_config.py` for training
data — so the N1.5 path needs a wrapper mirroring
`gr00t/eval/wrappers/robocasa_wrapper.py`, an `eval_policy_libero.py` mirroring
`eval_policy_robocasa.py`, and a third environment (deas-rc cloned, with
robosuite 1.4.0 replacing 1.5.2).

The blocker is the observation mapping. `LiberoDataConfig` expects
`state.eef_pos_absolute`, `state.eef_rot_absolute` and `state.gripper_close`,
while gr00t17's `LiberoEnv` emits `state.x/y/z/roll/pitch/yaw/gripper`. Which
LIBERO quantity feeds each N1.5 key, and in which rotation convention, is fixed
by the dataset the N1.5 checkpoint was trained on, not by anything in this
repository. A finetuned N1.5 LIBERO checkpoint settles it: its
`experiment_cfg/metadata.json` declares each state key's shape and rotation
type. Guessing instead would produce an evaluation that runs and silently
scores garbage.
