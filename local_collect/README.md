# Re-collecting training data from real evaluation runs

Runs a policy in the simulator, keeps the videos, and turns the rollouts into
GR00T LeRobot v2.1 datasets that load straight into DEAS critic / IQL training
alongside the human demos — the `rollouts/` half of what
`deas-gr00t15/bash_scripts/train_deas_critic.sh` expects.

| File | Purpose |
| --- | --- |
| `collect_robocasa.sh` | RoboCasa: evaluate, then replay into a dataset |
| `robocasa_replay_to_lerobot.py` | Replay recorded states, streaming frames to mp4 |
| `robocasa_to_lerobot.py` | Convert an already-replayed observation HDF5 |
| `collect_libero.sh` | LIBERO: collect in parallel, then assemble |
| `collect_libero.py` | Parallel collection runner (LIBERO island) |
| `libero_recorder.py` | Per-episode recorder placed on the raw LIBERO env |
| `libero_shards_to_lerobot.py` | Assemble shards into datasets |

Successful and failed episodes are both kept, because IQL needs the failures.
Every episode records its outcome in `meta/episodes.jsonl`, so success-only
subsets can be taken later without re-running anything.

## Images are only ever stored as video

Neither pipeline writes a raw image array to disk. This is worth stating because
the obvious routes do:

- RoboCasa's own `dataset_states_to_obs.py` replays states and writes the
  regenerated frames back into HDF5 as raw uint8 — at 128x128x3, three cameras
  and ~300 steps that is roughly 44 MB per episode, ~2 GB for a 50-episode task
  — and the result is then read again to be encoded as video anyway.
  `robocasa_replay_to_lerobot.py` does the same replay and appends each frame to
  its mp4 writer as it is produced, so the intermediate never exists and only
  one frame per camera is ever in memory.
- For LIBERO, `libero_recorder.py` encodes as it steps, inside each worker
  process, so images never cross a process boundary either. Assembly moves the
  finished mp4s into place; nothing is decoded or re-encoded.

## RoboCasa

```bash
ACTOR=/path/to/ckpt ENV_NAME=CoffeeSetupMug N_EPISODES=50 N_ENVS=8 \
  local_collect/collect_robocasa.sh
```

Stage 1 evaluates with `--collect_data` and `--save_video`. `DataCollectionWrapper`
records simulator states, actions, rewards and dones into `demo.hdf5`; images are
not stored there. Stage 2 replays those states to regenerate the camera views and
writes the dataset. `STAGE=eval` or `STAGE=replay` re-runs one stage.

Best-of-N rollouts: add `CRITIC=/path/to/critic MODEL_TYPE=deas`. Other knobs:
`SEED` `GPU_DEVICE` `CAMERA_SIZE` `ACTION_HORIZON` `DENOISING_STEPS` `NUM_SAMPLES`
`TEMPERATURE` `SAVE_VIDEO` `SAVE_TRACE` `OUTPUT_ROOT`.

Output: `eval/videos` (per-episode rollout videos), `raw/demo.hdf5`,
`dataset/<task>` (the LeRobot dataset, with its own per-camera videos).

Rewards and dones are copied from the recording rather than re-inferred, so the
dataset carries the signal the evaluated policy actually produced.

## LIBERO

```bash
MODEL=/path/to/n17 SUITES=libero_spatial N_EPISODES=20 N_ENVS=8 \
  local_collect/collect_libero.sh
```

Collection reuses the evaluation environment stack — `--n-envs` spawned LIBERO
simulators under gr00t17's video and multi-step wrappers — with an
`EpisodeRecorder` inserted directly on the raw `LiberoEnv`. That position
matters: above the `MultiStepWrapper` only the policy's action chunks are
visible, while training data needs one row per simulator step.

Each worker writes its own shards, so nothing large is pickled between
processes. Episodes are finalised on reset, which is also when a vector env
autoresets, and on close, so an interrupted run still leaves complete episodes.

`GROUP_BY=suite` (default) produces one dataset per suite with the tasks
separated by `task_index`; `GROUP_BY=task` produces one per task. `STAGE=collect`
or `STAGE=assemble` re-runs one stage. Other knobs: `SEED` `FPS` `GPU_DEVICE`
`N_ACTION_STEPS` `SAVE_VIDEO` `MOVE_SHARDS` `OUTPUT_ROOT`.

## What the datasets contain

`modality.json` carries `state`, `action`, `video`, `annotation` **and** the
`reward`, `done`, `next_state`, `next_video` sections that
`LeRobotRLModalityMetadata` requires. Without those four a dataset loaded with
`use_rl=True` fails with `unexpected modality: reward`. They are additive, so
non-RL configs ignore them and the same dataset also serves BC retraining.

`next.reward` and `next.done` are stored as scalar parquet columns, not
length-1 lists: `get_reward_or_done` stacks them and asserts a 1-D result.

`meta/stats.json` is deliberately absent; `LeRobotSingleDataset` computes and
caches it on first load from the parquet files it will actually read.

RoboCasa key naming differs from `~/Value/robocasa/convert_robocasa_to_groot_lerobot_v21.py`,
whose output cannot be loaded by `single_panda_gripper_rl`: video keys here are
`left_view` / `right_view` / `wrist_view`, and action keys are
`end_effector_position` / `end_effector_rotation` / `gripper_close` /
`base_motion` / `control_mode`.

## Verified

- RoboCasa converter, on a synthesised robomimic HDF5: loads with both
  `single_panda_gripper_rl` (`state (1,64)`, `next_state (1,64)`,
  `action (16,32)`, `reward (4,)`, `done (4,)`) and the BC inference config.
- RoboCasa replay machinery: `create_env_for_data_processing` builds the env with
  all three cameras and returns `(128, 128, 3) uint8`.
- LIBERO end to end with random actions: the recorder produced real shards
  (two mp4s, `frames.parquet`, `shard.json`), the assembler built a dataset with
  all eight modality sections, and it loaded with deas-gr00t15's `libero` config
  (`state (1,64)`, `action (16,32)`).

Not yet exercised: either pipeline driven by a real checkpoint. The RoboCasa
eval-to-replay leg needs an N1.5 RoboCasa checkpoint, and LIBERO collection
needs an N1.7 one; neither is on this machine.

## The LIBERO gripper width

LIBERO reports `robot0_gripper_qpos`, two numbers, and the recorder stores both
under `state.gripper_close`. A checkpoint trained elsewhere may expect a single
scalar there. If you are matching an existing checkpoint rather than training on
this data, check its `experiment_cfg/metadata.json` and pass `--gripper-scalar`
to `libero_shards_to_lerobot.py` to collapse the pair to one openness value.
