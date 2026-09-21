# DEAS critic training smoke (GR00T N1.5)

Exercises `deas-gr00t15/scripts/gr00t_deas_critic_finetune.py` end to end against
a synthetic dataset, so the pipeline is known-good before the real dataset
arrives. Uses the `deas-rc` environment from `local_eval/setup_env.sh`.

| File | Purpose |
| --- | --- |
| `make_fake_dataset.py` | Synthetic LeRobot dataset shaped for `single_panda_gripper_rl` |
| `smoke_critic_train.sh` | Short training run: base actor from the Hub, critic initialised fresh |
| `benchmark_train.py` | Training speed across batch sizes, loader workers and GPU counts |

## Running it

```bash
local_train/make_fake_dataset.py --output ~/data/fake_robocasa_rl --episodes 4 --frames 64
local_train/smoke_critic_train.sh
```

Both honour environment overrides: `DATASET`, `OUTPUT_DIR`, `BASE_MODEL`,
`GPU_DEVICE`, `MAX_STEPS`, `BATCH_SIZE`, `CRITIC_ACTION_HORIZON`, `WORKERS`,
`REPORT_TO`.

## The fake dataset

`single_panda_gripper_rl` is an RL config: on top of video/state/action/language
it reads `reward`, `done`, `next_state` and `next_video`, so `modality.json` must
be a `LeRobotRLModalityMetadata` with all eight sections and the dataset must be
loaded with `use_rl=True`. The generator writes:

- state, 16 dims: `end_effector_position_relative` 3, `end_effector_rotation_relative` 4 (quaternion), `gripper_qpos` 2, `base_position` 3, `base_rotation` 4 (quaternion). `next_state` maps to the same columns; the loader shifts by the delta index.
- action, 12 dims: `end_effector_position` 3, `end_effector_rotation` 3 (axis-angle), `gripper_close` 1, `base_motion` 4, `control_mode` 1.
- video: `left_view`, `right_view`, `wrist_view` as h264 mp4, from smoothly varying frames so decoding and colour jitter act on real signal.
- `next.reward` and `next.done` as **scalar** parquet columns — `get_reward_or_done` does `np.stack(...)` and asserts a 1-D result, so length-1 lists fail.
- Half the episodes carry a terminal reward, so both the sparse-reward expansion branch and the all-zero branch get hit. The dataset directory name contains `robocasa`, which is what selects the 15-step reward expansion in `get_reward_or_done`.

`meta/stats.json` is deliberately not written; the loader computes and caches the
statistics on first load, which exercises that path too.

## What the smoke run verified

Base model `nvidia/GR00T-N1.5-3B` (downloaded to the shared HF cache), critic and
value heads initialised fresh by
`GR00T_N1_5_DEAS_Critic.from_pretrained(..., from_gr00t_n1_5=True)`, 4 steps at
batch 2 on one A100, `critic_action_horizon=4`:

- The dataset loads through the full RL transform stack: `state`/`next_state` (1,64), `action` (16,32), `eagle_content`/`next_eagle_content`, `reward`, `done`.
- Training runs and reports both losses, e.g. `loss 5.188 = critic_loss 4.651 + value_loss 0.537`, with twin-Q and target-V statistics. Peak GPU memory 5.9 GiB allocated.
- A checkpoint is written with the expected structure — 784 tensors: `backbone.eagle_model` 584, `critic_head.vl_self_attention` 64, `critic_head.value` 54, `critic_head.critic` 36 (twin Q1/Q2), `critic_head.target_critic` 36 (Polyak target), `critic_head.backbone_encoder` 8, `critic_head.vlln` 2. The actor's DiT is not in this checkpoint; the eval path takes actor and critic as separate `--actor_model_path` / `--critic_model_path`.
- `config.json` carries `critic_config`, `value_config` and `rl_config` (`num_atoms` 101 for HL-Gauss, `q_agg` min, `critic_action_horizon` 4, `discount1`/`discount2` 0.995, `expectile` 0.9).
- `experiment_cfg/metadata.json` is written for `new_embodiment` with the RL modalities.
- The checkpoint reloads with `GR00T_N1_5_DEAS_Critic.from_pretrained` and a forward pass reproduces the same loss structure.

The numbers are meaningless — the data is noise. Only the plumbing was tested.

Note when driving the model outside the HF `Trainer`: the training arguments set
`bf16=True`, so a manual forward needs
`torch.autocast('cuda', dtype=torch.bfloat16)` or layer norm raises
`expected scalar type BFloat16 but found Float`.

## Measuring training speed

```bash
local_train/benchmark_train.py --batch-sizes 2 4 8 --workers 0 8 --gpus 0 3 --steps 40
```

Each case is a real `gr00t_deas_critic_finetune.py` run. Speed comes from the
`performance.jsonl` that `TrainingMetricsCallback` writes: one line every
`logging_steps` optimizer steps, each holding the seconds-per-step for that
window alone. `--warmup-windows` (default 1) drops the leading windows, so model
loading, CUDA context creation and first-step allocator growth stay out of the
number. `logging_steps` is hardcoded to 10 in the training script, so `--steps`
should be a multiple of 10 and must leave at least one window after the warm-up.

The rates are end to end — data loading, host-to-device copies, forward,
backward and the optimizer step — which is what you need to plan a run. They are
not kernel benchmarks.

`--gpu-counts 1 2 4` sweeps GPU counts, taking the first N entries of `--gpus`;
anything above 1 goes through the script's own torchrun path. `--dry-run` prints
the commands.

Output under `local_outputs/train_benchmark/<timestamp>/`: `summary.json`,
`summary.csv`, and a per-case directory with `train.log` and `performance.jsonl`.
Every case ends with an unavoidable full checkpoint save, so the script deletes
those artefacts once the timings are read, keeping only logs and metrics; pass
`--keep-checkpoints` to retain them. One case freed 9.9 GB and left 196 KB.

Measured example, one A100 80GB, batch 2, no loader workers, 30 steps against the
4-episode fake dataset: warm-up window 0.340 s/step, then 0.299 and 0.288, giving
**0.293 s/step, 6.8 samples/s, 5.89 GiB peak allocated**. Walltime for the case
was 144 s, most of it model loading.

One caveat on the numbers: the default fake dataset is 4 episodes of 64 frames,
about 276 KB, so it sits in page cache and video decoding costs almost nothing.
That makes the figures an upper bound and makes a `--workers` sweep
uninformative. For a data-loading-realistic measurement, generate a bigger set
first, or point `--dataset-path` at the real data:

```bash
local_train/make_fake_dataset.py --output ~/data/fake_robocasa_rl_big \
    --episodes 40 --frames 300 --image-size 256
```

## When the real dataset arrives

Point `--dataset-path` at it (the script takes several paths and builds a
`LeRobotMixtureDataset`), keep `--data-config single_panda_gripper_rl`, and raise
`--max-steps`, `--batch-size` and `--num-gpus`. `deas-gr00t15/bash_scripts/train_deas_critic.sh`
is the reference invocation for the four-task RoboCasa setup.

A run leaves about 9 GB per saved checkpoint (4 GB model plus 1.1 GB optimizer
state, `save_total_limit=8`); this filesystem is at 94%, so set `--save-steps`
with that in mind.
