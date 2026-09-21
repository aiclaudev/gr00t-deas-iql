# Local parallel RoboCasa evaluation (GR00T N1.5 / DEAS)

Run fine-tuned `deas-gr00t15` checkpoints against RoboCasa on this machine, with
several simulators stepping in parallel.

## Where the parallelism comes from

The repository already has it. `gr00t/eval/wrappers/robocasa_wrapper.py`'s
`load_robocasa_gym_env(..., n_envs=N)` builds a `gymnasium.vector.AsyncVectorEnv`
of N spawned RoboCasa processes, and `gr00t/eval/rollout.py`'s
`evaluate_vector_policy` steps all of them together, so each `policy.get_action`
is one batched inference over every live environment. That is the same
lockstep-batch shape as the ZMQ parallel eval in `~/Value/Isaac-GR00T`, minus
the process boundary and the msgpack round trip.

The ZMQ split existed there because RoboCasa 0.2.0 asserts NumPy 1.23.x, which
cannot coexist with GR00T's stack in one environment. This repository ships
`scripts/robocasa/robocasa-numpy126.patch`, which adds an opt-in for NumPy
1.26.4 behind `ROBOCASA_ALLOW_NUMPY_126`, so the simulator and the policy run in
one process and no server is needed.

## Files

| File | Purpose |
| --- | --- |
| `setup_env.sh` | Build the `deas-rc` conda environment; idempotent |
| `env.sh` | Runtime exports; sourced by everything else |
| `fetch_weights.py` | Download a checkpoint from the Hugging Face Hub |
| `run_eval.sh` | One task, one checkpoint |
| `run_suite.py` | Several tasks and seeds across GPUs, then the summary table |

## Setup

```bash
local_eval/setup_env.sh
```

This clones the existing `gr00t-train` environment — which already carries a
compiled `flash-attn` wheel for this python/torch ABI, and
`gr00t/model/backbone/eagle2_hg_model/radio_model.py` imports `flash_attn` at
module level — then repoints `gr00t` at `deas-gr00t15`, adds the simulator
runtime, and applies the RoboCasa NumPy patch.

`robosuite` and `robocasa` are used from `~/workspace/` over `PYTHONPATH` rather
than pip-installed, so the existing `robocasa-eval` environment keeps working
from the same sources. The NumPy patch is backwards compatible: the 1.23.x
assertion path is untouched.

Verified after setup: torch 2.5.1+cu124, numpy 1.26.4, gymnasium 1.0.0,
mujoco 3.2.6, robosuite 1.5.2, robocasa 0.2.0, flash-attn 2.7.1.post4.

## Getting weights

```bash
ACTOR=$(local_eval/fetch_weights.py --repo-id my-org/gr00t-n15-robocasa | tail -1)

# A repository holding several checkpoints:
ACTOR=$(local_eval/fetch_weights.py --repo-id my-org/runs --subfolder checkpoint-20000 | tail -1)
```

Private repositories read `$HF_TOKEN`. A checkpoint directory must contain
`config.json` and `experiment_cfg/metadata.json`; `fetch_weights.py` says so and
lists the plausible subdirectories when it does not.

`run_suite.py` can also resolve checkpoints itself:
`--actor hf://my-org/runs@v2#checkpoint-20000`.

## One task

```bash
ACTOR=/path/to/checkpoint ENV_NAME=CoffeeSetupMug N_ENVS=8 N_EPISODES=50 \
  GPU_DEVICE=0 local_eval/run_eval.sh
```

Best-of-N with a critic instead of plain BC:

```bash
ACTOR=/path/to/actor CRITIC=/path/to/critic MODEL_TYPE=deas DEAS_BACKEND=iql \
  NUM_SAMPLES=8 ENV_NAME=CoffeeSetupMug N_ENVS=8 local_eval/run_eval.sh
```

## Several tasks

```bash
local_eval/run_suite.py \
  --actor /path/to/checkpoint \
  --output-root ~/iclr2027/runs/bc-eval-1 \
  --tasks CoffeeSetupMug PnPMicrowaveToCounter TurnOffStove PnPCounterToMicrowave \
  --episodes 50 --n-envs 8 --gpus 0 3 --save-video
```

Each job runs `--n-envs` simulators in parallel and occupies one GPU; jobs are
spread over `--gpus` (`--jobs-per-gpu` to pack more onto each). `--dry-run`
prints the plan and the exact commands without writing anything.

Add `--eval-seeds 42 7 13` for one run per seed per task; the summary then also
carries a per-seed mean and standard deviation. `--skip-completed` reuses
finished job directories, which makes a re-run after a partial failure cheap.

Best-of-N: add `--critic <path>` (this switches the method to `deas`), with
`--num-samples`, `--temperature` and `--deas-backend`.

Output under `--output-root`:

```
manifest.json                                   plan and live job state
results/seed-<t>/eval-<e>/<task>/result.json    per-run result, videos/, inference/
aggregate/summary.md, summary.json, runs.csv, by_task.csv
```

The manifest is the schema `scripts/robocasa/aggregate_results.py` validates, so
that aggregator produces the summary unchanged. It is strict on purpose: a run
whose recorded configuration disagrees with the plan is reported as invalid
rather than folded into the success rate, and missing runs are never counted as
zero successes. Unlike `scripts/robocasa/submit_evaluations.py`, which is Slurm-
and NAS-specific, `run_suite.py` runs locally.

## Videos and Q curves

`--save-video` writes per-episode MP4s; `--save-inference-inputs` additionally
records observations, RNG state, candidates and Q values. With both, the
existing renderers work on the output directory:

```bash
# One success/failure pair, side by side, sharing a Q scale
conda run -n deas-rc python deas-gr00t15/scripts/robocasa/render_q_comparison.py \
  --task-dir <output-root>/results/seed-0/eval-42/CoffeeSetupMug \
  --output coffee.mp4 --title Coffee --pair-index 0

# Every recorded BoN run under a results tree
conda run -n deas-rc python deas-gr00t15/scripts/robocasa/render_q_comparisons.py \
  --results <output-root>/results --output <output-root>/q-videos
```

`scripts/robocasa/replay_inference.py` replays a saved trace and, with
`--recompute`, re-runs inference to check the recorded actions reproduce.

## Evaluation protocol

`evaluation_protocol` in `scripts/eval_policy_robocasa.py` fixes the held-out
settings regardless of the legacy `--layout`/`--style` flags: object instance
split B, layout/style pairs (1,1) (2,2) (4,4) (6,9) (7,10), 256x256 cameras, no
camera randomisation, termination on first success. Environment seeding is
`evaluation seed + env index`, so a given `--eval-seeds` value with a given
`--n-envs` reproduces the same scenes. **Keep `--n-envs` fixed when comparing
checkpoints**: changing it changes the per-environment seeds and therefore the
scenes being evaluated.

## Verified so far

- The environment builds and imports cleanly (versions above).
- `scripts/robocasa/smoke.py --n-envs 4` passes: EGL offscreen rendering, three
  256x256 camera views with real image variance, and batched action chunks.
- The `run_suite.py` manifest is accepted by `aggregate_results.py` for both the
  BC and the best-of-N paths, including its video validation.

Not yet exercised: loading an actual fine-tuned checkpoint and running batched
inference, because no such checkpoint is on this machine yet. The
`nvidia/GR00T-N1.5-3B` entry in the Hugging Face cache holds no weights, and the
base model carries no `new_embodiment` metadata for this data config, so it is
not a useful stand-in.
