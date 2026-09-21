# B200 parallel evaluation

Use these wrappers from the repository root.
All real evaluations and timing comparisons run through Slurm. Check
`snode --json` before submission; explicit `--qos=own` is included.

## Environments

- RoboCasa / N1.5: existing `../DEAS-Isaac-GR00T/.venv-robocasa`, torch 2.8 + cu128.
  Source simulators: `../bench/robosuite` and `../bench/robocasa`.
- LIBERO / N1.7: separate `../eval_envs/libero17/.venv`, torch 2.9 + cu128,
  robosuite 1.4, mujoco 3.3.1, gymnasium 0.29.1. Checkout at
  `../Isaac-GR00T/external_dependencies/LIBERO`.
- This setup covers the two combinations implemented by PARALLEL_EVAL_SETUP.md.
  It does not add N1.5 LIBERO or N1.7 RoboCasa adapters.
- `env.sh` pins five CPU thread pools to one thread, shared HF cache paths,
  GPU-local EGL index 0, and a separate LIBERO configuration directory.
  The A100 document's cu124 wheel is not installed on B200.

## RoboCasa SVF BoN

```bash
ROOT="$(pwd)"
WORK="$(dirname "$ROOT")"
ACTOR=/path/to/exported/actor
CRITIC=/path/to/exported/critic
REFERENCE=/path/to/bc2-reference
snode --json
sbatch "$ROOT/cluster_eval/robocasa.sbatch" "$ROOT" \
  --actor "$ACTOR" --critic "$CRITIC" \
  --deas-backend svf --critic-reference-actor "$REFERENCE" \
  --output-root "$WORK/parallel_evals/svf-bon10-seed42" \
  --episodes 50 --eval-seeds 42 --training-seed 42 \
  --save-video --save-inference-inputs
```

One GPU / 12 CPUs / 192 GiB / 12 hours. Eight environments are batched per
policy call; the four tasks run sequentially. BoN defaults to 10. Keep n_envs
fixed when comparing models. Omit critic and reference options for plain BC.
The launcher uses the personal interpreter directly, without conda run.

## LIBERO N1.7

```bash
ROOT="$(pwd)"
WORK="$(dirname "$ROOT")"
MODEL=/path/to/prepared/libero-checkpoint
snode --json
sbatch "$ROOT/cluster_eval/libero.sbatch" "$ROOT" "$MODEL" \
  "$WORK/parallel_evals/libero17-fewshot" 42 50 8
```

One GPU / 24 CPUs / 192 GiB / 4 hours. Four suites concurrently on that GPU,
eight environments per suite. Final arguments: seed, episodes/task, n_envs.
The remaining own CPU budget matters: both wrappers together request 36 CPUs.
For a checkpoint downloaded from another machine, use
`local_libero/prepare_checkpoint.py` first to fix stale processor paths.

## Smoke checks

```bash
sbatch cluster_eval/smoke.sbatch "$PWD" robocasa /path/to/rc-smoke
sbatch cluster_eval/smoke.sbatch "$PWD" libero /path/to/libero-smoke
```

These check parallel simulator reset/step/rendering, not policy success rate or
throughput. Reports are written under the supplied output directory.
Actual evaluation is not automatically submitted by setup.

To rebuild the LIBERO environment on this server, run
`bash cluster_eval/setup_libero.sh` on login. It skips GPU rendering during
installation; `smoke.sbatch` handles that on a worker. The GLVND/EGL libraries
come from the existing personal `groot-train/lib` directory (`EVAL_GL_LIB`
overrides it).

## Verification

B200 worker smoke checks passed for RoboCasa parallel reset/chunk stepping and
three-camera rendering, and for LIBERO with two environments and two steps.
LIBERO registered 130 tasks; the N1.7 model import and the SVF direct-interpreter
launch-plan regression also passed. Smoke now defaults to two environments.
These checks do not measure success rate, model-Q equivalence, or throughput.
