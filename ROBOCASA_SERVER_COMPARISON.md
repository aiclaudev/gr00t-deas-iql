# RoboCasa evaluation: B200 reference and remote comparison

This records the B200 setup inspected on 2026-09-21. The A100 environment
documented in PARALLEL_EVAL_SETUP.md is a separate environment.

## Code and pipeline

| Stage | Source |
| --- | --- |
| Task/seed launcher, shared-memory observations by default | local_eval/run_suite.py |
| B200 Slurm launcher and environment | cluster_eval/robocasa.sbatch, cluster_eval/env.sh |
| Evaluation protocol and BC/DEAS/IQL/SVF backend selection | deas-gr00t15/scripts/eval_policy_robocasa.py |
| Episode termination/autoreset loop | deas-gr00t15/gr00t/eval/rollout.py |
| Raw observations, RNG, explicit BC initial noise, reference actions | deas-gr00t15/gr00t/eval/inference_trace.py |
| BC action comparison without simulator | deas-gr00t15/scripts/robocasa/replay_bc_inputs.py |
| Simulator-state replay into LeRobot v2.1 with camera MP4s | local_collect/robocasa_replay_to_lerobot.py |
| Two episodes per task, conversion, reference replay | cluster_eval/robocasa_bc_replay.sbatch |
| Artifact validation and optional HF dataset upload | local_collect/package_bc_replay.py |

The runtime workspaces remain siblings of this combined repository. A source sync
dry run produced no differences before this documentation was added.

## B200 environment

Python 3.11.10, torch 2.8.0+cu128, torchvision 0.23.0+cu128,
transformers 4.51.3, flash-attn 2.8.3, numpy 1.26.4,
diffusers 0.30.2, gymnasium 1.0.0, mujoco 3.2.6,
robosuite 1.5.2, robocasa 0.2.0, safetensors 0.7.0.
RoboCasa/robosuite are imported from source checkouts selected by PYTHONPATH.
Package versions alone do not establish that those source trees match.

The actor computes in BF16. Copying the A100 torch 2.5.1/cu124 settings would not
reproduce this package environment. Record GPU, driver, package versions, source
revision, model revision, and normalization metadata when comparing.

## Checkpoints and normalization

- Actor: RLobot-jun/gr00t-n1.5-robocasa-bc2-30k,
  revision 8c057ce617b4dd87544f5fc36acb2dbb30b8ceaf.
- Optional DEAS critic: RLobot-jun/gr00t-n1.5-robocasa-deas-critic-bc2-30k-30k,
  revision dc96e83e755c51066fda18352ac638e6294fb7cd.
- N1.5 reads state/action statistics from experiment_cfg/metadata.json.
  Its Eagle image/text processor comes from the code/base processor assets;
  these N1.5 exports do not require an N1.7-style processor/ directory.
- For this DEAS critic, explicitly select --deas-backend checkpoint and
  --num-samples 10. The generic launcher's defaults are not this checkpoint's
  backend/BoN configuration.
- CheckpointDEASBoNPolicy uses actor statistics to produce robot actions, then
  normalizes those actions with the critic's own statistics. Preserve the
  checkpoint's online_q_feature_passes=2.

## Protocol to match

Four tasks: CoffeeSetupMug, PnPCounterToMicrowave, PnPMicrowaveToCounter,
TurnOffStove. Robot PandaOmron; object split B; layout/style pairs
(1,1), (2,2), (4,4), (6,9), (7,10). Three cameras, 256x256.
No generative textures or camera randomization. Stop on first success, native
termination, or the task's registry horizon. Action horizon 16, execute horizon
16, four denoising steps, action noise zero.

Keep environment count fixed: environment seed is evaluation seed + env index.
The launcher adds --shared_memory; direct calls to eval_policy_robocasa.py must
add it explicitly. Set GR00T_EVAL_DIAGNOSTICS=1 for policy, step, and reset logs.

The completed B200 shared-memory boundary test used 2 environments, Coffee,
4 episodes, seed 42, BC2-30k plus DEAS critic-30k, BoN10. It completed all four
episodes (0/4 successful, horizon 600); this verifies episode progression for
that configuration, not policy quality or all-task parallel reliability.
The new BC-only input-comparison job was submitted separately; submission does
not establish that its data or replay validation has completed.

## Run on a remote server

Activate its N1.5/RoboCasa environment. Set these paths for that server:

```bash
export DEAS_ROOT=/path/to/gr00t-deas-iql/deas-gr00t15
export ROBOSUITE_DIR=/path/to/robosuite
export ROBOCASA_DIR=/path/to/robocasa
source local_eval/env.sh
python local_eval/run_suite.py \
  --python-executable "$(command -v python)" \
  --actor /path/to/bc2-30k --output-root /path/to/new-bc2-eval \
  --tasks CoffeeSetupMug PnPCounterToMicrowave PnPMicrowaveToCounter TurnOffStove \
  --episodes 2 --n-envs 1 --gpus 0 --jobs-per-gpu 1 \
  --eval-seeds 42 --training-seed 42 \
  --action-horizon 16 --execute-horizon 16 --denoising-steps 4 \
  --save-video --save-inference-inputs
```

Run GPU commands inside an allocation if that server requires it.
For the B200 cluster, use the sbatch launchers, not the login node.
To compare the prior DEAS test, use Coffee, --episodes 4 --n-envs 2,
and add --critic /path/to/deas-critic --deas-backend checkpoint --num-samples 10.

## Compare identical observations

The requested HF dataset is RLobot-jun/robocasa-bc2-30k-replay-2ep.
It is published only after the recording, conversion, and reference replay pass.
Once it exists, download the dataset and pinned actor, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python /path/to/dataset/scripts/replay_bc_inputs.py \
  --dataset /path/to/dataset --actor /path/to/bc2-30k \
  --report /path/to/comparison.json
```

This uses lossless NPZ observations and explicit initial flow noise. MP4 frames
are for viewing/training; compression makes them unsuitable for exact input
comparisons. The report contains per-action max/mean absolute error and bitwise
equality. Default: first/middle/last calls per task, atol=1e-5, rtol=0.
Use --max-calls-per-task 0 for every call. A mismatch writes the report and exits
with code 1. Cross-hardware BF16 results need not be bitwise identical.

Replay isolates policy inference from simulator rendering/reset differences.
It compares BC actions and does not evaluate critic Q values.
