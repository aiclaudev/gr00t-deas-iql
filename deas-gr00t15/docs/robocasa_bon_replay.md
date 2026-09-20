# Seed42 DEAS BoN evaluation and inference replay

## Evaluation protocol

- Actor: `output/deas-training/20260918T183345.797990880Z/02-bc-rollout` (step 10000).
- Critic: `output/deas-training/20260918T183345.797990880Z/03-critic` (step 10000).
- Training seed 42; evaluation seeds 0, 1, 2.
- Tasks: CoffeeSetupMug, PnPMicrowaveToCounter, TurnOffStove, PnPCounterToMicrowave.
- 50 episodes per task and evaluation seed: 600 total.
- BoN: 10 candidates, greedy maximum of min(Q1,Q2), temperature 0, 4 denoising steps.
- Predict 16 actions and execute 16 for the initial baseline.
- Held-out object split B, layout/style pairs (1,1), (2,2), (4,4), (6,9), (7,10).
- Each task/seed is a separate own-QOS sbatch: 1 GPU, 8 CPU, 96 GiB RAM.
  Subsequent seeds for a task use afterany dependencies, at most 4 evaluation GPUs.
- All episodes get MP4 videos. W&B receives scalar metrics and configuration under
  `aiclaudev / gr00t1.5 finetune`; videos, input images, model weights stay on NAS.

## Checkpoint-specific critic inputs

`--deas_backend checkpoint` loads the actor and critic separately. It preserves
actor future tokens and each checkpoint's frozen vision/language transformation.
Actor actions are unnormalized to physical actions, then normalized with the
critic's own metadata for scoring. Unused critic action dimensions are zero-padded.
The original physical candidate is executed.

The existing critic training forward mutates the same backbone features in
`compute_value_loss`, then again in `compute_critic_loss`. Therefore the saved
online Q was trained on two transformations of current-observation features.
This evaluation explicitly uses two passes to match that checkpoint. A CPU test
executes the actual training methods to verify this behavior. Training code and
existing checkpoint weights were not changed; this does not repair any effects
that the original training implementation had on learning quality.

## Saved artifacts

The submission root contains `manifest.json`, Slurm logs, source hashes and a
`source-snapshot/repo` copy used by the evaluation workers. Later edits to the
working tree do not change their policy or rollout implementation.

Each `results/eval-seed-<seed>/<task>/` directory contains:

- `result.json`, per-episode metrics and videos under `videos/`.
- `inference/index.jsonl`: call index, episode numbers/steps, reset flags and
  actual number of executed actions.
- `inference/call-XXXXXX.npz`: raw policy observation (images, state, language),
  Python/NumPy/Torch CPU/CUDA RNG states, 10 candidate action chunks, Q scores,
  selected 16-step output, and the action chunk supplied to the environment.

NPZ files use numeric/string arrays only; load with `allow_pickle=False`.
The files describe policy inference, not a full simulator-state checkpoint.
A vector autoreset call may have `executed_steps: [0]`: its output was not executed.

## Restore or recompute a recorded output

Activate `scripts/robocasa/environment.sh` with TMPDIR set to a directory in
`/home/nas_main/dohyunlee`. Set PYTHONPATH to put the run's
`source-snapshot/repo` before the working repo.

Run the snapshot's `scripts/robocasa/replay_inference.py` with:

```text
--trace /home/nas_main/dohyunlee/.../inference/call-000000.npz
--report /home/nas_main/dohyunlee/.../replay-report.json
```

Without `--recompute`, it reconstructs the original output exactly on CPU from
saved candidates and Q scores. It writes the reconstructed actions beside the
report as `<report-name>.actions.npz` (override with `--output-actions`).
With `--recompute`, submit it through sbatch on
one GPU using the same venv and NGC 25.04 image. It loads both checkpoints,
restores the recorded random states and numerical settings, and reruns inference
on the saved observation. The report distinguishes bitwise equality from numeric
agreement (absolute tolerance 1e-5). Keep the checkpoint directories available.
GPU/library differences can affect recomputed numeric equality; the original
selected output remains directly recoverable from the recording.

The one-episode validation batch automatically runs GPU recomputation of its
first inference call after video recording and evaluation complete.

A worker submission wrapper is also provided:

```bash
sbatch slurm/robocasa_replay.sbatch \
  /home/nas_main/dohyunlee/.../RUN_ROOT \
  /home/nas_main/dohyunlee/.../RUN_ROOT/results/eval-seed-0/CoffeeSetupMug/inference/call-000000.npz \
  /home/nas_main/dohyunlee/.../replay-report.json
```

Run `snode --json` before replay submission. This requests one own GPU for at most
15 minutes; it may wait while all four own GPUs are evaluating.

## Execute horizon option

The evaluator accepts `--execute_horizon 8` or `--execute-horizon 8` together with
`--action_horizon 16`. The default execute horizon equals the action horizon.
Allowed range is 1 through action_horizon.

The loaded actor generates 16 actions; BoN scores all 16 with the existing
16-step critic. Only the selected candidate's first 8 actions reach the simulator.
Then a new observation triggers a fresh 16-step prediction. The model stays
loaded in GPU memory. The discarded final 8 actions are not reused.

Videos still record every simulator step. Traces retain the full 16-step output,
the 8-step supplied prefix and the actual executed count (possibly shorter at an
episode boundary). Both horizons appear in result config and W&B; aggregation
rejects mismatched execute horizons.

For future batches, add `--execute-horizon 8` to
`scripts/robocasa/submit_bon_evaluations.py` and choose a new output root.
Without `--submit`, this prints a plan only. More frequent inference increases
compute; estimate a new time limit before a production execute8 evaluation.

The submitted baseline below uses a pre-option source snapshot and executes all
16 actions. The execute8 option was verified with a CPU environment test proving
observations arrive at steps 0, 8 and 16, unused action tails are discarded, and
a partial final chunk executes only its remaining 2 steps. No additional execute8
600-episode campaign was submitted.

## Submitted baseline (2026-09-19 UTC)

Root: `output/robocasa-bon/seed42-bon10-eval012-50ep-video-inputs-20260919`

| Task | Eval seed 0 | Eval seed 1 | Eval seed 2 |
|---|---|---|---|
| CoffeeSetupMug | 137473 | 137477 | 137481 |
| PnPMicrowaveToCounter | 137474 | 137478 | 137482 |
| TurnOffStove | 137475 | 137479 | 137489 |
| PnPCounterToMicrowave | 137476 | 137480 | 137490 |

CPU result aggregation: 137491, after all evaluations finish. Each GPU job has a
90-minute cancel boundary. The single recorded Coffee smoke took 52.7 seconds
for one 600-step rollout and 111.2 seconds including setup. Per-job runtime for
50 episodes is roughly 45–70 minutes, allowing scene/reset variability; queue
waiting is additional. The smoke recorded 601 video frames and 38 policy calls
(about 10.7 MB of input/candidate archives). All five action fields recomputed
bit-for-bit identically from the first saved input/RNG snapshot on a GPU worker.
Validation job: 137450.
