# BC2 vs DEAS BoN50: local evaluation logging

Submitted with `--report-to none`: W&B is disabled. Results and videos stay on NAS.

- BC2 actor: completed step 10000 of 02-bc-rollout.
- DEAS: same BC2 actor plus completed step-10000 Q, BoN N=50, temperature=0.
- Both: 4 tasks × evaluation seeds 0,1,2 × 50 episodes = 600 episodes per method.
- Action/execute horizon 16, denoising steps 4, held-out object split B.
- One GPU / 8 CPU / 96 GiB per task job, sub-own. At most four distinct tasks in flight.
- Per task: BC2 seeds 0→1→2, then BoN50 seeds 0→1→2 (afterany dependencies).
- A failed evaluation is recorded as incomplete; it does not become a zero-success result.
- BC2 time limit 01:30:00; BoN50 time limit 04:00:00. These are cancellation limits, not ETAs.

Run root: `/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/robocasa-comparison/seed42-bc2-vs-bon50-eval012-50ep-20260919`

## Jobs

| Method | Task | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| BC2 | CoffeeSetupMug | 138485 | 138489 | 138493 |
| BC2 | PnPMicrowaveToCounter | 138486 | 138490 | 138494 |
| BC2 | TurnOffStove | 138487 | 138491 | 138495 |
| BC2 | PnPCounterToMicrowave | 138488 | 138492 | 138496 |
| BoN50 | CoffeeSetupMug | 138498 | 138502 | 138506 |
| BoN50 | PnPMicrowaveToCounter | 138499 | 138503 | 138507 |
| BoN50 | TurnOffStove | 138500 | 138504 | 138508 |
| BoN50 | PnPCounterToMicrowave | 138501 | 138505 | 138509 |

BC summary: 138497; BoN summary: 138510; combined summary: 138511.

## Local files

- `<bc2|bon>/results/eval-seed-<seed>/<task>/eval.csv` and `episodes.jsonl`: episode success, length, elapsed rollout time.
- Same directory `result.json`: current/final success rate, counts, mean episode length, walltime, settings, seeds, paths, error if any.
- Same directory `videos/`: per-episode MP4 videos.
- BoN `inference/`: local inputs, candidates, Q scores and selected actions for replay.
- `<bc2|bon>/logs/`: Slurm stdout/stderr.
- `aggregate/summary.json`, `summary.md`, `runs.csv`, `by_task.csv`: final combined comparison after all 24 evaluation jobs terminate.
- `manifest.json`: exact commands, model paths, submission responses, dependencies and job IDs.
