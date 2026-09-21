# GR00T DEAS / IQL research code

Combined working snapshots for RoboCasa experiments with GR00T 1.5 and GR00T N1.7.

## Components and current status

| Directory | Functionality |
| --- | --- |
| `deas-gr00t15/` | Original DEAS HL-Gauss critic; scalar chunk IQL training; BC and best-of-N evaluation; SVF research code; inference trace replay; success/failure videos with Q curves |
| `gr00t17/` | N1.7 policy and native training loader; attached scalar IQL Q/V critic and optional scalar soft-value head |

The N1.5 scalar IQL path has been used for training and RoboCasa BoN evaluation.
The N1.7 critic attachment has passed a short forward/backward and checkpoint smoke test; this is not a claim of a completed N1.7 critic training/evaluation pipeline.
**N1.7 HL-Gauss integration is not implemented.** The attached N1.7 soft-value head uses MSE. SVF components are research code; availability does not imply end-to-end validation of every configuration.

IQL uses scalar twin Q heads and an expectile V objective, with complete action-chunk masking. The existing DEAS HL-Gauss implementation remains available separately.

## Setup

These are two separate Python projects, both exposing a package named `gr00t`.
**Use a separate environment for each; do not install both in one environment.**

- [GR00T 1.5 / DEAS setup and usage](deas-gr00t15/README.md)
- [GR00T N1.7 setup and usage](gr00t17/README.md)
- [N1.7 critic attachment](gr00t17/docs/n17_iql_critic.md)

Model weights, datasets, experiment outputs, inference traces, videos, and personal environments are not included. Obtain their dependencies and weights using the instructions and applicable licenses in each project. Some scripts/configurations retain environment-specific NAS paths and Slurm settings from the research workspace; change those to your own paths and cluster resources before running. This snapshot is not a single portable launcher.

## Useful N1.5 entry points

- `scripts/train_chunk_iql.py`: chunk IQL critic training
- `scripts/eval_policy_robocasa.py`: policy and BoN evaluation
- `scripts/robocasa/replay_inference.py`: replay saved inference inputs/RNG
- `scripts/robocasa/render_q_comparison.py`: render success/failure comparisons with saved Q values

For example, from the N1.5 environment:

```bash
python deas-gr00t15/scripts/robocasa/render_q_comparison.py \
  --task-dir /path/to/evaluation/results/bc1/CoffeeSetupMug \
  --output /path/to/coffee-comparison.mp4 \
  --title Coffee --pair-index 0
```

The renderer needs NumPy, OpenCV and FFmpeg. It reads existing videos and inference traces, uses no GPU, and does not rerun the model. Left: success; right: failure. The Q scale is shared. It currently supports a single environment, deterministic argmax BoN, and one recorded frame per simulator step plus the reset frame.

## Provenance and licensing

This repository packages working-tree snapshots, including local research changes. Source commit IDs are in [PROVENANCE.json](PROVENANCE.json). The original upstream Git histories are not bundled. Original license and attribution files are preserved in each project:

- [DEAS / GR00T 1.5 license](deas-gr00t15/LICENSE)
- [GR00T N1.7 license](gr00t17/LICENSE)

The component licenses govern their respective files; this repository does not replace them with a blanket license. Public source availability does not include model or dataset redistribution rights.

## Maintaining both versions

The sibling development workspaces are synchronized into this repository with
`python tools/sync_workspace.py --apply` (omit `--apply` to preview changes).
Review and commit the resulting changes here. Runtime environments and running
jobs remain in their existing locations. This is an explicit sync workflow,
not a background synchronizer; source deletions require manual review.

Latest additions include policy-TD SVF from BC2 with DiT LoRA, shared Monte Carlo
samples for temperature and soft-value targets, SDE early termination once all
paths reach t=1, configurable training microbatch size, critic episode caching,
and the N1.7 W&B run-name fix. The policy-TD trainer is
`deas-gr00t15/scripts/train_policy_td_svf.py`.
