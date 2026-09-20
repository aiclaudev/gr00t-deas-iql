# SVF: 10,000-step actor LR schedule, pause at 5,000

## Confirmed preparation settings

- Actor: initialize from the existing BC2 checkpoint.
- Fixed reference policy: frozen BC2 copy.
- Teacher Q: frozen trained DEAS critic.
- Soft value: MSE, initialized with `critic-trunk`.
- Copy the Q hidden trunk (including the corresponding normalization layers).
  Keep the new scalar readout initialization; the distributional Q output has
  101 logits and is not copied into a single MSE output.
- Initialize new first-layer time-input weights to zero; they remain trainable.
  This does not zero every layer in the value network.
- Actor initial LR: `1e-5`; cosine decay over **10,000 optimizer updates**.
- Warmup: 0; final actor LR ratio: 0.
- Soft-value LR: fixed `3e-4` throughout.
- First stage: stop after **5,000 additional optimizer updates** and save.
- Sweep dimensions: only kappa and g. Candidate values will be specified by the user.
- This change prepares training. No GPU run or Slurm submission was performed.

Preparation config:
[/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/configs/svf_joint_seed42_staged.json](/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/configs/svf_joint_seed42_staged.json)

The config retains the earlier baseline kappa=0.4 and g=0.25 as placeholders,
not a new sweep grid. It also retains the existing four-GPU launch template.
The proposed one-GPU parallel sweep still needs its launcher/resource changes
and SVF-specific runtime validation after the final sweep plan is specified.

## Schedule and checkpoint meaning

`--steps 10000` determines both the final target and the LR horizon.
`--stop-after-steps 5000` only caps this invocation; it does not shorten the
schedule or enter the immutable resume identity.

For completed updates n, the actor LR used by the *next* update is:

```text
actor_lr_next(n) = 1e-5 * (1 + cos(pi * n / 10000)) / 2
soft_value_lr(n) = 3e-4
```

- Update 1 uses actor LR `1e-5`.
- After update 5,000, the checkpoint contains next-update actor LR `5e-6`.
- Resume from checkpoint-5000 starts at update 5,001 with that LR.
- After update 10,000, the next-update actor LR is zero and training is complete.

Logging separates the LR actually applied by an update (`actor_lr`, `value_lr`)
from the LR for the next update (`actor_lr_next_update`, `value_lr_next_update`).
`eta_seconds` refers to the full target; `stage_eta_seconds` refers to the
current invocation's stopping point.

## Dry-run commands

The commands below only print submission commands; neither contains `--submit`.

```bash
cd /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T

# First stage: full schedule stays at 10,000; stop and checkpoint at 5,000.
/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python \
  scripts/submit_svf.py --config configs/svf_joint_seed42_staged.json

# Continue a selected trial from its own checkpoint to 10,000.
# Replace the example run path with that trial's actual run root.
/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python \
  scripts/submit_svf.py --config configs/svf_joint_seed42_staged.json \
  --run-root /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/svf-joint/SELECTED_RUN \
  --resume /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/svf-joint/SELECTED_RUN/train/checkpoint-5000
```

The normal staged run preserves W&B logging. An explicit `--preflight-steps`
overrides the config's stage cap, stays limited to 1–100 updates, and disables
W&B. The two explicit CLI stop options cannot be combined.

## Resume guarantees and limits

Each checkpoint stores actor weights/adapters, soft-value weights, AdamW state,
LR scheduler state, completed update/data positions, per-rank RNG state, and
W&B run ID. Checkpoints also contain an exported actor for later evaluation.

Resume requires the same total steps, schedule, kappa/g, loss/initialization,
teachers, batch layout, GPU world size, and other learning settings. Changing
those would define a different continuation; the loader rejects the mismatch.
Each kappa/g candidate resumes from its own checkpoint with its original values.

Legacy constant-LR checkpoints can still resume with the constant schedule.
A cosine run requires saved scheduler state and cannot silently restart warmup
or decay from zero. Saving a scheduled checkpoint without its scheduler is
rejected. Loading checks scheduler step and next LR against the full horizon.

CPU regression tests cover an actual 10,000-update sequence against a
5,000-update save plus 5,000-update restore: all LR values, final parameters,
and Adam states match exactly. These are small-model CPU tests; full SVF GPU
execution and cross-device bitwise determinism are not established by them.
