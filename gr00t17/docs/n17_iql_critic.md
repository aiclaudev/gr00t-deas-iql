# GR00T N1.7 scalar IQL critic

`gr00t.model.iql.critic.N17IQLCritic` attaches the existing DEAS-shaped critic to
N1.7 checkpoint features. The production BC1/BC2 scripts are unchanged.

## Architecture

- Freeze the checkpoint's backbone, VLLN and VL self-attention in eval mode.
- Apply the transferred feature processing once and mean-pool the 2048 channels.
- Train the same category-specific 2048 → 1024 → 1024 → 1024 → 64 tanh projection.
- Q1/Q2: four 512-wide Linear/LayerNorm/GELU hidden layers, scalar outputs.
- V: 256-wide BRONet with four residual blocks, scalar output.
- State/action use the N1.7 processor's native 132-dimensional padded layout.
  Thus Q's input is 64 + 132 + 16*132 = 2308; V's input is 64 + 132 = 196.
  Only these input dimensions differ from the N1.5 critic. Hidden widths are unchanged.
- The critic evaluates the first 16 actions of the actor's native 40-slot output.
- BF16 autocast for projection/Q/V matrix products; FP32 master weights, optimizer,
  tanh, TD targets and losses. The frozen backbone loads in BF16.

## Training interface

```python
from gr00t.model.iql.critic import N17IQLCritic

critic = N17IQLCritic.from_actor_checkpoint(bc2_checkpoint, horizon=16).cuda().train()
# current and next_inputs are N1.7 processor.collator(samples)['inputs'].
result = critic(current, next_inputs, chunk_return, bootstrap_mask, chunk_valid,
                discount=0.99, expectile=0.7)
optimizer.zero_grad(set_to_none=True)
result['loss'].backward()
optimizer.step()
critic.head.update_target(0.005)
```

Use the **same actor checkpoint processor and normalization**, including its
embodiment IDs, state ordering and action ordering. N1.5's processed tensors are
not compatible. Training data must supply observation t, observation t+16,
normalized actions t:t+16 and the QC return/masks. `core.chunk_fields` implements
complete-chunk validity separately from terminal bootstrapping, as in the 1.5
IQL experiment. Preserve the original success-last15 expansion and reward-1
preprocessing when constructing returns; infer terminal flags from unexpanded
rewards. Exclude unknown final boundaries without a true next observation.

This addition supplies the model/loss/scoring API, not a production N1.7 paired
RL dataset/training launcher. A production sampler still needs to pair frames,
preserve current/next augmentation consistency, and supply the QC fields.

## Scoring and checkpoints

`critic.score_actions(inputs)` returns min(Q1,Q2). Alternatively pass normalized
candidate `actions` and their `action_mask`; batch observations must correspond
to the candidates. The caller handles generating and expanding BoN candidates.

`critic.save_head(directory)` saves projection, Q/V, target-Q and the absolute
actor checkpoint reference. `N17IQLCritic.load_head(directory)` reconstructs it.
The actor checkpoint must remain accessible; this is not a self-contained copy
of the backbone or an optimizer-resume checkpoint. Actor-policy weights are not
modified, and no critic is automatically inserted into an already running BC job.

## Validation

CPU regression cases: frozen feature/target gradients, terminal target,
action padding and 16-vs-40 horizon, and QC episode boundary validity.
`scripts/smoke_n17_iql.py` loads the actual actor checkpoint and two real paired
RoboCasa samples, performs one update, checks frozen gradients, and verifies a
critic-head save/load round trip. It does not measure throughput or submit jobs.
The actor's processor is in eval mode for this deterministic correctness check.

## Optional SVF soft value

```python
critic.enable_soft_value()  # Call before constructing the soft-value optimizer.
soft_optimizer = torch.optim.Adam(critic.head.soft_value.parameters(), lr=3e-4)
result = critic.soft_value_loss(inputs, x_t, t, teacher_qs, temperature,
                               chunk_valid=chunk_valid)
soft_optimizer.zero_grad(set_to_none=True)
result['loss'].backward()
soft_optimizer.step()
```

This adds the existing SVF `DoubleSoftValue` implementation independently of
IQL's V(s). Two 512x4 LayerNorm/GELU MLPs receive 64-dimensional projected
features, native N1.7 state, the first16 normalized noisy actions, and a
64-dimensional Fourier time embedding. Each predicts one scalar V(s,x_t,t).
The attached variant uses **MSE**, not HL-Gauss. Its scalar target is
`lambda * (logsumexp(teacher_qs/lambda, dim=0) - log(K))`.

`teacher_qs[K,B]` must be frozen critic scores of frozen BC continuations from
that same x_t,t. The continuation sampler and guided actor update are **not**
implemented by this attachment. SVF lambda is a separate temperature; it is not
IQL's expectile (0.7). The caller supplies lambda and MC teacher scores.

Soft-head computation and input gradients remain FP32, matching the existing
SVF code. Conditioning and targets are detached, so soft-value regression does
not update the backbone, 64-d projection, or IQL Q/V. Action padding and slots
beyond16 have zero input gradients; invalid chunks are excluded from regression.
`soft_values` retains the gradient to x_t for future guidance use. `save_head`
and `load_head` also save/reconstruct the optional soft head. Existing IQL-only
saved heads continue to load without creating one. No training job is submitted.
