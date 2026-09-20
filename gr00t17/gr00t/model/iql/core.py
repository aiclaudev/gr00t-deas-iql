"""Scalar IQL objectives; episode validity is independent of bootstrapping."""
from copy import deepcopy
import torch
from torch import nn


def chunk_fields(rewards, boundaries, terminated, start, horizon, discount):
    """Return QC validity, discounted return, and a separate bootstrap mask.

    An end transition itself is valid; transitions after it are invalid. An
    unknown/timeout final transition lacking its true next observation is
    excluded rather than bootstrapping from a clipped duplicate state.
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    boundaries = torch.as_tensor(boundaries, dtype=torch.bool)
    terminated = torch.as_tensor(terminated, dtype=torch.bool)
    if not (rewards.ndim == 1 and rewards.shape == boundaries.shape == terminated.shape):
        raise ValueError('Expected equal one-dimensional transition columns')
    if not 0 <= start < len(rewards) or horizon < 1:
        raise ValueError('Invalid start/horizon')
    if not 0 < discount <= 1:
        raise ValueError('Invalid discount')
    valid = torch.zeros(horizon, dtype=torch.bool)
    total = torch.tensor(0.0)
    alive = True
    terminal = False
    for j in range(horizon):
        i = start + j
        if alive and i < len(rewards):
            valid[j] = True
            total += discount ** j * rewards[i]
            terminal = terminal or bool(terminated[i])
            alive = not bool(boundaries[i] or terminated[i])
    full = bool(valid[-1])
    has_next = start + horizon < len(rewards)
    learn = full and (terminal or has_next)
    return dict(valid=valid, chunk_valid=float(learn), chunk_return=float(total),
                bootstrap_mask=float(not terminal), next_index=min(start + horizon, len(rewards)-1))


def masked_mean(values, valid):
    valid = valid.to(values)
    # Mask before reduction: invalid NaNs must not contaminate the objective.
    return torch.where(valid.bool(), values, torch.zeros_like(values)).sum() / valid.sum().clamp_min(1)


def expectile_loss(diff, expectile, valid):
    if not 0 < expectile < 1:
        raise ValueError('expectile must be in (0,1)')
    weights = torch.where(diff > 0, expectile, 1 - expectile)
    return masked_mean(weights * diff.square(), valid)


def td_target(chunk_return, bootstrap_mask, next_value, discount, horizon):
    return chunk_return + discount ** horizon * torch.where(
        bootstrap_mask.bool(), next_value, torch.zeros_like(next_value))


class ScalarIQL(nn.Module):
    """Original DEAS projection and Q/V hidden networks, with scalar outputs."""
    def __init__(self, feature_dim=64, state_dim=64, action_dim=32,
                 horizon=16, hidden=512, depth=4, value_hidden=256):
        super().__init__()
        from .networks import DoubleCritic, Value
        self.horizon = horizon
        self.critic = DoubleCritic(feature_dim + state_dim + horizon * action_dim,
                                   [hidden] * depth, output_dim=1)
        self.value = Value(feature_dim + state_dim, value_hidden, depth, output_dim=1)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)

    @property
    def q1(self): return self.critic.Q1
    @property
    def q2(self): return self.critic.Q2
    @property
    def target_q1(self): return self.target_critic.Q1

    def q(self, features, states, actions, target=False):
        network = self.target_critic if target else self.critic
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16, enabled=features.is_cuda):
            q1, q2 = network(features.float(), states.float(), actions[:, :self.horizon].float())
        return q1.float(), q2.float()

    def v(self, features, states):
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16, enabled=features.is_cuda):
            value = self.value(features.float(), states.float())
        return value.float()

    @torch.no_grad()
    def update_target(self, tau):
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.lerp_(p, tau)
