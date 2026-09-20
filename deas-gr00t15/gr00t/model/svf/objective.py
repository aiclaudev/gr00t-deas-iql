"""SVF objectives with frozen BC and Q teachers and a trainable soft value.

Reference velocities receive [K*B,H,D] in K-block order during Monte Carlo
rollouts. teacher_score receives [K,B,H,D] and returns [K,B]. Padding evolves
in the reference latent exactly as in BC; only valid final coordinates are
clipped. The teacher adapter ignores padding, and soft-value inputs, actor
losses and guidance explicitly mask it.
"""

from dataclasses import dataclass
import math
from typing import Callable, Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class SVFConfig:
    kappa: float = 0.4
    g: float = 0.25
    K: int = 8
    t_min: float = 0.1
    flow_steps: int = 10
    guidance_clip: Optional[float] = 2.0
    lambda_epsilon: float = 1e-3

    def __post_init__(self):
        for name in ("kappa", "g", "lambda_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.K < 2 or self.flow_steps < 1:
            raise ValueError("K must be at least two and flow_steps must be positive")
        if not 0 < self.t_min < 1:
            raise ValueError("t_min must lie strictly between zero and one")
        if self.guidance_clip is not None and (not math.isfinite(self.guidance_clip) or self.guidance_clip <= 0):
            raise ValueError("guidance_clip must be positive or None")

    @property
    def c(self):
        """lambda = c * Q spread; g fixes kappa**2 / c."""
        return self.kappa ** 2 / self.g


def _mask_like(action_mask, actions):
    mask = torch.as_tensor(action_mask, device=actions.device, dtype=actions.dtype)
    try:
        return torch.broadcast_to(mask, actions.shape)
    except RuntimeError as exc:
        raise ValueError("action_mask must broadcast to [B,H,D]") from exc


def _batch_time(t, actions):
    if t.numel() != actions.shape[0]:
        raise ValueError("one time value is required for each batch element")
    return t.reshape(-1).to(device=actions.device, dtype=torch.float32)


@torch.no_grad()
def reference_endpoints(velocity_fn: Callable, x_t, t, action_mask,
                        config: SVFConfig, generator=None):
    """K vectorized Euler-Maruyama BC continuations, using FMRL's clip span."""
    if x_t.ndim != 3:
        raise ValueError("x_t must have shape [B,H,D]")
    batch, horizon, dim = x_t.shape
    mask = _mask_like(action_mask, x_t).bool()
    x = x_t.float().unsqueeze(0).expand(config.K, -1, -1, -1).clone()
    x = x.reshape(config.K * batch, horizon, dim)
    s = _batch_time(t, x_t).repeat(config.K)
    for _ in range(config.flow_steps):
        ds = (1.0 - s).clamp(min=0.0, max=1.0 / config.flow_steps)
        safe_s = s.clamp_min(config.t_min)
        velocity = velocity_fn(x, s).float()
        if velocity.shape != x.shape:
            raise ValueError("reference velocity returned a different action shape")
        sb, db = safe_s[:, None, None], ds[:, None, None]
        drift = velocity - config.kappa ** 2 * (x - sb * velocity) / sb
        noise_scale = config.kappa * (2.0 * (1.0 - sb).clamp_min(0.0) / sb * db).sqrt()
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = x + drift * db + noise_scale * noise
        s = (s + ds).clamp_max(1.0)
    x = x.reshape(config.K, batch, horizon, dim)
    return torch.where(mask.unsqueeze(0), x.clamp(-1.0, 1.0), x).detach()


def estimate_lambda(qs, config: SVFConfig, valid=None):
    """Independent MC Q draws determine one detached temperature per update."""
    if qs.ndim != 2 or qs.shape[0] < 2:
        raise ValueError("qs must have shape [K>=2,B]")
    spread = qs.detach().float().std(dim=0, correction=0)
    weights = torch.ones_like(spread) if valid is None else valid.to(spread)
    statistics = torch.stack(((spread * weights).sum(), weights.sum()))
    if dist.is_available() and dist.is_initialized():
        # One shared temperature per distributed microstep. This pools states
        # across ranks, not across later gradient-accumulation microsteps.
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    mean_spread = statistics[0] / statistics[1].clamp_min(1.0)
    return (config.c * mean_spread.clamp_min(config.lambda_epsilon)).detach()


def soft_value_target(qs, lam):
    """Stable lambda*logmeanexp(Q/lambda); the whole target is detached."""
    if qs.ndim != 2 or qs.shape[0] == 0:
        raise ValueError("qs must have shape [K,B]")
    qs = qs.detach().float()
    lam = torch.as_tensor(lam, device=qs.device, dtype=qs.dtype).detach()
    return (lam * (torch.logsumexp(qs / lam, dim=0) - math.log(qs.shape[0]))).detach()


def actor_mse(prediction, target, action_mask):
    """Mean squared velocity error over valid action coordinates only."""
    mask = _mask_like(action_mask, prediction)
    return ((prediction.float() - target.float()).square() * mask).sum() / mask.sum().clamp_min(1.0)


def guidance_from_value(value_fn, x_t, t, base_velocity, action_mask,
                        config: SVFConfig, lam):
    """Differentiate V only w.r.t. x, then detach before actor regression.

    No V parameter .grad is populated by this operation and no second-order
    graph is retained. value_fn maps masked x to [B,2] (or [B]).
    """
    mask = _mask_like(action_mask, x_t).float()
    with torch.enable_grad():
        x = x_t.detach().float().requires_grad_(True)
        values = value_fn(x * mask)
        if values.ndim == 2:
            values = values.mean(dim=-1)
        if values.shape != (x.shape[0],):
            raise ValueError("value_fn must return [B] or [B,heads]")
        grad = torch.autograd.grad(values.sum(), x, create_graph=False)[0].detach() * mask
    t = _batch_time(t, x_t)
    safe_t = t.clamp_min(config.t_min)
    lam = torch.as_tensor(lam, device=x.device, dtype=x.dtype).detach()
    coef = torch.where(t >= config.t_min,
                       config.kappa ** 2 * (1.0 - safe_t) / (safe_t * lam),
                       torch.zeros_like(t))
    guidance = coef[:, None, None] * grad
    valid = mask.flatten(1).sum(-1) > 0
    weights = valid.to(x.dtype)
    count = weights.sum().clamp_min(1.0)
    base_norm = (base_velocity.detach().float() * mask).flatten(1).norm(dim=-1)
    mean_base_norm = (base_norm * weights).sum() / count
    guidance_norm = guidance.flatten(1).norm(dim=-1)
    clip_fraction = torch.zeros((), device=x.device)
    if config.guidance_clip is not None:
        cap = config.guidance_clip * mean_base_norm
        scales = (cap / guidance_norm.clamp_min(1e-8)).clamp_max(1.0)
        clip_fraction = (((scales < 1.0) & valid).float()).sum() / count
        guidance = guidance * scales[:, None, None]
    mean_norm = (guidance.flatten(1).norm(dim=-1) * weights).sum() / count
    return guidance.detach(), {
        "guidance_norm": mean_norm.detach(),
        "base_velocity_norm": mean_base_norm.detach(),
        "guidance_ratio": (mean_norm / mean_base_norm.clamp_min(1e-8)).detach(),
        "gradient_norm": ((grad.flatten(1).norm(dim=-1) * weights).sum() / count).detach(),
        "clip_fraction": clip_fraction.detach(),
        "coefficient_mean": ((coef * weights).sum() / count).detach(),
    }


def joint_losses(soft_value, velocity_fn, reference_velocity_fn, teacher_score,
                 features, states, actions, action_mask, config: SVFConfig,
                 generator=None):
    """Joint V regression and actor regression against frozen BC/Q teachers.

    The lambda estimator and V target use independent noise, times and SDE
    paths. Actor anchors are independent and cover [0,1), including pure BC
    targets below t_min. Conditioning passed to V is always detached.
    """
    actions = actions.detach().float()
    mask = _mask_like(action_mask, actions).float()
    batch = actions.shape[0]
    device = actions.device
    valid = mask.flatten(1).sum(-1) > 0
    features, states = features.detach(), states.detach()

    def anchor(t_min):
        noise = torch.randn(actions.shape, device=device, dtype=torch.float32, generator=generator)
        t = torch.rand(batch, device=device, generator=generator) * (1.0 - t_min) + t_min
        x = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
        return x, t, actions - noise

    with torch.no_grad():
        lambda_x, lambda_t, _ = anchor(config.t_min)
        lambda_ends = reference_endpoints(reference_velocity_fn, lambda_x, lambda_t,
                                         mask, config, generator)
        lambda_qs = teacher_score(lambda_ends).detach().float()
        if lambda_qs.shape != (config.K, batch):
            raise ValueError("teacher_score must return [K,B]")
        lam = estimate_lambda(lambda_qs, config, valid)
        value_x, value_t, _ = anchor(config.t_min)
        ends = reference_endpoints(reference_velocity_fn, value_x, value_t,
                                   mask, config, generator)
        qs = teacher_score(ends).detach().float()
        if qs.shape != (config.K, batch):
            raise ValueError("teacher_score must return [K,B]")
        target = soft_value_target(qs, lam)
    value_loss_type = getattr(soft_value, "loss_type", "mse")
    if value_loss_type == "hl-gauss":
        logits = soft_value.forward_logits(features, states, value_x * mask, value_t)
        predicted_values = soft_value.values_from_logits(logits)
        # Invalid padded examples contribute neither loss nor diagnostics. Use a
        # harmless in-support target for them before the strict projection check.
        projected_targets = torch.where(valid, target, torch.full_like(target, soft_value.value_min))
        per_head_loss = soft_value.loss_from_logits(logits, projected_targets)
    elif value_loss_type == "mse":
        predicted_values = soft_value(features, states, value_x * mask, value_t)
        per_head_loss = (predicted_values.float() - target[:, None]).square()
    else:
        raise ValueError("Unsupported soft-value loss type")
    if predicted_values.shape != (batch, 2) or per_head_loss.shape != (batch, 2):
        raise ValueError("soft_value must return exactly two heads [B,2]")
    count = valid.sum().clamp_min(1)
    value_loss = (per_head_loss.mean(-1) * valid).sum() / count
    raw_residual = predicted_values.detach().float() - target[:, None]
    value_diagnostics = {
        "value_target_mae": (raw_residual.abs().mean(-1) * valid).sum() / count,
        "value_target_rmse": ((raw_residual.square().mean(-1) * valid).sum() / count).sqrt(),
    }
    if value_loss_type == "hl-gauss":
        with torch.no_grad():
            probabilities = logits.detach().float().softmax(-1)
            log_probabilities = logits.detach().float().log_softmax(-1)
            projected = soft_value.target_probabilities(projected_targets)
            centers = soft_value.bin_centers.float().to(projected.device)
            projection_bias = (projected * centers).sum(-1) - target
            value_diagnostics.update({
                "value_prediction_entropy": (-(probabilities * log_probabilities).sum(-1).mean(-1) * valid).sum() / count,
                "value_target_entropy": (-(projected * projected.clamp_min(1e-30).log()).sum(-1) * valid).sum() / count,
                "value_projection_bias": (projection_bias * valid).sum() / count,
                "value_projection_mae": (projection_bias.abs() * valid).sum() / count,
            })

    actor_x, actor_t, base_velocity = anchor(0.0)
    guidance, info = guidance_from_value(
        lambda x: soft_value(features, states, x, actor_t), actor_x, actor_t,
        base_velocity, mask, config, lam)
    predicted_velocity = velocity_fn(actor_x, actor_t)
    actor_loss = actor_mse(predicted_velocity, (base_velocity + guidance).detach(), mask)
    loss = value_loss + actor_loss
    info.update({key: value.detach() for key, value in value_diagnostics.items()})
    info.update({
        "total_loss": loss.detach(), "soft_value_loss": value_loss.detach(),
        "actor_loss": actor_loss.detach(), "lambda": lam,
        "q_spread": qs.std(dim=0, correction=0).mean().detach(),
        "lambda_q_spread": lambda_qs.std(dim=0, correction=0).mean().detach(),
        "weight_max": (qs / lam).softmax(dim=0).amax(dim=0).mean().detach(),
        "value_target_mean": target.mean().detach(),
        "value_prediction_mean": predicted_values.mean().detach(),
        "valid_samples": valid.sum().detach(),
    })
    return loss, info
