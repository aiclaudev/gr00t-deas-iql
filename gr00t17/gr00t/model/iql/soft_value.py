"""Small time-conditioned soft values for the frozen GR00T/DEAS teachers."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def fourier_time_embedding(t: torch.Tensor, dim: int = 64, max_frequency: float = 256.0):
    """FMRL time features: log-spaced angular frequencies, with no 2*pi."""
    if dim <= 0 or dim % 2:
        raise ValueError("time embedding dimension must be a positive even number")
    t = t.reshape(-1).float()
    frequencies = torch.exp(torch.linspace(0.0, math.log(max_frequency), dim // 2,
                                           device=t.device, dtype=torch.float32))
    angles = t[:, None] * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class DoubleSoftValue(nn.Module):
    """Two independent MLPs predicting V(s, x_t, t), one scalar per head.

    Callers mask padded action coordinates before passing x_t. Both losses use
    the same Monte Carlo scalar target. MSE predicts scalars directly; HL-Gauss
    predicts categorical logits and exposes their differentiable expectations.
    """

    def __init__(self, feature_dim=64, state_dim=64, action_horizon=16,
                 action_dim=32, time_dim=64, hidden_dim=512, depth=4,
                 loss_type="mse", num_bins=101, value_min=-100.0,
                 value_max=0.0, sigma=None):
        super().__init__()
        if time_dim <= 0 or time_dim % 2 or depth < 1:
            raise ValueError("time_dim must be positive/even and depth must be positive")
        if loss_type not in ("mse", "hl-gauss"):
            raise ValueError("loss_type must be 'mse' or 'hl-gauss'")
        self.loss_type = loss_type
        if loss_type == "hl-gauss":
            if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
                raise ValueError("num_bins must be an integer >= 2")
            if not math.isfinite(value_min) or not math.isfinite(value_max) or value_min >= value_max:
                raise ValueError("value_min and value_max must be finite and ordered")
            self.num_bins = num_bins
            self.value_min, self.value_max = float(value_min), float(value_max)
            self.sigma = float(sigma) if sigma is not None else .75 * (value_max - value_min) / num_bins
            if not math.isfinite(self.sigma) or self.sigma <= 0:
                raise ValueError("sigma must be finite and positive")
            edges = torch.linspace(value_min, value_max, num_bins + 1, dtype=torch.float32)
            sigma_tensor = torch.tensor(self.sigma, dtype=torch.float32)
            if not torch.isfinite(edges).all() or not (edges[1:] > edges[:-1]).all():
                raise ValueError("Support edges must be distinct and finite in FP32")
            if not torch.isfinite(sigma_tensor) or sigma_tensor <= 0:
                raise ValueError("sigma must be positive and finite in FP32")
            self.register_buffer("support_edges", edges)
            self.register_buffer("bin_centers", edges[:-1] / 2 + edges[1:] / 2)
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.time_dim = time_dim
        input_dim = feature_dim + state_dim + action_horizon * action_dim + time_dim
        heads = []
        for _ in range(2):
            layers = []
            for layer in range(depth):
                layers.extend((nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                               nn.LayerNorm(hidden_dim), nn.GELU()))
            layers.append(nn.Linear(hidden_dim, 1 if loss_type == "mse" else num_bins))
            heads.append(nn.Sequential(*layers))
        self.heads = nn.ModuleList(heads)

    @staticmethod
    def _conditioning(value, batch, width, name):
        if value.shape == (batch, 1, width):
            value = value[:, 0]
        if value.shape != (batch, width):
            raise ValueError(f"{name} must have shape [B,{width}] or [B,1,{width}], got {tuple(value.shape)}")
        return value

    def forward_logits(self, features, states, x_t, t):
        if x_t.ndim != 3 or x_t.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("x_t must have shape [B, action_horizon, action_dim]")
        batch = x_t.shape[0]
        features = self._conditioning(features, batch, self.feature_dim, "features")
        states = self._conditioning(states, batch, self.state_dim, "states")
        if t.numel() != batch:
            raise ValueError("t must contain one scalar per batch item")
        inputs = torch.cat((features, states, x_t.flatten(1),
                            fourier_time_embedding(t, self.time_dim)), dim=-1)
        inputs = inputs.to(dtype=self.heads[0][0].weight.dtype)
        predictions = [head(inputs) for head in self.heads]
        return (torch.cat(predictions, dim=-1) if self.loss_type == "mse"
                else torch.stack(predictions, dim=1))

    def values_from_logits(self, logits):
        """Return [B,2] values; HL-Gauss uses softmax expectation, never argmax."""
        if self.loss_type == "mse":
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("MSE predictions must have shape [B,2]")
            return logits
        if logits.ndim != 3 or logits.shape[1:] != (2, self.num_bins):
            raise ValueError("HL-Gauss logits must have shape [B,2,num_bins]")
        return (logits.float().softmax(dim=-1) * self.bin_centers.float().to(logits.device)).sum(-1)

    def forward(self, features, states, x_t, t):
        return self.values_from_logits(self.forward_logits(features, states, x_t, t))

    @torch.no_grad()
    def target_probabilities(self, target):
        """Project scalar [B] targets to normalized truncated Gaussian bins.

        Tail differences use erfc to avoid cancellation near 1 in the Gaussian
        CDF. This is the same histogram construction as the frozen DEAS teacher,
        with explicit finite/support checks and FP32 normalization.
        """
        if self.loss_type != "hl-gauss":
            raise ValueError("target_probabilities requires hl-gauss loss")
        target = torch.as_tensor(target).detach().float()
        if target.ndim != 1 or not torch.isfinite(target).all():
            raise ValueError("HL-Gauss target must be a finite vector [B]")
        edges = self.support_edges.to(device=target.device, dtype=torch.float32)
        if ((target < edges[0]) | (target > edges[-1])).any():
            raise ValueError("HL-Gauss target is outside the configured value support")
        # Divide before sqrt(2) to avoid overflow when a finite sigma is large.
        standardized = ((edges[None, :] - target[:, None]) / self.sigma) / math.sqrt(2.0)
        lower, upper = standardized[:, :-1], standardized[:, 1:]
        positive_mass = .5 * (torch.erfc(lower) - torch.erfc(upper))
        negative_mass = .5 * (torch.erfc(-upper) - torch.erfc(-lower))
        crossing_mass = .5 * (torch.erf(upper) - torch.erf(lower))
        mass = torch.where(lower >= 0, positive_mass,
                           torch.where(upper <= 0, negative_mass, crossing_mass)).clamp_min(0)
        normalizer = mass.sum(-1, keepdim=True)
        if not torch.isfinite(mass).all() or not torch.isfinite(normalizer).all() or (normalizer <= 0).any():
            raise ValueError("HL-Gauss projection cannot be normalized in FP32")
        return mass / normalizer

    def loss_from_logits(self, logits, target):
        """Return unreduced [B,2] losses; targets never receive gradients."""
        values = self.values_from_logits(logits)
        target = torch.as_tensor(target, device=logits.device).detach().float()
        if target.ndim != 1 or target.shape[0] != values.shape[0] or not torch.isfinite(target).all():
            raise ValueError("Soft-value target must be finite with shape [B]")
        if self.loss_type == "mse":
            return (values.float() - target[:, None]).square()
        probabilities = self.target_probabilities(target)
        return -(probabilities[:, None, :] * F.log_softmax(logits.float(), dim=-1)).sum(-1)

    def loss_configuration(self):
        if self.loss_type == "mse":
            return {"loss_type": "mse"}
        return {"loss_type": "hl-gauss", "num_bins": self.num_bins,
                "value_min": self.value_min, "value_max": self.value_max,
                "sigma": self.sigma}
