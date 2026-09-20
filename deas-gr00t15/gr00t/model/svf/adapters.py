"""Torch adapters for frozen GR00T/DEAS teachers and a trainable flow actor.

These adapters intentionally do not use Policy.get_action (inference_mode/NumPy),
merge actor and critic weights, or change existing DEAS training code.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


PANDA_STATE_KEYS = (
    "end_effector_position_relative", "end_effector_rotation_relative",
    "gripper_qpos", "base_position", "base_rotation",
)
PANDA_ACTION_KEYS = (
    "end_effector_position", "end_effector_rotation", "gripper_close",
    "base_motion", "control_mode",
)
PANDA_STATE_ROTATIONS = {
    "end_effector_rotation_relative": "rotation_6d", "base_rotation": "rotation_6d",
}


@dataclass(frozen=True)
class HeadContext:
    """Head inputs for one batch; trainable-head contexts belong to one graph.

    Rebuild after every optimizer update. No input BatchFeature is mutated.
    """
    backbone_features: Tensor
    state_features: Tensor
    future_tokens: Tensor
    embodiment_id: Tensor
    attention_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.backbone_features.shape[0]

    def repeat_batches(self, count: int) -> "HeadContext":
        """Repeat entire B-sized batches, matching flattening of [K,B,...]."""
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        def repeat(x):
            return None if x is None else x.repeat(count, *([1] * (x.ndim - 1)))
        return HeadContext(*(repeat(getattr(self, name)) for name in self.__dataclass_fields__))

    def repeat_interleave(self, count: int) -> "HeadContext":
        """Repeat individual examples (distinct from the SVF [K,B] order)."""
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        return HeadContext(*(
            None if (x := getattr(self, name)) is None else x.repeat_interleave(count, dim=0)
            for name in self.__dataclass_fields__
        ))

    def slice_batch(self, start: int, stop: int | None = None) -> "HeadContext":
        return HeadContext(*(
            None if (x := getattr(self, name)) is None else x[start:stop]
            for name in self.__dataclass_fields__
        ))


def prepare_head_context(head: nn.Module, raw_backbone_output: Mapping[str, Tensor],
                         action_input: Mapping[str, Tensor]) -> HeadContext:
    """Apply actor head preprocessing once, retaining projector gradients.

    The caller controls freezing/no_grad. In particular a trainable head needs
    gradients through its VLLN, VL self-attention, state encoder and future tokens.
    """
    parameter = next(head.parameters())
    features = raw_backbone_output["backbone_features"].to(parameter)
    features = head.vl_self_attention(head.vlln(features))
    embodiment_id = action_input["embodiment_id"].to(device=parameter.device)
    state_features = head.state_encoder(action_input["state"].to(parameter), embodiment_id)
    future_tokens = head.future_tokens.weight.unsqueeze(0).expand(features.shape[0], -1, -1)
    return HeadContext(features, state_features, future_tokens, embodiment_id,
                       raw_backbone_output.get("backbone_attention_mask"))


def velocity_from_head(head: nn.Module, context: HeadContext, x_t: Tensor,
                       t: Tensor | float) -> Tensor:
    """Differentiable TRAIN-forward vector field, with t=0 noise and t=1 data.

    Unlike the legacy sampling loop this follows the training forward's cross
    attention mask and return_all_hidden_states=False exactly.
    """
    if x_t.ndim != 3 or x_t.shape[0] != context.batch_size:
        raise ValueError("x_t must be [context batch, horizon, action dimension]")
    if x_t.shape[-1] != head.config.action_dim:
        raise ValueError("x_t action dimension differs from the head configuration")
    times = torch.as_tensor(t, device=x_t.device)
    if times.numel() == 1:
        times = times.reshape(1).expand(x_t.shape[0])
    elif times.numel() == x_t.shape[0]:
        times = times.reshape(x_t.shape[0])
    else:
        raise ValueError("t must be scalar or have one value per batch example")
    time_buckets = (times * head.num_timestep_buckets).long()
    head_actions = x_t.to(context.state_features)
    action_features = head.action_encoder(head_actions, time_buckets, context.embodiment_id)
    if head.config.add_pos_embed:
        positions = torch.arange(x_t.shape[1], device=x_t.device)
        action_features = action_features + head.position_embedding(positions).unsqueeze(0)
    hidden_states = torch.cat((context.state_features, context.future_tokens, action_features), dim=1)
    output = head.model(
        hidden_states=hidden_states,
        encoder_hidden_states=context.backbone_features,
        encoder_attention_mask=context.attention_mask,
        timestep=time_buckets,
        return_all_hidden_states=False,
    )
    return head.action_decoder(output, context.embodiment_id)[:, -x_t.shape[1]:]


def _metadata(value: Mapping[str, Any] | str | Path, embodiment: str) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if "statistics" not in value:
        value = value[embodiment]
    if "statistics" not in value or "modalities" not in value:
        raise ValueError("metadata requires statistics and modalities")
    return value


def _bounds(metadata: Mapping[str, Any], modality: str, key: str,
            target_rotation: str | None = None) -> tuple[list[float], list[float]]:
    spec = metadata["modalities"][modality][key]
    shape = spec["shape"]
    if len(shape) != 1:
        raise ValueError(f"{modality}.{key} must be a vector")
    if target_rotation and spec.get("rotation_type") != target_rotation:
        if not spec.get("absolute") or not spec.get("rotation_type"):
            raise ValueError(f"Unsupported rotation conversion for {modality}.{key}")
        if target_rotation != "rotation_6d":
            raise ValueError("Only the Panda rotation_6d target is supported")
        return [-1.0] * 6, [1.0] * 6
    stats = metadata["statistics"][modality][key]
    low, high = stats["min"], stats["max"]
    if len(low) != shape[0] or len(high) != shape[0]:
        raise ValueError(f"Statistics do not match {modality}.{key} shape")
    if any(not torch.isfinite(torch.tensor(v)).item() for v in [*low, *high]):
        raise ValueError(f"Nonfinite statistics for {modality}.{key}")
    if any(hi < lo for lo, hi in zip(low, high)):
        raise ValueError(f"Reversed min/max for {modality}.{key}")
    return low, high


class ActionCoordinateAdapter(nn.Module):
    """Actor-normalized -> critic-normalized Panda state/action coordinates.

    Layout order follows single_panda_gripper_rl[_inference]. Rotation6d state
    channels use the transform's fixed [-1,1] bounds, not raw quaternion stats.
    min_max does NOT clip in this repository; extrapolation is preserved.
    Padded channels and constant critic-range channels are always zero.
    """
    def __init__(self, state_scale: Tensor, state_bias: Tensor, state_mask: Tensor,
                 action_scale: Tensor, action_bias: Tensor, action_mask: Tensor,
                 binary_mask: Tensor):
        super().__init__()
        for name, value in locals().copy().items():
            if name not in {"self", "__class__"}:
                self.register_buffer(name, value)

    @classmethod
    def from_metadata(cls, actor_metadata, critic_metadata, *, embodiment="new_embodiment",
                      state_keys: Sequence[str] = PANDA_STATE_KEYS,
                      action_keys: Sequence[str] = PANDA_ACTION_KEYS,
                      state_rotations: Mapping[str, str] | None = None,
                      max_state_dim: int = 64, max_action_dim: int = 32):
        actor, critic = _metadata(actor_metadata, embodiment), _metadata(critic_metadata, embodiment)
        rotations = PANDA_STATE_ROTATIONS if state_rotations is None else state_rotations

        def layout(modality, keys, width):
            scales, biases, binary = [], [], []
            for qualified_key in keys:
                key = qualified_key.removeprefix(modality + ".")
                rotation = rotations.get(key, rotations.get(modality + "." + key)) if modality == "state" else None
                amin, amax = _bounds(actor, modality, key, rotation)
                cmin, cmax = _bounds(critic, modality, key, rotation)
                if len(amin) != len(cmin):
                    raise ValueError(f"Actor/critic layout mismatch for {modality}.{key}")
                a_spec = actor["modalities"][modality][key]
                c_spec = critic["modalities"][modality][key]
                if a_spec.get("continuous", True) != c_spec.get("continuous", True):
                    raise ValueError(f"Actor/critic modality mismatch for {modality}.{key}")
                is_binary = not a_spec.get("continuous", True)
                if is_binary and (modality != "action" or key not in {"gripper_close", "control_mode"}):
                    raise ValueError(f"Unsupported noncontinuous coordinate {modality}.{key}")
                for al, ah, cl, ch in zip(amin, amax, cmin, cmax):
                    if is_binary:
                        scales.append(1.0); biases.append(0.0)
                    elif ch == cl:
                        scales.append(0.0); biases.append(0.0)
                    else:
                        scales.append((ah - al) / (ch - cl))
                        biases.append(((ah + al) - (ch + cl)) / (ch - cl))
                    binary.append(is_binary)
            real = len(scales)
            if real > width:
                raise ValueError(f"{modality} layout exceeds configured width")
            pad = width - real
            return (torch.tensor(scales + [0.0] * pad, dtype=torch.float32),
                    torch.tensor(biases + [0.0] * pad, dtype=torch.float32),
                    torch.arange(width) < real,
                    torch.tensor(binary + [False] * pad, dtype=torch.bool))

        ss, sb, sm, _ = layout("state", state_keys, max_state_dim)
        acs, acb, am, bm = layout("action", action_keys, max_action_dim)
        return cls(ss, sb, sm, acs, acb, am, bm)

    @staticmethod
    def _convert(x: Tensor, scale: Tensor, bias: Tensor, mask: Tensor) -> Tensor:
        if not x.is_floating_point() or x.shape[-1] != scale.numel():
            raise ValueError("Coordinate tensor must be floating and match the padded dimension")
        result = x * scale.to(x) + bias.to(x)
        return torch.where(mask.to(x.device), result, torch.zeros_like(result))

    def convert_state(self, states: Tensor) -> Tensor:
        return self._convert(states, self.state_scale, self.state_bias, self.state_mask)

    def convert_action(self, actions: Tensor, *, threshold_binary: bool = True) -> Tensor:
        result = self._convert(actions, self.action_scale, self.action_bias, self.action_mask)
        if threshold_binary:
            result = torch.where(self.binary_mask.to(actions.device), (actions > 0.5).to(actions.dtype), result)
        return result


def _critic_head(critic: nn.Module) -> nn.Module:
    return getattr(critic, "critic_head", critic)


@torch.no_grad()
def prepare_critic_features(critic: nn.Module, raw_backbone_output: Mapping[str, Tensor],
                            embodiment_id: Tensor, *, passes: int = 2) -> Tensor:
    """Reproduce the saved online-Q training feature path, without mutation.

    Existing DEASCritic.forward processes its SAME BatchFeature in value loss
    and then in Q loss. Therefore current online-Q checkpoints require TWO
    VLLN/self-attention passes. Keep this explicit in saved SVF provenance;
    passes=1 is only for a checkpoint trained after that training bug is fixed.
    """
    if not isinstance(passes, int) or passes < 1:
        raise ValueError("passes must be a positive integer")
    head = _critic_head(critic)
    head.eval()
    parameter = next(head.parameters())
    features = raw_backbone_output["backbone_features"].to(parameter)
    embodiment_id = embodiment_id.to(device=parameter.device)
    for _ in range(passes):
        features = head.vl_self_attention(head.vlln(features))
    return torch.tanh(head.backbone_encoder(features.mean(dim=1, keepdim=True), embodiment_id))


def score_critic(critic: nn.Module, features: Tensor, states: Tensor, actions: Tensor) -> Tensor:
    """Online min(Q1,Q2), decoded from categorical HLG logits.

    Inputs must ALREADY be critic-normalized with zero padding. This function
    keeps action gradients when requested; teacher target calls use no_grad.
    """
    head = _critic_head(critic)
    if actions.ndim != 3 or actions.shape[1] < head.critic_action_horizon:
        raise ValueError("Critic requires [B,H,D] actions covering its trained horizon")
    parameter = next(head.critic.parameters())
    q1_logits, q2_logits = head.critic(
        features.to(parameter), states.to(parameter),
        actions[:, :head.critic_action_horizon].to(parameter),
    )
    q1 = head.hlg.transform_from_probs(torch.softmax(q1_logits.float(), dim=-1))
    q2 = head.hlg.transform_from_probs(torch.softmax(q2_logits.float(), dim=-1))
    return torch.minimum(q1, q2)


def load_frozen_critic(checkpoint: str | Path, *, device="cpu", dtype=torch.bfloat16):
    """Load a separate full critic checkpoint, preserving its own preprocessing."""
    from gr00t.model.gr00t_n1_deas_critic import GR00T_N1_5_DEAS_Critic
    model = GR00T_N1_5_DEAS_Critic.from_pretrained(
        str(checkpoint), torch_dtype=dtype, tune_visual=False, tune_llm=False,
        tune_critic=False, tune_value=False,
    )
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)
