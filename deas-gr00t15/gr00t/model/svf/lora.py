"""DiT attention LoRA for SVF, with plain GR00T checkpoint export.

The adapter starts at zero and leaves every BC parameter frozen. Only the
attention projections inside ``action_head.model.transformer_blocks`` are
adapted; the separate visual-language self-attention and projectors are excluded.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActorTuningConfig:
    mode: str = "full"
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.mode not in ("full", "dit-lora"):
            raise ValueError("Actor tuning mode must be 'full' or 'dit-lora'")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        return self


class LoRALinear(nn.Module):
    """Frozen Linear plus an FP32 low-rank update; B=0 preserves BC initially."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        ActorTuningConfig("dit-lora", rank, alpha, dropout)
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires an ordinary torch.nn.Linear")
        self.base = base.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                               dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                               dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.train(base.training)

    @property
    def weight(self):
        # Diffusers attention processors inspect this attribute for shape/dtype.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        output = self.base(inputs)
        # Autocast may compute these matmuls in BF16, while optimizer parameters
        # remain FP32. Preserve the gradient through the final dtype conversion.
        update = F.linear(F.linear(self.dropout(inputs.to(self.lora_A.dtype)),
                                   self.lora_A), self.lora_B)
        return output + (update * self.scaling).to(output.dtype)


def is_lora_parameter(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in ("lora_A", "lora_B")


def apply_dit_lora(action_head: nn.Module, config: ActorTuningConfig) -> list[str]:
    """Freeze the full action head and wrap only DiT Q/K/V/output projections.

    ``full`` is a no-op, preserving the caller's existing tuning flags. Invalid
    structures fail before freezing or changing any module. Calling twice raises.
    Returned paths are relative to the action head.
    """
    config.validate()
    if config.mode == "full":
        return []
    if any(isinstance(module, LoRALinear) for module in action_head.modules()):
        raise ValueError("DiT LoRA is already installed on this action head")
    blocks = getattr(getattr(action_head, "model", None), "transformer_blocks", None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential)) or not len(blocks):
        raise ValueError("Expected action_head.model.transformer_blocks with at least one block")
    targets = []
    for index, block in enumerate(blocks):
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            path = f"model.transformer_blocks.{index}.attn1.{projection}"
            try:
                module = action_head.get_submodule(path)
            except AttributeError as error:
                raise ValueError(f"Missing DiT LoRA target: {path}") from error
            if not isinstance(module, nn.Linear):
                raise TypeError(f"DiT LoRA target is not Linear: {path}")
            targets.append((path, module))
    action_head.requires_grad_(False)
    for path, module in targets:
        parent_path, attribute = path.rsplit(".", 1)
        parent = action_head.get_submodule(parent_path)
        setattr(parent, attribute, LoRALinear(module, config.rank, config.alpha, config.dropout))
    return [path for path, _ in targets]


def _merged_state_dict(module: nn.Module) -> OrderedDict:
    """Return CPU tensors with vanilla keys without changing live parameters.

    Each projection is merged in FP32 on the CPU, then cast to the original base
    dtype. Other weights/buffers retain their original dtype. No full GPU copy.
    """
    adapters = {name: child for name, child in module.named_modules()
                if isinstance(child, LoRALinear)}
    original = module.state_dict()
    removed = set()
    replacements = {}
    metadata = OrderedDict(getattr(original, "_metadata", {}))
    for name, child in adapters.items():
        prefix = name + "." if name else ""
        base = child.base.weight.detach().to(device="cpu", dtype=torch.float32)
        delta = child.lora_B.detach().float().cpu() @ child.lora_A.detach().float().cpu()
        replacements[prefix + "base.weight"] = (
            prefix + "weight", (base + child.scaling * delta).to(child.base.weight.dtype))
        if child.base.bias is not None:
            replacements[prefix + "base.bias"] = (prefix + "bias", child.base.bias.detach().cpu().clone())
        removed.update((prefix + "lora_A", prefix + "lora_B"))
        # Match the metadata of an ordinary Linear at the pre-adapter path.
        metadata[name] = metadata.get(prefix + "base", {})
        for key in tuple(metadata):
            if key.startswith(prefix) and key != name and (prefix or key in ("base", "dropout")):
                metadata.pop(key)
    merged = OrderedDict()
    for key, tensor in original.items():
        if key in removed:
            continue
        if key in replacements:
            merged_key, merged_tensor = replacements[key]
            merged[merged_key] = merged_tensor
        else:
            merged[key] = tensor.detach().cpu().clone()
    merged._metadata = metadata
    return merged


def merged_action_head_state_dict(action_head: nn.Module) -> OrderedDict:
    return _merged_state_dict(action_head)


def merged_actor_state_dict(actor: nn.Module) -> OrderedDict:
    """State dict accepted by actor.save_pretrained(..., state_dict=result)."""
    return _merged_state_dict(actor)


__all__ = ["ActorTuningConfig", "LoRALinear", "apply_dit_lora", "is_lora_parameter",
           "merged_action_head_state_dict", "merged_actor_state_dict"]
