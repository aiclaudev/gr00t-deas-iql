"""Initialize soft-value features or full HLGauss heads from DEAS online Q.

Trunk transfer retains the fresh readout. Full transfer also copies the logits
readout, but is valid only with exactly matching HLGauss support and smoothing.
Binary actions use a smooth extension because thresholding destroys guidance.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _finite(tensor, name):
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain finite floating-point values")


def _linear(layer, name):
    if type(layer) is not nn.Linear or layer.bias is None:
        raise ValueError(f"{name} must be an ordinary Linear with bias")
    if layer.weight.shape != (layer.out_features, layer.in_features) or layer.bias.shape != (layer.out_features,):
        raise ValueError(f"{name} has inconsistent Linear shapes")


def _hlgauss_configuration(soft_value, critic_head):
    """Validate distributional equivalence before preparing any parameter copy."""
    configuration = getattr(soft_value, "loss_configuration", None)
    if not callable(configuration):
        raise ValueError("Full critic initialization requires HLGauss loss_configuration()")
    config = configuration()
    keys = {"loss_type", "num_bins", "value_min", "value_max", "sigma"}
    if not isinstance(config, dict) or set(config) != keys or config["loss_type"] != "hl-gauss":
        raise ValueError("Full critic initialization requires an hl-gauss support configuration")
    if isinstance(config["num_bins"], bool) or not isinstance(config["num_bins"], int) or config["num_bins"] < 2:
        raise ValueError("HLGauss num_bins must be an integer of at least two")
    for name in ("value_min", "value_max", "sigma"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"HLGauss {name} must be finite")
    if config["value_max"] <= config["value_min"] or config["sigma"] <= 0:
        raise ValueError("HLGauss support range and sigma must be positive")
    hlg = getattr(critic_head, "hlg", None)
    if hlg is None:
        raise ValueError("Full critic initialization requires critic_head.hlg")
    for name, teacher_name in (("num_bins", "num_bins"), ("value_min", "min_value"),
                               ("value_max", "max_value"), ("sigma", "sigma")):
        source_value = getattr(hlg, teacher_name, None)
        if (isinstance(source_value, bool) or not isinstance(source_value, (int, float))
                or not math.isfinite(source_value) or source_value != config[name]):
            raise ValueError(f"Teacher and soft-value HLGauss {name} must match exactly")
        if getattr(soft_value, name, None) != config[name]:
            raise ValueError(f"Soft-value HLGauss {name} disagrees with its configuration")
    # These buffers drive the actual decoded expectation and target support.
    edges = getattr(soft_value, "support_edges", None)
    centers = getattr(soft_value, "bin_centers", None)
    if (not isinstance(edges, torch.Tensor) or not isinstance(centers, torch.Tensor)
            or edges.shape != (config["num_bins"] + 1,) or centers.shape != (config["num_bins"],)):
        raise ValueError("Soft-value HLGauss support buffers have invalid shapes")
    _finite(edges, "Soft-value support edges")
    _finite(centers, "Soft-value bin centers")
    # Build as the network constructor does, before a possible device transfer;
    # CUDA and CPU linspace can differ by an ulp despite identical parameters.
    expected_edges_cpu = torch.linspace(config["value_min"], config["value_max"], config["num_bins"] + 1,
                                        dtype=edges.dtype, device="cpu")
    expected_edges = expected_edges_cpu.to(edges)
    expected_centers = (expected_edges_cpu[:-1] / 2 + expected_edges_cpu[1:] / 2).to(centers)
    if not torch.equal(edges, expected_edges) or not torch.equal(centers, expected_centers):
        raise ValueError("Soft-value HLGauss support buffers disagree with their configuration")
    return dict(config)


@torch.no_grad()
def initialize_soft_value_from_critic(soft_value, critic_head, coordinates, *, copy_output=False) -> dict:
    """Copy Q1/Q2 layers into soft-value heads with coordinate correction.

    Features/state are already in critic coordinates. For each action timestep,
    ``x_critic = scale*x_actor + bias`` is absorbed into the first layer. Padded
    action columns are zeroed and all time columns are zero. Binary coordinates
    remain continuous, rather than applying the teacher's >0.5 threshold.

    All structures, shapes, values, and prepared copies are validated before any
    destination mutation. With copy_output=False, the random scalar or categorical
    final layers remain untouched. With copy_output=True, categorical readouts
    are copied too, requiring exactly matched hl-gauss support and sigma. Source
    weights and trainability stay unchanged; initialize before making an optimizer.
    """
    if not isinstance(copy_output, bool):
        raise ValueError("copy_output must be a boolean")
    loss_type = getattr(soft_value, "loss_type", "mse")
    if loss_type not in {"mse", "hl-gauss"}:
        raise ValueError("Soft-value loss_type must be mse or hl-gauss")
    if copy_output and loss_type != "hl-gauss":
        raise ValueError("Full critic initialization requires loss_type='hl-gauss'")
    support = _hlgauss_configuration(soft_value, critic_head) if copy_output else None
    expected_output_dim = 1 if loss_type == "mse" else getattr(soft_value, "num_bins", None)
    if (isinstance(expected_output_dim, bool) or not isinstance(expected_output_dim, int)
            or expected_output_dim < (1 if loss_type == "mse" else 2)):
        raise ValueError("Soft-value output dimension is invalid for its loss type")
    dimensions = {}
    for name in ("feature_dim", "state_dim", "action_horizon", "action_dim", "time_dim"):
        value = getattr(soft_value, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"soft_value.{name} must be a positive integer")
        dimensions[name] = value
    feature_dim, state_dim = dimensions["feature_dim"], dimensions["state_dim"]
    horizon, action_dim, time_dim = (dimensions[key] for key in
                                     ("action_horizon", "action_dim", "time_dim"))
    conditioning_dim = feature_dim + state_dim
    q_input_dim = conditioning_dim + horizon * action_dim
    heads = getattr(soft_value, "heads", None)
    if not isinstance(heads, nn.ModuleList) or len(heads) != 2:
        raise ValueError("soft_value must contain exactly two heads")
    critic = getattr(critic_head, "critic", None)
    sources = [getattr(getattr(critic, name, None), "mlp", None) for name in ("Q1", "Q2")]
    if not all(type(source) is nn.Sequential for source in sources):
        raise ValueError("Expected critic_head.critic.Q1/Q2.mlp Sequential networks")
    source_storages = {(str(p.device), p.untyped_storage().data_ptr()) for source in sources for p in source.parameters()}
    if any((str(p.device), p.untyped_storage().data_ptr()) in source_storages for p in soft_value.parameters()):
        raise ValueError("Soft value and teacher must have independent parameter storage")
    for name in ("action_scale", "action_bias", "action_mask", "binary_mask"):
        tensor = getattr(coordinates, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (action_dim,):
            raise ValueError(f"coordinates.{name} must have shape [{action_dim}]")
        if name in ("action_mask", "binary_mask"):
            if tensor.dtype != torch.bool:
                raise ValueError(f"coordinates.{name} must be a boolean mask")
        else:
            _finite(tensor, f"coordinates.{name}")
    # Binary identity coordinates match the teacher at exactly 0 and 1. Values
    # between/outside these endpoints intentionally follow a smooth extension.
    binary = coordinates.binary_mask.detach().cpu()
    valid = coordinates.action_mask.detach().cpu()
    scale = coordinates.action_scale.detach().float().cpu()
    bias = coordinates.action_bias.detach().float().cpu()
    if (binary & ~valid).any().item():
        raise ValueError("Binary action channels cannot be padding")
    if not torch.equal(scale[binary], torch.ones_like(scale[binary])) or not torch.equal(bias[binary], torch.zeros_like(bias[binary])):
        raise ValueError("Binary action coordinates require identity scale and zero bias")
    scale = torch.where(valid, scale, torch.zeros_like(scale)).repeat(horizon)
    bias = torch.where(valid, bias, torch.zeros_like(bias)).repeat(horizon)

    copies = []
    head_descriptions = []
    for head_index, (source, destination) in enumerate(zip(sources, heads)):
        prefix = f"head {head_index}"
        if type(destination) is not nn.Sequential or len(source) != len(destination):
            raise ValueError(f"{prefix} source/destination layer counts must match")
        if len(source) < 4 or (len(source) - 1) % 3:
            raise ValueError(f"{prefix} requires Linear/LayerNorm/GELU blocks plus final Linear")
        for name, parameter in source.named_parameters():
            _finite(parameter, f"{prefix} source {name}")
        for name, parameter in destination.named_parameters():
            _finite(parameter, f"{prefix} destination {name}")
            if parameter.dtype != torch.float32:
                raise ValueError("Soft-value destination parameters must be FP32")
        _linear(source[-1], f"{prefix} source final")
        _linear(destination[-1], f"{prefix} destination final")
        if source[-1].out_features < 2 or destination[-1].out_features != expected_output_dim:
            raise ValueError(f"{prefix} output dimensions are invalid for distributional Q and {loss_type} soft value")
        if copy_output and source[-1].out_features != expected_output_dim:
            raise ValueError(f"{prefix} teacher output dimension must match the HLGauss support")
        if source[-1].in_features != destination[-1].in_features:
            raise ValueError(f"{prefix} final hidden dimensions differ")
        source_width = q_input_dim
        destination_width = q_input_dim + time_dim
        for index in range(0, len(source) - 1, 3):
            s_linear, d_linear = source[index], destination[index]
            _linear(s_linear, f"{prefix} source layer {index}")
            _linear(d_linear, f"{prefix} destination layer {index}")
            if s_linear.in_features != source_width or d_linear.in_features != destination_width:
                raise ValueError(f"{prefix} input dimensions differ at layer {index}")
            if s_linear.out_features != d_linear.out_features:
                raise ValueError(f"{prefix} hidden dimensions differ at layer {index}")
            source_width = destination_width = s_linear.out_features
            s_norm, d_norm = source[index + 1], destination[index + 1]
            if type(s_norm) is not nn.LayerNorm or type(d_norm) is not nn.LayerNorm:
                raise ValueError(f"{prefix} layer {index + 1} must be LayerNorm")
            if (s_norm.normalized_shape != (source_width,) or d_norm.normalized_shape != (destination_width,)
                    or not s_norm.elementwise_affine or not d_norm.elementwise_affine
                    or s_norm.bias is None or d_norm.bias is None or s_norm.eps != d_norm.eps):
                raise ValueError(f"{prefix} LayerNorm configuration differs")
            s_activation, d_activation = source[index + 2], destination[index + 2]
            if type(s_activation) is not nn.GELU or type(d_activation) is not nn.GELU or s_activation.approximate != d_activation.approximate:
                raise ValueError(f"{prefix} GELU configuration differs")
            weight = s_linear.weight.detach().to(device=d_linear.weight.device, dtype=torch.float32).clone()
            layer_bias = s_linear.bias.detach().to(device=d_linear.bias.device, dtype=torch.float32).clone()
            if index == 0:
                action_weight = weight[:, conditioning_dim:]
                adjusted = torch.zeros_like(d_linear.weight)
                adjusted[:, :conditioning_dim] = weight[:, :conditioning_dim]
                adjusted[:, conditioning_dim:q_input_dim] = action_weight * scale.to(weight)
                layer_bias = layer_bias + action_weight @ bias.to(weight)
                weight = adjusted
            copies.extend(((d_linear.weight, weight), (d_linear.bias, layer_bias),
                           (d_norm.weight, s_norm.weight.detach().to(d_norm.weight).clone()),
                           (d_norm.bias, s_norm.bias.detach().to(d_norm.bias).clone())))
        if source[-1].in_features != source_width or destination[-1].in_features != destination_width:
            raise ValueError(f"{prefix} final input width does not match its trunk")
        if copy_output:
            copies.extend(((destination[-1].weight, source[-1].weight.detach().to(destination[-1].weight).clone()),
                           (destination[-1].bias, source[-1].bias.detach().to(destination[-1].bias).clone())))
        head_descriptions.append({"source": f"critic.Q{head_index + 1}.mlp",
                                  "destination": f"heads.{head_index}",
                                  "hidden_layers": (len(source) - 1) // 3,
                                  "source_output_bins": source[-1].out_features})
    for destination, value in copies:
        if destination.shape != value.shape:
            raise ValueError("Prepared critic-trunk parameter shape does not match destination")
        _finite(value, "Prepared critic-trunk parameter")
    for destination, value in copies:
        destination.copy_(value)
    return {
        "mode": "critic-full" if copy_output else "critic-trunk",
        "source": "online_q1_q2_complete_heads" if copy_output else "online_q1_q2_hidden_layers",
        "loss_type": loss_type,
        "hl_gauss_configuration": support,
        "heads": head_descriptions,
        "input_order": ["critic_features", "critic_state", "actor_action_flat", "fourier_time"],
        "q_input_dim": q_input_dim, "soft_value_input_dim": q_input_dim + time_dim,
        "time_columns": "zero_initialized_trainable",
        "action_coordinates": "affine_actor_to_critic_folded_into_first_linear",
        "padded_action_columns": "zero",
        "binary_actions": "smooth_continuous_extension_without_hard_threshold",
        "binary_channels": binary.nonzero(as_tuple=False).flatten().tolist(),
        "scalar_readout": "original_random_initialization_unchanged" if loss_type == "mse" else "not_used",
        "output_readout": "copied_categorical_logits" if copy_output else "original_random_initialization_unchanged",
        "preserves_q_output": False,
        "preserves_each_continuous_q_head_at_initialization": copy_output,
        "q_equivalence_limitations": [
            "noisy_binary_actions_use_a_continuous_extension_instead_of_teacher_threshold",
            "teacher_min_and_soft_value_mean_aggregation_differ",
        ],
        "teacher_aggregation": "min_q1_q2_unchanged",
        "soft_value_aggregation": "mean_two_scalar_heads_unchanged" if loss_type == "mse" else "mean_two_decoded_hlgauss_heads",
        "copied_parameter_tensors": len(copies),
    }


__all__ = ["initialize_soft_value_from_critic"]
