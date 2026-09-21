"""Precision boundary for SVF online, target and evaluation Q networks."""
import torch


def q_forward_fp32(network, features, states, actions):
    """Keep Q matmuls in FP32, including inside an outer BF16 autocast.

    TF32 is disabled only for this call; other model precision settings remain
    unchanged. Q parameters must remain FP32 (the training/export default).
    """
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=features.device.type, enabled=False):
            return network(features.float(), states.float(), actions.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def projection_forward_fp32(network, pooled, embodiment_id):
    """Preserve full precision through the learned feature projection."""
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            return network(pooled.float(), embodiment_id).tanh()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
