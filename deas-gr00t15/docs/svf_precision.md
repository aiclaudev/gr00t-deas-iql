# Policy-TD SVF precision

Use the original mixed-precision setup: explicit BF16 autocast for frozen
backbone/reference, actor, projection and Q scoring; TF32 enabled. Trainable
master weights, optimizer, targets and losses remain FP32. Soft-value code
retains its original FP32 path outside BF16 autocast.

This follows the DEAS baseline precision settings (bf16=True, tf32=True,
compute_dtype=bfloat16), rather than the experimental strict FP32 critic path.
The optional precision helpers are not used by the default trainer or scorer.
No learning-rate, reward, discount or SVF algorithm settings are changed.
