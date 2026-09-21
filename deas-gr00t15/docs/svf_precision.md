# Policy-TD SVF critic precision

The default policy-TD SVF trainer now uses FP32 for learned projection, online
and EMA target Q networks, and soft-value heads. Float32 matmul precision is
highest and TF32 is disabled for the training process, covering backward as
well as forward. Weights, optimizer states, targets and losses stay FP32.
Frozen VLM/reference-head and actor velocity calls keep explicit BF16 autocast.
Feature pooling converts to FP32 before the mean. This does not recover detail
already lost in the frozen BF16 feature extractor.

SVF BoN uses the same FP32 pooling/projection/Q path. Existing checkpoints can
be scored with it, but scores may differ from their previous BF16 evaluation;
record code revision when comparing results. No state-dict architecture change
or automatic checkpoint conversion is performed.

The original upstream DEAS critic trainer sets bf16=True, tf32=True and
compute_dtype=bfloat16. That separate legacy training path and the IQL trainer
are not changed by this policy-TD SVF precision update.
