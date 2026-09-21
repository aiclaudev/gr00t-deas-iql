# IQL critic encoder comparison

`train_chunk_iql.py --critic-encoder none` removes the category-specific
2048-to-64 MLP and its final tanh. Frozen BC2 VLM, VLLN and VL self-attention
remain; their mean-pooled features enter Q and V directly.

- `deas` (default): original learned 64-dimensional tanh representation.
- `none`: no learned feature projection or tanh; use the full backbone feature
  width (2048 for this BC2 model). Q and V input layers are resized accordingly.
- Q hidden layers, V BRONet, scalar outputs, IQL losses, QC validity and reward
  shaping are unchanged. Precision follows the existing BF16 configuration.
- Train a fresh critic from the same BC2 actor; a 64-dimensional critic head is
  not shape-compatible with the direct feature model.
- Checkpoints record `critic_encoder`. The IQL BoN loader and rescoring tool use
  this field; older configurations without the field retain the original mode.
- Training logs include Q/V minimum, maximum and standard deviation every ten
  updates alongside the existing means.

## Inspect a saved DEAS encoder

Run `scripts/rescore_chunk_iql.py` in sbatch with `--checkpoint`, `--source-root`,
`--output` and `--inspect-encoder`. The source root can be the original training
snapshot for legacy checkpoints. It records each sampled state's 64 values
before and after tanh in `encoder_values.jsonl`, with task/episode/start IDs.
`summary.json` includes exact ±1 saturation, local tanh derivative, variation
per dimension, and counts of distinct output/sign vectors. No training updates
are performed. Diagnostic samples are outcome-stratified, not a held-out
policy success evaluation.
