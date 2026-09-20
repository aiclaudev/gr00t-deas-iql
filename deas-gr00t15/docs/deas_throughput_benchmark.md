# DEAS throughput comparison: one GPU vs four GPUs

## Scope

Representative workload: existing IQL critic forward/backward, frozen VLM,
trainable critic/value head, FP32 optimizer and BF16 autocast. This is a speed
comparison, not another production training run. The saved critic checkpoint
at step 4000 is loaded independently for every condition; no production model
or checkpoint is changed. SVF has extra reference-flow sampling work, so the
result measures this IQL workload rather than predicting SVF speed exactly.

Login benchmarks are prohibited by the B200 cluster guide. These commands are
for Slurm workers only. The preparation and CPU tests do not execute a benchmark.

## Six independent conditions

| Backend | GPUs | Batch per GPU | Global batch | CPU total | Loader workers total | RAM |
|---|---:|---:|---:|---:|---:|---:|
| Existing Decord | 1 | 128 | 128 | 24 | 16 | 192 GiB |
| Existing Decord | 4 | 32 | 128 | 24 | 16 | 192 GiB |
| Decord with reader cache | 1 | 128 | 128 | 24 | 16 | 192 GiB |
| Decord with reader cache | 4 | 32 | 128 | 24 | 16 | 192 GiB |
| TorchCodec CPU with reader cache | 1 | 128 | 128 | 24 | 16 | 192 GiB |
| TorchCodec CPU with reader cache | 4 | 32 | 128 | 24 | 16 | 192 GiB |

The GPU-job CPU ceiling is 30 per GPU; 24 CPUs allows an identical CPU allocation
for both sizes. Defaulting to 8 CPU per GPU would compare 8 CPUs against 32 CPUs
and confound GPU scaling with data loading capacity. This differs from the
current production critic's 32-CPU allocation and must be retained in reports.

Every case uses seed 42, the same global sample draw positions, four-task
`demos + rollouts`, and checkpoint-pinned normalization. Each case runs five
warm-up updates then twenty measured updates. CPU worker count is 16 for the
single process or four per rank for four processes. Per-case walltime is bounded
at 15 minutes; actual speed and memory fit are not known until execution.

## Decoder comparison details

The benchmark installs a process-local loader adapter. Existing training files
and running jobs retain their original Decord path. Baseline preserves its
current reader initialization and thread settings. The two cached conditions
use the same bounded reader cache and explicit one-thread decoding.

TorchCodec retrieves the same nearest-timestamp frame indices as Decord. To
preserve alignment, the compatibility adapter obtains the Decord timestamp
index once per cache miss, then uses TorchCodec for frame decoding. This extra
initialization is included in measured data loading. It is not claimed to be
an optimized pure-TorchCodec implementation. The current NumPy/PIL transform
path stays on CPU; CUDA video decoding is not enabled.

Parquet reading, state-array conversion and augmentation remain identical.
Thus this benchmark isolates the video-loader changes while retaining those
other potential CPU bottlenecks. A short random-access workload may have low
reader-cache hit rates; report measured results without assuming cache gains.

## Commands

```bash
cd /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
# Read-only plan: no directories, GPU usage or submissions.
python scripts/submit_deas_benchmark.py --after-job 136451
# Actual submission, only when worker execution is authorized:
python scripts/submit_deas_benchmark.py --after-job 136451 --submit
```

Each condition is a separate own job, connected with `afterany` to prevent
benchmark loaders competing against one another. A failed or OOM condition is
reported as such and does not prevent the other independent conditions running.
Existing user jobs, including cos_fwm, are never canceled. No monitor, automatic
SVF launch or nested submission is installed.

Inputs are known project paths and outputs remain under
`output/deas-benchmark/<run>/`. The submission manifest records every accepted
job ID even if a later submission fails. Worker image is NGC PyTorch 25.04;
Python comes from the existing personal `groot-train` environment.

## Interpretation

Compare warm-up-excluded wallclock seconds/update and global samples/second for
latency. Also compare projected GPU-hours for 10,000 updates for resource cost:
one GPU can be more economical while four GPUs have lower latency. Record the
largest rank's allocated/reserved VRAM and the visible data-loader wait.

CUDA event intervals include synchronization and DDP/NCCL waiting; they are not
pure kernel-compute utilization. DataLoader blocking time excludes CPU work
hidden by prefetch. Model startup, cache warm-up, checkpoint saving and logging
are not represented by a simple 10,000-update speed extrapolation.
