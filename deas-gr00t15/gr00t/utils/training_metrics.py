"""Wall-clock training metrics, collected without extra CUDA synchronization."""

import json
import os
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


class TrainingMetricsCallback(TrainerCallback):
    """Add metrics before the reporting callbacks consume the same log dict.

    Rates use completed optimizer steps and global batch size. They include data
    loading and checkpoint time between samples, and are not kernel benchmarks.
    GPU allocation hours start at the worker script, excluding image startup.
    CUDA allocator peaks describe rank 0 only; W&B system stats cover all GPUs.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        self.start_step = state.global_step
        self.window_start = self.started
        self.window_step = self.start_step
        self.snapshot = {}
        self.snapshot_step = None
        self.metrics_path = Path(args.output_dir) / "performance.jsonl"
        self.world_size = max(1, args.world_size)
        self.global_batch = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.world_size
        )
        try:
            self.job_start = float(os.environ["DEAS_JOB_START_UNIX"])
        except (KeyError, ValueError):
            self.job_start = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        interval = max(1, int(state.logging_steps or args.logging_steps))
        step = state.global_step
        if step % interval and step != state.max_steps:
            return control
        now = time.perf_counter()
        elapsed = max(0.0, now - self.started)
        window = now - self.window_start
        completed = step - self.window_step
        job_elapsed = max(0.0, time.time() - self.job_start)
        self.snapshot = {
            "timing/train_elapsed_seconds": elapsed,
            "timing/worker_script_elapsed_seconds": job_elapsed,
            "progress/samples_seen_global": step * self.global_batch,
            "progress/samples_this_session": (step - self.start_step) * self.global_batch,
            "usage/gpu_hours_since_worker_start": job_elapsed * self.world_size / 3600,
        }
        if completed > 0 and window > 0:
            self.snapshot.update({
                "timing/seconds_per_step": window / completed,
                "timing/samples_per_second_global": completed * self.global_batch / window,
                "timing/remaining_train_seconds_estimate": (
                    max(0, state.max_steps - step) * window / completed
                ),
            })
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            self.snapshot.update({
                "memory/rank0_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "memory/rank0_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            })
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps({"step": step, **self.snapshot}, allow_nan=False) + "\n")
        self.snapshot_step = step
        self.window_start = now
        self.window_step = step
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            if state.global_step == getattr(self, "snapshot_step", None):
                logs.update(self.snapshot)
        return control
