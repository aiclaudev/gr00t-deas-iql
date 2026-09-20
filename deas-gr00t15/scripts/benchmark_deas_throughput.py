#!/usr/bin/env python3
"""Worker-only IQL update throughput: identical samples and global batch on 1/4 GPUs.

This never exports a model or resumes the production run. Timing includes data
loading, H2D copies, forward/backward, gradient clipping, Adam and Q EMA. CUDA
events include stream stalls and DDP communication: they are not pure compute.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.train_svf import (DeterministicSamples, MicrobatchSampler,
                               atomic_json, validate_dataset_metadata)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset-path", nargs="+", required=True)
    p.add_argument("--global-batch", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=20)
    p.add_argument("--global-workers", type=int, default=16)
    p.add_argument("--video-backend", choices=("decord", "decord_cached", "torchcodec"), default="decord")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validate-config", action="store_true")
    p.add_argument("--world-size", type=int, help="Validation only; real runs use torchrun WORLD_SIZE")
    return p


def layout(global_batch, global_workers, world_size):
    if world_size not in (1, 4):
        raise ValueError("This comparison supports exactly 1 or 4 GPUs")
    if global_batch <= 0 or global_batch % world_size:
        raise ValueError("global-batch must be positive and divisible by world size")
    if global_workers < 0 or global_workers % world_size:
        raise ValueError("global-workers must be nonnegative and divisible by world size")
    return {"world_size": world_size, "global_batch": global_batch,
            "batch_per_rank": global_batch // world_size,
            "global_workers": global_workers, "workers_per_rank": global_workers // world_size,
            "gradient_accumulation_steps": 1,
            "prefetch_factor": world_size if global_workers else None,
            "prefetched_samples_limit": global_workers * global_batch}


def validate_args(args, world_size):
    schedule = layout(args.global_batch, args.global_workers, world_size)
    if args.warmup_steps < 1 or args.measure_steps < 2:
        raise ValueError("Use at least one warmup and two measured steps")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.world_size is not None and not args.validate_config:
        raise ValueError("--world-size is only for --validate-config")
    args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    args.output = str(Path(args.output).expanduser().resolve())
    args.dataset_path = [str(Path(p).expanduser().resolve()) for p in args.dataset_path]
    if not Path(args.output).is_relative_to(Path("/home/nas_main/dohyunlee")):
        raise ValueError("Output must remain within /home/nas_main/dohyunlee")
    checkpoint = Path(args.checkpoint)
    for name in ("config.json", "experiment_cfg/metadata.json", "model.safetensors", "optimizer.pt"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Missing checkpoint artifact: {checkpoint / name}")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "gr00t_n1_5_deass_critic":
        raise ValueError("Expected an existing DEAS IQL critic checkpoint")
    load_critic_metadata(checkpoint)
    for path in args.dataset_path:
        validate_dataset_metadata(Path(path))
    return schedule



def load_critic_metadata(checkpoint):
    """Preserve transition modalities when restoring IQL normalization metadata.

    DatasetMetadata is the BC schema: parsing an IQL checkpoint with it silently
    discards next_state, next_video, reward and done before RL transforms run.
    Validate the RL schema in preflight as well as during dataset construction.
    """
    from gr00t.data.schema import RLDatasetMetadata

    metadata = json.loads((Path(checkpoint) / "experiment_cfg/metadata.json").read_text())
    if "new_embodiment" not in metadata:
        raise ValueError("Missing new_embodiment normalization metadata")
    return RLDatasetMetadata.model_validate(metadata["new_embodiment"])


def checkpoint_tensor_dtypes(checkpoint):
    """Read only the safetensors JSON header, never the multi-GB weight body."""
    with (Path(checkpoint) / "model.safetensors").open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Invalid safetensors header")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if not 0 < header_size <= 64 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(header_size))
    return {key: value["dtype"] for key, value in header.items() if key != "__metadata__"}


def restore_saved_precision(model, checkpoint):
    """Match the production checkpoint: frozen backbone BF16, IQL head FP32."""
    import torch
    dtypes = checkpoint_tensor_dtypes(checkpoint)
    for prefix, expected in (("backbone.", "BF16"), ("critic_head.", "F32")):
        selected = {key: value for key, value in dtypes.items() if key.startswith(prefix)}
        if not selected or set(selected.values()) != {expected}:
            raise ValueError(f"Unexpected saved precision for {prefix}; do not silently benchmark different precision")
    model.backbone.to(dtype=torch.bfloat16)
    model.critic_head.float()
    state = model.state_dict()
    for key, expected in dtypes.items():
        if key not in state or state[key].dtype != {"BF16": torch.bfloat16, "F32": torch.float32}.get(expected):
            raise ValueError(f"Model precision/key differs from saved checkpoint: {key}")
    return {"backbone": "bfloat16", "critic_head": "float32", "optimizer_state": "float32"}


def restore_checkpoint_optimizer(model, checkpoint):
    """Restore the production Adam layout, including the frozen embedding alias.

    The IQL checkpoint was saved with tied Qwen token/output embeddings. The
    nested Hugging Face reload can leave lm_head as a new, separate parameter,
    shifting every subsequent optimizer ID. Only repair that verified alias;
    reject any other layout or moment-shape mismatch before training.
    """
    import torch

    saved = torch.load(Path(checkpoint) / "optimizer.pt", map_location="cpu",
                       weights_only=True, mmap=True)
    groups = saved["param_groups"]
    if len(groups) != 1:
        raise ValueError("Expected the production IQL Adam single parameter group")
    ids = groups[0]["params"]
    named = list(model.named_parameters())
    repaired_alias = False
    if len(named) != len(ids):
        dtypes = checkpoint_tensor_dtypes(checkpoint)
        embedding_name = "backbone.eagle_model.language_model.model.embed_tokens.weight"
        output_name = "backbone.eagle_model.language_model.lm_head.weight"
        extra = {name for name, _ in named if name not in dtypes}
        lm = model.backbone.eagle_model.language_model
        if (len(named) != len(ids) + 1 or extra != {output_name}
                or embedding_name not in dtypes
                or not getattr(lm.config, "tie_word_embeddings", False)):
            raise ValueError("Optimizer layout differs from the checkpoint; no verified embedding alias repair")
        embedding = lm.get_input_embeddings().weight
        output = lm.get_output_embeddings().weight
        if embedding.shape != output.shape or embedding.requires_grad or output.requires_grad:
            raise ValueError("Expected compatible frozen input/output embeddings")
        lm.tie_weights()
        if lm.get_output_embeddings().weight is not lm.get_input_embeddings().weight:
            raise ValueError("Failed to restore checkpoint embedding weight sharing")
        named = list(model.named_parameters())
        repaired_alias = True
    if len(named) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Optimizer parameter count/IDs differ from the checkpoint")
    expected_states = {pid for pid, (_, parameter) in zip(ids, named) if parameter.requires_grad}
    if set(saved["state"]) != expected_states:
        raise ValueError("Checkpoint Adam state IDs do not match the model's trainable parameters")
    for pid, (name, parameter) in zip(ids, named):
        if pid not in saved["state"]:
            continue
        state = saved["state"][pid]
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != torch.float32 or parameter.dtype != torch.float32):
                raise ValueError(f"Checkpoint Adam {key} mismatch at parameter {pid}: {name}")
        if "step" not in state:
            raise ValueError(f"Checkpoint Adam step missing for {name}")
    optimizer = torch.optim.Adam([parameter for _, parameter in named], lr=groups[0]["lr"])
    optimizer.load_state_dict(saved)
    return optimizer, {"parameter_count": len(named), "state_count": len(saved["state"]),
                       "restored_frozen_embedding_alias": repaired_alias}


def worker_init(worker_id, *, backend):
    # Keep each data worker to one CPU thread. Main training ranks use two.
    import torch
    from gr00t.utils.video_benchmark import install_benchmark_video_backend
    torch.set_num_threads(1)
    install_benchmark_video_backend(backend, reader_cache_size=16, num_threads=1)


def build_dataset(args):
    from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
    from gr00t.data.schema import DatasetStatisticalValues, EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    horizon = config["critic_cfg"]["rl_config"]["critic_action_horizon"]
    pinned = load_critic_metadata(checkpoint)
    for path in args.dataset_path:
        for values in validate_dataset_metadata(Path(path)).values():
            DatasetStatisticalValues.model_validate(values)
    data_config = DATA_CONFIG_MAP["single_panda_gripper_rl"](AS=horizon)
    datasets = [LeRobotSingleDataset(
        dataset_path=path, modality_configs=data_config.modality_config(),
        transforms=data_config.transform(), embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend="decord", use_rl=True,
    ) for path in args.dataset_path]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = LeRobotMixtureDataset(
            data_mixture=[(d, 1.0) for d in datasets], mode="train", seed=args.seed,
            balance_dataset_weights=True, balance_trajectory_weights=True,
            metadata_config={"percentile_mixing_method": "weighted_average"}, use_rl=True,
        )
        dataset.merged_metadata["new_embodiment"] = pinned
    for child in datasets:
        child.set_transforms_metadata(pinned)
    return DeterministicSamples(dataset, args.seed)


def percentile(values, q):
    if not values:
        raise ValueError("Cannot summarize an empty measurement")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(reports, *, global_batch):
    if not reports or len({len(r["steps"]) for r in reports}) != 1:
        raise ValueError("All ranks must report the same measured steps")
    count = len(reports[0]["steps"])
    if count == 0:
        raise ValueError("No measured steps")
    elapsed = max(r["measured_elapsed_seconds"] for r in reports)
    if elapsed <= 0:
        raise ValueError("Measurement duration must be positive")
    # These are critical-rank envelopes, not sums across GPUs or CPU utilization.
    end_to_end = [max(r["steps"][i]["wall_seconds"] for r in reports) for i in range(count)]
    data_wait = [max(r["steps"][i]["data_wait_seconds"] for r in reports) for i in range(count)]
    event_ms = [max(r["steps"][i]["cuda_update_milliseconds"] for r in reports) for i in range(count)]
    seconds_per_update = elapsed / count
    return {
        "measured_updates": count, "global_batch": global_batch,
        "measured_interval_seconds_max_rank": elapsed,
        "updates_per_second": count / elapsed,
        "samples_per_second": count * global_batch / elapsed,
        "seconds_per_update_from_interval": seconds_per_update,
        "step_seconds_median_max_rank": statistics.median(end_to_end),
        "step_seconds_p90_max_rank": percentile(end_to_end, 0.9),
        "data_wait_seconds_mean_max_rank": statistics.mean(data_wait),
        "data_wait_seconds_p90_max_rank": percentile(data_wait, 0.9),
        "cuda_update_ms_mean_max_rank_including_ddp_waits": statistics.mean(event_ms),
        "peak_allocated_bytes_max_rank": max(r["peak_allocated_bytes"] for r in reports),
        "peak_reserved_bytes_max_rank": max(r["peak_reserved_bytes"] for r in reports),
        "estimated_10000_updates_wall_hours": seconds_per_update * 10000 / 3600,
        "estimated_10000_updates_gpu_hours": seconds_per_update * 10000 / 3600 * len(reports),
        "estimate_excludes": "startup, checkpoint saving, evaluation, queueing and dataset/cache drift",
        "timing_notes": "Warmup excluded. CUDA event includes H2D, stream stalls and DDP waits; not pure compute. Data wait is host queue wait, not CPU utilization. Interval includes per-step synchronization and measurement overhead.",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    # Enforce before even importing torch or constructing any model/dataset.
    if not args.validate_config and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Benchmarks must run in sbatch workers; login benchmarking is prohibited")
    world_size = args.world_size if args.validate_config and args.world_size else int(os.environ.get("WORLD_SIZE", "1"))
    schedule = validate_args(args, world_size)
    if args.validate_config:
        print(json.dumps({"valid": True, "layout": schedule, "checkpoint": args.checkpoint,
                          "video_backend": args.video_backend}, indent=2))
        return
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    output = Path(args.output)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_MODULES_CACHE"] = str(Path(os.environ.get("HF_MODULES_CACHE", str(output / ".cache/hf-modules"))) / f"rank-{rank}")
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from gr00t.model.gr00t_n1_deas_critic import GR00T_N1_5_DEAS_Critic
    from gr00t.model.transforms import DefaultDataCollator
    from gr00t.utils.experiment import PolyakUpdateCallback
    from gr00t.utils.video_benchmark import install_benchmark_video_backend

    records = []
    stage = "initializing"
    started = time.monotonic()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
            raise RuntimeError("Worker visible GPU count must match the torchrun world size")
        torch.cuda.set_device(local_rank)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if world_size > 1:
            dist.init_process_group("nccl")
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if (output / "result.json").exists():
                raise FileExistsError("Refusing to overwrite an existing benchmark result")
        if world_size > 1:
            dist.barrier()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        install_benchmark_video_backend(args.video_backend, reader_cache_size=16, num_threads=1)
        stage = "dataset"
        dataset = build_dataset(args)
        stage = "model"
        model = GR00T_N1_5_DEAS_Critic.from_pretrained(
            args.checkpoint, tune_visual=False, tune_llm=False, tune_critic=True, tune_value=True,
            torch_dtype=torch.float32, local_files_only=True,
        )
        precision = restore_saved_precision(model, args.checkpoint)
        model.to(torch.device("cuda", local_rank))
        model.compute_dtype = "bfloat16"
        model.config.compute_dtype = "bfloat16"
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable or any(p.dtype != torch.float32 for p in trainable):
            raise RuntimeError("IQL trainable parameters must be nonempty FP32 parameters")
        if any(p.requires_grad for p in model.backbone.parameters()):
            raise RuntimeError("The vision/LLM backbone must remain frozen")
        optimizer, optimizer_restoration = restore_checkpoint_optimizer(model, args.checkpoint)
        if rank == 0:
            print("Optimizer restoration: " + json.dumps(optimizer_restoration), flush=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        ema = PolyakUpdateCallback(model.critic_head.target_critic, model.critic_head.critic,
                                  tau=model.critic_head.config.rl_config["tau"])
        train_model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
            broadcast_buffers=False, bucket_cap_mb=100,
        ) if world_size > 1 else model
        model.train()
        # Input augmentation is separately seeded by absolute global draw position.
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        sampler = MicrobatchSampler(start=0, stop=args.warmup_steps + args.measure_steps,
                                    rank=rank, world_size=world_size, size=schedule["batch_per_rank"])
        loader = DataLoader(
            dataset, batch_sampler=sampler, collate_fn=DefaultDataCollator(),
            num_workers=schedule["workers_per_rank"], pin_memory=False,
            persistent_workers=schedule["workers_per_rank"] > 0,
            prefetch_factor=schedule["prefetch_factor"],
            worker_init_fn=partial(worker_init, backend=args.video_backend),
            generator=torch.Generator().manual_seed(args.seed + rank),
        )
        if rank == 0:
            print(f"Initialization complete; starting {args.warmup_steps} warmup and "
                  f"{args.measure_steps} measured updates", flush=True)
        batches = iter(loader)
        stage = "warmup"
        measurement_start = None
        for step in range(args.warmup_steps + args.measure_steps):
            if step == args.warmup_steps:
                torch.cuda.synchronize()
                if world_size > 1:
                    dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                measurement_start = time.monotonic()
                stage = "measurement"
            step_start = time.monotonic()
            inputs = next(batches)
            data_wait = time.monotonic() - step_start
            event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            event_start.record()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = train_model(inputs)["loss"]
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite IQL loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            ema.on_step_end(None, None, None)
            event_end.record()
            event_end.synchronize()
            wall_seconds = time.monotonic() - step_start
            if step >= args.warmup_steps:
                records.append({"step": step + 1, "measured_step": step - args.warmup_steps + 1,
                                "wall_seconds": wall_seconds, "data_wait_seconds": data_wait,
                                "cuda_update_milliseconds": event_start.elapsed_time(event_end),
                                "loss": float(loss.detach())})
            if rank == 0:
                print(f"{stage}: completed {step + 1}/{args.warmup_steps + args.measure_steps} updates; "
                      f"wall={wall_seconds:.3f}s data_wait={data_wait:.3f}s "
                      f"loss={float(loss.detach()):.6f}", flush=True)
        elapsed = time.monotonic() - measurement_start
        report = {"rank": rank, "steps": records, "measured_elapsed_seconds": elapsed,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        atomic_json(output / f"rank-{rank}.json", report)
        reports = [None] * world_size if rank == 0 else None
        if world_size > 1:
            dist.gather_object(report, reports, dst=0)
        else:
            reports = [report]
        if rank == 0:
            metadata = Path(args.checkpoint) / "experiment_cfg/metadata.json"
            result = {"schema": 1, "state": "completed", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                      "checkpoint": args.checkpoint, "normalization_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                      "dataset_paths": args.dataset_path, "seed": args.seed, "layout": schedule,
                      "prefetch_note": "Equal total queued-sample limit, but batches assigned to workers differ. Short runs include filesystem/cache and pipeline-fill effects.",
                      "decoder_counters": "DataLoader worker counters are not aggregated; no cache-hit claim is made.",
                      "video_backend": args.video_backend, "warmup_steps_excluded": args.warmup_steps,
                      "optimizer": "Adam from checkpoint, FP32 state/weights, original learning rate and constant schedule",
                      "precision": precision, "optimizer_restoration": optimizer_restoration,
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "gpu_names": [torch.cuda.get_device_name(i) for i in range(world_size)],
                      "total_process_seconds": time.monotonic() - started,
                      "summary": summarize(reports, global_batch=args.global_batch), "rank_reports": reports}
            atomic_json(output / "result.json", result)
            print(json.dumps(result["summary"], indent=2), flush=True)
    except BaseException as exc:
        # Each failed rank records its own failure; no collective is attempted,
        # so an OOM does not require peers to enter a failed-rank barrier.
        output.mkdir(parents=True, exist_ok=True)
        failure = {"schema": 1, "state": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed",
                   "rank": rank, "stage": stage, "error_type": type(exc).__name__, "error": str(exc),
                   "layout": schedule, "video_backend": args.video_backend,
                   "completed_measured_steps": len(records), "steps": records}
        atomic_json(output / f"failure-rank-{rank}.json", failure)
        if rank == 0 and not (output / "result.json").exists():
            atomic_json(output / "result.json", failure)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
