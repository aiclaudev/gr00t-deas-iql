#!/usr/bin/env python3
"""Joint SVF fine-tuning from a frozen BC reference and frozen DEAS Q.

Run with torchrun inside sbatch. --validate-config performs no model/GPU loading.
The actor's normalization metadata is pinned even when datasets are mixed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def batch_schedule(global_batch: int, microbatch: int, world_size: int) -> dict[str, int]:
    if min(global_batch, microbatch, world_size) <= 0:
        raise ValueError("Batch sizes and world size must be positive")
    if global_batch % (microbatch * world_size):
        raise ValueError("global batch must be divisible by microbatch size × world size")
    return {
        "global_batch_size": global_batch,
        "microbatch_size": microbatch,
        "world_size": world_size,
        "gradient_accumulation_steps": global_batch // (microbatch * world_size),
    }


def sample_seed(seed: int, position: int) -> int:
    # Stable across processes, Python hash randomization, and loader workers.
    value = (seed + position * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFF


class MicrobatchSampler:
    """Disjoint global draw positions; checkpointing needs only the next batch."""

    def __init__(self, *, start: int, stop: int, rank: int, world_size: int, size: int):
        if not (0 <= start <= stop and 0 <= rank < world_size and size > 0):
            raise ValueError("Invalid distributed sampler range")
        self.start, self.stop = start, stop
        self.rank, self.world_size, self.size = rank, world_size, size

    def __iter__(self):
        for batch in range(self.start, self.stop):
            first = (batch * self.world_size + self.rank) * self.size
            yield list(range(first, first + self.size))

    def __len__(self):
        return self.stop - self.start


class DeterministicSamples:
    """Seed each augmented sample by its draw position, including after resume.

    Mixture sampling itself is deterministic in (epoch, index, seed). Saving and
    restoring local RNG state also makes num_workers=0 safe for model RNG streams.
    """

    def __init__(self, dataset, seed: int):
        self.dataset, self.seed = dataset, seed
        self.length = len(dataset)
        if self.length <= 0:
            raise ValueError("Training dataset is empty")

    def __len__(self):
        return self.length

    def __getitem__(self, position):
        import numpy as np
        import torch

        epoch, index = divmod(position, self.length)
        self.dataset.set_epoch(epoch)
        # A single dataset also gets a deterministic permutation-like draw;
        # mixtures already sample trajectories and frames using their index.
        if not hasattr(self.dataset, "sample_step"):
            index = sample_seed(self.seed + epoch, index) % self.length
        py_state, np_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        seed = sample_seed(self.seed, position)
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor", required=True)
    p.add_argument("--critic", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset-path", nargs="+", required=True)
    p.add_argument("--global-batch-size", type=int, default=128)
    p.add_argument("--microbatch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--actor-lr", type=float, default=1e-5)
    p.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="constant",
                   help="Actor LR schedule over --steps; soft-value LR stays constant")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Optional actor warmup updates within the full schedule")
    p.add_argument("--min-lr-ratio", type=float, default=0.0,
                   help="Final actor LR / initial actor LR for cosine decay")
    p.add_argument("--actor-tuning", choices=("full", "dit-lora"), default="full",
                   help="Train the full action head or only DiT attention LoRA adapters")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--soft-value-init", choices=("random", "critic-trunk", "critic-full"), default="random",
                   help="Initialize randomly, copy the critic trunk, or copy all critic layers (HL-Gauss only)")
    p.add_argument("--soft-value-loss", choices=("mse", "hl-gauss"), default="mse",
                   help="Scalar MSE or HL-Gauss cross-entropy using the teacher critic support")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--report-to", choices=("none", "wandb"), default="none")
    p.add_argument("--run-name", default="seed42-svf-joint")
    p.add_argument("--wandb-project", default="gr00t1.5 finetune")
    p.add_argument("--wandb-entity", default="aiclaudev")
    p.add_argument("--kappa", type=float, default=0.4)
    p.add_argument("--g", type=float, default=0.25)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--t-min", type=float, default=0.1)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--guidance-clip", type=float, default=2.0)
    p.add_argument("--lambda-epsilon", type=float, default=1e-3)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after-steps", type=int,
                   help="Pause after this many additional updates; keep --steps for an exact resume")
    p.add_argument("--validate-config", action="store_true")
    p.add_argument("--world-size", type=int,
                   help="Validation only; torchrun WORLD_SIZE determines actual training")
    return p


def validate_dataset_metadata(path: Path) -> dict:
    """Read only known metadata; never trigger dataset-wide statistics generation."""
    for name in ("modality.json", "episodes.jsonl", "tasks.jsonl", "info.json", "stats.json"):
        item = path / "meta" / name
        if not item.is_file() or item.stat().st_size == 0:
            raise ValueError(f"Missing or empty dataset metadata: {item}; automatic statistics scans are disabled")
    stats = json.loads((path / "meta/stats.json").read_text())
    if not isinstance(stats, dict) or not stats:
        raise ValueError(f"Invalid dataset statistics: {path}")
    for feature, values in stats.items():
        if not isinstance(values, dict):
            raise ValueError(f"Invalid statistics for {feature} in {path}")
        for name in ("min", "max", "mean", "std", "q01", "q99"):
            value = values.get(name)
            if not isinstance(value, list) or not value:
                raise ValueError(f"Missing statistics array {feature}/{name} in {path}")
            def finite_numbers(items):
                return all(finite_numbers(x) if isinstance(x, list)
                           else isinstance(x, (int, float)) and math.isfinite(x) for x in items)
            if not finite_numbers(value):
                raise ValueError(f"Invalid statistics array {feature}/{name} in {path}")
    return stats


def validate_args(args, *, world_size: int) -> dict[str, Any]:
    schedule = batch_schedule(args.global_batch_size, args.microbatch_size, world_size)
    if args.soft_value_init == "critic-full" and args.soft_value_loss != "hl-gauss":
        raise ValueError("critic-full initialization requires --soft-value-loss hl-gauss")
    for key in ("steps", "save_steps", "log_steps", "candidates", "flow_steps"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    validate_lr_schedule(args)
    if args.candidates < 2:
        raise ValueError("candidates must be at least two for Q spread estimation")
    if not 0 <= args.num_workers <= 4:
        raise ValueError("num-workers must be between 0 and 4 per rank")
    if not 0 < args.t_min < 1:
        raise ValueError("t-min must be between zero and one")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("stop-after-steps must be positive")
    for key in ("actor_lr", "value_lr", "kappa", "g", "guidance_clip", "lambda_epsilon", "max_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.lora_rank <= 0:
        raise ValueError("lora-rank must be positive")
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        raise ValueError("lora-alpha must be finite and positive")
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        raise ValueError("lora-dropout must be finite and in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and nonnegative")
    for key in ("actor", "critic", "output"):
        setattr(args, key, str(Path(getattr(args, key)).expanduser().resolve()))
    args.dataset_path = [str(Path(p).expanduser().resolve()) for p in args.dataset_path]
    home = Path("/home/nas_main/dohyunlee")
    if not Path(args.output).is_relative_to(home):
        raise ValueError("Output must remain within /home/nas_main/dohyunlee")
    for path in (args.actor, args.critic, *args.dataset_path):
        if not Path(path).is_dir():
            raise ValueError(f"Missing input directory: {path}")
    for dataset_path in args.dataset_path:
        validate_dataset_metadata(Path(dataset_path))
    if not (Path(args.actor) / "experiment_cfg/metadata.json").is_file():
        raise ValueError("Actor checkpoint is missing normalization metadata")
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        if not (args.resume / "complete.json").is_file():
            raise ValueError("Resume requires a completed SVF checkpoint")
    return schedule


def svf_config_values(args) -> dict[str, Any]:
    return {"kappa": args.kappa, "g": args.g, "K": args.candidates,
            "t_min": args.t_min, "flow_steps": args.flow_steps,
            "guidance_clip": args.guidance_clip, "lambda_epsilon": args.lambda_epsilon}


def actor_tuning_values(args=None) -> dict[str, Any]:
    return {"mode": getattr(args, "actor_tuning", "full"),
            "rank": getattr(args, "lora_rank", 16),
            "alpha": getattr(args, "lora_alpha", 16.0),
            "dropout": getattr(args, "lora_dropout", 0.0)}


def lr_schedule_values(args=None) -> dict[str, Any]:
    return {"actor": getattr(args, "lr_scheduler", "constant"),
            "warmup_steps": getattr(args, "warmup_steps", 0),
            "min_lr_ratio": getattr(args, "min_lr_ratio", 0.0),
            "soft_value": "constant"}


def validate_lr_schedule(args):
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup-steps must be nonnegative and smaller than --steps")
    if not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("min-lr-ratio must be finite and between zero and one")
    if args.lr_scheduler == "constant" and (args.warmup_steps or args.min_lr_ratio):
        raise ValueError("warmup-steps and min-lr-ratio require the cosine actor scheduler")


def actor_lr_multiplier(completed_updates: int, *, total_steps: int,
                        config: dict[str, Any]) -> float:
    """LR for the next update, indexed by completed optimizer updates."""
    if config["actor"] == "constant":
        return 1.0
    warmup = config["warmup_steps"]
    if completed_updates < warmup:
        return completed_updates / warmup
    progress = min(1.0, max(0.0, (completed_updates - warmup) / (total_steps - warmup)))
    floor = config["min_lr_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def build_lr_scheduler(optimizer, args):
    import torch
    validate_lr_schedule(args)
    if [group.get("name") for group in optimizer.param_groups] != ["actor", "soft_value"]:
        raise ValueError("Scheduler expects actor and soft_value optimizer groups in that order")
    config = lr_schedule_values(args)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
        lambda completed: actor_lr_multiplier(completed, total_steps=args.steps, config=config),
        lambda completed: 1.0,
    ])


def restore_optimizer_and_scheduler(optimizer, scheduler, saved, args):
    """Restore both states and verify that the next LR belongs to the full horizon."""
    step = saved["step"]
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step <= args.steps:
        raise ValueError("Invalid checkpoint optimizer step")
    config = lr_schedule_values(args)
    base_lrs = [args.actor_lr, args.value_lr]
    expected_lrs = [args.actor_lr * actor_lr_multiplier(step, total_steps=args.steps, config=config),
                    args.value_lr]
    groups = saved["optimizer"]["param_groups"]
    if [group.get("name") for group in groups] != ["actor", "soft_value"]:
        raise ValueError("Checkpoint optimizer groups differ from actor/soft_value")
    def same_lrs(actual, expected):
        return len(actual) == len(expected) and all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-16) for a, b in zip(actual, expected))
    if not same_lrs([group["lr"] for group in groups], expected_lrs):
        raise ValueError("Checkpoint optimizer LR differs from the full-horizon schedule")
    state = saved.get("lr_scheduler")
    if state is None:
        # Earlier SVF checkpoints used a fixed LR and did not save a scheduler.
        if config != lr_schedule_values():
            raise ValueError("Cosine resume requires checkpoint lr_scheduler state")
        state = {**scheduler.state_dict(), "last_epoch": step, "_step_count": step + 1,
                 "_last_lr": expected_lrs}
    if (state.get("last_epoch") != step
            or not same_lrs(state.get("base_lrs", []), base_lrs)
            or not same_lrs(state.get("_last_lr", []), expected_lrs)):
        raise ValueError("Checkpoint scheduler step/LR differs from the training state")
    # Construct scheduler first, then load it and the optimizer: initialization
    # must never overwrite a restored, already-decayed optimizer LR.
    scheduler.load_state_dict(state)
    optimizer.load_state_dict(saved["optimizer"])


def run_identity(args, schedule) -> dict[str, Any]:
    return {"actor": args.actor, "critic": args.critic, "dataset_paths": args.dataset_path,
            "seed": args.seed, "steps": args.steps, "actor_lr": args.actor_lr,
            "value_lr": args.value_lr, "lr_schedule": lr_schedule_values(args),
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm, "data_config": "single_panda_gripper",
            "normalization": "actor_checkpoint_pinned", "mode": "joint_from_step_one",
            "batch": schedule, "svf": svf_config_values(args),
            "actor_tuning": actor_tuning_values(args),
            "soft_value_init": getattr(args, "soft_value_init", "random"),
            "soft_value_loss": getattr(args, "soft_value_loss", "mse")}


def assert_resume_compatible(saved: dict, expected: dict):
    # Missing fields in older checkpoints imply full actor tuning, random value
    # initialization, and scalar MSE. Never silently change architecture or loss.
    defaults = {"actor_tuning": actor_tuning_values(), "soft_value_init": "random",
                "soft_value_loss": "mse", "lr_schedule": lr_schedule_values()}
    saved = {**defaults, **saved}
    expected = {**defaults, **expected}
    if saved != expected:
        changed = sorted(k for k in set(saved) | set(expected) if saved.get(k) != expected.get(k))
        raise ValueError(f"Resume configuration differs: {', '.join(changed)}")


def assert_soft_value_resume_compatible(checkpoint: Path, expected: dict):
    """Reject changed HL-Gauss support before loading the saved model buffers."""
    config_path = checkpoint / "soft_value_config.json"
    if config_path.is_file():
        saved = json.loads(config_path.read_text())
        if not isinstance(saved, dict):
            raise ValueError("Invalid checkpoint soft_value_config.json: expected an object")
    elif expected.get("loss_type") == "mse":
        # Legacy scalar checkpoints predate support metadata; no bins are needed.
        saved = {"loss_type": "mse"}
    else:
        raise ValueError("HL-Gauss resume requires checkpoint soft_value_config.json")
    if saved != expected:
        changed = sorted(k for k in set(saved) | set(expected) if saved.get(k) != expected.get(k))
        raise ValueError(f"Resume soft-value configuration differs: {', '.join(changed)}")


def atomic_json(path: Path, data):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def build_dataset(args):
    from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
    from gr00t.data.schema import DatasetMetadata, DatasetStatisticalValues, EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP

    data_config = DATA_CONFIG_MAP["single_panda_gripper"]
    actor_metadata = json.loads((Path(args.actor) / "experiment_cfg/metadata.json").read_text())
    pinned = DatasetMetadata.model_validate(actor_metadata["new_embodiment"])
    # Validate with the exact dataset schema before constructing any dataset;
    # LeRobot otherwise falls back to scanning every parquet file on invalid stats.
    for path in args.dataset_path:
        for value in validate_dataset_metadata(Path(path)).values():
            DatasetStatisticalValues.model_validate(value)
    datasets = [LeRobotSingleDataset(
        dataset_path=path, modality_configs=data_config.modality_config(),
        transforms=data_config.transform(), embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend="decord",
    ) for path in args.dataset_path]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = LeRobotMixtureDataset(
            data_mixture=[(d, 1.0) for d in datasets], mode="train", seed=args.seed,
            balance_dataset_weights=True, balance_trajectory_weights=True,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        dataset.merged_metadata["new_embodiment"] = pinned
    # Mixture construction recomputes its statistics. Restore the actor's values
    # afterward, otherwise x_t and the fixed reference use different coordinates.
    for child in datasets:
        child.set_transforms_metadata(pinned)
    return DeterministicSamples(dataset, args.seed)


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def actor_tuning_provenance(model) -> dict[str, Any]:
    from dataclasses import asdict

    config = getattr(model, "actor_tuning", None)
    values = asdict(config) if config is not None else actor_tuning_values()
    return {**values, "target_modules": list(getattr(model, "lora_target_modules", []))}


def export_actor(model, destination: Path, actor_source: Path):
    destination.mkdir(parents=True, exist_ok=False)
    tuning = actor_tuning_provenance(model)
    kwargs = {"safe_serialization": True, "max_shard_size": "5GB"}
    if tuning["mode"] == "dit-lora":
        from gr00t.model.svf.lora import merged_actor_state_dict
        # Build vanilla weights on CPU; leave the live adapters and optimizer
        # untouched so checkpointing also works mid-run or before exact resume.
        kwargs["state_dict"] = merged_actor_state_dict(model.actor)
    model.actor.save_pretrained(destination, **kwargs)
    cfg = destination / "experiment_cfg"
    cfg.mkdir()
    shutil.copy2(actor_source / "experiment_cfg/metadata.json", cfg / "metadata.json")
    atomic_json(destination / "svf_actor_tuning.json", {
        **tuning, "source_actor": str(actor_source),
        "format": "merged_vanilla_gr00t" if tuning["mode"] == "dit-lora" else "vanilla_gr00t",
    })


def save_checkpoint(model, optimizer, *, output: Path, step: int, microbatches: int,
                    identity: dict, rank: int, world_size: int, elapsed: float,
                    wandb_id: str | None, scheduler=None):
    import torch
    import torch.distributed as dist

    if scheduler is not None and scheduler.last_epoch != step:
        raise ValueError("Scheduler and checkpoint optimizer step disagree")
    if scheduler is None and identity.get("lr_schedule", lr_schedule_values()) != lr_schedule_values():
        raise ValueError("Scheduled training requires scheduler state in checkpoints")
    target = output / f"checkpoint-{step}"
    partial = output / f".checkpoint-{step}.incomplete"
    if rank == 0:
        if target.exists() or partial.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
        partial.mkdir()
    if world_size > 1:
        dist.barrier()
    torch.save(capture_rng(), partial / f"rng-rank-{rank}.pt")
    if rank == 0:
        torch.save({"schema": 1, "step": step, "microbatches": microbatches,
                    "identity": identity, "actor_head": model.actor.action_head.state_dict(),
                    "soft_value": model.soft_value.state_dict(), "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "elapsed_seconds": elapsed, "wandb_run_id": wandb_id},
                   partial / "training_state.pt")
        export_actor(model, partial / "actor", Path(identity["actor"]))
        atomic_json(partial / "svf_config.json", identity["svf"])
        atomic_json(partial / "actor_tuning.json", actor_tuning_provenance(model))
        atomic_json(partial / "soft_value_initialization.json", getattr(
            model, "soft_value_initialization", {"mode": identity.get("soft_value_init", "random")}))
        value_config = getattr(model.soft_value, "loss_configuration", None)
        atomic_json(partial / "soft_value_config.json", value_config() if callable(value_config)
                    else {"loss_type": identity.get("soft_value_loss", "mse")})
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        atomic_json(partial / "complete.json", {"step": step, "world_size": world_size,
                                                "microbatches": microbatches})
        partial.rename(target)
        atomic_json(output / "latest_checkpoint.json", {"path": str(target), "step": step})
    if world_size > 1:
        dist.barrier()
    return target


def reduce_metrics(metrics, *, count: int, world_size: int):
    import torch
    import torch.distributed as dist
    names = sorted(metrics)
    values = torch.stack([metrics[name].float() / count for name in names])
    if world_size > 1:
        dist.all_reduce(values)
        values /= world_size
    return dict(zip(names, values.cpu().tolist()))


def main(argv=None):
    args = parser().parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    schedule = validate_args(args, world_size=(args.world_size or world_size) if args.validate_config else world_size)
    identity = run_identity(args, schedule)
    if args.validate_config:
        print(json.dumps(identity, indent=2))
        return 0
    if args.world_size is not None and args.world_size != world_size:
        raise ValueError("Use torchrun to set world size")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SVF training runs through sbatch; SLURM_JOB_ID is required")

    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if "HF_MODULES_CACHE" in os.environ:
        os.environ["HF_MODULES_CACHE"] = str(Path(os.environ["HF_MODULES_CACHE"]) / f"rank-{rank}")
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from gr00t.model.svf.objective import SVFConfig
    from gr00t.model.svf.model import JointSVFModel
    from gr00t.model.svf.lora import ActorTuningConfig
    from gr00t.model.transforms import DefaultDataCollator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SVF training")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    wandb_run = None
    owns_output = False
    status = {"state": "initializing", "step": 0, "steps": args.steps, "identity": identity}
    try:
        if rank == 0:
            if args.resume is None:
                output.mkdir(parents=True, exist_ok=False)
            else:
                output.mkdir(parents=True, exist_ok=True)
                if (output / "run_config.json").is_file():
                    assert_resume_compatible(json.loads((output / "run_config.json").read_text()), identity)
            owns_output = True
            atomic_json(output / "run_config.json", identity)
            atomic_json(output / "status.json", status)
        if world_size > 1:
            dist.barrier()
        dataset = build_dataset(args)
        model = JointSVFModel(args.actor, args.critic, SVFConfig(**svf_config_values(args)), device,
                              actor_tuning=ActorTuningConfig(**actor_tuning_values(args)),
                              soft_value_init=args.soft_value_init,
                              soft_value_loss=args.soft_value_loss)
        actor_params = [p for p in model.actor.action_head.parameters() if p.requires_grad]
        value_params = [p for p in model.soft_value.parameters() if p.requires_grad]
        ids = {id(p) for p in actor_params + value_params}
        if not actor_params or not value_params or len(ids) != len(actor_params + value_params):
            raise RuntimeError("Actor and value optimizer groups must be nonempty and disjoint")
        if any(p.requires_grad and id(p) not in ids for p in model.parameters()):
            raise RuntimeError("Unexpected trainable parameter outside actor head and soft value")
        if any(p.dtype != torch.float32 for p in actor_params + value_params):
            raise RuntimeError("Trainable weights must remain FP32 for AdamW")
        description = model.training_description()
        description["trainable_parameters"] = {
            "actor": sum(p.numel() for p in actor_params),
            "soft_value": sum(p.numel() for p in value_params),
        }
        if rank == 0:
            atomic_json(output / "model_description.json", description)
        optimizer = torch.optim.AdamW([
            {"params": actor_params, "lr": args.actor_lr, "name": "actor"},
            {"params": value_params, "lr": args.value_lr, "name": "soft_value"},
        ], weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args)
        start_step, completed_microbatches, previous_elapsed, wandb_id = 0, 0, 0.0, None
        if args.resume:
            # These files are created by this training script in the user's workspace.
            saved = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
            assert_resume_compatible(saved["identity"], identity)
            assert_soft_value_resume_compatible(args.resume, model.soft_value.loss_configuration())
            model.actor.action_head.load_state_dict(saved["actor_head"], strict=True)
            model.soft_value.load_state_dict(saved["soft_value"], strict=True)
            restore_optimizer_and_scheduler(optimizer, scheduler, saved, args)
            start_step, completed_microbatches = saved["step"], saved["microbatches"]
            previous_elapsed, wandb_id = saved["elapsed_seconds"], saved.get("wandb_run_id")
            if completed_microbatches != start_step * schedule["gradient_accumulation_steps"]:
                raise ValueError("Checkpoint data position disagrees with optimizer step")
            del saved
        if start_step >= args.steps:
            raise ValueError("Checkpoint already reached the requested training steps")
        train_model = DistributedDataParallel(model, device_ids=[local_rank],
                            broadcast_buffers=False, find_unused_parameters=False) if world_size > 1 else model
        model.train()
        # Models initialize identically, then stochastic training streams differ by rank.
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        final_step = min(args.steps, start_step + args.stop_after_steps) if args.stop_after_steps else args.steps
        accumulation = schedule["gradient_accumulation_steps"]
        sampler = MicrobatchSampler(start=completed_microbatches, stop=final_step * accumulation,
                                    rank=rank, world_size=world_size, size=args.microbatch_size)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=DefaultDataCollator(),
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=False,
                            generator=torch.Generator().manual_seed(args.seed + rank))
        batches = iter(loader)
        if args.resume:
            restore_rng(torch.load(args.resume / f"rng-rank-{rank}.pt", map_location="cpu", weights_only=False))
        if rank == 0 and args.report_to == "wandb":
            import wandb
            os.environ["WANDB_LOG_MODEL"] = "false"
            os.environ["WANDB_WATCH"] = "false"
            os.environ.setdefault("WANDB_MODE", "online")
            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                  name=args.run_name, config={**identity, "model": description}, dir=str(output),
                                  id=wandb_id, resume="must" if wandb_id else None)
            wandb_id = wandb_run.id
            atomic_json(output / "wandb_config.json", {"project": args.wandb_project,
                        "entity": args.wandb_entity, "run_id": wandb_id})
        begin = time.monotonic()
        step = start_step
        latest = None
        for step in range(start_step + 1, final_step + 1):
            optimizer.zero_grad(set_to_none=True)
            metrics = {}
            for micro in range(accumulation):
                inputs = next(batches)
                sync = train_model.no_sync() if world_size > 1 and micro < accumulation - 1 else nullcontext()
                with sync:
                    result = train_model(inputs)
                    loss = result["loss"]
                    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                    if world_size > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError(f"Nonfinite loss at step {step}")
                    (loss / accumulation).backward()
                for name, value in {"loss": loss.detach(), **result.get("metrics", {})}.items():
                    value = torch.as_tensor(value, device=device).detach().float().mean()
                    metrics[name] = metrics.get(name, torch.zeros((), device=device)) + value
                completed_microbatches += 1
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm,
                                                             error_if_nonfinite=True)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(value_params, args.max_grad_norm,
                                                             error_if_nonfinite=True)
            applied_lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
            optimizer.step()
            scheduler.step()
            elapsed = previous_elapsed + time.monotonic() - begin
            if step == start_step + 1 or step % args.log_steps == 0 or step == final_step:
                values = reduce_metrics(metrics, count=accumulation, world_size=world_size)
                if not all(math.isfinite(v) for v in values.values()):
                    raise FloatingPointError(f"Nonfinite metric at step {step}")
                peak_memory = torch.tensor([torch.cuda.max_memory_allocated(device),
                                            torch.cuda.max_memory_reserved(device)],
                                           device=device, dtype=torch.float64)
                if world_size > 1:
                    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
                max_allocated, max_reserved = peak_memory.cpu().tolist()
                seconds_per_step = (time.monotonic() - begin) / (step - start_step)
                values.update({"actor_grad_norm": float(actor_grad_norm),
                               "value_grad_norm": float(value_grad_norm), "actor_lr": applied_lrs["actor"],
                               "value_lr": applied_lrs["soft_value"],
                               "actor_lr_next_update": scheduler.get_last_lr()[0],
                               "value_lr_next_update": scheduler.get_last_lr()[1],
                               "walltime_seconds": elapsed,
                               "samples_seen": step * args.global_batch_size,
                               "steps_per_second": 1.0 / max(seconds_per_step, 1e-9),
                               "seconds_per_step": seconds_per_step,
                               "eta_seconds": (args.steps - step) * seconds_per_step,
                               "stage_eta_seconds": (final_step - step) * seconds_per_step,
                               "max_gpu_allocated_gib": max_allocated / 2**30,
                               "max_gpu_reserved_gib": max_reserved / 2**30})
                if rank == 0:
                    status = {"state": "running", "step": step, "steps": args.steps,
                              "run_until_step": final_step,
                              "elapsed_seconds": elapsed, "metrics": values,
                              "latest_checkpoint": str(latest) if latest else None}
                    atomic_json(output / "status.json", status)
                    with (output / "metrics.jsonl").open("a") as log:
                        log.write(json.dumps({"step": step, **values}, allow_nan=False) + "\n")
                    print(json.dumps({"step": step, **values}), flush=True)
                    if wandb_run:
                        wandb_run.log({f"train/{k}": v for k, v in values.items()}, step=step)
            if step % args.save_steps == 0 or step == final_step:
                latest = save_checkpoint(model, optimizer, output=output, step=step,
                    microbatches=completed_microbatches, identity=identity, rank=rank,
                    world_size=world_size, elapsed=elapsed, wandb_id=wandb_id, scheduler=scheduler)
        if rank == 0:
            # The final actor path is directly consumable by RoboCasa policy loading.
            if final_step == args.steps:
                final_actor = output / "actor"
                if final_actor.exists():
                    raise FileExistsError(final_actor)
                final_actor.symlink_to((latest / "actor").relative_to(output), target_is_directory=True)
            status.update(state="completed" if final_step == args.steps else "paused",
                          step=step, latest_checkpoint=str(latest),
                          elapsed_seconds=previous_elapsed + time.monotonic() - begin)
            atomic_json(output / "status.json", status)
        if world_size > 1:
            dist.barrier()
        if wandb_run:
            wandb_run.finish()
        return 0
    except BaseException as error:
        if rank == 0 and owns_output and output.is_dir():
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(output / "status.json", status)
        if wandb_run:
            wandb_run.finish(exit_code=1)
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
