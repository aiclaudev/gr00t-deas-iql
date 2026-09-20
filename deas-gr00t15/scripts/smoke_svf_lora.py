#!/usr/bin/env python3
"""Bounded, one-GPU LoRA correctness check; never a throughput benchmark.

Runs exactly two optimizer updates on one CoffeeSetupMug sample with the
production SVF candidate settings. Production train_svf.py retains its Slurm
guard. Invoke this script in the foreground under an external timeout as well.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="One CoffeeSetupMug demos directory, with existing metadata")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory underneath this repository's output/code-checks")
    args = parser.parse_args()
    for key in ("actor", "critic", "dataset_path", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.output.is_relative_to((REPO_ROOT / "output/code-checks").resolve()):
        parser.error("Smoke artifacts must stay in the repository's output/code-checks directory")
    for path in (args.actor, args.critic, args.dataset_path):
        if not path.is_dir():
            parser.error(f"Missing input directory: {path}")
    if args.dataset_path.name != "CoffeeSetupMug" or args.dataset_path.parent.name != "demos":
        parser.error("This bounded check accepts only the CoffeeSetupMug demos dataset")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("The login correctness check must run as one process")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].strip():
        parser.error("Explicitly select exactly one checked GPU with CUDA_VISIBLE_DEVICES")
    return args


def write_report(path, report):
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def assert_tree_equal(actual, expected):
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected.cpu()):
            raise AssertionError("Checkpoint tensor differs after restore")
    elif isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError("Checkpoint dictionary keys differ after restore")
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("Checkpoint sequence differs after restore")
        for a, b in zip(actual, expected):
            assert_tree_equal(a, b)
    elif actual != expected:
        raise AssertionError("Checkpoint scalar differs after restore")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    report = {
        "state": "initializing", "purpose": "login_lora_correctness_only",
        "actor": str(args.actor), "critic": str(args.critic),
        "dataset": str(args.dataset_path), "seed": 42,
        "batch_size": 1, "optimizer_updates": 2, "num_workers": 0,
        "candidates": 8, "flow_steps": 10, "wandb": False,
        "checks": {}, "updates": [],
    }
    write_report(report_path, report)

    def deadline(_signum, _frame):
        raise TimeoutError("Bounded login correctness check exceeded its 540-second guard")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(540)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(args.output / "hf_modules"))
    scratch = args.output / "scratch"
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch)
    os.environ["TMP"] = str(scratch)
    os.environ["TEMP"] = str(scratch)

    try:
        import numpy as np
        import torch
        from gr00t.model.svf.adapters import prepare_head_context, velocity_from_head
        from gr00t.model.svf.lora import ActorTuningConfig, is_lora_parameter
        from gr00t.model.svf.model import JointSVFModel
        from gr00t.model.svf.objective import SVFConfig
        from gr00t.model.transforms import DefaultDataCollator
        from scripts.train_svf import build_dataset

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device must be visible for the login check")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        dataset = build_dataset(SimpleNamespace(
            actor=str(args.actor), dataset_path=[str(args.dataset_path)], seed=42,
        ))
        # Deliberately no DataLoader workers/prefetch or sample loop.
        inputs = DefaultDataCollator()([dataset[0]])
        if inputs["action"].shape[0] != 1:
            raise AssertionError("Correctness sample unexpectedly has a batch larger than one")
        model = JointSVFModel(args.actor, args.critic, SVFConfig(), device,
            actor_tuning=ActorTuningConfig(mode="dit-lora", rank=16, alpha=16, dropout=0.0))
        named = dict(model.named_parameters())
        adapters = {name: parameter for name, parameter in named.items()
                    if is_lora_parameter(name)}
        value_params = dict(model.soft_value.named_parameters())
        if not adapters or not value_params:
            raise AssertionError("Missing LoRA or soft-value parameters")
        for name, parameter in named.items():
            should_train = name in adapters or name.startswith("soft_value.")
            if parameter.requires_grad != should_train:
                raise AssertionError(f"Unexpected trainability: {name}")
            if should_train and parameter.dtype != torch.float32:
                raise AssertionError(f"Trainable parameter is not FP32: {name}")
            if name in adapters and not name.startswith("actor.action_head.model."):
                raise AssertionError(f"LoRA is outside the actor DiT: {name}")
        b_names = [name for name in adapters if name.endswith(".lora_B")]
        a_names = [name for name in adapters if name.endswith(".lora_A")]
        if not b_names or not a_names or not all(torch.count_nonzero(adapters[n]).item() == 0 for n in b_names):
            raise AssertionError("LoRA must initialize with all B matrices at zero")
        report["checks"].update(initial_lora_B_zero=True, only_dit_lora_and_soft_value_trainable=True,
                                trainable_parameters_fp32=True)
        report["trainable_parameters"] = {
            "lora": sum(p.numel() for p in adapters.values()),
            "soft_value": sum(p.numel() for p in value_params.values()),
        }
        frozen_versions = {name: p._version for name, p in named.items() if not p.requires_grad}
        initial_values = {name: p.detach().cpu().clone() for name, p in value_params.items()}
        groups = [
            {"params": list(adapters.values()), "lr": 1e-5, "name": "actor"},
            {"params": list(value_params.values()), "lr": 3e-4, "name": "soft_value"},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
        report["state"] = "checking_two_updates"
        write_report(report_path, report)
        model.train()
        for update in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            result = model(inputs)
            loss = result["loss"]
            metrics = {name: float(torch.as_tensor(value).detach().float().mean().cpu())
                       for name, value in result.get("metrics", {}).items()}
            metrics["loss"] = float(loss.detach().float().cpu())
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite SVF loss or diagnostic metric")
            loss.backward()
            for name, p in named.items():
                if not p.requires_grad and p.grad is not None:
                    raise AssertionError(f"Frozen parameter received a gradient: {name}")
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f"Nonfinite gradient: {name}")
            if update == 2:
                a_has_gradient = any(adapters[n].grad is not None and
                                    torch.count_nonzero(adapters[n].grad).item() > 0 for n in a_names)
                if not a_has_gradient:
                    raise AssertionError("No LoRA A gradient after B's first update")
                report["checks"]["lora_A_receives_gradient_after_B_update"] = True
            if not any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                       for p in value_params.values()):
                raise AssertionError("Soft value did not receive a gradient")
            torch.nn.utils.clip_grad_norm_(list(adapters.values()), 1.0, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(list(value_params.values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            report["updates"].append({"update": update, **metrics})
            write_report(report_path, report)
            print(json.dumps({"correctness_update": update, **metrics}, allow_nan=False), flush=True)
        if not any(torch.count_nonzero(adapters[n]).item() > 0 for n in b_names):
            raise AssertionError("LoRA B did not change after the optimizer updates")
        if not any(not torch.equal(p.detach().cpu(), initial_values[n]) for n, p in value_params.items()):
            raise AssertionError("Soft value did not change after the optimizer updates")
        if any(named[n]._version != version for n, version in frozen_versions.items()):
            raise AssertionError("A frozen parameter was modified during the updates")
        report["checks"].update(lora_B_updated=True, soft_value_updated=True,
                                frozen_parameters_unchanged=True, no_frozen_gradients=True)
        del initial_values, result, loss
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        # Reuse one fixed observation context. This checks restoration of the
        # actual DiT prediction without another expensive SVF Monte Carlo draw.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_inputs, action_inputs = model.actor.prepare_input(inputs)
            raw = model.actor.backbone(backbone_inputs)
            context = prepare_head_context(model.actor.action_head, raw, action_inputs)
        probe_x = torch.zeros_like(action_inputs.action, dtype=torch.float32)
        probe_t = torch.full((1,), 0.5, device=device)

        def predictions():
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    velocity = velocity_from_head(model.actor.action_head, context, probe_x, probe_t)
                value = model.soft_value(torch.zeros(1, 64, device=device),
                                         torch.zeros(1, 64, device=device), probe_x, probe_t)
            return velocity.detach().float().cpu(), value.detach().float().cpu()

        before_predictions = predictions()
        snapshot = {"schema": 1, "updates": 2,
                    "adapters": {name: p.detach().cpu().clone() for name, p in adapters.items()},
                    "soft_value": cpu_tree(model.soft_value.state_dict()),
                    "optimizer": cpu_tree(optimizer.state_dict())}
        checkpoint = args.output / "smoke_state.pt"
        torch.save(snapshot, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert_tree_equal(loaded, snapshot)
        with torch.no_grad():
            for p in adapters.values():
                p.add_(0.01)
            for p in value_params.values():
                p.add_(0.01)
            for name, p in adapters.items():
                p.copy_(loaded["adapters"][name])
        model.soft_value.load_state_dict(loaded["soft_value"], strict=True)
        restored_optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
        restored_optimizer.load_state_dict(loaded["optimizer"])
        assert_tree_equal({name: p.detach().cpu() for name, p in adapters.items()}, loaded["adapters"])
        assert_tree_equal(model.soft_value.state_dict(), loaded["soft_value"])
        assert_tree_equal(restored_optimizer.state_dict(), loaded["optimizer"])
        after_predictions = predictions()
        differences = {}
        for name, before, after in zip(("actor_velocity", "soft_value"), before_predictions, after_predictions):
            if not torch.isfinite(after).all() or not torch.allclose(before, after, atol=1e-5, rtol=1e-5):
                raise AssertionError(f"Restored {name} prediction differs")
            differences[name] = float((before - after).abs().max())
        report["checks"].update(adapter_and_value_checkpoint_exact=True, optimizer_checkpoint_exact=True,
                                restored_predictions_match=True)
        report["restored_prediction_max_abs_difference"] = differences
        report["peak_vram_gib"] = {"allocated": torch.cuda.max_memory_allocated(device) / 2**30,
                                   "reserved": torch.cuda.max_memory_reserved(device) / 2**30}
        report["checkpoint"] = str(checkpoint)
        report["state"] = "passed"
        write_report(report_path, report)
        print(json.dumps({"correctness": "passed", "report": str(report_path)}, allow_nan=False), flush=True)
        return 0
    except BaseException as exc:
        report.update(state="failed", error_type=type(exc).__name__, error=str(exc))
        write_report(report_path, report)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
