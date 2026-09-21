#!/usr/bin/env python3
"""Measure DEAS critic training speed across batch sizes, loader workers and GPU counts.

Each case is a real `scripts/gr00t_deas_critic_finetune.py` run. Speed is read
from the `performance.jsonl` that TrainingMetricsCallback writes: one line every
`logging_steps` optimizer steps, each carrying the seconds-per-step for that
window alone. The first window is discarded as warm-up, so model load, CUDA
context creation and the first-step allocator growth do not pollute the number.

These are end-to-end training rates. They include data loading, host-to-device
copies, forward, backward and the optimizer step, which is what matters for
planning a run; they are not kernel benchmarks.

    local_train/benchmark_train.py --batch-sizes 2 4 8 --steps 40

Every case leaves a checkpoint behind, roughly 5-10 GB: TrainRunner.train always
calls safe_save_model_for_hf_trainer, and transformers saves again when
global_step reaches max_steps. Rather than fight either of those, the script
deletes the large files once the timings have been read. Pass --keep-checkpoints
to retain them.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

LOCAL_TRAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_TRAIN_DIR.parent
DEAS_ROOT = REPO_ROOT / "deas-gr00t15"
TRAIN_SCRIPT = DEAS_ROOT / "scripts" / "gr00t_deas_critic_finetune.py"
ENV_SCRIPT = REPO_ROOT / "local_eval" / "env.sh"
# gr00t_deas_critic_finetune.py hardcodes TrainingArguments(logging_steps=10.0),
# which is also the width of each performance.jsonl window.
LOGGING_STEPS = 10
# Large files a finished case leaves behind, removed unless --keep-checkpoints.
BULK = ("model.safetensors", "optimizer.pt", "training_args.bin", "rng_state.pth", "scheduler.pt")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", nargs="+",
                        default=["/home/junhyeong/data/fake_robocasa_rl"],
                        help="One or more LeRobot datasets; several become a mixture dataset")
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--data-config", default="single_panda_gripper_rl")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Default: local_outputs/train_benchmark/<timestamp>")
    parser.add_argument("--batch-sizes", nargs="+", type=positive, default=[2],
                        help="Per-GPU batch sizes to sweep")
    parser.add_argument("--workers", nargs="+", type=nonnegative, default=[0],
                        help="Dataloader worker counts to sweep")
    parser.add_argument("--gpu-counts", nargs="+", type=positive, default=[1],
                        help="GPU counts to sweep; each takes the first N of --gpus")
    parser.add_argument("--gpus", nargs="+", type=nonnegative, default=[0],
                        help="Pool of GPU indices the cases are placed on")
    parser.add_argument("--steps", type=positive, default=40,
                        help=f"Optimizer steps per case; a multiple of {LOGGING_STEPS} measures cleanly")
    parser.add_argument("--warmup-windows", type=positive, default=1,
                        help=f"Leading {LOGGING_STEPS}-step windows to discard")
    parser.add_argument("--critic-action-horizon", type=positive, default=4)
    parser.add_argument("--conda-env", default="deas-rc")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep each case's checkpoint instead of deleting it after timing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    windows = args.steps // LOGGING_STEPS
    if windows <= args.warmup_windows:
        parser.error(
            f"--steps {args.steps} gives {windows} windows of {LOGGING_STEPS} steps, "
            f"which leaves nothing after discarding {args.warmup_windows}; raise --steps")
    for count in args.gpu_counts:
        if count > len(args.gpus):
            parser.error(f"--gpu-counts {count} needs at least {count} entries in --gpus")
    return args


def training_environment(devices):
    """local_eval/env.sh, plus the GPU selection for this case."""
    dump = subprocess.run(["bash", "-c", f"set -a; source {ENV_SCRIPT}; env -0"],
                          check=True, capture_output=True).stdout
    environment = dict(e.split("=", 1) for e in dump.decode().split("\0") if "=" in e)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    return environment


def case_command(args, output_dir, batch_size, workers, gpu_count):
    # The training script re-executes itself under torchrun when gpu_count > 1.
    return [
        "conda", "run", "--no-capture-output", "-n", args.conda_env,
        "python", str(TRAIN_SCRIPT),
        "--dataset-path", *args.dataset_path,
        "--output-dir", str(output_dir),
        "--base-model-path", args.base_model,
        "--data-config", args.data_config,
        "--num-gpus", str(gpu_count),
        "--batch-size", str(batch_size),
        "--max-steps", str(args.steps),
        # No mid-run checkpoint: the final save is unavoidable, extra ones are not.
        "--save-steps", str(args.steps + 1),
        "--critic-action-horizon", str(args.critic_action_horizon),
        "--dataloader-num-workers", str(workers),
        "--report-to", "tensorboard",
        "--run-name", output_dir.name,
    ]


def read_windows(output_dir):
    path = output_dir / "performance.jsonl"
    if not path.is_file():
        return []
    windows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "timing/seconds_per_step" in record:
            windows.append(record)
    return windows


def summarise(args, windows, batch_size, gpu_count):
    measured = windows[args.warmup_windows:]
    if not measured:
        return {"status": "no measured windows"}
    per_step = [w["timing/seconds_per_step"] for w in measured]
    samples = [w["timing/samples_per_second_global"] for w in measured]
    return {
        "status": "ok",
        "measured_windows": len(measured),
        "seconds_per_step_mean": statistics.mean(per_step),
        "seconds_per_step_median": statistics.median(per_step),
        "seconds_per_step_stdev": statistics.stdev(per_step) if len(per_step) > 1 else 0.0,
        "steps_per_second": 1.0 / statistics.mean(per_step),
        "samples_per_second": statistics.mean(samples),
        "global_batch": batch_size * gpu_count,
        "peak_allocated_gib": max(w.get("memory/rank0_peak_allocated_gib", 0) for w in measured),
        "peak_reserved_gib": max(w.get("memory/rank0_peak_reserved_gib", 0) for w in measured),
    }


def prune(output_dir):
    """Drop the multi-gigabyte artefacts, keep logs and metrics."""
    freed = 0
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            freed += sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            shutil.rmtree(child)
        elif child.is_file() and child.name in BULK:
            freed += child.stat().st_size
            child.unlink()
    return freed


def write_outputs(root, report):
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [dict(case["case"], **{k: v for k, v in case["result"].items()})
            for case in report["cases"]]
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def print_table(report):
    header = f"{'gpus':>4} {'batch':>6} {'workers':>8} {'s/step':>9} {'steps/s':>9} {'samples/s':>10} {'peak GiB':>9}"
    print("\n" + header)
    print("-" * len(header))
    for case in report["cases"]:
        spec, result = case["case"], case["result"]
        if result.get("status") != "ok":
            print(f"{spec['gpu_count']:>4} {spec['batch_size']:>6} {spec['workers']:>8}"
                  f"   {result.get('status', 'failed')}")
            continue
        print(f"{spec['gpu_count']:>4} {spec['batch_size']:>6} {spec['workers']:>8} "
              f"{result['seconds_per_step_mean']:>9.3f} {result['steps_per_second']:>9.3f} "
              f"{result['samples_per_second']:>10.2f} {result['peak_allocated_gib']:>9.2f}")


def main(argv=None):
    args = arguments(argv)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.output_root or REPO_ROOT / "local_outputs" / "train_benchmark" / stamp).resolve()

    cases = [{"gpu_count": g, "batch_size": b, "workers": w}
             for g in args.gpu_counts for b in args.batch_sizes for w in args.workers]

    if args.dry_run:
        for spec in cases:
            directory = root / f"g{spec['gpu_count']}_b{spec['batch_size']}_w{spec['workers']}"
            print(" ".join(case_command(args, directory, spec["batch_size"],
                                        spec["workers"], spec["gpu_count"])))
        return 0

    root.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "settings": {
            "dataset_path": args.dataset_path, "base_model": args.base_model,
            "data_config": args.data_config, "steps": args.steps,
            "logging_steps": LOGGING_STEPS, "warmup_windows": args.warmup_windows,
            "critic_action_horizon": args.critic_action_horizon, "gpu_pool": args.gpus,
        },
        "note": ("End-to-end training rates including data loading and the optimizer step. "
                 "The first window is warm-up and is excluded."),
        "cases": [],
    }

    for spec in cases:
        name = f"g{spec['gpu_count']}_b{spec['batch_size']}_w{spec['workers']}"
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        devices = args.gpus[:spec["gpu_count"]]
        command = case_command(args, directory, spec["batch_size"], spec["workers"],
                               spec["gpu_count"])
        print(f"\n=== {name}: GPUs {devices}, batch {spec['batch_size']}/GPU, "
              f"{spec['workers']} workers, {args.steps} steps")
        started = time.monotonic()
        with (directory / "train.log").open("w") as log:
            finished = subprocess.run(command, cwd=str(DEAS_ROOT),
                                      env=training_environment(devices),
                                      stdout=log, stderr=subprocess.STDOUT)
        walltime = time.monotonic() - started

        if finished.returncode != 0:
            result = {"status": f"exit {finished.returncode}; see {directory / 'train.log'}"}
        else:
            windows = read_windows(directory)
            result = summarise(args, windows, spec["batch_size"], spec["gpu_count"])
            result["all_windows"] = [w["timing/seconds_per_step"] for w in windows]
        result["walltime_seconds"] = walltime
        if not args.keep_checkpoints:
            result["freed_bytes"] = prune(directory)
        report["cases"].append({"case": spec, "output_dir": str(directory), "result": result})
        write_outputs(root, report)
        print(f"    {result.get('status')}"
              + (f", {result['seconds_per_step_mean']:.3f} s/step"
                 if result.get("status") == "ok" else ""))

    print_table(report)
    print(f"\nWrote {root / 'summary.json'} and {root / 'summary.csv'}")
    return 0 if all(c["result"].get("status") == "ok" for c in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
