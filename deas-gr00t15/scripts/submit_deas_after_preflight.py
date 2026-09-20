#!/usr/bin/env python3
"""Verify completed calibration jobs, then submit the authorized seed sweep once."""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAL_ROOT = Path("/home/nas_main/dohyunlee")
STAGES = {"bc-demo": "01-bc-demo", "bc-rollout": "02-bc-rollout", "critic": "03-critic"}
PROJECT = "gr00t1.5 finetune"
ENTITY = "aiclaudev"
PREFLIGHT_STEPS = 50
RESULT_PREFIX = "DEAS_SWEEP_RESULT="


class PendingVerification(RuntimeError):
    """A finished job's accounting or uploaded summary has not propagated yet."""


def retry_pending(check, deadline):
    for attempt in range(3):
        if time.monotonic() >= deadline:
            raise TimeoutError("Preflight accounting/upload verification exceeded 90 seconds")
        try:
            return check()
        except PendingVerification:
            if attempt == 2:
                raise
            time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))


def contained(path, parent):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(parent) or resolved == parent:
        raise ValueError(f"Path escapes its expected directory: {path}")
    return resolved


def read_json(path, parent):
    path = contained(path, parent)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty JSON file: {path}")
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def finite_number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or non-finite measurement: {name}")
    if positive and value <= 0:
        raise ValueError(f"Measurement must be positive: {name}")
    return float(value)


def read_manifests(root):
    arguments_path = contained(root / "arguments.tsv", root)
    with arguments_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    arguments = {row["key"]: row["value"] for row in rows}
    if len(arguments) != len(rows):
        raise ValueError("Duplicate keys in preflight arguments.tsv")
    expected = {
        "repo": str(REPO_ROOT), "gpus": "4", "batch_per_gpu": "32", "global_batch": "128",
        "steps_per_stage": "50", "seed": "42", "account": "sub", "qos": "own",
        "wandb_project": PROJECT, "wandb_entity": ENTITY, "wandb_mode": "online",
        "wandb_log_model": "false",
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            raise ValueError(f"Unexpected preflight setting {key}: expected {value!r}")
    with contained(root / "jobs.tsv", root).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    jobs = {row["stage"]: row["job_id"] for row in rows}
    if len(rows) != len(STAGES) or set(jobs) != set(STAGES):
        raise ValueError("jobs.tsv must contain exactly the three preflight stages")
    if len(set(jobs.values())) != len(STAGES) or any(not re.fullmatch(r"[0-9]+", jid) for jid in jobs.values()):
        raise ValueError("Invalid or duplicate preflight job IDs")
    return jobs


def read_accounting(jobs, deadline):
    command = [
        "sacct", "-X", "--noheader", "--parsable2", "--jobs", ",".join(jobs.values()),
        "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True,
                                timeout=min(15.0, max(0.1, deadline - time.monotonic())))
    except subprocess.SubprocessError:
        raise PendingVerification("sacct is temporarily unavailable") from None
    records = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4 or fields[0] not in jobs.values() or fields[0] in records:
            raise ValueError("Unexpected or duplicate sacct record")
        jid, state, elapsed, exit_code = fields
        if state in ("PENDING", "CONFIGURING", "RUNNING", "COMPLETING"):
            raise PendingVerification(f"sacct completion for job {jid} has not propagated")
        if state != "COMPLETED" or exit_code != "0:0" or not elapsed.isdecimal():
            raise ValueError(f"Preflight job {jid} is not successfully completed with a measured duration")
        if int(elapsed) <= 0:
            raise PendingVerification(f"sacct duration for job {jid} has not propagated")
        records[jid] = {"state": state, "elapsed_seconds": int(elapsed), "exit_code": exit_code}
    if set(records) != set(jobs.values()):
        raise PendingVerification("sacct does not yet report all completed preflight jobs")
    return records


def verify_model(stage_dir):
    config = read_json(stage_dir / "config.json", stage_dir)
    if not config:
        raise ValueError(f"Empty model configuration: {stage_dir}")
    indexes = [stage_dir / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")]
    indexes = [path for path in indexes if path.is_file()]
    single_files = [stage_dir / name for name in ("model.safetensors", "pytorch_model.bin")]
    single_files = [path for path in single_files if path.is_file()]
    if len(indexes) + len(single_files) != 1:
        raise ValueError(f"Expected one saved model or shard index: {stage_dir}")
    if single_files:
        model = contained(single_files[0], stage_dir)
        if model.stat().st_size <= 0:
            raise ValueError(f"Empty saved model: {model}")
        return {"index": None, "shards": [model.name]}
    index = read_json(indexes[0], stage_dir)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Empty model weight map: {indexes[0]}")
    shards = set()
    for filename in weight_map.values():
        if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError("Model shard index contains an unexpected filename")
        shard = contained(stage_dir / filename, stage_dir)
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise ValueError(f"Missing or empty model shard: {shard}")
        shards.add(filename)
    return {"index": indexes[0].name, "shards": sorted(shards)}


def inspect_stage(stage_dir, elapsed, target_steps):
    state = read_json(stage_dir / "trainer_state.json", stage_dir)
    if state.get("global_step") != PREFLIGHT_STEPS:
        raise ValueError(f"Preflight did not finish exactly 50 steps: {stage_dir}")
    losses = [entry["train_loss"] for entry in state.get("log_history", [])
              if "train_loss" in entry and entry.get("step") == PREFLIGHT_STEPS]
    if len(losses) != 1:
        raise ValueError(f"Missing or ambiguous final train_loss: {stage_dir}")
    loss = finite_number(losses[0], "final train_loss")
    model = verify_model(stage_dir)
    wandb_config = read_json(stage_dir / "wandb_config.json", stage_dir)
    for key, expected in (("entity", ENTITY), ("project", PROJECT), ("mode", "online")):
        if wandb_config.get(key) != expected:
            raise ValueError(f"Unexpected W&B {key} for {stage_dir.name}")
    if not isinstance(wandb_config.get("run_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", wandb_config["run_id"]):
        raise ValueError(f"Missing or invalid W&B run ID for {stage_dir.name}")
    measurements = {}
    with contained(stage_dir / "performance.jsonl", stage_dir).open() as stream:
        for line in stream:
            record = json.loads(line)
            step = record.get("step")
            if step in (30, 40, 50):
                if step in measurements:
                    raise ValueError(f"Duplicate performance measurement at step {step}")
                measurements[step] = record
    if set(measurements) != {30, 40, 50}:
        raise ValueError(f"Missing 30/40/50-step performance measurements: {stage_dir}")
    rates = {str(step): finite_number(record.get("timing/seconds_per_step"), f"step {step} rate", True)
             for step, record in measurements.items()}
    last = measurements[50]
    train_elapsed = finite_number(last.get("timing/train_elapsed_seconds"), "train elapsed", True)
    worker_elapsed = finite_number(last.get("timing/worker_script_elapsed_seconds"), "worker elapsed", True)
    startup = worker_elapsed - train_elapsed
    if startup < 0:
        raise ValueError("Worker duration is shorter than training duration")
    cleanup_save = max(elapsed - worker_elapsed, 0.0)
    checkpoint_count = target_steps // 1000 + 1
    rate = max(rates.values())
    unrounded = 1.3 * (rate * target_steps + startup + checkpoint_count * cleanup_save)
    budget_seconds = max(1800, math.ceil(unrounded / 300) * 300)
    hours, remainder = divmod(budget_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return {
        "trainer_global_step": PREFLIGHT_STEPS, "train_loss": loss, "model": model,
        "wandb": wandb_config, "seconds_per_step_samples": rates, "seconds_per_step": rate,
        "train_elapsed_seconds": train_elapsed, "worker_script_elapsed_seconds": worker_elapsed,
        "startup_seconds": startup, "cleanup_and_save_seconds": cleanup_save,
        "checkpoint_save_count_budget": checkpoint_count, "unrounded_budget_seconds": unrounded,
        "time_limit_seconds": budget_seconds, "time_limit": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
    }


def verify_wandb(stages, deadline):
    import wandb

    api = wandb.Api(timeout=min(10.0, max(0.1, deadline - time.monotonic())))
    for name, stage in stages.items():
        if time.monotonic() >= deadline:
            raise TimeoutError("Preflight accounting/upload verification exceeded 90 seconds")
        run_id = stage["wandb"]["run_id"]
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            summary = dict(run.summary)
            config = dict(run.config)
            remote_state = run.state
        except Exception:
            raise PendingVerification(f"W&B verification request failed for {name}") from None
        for key, expected in (("seed", 42), ("num_gpus", 4), ("batch_size", 32)):
            if config.get(key) != expected:
                raise ValueError(f"W&B run config mismatch for {name}: {key}")
        uploaded_step = summary.get("train/global_step")
        if remote_state in ("failed", "crashed", "killed") or (uploaded_step is not None and uploaded_step > PREFLIGHT_STEPS):
            raise ValueError(f"W&B reports an unsuccessful or unexpected run for {name}")
        if remote_state != "finished" or uploaded_step != PREFLIGHT_STEPS or "train_loss" not in summary:
            raise PendingVerification(f"W&B has not confirmed the finished 50-step run for {name}")
        uploaded_loss = finite_number(summary["train_loss"], f"uploaded {name} train_loss")
        stage["wandb_verified"] = {
            "state": remote_state, "global_step": summary["train/global_step"], "train_loss": uploaded_loss,
            "seed": config["seed"], "num_gpus": config["num_gpus"], "batch_size": config["batch_size"],
        }


def write_json(path, value):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def submit_sweep(production_dir, command, seeds):
    last_result = None
    with (production_dir / "submission.log").open("x") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if line.startswith(RESULT_PREFIX):
                last_result = line[len(RESULT_PREFIX):].strip()
        returncode = process.wait()
    if last_result is not None:
        summary = json.loads(last_result)
        write_json(production_dir / "summary.json", summary)
    else:
        raise RuntimeError("No sweep result was returned; inspect submission.log before any manual retry")
    if returncode:
        raise RuntimeError(f"Sweep submission failed with exit {returncode}; partial job IDs are retained")
    runs = summary.get("runs", [])
    if [run.get("seed") for run in runs] != seeds:
        raise ValueError("Sweep result does not match requested seeds; do not repeat the submission")
    job_ids = [str(run["job_ids"][stage]) for run in runs for stage in ("bc_demo", "bc_rollout", "critic")]
    if len(set(job_ids)) != len(seeds) * 3 or any(not re.fullmatch(r"[0-9]+", jid) for jid in job_ids):
        raise ValueError("Invalid or duplicate production job IDs; inspect the retained submission log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", required=True, type=Path)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--target-steps", type=int, default=10000)
    args = parser.parse_args()
    if not args.preflight_root.is_absolute() or not REPO_ROOT.is_relative_to(PERSONAL_ROOT):
        parser.error("Use an absolute preflight path inside this personal repository's output directory")
    root = contained(args.preflight_root, (REPO_ROOT / "output").resolve(strict=True))
    if not root.is_dir():
        parser.error("Preflight root must be a directory")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", args.seeds):
        parser.error("--seeds must be comma-separated nonnegative integers without leading zeros")
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or args.target_steps <= 0:
        parser.error("Seeds must be distinct and --target-steps must be positive")
    production_dir = root / "production"
    if production_dir.exists() or production_dir.is_symlink():
        raise FileExistsError("production already exists; refusing a potentially duplicate submission")
    jobs = read_manifests(root)
    verification_deadline = time.monotonic() + 90.0
    accounting = retry_pending(lambda: read_accounting(jobs, verification_deadline), verification_deadline)
    stages = {}
    for name, dirname in STAGES.items():
        stage_dir = contained(root / dirname, root)
        stages[name] = inspect_stage(stage_dir, accounting[jobs[name]]["elapsed_seconds"], args.target_steps)
        stages[name].update({"job_id": jobs[name], "accounting": accounting[jobs[name]]})
    retry_pending(lambda: verify_wandb(stages, verification_deadline), verification_deadline)
    command = ["bash", str(REPO_ROOT / "bash_scripts/submit_deas_seed_sweep.sh"),
               "--seeds", args.seeds, "--gpus", "4", "--steps", str(args.target_steps)]
    for name in STAGES:
        command.extend([f"--time-{name}", stages[name]["time_limit"]])
    plan = {
        "schema": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_root": str(root), "seeds": seeds, "target_steps": args.target_steps,
        "gpus_per_job": 4, "global_batch": 128, "production_job_count": len(seeds) * 3,
        "formula": "max(1800, ceil(1.3 * (max(rate30, rate40, rate50) * target_steps + startup + (floor(target_steps/1000)+1) * cleanup_and_save) / 300) * 300)",
        "stages": stages, "submission_command": command,
    }
    # This is a one-time submission claim and completed-job output, never live worker control.
    production_dir.mkdir()
    write_json(production_dir / "plan.json", plan)
    for name, stage in stages.items():
        print(f"{name}: completed calibration job {stage['job_id']}; time limit {stage['time_limit']}", flush=True)
    submit_sweep(production_dir, command, seeds)
    print(f"Production submission recorded in {production_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preflight gate failed: {exc}", file=sys.stderr)
        sys.exit(1)
