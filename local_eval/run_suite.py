#!/usr/bin/env python3
"""Run RoboCasa evaluations for several tasks on this machine and summarise them.

Each job is one `scripts/eval_policy_robocasa.py` process that internally runs
--n-envs RoboCasa simulators in parallel through AsyncVectorEnv, so a job
already saturates several CPU cores and one GPU. Jobs themselves are spread
over the GPUs given by --gpus.

The manifest this writes is the same schema `scripts/robocasa/aggregate_results.py`
consumes, so the existing aggregator validates and summarises the results
without changes. Unlike scripts/robocasa/submit_evaluations.py this runs
locally and needs no Slurm.

Checkpoints may be local directories or `hf://` specifications:

    hf://my-org/gr00t-n15-robocasa
    hf://my-org/gr00t-n15-robocasa@v2#checkpoint-20000
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

LOCAL_EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_EVAL_DIR.parent
DEAS_ROOT = REPO_ROOT / "deas-gr00t15"
EVAL_SCRIPT = DEAS_ROOT / "scripts" / "eval_policy_robocasa.py"
AGGREGATOR = DEAS_ROOT / "scripts" / "robocasa" / "aggregate_results.py"
REQUIRED_CHECKPOINT_FILES = ("config.json", "experiment_cfg/metadata.json")

# The four RoboCasa tasks this repository's own submission script evaluates.
DEFAULT_TASKS = ("CoffeeSetupMug", "PnPMicrowaveToCounter", "TurnOffStove", "PnPCounterToMicrowave")


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
    parser.add_argument("--actor", required=True,
                        help="Actor checkpoint directory, or an hf:// specification")
    parser.add_argument("--critic", default=None,
                        help="Critic checkpoint for best-of-N; implies --method deas")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory for the manifest, per-job results and the summary")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--episodes", type=positive, default=50)
    parser.add_argument("--n-envs", type=positive, default=8,
                        help="RoboCasa simulators run in parallel inside each job")
    parser.add_argument("--gpus", nargs="+", type=nonnegative, default=[0],
                        help="GPU indices jobs are distributed over")
    parser.add_argument("--jobs-per-gpu", type=positive, default=1)
    parser.add_argument("--eval-seeds", nargs="+", type=nonnegative, default=[42],
                        help="One evaluation run per seed per task; several seeds give error bars")
    parser.add_argument("--training-seed", type=nonnegative, default=0,
                        help="Recorded as checkpoint provenance, not used by the simulator")
    parser.add_argument("--action-horizon", type=positive, default=16)
    parser.add_argument("--execute-horizon", type=positive, default=None)
    parser.add_argument("--denoising-steps", type=positive, default=4)
    parser.add_argument("--data-config", default="single_panda_gripper_rl_inference")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--deas-backend", choices=("legacy", "checkpoint", "iql", "svf"), default="iql")
    parser.add_argument("--critic-reference-actor", default=None,
                        help="For --deas-backend svf: the checkpoint the SVF run started from "
                             "(BC2), whose frozen features and normalization the critic scores in")
    parser.add_argument("--num-samples", type=positive, default=4,
                        help="Best-of-N sample count; only used with --critic")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-inference-inputs", action="store_true",
                        help="Record observations, RNG state and Q values for replay and rendering")
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--python-executable", default=None, help="Use this interpreter directly instead of conda run")
    parser.add_argument("--conda-env", default=os.environ.get("DEAS_CONDA_ENV", "deas-rc"))
    parser.add_argument("--skip-completed", action="store_true",
                        help="Leave already-completed job directories alone and reuse their results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write nothing; print the plan and the commands")
    args = parser.parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error("--eval-seeds must not contain duplicates")
    if args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if args.execute_horizon is not None and not 1 <= args.execute_horizon <= args.action_horizon:
        parser.error("--execute-horizon must be between 1 and --action-horizon")
    if args.deas_backend == "svf" and args.critic and not args.critic_reference_actor:
        parser.error("--deas-backend svf requires --critic-reference-actor")
    return args


def resolve_checkpoint(spec, *, label):
    """Accept a local directory or hf://repo_id[@revision][#subfolder]."""
    if not spec.startswith("hf://"):
        path = Path(spec).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"{label} checkpoint is not a directory: {path}")
        return path
    remainder = spec[len("hf://"):]
    remainder, _, subfolder = remainder.partition("#")
    repo_id, _, revision = remainder.partition("@")
    if not repo_id:
        raise SystemExit(f"{label} specification is missing a repository: {spec}")
    from huggingface_hub import snapshot_download

    print(f"Downloading {label} checkpoint {repo_id}"
          + (f"@{revision}" if revision else "") + (f"#{subfolder}" if subfolder else ""))
    root = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision or None,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        allow_patterns=[f"{subfolder.strip('/')}/**"] if subfolder else None,
    ))
    return (root / subfolder.strip("/")) if subfolder else root


def check_checkpoint(path, *, label):
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        raise SystemExit(f"{label} checkpoint {path} is missing {missing}")


def simulator_environment(gpu):
    """Everything local_eval/env.sh exports, pinned to one GPU."""
    dump = subprocess.run(
        ["bash", "-c", f'set -a; source {shlex.quote(str(LOCAL_EVAL_DIR / "env.sh"))}; env -0'],
        check=True, capture_output=True,
    ).stdout
    environment = dict(
        entry.split("=", 1) for entry in dump.decode().split("\0") if "=" in entry
    )
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    return environment


def build_plan(args, actor, critic):
    method = "deas" if critic else "gr00tn15"
    execute_horizon = args.execute_horizon or args.action_horizon
    root = args.output_root.expanduser().resolve()
    config = {
        "episodes": args.episodes,
        "n_envs": args.n_envs,
        "action_horizon": args.action_horizon,
        "execute_horizon": execute_horizon,
        "denoising_steps": args.denoising_steps,
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "save_video": args.save_video,
        "save_inference_inputs": args.save_inference_inputs,
        "report_to": args.report_to,
        "tasks": list(args.tasks),
        "eval_seeds": list(args.eval_seeds),
        "gpus": list(args.gpus),
        "jobs_per_gpu": args.jobs_per_gpu,
        "wandb_group": args.wandb_group,
    }
    if method == "deas":
        config.update(num_samples=args.num_samples, temperature=args.temperature,
                      deas_backend=args.deas_backend)
        if args.deas_backend == "svf":
            config["critic_reference_actor"] = str(args.critic_reference_actor)

    jobs = []
    for eval_seed in args.eval_seeds:
        for task in args.tasks:
            key = f"seed-{args.training_seed}-eval-{eval_seed}-{task}-{method}"
            output_dir = root / "results" / f"seed-{args.training_seed}" / f"eval-{eval_seed}" / task
            launcher = ([args.python_executable] if args.python_executable else
                        ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python"])
            command = launcher + [str(EVAL_SCRIPT),
                "--actor_model_path", str(actor),
                "--model_type", method,
                "--env_name", task,
                "--num_episodes", str(args.episodes),
                "--n_envs", str(args.n_envs),
                "--seed", str(eval_seed),
                "--training_seed", str(args.training_seed),
                "--data_config", args.data_config,
                "--embodiment_tag", args.embodiment_tag,
                "--action_horizon", str(args.action_horizon),
                "--execute_horizon", str(execute_horizon),
                "--denoising_steps", str(args.denoising_steps),
                "--output_path", str(output_dir),
                "--report_to", args.report_to,
                "--run_name", key,
            ]
            if critic:
                command += ["--critic_model_path", str(critic),
                            "--deas_backend", args.deas_backend,
                            "--num_samples", str(args.num_samples),
                            "--temperature", str(args.temperature)]
                if args.deas_backend == "svf":
                    command += ["--critic_reference_actor", str(args.critic_reference_actor)]
            if args.wandb_group:
                command += ["--wandb_group", args.wandb_group]
            if args.save_video:
                command.append("--save_video")
            if args.save_inference_inputs:
                command.append("--save_inference_inputs")
            jobs.append({
                "key": key,
                "training_seed": args.training_seed,
                "eval_seed": eval_seed,
                "task": task,
                "method": method,
                "model_type": method,
                "actor": str(actor),
                "critic": str(critic) if critic else None,
                "expected_episodes": args.episodes,
                "save_video": args.save_video,
                "video_dir": str(output_dir / "videos") if args.save_video else None,
                "output_dir": str(output_dir),
                "result_path": str(output_dir / "result.json"),
                "job_id": None,
                "command": command,
                "submission_state": "planned",
            })
    return {
        "schema_version": 1,
        "status": "planned",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"training_summary": None, "actor": str(actor), "critic": str(critic) if critic else None},
        "config": config,
        "output_root": str(root),
        "jobs": jobs,
        "runner": {"kind": "local", "host": os.uname().nodename, "conda_env": args.conda_env},
        "aggregator": {"output_dir": str(root / "aggregate"), "submission_state": "planned"},
    }


# Every worker thread refreshes the manifest, so the scratch file needs a name
# of its own: with a shared one, two threads race and whichever renames second
# finds the file already moved and dies with FileNotFoundError.
_manifest_lock = threading.Lock()


def save_manifest(plan):
    path = Path(plan["output_root"]) / "manifest.json"
    temporary = path.with_name(f"manifest.json.{os.getpid()}.{threading.get_ident()}.tmp")
    with _manifest_lock:
        temporary.write_text(json.dumps(plan, indent=2) + "\n")
        temporary.replace(path)
    return path


def already_completed(job):
    result = Path(job["result_path"])
    if not result.is_file():
        return False
    try:
        return json.loads(result.read_text()).get("status") == "completed"
    except (OSError, ValueError):
        return False


def run_jobs(plan, args):
    """Run the planned jobs, at most jobs_per_gpu concurrently on each GPU."""
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    pending = list(plan["jobs"])
    lock = threading.Lock()
    started = time.monotonic()

    def worker(gpu):
        environment = simulator_environment(gpu)
        while True:
            with lock:
                if not pending:
                    return
                job = pending.pop(0)
            if args.skip_completed and already_completed(job):
                job["submission_state"] = "reused"
                print(f"[gpu {gpu}] reusing completed {job['key']}")
                continue
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            job["submission_state"] = "running"
            job["gpu"] = gpu
            save_manifest(plan)
            print(f"[gpu {gpu}] starting {job['key']}")
            log_path = output_dir / "eval.log"
            with log_path.open("w") as log:
                completed = subprocess.run(job["command"], cwd=str(DEAS_ROOT), env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
            job["returncode"] = completed.returncode
            job["submission_state"] = "finished" if completed.returncode == 0 else "failed"
            print(f"[gpu {gpu}] {job['key']} exited {completed.returncode} "
                  f"({time.monotonic() - started:.0f}s elapsed); log: {log_path}")
            save_manifest(plan)

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main(argv=None):
    args = arguments(argv)
    actor = resolve_checkpoint(args.actor, label="actor")
    check_checkpoint(actor, label="actor")
    critic = None
    if args.critic:
        critic = resolve_checkpoint(args.critic, label="critic")
        if args.deas_backend == "svf":
            for name in ("config.json", "q_projection.safetensors"):
                if not (critic / name).is_file():
                    raise SystemExit(f"SVF critic is missing {name}: {critic}")
            args.critic_reference_actor = resolve_checkpoint(args.critic_reference_actor, label="reference actor")
            check_checkpoint(args.critic_reference_actor, label="reference actor")
        else:
            check_checkpoint(critic, label="critic")

    plan = build_plan(args, actor, critic)

    if args.dry_run:
        print(json.dumps({"output_root": plan["output_root"], "config": plan["config"],
                          "jobs": [job["key"] for job in plan["jobs"]]}, indent=2))
        for job in plan["jobs"]:
            print("\n" + " ".join(shlex.quote(part) for part in job["command"]))
        return 0

    Path(plan["output_root"]).mkdir(parents=True, exist_ok=True)
    save_manifest(plan)
    run_jobs(plan, args)
    plan["status"] = "finished"
    manifest_path = save_manifest(plan)

    aggregate = subprocess.run(
        [sys.executable, str(AGGREGATOR), "--manifest", str(manifest_path), "--allow-incomplete"],
        check=False,
    )
    summary = Path(plan["output_root"]) / "aggregate" / "summary.md"
    if summary.is_file():
        print()
        print(summary.read_text())
    failed = [job["key"] for job in plan["jobs"] if job.get("submission_state") == "failed"]
    if failed:
        print(f"Failed jobs: {failed}", file=sys.stderr)
        return 1
    return aggregate.returncode


if __name__ == "__main__":
    raise SystemExit(main())
