#!/usr/bin/env python3
"""Evaluate a GR00T N1.7 checkpoint on LIBERO, in parallel, across whole suites.

Runs inside the LIBERO island environment built by local_libero/setup_env.sh.

Parallelism is gr00t17's own: run_rollout_gymnasium_policy builds an
AsyncVectorEnv of --n-envs spawned LIBERO simulators and steps them together,
so each policy call is one batched inference over every live environment.

The policy is loaded once and reused for every task. gr00t17's own
rollout_policy.py entrypoint evaluates a single task per process, which would
reload a multi-billion parameter checkpoint 133 times for the full benchmark.

Results are written per task as they finish, so a long run can be resumed with
--skip-completed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
# LIBERO's own long-horizon suites need more steps than the short ones.
DEFAULT_MAX_EPISODE_STEPS = {"libero_10": 520, "libero_90": 520}
DEFAULT_MAX_EPISODE_STEPS_OTHER = 280


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("policy")
    source.add_argument("--model-path", default="",
                        help="Local checkpoint directory or Hub id")
    source.add_argument("--policy-client-host", default="",
                        help="Evaluate against a running policy server instead")
    source.add_argument("--policy-client-port", type=int, default=None)

    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=list(SUITES),
                        help=f"Any of {' '.join(SUITES)}")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Evaluate only these task names, ignoring --suites membership")
    parser.add_argument("--n-episodes", type=positive, default=50,
                        help="Episodes per task; LIBERO's usual protocol is 50")
    parser.add_argument("--n-envs", type=positive, default=8,
                        help="LIBERO simulators stepped in parallel per task")
    parser.add_argument("--n-action-steps", type=positive, default=8)
    parser.add_argument("--max-episode-steps", type=positive, default=None,
                        help="Default: 520 for libero_10/libero_90, 280 otherwise")
    parser.add_argument("--seed", type=int, default=0,
                        help="Per-env seeds are seed+index, so runs are reproducible")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Reuse task results already present under --output-root")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the tasks and the estimated cost, evaluate nothing")
    parser.add_argument("--summarise-only", action="store_true",
                        help="Rebuild summary.json/summary.md from existing results and exit. "
                             "Use after running suites as separate processes into one root, "
                             "where each would otherwise write a summary covering only its own "
                             "suite. Loads no policy.")
    args = parser.parse_args(argv)

    if not args.summarise_only and bool(args.model_path) == bool(args.policy_client_host):
        parser.error("Give exactly one of --model-path or --policy-client-host")
    if args.policy_client_host and args.policy_client_port is None:
        parser.error("--policy-client-host needs --policy-client-port")
    for suite in args.suites:
        if suite not in SUITES:
            parser.error(f"Unknown suite {suite}; known: {' '.join(SUITES)}")
    return args


def plan_tasks(args):
    from libero.libero import benchmark

    catalog = benchmark.get_benchmark_dict()
    plan = []
    for suite in args.suites:
        for name in catalog[suite]().get_task_names():
            if args.tasks and name not in args.tasks:
                continue
            steps = args.max_episode_steps or DEFAULT_MAX_EPISODE_STEPS.get(
                suite, DEFAULT_MAX_EPISODE_STEPS_OTHER)
            plan.append({"suite": suite, "task": name, "env_name": f"libero_sim/{name}",
                         "max_episode_steps": steps})
    if args.tasks:
        found = {entry["task"] for entry in plan}
        missing = [name for name in args.tasks if name not in found]
        if missing:
            raise SystemExit(f"Tasks not found in {args.suites}: {missing}")
    return plan


def result_path(root, entry):
    return root / "results" / entry["suite"] / f"{entry['task']}.json"


def load_result(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if value.get("status") == "completed" else None


def summarise(root, plan, settings):
    rows = []
    for entry in plan:
        record = load_result(result_path(root, entry)) or {}
        rows.append({
            "suite": entry["suite"], "task": entry["task"],
            "status": record.get("status", "missing"),
            "episodes": record.get("episodes"),
            "successes": record.get("successes"),
            "success_rate": record.get("success_rate"),
            "mean_episode_length": record.get("mean_episode_length"),
            "seconds": record.get("seconds"),
        })
    groups = []
    for suite in dict.fromkeys(row["suite"] for row in rows):
        done = [r for r in rows if r["suite"] == suite and r["status"] == "completed"]
        planned = [r for r in rows if r["suite"] == suite]
        episodes = sum(r["episodes"] for r in done)
        successes = sum(r["successes"] for r in done)
        rates = [r["success_rate"] for r in done]
        groups.append({
            "suite": suite, "planned_tasks": len(planned), "completed_tasks": len(done),
            "evaluated_episodes": episodes, "successful_episodes": successes,
            "pooled_success_rate": successes / episodes if episodes else None,
            # LIBERO is normally reported as the mean over tasks, not pooled.
            "task_mean_success_rate": statistics.mean(rates) if rates else None,
        })
    return {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "settings": settings, "tasks": rows, "suites": groups}


def write_summary(root, report):
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# LIBERO evaluation (GR00T N1.7)", "",
             "Success rate per suite is the mean over tasks, which is how LIBERO is",
             "normally reported; the pooled rate over all episodes is given alongside.", "",
             "| Suite | Tasks | Episodes | Task-mean | Pooled |", "|---|---:|---:|---:|---:|"]
    for group in report["suites"]:
        def percent(value):
            return "—" if value is None else f"{100 * value:.1f}%"
        lines.append(f"| {group['suite']} | {group['completed_tasks']}/{group['planned_tasks']} "
                     f"| {group['evaluated_episodes']} | {percent(group['task_mean_success_rate'])} "
                     f"| {percent(group['pooled_success_rate'])} |")
    incomplete = [r for r in report["tasks"] if r["status"] != "completed"]
    if incomplete:
        lines += ["", f"## Not completed ({len(incomplete)})", ""]
        lines += [f"- {r['suite']}/{r['task']}: {r['status']}" for r in incomplete[:50]]
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = arguments(argv)
    root = args.output_root.expanduser().resolve()

    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    plan = plan_tasks(args)

    settings = {
        "model_path": args.model_path or None,
        "policy_client": (f"{args.policy_client_host}:{args.policy_client_port}"
                          if args.policy_client_host else None),
        "suites": args.suites, "n_episodes": args.n_episodes, "n_envs": args.n_envs,
        "n_action_steps": args.n_action_steps, "seed": args.seed,
        "save_video": args.save_video, "tasks": len(plan),
    }

    if args.dry_run:
        print(json.dumps({"settings": settings,
                          "tasks": [f"{e['suite']}/{e['task']}" for e in plan]}, indent=2))
        print(f"\n{len(plan)} tasks x {args.n_episodes} episodes "
              f"= {len(plan) * args.n_episodes} episodes total")
        return 0

    root.mkdir(parents=True, exist_ok=True)

    if args.summarise_only:
        report = summarise(root, plan, settings)
        write_summary(root, report)
        print((root / "summary.md").read_text())
        print(f"Wrote {root / 'summary.json'}")
        return 0 if all(row["status"] == "completed" for row in report["tasks"]) else 1

    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
    from gr00t.eval.rollout_policy import (
        MultiStepConfig, VideoConfig, WrapperConfigs,
        create_gr00t_sim_policy, run_rollout_gymnasium_policy,
    )

    print(f"Loading the policy once for {len(plan)} tasks")
    embodiment_tag = get_embodiment_tag_from_env_name(plan[0]["env_name"])
    policy = create_gr00t_sim_policy(
        args.model_path, embodiment_tag,
        args.policy_client_host, args.policy_client_port,
    )
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)

    started = time.monotonic()
    for index, entry in enumerate(plan, start=1):
        path = result_path(root, entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        label = f"[{index}/{len(plan)}] {entry['suite']}/{entry['task']}"
        if args.skip_completed and load_result(path):
            print(f"{label}: reusing existing result")
            continue

        video_dir = str(root / "videos" / entry["suite"] / entry["task"]) if args.save_video else None
        wrapper_configs = WrapperConfigs(
            multistep=MultiStepConfig(
                contract=contract,
                max_episode_steps=entry["max_episode_steps"],
                terminate_on_success=True,
            ),
            video=VideoConfig(video_dir=video_dir,
                              max_episode_steps=entry["max_episode_steps"]),
        )
        print(f"{label}: {args.n_episodes} episodes, n_envs={args.n_envs}", flush=True)
        task_started = time.monotonic()
        try:
            _, successes, infos = run_rollout_gymnasium_policy(
                env_name=entry["env_name"], policy=policy,
                wrapper_configs=wrapper_configs, n_episodes=args.n_episodes,
                n_envs=args.n_envs, seed=args.seed,
            )
        except Exception as error:  # one bad task must not lose the whole sweep
            traceback.print_exc()
            path.write_text(json.dumps(
                {**entry, "status": "failed", "error": f"{type(error).__name__}: {error}",
                 "seconds": time.monotonic() - task_started}, indent=2) + "\n")
            continue

        lengths = infos.get("episode_lengths", [])
        # run_rollout_gymnasium_policy raises n_episodes to at least n_envs and
        # returns whatever the final batch completed, so it overshoots. The
        # protocol counts exactly --n-episodes per task; keep the first ones.
        successes = [bool(value) for value in successes][: args.n_episodes]
        lengths = list(lengths)[: args.n_episodes]
        record = {
            **entry, "status": "completed",
            "episodes": len(successes), "successes": sum(successes),
            "success_rate": sum(successes) / len(successes) if successes else None,
            "mean_episode_length": statistics.mean(lengths) if lengths else None,
            "seconds": time.monotonic() - task_started,
            "video_dir": video_dir,
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{label}: {record['successes']}/{record['episodes']} "
              f"({100 * record['success_rate']:.1f}%) in {record['seconds']:.0f}s "
              f"[{(time.monotonic() - started) / 60:.0f} min elapsed]", flush=True)
        write_summary(root, summarise(root, plan, settings))

    report = summarise(root, plan, settings)
    write_summary(root, report)
    print()
    print((root / "summary.md").read_text())
    print(f"Wrote {root / 'summary.json'}")
    return 0 if all(row["status"] == "completed" for row in report["tasks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
