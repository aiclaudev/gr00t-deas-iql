#!/usr/bin/env python3
"""Re-collect training data from a real LIBERO evaluation run, in parallel.

Runs inside the LIBERO island environment built by local_libero/setup_env.sh.

Environments are the same ones gr00t17 evaluates with — an AsyncVectorEnv of
--n-envs spawned LIBERO simulators, each carrying gr00t17's VideoRecordingWrapper
and MultiStepWrapper — with an EpisodeRecorder inserted directly on the raw
LiberoEnv so that every individual simulator step is captured, not the action
chunks the policy issues.

Each worker writes its own episode shards, so nothing large crosses a process
boundary. Assemble them into a dataset afterwards with
libero_shards_to_lerobot.py.

    collect_libero.py --model-path /path/to/n17 --output-root ~/data/libero_rollouts \\
        --suites libero_spatial --n-episodes 20 --n-envs 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from functools import partial
from pathlib import Path

LOCAL_COLLECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_COLLECT_DIR))
sys.path.insert(0, str(LOCAL_COLLECT_DIR.parent / "local_libero"))

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
DEFAULT_MAX_EPISODE_STEPS = {"libero_10": 520, "libero_90": 520}
DEFAULT_MAX_EPISODE_STEPS_OTHER = 280


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def make_collecting_env(env_idx: int, *, env_name: str, task_name: str, shard_dir: str,
                        wrapper_configs, fps: int):
    """Module-level so spawned workers can pickle it."""
    import gymnasium as gym

    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

    from libero_recorder import EpisodeRecorder

    register_libero_envs()
    env = gym.make(env_name)
    env = EpisodeRecorder(env, shard_dir=shard_dir, task_name=task_name,
                          env_index=env_idx, fps=fps,
                          max_episode_steps=wrapper_configs.multistep.max_episode_steps)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper

        env = VideoRecordingWrapper(
            env,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            overlay_text=wrapper_configs.video.overlay_text,
            record_video_keys=wrapper_configs.video.record_video_keys,
        )
    return MultiStepWrapper(
        env,
        contract=wrapper_configs.multistep.contract,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )


def collect_task(policy, contract, entry, args, root):
    import gymnasium as gym
    import numpy as np

    from gr00t.eval.rollout_policy import MultiStepConfig, VideoConfig, WrapperConfigs

    shard_dir = root / "shards" / entry["suite"] / entry["task"]
    shard_dir.mkdir(parents=True, exist_ok=True)
    video_dir = str(root / "eval_videos" / entry["suite"] / entry["task"]) if args.save_video else None

    wrapper_configs = WrapperConfigs(
        multistep=MultiStepConfig(contract=contract,
                                  max_episode_steps=entry["max_episode_steps"],
                                  terminate_on_success=True),
        video=VideoConfig(video_dir=video_dir, max_episode_steps=entry["max_episode_steps"]),
    )
    env_fns = [
        partial(make_collecting_env, idx, env_name=entry["env_name"], task_name=entry["task"],
                shard_dir=str(shard_dir), wrapper_configs=wrapper_configs, fps=args.fps)
        for idx in range(args.n_envs)
    ]
    env = (gym.vector.SyncVectorEnv(env_fns) if args.n_envs == 1
           else gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn"))

    completed = 0
    successes = 0
    accepted_shards = []
    episode_indices = [0] * args.n_envs
    try:
        seeds = [int(np.random.SeedSequence([args.seed, i]).generate_state(1)[0])
                 for i in range(args.n_envs)]
        observations, _ = env.reset(seed=seeds)
        policy.reset()
        live = [False] * args.n_envs
        while completed < args.n_episodes:
            actions, _ = policy.get_action(observations)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for idx in range(args.n_envs):
                finished = bool(terminations[idx] or truncations[idx])
                if not finished:
                    live[idx] = True
                    continue
                final = infos.get("final_info", [None] * args.n_envs)[idx]
                success = bool(np.any(final["success"])) if final and "success" in final else False
                completed += 1
                successes += success
                accepted_shards.append(f"env{idx:03d}_ep{episode_indices[idx]:05d}")
                episode_indices[idx] += 1
                print(f"EPISODE_DONE env={idx} count={completed} success={success}", flush=True)
                live[idx] = False
                if completed >= args.n_episodes:
                    break
    finally:
        # close() finalises each worker's in-flight shard, so partial runs still
        # leave complete episodes behind.
        try:
            env.close()
        except Exception as error:
            print(f"env.close() failed: {error}")
    # Keep exactly the episodes counted in SR; vector autoreset can save extras.
    (shard_dir / "accepted_episodes.json").write_text(
        json.dumps(accepted_shards, indent=2) + "\n")
    return {"completed": completed, "successes": successes,
            "success_rate": successes / completed if completed else None,
            "env_seeds": seeds, "shard_dir": str(shard_dir)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--policy-client-host", default="")
    parser.add_argument("--policy-client-port", type=int, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--n-episodes", type=positive, default=20,
                        help="Episodes collected per task")
    parser.add_argument("--n-envs", type=positive, default=8)
    parser.add_argument("--n-action-steps", type=positive, default=8)
    parser.add_argument("--max-episode-steps", type=positive, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=positive, default=20)
    parser.add_argument("--save-video", action="store_true",
                        help="Also keep gr00t17's own rollout videos, separate from the dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if bool(args.model_path) == bool(args.policy_client_host):
        parser.error("Give exactly one of --model-path or --policy-client-host")

    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    from list_tasks import tasks_by_suite

    catalog = tasks_by_suite(args.suites)
    plan = []
    for suite, names in catalog.items():
        for name in names:
            if args.tasks and name not in args.tasks:
                continue
            plan.append({
                "suite": suite, "task": name, "env_name": f"libero_sim/{name}",
                "max_episode_steps": args.max_episode_steps or DEFAULT_MAX_EPISODE_STEPS.get(
                    suite, DEFAULT_MAX_EPISODE_STEPS_OTHER),
            })

    root = args.output_root.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"tasks": [f"{e['suite']}/{e['task']}" for e in plan],
                          "episodes_per_task": args.n_episodes,
                          "total_episodes": len(plan) * args.n_episodes}, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)

    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
    from gr00t.eval.rollout_policy import create_gr00t_sim_policy

    print(f"Loading the policy once for {len(plan)} tasks")
    policy = create_gr00t_sim_policy(
        args.model_path, get_embodiment_tag_from_env_name(plan[0]["env_name"]),
        args.policy_client_host, args.policy_client_port)
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "settings": {"model_path": args.model_path or None, "suites": args.suites,
                     "n_episodes": args.n_episodes, "n_envs": args.n_envs,
                     "n_action_steps": args.n_action_steps, "seed": args.seed, "fps": args.fps},
        "tasks": [],
    }
    started = time.monotonic()
    for index, entry in enumerate(plan, start=1):
        label = f"[{index}/{len(plan)}] {entry['suite']}/{entry['task']}"
        print(f"{label}: collecting {args.n_episodes} episodes, n_envs={args.n_envs}", flush=True)
        try:
            result = collect_task(policy, contract, entry, args, root)
            record = {**entry, "status": "completed", **result}
            print(f"{label}: {result['completed']} episodes, "
                  f"{result['successes']} successful "
                  f"[{(time.monotonic() - started) / 60:.0f} min elapsed]", flush=True)
        except Exception as error:
            traceback.print_exc()
            record = {**entry, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        manifest["tasks"].append(record)
        (root / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nShards under {root / 'shards'}")
    print("Assemble with local_collect/libero_shards_to_lerobot.py")
    return 1 if any(t["status"] != "completed" for t in manifest["tasks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
