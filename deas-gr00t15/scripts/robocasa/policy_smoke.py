"""Record bounded actor previews; these clips are not full-episode evaluations."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

TASKS = ("CoffeeSetupMug", "PnPMicrowaveToCounter", "TurnOffStove", "PnPCounterToMicrowave")
ACTION_HORIZON = 16


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, help="Preview only this task; default: all four")
    parser.add_argument("--inference-steps", type=positive_int, default=5,
                        help="Maximum policy calls per task (each predicts 16 simulator actions)")
    parser.add_argument("--video-fps", type=positive_int, default=15,
                        help="Saved video playback FPS; does not change simulator control frequency")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU for this bounded actor preview")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env
    config = DATA_CONFIG_MAP["single_panda_gripper_rl_inference"](AS=ACTION_HORIZON)
    policy = Gr00tPolicy(model_path=str(args.actor.resolve()), embodiment_tag="new_embodiment",
                        modality_config=config.modality_config(), modality_transform=config.transform(),
                        denoising_steps=4, device="cuda:0")
    results = []
    report = {
        "scope": "bounded_actor_preview_not_full_episode_evaluation",
        "actor": str(args.actor.resolve()),
        "seed": args.seed,
        "max_inference_calls_per_task": args.inference_steps,
        "action_horizon": ACTION_HORIZON,
        "max_simulator_steps_per_task": args.inference_steps * ACTION_HORIZON,
        "video_fps": args.video_fps,
        "tasks": results,
    }
    for task in (args.task,) if args.task else TASKS:
        env = load_robocasa_gym_env(task, n_envs=1, seed=args.seed, obj_instance_split="B",
                                  layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
                                  action_horizon=ACTION_HORIZON, camera_widths=256, camera_heights=256,
                                  video_path=args.output / task / "videos", video_fps=args.video_fps)
        inference_calls = 0
        simulator_steps = 0
        stop_reason = "inference_step_limit"
        was_terminated = False
        was_truncated = False
        try:
            observation, _ = env.reset()
            for _ in range(args.inference_steps):
                with torch.inference_mode():
                    actions = policy.get_action(observation)
                if not actions:
                    raise ValueError("Policy returned no actions")
                for name, values in actions.items():
                    if values.shape[0:2] != (1, ACTION_HORIZON) or not np.isfinite(values).all():
                        raise ValueError(f"Unexpected or nonfinite actions: {name}, {values.shape}")
                inference_calls += 1
                observation, reward, terminated, truncated, info = env.step(actions)
                if not np.isfinite(reward).all():
                    raise ValueError("Simulator returned nonfinite rewards")
                executed_steps = int(info["num_executed_steps"][0])
                if not 0 < executed_steps <= ACTION_HORIZON:
                    raise ValueError(f"Unexpected simulator step count: {executed_steps}")
                simulator_steps += executed_steps
                was_terminated = bool(np.asarray(terminated).reshape(-1)[0])
                was_truncated = bool(np.asarray(truncated).reshape(-1)[0])
                if was_terminated or was_truncated:
                    stop_reason = ("terminated_and_truncated" if was_terminated and was_truncated
                                   else "terminated" if was_terminated else "truncated")
                    break
            results.append({
                "task": task,
                "inference_calls": inference_calls,
                "simulator_steps": simulator_steps,
                "stop_reason": stop_reason,
                "terminated": was_terminated,
                "truncated": was_truncated,
                "action_shapes": {name: list(value.shape) for name, value in actions.items()},
                "video_fps": args.video_fps,
                "status": "passed",
            })
        finally:
            # Flush the single partial episode at the call limit or a natural end.
            # Never reset again: doing so would start a second clip/episode.
            env.close()
        videos = list((args.output / task / "videos" / "env_0").glob("*.mp4"))
        if len(videos) != 1 or videos[0].stat().st_size == 0:
            raise ValueError(f"Expected one nonempty preview video for {task}")
        results[-1]["video"] = str(videos[0].resolve())
        results[-1]["video_bytes"] = videos[0].stat().st_size
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
