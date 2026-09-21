#!/usr/bin/env python3
"""Generate a synthetic LeRobot dataset shaped for the `single_panda_gripper_rl` config.

The point is to exercise the DEAS critic training pipeline end to end before the
real dataset exists: the schema, the RL modalities (reward, done, next_state,
next_video), video decoding, normalisation statistics, and the forward/backward
pass. The numbers are noise and mean nothing.

Layout written:

    <output>/meta/{info,modality}.json, {tasks,episodes}.jsonl
    <output>/data/chunk-000/episode_%06d.parquet
    <output>/videos/chunk-000/observation.images.<view>/episode_%06d.mp4

meta/stats.json is deliberately not written; LeRobotSingleDataset computes it
from the parquet files on first load, which also exercises that path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# state vector layout, matching the keys SinglePandaGripperRLDataConfig reads
STATE_LAYOUT = [
    ("end_effector_position_relative", 3, None),
    ("end_effector_rotation_relative", 4, "quaternion"),
    ("gripper_qpos", 2, None),
    ("base_position", 3, None),
    ("base_rotation", 4, "quaternion"),
]
# action vector layout
ACTION_LAYOUT = [
    ("end_effector_position", 3, None),
    ("end_effector_rotation", 3, "axis_angle"),
    ("gripper_close", 1, None),
    ("base_motion", 4, None),
    ("control_mode", 1, None),
]
VIEWS = ["left_view", "right_view", "wrist_view"]
TASKS = [
    "pick the mug from the counter and place it in the coffee machine",
    "open the microwave and take out the bowl",
]


def spans(layout):
    offset = 0
    for name, width, rotation in layout:
        yield name, offset, offset + width, rotation
        offset += width


def build_modality_json():
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {},
                "reward": {}, "done": {}, "next_state": {}, "next_video": {}}
    for name, start, end, rotation in spans(STATE_LAYOUT):
        entry = {"start": start, "end": end, "dtype": "float32", "absolute": True,
                 "original_key": "observation.state"}
        if rotation:
            entry["rotation_type"] = rotation
        modality["state"][name] = entry
        # next_state reads the same columns; the loader shifts by the delta index.
        modality["next_state"][name] = dict(entry)
    for name, start, end, rotation in spans(ACTION_LAYOUT):
        entry = {"start": start, "end": end, "dtype": "float32", "absolute": False,
                 "original_key": "action"}
        if rotation:
            entry["rotation_type"] = rotation
        modality["action"][name] = entry
    for view in VIEWS:
        entry = {"original_key": f"observation.images.{view}"}
        modality["video"][view] = entry
        modality["next_video"][view] = dict(entry)
    modality["annotation"]["human.action.task_description"] = {"original_key": "task_index"}
    modality["reward"]["next.reward"] = {"original_key": "next.reward", "dtype": "float32"}
    modality["done"]["next.done"] = {"original_key": "next.done", "dtype": "bool"}
    return modality


def build_info_json(args, total_frames, state_dim, action_dim):
    def named(dim, prefix):
        return {"dtype": "float32", "shape": [dim], "names": [f"{prefix}_{i}" for i in range(dim)]}

    features = {
        "observation.state": named(state_dim, "state"),
        "action": named(action_dim, "action"),
        "timestamp": {"dtype": "float32", "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    for view in VIEWS:
        features[f"observation.images.{view}"] = {
            "dtype": "video",
            "shape": [args.image_size, args.image_size, 3],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": float(args.fps), "video.codec": "h264",
                           "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                           "has_audio": False},
        }
    return {
        "codebase_version": "v2.1",
        "robot_type": "PandaMobile",
        "total_episodes": args.episodes,
        "total_frames": total_frames,
        "total_tasks": len(TASKS),
        "total_videos": args.episodes * len(VIEWS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": float(args.fps),
        "splits": {"train": f"0:{args.episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def unit_quaternions(rng, count):
    raw = rng.normal(size=(count, 4))
    return (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)


def episode_frames(rng, args, episode):
    """Smoothly varying frames, so video compression and colour jitter have real signal."""
    size, count = args.image_size, args.frames
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    frames = np.empty((count, size, size, 3), dtype=np.uint8)
    for step in range(count):
        phase = 2 * np.pi * (step / count + episode / max(args.episodes, 1))
        channels = [
            0.5 + 0.5 * np.sin(phase + 6 * x),
            0.5 + 0.5 * np.cos(phase + 6 * y),
            0.5 + 0.5 * np.sin(phase + 3 * (x + y)),
        ]
        image = np.stack(channels, axis=-1) * 255
        image += rng.normal(scale=6.0, size=image.shape)
        frames[step] = np.clip(image, 0, 255).astype(np.uint8)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=64,
                        help="Frames per episode; must exceed the action horizon")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import imageio.v2 as imageio
    import pandas as pd

    state_dim = sum(width for _, width, _ in STATE_LAYOUT)
    action_dim = sum(width for _, width, _ in ACTION_LAYOUT)
    rng = np.random.default_rng(args.seed)
    output = args.output
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data/chunk-000").mkdir(parents=True, exist_ok=True)

    episode_records, global_index = [], 0
    for episode in range(args.episodes):
        count = args.frames
        task_index = episode % len(TASKS)

        state = np.concatenate([
            rng.uniform(-0.4, 0.4, (count, 3)),
            unit_quaternions(rng, count),
            rng.uniform(0.0, 0.04, (count, 2)),
            rng.uniform(-1.0, 1.0, (count, 3)),
            unit_quaternions(rng, count),
        ], axis=1).astype(np.float32)
        action = np.concatenate([
            rng.uniform(-1.0, 1.0, (count, 3)),
            rng.uniform(-0.5, 0.5, (count, 3)),
            rng.choice([-1.0, 1.0], (count, 1)),
            rng.uniform(-1.0, 1.0, (count, 4)),
            rng.choice([0.0, 1.0], (count, 1)),
        ], axis=1).astype(np.float32)

        # Half the episodes succeed, so both the sparse-reward and the all-zero
        # branch of the reward handling get exercised.
        reward = np.zeros(count, dtype=np.float32)
        done = np.zeros(count, dtype=bool)
        if episode % 2 == 0:
            reward[-1] = 1.0
        done[-1] = True

        frame = pd.DataFrame({
            "observation.state": list(state),
            "action": list(action),
            "timestamp": (np.arange(count) / args.fps).astype(np.float32),
            "annotation.human.action.task_description": np.full(count, task_index, dtype=np.int64),
            "task_index": np.full(count, task_index, dtype=np.int64),
            "episode_index": np.full(count, episode, dtype=np.int64),
            "frame_index": np.arange(count, dtype=np.int64),
            "index": np.arange(global_index, global_index + count, dtype=np.int64),
            "next.reward": reward,
            "next.done": done,
        })
        frame.to_parquet(output / f"data/chunk-000/episode_{episode:06d}.parquet", index=False)
        global_index += count

        frames = episode_frames(rng, args, episode)
        for view in VIEWS:
            directory = output / f"videos/chunk-000/observation.images.{view}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"episode_{episode:06d}.mp4"
            with imageio.get_writer(path, fps=args.fps, codec="libx264",
                                    pixelformat="yuv420p", macro_block_size=1) as writer:
                offset = VIEWS.index(view) * 40
                for image in np.roll(frames, offset, axis=2):
                    writer.append_data(image)

        episode_records.append({"episode_index": episode, "tasks": [TASKS[task_index]],
                                "length": count})

    (output / "meta/modality.json").write_text(json.dumps(build_modality_json(), indent=4) + "\n")
    (output / "meta/info.json").write_text(
        json.dumps(build_info_json(args, global_index, state_dim, action_dim), indent=4) + "\n")
    (output / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(TASKS)))
    (output / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in episode_records))

    print(f"Wrote {args.episodes} episodes x {args.frames} frames "
          f"(state {state_dim}, action {action_dim}) to {output}")


if __name__ == "__main__":
    raise SystemExit(main())
