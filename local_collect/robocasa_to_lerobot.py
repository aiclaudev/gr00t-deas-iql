#!/usr/bin/env python3
"""Convert replayed RoboCasa eval rollouts into a GR00T LeRobot v2.1 dataset.

Input is a robomimic-format HDF5 that already carries observations, i.e. the
output of robocasa's dataset_states_to_obs.py. Output loads directly with
deas-gr00t15's `single_panda_gripper_rl` config, so re-collected rollouts can
go straight into DEAS critic / IQL training alongside the human demos.

Adapted from ~/Value/robocasa/convert_robocasa_to_groot_lerobot_v21.py, which
targets a different key naming and omits the RL modalities. Three differences:

- video keys are left_view / right_view / wrist_view, not robot0_agentview_left
  and friends;
- action keys are end_effector_position / end_effector_rotation /
  gripper_close / base_motion / control_mode;
- modality.json also carries the reward, done, next_state and next_video
  sections that LeRobotRLModalityMetadata requires. Without them a dataset
  loaded with use_rl=True fails with "unexpected modality: reward".

The extra sections are additive: configs that do not use RL ignore them, so the
same dataset also serves BC retraining.

Every episode is kept, successful or not, because IQL needs the failures. Each
episode records its outcome in meta/episodes.jsonl.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

DEFAULT_CHUNK_SIZE = 1000
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
LANG_KEY = "annotation.human.action.task_description"

# (gr00t key, robomimic obs key, width, rotation type)
STATE_SPECS: list[tuple[str, str, int, str | None]] = [
    ("gripper_qpos", "robot0_gripper_qpos", 2, None),
    ("base_position", "robot0_base_pos", 3, None),
    ("base_rotation", "robot0_base_quat", 4, "quaternion"),
    ("end_effector_position_relative", "robot0_base_to_eef_pos", 3, None),
    ("end_effector_rotation_relative", "robot0_base_to_eef_quat", 4, "quaternion"),
    ("gripper_qvel", "robot0_gripper_qvel", 2, None),
    ("end_effector_position_absolute", "robot0_eef_pos", 3, None),
    ("end_effector_rotation_absolute", "robot0_eef_quat", 4, "quaternion"),
    ("joint_position", "robot0_joint_pos", 7, None),
    ("joint_position_cos", "robot0_joint_pos_cos", 7, None),
    ("joint_position_sin", "robot0_joint_pos_sin", 7, None),
    ("joint_velocity", "robot0_joint_vel", 7, None),
]
# (gr00t key, robomimic obs key)
VIDEO_SPECS: list[tuple[str, str]] = [
    ("left_view", "robot0_agentview_left_image"),
    ("right_view", "robot0_agentview_right_image"),
    ("wrist_view", "robot0_eye_in_hand_image"),
]
# (gr00t key, width, absolute, rotation type)
ACTION_SPECS: list[tuple[str, int, bool, str | None]] = [
    ("end_effector_position", 3, False, None),
    ("end_effector_rotation", 3, False, "axis_angle"),
    ("gripper_close", 1, False, None),
    ("base_motion", 4, False, None),
    ("control_mode", 1, False, None),
]
ACTION_DIM = sum(size for _, size, _, _ in ACTION_SPECS)
STATE_DIM = sum(size for _, _, size, _ in STATE_SPECS)


def build_modality_meta() -> dict:
    state_meta, start = {}, 0
    for key, _, size, rotation in STATE_SPECS:
        entry = {"start": start, "end": start + size, "dtype": "float32",
                 "absolute": True, "original_key": "observation.state"}
        if rotation:
            entry["rotation_type"] = rotation
        state_meta[key] = entry
        start += size

    action_meta, start = {}, 0
    for key, size, absolute, rotation in ACTION_SPECS:
        entry = {"start": start, "end": start + size, "dtype": "float32",
                 "absolute": absolute, "original_key": "action"}
        if rotation:
            entry["rotation_type"] = rotation
        action_meta[key] = entry
        start += size

    video_meta = {key: {"original_key": f"observation.images.{key}"} for key, _ in VIDEO_SPECS}

    return {
        "state": state_meta,
        "action": action_meta,
        "video": video_meta,
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        # The RL modalities. next_state and next_video read the same columns as
        # state and video; the loader shifts them by the configured delta index.
        "reward": {"next.reward": {"original_key": "next.reward", "dtype": "float32"}},
        "done": {"next.done": {"original_key": "next.done", "dtype": "bool"}},
        "next_state": {key: dict(value) for key, value in state_meta.items()},
        "next_video": {key: dict(value) for key, value in video_meta.items()},
    }


def build_feature_meta(image_shape, fps: float) -> dict:
    height, width, channels = image_shape
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM],
                              "names": [f"state_{i}" for i in range(STATE_DIM)]},
        "action": {"dtype": "float32", "shape": [ACTION_DIM],
                   "names": [f"action_{i}" for i in range(ACTION_DIM)]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        LANG_KEY: {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    video_info = {"video.fps": float(fps), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                  "video.is_depth_map": False, "has_audio": False}
    for key, _ in VIDEO_SPECS:
        features[f"observation.images.{key}"] = {
            "dtype": "video", "shape": [height, width, channels],
            "names": ["height", "width", "channel"], "video_info": video_info,
        }
    return features


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n")


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def concat_state(obs_group) -> np.ndarray:
    parts = []
    for key, source, size, _ in STATE_SPECS:
        if source not in obs_group:
            raise KeyError(f"Observation {source} missing; was dataset_states_to_obs.py run?")
        array = np.asarray(obs_group[source][()], dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[1] != size:
            raise ValueError(f"{source}: expected width {size}, got {array.shape[1]}")
        parts.append(array)
    return np.concatenate(parts, axis=1)


def demo_names_in_order(handle, filter_key: str | None):
    if filter_key:
        names = [name.decode() for name in handle[f"mask/{filter_key}"][()]]
    else:
        names = list(handle["data"].keys())
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def convert(hdf5_path: Path, output_root: Path, *, filter_key, max_episodes,
            overwrite, fps_override) -> Path:
    with h5py.File(hdf5_path, "r") as handle:
        env_args = json.loads(handle["data"].attrs["env_args"])
        task_name = env_args["env_name"]
        dataset_root = output_root / task_name
        if dataset_root.exists():
            if not overwrite:
                raise FileExistsError(f"{dataset_root} exists; pass --overwrite")
            shutil.rmtree(dataset_root)

        names = demo_names_in_order(handle, filter_key)
        if max_episodes is not None:
            names = names[:max_episodes]
        if not names:
            raise SystemExit(f"{hdf5_path}: no demos selected")

        fps = float(fps_override or env_args["env_kwargs"].get("control_freq", 20))
        first_image = np.asarray(handle["data"][names[0]]["obs"][VIDEO_SPECS[0][1]][0])

        ordered_tasks, task_to_index = [], {}
        episode_rows, state_arrays, action_arrays = [], [], []
        global_index = 0
        successes = 0

        for episode_index, name in enumerate(names):
            demo = handle["data"][name]
            obs = demo["obs"]
            ep_meta = json.loads(demo.attrs["ep_meta"]) if "ep_meta" in demo.attrs else {}
            task_text = ep_meta.get("lang", task_name)
            if task_text not in task_to_index:
                task_to_index[task_text] = len(ordered_tasks)
                ordered_tasks.append(task_text)
            task_index = task_to_index[task_text]

            state = concat_state(obs)
            action = np.asarray(demo["actions"][()], dtype=np.float32)
            rewards = np.asarray(demo["rewards"][()], dtype=np.float32)
            dones = np.asarray(demo["dones"][()], dtype=bool)
            frames = state.shape[0]
            if action.shape[1] != ACTION_DIM:
                raise ValueError(f"{name}: expected action dim {ACTION_DIM}, got {action.shape[1]}")
            for array, label in ((action, "actions"), (rewards, "rewards"), (dones, "dones")):
                if len(array) != frames:
                    raise ValueError(f"{name}: {label} has {len(array)} rows, state has {frames}")

            succeeded = bool(rewards.max() > 0)
            successes += succeeded

            frame_df = pd.DataFrame({
                "observation.state": list(state),
                "action": list(action),
                "timestamp": (np.arange(frames) / fps).astype(np.float32),
                LANG_KEY: np.full(frames, task_index, dtype=np.int64),
                "task_index": np.full(frames, task_index, dtype=np.int64),
                "episode_index": np.full(frames, episode_index, dtype=np.int64),
                "frame_index": np.arange(frames, dtype=np.int64),
                "index": np.arange(global_index, global_index + frames, dtype=np.int64),
                # Scalars, not length-1 lists: get_reward_or_done stacks these
                # and asserts the result is one-dimensional.
                "next.reward": rewards,
                "next.done": dones,
            })
            data_path = dataset_root / DATA_PATH_TEMPLATE.format(
                episode_chunk=episode_index // DEFAULT_CHUNK_SIZE, episode_index=episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            frame_df.to_parquet(data_path, index=False)

            for key, source in VIDEO_SPECS:
                write_video(
                    dataset_root / VIDEO_PATH_TEMPLATE.format(
                        episode_chunk=episode_index // DEFAULT_CHUNK_SIZE,
                        video_key=f"observation.images.{key}", episode_index=episode_index),
                    np.asarray(obs[source][()], dtype=np.uint8), fps=fps)

            episode_rows.append({"episode_index": episode_index, "tasks": [task_text],
                                 "length": frames, "source_demo": name, "success": succeeded})
            state_arrays.append(state)
            action_arrays.append(action)
            global_index += frames

        write_json(dataset_root / "meta" / "info.json", {
            "codebase_version": "v2.1", "robot_type": env_args["env_kwargs"].get("robots", "PandaOmron"),
            "total_episodes": len(episode_rows), "total_frames": global_index,
            "total_tasks": len(ordered_tasks), "total_videos": len(episode_rows) * len(VIDEO_SPECS),
            "total_chunks": 1, "chunks_size": DEFAULT_CHUNK_SIZE, "fps": fps,
            "splits": {"train": f"0:{len(episode_rows)}"},
            "data_path": DATA_PATH_TEMPLATE, "video_path": VIDEO_PATH_TEMPLATE,
            "features": build_feature_meta(first_image.shape, fps),
        })
        write_json(dataset_root / "meta" / "modality.json", build_modality_meta())
        (dataset_root / "meta" / "tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": i, "task": t}) + "\n"
                    for i, t in enumerate(ordered_tasks)))
        (dataset_root / "meta" / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in episode_rows))
        # meta/stats.json is left out on purpose: LeRobotSingleDataset computes
        # and caches it on first load, from the parquet files it will actually read.

        print(f"[ok] {task_name}: {len(episode_rows)} episodes "
              f"({successes} successful), {global_index} frames -> {dataset_root}")
        return dataset_root


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, nargs="+", required=True,
                        help="Replayed robomimic HDF5 files (with observations)")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="A dataset directory is written per task underneath")
    parser.add_argument("--filter-key", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in args.hdf5:
        convert(path, args.output_root.expanduser().resolve(), filter_key=args.filter_key,
                max_episodes=args.max_episodes, overwrite=args.overwrite, fps_override=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
