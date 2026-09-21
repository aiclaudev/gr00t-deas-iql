#!/usr/bin/env python3
"""Replay recorded RoboCasa rollouts straight into a GR00T LeRobot v2.1 dataset.

The collected demo.hdf5 holds simulator states, actions, rewards and dones but
no images. robocasa's dataset_states_to_obs.py regenerates the camera views by
replaying those states, but writes them back into HDF5 as raw uint8 arrays: at
128x128x3, three cameras and ~300 steps that is roughly 44 MB per episode, and
the whole thing is then read once more to be re-encoded as video. This does the
same replay and streams each frame straight into its mp4 writer, so the
intermediate never exists. Only one frame per camera is ever held in memory.

Output loads with deas-gr00t15's `single_panda_gripper_rl`; the metadata
builders are shared with robocasa_to_lerobot.py, which remains the path to use
when a replayed observation HDF5 already exists.

Rewards and dones are copied from the recording rather than re-inferred, so the
dataset carries the signal the evaluated policy actually produced. Successful
and failed episodes are both kept.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robocasa_to_lerobot import (
    ACTION_DIM,
    DATA_PATH_TEMPLATE,
    DEFAULT_CHUNK_SIZE,
    LANG_KEY,
    STATE_SPECS,
    VIDEO_PATH_TEMPLATE,
    VIDEO_SPECS,
    build_feature_meta,
    build_modality_meta,
    demo_names_in_order,
    write_json,
)


def low_dimensional_state(obs: dict) -> np.ndarray:
    parts = []
    for _, source, size, _ in STATE_SPECS:
        if source not in obs:
            raise KeyError(f"Replayed observation is missing {source}")
        array = np.asarray(obs[source], dtype=np.float32).reshape(-1)
        if array.size != size:
            raise ValueError(f"{source}: expected width {size}, got {array.size}")
        parts.append(array)
    return np.concatenate(parts)


def open_writers(dataset_root: Path, episode_index: int, fps: float):
    writers = {}
    for key, _ in VIDEO_SPECS:
        path = dataset_root / VIDEO_PATH_TEMPLATE.format(
            episode_chunk=episode_index // DEFAULT_CHUNK_SIZE,
            video_key=f"observation.images.{key}", episode_index=episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        writers[key] = imageio.get_writer(path, fps=fps, codec="libx264",
                                          pixelformat="yuv420p", macro_block_size=1)
    return writers


def replay(hdf5_path: Path, output_root: Path, *, camera_size, max_episodes,
           overwrite, fps_override, generative_textures, randomize_cameras) -> Path:
    import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
    import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=str(hdf5_path))
    if generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True

    camera_names = [source.removesuffix("_image") for _, source in VIDEO_SPECS]
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=camera_size,
        camera_width=camera_size,
        reward_shaping=False,
    )

    with h5py.File(hdf5_path, "r") as handle:
        env_args = json.loads(handle["data"].attrs["env_args"])
        task_name = env_args["env_name"]
        dataset_root = output_root / task_name
        if dataset_root.exists():
            if not overwrite:
                raise FileExistsError(f"{dataset_root} exists; pass --overwrite")
            shutil.rmtree(dataset_root)

        names = demo_names_in_order(handle, None)
        if max_episodes is not None:
            names = names[:max_episodes]
        if not names:
            raise SystemExit(f"{hdf5_path}: no demos found")

        fps = float(fps_override or env_meta["env_kwargs"].get("control_freq", 20))
        ordered_tasks, task_to_index = [], {}
        episode_rows, state_arrays, action_arrays = [], [], []
        global_index, successes = 0, 0

        for episode_index, name in enumerate(names):
            demo = handle["data"][name]
            states = np.asarray(demo["states"][()])
            actions = np.asarray(demo["actions"][()], dtype=np.float32)
            rewards = np.asarray(demo["rewards"][()], dtype=np.float32)
            dones = np.asarray(demo["dones"][()], dtype=bool)
            frames = states.shape[0]
            if actions.shape[0] != frames or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"{name}: actions {actions.shape} do not match "
                                 f"{frames} states of width {ACTION_DIM}")

            initial_state = {"states": states[0], "model": demo.attrs["model_file"]}
            if "ep_meta" in demo.attrs:
                initial_state["ep_meta"] = demo.attrs["ep_meta"]
            env.reset()
            env.reset_to(initial_state)
            ep_meta = json.loads(env.env.get_ep_meta_json()) if hasattr(env.env, "get_ep_meta_json") \
                else env.env.get_ep_meta()
            task_text = ep_meta.get("lang", task_name)
            if task_text not in task_to_index:
                task_to_index[task_text] = len(ordered_tasks)
                ordered_tasks.append(task_text)
            task_index = task_to_index[task_text]

            writers = open_writers(dataset_root, episode_index, fps)
            state_rows = []
            try:
                for step in range(frames):
                    # Same per-step state reload dataset_states_to_obs.py uses.
                    obs = env.reset_to({"states": states[step]})
                    state_rows.append(low_dimensional_state(obs))
                    for key, source in VIDEO_SPECS:
                        frame = np.asarray(obs[source])
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                        writers[key].append_data(frame)
            finally:
                for writer in writers.values():
                    writer.close()

            state = np.stack(state_rows).astype(np.float32)
            succeeded = bool(rewards.max() > 0)
            successes += succeeded

            frame_df = pd.DataFrame({
                "observation.state": list(state),
                "action": list(actions),
                "timestamp": (np.arange(frames) / fps).astype(np.float32),
                LANG_KEY: np.full(frames, task_index, dtype=np.int64),
                "task_index": np.full(frames, task_index, dtype=np.int64),
                "episode_index": np.full(frames, episode_index, dtype=np.int64),
                "frame_index": np.arange(frames, dtype=np.int64),
                "index": np.arange(global_index, global_index + frames, dtype=np.int64),
                "next.reward": rewards,
                "next.done": dones,
            })
            data_path = dataset_root / DATA_PATH_TEMPLATE.format(
                episode_chunk=episode_index // DEFAULT_CHUNK_SIZE, episode_index=episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            frame_df.to_parquet(data_path, index=False)

            episode_rows.append({"episode_index": episode_index, "tasks": [task_text],
                                 "length": frames, "source_demo": name, "success": succeeded})
            state_arrays.append(state)
            action_arrays.append(actions)
            global_index += frames
            print(f"  [{episode_index + 1}/{len(names)}] {name}: {frames} frames, "
                  f"{'success' if succeeded else 'failure'}", flush=True)

        write_json(dataset_root / "meta" / "info.json", {
            "codebase_version": "v2.1",
            "robot_type": env_meta["env_kwargs"].get("robots", "PandaOmron"),
            "total_episodes": len(episode_rows), "total_frames": global_index,
            "total_tasks": len(ordered_tasks),
            "total_videos": len(episode_rows) * len(VIDEO_SPECS),
            "total_chunks": 1, "chunks_size": DEFAULT_CHUNK_SIZE, "fps": fps,
            "splits": {"train": f"0:{len(episode_rows)}"},
            "data_path": DATA_PATH_TEMPLATE, "video_path": VIDEO_PATH_TEMPLATE,
            "features": build_feature_meta((camera_size, camera_size, 3), fps),
        })
        write_json(dataset_root / "meta" / "modality.json", build_modality_meta())
        (dataset_root / "meta" / "tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": i, "task": t}) + "\n"
                    for i, t in enumerate(ordered_tasks)))
        (dataset_root / "meta" / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in episode_rows))

        print(f"[ok] {task_name}: {len(episode_rows)} episodes "
              f"({successes} successful), {global_index} frames -> {dataset_root}")
    env.env.close()
    return dataset_root


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, nargs="+", required=True,
                        help="Collected demo.hdf5 files (states and actions, no images)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generative-textures", action="store_true")
    parser.add_argument("--randomize-cameras", action="store_true")
    args = parser.parse_args()

    for path in args.hdf5:
        replay(path, args.output_root.expanduser().resolve(),
               camera_size=args.camera_size, max_episodes=args.max_episodes,
               overwrite=args.overwrite, fps_override=args.fps,
               generative_textures=args.generative_textures,
               randomize_cameras=args.randomize_cameras)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
