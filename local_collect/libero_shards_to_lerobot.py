#!/usr/bin/env python3
"""Assemble LIBERO collection shards into GR00T LeRobot v2.1 datasets.

collect_libero.py writes one shard per episode from each worker process, with
the camera views already encoded as mp4. This renumbers the episodes, moves the
videos into the LeRobot layout and writes the metadata; no frame is ever
decoded or re-encoded, so assembly is a file move plus a small parquet rewrite.

By default one dataset is produced per suite, with the suite's tasks
distinguished by task_index. Use --group-by task for one dataset per task.

modality.json is written to match deas-gr00t15's `LiberoDataConfig`
(`state.eef_pos_absolute`, `state.eef_rot_absolute`, `state.gripper_close`;
`action.eef_pos_delta`, `action.eef_rot_delta`, `action.gripper_close`) and
also carries the reward / done / next_state / next_video sections the RL
configs require.

Note on the gripper width: LIBERO reports `robot0_gripper_qpos`, two numbers,
and the recorder stores both under `gripper_close`. A policy trained elsewhere
may expect a single scalar there; if you are matching an existing checkpoint,
check its experiment_cfg/metadata.json and pass --gripper-scalar to collapse
the pair to one value instead.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CHUNK_SIZE = 1000
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
LANG_KEY = "annotation.human.action.task_description"
VIDEO_KEYS = ("front_view", "left_wrist_view")


def build_modality_meta(state_widths, action_widths) -> dict:
    def spans(specs):
        meta, start = {}, 0
        for key, width, rotation, absolute, original in specs:
            entry = {"start": start, "end": start + width, "dtype": "float32",
                     "absolute": absolute, "original_key": original}
            if rotation:
                entry["rotation_type"] = rotation
            meta[key] = entry
            start += width
        return meta

    state_meta = spans([
        ("eef_pos_absolute", state_widths[0], None, True, "observation.state"),
        ("eef_rot_absolute", state_widths[1], "axis_angle", True, "observation.state"),
        ("gripper_close", state_widths[2], None, True, "observation.state"),
    ])
    action_meta = spans([
        ("eef_pos_delta", action_widths[0], None, False, "action"),
        ("eef_rot_delta", action_widths[1], "axis_angle", False, "action"),
        ("gripper_close", action_widths[2], None, False, "action"),
    ])
    video_meta = {key: {"original_key": f"observation.images.{key}"} for key in VIDEO_KEYS}
    return {
        "state": state_meta,
        "action": action_meta,
        "video": video_meta,
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        "reward": {"next.reward": {"original_key": "next.reward", "dtype": "float32"}},
        "done": {"next.done": {"original_key": "next.done", "dtype": "bool"}},
        "next_state": {key: dict(value) for key, value in state_meta.items()},
        "next_video": {key: dict(value) for key, value in video_meta.items()},
    }


def build_feature_meta(state_dim, action_dim, image_shape, fps) -> dict:
    height, width, channels = image_shape
    video_info = {"video.fps": float(fps), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                  "video.is_depth_map": False, "has_audio": False}
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim],
                              "names": [f"state_{i}" for i in range(state_dim)]},
        "action": {"dtype": "float32", "shape": [action_dim],
                   "names": [f"action_{i}" for i in range(action_dim)]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        LANG_KEY: {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    for key in VIDEO_KEYS:
        features[f"observation.images.{key}"] = {
            "dtype": "video", "shape": [height, width, channels],
            "names": ["height", "width", "channel"], "video_info": video_info}
    return features


def probe_video(path: Path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        return np.asarray(reader.get_data(0)).shape
    finally:
        reader.close()


def load_shards(shards_root: Path, group_by: str):
    groups = defaultdict(list)
    for meta_path in sorted(shards_root.rglob("shard.json")):
        shard = meta_path.parent
        if not (shard / "frames.parquet").is_file():
            print(f"[skip] {shard}: no frames.parquet")
            continue
        meta = json.loads(meta_path.read_text())
        if not meta.get("complete", True):
            print(f"[skip] {shard}: interrupted partial episode")
            continue
        # <shards_root>/<suite>/<task>/<shard>
        suite = shard.parent.parent.name
        key = suite if group_by == "suite" else f"{suite}/{meta['task_name']}"
        groups[key].append((shard, meta))
    return groups


def assemble(name: str, shards, output_root: Path, *, overwrite, move, gripper_scalar):
    dataset_root = output_root / name.replace("/", "__")
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_root} exists; pass --overwrite")
        shutil.rmtree(dataset_root)

    ordered_tasks, task_to_index = [], {}
    episode_rows = []
    global_index = 0
    successes = 0
    fps = shards[0][1]["fps"]
    state_dim = action_dim = None
    image_shape = None

    for episode_index, (shard, meta) in enumerate(shards):
        frame_df = pd.read_parquet(shard / "frames.parquet")
        state = np.stack(frame_df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(frame_df["action"].to_numpy()).astype(np.float32)
        if gripper_scalar:
            # LIBERO's gripper qpos is a symmetric pair; its difference is the
            # single "openness" scalar some pipelines train on.
            state = np.concatenate([state[:, :6], (state[:, 6:7] - state[:, 7:8])], axis=1)
        frames = state.shape[0]
        state_dim = state_dim or state.shape[1]
        action_dim = action_dim or action.shape[1]

        task_text = meta["task_text"]
        if task_text not in task_to_index:
            task_to_index[task_text] = len(ordered_tasks)
            ordered_tasks.append(task_text)
        task_index = task_to_index[task_text]
        successes += bool(meta["success"])

        pd.DataFrame({
            "observation.state": list(state),
            "action": list(action),
            "timestamp": (np.arange(frames) / fps).astype(np.float32),
            LANG_KEY: np.full(frames, task_index, dtype=np.int64),
            "task_index": np.full(frames, task_index, dtype=np.int64),
            "episode_index": np.full(frames, episode_index, dtype=np.int64),
            "frame_index": np.arange(frames, dtype=np.int64),
            "index": np.arange(global_index, global_index + frames, dtype=np.int64),
            "next.reward": frame_df["next.reward"].to_numpy().astype(np.float32),
            "next.done": frame_df["next.done"].to_numpy().astype(bool),
            **({"next.terminated": frame_df["next.terminated"].to_numpy().astype(bool)}
               if "next.terminated" in frame_df else {}),
        }).to_parquet(
            _prepared(dataset_root / DATA_PATH_TEMPLATE.format(
                episode_chunk=episode_index // DEFAULT_CHUNK_SIZE,
                episode_index=episode_index)), index=False)

        for key in VIDEO_KEYS:
            source = shard / "videos" / f"{key}.mp4"
            if image_shape is None:
                image_shape = probe_video(source)
            target = _prepared(dataset_root / VIDEO_PATH_TEMPLATE.format(
                episode_chunk=episode_index // DEFAULT_CHUNK_SIZE,
                video_key=f"observation.images.{key}", episode_index=episode_index))
            # Already encoded by the recorder; never re-encode.
            (shutil.move if move else shutil.copy2)(str(source), str(target))

        episode_rows.append({"episode_index": episode_index, "tasks": [task_text],
                             "length": int(frames), "success": bool(meta["success"]),
                             "source_shard": shard.name, "task_name": meta["task_name"]})
        global_index += frames

    gripper_state_width = state_dim - 6
    gripper_action_width = action_dim - 6
    _write_json(dataset_root / "meta" / "info.json", {
        "codebase_version": "v2.1", "robot_type": "Panda",
        "total_episodes": len(episode_rows), "total_frames": global_index,
        "total_tasks": len(ordered_tasks), "total_videos": len(episode_rows) * len(VIDEO_KEYS),
        "total_chunks": 1, "chunks_size": DEFAULT_CHUNK_SIZE, "fps": float(fps),
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": DATA_PATH_TEMPLATE, "video_path": VIDEO_PATH_TEMPLATE,
        "features": build_feature_meta(state_dim, action_dim, image_shape, fps)})
    _write_json(dataset_root / "meta" / "modality.json",
                build_modality_meta((3, 3, gripper_state_width),
                                    (3, 3, gripper_action_width)))
    (dataset_root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n"
                for i, t in enumerate(ordered_tasks)))
    (dataset_root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows))

    print(f"[ok] {name}: {len(episode_rows)} episodes ({successes} successful), "
          f"{global_index} frames, {len(ordered_tasks)} tasks -> {dataset_root}")
    return dataset_root


def _prepared(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    _prepared(path).write_text(json.dumps(payload, indent=4) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=Path, required=True,
                        help="The shards/ directory collect_libero.py wrote")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group-by", choices=("suite", "task"), default="suite")
    parser.add_argument("--gripper-scalar", action="store_true",
                        help="Collapse the gripper qpos pair to one openness value")
    parser.add_argument("--move", action="store_true",
                        help="Move the shard videos instead of copying, freeing the shards")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    groups = load_shards(args.shards.expanduser().resolve(), args.group_by)
    if not groups:
        raise SystemExit(f"No shards under {args.shards}")
    for name, shards in sorted(groups.items()):
        assemble(name, sorted(shards, key=lambda item: item[0].name),
                 args.output_root.expanduser().resolve(), overwrite=args.overwrite,
                 move=args.move, gripper_scalar=args.gripper_scalar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
