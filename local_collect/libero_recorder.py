#!/usr/bin/env python3
"""Record LIBERO env steps to disk, one shard per episode, images as video.

Sits directly on the raw LiberoEnv, below the video and multi-step wrappers, so
it sees every individual simulator step rather than the action chunks the
policy issues.

Each episode becomes a shard directory holding one mp4 per camera plus a
parquet of the low-dimensional stream. Frames go into the mp4 writers as they
arrive, so an episode never holds more than one frame per camera in memory and
no raw image array is ever written to disk. Collection runs across spawned
worker processes, so each worker writes its own shards and
libero_shards_to_lerobot.py assembles them afterwards.

Episodes are finalised on reset, which is also when a vector env autoresets,
and on close, so a run that stops early still leaves complete shards behind.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

# (gr00t key, LiberoEnv obs key)
VIDEO_SOURCES: list[tuple[str, str]] = [
    ("front_view", "video.image"),
    ("left_wrist_view", "video.wrist_image"),
]
# (gr00t key, LiberoEnv obs keys concatenated in order, rotation type)
STATE_SOURCES: list[tuple[str, tuple[str, ...], str | None]] = [
    ("eef_pos_absolute", ("state.x", "state.y", "state.z"), None),
    ("eef_rot_absolute", ("state.roll", "state.pitch", "state.yaw"), "axis_angle"),
    ("gripper_close", ("state.gripper",), None),
]
# (gr00t key, action-dict keys concatenated in order, rotation type)
ACTION_SOURCES: list[tuple[str, tuple[str, ...], str | None]] = [
    ("eef_pos_delta", ("action.x", "action.y", "action.z"), None),
    ("eef_rot_delta", ("action.roll", "action.pitch", "action.yaw"), "axis_angle"),
    ("gripper_close", ("action.gripper",), None),
]


def flatten(source: dict, keys) -> np.ndarray:
    parts = []
    for key in keys:
        parts.append(np.asarray(source[key], dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


class EpisodeRecorder(gym.Wrapper):
    """Stream one shard per episode into ``shard_dir``."""

    def __init__(self, env, shard_dir, task_name: str, env_index: int = 0, fps: int = 20,
                 min_length: int = 2, max_episode_steps: int | None = None):
        super().__init__(env)
        self.shard_dir = Path(shard_dir)
        self.task_name = task_name
        self.env_index = env_index
        self.fps = fps
        self.min_length = min_length
        self.max_episode_steps = max_episode_steps
        self._observation = None
        self._episode = 0
        self._reset_writers()

    def _reset_writers(self):
        self._writers = None
        self._shard = None
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._successes = []
        self._terminated = []
        self._task_text = None

    def _open_shard(self):
        import imageio.v2 as imageio

        self._shard = self.shard_dir / f"env{self.env_index:03d}_ep{self._episode:05d}"
        (self._shard / "videos").mkdir(parents=True, exist_ok=True)
        self._writers = {
            key: imageio.get_writer(self._shard / "videos" / f"{key}.mp4", fps=self.fps,
                                    codec="libx264", pixelformat="yuv420p", macro_block_size=1)
            for key, _ in VIDEO_SOURCES
        }

    def _record_frame(self, observation):
        if self._writers is None:
            self._open_shard()
        for key, source in VIDEO_SOURCES:
            frame = np.asarray(observation[source])
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            self._writers[key].append_data(frame)
        self._states.append(flatten(observation, [k for _, keys, _ in STATE_SOURCES for k in keys]))
        if self._task_text is None:
            self._task_text = str(observation["annotation.human.action.task_description"])

    def _finalise(self):
        if self._shard is None:
            return
        for writer in self._writers.values():
            writer.close()
        # Only pre-action observations are encoded: one MP4 frame per row.
        frames = min(len(self._states), len(self._actions))
        if frames < self.min_length:
            import shutil

            shutil.rmtree(self._shard, ignore_errors=True)
            self._reset_writers()
            self._episode += 1
            return

        import pandas as pd

        state = np.stack(self._states[:frames]).astype(np.float32)
        action = np.stack(self._actions[:frames]).astype(np.float32)
        reward = np.asarray(self._rewards[:frames], dtype=np.float32)
        done = np.asarray(self._dones[:frames], dtype=bool)
        success = bool(np.any(self._successes[:frames]))
        complete = bool(done[-1])

        pd.DataFrame({
            "observation.state": list(state),
            "action": list(action),
            "next.reward": reward,
            "next.done": done,
            "next.terminated": np.asarray(self._terminated[:frames], dtype=bool),
        }).to_parquet(self._shard / "frames.parquet", index=False)

        (self._shard / "shard.json").write_text(json.dumps({
            "task_name": self.task_name,
            "task_text": self._task_text or self.task_name,
            "env_index": self.env_index,
            "episode": self._episode,
            "length": int(frames),
            "success": success,
            "complete": complete,
            "fps": self.fps,
            "video_keys": [key for key, _ in VIDEO_SOURCES],
        }, indent=2) + "\n")

        self._reset_writers()
        self._episode += 1

    def reset(self, **kwargs):
        self._finalise()
        observation, info = self.env.reset(**kwargs)
        self._observation = observation
        return observation, info

    def step(self, action):
        self._record_frame(self._observation)
        self._actions.append(flatten(action, [k for _, keys, _ in ACTION_SOURCES for k in keys]))
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._rewards.append(float(reward))
        success = bool(info.get("success", False))
        at_limit = self.max_episode_steps is not None and len(self._actions) >= self.max_episode_steps
        self._dones.append(bool(terminated or truncated or success or at_limit))
        self._terminated.append(bool(terminated or success))
        self._successes.append(bool(info.get("success", False)))
        self._observation = observation
        return observation, reward, terminated, truncated, info

    def close(self):
        self._finalise()
        return self.env.close()
