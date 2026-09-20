"""CPU checks for RoboCasa success termination, without loading a simulator or model."""
import ast
from pathlib import Path
import string

import av
import gymnasium as gym
import numpy as np
import pytest

from gr00t.eval.rollout import evaluate_vector_policy
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo


def load_wrapper_class():
    # Exercise the real wrapper without importing RoboCasa / MuJoCo assets.
    source = Path(__file__).resolve().parents[1] / "gr00t/eval/wrappers/robocasa_wrapper.py"
    tree = ast.parse(source.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "RoboCasaWrapper")
    namespace = {"gym": gym, "np": np, "string": string}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["RoboCasaWrapper"]


RoboCasaWrapper = load_wrapper_class()


class TinyRoboCasa(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    render_mode = "rgb_array"
    observation_space = gym.spaces.Dict({
        "robot0_base_pos": gym.spaces.Box(-1, 1, (3,), dtype=np.float32),
    })
    action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float32)

    def __init__(self, success_at=3, reset_success=False, termination_at=None, truncation_at=None):
        self.success_at = success_at
        self.reset_success = reset_success
        self.termination_at = termination_at
        self.truncation_at = truncation_at
        self.steps = 0
        self.total_steps = 0
        self.sim = self

    def get_ep_meta(self):
        return {"lang": "complete the task"}

    def _check_success(self):
        return {"task": (self.steps == 0 and self.reset_success) or self.steps == self.success_at}

    def observation(self):
        return {"robot0_base_pos": np.zeros(3, dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        return (self.observation(), 0., self.steps == self.termination_at,
                self.steps == self.truncation_at, {})

    def render(self, **kwargs):
        return np.full((24, 32, 3), self.steps, dtype=np.uint8)


def make_chunked(raw, terminate_on_success=True, video_folder=None):
    wrapped = RoboCasaWrapper(raw, terminate_on_success=terminate_on_success)
    if video_folder is not None:
        wrapped = RecordVideo(wrapped, video_folder, episode_trigger=lambda episode: True, fps=20)
    return MultiStepWrapper(wrapped, np.array([0]), np.array([0]),
                            n_action_steps=16, max_episode_steps=20)


class ZeroPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return {key: np.zeros_like(value) for key, value in self.env.action_space.sample().items()}


def test_first_success_stops_inside_sixteen_action_chunk():
    raw = TinyRoboCasa(success_at=3)
    env = make_chunked(raw)
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(ZeroPolicy(env).get_action(None))
        assert terminated and not truncated
        assert raw.steps == info["num_executed_steps"] == 3
        assert info["success"].tolist() == [True]
        assert info["dones"].tolist() == [False, False, True]
    finally:
        env.close()


@pytest.mark.parametrize("success_at,terminate_on_success", [(None, True), (3, False)])
def test_failure_or_explicit_legacy_mode_reaches_task_horizon(success_at, terminate_on_success):
    raw = TinyRoboCasa(success_at=success_at)
    env = make_chunked(raw, terminate_on_success=terminate_on_success)
    try:
        env.reset()
        action = ZeroPolicy(env).get_action(None)
        _, _, terminated, truncated, info = env.step(action)
        assert raw.steps == info["num_executed_steps"] == 16
        assert not terminated and not truncated
        _, _, terminated, truncated, info = env.step(action)
        assert raw.steps == 20 and info["num_executed_steps"] == 4
        assert not terminated and truncated
    finally:
        env.close()


@pytest.mark.parametrize("native_end", ["termination_at", "truncation_at"])
def test_native_episode_end_is_preserved(native_end):
    raw = TinyRoboCasa(success_at=None, **{native_end: 2})
    env = make_chunked(raw)
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(ZeroPolicy(env).get_action(None))
        assert raw.steps == info["num_executed_steps"] == 2
        assert terminated == (native_end == "termination_at")
        assert truncated == (native_end == "truncation_at")
        assert info["success"].tolist() == [False]
    finally:
        env.close()


def test_reset_success_does_not_count_as_completed_success():
    raw = TinyRoboCasa(success_at=None, reset_success=True)
    env = gym.vector.SyncVectorEnv([lambda: make_chunked(raw)])
    try:
        successes, lengths = evaluate_vector_policy(ZeroPolicy(env), env, 2)
    finally:
        env.close()
    assert successes == [False, False]
    assert lengths == [20, 20]
    assert raw.total_steps == 40


def test_next_step_autoreset_counts_exact_requested_episodes():
    raw = TinyRoboCasa(success_at=3)
    env = gym.vector.SyncVectorEnv([lambda: make_chunked(raw)])
    records = []
    try:
        successes, lengths = evaluate_vector_policy(
            ZeroPolicy(env), env, 5, episode_callback=records.append)
    finally:
        env.close()
    assert successes == [True] * 5
    assert lengths == [3] * 5
    assert raw.total_steps == 15
    assert [record["episode"] for record in records] == list(range(5))
    assert [record["length"] for record in records] == lengths


@pytest.mark.parametrize("success_at,episode_length", [(3, 3), (None, 20)])
def test_success_and_failure_videos_are_finalized(tmp_path, success_at, episode_length):
    raw = TinyRoboCasa(success_at=success_at)
    env = gym.vector.SyncVectorEnv([lambda: make_chunked(raw, video_folder=tmp_path)])
    try:
        successes, lengths = evaluate_vector_policy(ZeroPolicy(env), env, 3)
        assert successes == [success_at is not None] * 3
        assert lengths == [episode_length] * 3
        if success_at is not None:
            # Success closes even the final video's encoder immediately, before env.close().
            assert len(list(tmp_path.glob("*.mp4"))) == 3
            assert not list(tmp_path.glob("*.partial"))
    finally:
        env.close()
    videos = sorted(tmp_path.glob("*.mp4"))
    assert len(videos) == 3
    assert not list(tmp_path.glob("*.partial"))
    for video in videos:
        with av.open(str(video)) as container:
            assert float(container.streams.video[0].average_rate) == 20
            assert sum(1 for _ in container.decode(video=0)) == episode_length + 1
