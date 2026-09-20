"""CPU-only MP4 checks with tiny rendered frames; no simulator, model or GPU."""
import av
import gymnasium as gym
import numpy as np
import pytest

from gr00t.eval.rollout import evaluate_vector_policy
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo


class TinyRenderedKitchen(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    render_mode = "rgb_array"
    observation_space = gym.spaces.Dict({"state.x": gym.spaces.Box(-1, 1, (1,))})
    action_space = gym.spaces.Dict({"action.x": gym.spaces.Box(-1, 1, (1,))})

    def __init__(self, terminate_at=None):
        self.episode = -1
        self.steps = 0
        self.terminate_at = terminate_at
        self.closed = False
        self.fail_reset = False
        self.frame = np.empty((24, 32, 3), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        if self.fail_reset:
            raise RuntimeError("simulator reset failed")
        self.episode += 1
        self.steps = 0
        return {"state.x": np.zeros(1, dtype=np.float32)}, {"success": False}

    def step(self, action):
        self.steps += 1
        return {"state.x": np.zeros(1, dtype=np.float32)}, 0., self.steps == self.terminate_at, False, {"success": True}

    def render(self):
        self.frame[:] = self.episode * 20 + self.steps
        return self.frame[::-1]  # Match RoboCasa's vertically flipped, negative-stride view.

    def close(self):
        self.closed = True


class ZeroPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return {key: np.zeros_like(value) for key, value in self.env.action_space.sample().items()}


def decode(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        codec = stream.codec_context.name
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    return fps, codec, frames


def test_ten_outer_truncated_episodes_yield_ten_decodable_videos(tmp_path):
    videos = tmp_path / "videos" / "env_0"
    base = TinyRenderedKitchen()
    recorder = RecordVideo(base, videos, episode_trigger=lambda episode: True, fps=20)
    chunked = MultiStepWrapper(recorder, np.array([0]), np.array([0]), 4, max_episode_steps=6)
    env = gym.vector.SyncVectorEnv([lambda: chunked])
    try:
        successes, lengths = evaluate_vector_policy(ZeroPolicy(env), env, 10)
        assert successes == [True] * 10
        assert lengths == [6] * 10
        assert not hasattr(recorder, "recorded_frames")  # Frames are streamed, not retained per episode.
    finally:
        env.close()
    paths = sorted(videos.glob("*.mp4"))
    assert len(paths) == 10
    assert not list(videos.glob("*.partial"))
    assert base.closed
    for episode, path in enumerate(paths):
        assert path.name == f"rl-video-episode-{episode}.mp4"
        fps, codec, frames = decode(path)
        assert fps == 20
        assert codec == "h264"
        assert len(frames) == 7  # Reset image plus all six executed simulator steps.
        assert frames[0].shape == (24, 32, 3)
        assert abs(float(frames[0].mean()) - episode * 20) < 3
        assert abs(float(frames[-1].mean()) - (episode * 20 + 6)) < 3


def test_native_termination_flushes_and_reset_only_episode_is_omitted(tmp_path):
    env = RecordVideo(TinyRenderedKitchen(terminate_at=2), tmp_path,
                      episode_trigger=lambda episode: True, fps=20)
    env.reset()
    env.step({"action.x": np.zeros(1)})
    env.step({"action.x": np.zeros(1)})
    assert (tmp_path / "rl-video-episode-0.mp4").exists()
    assert not env.recording
    env.reset()
    env.close()
    assert len(list(tmp_path.glob("*.mp4"))) == 1
    assert len(decode(tmp_path / "rl-video-episode-0.mp4")[2]) == 3
    assert not list(tmp_path.glob("*.partial"))


def test_reset_failure_preserves_the_preceding_video(tmp_path):
    base = TinyRenderedKitchen()
    env = RecordVideo(base, tmp_path, episode_trigger=lambda episode: True)
    env.reset()
    env.step({"action.x": np.zeros(1)})
    base.fail_reset = True
    try:
        with pytest.raises(RuntimeError, match="simulator reset failed"):
            env.reset()
    finally:
        env.close()
    assert len(decode(tmp_path / "rl-video-episode-0.mp4")[2]) == 2
    assert base.closed
    assert not list(tmp_path.glob("*.partial"))
