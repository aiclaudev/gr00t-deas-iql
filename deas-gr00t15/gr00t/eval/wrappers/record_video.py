
from __future__ import annotations

import os
import sys
from typing import Any, Callable, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame


class RecordVideo(gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs):
    """Streams episode RGB frames to H.264 MP4 using the installed PyAV encoder.

    Reset and close finalize episodes when an outer wrapper imposes truncation.
    Recordings containing only a reset frame are discarded.

    .. py:currentmodule:: gymnasium.utils.save_video

    Usually, you only want to record episodes intermittently, say every hundredth episode or at every thousandth environment step.
    To do this, you can specify ``episode_trigger`` or ``step_trigger``.
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.

    The ``episode_trigger`` should return ``True`` on the episode when recording should start.
    The ``step_trigger`` should return ``True`` on the n-th environment step that the recording should be started, where n sums over all previous episodes.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed, i.e. :func:`capped_cubic_video_schedule`.
    This function starts a video at every episode that is a power of 3 until 1000 and then every 1000 episodes.
    By default, the recording will be stopped once reset is called.
    However, you can also create recordings of fixed length (possibly spanning several episodes)
    by passing a strictly positive value for ``video_length``.

    No vector version of the wrapper exists.

    Examples - Run the environment for 50 episodes, and save the video every 10 episodes starting from the 0th:
        >>> import os
import sys
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 10 == 0
        >>> env = RecordVideo(env, video_folder="./save_videos1", episode_trigger=trigger, disable_logger=True)
        >>> for i in range(50):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos1"))
        5

    Examples - Run the environment for 5 episodes, start a recording every 200th step, making sure each video is 100 frames long:
        >>> import os
import sys
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 200 == 0
        >>> env = RecordVideo(env, video_folder="./save_videos2", step_trigger=trigger, video_length=100, disable_logger=True)
        >>> for i in range(5):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     _ = env.action_space.seed(123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos2"))
        2

    Examples - Run 3 episodes, record everything, but in chunks of 1000 frames:
        >>> import os
import sys
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> env = RecordVideo(env, video_folder="./save_videos3", video_length=1000, disable_logger=True)
        >>> for i in range(3):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos3"))
        2

    Change logs:
     * v0.25.0 - Initially added to replace ``wrappers.monitoring.VideoRecorder``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            fps (int): The frame per second in the video. Provides a custom video fps for environment, if ``None`` then
                the environment metadata ``render_fps`` key is used if it exists, otherwise a default value of 30 is used.
            disable_logger (bool): Retained for compatibility; PyAV does not display a progress bar
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        if fps <= 0 or video_length < 0:
            raise ValueError("fps must be positive and video_length must be nonnegative")
        self.frames_per_sec: int = fps
        self.name_prefix: str = name_prefix
        self._video_name: str | None = None
        self.video_length: int = video_length if video_length != 0 else float("inf")
        self.recording: bool = False
        self._container = None
        self._stream = None
        self._frame_shape = None
        self._video_path: str | None = None
        self._partial_path: str | None = None
        self._frames_recorded = 0
        self._recorded_steps = 0
        self.render_history: list[RenderFrame] = []

        self.step_id = -1
        self.episode_id = -1

        try:
            import av
            av.Codec("libx264", "w")
        except (ImportError, ValueError) as exc:
            raise error.DependencyNotInstalled("Video recording requires PyAV with the libx264 encoder") from exc
        self._av = av

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."
        frame = self.env.render()
        if isinstance(frame, list):
            if not frame:
                return
            self.render_history += frame
            frame = frame[-1]
        self._write_frame(frame)

    def _write_frame(self, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError("Video render must return an H x W x RGB/RGBA numpy array")
        if frame.dtype != np.uint8:
            raise ValueError(f"Video render must return uint8 pixels, got {frame.dtype}")
        if self._frame_shape is not None and frame.shape != self._frame_shape:
            raise ValueError("Video frame shape changed within an episode")
        if self._container is None:
            self._frame_shape = frame.shape
            self._video_path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            self._partial_path = self._video_path + ".partial"
            if os.path.exists(self._video_path) or os.path.exists(self._partial_path):
                raise FileExistsError(f"Video already exists: {self._video_path}")
            self._container = self._av.open(self._partial_path, mode="w", format="mp4")
            self._stream = self._container.add_stream("libx264", rate=self.frames_per_sec)
            self._stream.width = frame.shape[1] + frame.shape[1] % 2
            self._stream.height = frame.shape[0] + frame.shape[0] % 2
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": "23", "preset": "veryfast"}
            self._stream.codec_context.thread_count = 1
        # RoboCasa vertically flips its render view, which produces negative strides.
        # Pad odd dimensions for widely supported yuv420p output, then make contiguous.
        frame = np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)), mode="edge")
        pixel_format = "rgb24" if frame.shape[-1] == 3 else "rgba"
        video_frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format=pixel_format)
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)
        self._frames_recorded += 1

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        # Finish the previous episode before a potentially failing simulator reset.
        if self.recording and self.video_length == float("inf"):
            self.stop_recording()
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            if self._frames_recorded >= self.video_length:
                self.stop_recording()

        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._recorded_steps += 1
            self._capture_frame()
            if self._frames_recorded >= self.video_length or (
                self.video_length == float("inf") and (terminated or truncated)
            ):
                self.stop_recording()

        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        render_out = super().render()
        if self.recording and isinstance(render_out, list):
            for frame in render_out:
                self._write_frame(frame)
        if self.render_history:
            history, self.render_history = self.render_history, []
            return history + (render_out if isinstance(render_out, list) else [render_out])
        return render_out

    def close(self):
        """Flush the last MP4 even if simulator cleanup raises."""
        try:
            if self.recording:
                self.stop_recording()
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                super().close()
            except BaseException as close_error:
                if not active_error:
                    raise
                logger.warn(f"Environment cleanup also failed: {close_error}")

    def start_recording(self, video_name: str):
        if self.recording:
            self.stop_recording()
        self.recording = True
        self._video_name = video_name
        self._frames_recorded = 0
        self._recorded_steps = 0

    def stop_recording(self):
        """Flush the encoder and publish a complete MP4; omit reset-only recordings."""
        assert self.recording, "stop_recording was called, but no recording was started"
        try:
            if self._container is not None:
                for packet in self._stream.encode(None):
                    self._container.mux(packet)
                self._container.close()
                self._container = None
                if self._frames_recorded and self._recorded_steps:
                    os.replace(self._partial_path, self._video_path)
                else:
                    os.unlink(self._partial_path)
        except BaseException:
            if self._container is not None:
                try:
                    self._container.close()
                except BaseException as close_error:
                    logger.warn(f"Video encoder cleanup also failed: {close_error}")
            if self._partial_path and os.path.exists(self._partial_path):
                try:
                    os.unlink(self._partial_path)
                except OSError as cleanup_error:
                    logger.warn(f"Could not remove unfinished video: {cleanup_error}")
            raise
        finally:
            self._container = None
            self._stream = None
            self._frame_shape = None
            self._video_path = None
            self._partial_path = None
            self._frames_recorded = 0
            self._recorded_steps = 0
            self.recording = False
            self._video_name = None

    def __del__(self):
        if getattr(self, "_frames_recorded", 0):
            logger.warn("Unable to save last video! Did you call close()?")
