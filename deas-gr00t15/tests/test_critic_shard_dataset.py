from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
from gr00t.data.bc_shard_dataset import cache_episode, CriticShardDataset
from gr00t.utils.video import get_frames_by_timestamps


def test_current_and_next_share_decode_and_preserve_boundary(tmp_path):
    path = tmp_path / 'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((32, 32, 3), i * 25, dtype=np.uint8))
    writer.release()
    timestamps = np.arange(8) / 10
    calls = []
    def video_path(*args):
        calls.append(args)
        return path
    source = SimpleNamespace(
        get_trajectory_data=lambda _: pd.DataFrame({'timestamp': timestamps}),
        get_video_path=video_path,
        modality_keys={'video': ['video.camera'], 'next_video': ['next_video.camera']},
        _delta_indices={'video.camera': np.array([-1, 0]), 'next_video.camera': np.array([4, 16])},
    )
    cached, size = cache_episode(source, 0, 1024**2)
    assert len(calls) == 1
    assert size == 8 * 32 * 32 * 3
    for step in (0, 3, 7):
        for key, offsets in source._delta_indices.items():
            expected = get_frames_by_timestamps(str(path), timestamps[np.clip(step + offsets, 0, 7)], video_backend_kwargs={'num_threads': 1})
            np.testing.assert_array_equal(cached.get_video(0, key.split('.')[0], key, step), expected)


def test_rl_sampling_weights_preserved():
    source = SimpleNamespace(trajectory_ids=np.array([3, 8]), trajectory_lengths=np.array([10, 30]), use_rl=True, video_backend='decord')
    stream = CriticShardDataset(source)
    np.testing.assert_allclose(stream.episode_weights[0], [.25, .75])
