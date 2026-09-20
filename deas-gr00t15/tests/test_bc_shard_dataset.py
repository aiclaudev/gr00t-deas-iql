from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import cv2
from gr00t.data.bc_shard_dataset import cache_episode, BCShardDataset
from gr00t.utils.video import get_frames_by_timestamps


def test_cached_frames_match_random_reader_and_padding(tmp_path):
    path = tmp_path / 'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((32, 32, 3), i*25, dtype=np.uint8))
    writer.release()
    timestamps = np.array([0., .051, .15, .29, .42, .51, .6, .7])
    frame = pd.DataFrame({'timestamp': timestamps})
    source = SimpleNamespace(
        get_trajectory_data=lambda _: frame,
        get_video_path=lambda *_: path,
        modality_keys={'video': ['video.camera']},
        delta_indices={'video.camera': np.array([-1, 0, 1, 4])},
    )
    source._delta_indices = source.delta_indices
    cached, size = cache_episode(source, 0, 1024**2)
    assert size == 8*32*32*3
    for step in [0, 3, 7]:
        indices = np.clip(source.delta_indices['video.camera']+step, 0, 7)
        expected = get_frames_by_timestamps(str(path), timestamps[indices], video_backend_kwargs={'num_threads': 1})
        np.testing.assert_array_equal(cached.get_video(0, 'video', 'video.camera', step), expected)
    with pytest.raises(MemoryError):
        cache_episode(source, 0, 1)


def test_sampling_weights_and_shard_shuffle(monkeypatch):
    import gr00t.data.bc_shard_dataset as module
    source = SimpleNamespace(trajectory_ids=np.array([3, 8]), trajectory_lengths=np.array([10, 30]), use_rl=False, video_backend='decord')
    stream = BCShardDataset(source, episodes_per_shard=4, samples_per_episode=16)
    np.testing.assert_allclose(stream.episode_weights[0], [.25, .75])
    monkeypatch.setattr(module, 'cache_episode', lambda source, eid, budget: (SimpleNamespace(curr_traj_data=range(10 if eid==3 else 30), eid=eid), 1))
    episodes, positions = stream._load_shard(42)
    assert len(positions) == 64
    assert all(sum(i == e for i, _ in positions) == 16 for e in range(4))
    assert all(0 <= step < len(episodes[e].curr_traj_data) for e, step in positions)
    assert positions == stream._load_shard(42)[1]


def test_iterable_trainer_sampler():
    from gr00t.experiment.trainer import DualBrainTrainer
    source = SimpleNamespace(trajectory_ids=np.array([0]), trajectory_lengths=np.array([8]), use_rl=False, video_backend='decord')
    stream = BCShardDataset(source)
    assert DualBrainTrainer._get_train_sampler(SimpleNamespace(train_dataset=stream)) is None


def test_runner_accepts_stream_without_length(monkeypatch, tmp_path):
    import gr00t.experiment.runner as module
    from torch.utils.data import IterableDataset
    class Stream(IterableDataset):
        def __iter__(self):
            yield {}
    class Trainer:
        def __init__(self, **kwargs):
            self.train_dataset = kwargs['train_dataset']
        def add_callback(self, callback):
            pass
        def get_train_dataloader(self):
            raise AssertionError('Logging must not construct or size the streaming loader')
    monkeypatch.setattr(module, 'DualBrainTrainer', Trainer)
    runner = SimpleNamespace(exp_cfg_dir=tmp_path)
    result = module.TrainRunner.create_trainer(runner, None, SimpleNamespace(run_name='test'), Stream(), None, None)
    assert isinstance(result, Trainer)
