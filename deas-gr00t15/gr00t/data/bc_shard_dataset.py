"""N1.7-style bounded RAM shards and asynchronous next-shard decoding for BC.

Keeps N1.5 transforms, metadata, padding and nearest-timestamp video semantics.
Episode selection follows the existing mixture weights. Each selected episode
contributes the same number of uniformly sampled frames, preserving marginal
sample weights while changing sample order/correlation relative to random reads.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset


class _CachedEpisode(LeRobotSingleDataset):
    def get_video(self, trajectory_id, modality, key, base_index):
        indices = np.clip(self.delta_indices[key] + base_index, 0, len(self.curr_traj_data) - 1)
        cache_key = "video." + key.split(".", 1)[1]
        return self.frames[cache_key][indices]


def cache_episode(source, episode_id, byte_limit):
    import decord

    cached = object.__new__(_CachedEpisode)
    cached.__dict__ = source.__dict__.copy()
    cached.curr_traj_id = None
    cached.curr_traj_data = None
    cached.curr_traj_data = source.get_trajectory_data(episode_id)
    cached.curr_traj_id = episode_id
    cached.frames = {}
    timestamps = cached.curr_traj_data['timestamp'].to_numpy()
    used = 0
    video_keys = list(dict.fromkeys(
        "video." + key.split(".", 1)[1]
        for modality in ("video", "next_video")
        for key in source.modality_keys.get(modality, [])
    ))
    for key in video_keys:
        path = source.get_video_path(episode_id, key.split('.', 1)[1])
        reader = decord.VideoReader(str(path), num_threads=1)
        frame_times = reader.get_frame_timestamp(range(len(reader)))[:, :1]
        # Identical nearest-start-time mapping and tie handling to DEAS.
        mapping = np.concatenate([
            np.abs(frame_times - timestamps[i:i+256]).argmin(axis=0)
            for i in range(0, len(timestamps), 256)
        ])
        first = reader.get_batch(mapping[:1]).asnumpy()
        required = len(mapping) * first[0].nbytes
        if used + required > byte_limit:
            raise MemoryError(f'Episode {episode_id}: shard RAM budget exceeded ({byte_limit} bytes)')
        frames = np.empty((len(mapping), *first.shape[1:]), dtype=first.dtype)
        for i in range(0, len(mapping), 32):
            frames[i:i+32] = reader.get_batch(mapping[i:i+32]).asnumpy()
        cached.frames[key] = frames
        used += required
        del reader
    return cached, used


class BCShardDataset(IterableDataset):
    def __init__(self, source, seed=42, episodes_per_shard=4,
                 samples_per_episode=256, max_shard_gib=4, allow_rl=False):
        super().__init__()
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('BC shard loader currently supports single-GPU training only')
        if episodes_per_shard < 1 or samples_per_episode < 1 or max_shard_gib <= 0:
            raise ValueError('Shard settings must be positive')
        self.source_dataset = source
        self.seed = seed
        self.episodes_per_shard = episodes_per_shard
        self.samples_per_episode = samples_per_episode
        self.byte_limit = int(max_shard_gib * 2**30)
        if isinstance(source, LeRobotMixtureDataset):
            self.datasets = source.datasets
            self.weights = source.dataset_sampling_weights
            self.episode_weights = source.trajectory_sampling_weights
        else:
            self.datasets = [source]
            self.weights = np.array([1.])
            lengths = np.asarray(source.trajectory_lengths, dtype=float)
            self.episode_weights = [lengths / lengths.sum()]
        if not allow_rl and any(d.use_rl for d in self.datasets):
            raise ValueError('BC shard loader cannot be used for RL datasets')
        if any(d.video_backend != 'decord' for d in self.datasets):
            raise ValueError('BC shard loader preserves Decord frame semantics; use video_backend=decord')

    def _load_shard(self, seed):
        rng = np.random.default_rng(seed)
        episodes, positions = [], []
        remaining = self.byte_limit
        for _ in range(self.episodes_per_shard):
            d = int(rng.choice(len(self.datasets), p=self.weights))
            source = self.datasets[d]
            e = int(rng.choice(len(source.trajectory_ids), p=self.episode_weights[d]))
            cached, used = cache_episode(source, source.trajectory_ids[e], remaining)
            remaining -= used
            # Transforms execute on the foreground worker only, not in prefetch.
            episodes.append(cached)
            steps = rng.integers(0, len(cached.curr_traj_data), size=self.samples_per_episode)
            positions.extend((len(episodes)-1, int(step)) for step in steps)
        rng.shuffle(positions)
        return episodes, positions

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, worker_id]))
        # At most current + next raw shard per worker; no transformed-frame cache.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._load_shard, int(rng.integers(2**63)))
            while True:
                start = time.perf_counter()
                episodes, positions = future.result()
                print(f'BC_SHARD worker={worker_id} samples={len(positions)} wait_seconds={time.perf_counter()-start:.4f}', flush=True)
                future = pool.submit(self._load_shard, int(rng.integers(2**63)))
                for episode_index, step in positions:
                    episode = episodes[episode_index]
                    yield episode.transforms(episode.get_step_data(episode.curr_traj_id, step))
                del episode, episodes, positions


class CriticShardDataset(BCShardDataset):
    """Cache current/next RGB together; retain the original RL sample and transforms."""
    def __init__(self, source, **kwargs):
        super().__init__(source, allow_rl=True, **kwargs)
        if not all(d.use_rl for d in self.datasets):
            raise ValueError("Critic shard loader requires RL datasets")
