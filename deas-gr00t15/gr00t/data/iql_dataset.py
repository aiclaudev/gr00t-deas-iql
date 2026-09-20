"""Episode bulk decoding + next-episode prefetch for QC/IQL on DEAS data.

Original dataset files are read-only. Original critic normalization is pinned.
Only observed complete chunks enter the loss. A successful terminal at the final
transition is included; a final unknown failure lacks a true next observation
and is excluded. DEAS success reward expansion and negative reward conversion are preserved.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import RLDatasetMetadata, DatasetStatisticalValues
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import SinglePandaGripperRLDataConfig
from gr00t.model.iql.core import chunk_fields


def eligible_starts(rewards, boundaries, terminated, horizon):
    n = len(rewards)
    if n < horizon:
        return np.empty(0,dtype=np.int64)
    boundaries = np.asarray(boundaries,bool) | np.asarray(terminated,bool)
    starts = np.arange(n-horizon+1)
    cumulative = np.r_[0,np.cumsum(boundaries)]
    # End itself is valid, but no previous transition may be a boundary.
    valid = (cumulative[starts+horizon-1]-cumulative[starts]) == 0
    valid &= (starts+horizon < n) | np.asarray(terminated,bool)[starts+horizon-1]
    return starts[valid]


class EpisodeDataset(LeRobotSingleDataset):
    def __init__(self,path,actor_metadata,horizon):
        config=SinglePandaGripperRLDataConfig(AS=horizon)
        for entry in json.loads((Path(path)/'meta/stats.json').read_text()).values():
            DatasetStatisticalValues.model_validate(entry)
        super().__init__(path, config.modality_config(),EmbodimentTag.NEW_EMBODIMENT,
                         transforms=config.transform(),use_rl=True)
        metadata=deepcopy(actor_metadata)
        own=self.metadata.model_dump(mode='json')
        for key in ['reward','done']:
            metadata['modalities'][key]=own['modalities'][key]
        metadata['modalities']['next_state']=deepcopy(metadata['modalities']['state'])
        metadata['modalities']['next_video']=deepcopy(metadata['modalities']['video'])
        self.set_transforms_metadata(RLDatasetMetadata.model_validate(metadata))
        self.horizon=horizon
        self.episode_videos={}

    def load_episode(self,episode_id):
        # Called by one prefetch thread; does not touch the active episode cache.
        import decord
        from torchcodec.decoders import VideoDecoder
        path=self.dataset_path/self.data_path_pattern.format(
            episode_chunk=self.get_episode_chunk(episode_id),episode_index=episode_id)
        frame=pd.read_parquet(path)
        rewards=frame['next.reward'].to_numpy(dtype=np.float32)
        if not np.isfinite(rewards).all() or ((rewards<0)|(rewards>1)).any():
            raise ValueError('Expected finite original sparse rewards in [0,1]')
        terminated=(frame['next.terminated'].to_numpy(bool) if 'next.terminated' in frame
                    else rewards>0)
        boundaries=frame['next.done'].to_numpy(bool).copy() | terminated
        boundaries[-1]=True
        starts=eligible_starts(rewards,boundaries,terminated,self.horizon)
        if len(starts)==0:
            return None
        training_rewards=rewards.copy()
        if rewards.sum()>0: training_rewards[-15:]=1
        training_rewards-=1
        videos={}
        timestamps=frame['timestamp'].to_numpy()
        for key in self.modality_keys['video']:
            name=key.split('.',1)[1]
            video=str(self.get_video_path(episode_id,name))
            reader=decord.VideoReader(video,num_threads=1)
            frame_times=reader.get_frame_timestamp(range(len(reader)))[:,:1]
            # Preserve DEAS's nearest-frame-start mapping, including tie behavior.
            indices=np.abs(frame_times-timestamps).argmin(axis=0)
            del reader
            decoder=VideoDecoder(video,device='cpu',dimension_order='NHWC',num_ffmpeg_threads=1)
            videos[name]=decoder.get_frames_at(indices.tolist()).data.numpy()
            del decoder
        return dict(id=episode_id,frame=frame,videos=videos,rewards=training_rewards,
                    boundaries=boundaries,terminated=terminated,starts=starts)

    def activate(self,episode):
        self.curr_traj_id=episode['id'];self.curr_traj_data=episode['frame']
        self.episode_videos=episode['videos']

    def get_trajectory_data(self,trajectory_id):
        if trajectory_id!=self.curr_traj_id or self.curr_traj_data is None:
            raise RuntimeError('Activate the requested episode before sampling')
        return self.curr_traj_data

    def get_video(self,trajectory_id,modality,key,base_index):
        indices=np.clip(self.delta_indices[key]+base_index,0,len(self.curr_traj_data)-1)
        return self.episode_videos[key.split('.',1)[1]][indices]

    def get_reward_or_done(self,trajectory_id,modality,key,base_index):
        return super().get_reward_or_done(trajectory_id,modality,key,base_index)


class ChunkEpisodeStream(IterableDataset):
    def __init__(self,paths,actor_path,seed=42,horizon=16,discount=.99,samples_per_episode=64,normalization_metadata=None):
        self.paths=list(paths);self.actor_path=str(actor_path)
        self.seed=seed;self.horizon=horizon;self.discount=discount
        self.samples_per_episode=samples_per_episode
        actor=json.loads(Path(normalization_metadata).read_text())
        self.actor_metadata=actor['new_embodiment']

    def __iter__(self):
        worker=get_worker_info();worker_id=worker.id if worker else 0
        seed=self.seed+100003*worker_id
        rng=np.random.default_rng(seed)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        datasets=[EpisodeDataset(p,self.actor_metadata,self.horizon) for p in self.paths]
        dataset_weights=np.array([sum(ds.trajectory_lengths) for ds in datasets],dtype=float)
        dataset_weights/=dataset_weights.sum()
        trajectory_weights=[np.array(ds.trajectory_lengths,dtype=float)/sum(ds.trajectory_lengths) for ds in datasets]
        # The loader thread has its own RNG; augmentation uses the main worker RNG.
        def read_next():
            for _ in range(100):
                i=int(rng.choice(len(datasets),p=dataset_weights));ds=datasets[i]
                ep=ds.load_episode(int(rng.choice(ds.trajectory_ids,p=trajectory_weights[i])))
                if ep is not None:
                    starts=rng.choice(ep['starts'],size=self.samples_per_episode,
                                      replace=len(ep['starts'])<self.samples_per_episode)
                    return i,ep,starts
            raise RuntimeError('No valid QC chunks found in 100 episode draws')
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(read_next)
            while True:
                i,ep,starts=future.result()
                future=pool.submit(read_next)
                ds=datasets[i];ds.activate(ep)
                for start in starts:
                    start=int(start)
                    fields=chunk_fields(ep['rewards'],ep['boundaries'],ep['terminated'],
                                        start,self.horizon,self.discount)
                    assert fields['chunk_valid']==1
                    sample=ds.transforms(ds.get_step_data(ep['id'],start))
                    sample.update(chunk_valid=np.float32(fields['chunk_valid']),
                                  chunk_return=np.float32(fields['chunk_return']),
                                  bootstrap_mask=np.float32(fields['bootstrap_mask']),
                                  qc_valid=fields['valid'].numpy(),
                                  dataset_index=np.int64(i),episode_index=np.int64(ep['id']),
                                  start_index=np.int64(start))
                    yield sample
                # Release the active episode before moving to the prefetched one.
                ds.curr_traj_data=None;ds.curr_traj_id=None;ds.episode_videos={}
                del ep
