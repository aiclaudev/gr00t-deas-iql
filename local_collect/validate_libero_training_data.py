#!/usr/bin/env python3
"""Validate collected MP4/row alignment and actual N1.7 training preprocessing."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import imageio.v2 as imageio


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    root=args.dataset
    info=json.loads((root/'meta/info.json').read_text())
    episodes=[json.loads(s) for s in (root/'meta/episodes.jsonl').read_text().splitlines()]
    assert len(episodes)>=4, 'Fewer than four complete collected episodes'
    checked=[]
    for ep in episodes:
        fmt=dict(episode_chunk=ep['episode_index']//info['chunks_size'],episode_index=ep['episode_index'])
        frame=pd.read_parquet(root/info['data_path'].format(**fmt));n=len(frame)
        assert n==ep['length'] and frame['next.done'].iloc[-1]
        assert np.array_equal(frame['frame_index'],np.arange(n))
        np.testing.assert_allclose(frame['timestamp'],np.arange(n)/info['fps'],atol=1e-5)
        assert np.isfinite(np.stack(frame['action'])).all()
        assert np.isfinite(np.stack(frame['observation.state'])).all()
        for key,feature in info['features'].items():
            if feature['dtype']!='video':continue
            path=root/info['video_path'].format(**fmt,video_key=key)
            assert path.suffix=='.mp4'
            reader=imageio.get_reader(path);count=0
            try:
                assert abs(reader.get_meta_data()['fps']-info['fps'])<.01
                for image in reader:
                    assert list(image.shape)==feature['shape']
                    if count==0:assert image.std()>1, 'Blank camera image'
                    count+=1
            finally:reader.close()
            assert count==n, f'{path}: MP4 frames={count}, rows={n}'
            checked.append(dict(episode=ep['episode_index'],camera=key,frames=count))
    # Add N1.7 aliases without changing stored states/actions, videos, or N1.5 keys.
    path=root/'meta/modality.json';meta=json.loads(path.read_text())
    for modality,width in [('state',8),('action',7)]:
        for i,key in enumerate(['x','y','z','roll','pitch','yaw']):
            meta[modality][key]={'start':i,'end':i+1,'original_key':'observation.state' if modality=='state' else 'action'}
        meta[modality]['gripper']={'start':6,'end':width,'original_key':'observation.state' if modality=='state' else 'action'}
    meta['video']['image']=dict(meta['video']['front_view'])
    meta['video']['wrist_image']=dict(meta['video']['left_wrist_view'])
    path.write_text(json.dumps(meta,indent=2)+'\n')
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_stats, generate_rel_stats
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from gr00t.utils import video_utils
    video_utils._DEFAULT_DECODER_KWARGS['num_ffmpeg_threads']=1
    generate_stats(root);generate_rel_stats(root,EmbodimentTag('libero_sim'))
    processor=Gr00tN1d7Processor.from_pretrained(args.model/'processor')
    processor.train()
    ds=ShardedSingleStepDataset(root,EmbodimentTag('libero_sim'),MODALITY_CONFIGS['libero_sim'],
                               shard_size=64,episode_sampling_rate=1.0,allow_padding=False)
    mixture=ShardedMixtureDataset([ds],[1.0],processor,num_shards_per_epoch=1,training=True)
    samples=[]
    for ep_index in range(len(episodes)):
        episode=ds.episode_loader[ep_index]
        for start in [0,max(0,ds.get_effective_episode_length(ep_index)-1)]:
            sample=ds.get_datapoint(episode,start)
            assert sample is not None
            samples.append({'episode':ep_index,'start':start,'sample_type':type(sample).__name__,
                            'keys':list(sample) if isinstance(sample,dict) else None})
    report=dict(status='passed',episodes=len(episodes),mp4_checks=checked,training_samples=samples,
                training_loader='N1.7 ShardedSingleStepDataset + checkpoint Gr00tN1d7Processor',
                scope='Dataset decoding and training-input preprocessing; no model update or success-rate benchmark')
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('TRAINING_DATA_VALIDATED',json.dumps(report),flush=True)

if __name__=='__main__':main()
