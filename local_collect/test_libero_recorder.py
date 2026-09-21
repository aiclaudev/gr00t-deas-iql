import json
import sys
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from libero_recorder import EpisodeRecorder

class TinyEnv(gym.Env):
    observation_space=gym.spaces.Dict({})
    action_space=gym.spaces.Dict({})
    def obs(self):
        return {'video.image':np.full((16,16,3),self.i*40,np.uint8),
                'video.wrist_image':np.full((16,16,3),self.i*40,np.uint8),
                **{f'state.{k}':np.array([self.i],np.float32) for k in ['x','y','z','roll','pitch','yaw']},
                'state.gripper':np.zeros(2),'annotation.human.action.task_description':'test'}
    def reset(self,**kwargs):self.i=0;return self.obs(),{}
    def step(self,a):self.i+=1;return self.obs(),0.,False,False,{'success':False}

def test_video_matches_pre_action_rows_and_partial_is_flagged(tmp_path):
    env=EpisodeRecorder(TinyEnv(),tmp_path,'test',max_episode_steps=3)
    action={f'action.{k}':np.array([.2]) for k in ['x','y','z','roll','pitch','yaw','gripper']}
    env.reset()
    for _ in range(3):env.step(action)
    env.reset()
    for _ in range(2):env.step(action)
    env.close()
    first=tmp_path/'env000_ep00000'
    frame=pd.read_parquet(first/'frames.parquet')
    assert len(frame)==3 and frame['next.done'].tolist()==[False,False,True]
    assert [x[0] for x in frame['observation.state']]==[0,1,2]
    reader=imageio.get_reader(first/'videos/front_view.mp4')
    try: images=list(iter(reader))
    finally:reader.close()
    assert len(images)==3
    assert np.allclose([im.mean() for im in images],[0,40,80],atol=3)
    assert json.loads((first/'shard.json').read_text())['complete']
    assert not json.loads((tmp_path/'env000_ep00001/shard.json').read_text())['complete']
