"""Worker preflight for both actors, BoN50 and exact input/RNG replay."""
import json,sys,random,gc
from pathlib import Path
import numpy as np
import torch
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.iql_bon_policy import CheckpointIQLBoNPolicy
from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env

root=Path(sys.argv[1]);cfg=json.loads((root/'manifest.json').read_text());reports=[]
for label,actor in cfg['actors'].items():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    data=DATA_CONFIG_MAP['single_panda_gripper_rl_inference'](AS=16)
    policy=CheckpointIQLBoNPolicy(actor,cfg['critic'],'new_embodiment',data.modality_config(),data.transform(),num_samples=50)
    env=load_robocasa_gym_env('CoffeeSetupMug',n_envs=1,seed=42,obj_instance_split='B',layout_and_style_ids=((1,1),(2,2),(4,4),(6,9),(7,10)),action_horizon=16,camera_widths=256,camera_heights=256)
    try:
        obs,_=env.reset();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();nr=np.random.get_state();pr=random.getstate()
        action=policy.get_action(obs);scores=policy.last_scores.clone()
        assert scores.shape==(50,1) and torch.isfinite(scores).all()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);np.random.set_state(nr);random.setstate(pr)
        replay=policy.get_action(obs)
        for k in action:np.testing.assert_allclose(action[k],replay[k],atol=1e-5,rtol=0)
        torch.testing.assert_close(policy.last_scores,scores,atol=1e-5,rtol=0)
        _,reward,terminated,truncated,info=env.step(action)
        assert np.isfinite(reward).all();assert 0<int(info['num_executed_steps'][0])<=16
        reports.append(dict(actor=label,status='passed',scores_shape=list(scores.shape),rng_replay=True,simulator_step=True))
    finally:env.close()
    del policy,env;gc.collect();torch.cuda.empty_cache()
(root/'preflight.json').write_text(json.dumps(reports,indent=2)+'\n');print('PREFLIGHT_PASSED',json.dumps(reports),flush=True)
