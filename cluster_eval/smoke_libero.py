"""Short two-process rendering/step correctness check; no policy benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
import gymnasium as gym


def make_env():
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    register_libero_envs()
    from libero.libero import benchmark
    task=benchmark.get_benchmark_dict()['libero_spatial']().get_task_names()[0]
    return gym.make('libero_sim/'+task)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    env=gym.vector.AsyncVectorEnv([make_env,make_env],context='spawn',shared_memory=False)
    try:
        obs,_=env.reset(seed=42)
        shapes={k:list(np.asarray(v).shape) for k,v in obs.items()}
        for _ in range(2):
            obs,reward,terminated,truncated,info=env.step(env.action_space.sample())
            assert np.isfinite(reward).all()
        report=dict(status='passed',n_envs=2,steps=2,observation_shapes=shapes)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    finally:env.close()

if __name__=='__main__':main()
