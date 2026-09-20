"""Check real kitchen assets, EGL, observations and chunked actions; no policy score."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-name', default='CoffeeSetupMug')
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env
    from gr00t.model.policy import Gr00tPolicy, Gr00tDEASDualBoNPolicy  # noqa: F401
    args.output.mkdir(parents=True, exist_ok=True)
    env = load_robocasa_gym_env(
        args.env_name, n_envs=args.n_envs, seed=42,
        obj_instance_split='B', layout_and_style_ids=((1, 1),),
        action_horizon=4, camera_widths=256, camera_heights=256,
    )
    try:
        obs, _ = env.reset()
        shapes = {key: list(np.asarray(value).shape) for key, value in obs.items()}
        for key in ('video.left_view', 'video.right_view', 'video.wrist_view'):
            frames = obs[key]
            assert frames.shape == (args.n_envs, 1, 256, 256, 3), (key, frames.shape)
            assert frames.dtype == np.uint8 and frames.std() > 1, key
            imageio.imwrite(args.output / f'{key}.png', frames[0, 0])
        assert all(isinstance(x, str) and x for x in obs['annotation.human.action.task_description'])
        actions = {key: np.zeros_like(value) for key, value in env.action_space.sample().items()}
        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step(actions)
            assert 'success' in info
            assert np.isfinite(reward).all()
        obs, _ = env.reset()
        report = dict(status='passed', task=args.env_name, n_envs=args.n_envs,
                      simulator_steps_per_env=8, observation_shapes=shapes,
                      action_shapes={k: list(v.shape) for k, v in actions.items()},
                      versions={p: importlib.metadata.version(p) for p in
                                ('torch', 'numpy', 'gymnasium', 'mujoco', 'robosuite', 'robocasa')})
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
