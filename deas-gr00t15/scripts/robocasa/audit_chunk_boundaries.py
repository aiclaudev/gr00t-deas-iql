"""CPU-only audit of frozen DEAS boundary handling; no models or video decoding.

Reads one episode per dataset from a training manifest, executes the snapshot's
actual reward/done loader and target arithmetic, and saves the evidence as JSON.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch


def method(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    snapshot = Path(manifest['source_snapshot'])
    loader_path = snapshot / 'gr00t/data/dataset.py'
    critic_path = snapshot / 'gr00t/model/action_head/deas_critic.py'
    loader_nodes = [method(loader_path, 'LeRobotSingleDataset', name)
                    for name in ('retrieve_data_and_pad', 'get_reward_or_done')]
    namespace = {'np': np}
    exec(compile(ast.Module(body=loader_nodes, type_ignores=[]), str(loader_path), 'exec'), namespace)
    Loader = type('SnapshotBoundaryLoader', (), {n.name: namespace[n.name] for n in loader_nodes})

    # Execute the original arithmetic, without constructing a critic or VLM.
    loss = method(critic_path, 'DEASCritic', 'compute_critic_loss')
    first = next(i for i, n in enumerate(loss.body)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == 'done')
    last = next(i for i, n in enumerate(loss.body)
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == 'scaled_rewards')
    target = next(n for n in ast.walk(loss)
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'target_v')
    arithmetic = compile(ast.Module(body=loss.body[first:last + 1] + [target], type_ignores=[]),
                         str(critic_path), 'exec')
    config = json.loads((Path(manifest['run_root']) / '03-critic/config.json').read_text())['critic_cfg']['rl_config']
    horizon = config['critic_action_horizon']
    critic_stub = SimpleNamespace(rl_config=SimpleNamespace(**config), critic_action_horizon=horizon)

    report = {
        'manifest': str(args.manifest.resolve()), 'snapshot': str(snapshot),
        'scope': 'First episode from each of eight datasets; reward/done columns only. Not a dataset-wide estimate.',
        'source_sha256': {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (loader_path, critic_path)},
        'rl_config': config, 'episodes': [],
        'qc_reference': {
            'agent': 'https://github.com/ColinQiyangLi/qc/blob/main/agents/acfql.py',
            'dataset': 'https://github.com/ColinQiyangLi/qc/blob/main/utils/datasets.py',
            'note': 'Official QC uses cumulative bootstrap masks and valid[-1] on critic loss; its FQL learner is not IQL.'},
    }
    for relative_path, digest in report['source_sha256'].items():
        assert manifest['source_hashes'][relative_path] == digest, relative_path
    for dataset_name in manifest['datasets']:
        dataset = Path(dataset_name)
        info = json.loads((dataset / 'meta/info.json').read_text())
        with (dataset / 'meta/episodes.jsonl').open() as handle:
            episode = json.loads(next(handle))
        episode_id = episode['episode_index']
        data_path = dataset / info['data_path'].format(
            episode_chunk=episode_id // info['chunks_size'], episode_index=episode_id)
        frame = pq.read_table(data_path, columns=['next.reward', 'next.done'], use_threads=False).to_pandas()
        length = len(frame)
        assert length == episode['length'] and length > horizon
        done_raw = np.asarray(frame['next.done'], dtype=float)
        reward_raw = np.asarray(frame['next.reward'], dtype=float)
        assert np.isin(done_raw, [0, 1]).all()
        loader = Loader()
        loader.delta_indices = {key: np.arange(horizon) for key in ('reward.next.reward', 'done.next.done')}
        loader.get_trajectory_index = lambda _: 0
        loader.trajectory_lengths = [length]
        loader.curr_traj_data = frame
        loader.dataset_path = dataset
        loader.use_rl = True
        loader.lerobot_modality_meta = SimpleNamespace(
            reward={'next.reward': SimpleNamespace(original_key=None)},
            done={'next.done': SimpleNamespace(original_key=None)})
        entry = {
            'dataset': dataset_name, 'episode': episode_id, 'file': str(data_path), 'length': length,
            'raw_reward_nonzero_indices': np.flatnonzero(reward_raw).tolist(),
            'raw_done_indices': np.flatnonzero(done_raw).tolist(),
            'termination_semantics': 'next.done alone does not identify true terminal versus timeout',
            'cases': [],
        }
        for remaining in (horizon + 1, horizon, 8, 1):
            start = length - remaining
            reward = loader.get_reward_or_done(episode_id, 'reward', 'reward.next.reward', start)
            done = loader.get_reward_or_done(episode_id, 'done', 'done.next.done', start)
            in_episode = np.arange(horizon) < remaining
            # These validity calculations apply IF next.done denotes an episode boundary.
            before_boundary = np.concatenate(([True], np.cumprod(1 - done[:-1]).astype(bool)))
            valid = in_episode & before_boundary
            bootstrap = float(np.prod(1 - done))
            values = {}
            for next_value in (0.0, -50.0):
                ns = dict(torch=torch, self=critic_stub,
                          action_input=SimpleNamespace(
                              reward=torch.tensor(reward[None], dtype=torch.float32),
                              done=torch.tensor(done[None], dtype=torch.float32)),
                          vs=torch.tensor([next_value]))
                exec(arithmetic, ns)
                values[str(next_value)] = float(ns['target_v'].item())
            shifted = reward - int(config['negative_reward'])
            padding_contribution = float(np.sum(shifted * (~in_episode) * config['discount1'] ** np.arange(horizon)))
            case = {
                'start': start, 'real_steps_remaining': remaining,
                'loaded_reward': reward.tolist(), 'loaded_done': done.tolist(),
                'deas_done_product': float(np.prod(done)),
                'deas_bootstrap_coefficient': float(config['discount2'] ** (config['nstep'] * horizon) * (1 - np.prod(done))),
                'deas_shifted_discounted_reward': float(ns['scaled_rewards'].item()),
                'deas_target_with_hypothetical_next_v': values,
                'padding_only_reward_contribution': padding_contribution,
                'qc_valid_if_done_is_boundary': valid.astype(int).tolist(),
                'qc_critic_loss_weight_if_done_is_boundary': int(valid[-1]),
                'bootstrap_mask_if_done_is_terminal': bootstrap,
            }
            assert np.prod(done) == 0 or np.all(done == 1)
            if remaining < horizon:
                assert case['qc_critic_loss_weight_if_done_is_boundary'] == 0
                if config['negative_reward']:
                    assert padding_contribution < 0
                else:
                    assert padding_contribution == 0
            entry['cases'].append(case)
        report['episodes'].append(entry)
        print(json.dumps({
            'dataset': '/'.join(dataset.parts[-2:]), 'length': length,
            'reward_nonzero': len(entry['raw_reward_nonzero_indices']),
            'done_indices': entry['raw_done_indices'],
            'last8_reward': entry['cases'][2]['loaded_reward'],
            'last8_deas_target_v_minus50': entry['cases'][2]['deas_target_with_hypothetical_next_v']['-50.0'],
        }), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
