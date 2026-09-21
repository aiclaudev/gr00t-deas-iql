#!/usr/bin/env python3
"""Rescore a bounded, outcome-stratified dataset sample with a saved QC/IQL critic.

No simulator or learning updates. Reports sample maxima, not maxima of batch means.
Use --source-root to import the immutable training source snapshot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--points', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-episodes-inspected', type=int, default=128)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Dataset rescoring must run in sbatch')
    assert args.points >= 2 and args.batch_size > 0
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file
    from gr00t.data.iql_dataset import EpisodeDataset, eligible_starts
    from gr00t.model.iql.core import chunk_fields
    from gr00t.model.iql.model import ChunkIQLCritic
    from gr00t.model.transforms import DefaultDataCollator

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    critic_root = args.checkpoint.parent
    cfg = json.loads((args.checkpoint / 'config.json').read_text())
    assert cfg['algorithm'] == 'scalar IQL + QC complete chunks'
    normalization = (critic_root / 'normalization_metadata.json').read_bytes()
    metadata = json.loads(normalization)['new_embodiment']
    horizon, discount = cfg['horizon'], cfg['discount']
    print('LOAD_MODEL', str(args.checkpoint), flush=True)
    model = ChunkIQLCritic(cfg['actor'], horizon=horizon)
    payload = torch.load(args.checkpoint / 'training.pt', map_location='cpu',
                         weights_only=False, mmap=True)
    step = payload['step']
    model.head.load_state_dict(payload['head'], strict=True)
    del payload
    frozen = load_file(str(critic_root / 'frozen_features.safetensors'))
    missing, unexpected = model.load_state_dict(frozen, strict=False)
    assert not unexpected and all(k.startswith('head.') for k in missing), (missing, unexpected)
    del frozen
    model.to('cuda:0').eval().requires_grad_(False)
    collator = DefaultDataCollator()
    rows, selection, coverage = [], [], []
    output_file = args.output / 'scores.jsonl'
    with output_file.open('w') as stream, torch.inference_mode():
        for path_string in cfg['dataset_path']:
            path = Path(path_string)
            ds = EpisodeDataset(path, metadata, horizon)
            ds.transforms.eval()
            source, task = path.parent.name, path.name
            quotas = {'success': 2} if source == 'demos' else {'success': 3, 'failure': 3}
            counts = {k: 0 for k in quotas}
            inspected = 0
            for episode_id in rng.permutation(ds.trajectory_ids)[:args.max_episodes_inspected]:
                if counts == quotas:
                    break
                inspected += 1
                episode_id = int(episode_id)
                parquet = ds.dataset_path / ds.data_path_pattern.format(
                    episode_chunk=ds.get_episode_chunk(episode_id), episode_index=episode_id)
                frame = pd.read_parquet(parquet)
                rewards = frame['next.reward'].to_numpy(dtype=np.float32)
                outcome = 'success' if rewards.sum() > 0 else 'failure'
                if counts.get(outcome, 0) >= quotas.get(outcome, 0):
                    continue
                terminated = frame['next.terminated'].to_numpy(bool) if 'next.terminated' in frame else rewards > 0
                boundaries = frame['next.done'].to_numpy(bool).copy() | terminated
                boundaries[-1] = True
                valid = eligible_starts(rewards, boundaries, terminated, horizon)
                if not len(valid):
                    continue
                ep = ds.load_episode(episode_id)
                assert ep is not None
                ds.activate(ep)
                # Include first/last valid starts, plus any successful terminal chunk.
                starts = valid[np.linspace(0, len(valid)-1, min(args.points, len(valid))).round().astype(int)]
                terminal_starts = valid[terminated[valid+horizon-1]]
                starts = np.unique(np.r_[starts, terminal_starts]).astype(int)
                selection.append(dict(task=task, source=source, episode=episode_id,
                                      outcome=outcome, starts=starts.tolist()))
                for offset in range(0, len(starts), args.batch_size):
                    group = starts[offset:offset+args.batch_size]
                    samples = [ds.transforms(ds.get_step_data(episode_id, int(s))) for s in group]
                    batch = collator(samples)
                    batch = {k: v.to('cuda:0') if torch.is_tensor(v) else v for k, v in batch.items()}
                    features = model.project(model.encode(batch), batch['embodiment_id'])
                    states = batch['state'].float() * batch['state_mask'].float()
                    actions = batch['action'].float() * batch['action_mask'].float()
                    q1, q2 = model.head.q(features, states, actions)
                    value = model.head.v(features, states)
                    numbers = dict(q1=q1, q2=q2, q_min=torch.minimum(q1, q2), v=value,
                                   projection_saturation=(features.abs() >= .999).float().flatten(1).mean(1))
                    numbers = {k: v.flatten().cpu().numpy() for k, v in numbers.items()}
                    assert all(len(v) == len(group) and np.isfinite(v).all() for v in numbers.values())
                    for i, start in enumerate(group):
                        start = int(start)
                        fields = chunk_fields(ep['rewards'], ep['boundaries'], ep['terminated'], start, horizon, discount)
                        assert fields['chunk_valid'] == 1
                        row = dict(task=task, source=source, episode=episode_id, outcome=outcome,
                                   start=start, length=len(rewards), progress=start/max(1,len(rewards)-1),
                                   terminal=not bool(fields['bootstrap_mask']), chunk_return=fields['chunk_return'],
                                   **{k: float(v[i]) for k, v in numbers.items()})
                        rows.append(row)
                        stream.write(json.dumps(row) + '\n')
                    stream.flush()
                counts[outcome] += 1
                print('SCORED', task, source, episode_id, outcome, len(starts), 'TOTAL', len(rows), flush=True)
                ds.curr_traj_data = None
                ds.curr_traj_id = None
                ds.episode_videos = {}
                del ep
            coverage.append(dict(task=task, source=source, requested=quotas, actual=counts, inspected=inspected))
            del ds

    def stats(subset):
        result = {'n': len(subset)}
        for key in ('q1', 'q2', 'q_min', 'v', 'projection_saturation'):
            values = np.array([r[key] for r in subset], dtype=float)
            if len(values):
                result[key] = dict(min=float(values.min()), mean=float(values.mean()), max=float(values.max()),
                                   std=float(values.std()), p05=float(np.quantile(values,.05)), p95=float(np.quantile(values,.95)))
        return result

    grouped = {'all': stats(rows)}
    for key in ('task', 'outcome', 'source', 'terminal'):
        for val in sorted({r[key] for r in rows}):
            grouped[f'{key}={val}'] = stats([r for r in rows if r[key] == val])
    for task in sorted({r['task'] for r in rows}):
        for outcome in ('success', 'failure'):
            grouped[f'{task}/{outcome}'] = stats([r for r in rows if r['task'] == task and r['outcome'] == outcome])
    summary = dict(checkpoint=str(args.checkpoint), step=step, training_batch=cfg['batch_size'], seed=args.seed,
                   source_root=str(args.source_root), precision=cfg['precision'], normalization_sha256=hashlib.sha256(normalization).hexdigest(),
                   sampling='Outcome-stratified known training trajectories; evenly spaced complete chunks including terminal starts; eval transforms; dataset actions, no simulator.',
                   scope='These are maxima over the sampled states, not the entire dataset. This is not held-out generalization or policy success rate.',
                   selection=selection, coverage=coverage, statistics=grouped)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('RESCORE_COMPLETE', json.dumps(grouped['all']), flush=True)


if __name__ == '__main__':
    main()
