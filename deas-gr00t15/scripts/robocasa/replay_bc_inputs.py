#!/usr/bin/env python3
"""Compare BC actions on lossless recorded observations, without a simulator."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--actor', required=True, help='Downloaded BC2 checkpoint directory')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--max-calls-per-task', type=int, default=3, help='0 selects every call')
    parser.add_argument('--atol', type=float, default=1e-5)
    args = parser.parse_args()
    if args.max_calls_per_task < 0:
        parser.error('--max-calls-per-task must be nonnegative')
    try:
        from inference_trace import load_trace, prefixed, restore_rng
    except ModuleNotFoundError:
        from gr00t.eval.inference_trace import load_trace, prefixed, restore_rng
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    groups = sorted((args.dataset / 'eval').glob('*/inference'))
    if not groups:
        raise ValueError('Expected dataset/eval/<task>/inference/call-*.npz')
    first = sorted(groups[0].glob('call-*.npz'))[0]
    meta, _ = load_trace(first)
    cfg = meta['config']
    data = DATA_CONFIG_MAP[cfg['data_config']](AS=cfg['action_horizon'])
    policy = Gr00tPolicy(args.actor, cfg['embodiment_tag'], data.modality_config(),
                         data.transform(), cfg['denoising_steps'], device='cuda:0')
    rows = []
    for group in groups:
        paths = sorted(group.glob('call-*.npz'))
        if args.max_calls_per_task and len(paths) > args.max_calls_per_task:
            indices = np.linspace(0, len(paths) - 1, args.max_calls_per_task, dtype=int)
            paths = [paths[i] for i in indices]
        for path in paths:
            metadata, arrays = load_trace(path)
            if 'initial_noise' not in arrays:
                raise ValueError(f'{path}: missing explicit initial noise')
            restore_rng(metadata['rng'], arrays)
            noise = torch.from_numpy(arrays['initial_noise'])
            original_randn = torch.randn
            calls = []

            def recorded_randn(*shape, **kwargs):
                requested = kwargs.get('size', shape)
                if len(requested) == 1 and isinstance(requested[0], (tuple, list, torch.Size)):
                    requested = requested[0]
                if tuple(requested) == tuple(noise.shape):
                    calls.append(True)
                    return noise.to(device=kwargs.get('device', 'cpu'),
                                    dtype=kwargs.get('dtype', torch.float32)).clone()
                return original_randn(*shape, **kwargs)

            # Works with the original N1.5 head as well: replace only its one
            # initial action-noise draw, not the model's numerical operations.
            with patch('torch.randn', recorded_randn):
                actual = policy.get_action(prefixed(arrays, 'input::'))
            if len(calls) != 1:
                raise ValueError(f'Expected one initial noise draw, found {len(calls)}')
            expected = prefixed(arrays, 'output::')
            if actual.keys() != expected.keys():
                raise ValueError('Output action keys differ from reference')
            errors = {}
            for key in expected:
                a, b = np.asarray(actual[key]), expected[key]
                if a.shape != b.shape:
                    raise ValueError(f'{key}: shape mismatch {a.shape} vs {b.shape}')
                errors[key] = {'max_abs': float(np.max(np.abs(a - b))),
                               'mean_abs': float(np.mean(np.abs(a - b))),
                               'bitwise_exact': bool(np.array_equal(a, b)),
                               'close': bool(np.allclose(a, b, atol=args.atol, rtol=0))}
            row = {'task': group.parent.name, 'call': path.name, 'errors': errors,
                   'close': all(e['close'] for e in errors.values())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    metadata_path = Path(args.actor) / 'experiment_cfg/metadata.json'
    report = {'passed': all(r['close'] for r in rows), 'calls': len(rows), 'atol': args.atol,
              'rtol': 0, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'actor': str(args.actor),
              'normalization_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              'results': rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('BC_REPLAY_RESULT', json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
