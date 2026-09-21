#!/usr/bin/env python3
"""Verify and publish only the explicitly selected BC comparison artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

TASKS = ['CoffeeSetupMug', 'PnPCounterToMicrowave', 'PnPMicrowaveToCounter', 'TurnOffStove']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--repo', default='RLobot-jun/robocasa-bc2-30k-replay-2ep')
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.upload:
        import numpy as np
        report = json.loads((root / 'reference_replay.json').read_text())
        assert report['passed'], 'Local inference replay did not match'
        counts = {}
        for task in TASKS:
            info = json.loads((root / 'dataset' / task / 'meta/info.json').read_text())
            assert info['total_episodes'] == 2, (task, info['total_episodes'])
            traces = sorted((root / 'eval' / task / 'inference').glob('call-*.npz'))
            assert traces, task
            for trace in traces:
                with np.load(trace, allow_pickle=False) as arrays:
                    assert 'initial_noise' in arrays
                    assert any(k.startswith('input::video.') for k in arrays.files)
                    assert any(k.startswith('output::action.') for k in arrays.files)
            counts[task] = {'episodes': 2, 'inference_calls': len(traces)}
        versions = {}
        for name in ['torch', 'transformers', 'numpy', 'mujoco', 'robosuite', 'robocasa',
                     'safetensors', 'torchvision', 'flash-attn', 'timm']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest = {'actor': 'RLobot-jun/gr00t-n1.5-robocasa-bc2-30k',
                    'actor_revision': '8c057ce617b4dd87544f5fc36acb2dbb30b8ceaf',
                    'seed': 42, 'n_envs': 1, 'action_horizon': 16,
                    'execute_horizon': 16, 'denoising_steps': 4,
                    'tasks': counts, 'versions': versions,
                    'purpose': 'Same-observation BC action comparison; raw NPZ is lossless, MP4 is not'}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        files = {}
        for folder in ['eval', 'dataset', 'scripts', 'actor_metadata']:
            for path in sorted((root / folder).rglob('*')):
                if path.is_file() and path.suffix in {'.npz', '.json', '.jsonl', '.mp4', '.parquet', '.py'}:
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                    files[str(path.relative_to(root))] = {'bytes': path.stat().st_size,
                                                         'sha256': digest.hexdigest()}
        (root / 'files.json').write_text(json.dumps(files, indent=2) + '\n')
        print('PACKAGE_VALIDATED', json.dumps(counts), flush=True)
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))
    user = api.whoami()['name']
    assert args.repo.startswith(user + '/'), (user, args.repo)
    files = json.loads((root / 'files.json').read_text())
    report = json.loads((root / 'reference_replay.json').read_text())
    assert report['passed']
    allow = list(files) + ['README.md', 'manifest.json', 'files.json', 'reference_replay.json']
    api.create_repo(args.repo, repo_type='dataset', private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type='dataset', folder_path=str(root),
                      allow_patterns=allow, commit_message='BC2-30k RoboCasa lossless inference replay and LeRobot episodes')
    info = api.dataset_info(args.repo, files_metadata=True)
    actual = {item.rfilename: item.size for item in info.siblings}
    assert not info.private
    for name, value in files.items():
        assert actual.get(name) == value['bytes'], name
    receipt = {'url': 'https://huggingface.co/datasets/' + args.repo,
               'revision': info.sha, 'verified_files': len(files)}
    (root / 'uploaded.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('UPLOAD_VERIFIED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
