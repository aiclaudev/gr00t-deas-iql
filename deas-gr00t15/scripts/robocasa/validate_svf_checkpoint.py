#!/usr/bin/env python3
"""Validate a completed SVF actor export without importing torch or the simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

USER_ROOT = Path('/home/nas_main/dohyunlee')


def validate_checkpoint(checkpoint, expected_step, actor):
    """Return canonical paths only after the atomic checkpoint marker validates."""
    if type(expected_step) is not int or expected_step <= 0:
        raise ValueError('Expected SVF step must be a positive integer')
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    actor = Path(actor).expanduser().resolve(strict=True)
    for path in (checkpoint, actor):
        if not path.is_relative_to(USER_ROOT):
            raise ValueError(f'Checkpoint and actor must stay inside {USER_ROOT}: {path}')
        if not path.is_dir():
            raise ValueError(f'Expected a checkpoint directory: {path}')
    if actor != checkpoint / 'actor':
        raise ValueError('Evaluation actor must be exactly CHECKPOINT/actor')
    if checkpoint.name.startswith('.') or checkpoint.name.endswith('.incomplete'):
        raise ValueError('Cannot evaluate an incomplete checkpoint directory')
    marker_path = checkpoint / 'complete.json'
    if not marker_path.is_file():
        raise ValueError(f'SVF checkpoint completion marker is missing: {marker_path}')
    if marker_path.resolve() != marker_path:
        raise ValueError('SVF checkpoint completion marker must not be a symlink')
    marker = json.loads(marker_path.read_text())
    if not isinstance(marker, dict) or type(marker.get('step')) is not int:
        raise ValueError('SVF completion marker must contain an integer step')
    if marker['step'] != expected_step:
        raise ValueError(f'Expected SVF step {expected_step}, got {marker["step"]}')
    return checkpoint, actor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--actor', type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint, actor = validate_checkpoint(args.checkpoint, args.expected_step, args.actor)
    print(json.dumps({'checkpoint': str(checkpoint), 'step': args.expected_step,
                      'actor': str(actor), 'status': 'complete'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
