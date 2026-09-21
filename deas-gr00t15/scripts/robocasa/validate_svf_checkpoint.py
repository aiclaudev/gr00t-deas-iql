#!/usr/bin/env python3
"""Validate a completed SVF actor export without importing torch or the simulator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Where checkpoints are allowed to live. The training cluster confines them to
# one user root; elsewhere (a published checkpoint pulled onto a workstation)
# set SVF_ALLOWED_ROOT, or pass --allowed-root. An empty value disables the
# containment check.
DEFAULT_ALLOWED_ROOT = os.environ.get('SVF_ALLOWED_ROOT', '/home/nas_main/dohyunlee')


def validate_checkpoint(checkpoint, expected_step, actor, allowed_root=None):
    """Return canonical paths only after the atomic checkpoint marker validates.

    Two layouts are accepted. A training run writes CHECKPOINT/actor alongside a
    complete.json marker, and passing --expected-step requires both. A published
    checkpoint is just the actor directory itself, with no marker; omit
    --expected-step to accept that form.
    """
    if expected_step is not None and (type(expected_step) is not int or expected_step <= 0):
        raise ValueError('Expected SVF step must be a positive integer')
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    actor = Path(actor).expanduser().resolve(strict=True)

    root = DEFAULT_ALLOWED_ROOT if allowed_root is None else allowed_root
    if root:
        root = Path(root).expanduser().resolve()
        for path in (checkpoint, actor):
            if not path.is_relative_to(root):
                raise ValueError(f'Checkpoint and actor must stay inside {root}: {path}')
    for path in (checkpoint, actor):
        if not path.is_dir():
            raise ValueError(f'Expected a checkpoint directory: {path}')

    if actor not in (checkpoint / 'actor', checkpoint):
        raise ValueError('Evaluation actor must be CHECKPOINT/actor or CHECKPOINT itself')
    if checkpoint.name.startswith('.') or checkpoint.name.endswith('.incomplete'):
        raise ValueError('Cannot evaluate an incomplete checkpoint directory')

    if expected_step is None:
        return checkpoint, actor

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
    parser.add_argument('--expected-step', type=int, default=None,
                        help='Training-run checkpoints must match complete.json; '
                             'omit for a published checkpoint that carries no marker')
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--allowed-root', default=None,
                        help=f'Confine paths to this root (default: {DEFAULT_ALLOWED_ROOT!r}, '
                             'from $SVF_ALLOWED_ROOT). Pass an empty string to disable.')
    args = parser.parse_args(argv)
    checkpoint, actor = validate_checkpoint(args.checkpoint, args.expected_step, args.actor,
                                            allowed_root=args.allowed_root)
    print(json.dumps({'checkpoint': str(checkpoint), 'step': args.expected_step,
                      'actor': str(actor), 'status': 'complete'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
