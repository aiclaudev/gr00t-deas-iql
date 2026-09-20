#!/usr/bin/env python3
"""Prepare a bounded IQL benchmark: 1x128 vs 4x32, three video backends.

Default dry-run. Each condition is a separate own job; afterany serializes them
so their data loaders do not compete against each other. No production weights
are saved or overwritten, and no joint-SVF training is launched.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python')
CHECKPOINT = REPO / 'output/deas-training/20260918T183345.797990880Z/03-critic/checkpoint-4000'
TASKS = ('CoffeeSetupMug', 'PnPMicrowaveToCounter', 'TurnOffStove', 'PnPCounterToMicrowave')
DATASETS = [f'/home/nas_main/dohyunlee/jh_ws/data/deas_robocasa/{kind}/{task}'
            for task in TASKS for kind in ('demos', 'rollouts')]
BACKENDS = ('decord', 'decord_cached', 'torchcodec')


def cases():
    return [(gpus, backend) for backend in BACKENDS for gpus in (1, 4)]


def train_arguments(root, gpus, backend):
    destination = root / f'{backend}-{gpus}gpu'
    return ['--checkpoint', str(CHECKPOINT), '--output', str(destination),
            '--dataset-path', *DATASETS, '--global-batch', '128', '--global-workers', '16',
            '--warmup-steps', '5', '--measure-steps', '20', '--seed', '42',
            '--video-backend', backend]


def command(root, gpus, backend, duration, dependency):
    destination = root / f'{backend}-{gpus}gpu'
    cmd = ['sbatch', '--parsable', '--account=sub', '--qos=own', '--partition=compute',
           '--nodes=1', '--ntasks=1', '--gres=' + f'gpu:{gpus}', '--cpus-per-task=24',
           '--mem=192G', '--container=nvcr.io/nvidia/pytorch:25.04-py3', '--export=NONE',
           '--time=' + duration, '--job-name=' + f'bench-{backend}-{gpus}g',
           '--chdir=' + str(REPO), '--output=' + str(root / 'logs' / f'{backend}-{gpus}gpu-%j.out'),
           '--error=' + str(root / 'logs' / f'{backend}-{gpus}gpu-%j.err')]
    if dependency:
        cmd += ['--dependency=afterany:' + dependency, '--kill-on-invalid-dep=yes']
    return cmd + [str(REPO / 'slurm/deas_benchmark.sbatch'), str(gpus), str(destination),
                  *train_arguments(root, gpus, backend)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--run-root', type=Path)
    p.add_argument('--after-job', help='Start only after this job releases its allocation')
    p.add_argument('--time', default='00:15:00', help='Per-condition bound; default 15 minutes')
    args = p.parse_args(argv)
    if args.after_job and not args.after_job.isdecimal():
        p.error('--after-job must be numeric')
    if not re.fullmatch(r'[0-9]+:[0-5][0-9]:[0-5][0-9]', args.time) or not any(map(int, args.time.split(':'))):
        p.error('--time must be positive HH:MM:SS')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = (args.run_root or REPO / 'output/deas-benchmark' / stamp).resolve()
    if root == REPO / 'output/deas-benchmark' or not root.is_relative_to(REPO / 'output/deas-benchmark'):
        p.error('Run root must be a new directory under output/deas-benchmark')
    if root.exists():
        raise FileExistsError(root)
    snapshot = None
    if args.submit:
        for gpus, backend in cases():
            subprocess.run([str(PYTHON), str(REPO / 'scripts/benchmark_deas_throughput.py'),
                            *train_arguments(root, gpus, backend), '--validate-config',
                            '--world-size', str(gpus)], check=True)
        snapshot = json.loads(subprocess.check_output(['snode', '--json'], text=True))
        cap = snapshot['accounts']['sub']['per_user_own_cap']
        if cap['gpu'] < 4 or cap['cpu'] < 24 or cap['mem_mib'] < 192 * 1024:
            raise ValueError('Current own quota cannot fit this prepared benchmark')
        print(json.dumps({'capacity': snapshot['cluster'], 'own_cap': cap,
                          'user': snapshot['accounts']['sub']['users'].get(getpass.getuser(), {})}, indent=2))
        subprocess.run(['squeue', '--user=' + getpass.getuser(),
                        '--format=%.18i %.25j %.12q %.12T %.40R'], check=True)
        root.mkdir(parents=True)
        (root / 'logs').mkdir()
    records = []
    dependency = args.after_job
    for gpus, backend in cases():
        cmd = command(root, gpus, backend, args.time, dependency)
        print(shlex.join(cmd), flush=True)
        if args.submit:
            jid = subprocess.check_output(cmd, text=True).strip().split(';')[0]
            if not jid.isdecimal():
                raise RuntimeError(f'Unexpected sbatch response: {jid}')
            records.append({'job_id': jid, 'gpus': gpus, 'backend': backend,
                            'output': str(root / f'{backend}-{gpus}gpu'), 'command': cmd})
            dependency = jid
            # Persist each successful submission even if a later submission fails.
            (root / 'jobs.json').write_text(json.dumps({'jobs': records, 'resources': snapshot}, indent=2) + '\n')
        else:
            dependency = f'<{backend}_{gpus}GPU_JOB_ID>'
    print(('Submitted benchmark jobs to ' if args.submit else 'DRY RUN ONLY: ') + str(root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
