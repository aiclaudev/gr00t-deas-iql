#!/usr/bin/env python3
"""Prepare one joint-SVF worker command. Default is a read-only dry run.

Nothing is submitted without --submit and an explicit --time. No monitoring,
background process, automatic dependency or automatic follow-up is installed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python')
SCALAR_OPTIONS = ('seed', 'steps', 'global_batch_size', 'microbatch_size', 'actor_lr',
                  'value_lr', 'weight_decay', 'max_grad_norm', 'kappa', 'g', 'candidates',
                  't_min', 'flow_steps', 'guidance_clip', 'lambda_epsilon', 'save_steps',
                  'log_steps', 'num_workers', 'wandb_project', 'wandb_entity',
                  'actor_tuning', 'lora_rank', 'lora_alpha', 'lora_dropout', 'soft_value_init',
                  'soft_value_loss', 'lr_scheduler', 'warmup_steps', 'min_lr_ratio')
OPTION_DEFAULTS = {'actor_tuning': 'full', 'lora_rank': 16, 'lora_alpha': 16.0,
                   'lora_dropout': 0.0, 'soft_value_init': 'random', 'soft_value_loss': 'mse',
                   'lr_scheduler': 'constant', 'warmup_steps': 0, 'min_lr_ratio': 0.0}


def validate_stop_options(*, preflight_steps=None, stop_after_steps=None):
    if preflight_steps is not None and stop_after_steps is not None:
        raise ValueError('preflight_steps and stop_after_steps are mutually exclusive')
    if preflight_steps is not None and (
            not isinstance(preflight_steps, int) or isinstance(preflight_steps, bool)
            or not 1 <= preflight_steps <= 100):
        raise ValueError('preflight_steps must be an integer between 1 and 100')
    if stop_after_steps is not None and (
            not isinstance(stop_after_steps, int) or isinstance(stop_after_steps, bool)
            or stop_after_steps <= 0):
        raise ValueError('stop_after_steps must be a positive integer of additional updates')


def validate_config(cfg):
    tuning = {**OPTION_DEFAULTS, **cfg}
    steps = cfg['steps']
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError('steps must be a positive integer')
    if tuning['lr_scheduler'] not in ('constant', 'cosine'):
        raise ValueError('lr_scheduler must be constant or cosine (actor only)')
    warmup = tuning['warmup_steps']
    if (not isinstance(warmup, int) or isinstance(warmup, bool)
            or not 0 <= warmup < steps):
        raise ValueError('warmup_steps must be an integer in [0, steps) (actor only)')
    minimum = tuning['min_lr_ratio']
    if (not isinstance(minimum, (int, float)) or isinstance(minimum, bool)
            or not math.isfinite(minimum) or not 0 <= minimum <= 1):
        raise ValueError('min_lr_ratio must be finite and in [0, 1] (actor only)')
    if tuning['lr_scheduler'] == 'constant' and (warmup or minimum):
        raise ValueError('warmup_steps and min_lr_ratio require the cosine actor scheduler')
    validate_stop_options(stop_after_steps=cfg.get('stop_after_steps'))
    if tuning['soft_value_init'] not in ('random', 'critic-trunk', 'critic-full'):
        raise ValueError('soft_value_init must be random, critic-trunk or critic-full')
    if tuning['soft_value_loss'] not in ('mse', 'hl-gauss'):
        raise ValueError('soft_value_loss must be mse or hl-gauss')
    if tuning['soft_value_init'] == 'critic-full' and tuning['soft_value_loss'] != 'hl-gauss':
        raise ValueError('critic-full initialization requires soft_value_loss hl-gauss')
    if tuning['actor_tuning'] not in ('full', 'dit-lora'):
        raise ValueError('actor_tuning must be full or dit-lora')
    rank = tuning['lora_rank']
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError('lora_rank must be a positive integer')
    if not math.isfinite(tuning['lora_alpha']) or tuning['lora_alpha'] <= 0:
        raise ValueError('lora_alpha must be finite and positive')
    if not math.isfinite(tuning['lora_dropout']) or not 0 <= tuning['lora_dropout'] < 1:
        raise ValueError('lora_dropout must be finite and in [0, 1)')
    if (cfg['gpus'], cfg['cpus_per_gpu'], cfg['memory_gib'], cfg['account'], cfg['qos']) != (4, 8, 768, 'sub', 'own'):
        raise ValueError('This prepared launch uses four GPUs, 32 CPUs, 768 GiB and sub/own')
    if cfg['global_batch_size'] % (cfg['gpus'] * cfg['microbatch_size']):
        raise ValueError('Global batch must be divisible by GPUs times microbatch size')
    for value in (cfg['actor'], cfg['critic'], *cfg['dataset_paths']):
        if not Path(value).resolve().is_relative_to('/home/nas_main/dohyunlee'):
            raise ValueError(f'Input must remain in the personal workspace: {value}')
    if not re.fullmatch(r'[0-9]+', str(cfg['critic_job_id'])):
        raise ValueError('critic_job_id must be numeric')


def require_finished_teacher(cfg):
    critic = Path(cfg['critic'])
    for name in ('config.json', 'experiment_cfg/metadata.json', 'trainer_state.json'):
        if not (critic / name).is_file():
            raise ValueError(f'Final critic is not ready: missing {critic / name}')
    state = json.loads((critic / 'trainer_state.json').read_text())
    if state['global_step'] != cfg['critic_required_steps']:
        raise ValueError('Use the completed 10,000-step critic, not an intermediate checkpoint')
    if not ((critic / 'model.safetensors').is_file() or (critic / 'model.safetensors.index.json').is_file()):
        raise ValueError('Final critic model weights are missing')
    archived = subprocess.check_output(
        ['sacct', '-j', str(cfg['critic_job_id']), '-X', '-n', '-P', '-o', 'JobIDRaw,State'], text=True)
    states = [line.split('|')[1].split()[0] for line in archived.splitlines()
              if line.split('|')[0] == str(cfg['critic_job_id'])]
    if states != ['COMPLETED']:
        raise ValueError(f'Critic job must be COMPLETED before submission; found {states}')


def training_args(cfg, output, *, preflight_steps=None, stop_after_steps=None, resume=None):
    """Retain the full actor LR horizon; the soft-value LR remains constant.

    stop_after_steps limits additional updates in this invocation, including resume.
    Only bounded preflight runs disable configured experiment logging.
    """
    validate_stop_options(preflight_steps=preflight_steps, stop_after_steps=stop_after_steps)
    if preflight_steps is None and stop_after_steps is None:
        stop_after_steps = cfg.get('stop_after_steps')
    validate_stop_options(preflight_steps=preflight_steps, stop_after_steps=stop_after_steps)
    args = ['--actor', cfg['actor'], '--critic', cfg['critic'], '--output', str(output),
            '--dataset-path', *cfg['dataset_paths']]
    for key in SCALAR_OPTIONS:
        value = cfg[key] if key in cfg else OPTION_DEFAULTS[key]
        args += ['--' + key.replace('_', '-'), str(value)]
    args += ['--report-to', 'none' if preflight_steps is not None else cfg['report_to'],
             '--run-name', f"seed{cfg['seed']}-svf-joint-{output.parent.name}"]
    limit = preflight_steps if preflight_steps is not None else stop_after_steps
    if limit is not None:
        args += ['--stop-after-steps', str(limit)]
    if resume:
        args += ['--resume', str(resume)]
    return args


def build_command(cfg, root, time_limit, train_args):
    return ['sbatch', '--parsable', '--account=sub', '--qos=own', '--partition=compute',
            '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--cpus-per-gpu=8', '--mem=768G',
            '--container=nvcr.io/nvidia/pytorch:25.04-py3', '--export=NONE',
            '--time=' + time_limit, '--job-name=svf-joint', '--chdir=' + str(REPO),
            '--output=' + str(root / 'logs/svf-%j.out'), '--error=' + str(root / 'logs/svf-%j.err'),
            str(REPO / 'slurm/svf_joint.sbatch'), str(root), *train_args]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=REPO / 'configs/svf_joint_seed42.json')
    p.add_argument('--soft-value-init', choices=('random', 'critic-trunk', 'critic-full'),
                   help='Override the configuration soft-value initialization')
    p.add_argument('--soft-value-loss', choices=('mse', 'hl-gauss'),
                   help='Override the configuration soft-value loss without changing initialization')
    p.add_argument('--time', help='Measured expected walltime plus 20-30%%, HH:MM:SS')
    p.add_argument('--run-root', type=Path)
    stop = p.add_mutually_exclusive_group()
    stop.add_argument('--preflight-steps', type=int,
                      help='1..100 additional worker updates without W&B; retain full schedule for resume')
    stop.add_argument('--stop-after-steps', type=int,
                      help='Positive additional updates, preserving full schedule and configured W&B logging')
    p.add_argument('--resume', type=Path)
    p.add_argument('--submit', action='store_true', help='Actually submit; omitted means dry run')
    args = p.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if args.soft_value_init is not None:
        cfg['soft_value_init'] = args.soft_value_init
    if args.soft_value_loss is not None:
        cfg['soft_value_loss'] = args.soft_value_loss
    if args.stop_after_steps is not None:
        cfg['stop_after_steps'] = args.stop_after_steps
    validate_config(cfg)
    try:
        validate_stop_options(preflight_steps=args.preflight_steps,
                              stop_after_steps=args.stop_after_steps)
    except ValueError as exc:
        p.error(str(exc))
    if args.time and (not re.fullmatch(r'[0-9]+:[0-5][0-9]:[0-5][0-9]', args.time)
                      or not any(int(v) for v in args.time.split(':'))):
        p.error('--time must be a positive HH:MM:SS duration')
    if args.submit and not args.time:
        p.error('--submit needs an explicit measured --time')
    if args.resume and not args.run_root:
        p.error('--resume needs the existing --run-root')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = (args.run_root or REPO / 'output/svf-joint' / f'seed{cfg["seed"]}-{stamp}').resolve()
    if not root.is_relative_to(REPO / 'output/svf-joint'):
        p.error('--run-root must stay under this repository output/svf-joint')
    train_args = training_args(cfg, root / 'train', preflight_steps=args.preflight_steps,
                               resume=args.resume.resolve() if args.resume else None)
    command = build_command(cfg, root, args.time or '<WALLTIME_REQUIRED>', train_args)
    print(shlex.join(command), flush=True)
    if not args.submit:
        print('DRY RUN: no files created, no resource request and no job submitted.')
        return 0
    require_finished_teacher(cfg)
    subprocess.run([str(PYTHON), str(REPO / 'scripts/train_svf.py'), *train_args,
                    '--validate-config', '--world-size', str(cfg['gpus'])], check=True)
    snapshot = json.loads(subprocess.check_output(['snode', '--json'], text=True))
    account = snapshot['accounts']['sub']
    cap = account['per_user_own_cap']
    request = {'gpu': 4, 'cpu': 32, 'mem_mib': 768 * 1024}
    if any(request[k] > cap[k] for k in request):
        raise ValueError('Requested resources exceed the current per-user own cap')
    print(json.dumps({'request': request, 'own_cap': cap, 'account_available': account['avail'],
                      'current_user': account['users'].get(getpass.getuser(), {}),
                      'cluster_available': snapshot['cluster']['avail']}, indent=2))
    subprocess.run(['squeue', '--user=' + getpass.getuser(),
                    '--format=%.18i %.25j %.12q %.12T %.40R'], check=True)
    if not args.resume and root.exists():
        raise FileExistsError(f'Use a new run directory: {root}')
    if args.resume and not (args.resume / 'complete.json').is_file():
        raise ValueError('Resume checkpoint is incomplete')
    root.mkdir(parents=True, exist_ok=bool(args.resume))
    (root / 'logs').mkdir(exist_ok=True)
    submission = {'configuration': cfg, 'command': command, 'resources': snapshot}
    job_id = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not job_id.isdecimal():
        raise RuntimeError(f'Unrecognized sbatch response: {job_id}')
    submission['job_id'] = job_id
    (root / f'submission-{job_id}.json').write_text(json.dumps(submission, indent=2) + '\n')
    print(f'Submitted {job_id}; output: {root / "train"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
