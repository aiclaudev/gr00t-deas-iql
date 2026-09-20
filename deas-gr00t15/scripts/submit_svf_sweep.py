#!/usr/bin/env python3
"""Plan sequential SVF train -> four-task eval -> summary stages.

Default: read-only preview, no cluster calls. --submit requires explicit training
and evaluation walltimes and an evaluation episode count. Every independent
training run and task evaluation is a separate sbatch; no live worker control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_module('svf_single_submission', REPO / 'scripts/submit_svf.py')
evaluation = load_module('svf_eval_submission', REPO / 'scripts/robocasa/submit_evaluations.py')


def duration(value, label):
    if value is not None and (not isinstance(value, str) or
            not re.fullmatch(r'[0-9]+:[0-5][0-9]:[0-5][0-9]', value) or
            not any(int(x) for x in value.split(':'))):
        raise ValueError(f'{label} must be a positive HH:MM:SS duration')
    return value


def integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def grid_values(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{label} must be a nonempty list')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(f'{label} must contain finite positive numbers')
    if len(set(values)) != len(values):
        raise ValueError(f'{label} must not contain duplicates')
    return values


def number_key(value):
    return str(value).replace('.', 'p')


def job_record(key, kind, command, needs=(), dependency_kind='afterok'):
    return dict(key=key, kind=kind, command=command, depends_on=list(needs),
                dependency_kind=dependency_kind, job_id=None, submission_state='planned')


def build_plan(config, root, *, train_time=None, eval_time=None, eval_episodes=None):
    root = Path(root).expanduser().resolve()
    if not root.is_relative_to(REPO / 'output/svf-joint') or root == REPO / 'output/svf-joint':
        raise ValueError('run-root must be below this repository output/svf-joint')
    config_path = (REPO / config['training_config']).resolve()
    if not config_path.is_relative_to(REPO):
        raise ValueError('training_config must stay inside this repository')
    base = json.loads(config_path.read_text())
    train.validate_config(base)
    stage_steps = base.get('stop_after_steps')
    integer(stage_steps, 'stop_after_steps')
    if stage_steps > base['steps']:
        raise ValueError('stop_after_steps cannot exceed the full training horizon')
    kappas, gs = grid_values(config['kappas'], 'kappas'), grid_values(config['gs'], 'gs')
    ev = dict(config['evaluation'])
    episodes = ev.get('episodes') if eval_episodes is None else eval_episodes
    if episodes is not None:
        integer(episodes, 'eval episodes')
    train_time = duration(train_time if train_time is not None else config.get('train_time'), 'train_time')
    eval_time = duration(eval_time if eval_time is not None else ev.get('time'), 'eval time')
    tasks = ev['tasks']
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(evaluation.TASKS):
        raise ValueError('evaluation tasks must be distinct supported RoboCasa tasks')
    for name in ('n_envs', 'action_horizon', 'denoising_steps'):
        integer(ev[name], name)
    integer(ev['eval_seed'], 'eval_seed', 0)
    integer(base['seed'], 'training seed', 0)
    if type(ev['save_video']) is not bool or ev['report_to'] not in ('wandb', 'none'):
        raise ValueError('Invalid eval save_video or report_to')
    unresolved = [name for name, value in (('train_time', train_time), ('eval_time', eval_time),
                                          ('eval_episodes', episodes)) if value is None]
    arms, jobs = [], []
    previous_summary = None
    for kappa in kappas:
        for g in gs:
            index = len(arms) + 1
            key = f'{index:02d}-k{number_key(kappa)}-g{number_key(g)}'
            arm_root = root / key
            cfg = {**base, 'kappa': kappa, 'g': g}
            train.validate_config(cfg)
            train_args = train.training_args(cfg, arm_root / 'train')
            command = train.build_command(cfg, arm_root, train_time or '<TRAIN_TIME_REQUIRED>', train_args)
            command.insert(1, '--kill-on-invalid-dep=yes')
            command[command.index('--job-name=svf-joint')] = '--job-name=svf-' + key
            training = job_record(key + ':train', 'train', command,
                                  [previous_summary] if previous_summary else [])
            jobs.append(training)
            checkpoint = arm_root / 'train' / f'checkpoint-{stage_steps}'
            args = SimpleNamespace(output_root=arm_root / 'eval', training_summary=None,
                actor=checkpoint / 'actor', seed=base['seed'], eval_seed=ev['eval_seed'],
                tasks=tasks, episodes=episodes, n_envs=ev['n_envs'], time=eval_time or '<EVAL_TIME_REQUIRED>',
                report_to=ev['report_to'], save_video=ev['save_video'], action_horizon=ev['action_horizon'],
                denoising_steps=ev['denoising_steps'], qos='own')
            eval_plan = evaluation.build_plan(args, allow_future_actor=True)
            eval_plan['source'].update(svf_checkpoint=str(checkpoint), expected_svf_step=stage_steps,
                                       kappa=kappa, g=g)
            eval_plan['notes'] = ['Actor is a future SVF checkpoint, checked on the worker after training succeeds.',
                                  'Failed or missing evaluations make aggregation fail and block later training.']
            for record in eval_plan['jobs']:
                record['key'] = key + ':' + record['key']
                record.update(kind='eval', depends_on=[training['key']], dependency_kind='afterok')
                cmd = record['command']
                position = cmd.index(str(REPO / 'slurm/robocasa_pipeline_eval.sbatch'))
                forwarded = cmd[position + 1:]
                # The wrapper checks the SVF complete marker; the inner BC worker
                # must not expect a HuggingFace Trainer trainer_state.json.
                forwarded[8] = key + '-' + forwarded[8]
                cmd[position:] = [str(REPO / 'slurm/svf_robocasa_eval.sbatch'), str(checkpoint),
                                  str(stage_steps), *forwarded]
                jobs.append(record)
            _, summary_command = evaluation.aggregation_command({**eval_plan,
                'jobs': [{**j, 'job_id': f'<{j["key"]}>'} for j in eval_plan['jobs']]})
            summary_command = [v for v in summary_command if not v.startswith('--dependency=')]
            summary = job_record(key + ':summary', 'summary', summary_command,
                                 [j['key'] for j in eval_plan['jobs']], 'afterany')
            eval_plan['aggregator'] = summary
            jobs.append(summary)
            arms.append(dict(key=key, kappa=kappa, g=g, c=kappa ** 2 / g,
                             root=str(arm_root), training_config=cfg, evaluation=eval_plan))
            previous_summary = summary['key']
    keys = [j['key'] for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate job identities')
    return dict(schema_version=1, status='planned', output_root=str(root),
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                unresolved=unresolved, stage_steps=stage_steps, total_steps=base['steps'],
                train_time=train_time, eval_time=eval_time, eval_episodes=episodes,
                arms=arms, jobs=jobs)


def resolved_command(job, accepted):
    ids = []
    for key in job['depends_on']:
        jid = accepted.get(key)
        if jid is None or not re.fullmatch(r'[1-9][0-9]*', str(jid)):
            raise ValueError(f'Dependency is not accepted: {key}')
        ids.append(str(jid))
    command = [v for v in job['command'] if not v.startswith('--dependency=')]
    dependency = job['dependency_kind'] + ':' + ':'.join(ids) if ids else ''
    if dependency:
        command.insert(1, '--dependency=' + dependency)
    return command, dependency


def submit_plan(plan, run=subprocess.run):
    if plan['unresolved']:
        raise ValueError('Set these before submission: ' + ', '.join(plan['unresolved']))
    root = Path(plan['output_root'])
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'Use a fresh run-root; inspect the existing manifest first: {root}')
    first = plan['arms'][0]
    train.require_finished_teacher(first['training_config'])
    train_args = train.training_args(first['training_config'], Path(first['root']) / 'train')
    run([str(train.PYTHON), str(REPO / 'scripts/train_svf.py'), *train_args,
         '--validate-config', '--world-size', str(first['training_config']['gpus'])], check=True)
    response = run(['snode', '--json'], check=True, capture_output=True, text=True, timeout=60)
    capacity = json.loads(response.stdout)
    account = capacity['accounts']['sub']
    cfg = first['training_config']
    # Phases never overlap. All task evaluations may run together; aggregation
    # starts only once they terminate and must succeed before the next training.
    request = {'gpu': max(cfg['gpus'], len(first['evaluation']['jobs'])),
               'cpu': max(cfg['gpus'] * cfg['cpus_per_gpu'], 8 * len(first['evaluation']['jobs'])),
               'mem_mib': max(cfg['memory_gib'] * 1024, 96 * 1024 * len(first['evaluation']['jobs']))}
    if any(request[k] > account['per_user_own_cap'][k] for k in request):
        raise ValueError('Phase resources exceed the current sub own cap')
    plan['capacity_check'] = dict(peak_request=request, own_cap=account['per_user_own_cap'],
        current_user=account['users'].get(getpass.getuser(), {}), account_available=account['avail'],
        cluster_available=capacity['cluster']['avail'])
    root.mkdir(parents=True, exist_ok=False)
    for arm in plan['arms']:
        arm_root = Path(arm['root'])
        (arm_root / 'logs').mkdir(parents=True)
        (arm_root / 'training_config.json').write_text(json.dumps(arm['training_config'], indent=2) + '\n')
        (Path(arm['evaluation']['output_root']) / 'logs').mkdir(parents=True)
        evaluation.save_manifest(arm['evaluation'])
    plan['status'] = 'submitting'
    evaluation.save_manifest(plan)
    accepted = {}
    owners = {j['key']: arm['evaluation'] for arm in plan['arms']
              for j in [*arm['evaluation']['jobs'], arm['evaluation']['aggregator']]}
    try:
        for job in plan['jobs']:
            job['command'], job['dependency'] = resolved_command(job, accepted)
            evaluation.submit_one(plan, job, run)
            accepted[job['key']] = job['job_id']
            if job['key'] in owners:
                evaluation.save_manifest(owners[job['key']])
        plan['status'] = 'submitted'
        for arm in plan['arms']:
            arm['evaluation']['status'] = 'submitted'
            evaluation.save_manifest(arm['evaluation'])
        evaluation.save_manifest(plan)
    except BaseException as exc:
        plan['status'] = 'partial_failure'
        plan['error'] = f'{type(exc).__name__}: {exc}'
        evaluation.save_manifest(plan)
        raise
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/svf_sweep_seed42.json')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--train-time')
    parser.add_argument('--eval-time')
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--json', action='store_true', help='Print the full plan with commands and dependencies')
    parser.add_argument('--submit', action='store_true', help='Explicitly submit the whole chain; default only previews it')
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = args.run_root or REPO / 'output/svf-joint/sweeps' / stamp
    try:
        plan = build_plan(json.loads(args.config.read_text()), root, train_time=args.train_time,
                          eval_time=args.eval_time, eval_episodes=args.eval_episodes)
        if args.submit:
            submit_plan(plan)
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print(f"{len(plan['arms'])} arms / {len(plan['jobs'])} jobs; root={plan['output_root']}")
            print(f"Each: train {plan['stage_steps']} updates (LR horizon {plan['total_steps']}) -> "
                  f"{len(plan['arms'][0]['evaluation']['jobs'])} task evals -> summary -> next train")
            for arm in plan['arms']:
                print(f"  {arm['key']}: kappa={arm['kappa']}, g={arm['g']}, c={arm['c']:.6g}")
            if plan['unresolved']:
                print('Unset: ' + ', '.join(plan['unresolved']))
            print('Submitted; see manifest.json.' if args.submit else
                  'DRY RUN: no files created, no cluster calls, no jobs submitted.')
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'SVF sweep preparation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
