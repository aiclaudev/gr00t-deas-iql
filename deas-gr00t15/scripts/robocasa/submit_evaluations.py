#!/usr/bin/env python3
"""Plan actor-only RoboCasa evaluations; submission requires explicit --submit.

Dry-run is the default and performs no writes or cluster commands. Each task and
training seed gets one worker. Existing submission directories are never resumed:
inspect the retained manifest and queue before preparing a new explicit attempt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_ROOT = Path('/home/nas_main/dohyunlee')
TASKS = ('CoffeeSetupMug', 'PnPMicrowaveToCounter', 'TurnOffStove', 'PnPCounterToMicrowave')
IMAGE = 'nvcr.io/nvidia/pytorch:25.04-py3'


def owned_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(USER_ROOT):
        raise ValueError(f'Path must stay inside {USER_ROOT}: {path}')
    return path


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--training-summary', type=Path, help='Recorded training production summary.json')
    source.add_argument('--actor', type=Path, help='Existing RoboCasa BC actor checkpoint')
    parser.add_argument('--seed', type=nonnegative, help='Training seed; required only with --actor')
    parser.add_argument('--eval-seed', type=nonnegative, help='Constant evaluation seed; otherwise each training seed')
    parser.add_argument('--output-root', type=Path, required=True, help='New persistent submission directory under the user home')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--episodes', type=positive, default=50)
    parser.add_argument('--n-envs', type=positive, default=1)
    parser.add_argument('--save-video', action='store_true', help='Save per-episode RoboCasa videos locally under each task output/videos')
    parser.add_argument('--time', default='04:00:00', help='Worker CANCEL boundary; default is an unmeasured initial cap')
    parser.add_argument('--report-to', choices=('none', 'wandb'), default='wandb')
    parser.add_argument('--qos', choices=('own', 'extra'), default='own', help='own waits for protected quota; extra is preemptible (account sub)')
    parser.add_argument('--action-horizon', type=positive, default=16)
    parser.add_argument('--denoising-steps', type=positive, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true', help='Actually create the manifest and submit jobs')
    mode.add_argument('--dry-run', action='store_true', help='Print the plan only (default)')
    args = parser.parse_args(argv)
    if args.actor is not None and args.seed is None:
        parser.error('--actor requires --seed')
    if args.training_summary is not None and args.seed is not None:
        parser.error('--seed is only valid with --actor; use --eval-seed to override evaluation seeds')
    if len(set(args.tasks)) != len(args.tasks):
        parser.error('--tasks must not contain duplicates')
    if not re.fullmatch(r'\d+:[0-5]\d:[0-5]\d', args.time) or not any(int(x) for x in args.time.split(':')):
        parser.error('--time must be a positive HH:MM:SS value')
    return args


def numeric_job_id(value):
    value = str(value)
    if not re.fullmatch(r'[1-9][0-9]*', value):
        raise ValueError(f'Invalid recorded training job ID: {value!r}')
    return value


def build_plan(args, *, allow_future_actor=False):
    root = owned_path(args.output_root)
    source = {'training_summary': None, 'actor': None}
    barrier = None
    if args.training_summary is not None:
        summary_path = owned_path(args.training_summary)
        source['training_summary'] = str(summary_path)
        summary = json.loads(summary_path.read_text())
        runs = summary.get('runs')
        if not isinstance(runs, list) or not runs:
            raise ValueError('Training summary must contain a nonempty runs list')
        records = []
        seen = set()
        for run in runs:
            seed = run['seed']
            if type(seed) is not int or seed < 0 or seed in seen:
                raise ValueError('Training seeds must be distinct nonnegative integers')
            seen.add(seed)
            actor = owned_path(run['output_dirs']['bc_rollout'])
            expected_steps = None
            training_arguments = actor.parent / 'arguments.tsv'
            if training_arguments.is_file():
                values = dict(line.split('\t', 1) for line in training_arguments.read_text().splitlines() if '\t' in line)
                if 'steps_per_stage' in values:
                    expected_steps = int(values['steps_per_stage'])
                    if expected_steps <= 0:
                        raise ValueError('Recorded training step count must be positive')
            records.append((seed, actor, numeric_job_id(run['job_ids']['bc_rollout']), expected_steps))
            barrier = numeric_job_id(run['job_ids']['critic'])
    else:
        actor = owned_path(args.actor)
        # Only internal orchestration may plan a future actor. Its worker must
        # validate the completed checkpoint after the training dependency succeeds.
        # CLI single-actor mode retains its existing-file checks.
        if not allow_future_actor:
            for filename in ('config.json', 'experiment_cfg/metadata.json'):
                if not (actor / filename).is_file():
                    raise ValueError(f'Missing actor file: {actor / filename}')
        source['actor'] = str(actor)
        records = [(args.seed, actor, None, None)]
    group = 'robocasa-actor-' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    config = dict(episodes=args.episodes, n_envs=args.n_envs, eval_seed=args.eval_seed,
                  tasks=args.tasks, time_limit=args.time, report_to=args.report_to, save_video=args.save_video,
                  action_horizon=args.action_horizon, denoising_steps=args.denoising_steps,
                  account='sub', qos=args.qos, gpus_per_job=1, cpus_per_gpu=8, mem_gib=96,
                  barrier_job_id=barrier, wandb_group=group)
    jobs = []
    for seed, actor, actor_job, expected_steps in records:
        eval_seed = seed if args.eval_seed is None else args.eval_seed
        for task in args.tasks:
            key = f'seed-{seed}-{task}-gr00tn15'
            output = root / 'results' / f'seed-{seed}' / task
            dependencies = []
            if actor_job:
                dependencies.append(f'afterok:{actor_job}')
            if barrier:
                dependencies.append(f'afterany:{barrier}')
            dependency = ','.join(dependencies)  # AND: own actor success and sweep finished.
            command = [
                'sbatch', '--parsable', '--account=sub', f'--qos={args.qos}', '--partition=compute',
                '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-gpu=8', '--mem=96G',
                f'--container={IMAGE}', '--export=NONE', f'--time={args.time}',
                '--kill-on-invalid-dep=yes', '--job-name=deas-rc-actor',
                f'--comment={group}:{key}', f'--chdir={REPO_ROOT}',
                f'--output={root / "logs" / (key + "-%j.out")}',
                f'--error={root / "logs" / (key + "-%j.err")}',
            ]
            if dependency:
                command.append(f'--dependency={dependency}')
            command += [str(REPO_ROOT / 'slurm/robocasa_pipeline_eval.sbatch'), str(actor),
                        task, str(output), str(args.episodes), str(args.n_envs), str(eval_seed),
                        str(seed), args.report_to, key, group, str(args.action_horizon),
                        str(args.denoising_steps), str(expected_steps or 0), '1' if args.save_video else '0']
            jobs.append(dict(key=key, seed=seed, training_seed=seed, eval_seed=eval_seed,
                             task=task, method='gr00tn15', model_type='gr00tn15', actor=str(actor),
                             save_video=args.save_video, video_dir=str(output / 'videos') if args.save_video else None,
                             expected_episodes=args.episodes, expected_training_steps=expected_steps, output_dir=str(output),
                             result_path=str(output / 'result.json'), job_id=None,
                             planned_dependency=dependency, dependency=dependency, command=command, submission_state='planned'))
    return dict(schema_version=1, status='planned',
                created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source=source,
                config=config, output_root=str(root), jobs=jobs,
                notes=['Dry-run creates no files and makes no cluster calls.',
                       '04:00:00 is an unmeasured initial evaluation cap; calibrate on a worker.',
                       'afterok requires this actor training to succeed; afterany only waits for the final training job to end.',
                       'A failed actor cancels its evaluations; aggregate output reports missing results, not zero successes.'],
                aggregator=dict(job_id=None, dependency=None, output_dir=str(root / 'aggregate'),
                                command=None, submission_state='planned'))


def save_manifest(plan):
    root = Path(plan['output_root'])
    temporary = root / 'manifest.json.tmp'
    with temporary.open('w') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / 'manifest.json')


def aggregation_command(plan):
    root = Path(plan['output_root'])
    dependency = 'afterany:' + ':'.join(job['job_id'] for job in plan['jobs'])
    command = ['sbatch', '--parsable', '--account=sub', '--qos=own', '--partition=compute',
               '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=2G', '--time=00:10:00',
               '--container=docker.io/library/python:3.11', '--export=NONE',
               '--job-name=deas-rc-summary', '--kill-on-invalid-dep=yes',
               f'--dependency={dependency}', f'--chdir={REPO_ROOT}',
               f'--output={root / "logs/aggregate-%j.out"}',
               f'--error={root / "logs/aggregate-%j.err"}',
               str(REPO_ROOT / 'slurm/robocasa_collect_results.sbatch'),
               str(root / 'manifest.json'), str(root / 'aggregate')]
    return dependency, command


def submit_one(plan, record, run):
    record['submission_state'] = 'submitting'
    save_manifest(plan)  # A crash from this point is ambiguous; never auto-resubmit.
    try:
        response = run(record['command'], check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        record['submission_state'] = 'unknown'
        save_manifest(plan)
        raise
    record['submission_stdout'] = response.stdout
    record['submission_stderr'] = response.stderr
    if response.returncode:
        record['submission_state'] = 'rejected'
        save_manifest(plan)
        raise RuntimeError(f'sbatch returned {response.returncode}: {response.stderr.strip()}')
    candidate = response.stdout.strip().split(';', 1)[0]
    if not re.fullmatch(r'[1-9][0-9]*', candidate):
        record['submission_state'] = 'unknown'
        save_manifest(plan)
        raise RuntimeError('Ambiguous sbatch response; inspect the saved response and queue before any retry')
    record['job_id'] = candidate
    record['submission_state'] = 'accepted'
    save_manifest(plan)


def resolve_dependencies(plan, run):
    """Drop completed dependencies so archived job IDs need not remain in slurmctld."""
    requested = set()
    for job in plan['jobs']:
        for clause in filter(None, job['planned_dependency'].split(',')):
            kind, jid = clause.split(':')
            if kind not in ('afterok', 'afterany'):
                raise ValueError(f'Unsupported dependency kind: {kind}')
            requested.add(numeric_job_id(jid))
    if not requested:
        return
    response = run(['sacct', '-X', '--noheader', '--parsable2', '--jobs', ','.join(sorted(requested, key=int)),
                    '--format=JobIDRaw,State%40,ExitCode'],
                   check=True, capture_output=True, text=True, timeout=60)
    states = {}
    for line in response.stdout.splitlines():
        fields = line.strip().split('|')
        if not line.strip() or fields[0] not in requested:
            continue
        if len(fields) < 3 or fields[0] in states:
            raise ValueError('Ambiguous accounting record; refusing to submit evaluation')
        states[fields[0]] = {'state': fields[1].split()[0].rstrip('+') if fields[1].strip() else '',
                            'exit_code': fields[2].strip()}
    missing = requested - states.keys()
    if missing:
        raise ValueError('Training job state is unknown for ' + ','.join(sorted(missing)) +
                         '; cannot infer success from checkpoint files alone')
    terminal = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    live = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING',
            'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'RESV_DEL_HOLD', 'SPECIAL_EXIT',
            'STOPPED', 'STAGE_OUT'}
    for jid, record in states.items():
        if record['state'] not in terminal | live:
            raise ValueError(f'Unknown training state for {jid}: {record["state"]!r}')
    plan['dependency_accounting'] = states
    for job in plan['jobs']:
        unresolved = []
        for clause in filter(None, job['planned_dependency'].split(',')):
            kind, jid = clause.split(':')
            record = states[jid]
            if record['state'] in live:
                unresolved.append(clause)
            elif kind == 'afterok' and not (record['state'] == 'COMPLETED' and record['exit_code'] == '0:0'):
                raise ValueError(f'Actor training {jid} did not complete successfully: {record}')
        dependency = ','.join(unresolved)
        command = [argument for argument in job['command'] if not argument.startswith('--dependency=')]
        if dependency:
            script_index = command.index(str(REPO_ROOT / 'slurm/robocasa_pipeline_eval.sbatch'))
            command.insert(script_index, f'--dependency={dependency}')
        job.update(dependency=dependency, command=command)


def submit_plan(plan, run=subprocess.run):
    root = Path(plan['output_root'])
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'{root} already exists; refusing duplicate submission. Inspect its manifest and queue.')
    resolve_dependencies(plan, run)
    # Live checks run only after explicit --submit, before any worker is submitted.
    capacity_response = run(['snode', '--json'], check=True, capture_output=True, text=True, timeout=60)
    capacity = json.loads(capacity_response.stdout)
    account = capacity['accounts']['sub']
    cap = account['per_user_own_cap']
    request = {'gpu': 1, 'cpu': 8, 'mem_mib': 96 * 1024}
    if plan['config']['qos'] == 'own' and any(request[key] > cap[key] for key in request):
        raise ValueError(f'Per-job request exceeds sub own quota: {request}; cap={cap}')
    username = getpass.getuser()
    queue = run(['squeue', f'--user={username}', '--format=%.18i %.25j %.12q %.12T %.40R'],
                check=True, capture_output=True, text=True, timeout=60)
    current = account['users'].get(username, {}).get('own', {})
    remaining = {key: max(0, cap[key] - current.get(key, 0)) for key in request}
    quota_wait = plan['config']['qos'] == 'own' and any(request[key] > remaining[key] for key in request)
    capacity_wait = any(request[key] > account['avail'][key] or request[key] > capacity['cluster']['avail'][key] for key in request)
    plan['capacity_check'] = dict(request=request, qos=plan['config']['qos'], own_cap=cap, current_own=current,
                                  own_remaining=remaining, account_available=account['avail'],
                                  cluster_available=capacity['cluster']['avail'],
                                  queue_expected=quota_wait or capacity_wait,
                                  preemptible=plan['config']['qos'] == 'extra')
    # The atomic mkdir is the submission lock. There is intentionally no resume
    # mode: IDs/unknown responses remain inspectable if anything fails part-way.
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    (root / 'logs').mkdir()
    plan['status'] = 'submitting'
    save_manifest(plan)
    (root / 'capacity.json').write_text(json.dumps(capacity, indent=2) + '\n')
    (root / 'existing-jobs.txt').write_text(queue.stdout)
    try:
        previous_tasks = {}
        for job in plan['jobs']:
            # Optional task lanes: one task per GPU, subsequent evaluation seeds
            # wait for that same task without holding a worker between jobs.
            if plan['config'].get('sequential_by_task') and job['task'] in previous_tasks:
                dependency = ','.join(filter(None, [job.get('dependency'),
                    'afterany:' + previous_tasks[job['task']]]))
                job['dependency'] = dependency
                job['command'] = [v for v in job['command'] if not v.startswith('--dependency=')]
                job['command'].insert(1, '--dependency=' + dependency)
            submit_one(plan, job, run)
            previous_tasks[job['task']] = job['job_id']
        dependency, command = aggregation_command(plan)
        plan['aggregator'].update(dependency=dependency, command=command)
        submit_one(plan, plan['aggregator'], run)
        plan['status'] = 'submitted'
        save_manifest(plan)
    except BaseException as exc:
        plan['status'] = 'partial_failure'
        plan['error'] = f'{type(exc).__name__}: {exc}'
        save_manifest(plan)
        raise
    return plan


def main(argv=None):
    args = arguments(argv)
    try:
        plan = build_plan(args)
        if args.submit:
            submit_plan(plan)
        print(json.dumps(plan, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Evaluation preparation failed: {exc}', file=sys.stderr)
        print('No automatic retries or cancellations. Inspect any saved manifest before another submission.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
