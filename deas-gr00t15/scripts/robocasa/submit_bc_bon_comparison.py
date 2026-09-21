#!/usr/bin/env python3
"""Submit matched BC actor and DEAS BoN evaluations with protected own quota."""
import argparse
import datetime as dt
import getpass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import submit_evaluations as common
import submit_bon_evaluations as bon


def build_plan(args):
    root = common.owned_path(args.output_root)
    bc_label = getattr(args, 'bc_label', 'bc2')
    if bc_label not in ('bc1', 'bc2'):
        raise ValueError('bc_label must be bc1 or bc2')
    serial = bool(getattr(args, 'serial', False))
    report_to = getattr(args, 'report_to', 'none')
    if report_to not in ('wandb', 'none'):
        raise ValueError('report_to must be wandb or none')
    bon_args = argparse.Namespace(actor=args.actor, critic=args.critic, output_root=root / 'bon',
        eval_seeds=args.eval_seeds, tasks=args.tasks, episodes=args.episodes,
        execute_horizon=16, num_samples=args.num_samples, time=args.bon_time, report_to=report_to)
    bon_plan = bon.build_plan(bon_args)
    bc_jobs = []
    for seed in args.eval_seeds:
        bc_args = common.arguments(['--actor', str(args.actor), '--seed', '42', '--eval-seed', str(seed),
            '--output-root', str(root / bc_label), '--episodes', str(args.episodes), '--save-video',
            '--time', args.bc_time, '--report-to', report_to, '--tasks', *args.tasks])
        bc_plan = common.build_plan(bc_args)
        for job in bc_plan['jobs']:
            old_key, old_output = job['key'], job['output_dir']
            key = f'eval{seed}-' + old_key
            output = root / bc_label / 'results' / f'eval-seed-{seed}' / job['task']
            command = [value.replace(old_key, key).replace(old_output, str(output)) for value in job['command']]
            script = command.index(str(common.REPO_ROOT / 'slurm/robocasa_pipeline_eval.sbatch'))
            command[script + 13] = '10000'  # worker checks the completed BC Trainer step
            job.update(key=key, output_dir=str(output), result_path=str(output / 'result.json'),
                       video_dir=str(output / 'videos'), command=command, expected_training_steps=10000)
            bc_jobs.append(job)
    bc_plan['jobs'] = bc_jobs
    bc_plan['config'].update(eval_seed=None, eval_seeds=args.eval_seeds,
                             sequential_by_task=not serial, serial=serial, bc_label=bc_label)
    bc_plan['notes'] = [f'{bc_label.upper()} actor-only baseline, matched eval seeds/tasks to DEAS BoN.']
    bc_plan['config']['terminate_on_success'] = True
    bon_plan['config'].update(serial=serial, bc_label=bc_label, terminate_on_success=True)
    # Results have unique (training seed, eval seed, task, method). The combined
    # summary checks BoN sample count only for DEAS, and videos for both methods.
    config = {**bc_plan['config'], 'num_samples': args.num_samples, 'temperature': 0.0,
              'deas_backend': 'checkpoint', 'execute_horizon': 16, 'save_inference_inputs': False,
              'terminate_on_success': True}
    plan = dict(schema_version=1, status='planned', output_root=str(root),
        created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), config=config,
        source={'actor': str(common.owned_path(args.actor)), 'critic': str(common.owned_path(args.critic))},
        jobs=[*bc_jobs, *bon_plan['jobs']],
        aggregator=dict(job_id=None, dependency=None, output_dir=str(root / 'aggregate'),
                        command=None, submission_state='planned'),
        bc_plan=bc_plan, bon_plan=bon_plan)
    if serial:
        plan['config']['max_concurrent_gpus'] = 1
        previous_key = None
        for job in plan['jobs']:
            # Job IDs do not exist during dry-run; record the intended predecessor.
            job['serial_predecessor_key'] = previous_key
            previous_key = job['key']
    return plan


def _prepare_serial_children(plan, run):
    """Check one-GPU capacity and freeze BoN sources before submitting any job."""
    capacity = json.loads(run(['snode', '--json'], check=True, capture_output=True,
                              text=True, timeout=60).stdout)
    username = getpass.getuser()
    queue = run(['squeue', f'--user={username}', '--format=%.18i %.25j %.12q %.12T %.40R'],
                check=True, capture_output=True, text=True, timeout=60)
    account = capacity['accounts']['sub']
    cap = account['per_user_own_cap']
    request = {'gpu': 1, 'cpu': 8, 'mem_mib': 96 * 1024}
    if any(request[key] > cap[key] for key in request):
        raise ValueError(f'Serial evaluation exceeds own quota: {request}; cap={cap}')
    current = account['users'].get(username, {}).get('own', {})
    remaining = {key: max(0, cap[key] - current.get(key, 0)) for key in request}
    plan['capacity_check'] = dict(request=request, own_cap=cap, current_own=current,
        own_remaining=remaining, account_available=account['avail'],
        cluster_available=capacity['cluster']['avail'], max_concurrent_gpus=1,
        queue_expected=any(request[key] > remaining[key] or
                           request[key] > account['avail'][key] or
                           request[key] > capacity['cluster']['avail'][key] for key in request))
    root = Path(plan['output_root'])
    (root / 'capacity.json').write_text(json.dumps(capacity, indent=2) + '\n')
    (root / 'existing-jobs.txt').write_text(queue.stdout)
    for child in (plan['bc_plan'], plan['bon_plan']):
        child_root = Path(child['output_root'])
        child_root.mkdir()
        (child_root / 'logs').mkdir()
        child['status'] = 'submitting'
        common.save_manifest(child)
    child = plan['bon_plan']
    snapshots = Path(child['output_root']) / 'source-snapshot/repo'
    snapshots.mkdir(parents=True)
    shutil.copytree(common.REPO_ROOT / 'gr00t', snapshots / 'gr00t',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for relative in ('scripts/eval_policy_robocasa.py', 'scripts/robocasa/replay_inference.py',
                     'scripts/robocasa/aggregate_results.py', 'scripts/robocasa/submit_bon_evaluations.py',
                     'slurm/robocasa_bon_eval.sbatch'):
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((common.REPO_ROOT / relative).read_bytes())
    child['source_snapshot'] = str(snapshots)
    child['source_hashes'] = {str(path.relative_to(snapshots)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in snapshots.rglob('*') if path.is_file()}
    common.save_manifest(child)
    common.save_manifest(plan)


def _submit_serial_record(plan, child, record, run):
    # The parent journals the entire submission before every sbatch, including
    # nested aggregators. Both manifests retain accepted or ambiguous responses.
    record['submission_state'] = 'submitting'
    common.save_manifest(child)
    try:
        common.submit_one(plan, record, run)
    finally:
        common.save_manifest(child)
        common.save_manifest(plan)


def _submit_serial(plan, run):
    _prepare_serial_children(plan, run)
    previous = None
    for child in (plan['bc_plan'], plan['bon_plan']):
        try:
            for job in child['jobs']:
                if previous:
                    dependency = ','.join(filter(None, [job.get('dependency'), 'afterany:' + previous]))
                    job['dependency'] = dependency
                    job['command'] = [value for value in job['command'] if not value.startswith('--dependency=')]
                    job['command'].insert(1, '--dependency=' + dependency)
                _submit_serial_record(plan, child, job, run)
                previous = common.numeric_job_id(job['job_id'])
            dependency, command = common.aggregation_command(child)
            child['aggregator'].update(dependency=dependency, command=command)
            _submit_serial_record(plan, child, child['aggregator'], run)
            child['status'] = 'submitted'
            common.save_manifest(child)
            common.save_manifest(plan)
        except BaseException as exc:
            child['status'] = 'partial_failure'
            child['error'] = f'{type(exc).__name__}: {exc}'
            common.save_manifest(child)
            raise


def submit(plan, run=None):
    run = subprocess.run if run is None else run
    root = Path(plan['output_root'])
    if root.exists() or root.is_symlink():
        raise FileExistsError('Comparison root already exists; inspect its manifest before another attempt')
    root.mkdir(parents=True)
    (root / 'logs').mkdir()
    plan['status'] = 'submitting'
    common.save_manifest(plan)
    try:
        if plan['config'].get('serial'):
            _submit_serial(plan, run)
        else:
            common.submit_plan(plan['bc_plan'], run=run)
            common.save_manifest(plan)
            plan['bon_plan']['initial_task_dependencies'] = {
                job['task']: job['job_id'] for job in plan['bc_plan']['jobs']}
            bon.submit(plan['bon_plan'], run=run)
            common.save_manifest(plan)
        dependency, command = common.aggregation_command(plan)
        plan['aggregator'].update(dependency=dependency, command=command)
        common.submit_one(plan, plan['aggregator'], run)
        plan['status'] = 'submitted'
        common.save_manifest(plan)
    except BaseException as exc:
        plan['status'] = 'partial_failure'
        plan['error'] = f'{type(exc).__name__}: {exc}'
        common.save_manifest(plan)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--critic', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--eval-seeds', type=common.nonnegative, nargs='+', default=[0, 1, 2])
    p.add_argument('--tasks', choices=common.TASKS, nargs='+', default=list(common.TASKS))
    p.add_argument('--episodes', type=common.positive, default=50)
    p.add_argument('--num-samples', type=common.positive, default=10)
    p.add_argument('--bc-time', required=True)
    p.add_argument('--bon-time', required=True)
    p.add_argument('--report-to', choices=('wandb', 'none'), default='none')
    p.add_argument('--bc-label', choices=('bc1', 'bc2'), default='bc2',
                   help='BC training stage label used in paths and manifests')
    p.add_argument('--serial', action='store_true',
                   help='Chain every task/seed job, using at most one GPU for this comparison')
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    plan = build_plan(a)
    if a.submit:
        submit(plan)
    print(json.dumps({'status': plan['status'], 'output_root': plan['output_root'],
        'jobs': [{'method': j['method'], 'task': j['task'], 'eval_seed': j['eval_seed'],
                  'job_id': j['job_id']} for j in plan['jobs']],
        'aggregate_job_id': plan['aggregator']['job_id'],
        'episodes_per_method': a.episodes * len(a.tasks) * len(a.eval_seeds),
        'bon_num_samples': a.num_samples, 'bc_label': a.bc_label, 'serial': a.serial}, indent=2))


if __name__ == '__main__':
    main()
