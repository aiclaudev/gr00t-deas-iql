#!/usr/bin/env python3
"""Plan/submit task-specific 1-GPU BoN evaluations, sequential seeds per task."""
import argparse
import datetime as dt
import getpass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import submit_evaluations as common


def build_plan(args):
    num_samples = getattr(args, 'num_samples', 10)
    if type(num_samples) is not int or num_samples <= 0:
        raise ValueError('num_samples must be a positive integer')
    report_to = getattr(args, 'report_to', 'wandb')
    if report_to not in ('none', 'wandb'):
        raise ValueError('report_to must be none or wandb')
    actor, critic, root = map(common.owned_path, (args.actor,args.critic,args.output_root))
    for model in (actor,critic):
        state=json.loads((model/'trainer_state.json').read_text())
        if state.get('global_step')!=10000:
            raise ValueError(f'Expected 10000-step model: {model}')
        for name in ('config.json','experiment_cfg/metadata.json'):
            if not (model/name).is_file(): raise ValueError(f'Missing {model/name}')
        if not any((model/name).is_file() for name in ('model.safetensors','model.safetensors.index.json')):
            raise ValueError(f'Missing weights: {model}')
    if len(set(args.eval_seeds))!=len(args.eval_seeds): raise ValueError('Duplicate evaluation seed')
    if len(set(args.tasks))!=len(args.tasks): raise ValueError('Duplicate task')
    if not re.fullmatch(r'\d+:[0-5]\d:[0-5]\d',args.time) or not any(int(part) for part in args.time.split(':')):
        raise ValueError('Time limit must be a positive HH:MM:SS value')
    if not 1 <= args.execute_horizon <= 16: raise ValueError('execute_horizon must be between 1 and 16')
    group=f'robocasa-bon{num_samples}-'+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    jobs=[]
    for seed in args.eval_seeds:
        for task in args.tasks:
            key=f'train42-eval{seed}-{task}-bon{num_samples}-exec{args.execute_horizon}'
            output=root/'results'/f'eval-seed-{seed}'/task
            command=['sbatch','--parsable','--account=sub','--qos=own','--partition=compute',
                     '--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-gpu=8','--mem=96G',
                     f'--container={common.IMAGE}','--export=NONE',f'--time={args.time}',
                     '--kill-on-invalid-dep=yes','--job-name=deas-rc-bon',f'--chdir={common.REPO_ROOT}',
                     f'--comment={group}:{key}',f'--output={root}/logs/{key}-%j.out',
                     f'--error={root}/logs/{key}-%j.err',
                     str(common.REPO_ROOT/'slurm/robocasa_bon_eval.sbatch'),str(actor),str(critic),
                     task,str(output),str(args.episodes),str(seed),group,str(root/'source-snapshot/repo'),str(args.execute_horizon),str(num_samples),report_to]
            jobs.append(dict(key=key,training_seed=42,eval_seed=seed,task=task,method='deas',
                             actor=str(actor),critic=str(critic),expected_episodes=args.episodes,
                             output_dir=str(output),result_path=str(output/'result.json'),job_id=None,
                             command=command,dependency='',submission_state='planned'))
    return dict(schema_version=1,status='planned',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                output_root=str(root),jobs=jobs,config=dict(n_envs=1,action_horizon=16,execute_horizon=args.execute_horizon,denoising_steps=4,
                num_samples=num_samples,report_to=report_to,temperature=0.0,deas_backend='checkpoint',save_video=True,save_inference_inputs=True,
                critic_feature_passes=2,
                training_seed=42,eval_seeds=args.eval_seeds,episodes=args.episodes,tasks=args.tasks,
                qos='own',account='sub',gpus_per_job=1,cpus_per_gpu=8,mem_gib=96,
                time_limit=args.time,wandb_group=group),
                aggregator=dict(job_id=None,dependency=None,output_dir=str(root/'aggregate'),command=None,submission_state='planned'))


def submit(plan, run=subprocess.run):
    root=Path(plan['output_root'])
    if root.exists(): raise FileExistsError('Submission directory exists; inspect manifest before retrying')
    capacity=json.loads(run(['snode','--json'],check=True,capture_output=True,text=True).stdout)
    queue=run(['squeue',f'--user={getpass.getuser()}','--format=%.18i %.25j %.12q %.12T %.40R'],check=True,capture_output=True,text=True)
    cap=capacity['accounts']['sub']['per_user_own_cap']
    request={'gpu':4,'cpu':32,'mem_mib':4*96*1024}
    if any(request[k]>cap[k] for k in request): raise ValueError('Four parallel evaluations exceed own quota')
    root.parent.mkdir(parents=True,exist_ok=True);root.mkdir();(root/'logs').mkdir()
    (root/'capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
    (root/'existing-jobs.txt').write_text(queue.stdout)
    snapshots=root/'source-snapshot/repo';snapshots.mkdir(parents=True)
    shutil.copytree(common.REPO_ROOT/'gr00t', snapshots/'gr00t',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative in ('scripts/eval_policy_robocasa.py','scripts/robocasa/replay_inference.py',
                     'scripts/robocasa/aggregate_results.py','scripts/robocasa/submit_bon_evaluations.py',
                     'slurm/robocasa_bon_eval.sbatch'):
        target=snapshots/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((common.REPO_ROOT/relative).read_bytes())
    plan['source_snapshot']=str(snapshots)
    plan['source_hashes']={str(p.relative_to(snapshots)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in snapshots.rglob('*') if p.is_file()}
    plan['status']='submitting';common.save_manifest(plan)
    previous={task:common.numeric_job_id(jid) for task,jid in plan.get('initial_task_dependencies', {}).items()}
    try:
        for job in plan['jobs']:
            if job['task'] in previous:
                dependency='afterany:'+previous[job['task']]
                job['dependency']=dependency
                job['command'].insert(1,'--dependency='+dependency)
            common.submit_one(plan,job,run)
            previous[job['task']]=job['job_id']
            print(f"Accepted eval seed {job['eval_seed']} {job['task']}: {job['job_id']}",flush=True)
        dependency,command=common.aggregation_command(plan)
        plan['aggregator'].update(dependency=dependency,command=command)
        common.submit_one(plan,plan['aggregator'],run)
        plan['status']='submitted';common.save_manifest(plan)
    except BaseException as exc:
        plan['status']='partial_failure';plan['error']=str(exc);common.save_manifest(plan);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--critic',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--eval-seeds',type=common.nonnegative,nargs='+',default=[0,1,2])
    p.add_argument('--tasks',choices=common.TASKS,nargs='+',default=list(common.TASKS))
    p.add_argument('--episodes',type=common.positive,default=50)
    p.add_argument('--report-to',choices=('none','wandb'),default='wandb')
    p.add_argument('--num-samples',type=common.positive,default=10,help='Candidate actions per BoN inference call')
    p.add_argument('--execute-horizon','--execute_horizon',type=common.positive,default=16)
    p.add_argument('--time',required=True)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args();plan=build_plan(args)
    if args.submit: submit(plan)
    print(json.dumps(plan,indent=2))

if __name__=='__main__': main()
