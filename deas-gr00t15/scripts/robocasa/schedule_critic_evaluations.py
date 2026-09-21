#!/usr/bin/env python3
"""Queue two one-GPU task chains after their respective critic training jobs."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import submit_evaluations as common

VARIANTS = (
    ('lr3e4', '138923', '20260919T155726Z-critic-lr3e4-seed42', 2),
    ('lr3e4-featurefix', '138941', '20260919T160821Z-critic-lr3e4-featurefix-seed42', 1),
)
ACTOR_RELATIVE = 'output/deas-training/20260918T183345.797990880Z/02-bc-rollout'
DEFAULT_NUM_SAMPLES = 10


def make_plan(output_root, variant):
    label, jid, run_name, passes = variant
    root = common.owned_path(output_root) / label
    repo = common.REPO_ROOT
    actor = repo / ACTOR_RELATIVE
    training = repo / 'output/deas-training' / run_name
    manifest = json.loads((training / 'manifest.json').read_text())
    if str(manifest['job_id']) != jid or manifest['max_steps'] != 10000 or manifest['seed'] != 42:
        raise ValueError('Training identity does not match the requested evaluation')
    if json.loads((actor / 'trainer_state.json').read_text())['global_step'] != 10000:
        raise ValueError('Actor has not completed 10000 steps')
    marker = json.loads((training / '03-critic/checkpoint-5000/config.json').read_text())['critic_cfg'].get('online_q_feature_passes', 2)
    if marker != passes:
        raise ValueError('Training feature-path marker mismatch')
    group = 'critic-followup-' + label + '-' + hashlib.sha256(str(root).encode()).hexdigest()[:8]
    snapshot = root / 'source-snapshot/repo'
    jobs=[]
    for task in common.TASKS:
        key=f'{label}-eval0-{task}-bon{DEFAULT_NUM_SAMPLES}'
        output=root/'results/eval-seed-0'/task
        command=['sbatch','--parsable','--account=sub','--qos=own','--partition=compute',
            '--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-gpu=8','--mem=96G',
            f'--container={common.IMAGE}','--export=NONE','--time=01:30:00',
            '--kill-on-invalid-dep=yes',f'--job-name=rc-{label}',f'--chdir={repo}',
            f'--comment={group}:{task}',f'--output={root}/logs/{key}-%j.out',f'--error={root}/logs/{key}-%j.err',
            str(snapshot/'slurm/robocasa_after_critic_eval.sbatch'),str(actor),str(training/'03-critic'),
            task,str(output),'50','0',group,str(snapshot),'16',str(DEFAULT_NUM_SAMPLES),'wandb',str(passes)]
        jobs.append(dict(key=key,training_seed=42,eval_seed=0,task=task,method='deas',actor=str(actor),
            critic=str(training/'03-critic'),expected_episodes=50,output_dir=str(output),
            result_path=str(output/'result.json'),job_id=None,command=command,
            dependency=None,submission_state='planned'))
    return dict(schema_version=1,status='planned',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        output_root=str(root),source_snapshot=str(snapshot),training_job_id=jid,variant=label,
        config=dict(n_envs=1,action_horizon=16,execute_horizon=16,denoising_steps=4,num_samples=DEFAULT_NUM_SAMPLES,
            temperature=0.0,deas_backend='checkpoint',critic_feature_passes=passes,terminate_on_success=True,
            save_video=True,save_inference_inputs=True,report_to='wandb',training_seed=42,eval_seeds=[0],
            episodes=50,tasks=list(common.TASKS),qos='own',account='sub',gpus_per_job=1,cpus_per_gpu=8,
            mem_gib=96,time_limit='01:30:00',max_concurrent_gpus=1,wandb_group=group),
        jobs=jobs,aggregator=dict(job_id=None,dependency=None,output_dir=str(root/'aggregate'),
                                 command=None,submission_state='planned'))


def freeze(plan):
    root=Path(plan['output_root']);root.mkdir();(root/'logs').mkdir()
    snapshot=Path(plan['source_snapshot']);snapshot.mkdir(parents=True)
    shutil.copytree(common.REPO_ROOT/'gr00t',snapshot/'gr00t',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative in ('scripts/eval_policy_robocasa.py','scripts/robocasa/aggregate_results.py',
                     'scripts/robocasa/replay_inference.py','scripts/robocasa/schedule_critic_evaluations.py',
                     'slurm/robocasa_after_critic_eval.sbatch','slurm/robocasa_collect_results.sbatch'):
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((common.REPO_ROOT/relative).read_bytes())
    collector=snapshot/'slurm/robocasa_collect_results.sbatch'
    collector.write_text(collector.read_text().replace('"$REPO_ROOT/scripts/robocasa/aggregate_results.py"',
                                                      '"'+str(snapshot/'scripts/robocasa/aggregate_results.py')+'"'))
    plan['source_hashes']={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in snapshot.rglob('*') if p.is_file()}
    common.save_manifest(plan)


def submit(plans, root, run=subprocess.run):
    if root.exists():raise FileExistsError('Output exists; inspect manifests before any retry')
    capacity=json.loads(run(['snode','--json'],check=True,capture_output=True,text=True).stdout)
    cap=capacity['accounts']['sub']['per_user_own_cap']
    if cap['gpu']<2 or cap['cpu']<16 or cap['mem_mib']<196608:
        raise ValueError('Two simultaneous one-GPU evaluations exceed own cap')
    states={}
    for plan in plans:
        x=json.loads(run(['sjob','--json',plan['training_job_id']],check=True,capture_output=True,text=True).stdout)
        tasks=x.get('tasks',[])
        if len(tasks)!=1 or tasks[0]['state'] not in ('RUNNING','PENDING','CONFIGURING'):
            raise ValueError('Expected active training job; resolve terminal status before submission')
        states[plan['training_job_id']]=tasks[0]['state']
    root.mkdir(parents=True)
    (root/'capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
    (root/'training-states.json').write_text(json.dumps(states,indent=2)+'\n')
    for plan in plans:freeze(plan)
    for plan in plans:
        plan['status']='submitting';common.save_manifest(plan)
        previous=None
        try:
            for job in plan['jobs']:
                dependency='afterok:'+plan['training_job_id']
                if previous:dependency+=',afterany:'+previous
                job['dependency']=dependency
                job['command'].insert(1,'--dependency='+dependency)
                common.submit_one(plan,job,run)
                previous=job['job_id']
                print(plan['variant'],job['task'],job['job_id'],dependency,flush=True)
            dependency,command=common.aggregation_command(plan)
            command[command.index(str(common.REPO_ROOT/'slurm/robocasa_collect_results.sbatch'))]=str(Path(plan['source_snapshot'])/'slurm/robocasa_collect_results.sbatch')
            plan['aggregator'].update(dependency=dependency,command=command)
            common.submit_one(plan,plan['aggregator'],run)
            plan['status']='submitted';common.save_manifest(plan)
        except BaseException as exc:
            plan['status']='partial_failure';plan['error']=str(exc);common.save_manifest(plan);raise
    (root/'schedule.json').write_text(json.dumps({'max_concurrent_eval_gpus':2,'total_episodes':400,
        'runs':[{'variant':p['variant'],'training_job':p['training_job_id'],
                 'manifest':str(Path(p['output_root'])/'manifest.json'),
                 'jobs':[j['job_id'] for j in p['jobs']],'aggregate_job':p['aggregator']['job_id']} for p in plans]},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--submit',action='store_true');a=p.parse_args();root=common.owned_path(a.output_root)
    plans=[make_plan(root,v) for v in VARIANTS]
    if a.submit:submit(plans,root)
    else:print(json.dumps(plans,indent=2))

if __name__=='__main__':main()
