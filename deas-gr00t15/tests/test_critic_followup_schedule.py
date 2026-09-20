import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def scheduler(tmp_path, monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts/robocasa'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('critic_followup_scheduler',scripts/'schedule_critic_evaluations.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    repo=tmp_path/'repo';repo.mkdir()
    monkeypatch.setattr(module.common,'REPO_ROOT',repo)
    actor=repo/module.ACTOR_RELATIVE;actor.mkdir(parents=True)
    (actor/'trainer_state.json').write_text(json.dumps({'global_step':10000}))
    for label,jid,name,passes in module.VARIANTS:
        root=repo/'output/deas-training'/name
        checkpoint=root/'03-critic/checkpoint-5000';checkpoint.mkdir(parents=True)
        (root/'manifest.json').write_text(json.dumps({'job_id':jid,'max_steps':10000,'seed':42}))
        (checkpoint/'config.json').write_text(json.dumps({'critic_cfg':{'online_q_feature_passes':passes}}))
        assert not (root/'03-critic/trainer_state.json').exists()
    for name in ('gr00t/example.py','scripts/eval_policy_robocasa.py','scripts/robocasa/aggregate_results.py',
                 'scripts/robocasa/replay_inference.py','scripts/robocasa/schedule_critic_evaluations.py',
                 'slurm/robocasa_after_critic_eval.sbatch','slurm/robocasa_collect_results.sbatch'):
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# snapshot fixture\n')
    return module


def test_future_checkpoints_queue_two_separate_serial_chains(tmp_path,monkeypatch):
    m=scheduler(tmp_path,monkeypatch);root=tmp_path/'evaluation'
    plans=[m.make_plan(root,v) for v in m.VARIANTS]
    submitted=[]
    def run(command,**kwargs):
        if command[0]=='snode':
            out=json.dumps({'accounts':{'sub':{'per_user_own_cap':{'gpu':4,'cpu':56,'mem_mib':900000}}}})
        elif command[0]=='sjob':out=json.dumps({'tasks':[{'state':'RUNNING'}]})
        else:
            assert command[0]=='sbatch';submitted.append(command.copy());out=str(8000+len(submitted))+'\n'
        return subprocess.CompletedProcess(command,0,out,'')
    m.submit(plans,root,run=run)
    assert len(submitted)==10  # eight GPU jobs and two CPU summaries
    for plan in plans:
        prior=None
        for job in plan['jobs']:
            assert 'afterok:'+plan['training_job_id'] in job['dependency'].split(',')
            if prior:assert 'afterany:'+prior in job['dependency'].split(',')
            else:assert 'afterany:' not in job['dependency']
            assert '--gres=gpu:1' in job['command']
            assert '--kill-on-invalid-dep=yes' in job['command']
            assert job['command'][-1]==str(plan['config']['critic_feature_passes'])
            prior=job['job_id']
        saved=json.loads((Path(plan['output_root'])/'manifest.json').read_text())
        assert saved['status']=='submitted'
        assert saved['config']['terminate_on_success'] is True
        assert Path(saved['source_snapshot'],'gr00t/example.py').exists()


def test_failed_training_cannot_queue_evaluation(tmp_path,monkeypatch):
    m=scheduler(tmp_path,monkeypatch);root=tmp_path/'evaluation'
    plans=[m.make_plan(root,v) for v in m.VARIANTS]
    def run(command,**kwargs):
        assert command[0]!='sbatch'
        data=({'accounts':{'sub':{'per_user_own_cap':{'gpu':4,'cpu':56,'mem_mib':900000}}}}
              if command[0]=='snode' else {'tasks':[{'state':'FAILED'}]})
        return subprocess.CompletedProcess(command,0,json.dumps(data),'')
    with pytest.raises(ValueError,match='active training'):
        m.submit(plans,root,run=run)
    assert not root.exists()
