"""Submission-safety checks with fake cluster commands and fake model files."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts/robocasa'
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location('bc_bon_serial_test', SCRIPTS / 'submit_bc_bon_comparison.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
finally:
    sys.path.pop(0)


@pytest.fixture
def args(tmp_path, monkeypatch):
    common = comparison.common
    monkeypatch.setattr(common, 'USER_ROOT', tmp_path)
    for name in ('bc1', 'critic'):
        path = tmp_path / name
        (path / 'experiment_cfg').mkdir(parents=True)
        (path / 'trainer_state.json').write_text('{"global_step": 10000}')
        (path / 'config.json').write_text('{}')
        (path / 'experiment_cfg/metadata.json').write_text('{}')
        (path / 'model.safetensors').write_bytes(b'never loaded')
    fake_repo = tmp_path / 'repo'
    for relative in ('gr00t/__init__.py', 'scripts/eval_policy_robocasa.py',
                     'scripts/robocasa/replay_inference.py', 'scripts/robocasa/aggregate_results.py',
                     'scripts/robocasa/submit_bon_evaluations.py', 'slurm/robocasa_bon_eval.sbatch'):
        target = fake_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# source snapshot fixture\n')
    monkeypatch.setattr(common, 'REPO_ROOT', fake_repo)
    return argparse.Namespace(actor=tmp_path / 'bc1', critic=tmp_path / 'critic',
        output_root=tmp_path / 'evaluation', eval_seeds=[0], tasks=list(common.TASKS),
        episodes=50, num_samples=50, bc_time='01:00:00', bon_time='01:00:00',
        bc_label='bc1', serial=True, report_to='none')


class FakeCluster:
    def __init__(self, root, gpu_cap=1, fail_at=None, ambiguous_at=None):
        self.root, self.gpu_cap = root, gpu_cap
        self.fail_at, self.ambiguous_at = fail_at, ambiguous_at
        self.calls, self.submissions = [], []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if command[:2] == ['snode', '--json']:
            resources = {'gpu': self.gpu_cap, 'cpu': 56, 'mem_mib': 900000}
            capacity = {'accounts': {'sub': {'per_user_own_cap': resources,
                'users': {}, 'avail': resources}}, 'cluster': {'avail': resources}}
            return subprocess.CompletedProcess(command, 0, json.dumps(capacity), '')
        if command[0] == 'squeue':
            return subprocess.CompletedProcess(command, 0, 'mock queue\n', '')
        assert command[0] == 'sbatch'
        self.submissions.append(command)
        index = len(self.submissions)
        # The full comparison journal exists before any external submission.
        saved = json.loads((self.root / 'manifest.json').read_text())
        assert saved['status'] == 'submitting'
        if index == self.fail_at:
            return subprocess.CompletedProcess(command, 1, '', 'mock rejection')
        stdout = 'unrecognized' if index == self.ambiguous_at else f'{700 + index};cluster\n'
        return subprocess.CompletedProcess(command, 0, stdout, '')


def test_bc1_dry_plan_has_one_chain_and_local_reporting(args):
    plan = comparison.build_plan(args)
    assert not args.output_root.exists()
    assert plan['config']['bc_label'] == plan['bc_plan']['config']['bc_label'] == 'bc1'
    assert plan['config']['max_concurrent_gpus'] == 1
    assert len(plan['jobs']) == 8
    assert sum(job['expected_episodes'] for job in plan['jobs']) == 400
    assert plan['bc_plan']['output_root'] == str(args.output_root / 'bc1')
    assert plan['config']['report_to'] == plan['bon_plan']['config']['report_to'] == 'none'
    previous = None
    for job in plan['jobs']:
        assert job['serial_predecessor_key'] == previous
        previous = job['key']
        assert '--gres=gpu:1' in job['command']
        assert 'none' in job['command']
        assert job['actor'] == str(args.actor)
    assert all('/bc1/results/' in job['output_dir'] for job in plan['bc_plan']['jobs'])
    assert all(job['critic'] == str(args.critic) for job in plan['bon_plan']['jobs'])


def test_serial_submission_chains_eight_gpu_jobs_and_preserves_three_aggregates(args):
    plan = comparison.build_plan(args)
    cluster = FakeCluster(args.output_root)
    comparison.submit(plan, run=cluster)
    assert len(cluster.submissions) == 11
    assert plan['status'] == 'submitted'
    assert plan['capacity_check']['request']['gpu'] == 1
    assert not plan['capacity_check']['queue_expected']
    previous = None
    for job in plan['jobs']:
        expected = '' if previous is None else 'afterany:' + previous
        assert job['dependency'] == expected
        options = [value for value in job['command'] if value.startswith('--dependency=')]
        assert options == ([] if not expected else ['--dependency=' + expected])
        assert job['submission_state'] == 'accepted'
        previous = job['job_id']
    assert plan['bon_plan']['jobs'][0]['dependency'] == 'afterany:' + plan['bc_plan']['jobs'][-1]['job_id']
    for owner in (plan['bc_plan'], plan['bon_plan'], plan):
        agg = owner['aggregator']
        assert agg['dependency'] == 'afterany:' + ':'.join(job['job_id'] for job in owner['jobs'])
        assert '--cpus-per-task=2' in agg['command']
        assert not any(arg.startswith('--gres=') for arg in agg['command'])
        assert json.loads((Path(owner['output_root']) / 'manifest.json').read_text()) == owner
    snapshot = Path(plan['bon_plan']['source_snapshot'])
    assert (snapshot / 'gr00t/__init__.py').read_text() == '# source snapshot fixture\n'
    assert 'gr00t/__init__.py' in plan['bon_plan']['source_hashes']
    with pytest.raises(FileExistsError):
        comparison.submit(comparison.build_plan(args), run=lambda *_a, **_k: pytest.fail('duplicate must not contact cluster'))


def test_legacy_namespace_stays_bc2_with_parallel_task_lanes(args):
    del args.bc_label
    del args.serial
    plan = comparison.build_plan(args)
    assert plan['config']['bc_label'] == 'bc2'
    assert not plan['config']['serial']
    assert plan['bc_plan']['config']['sequential_by_task']
    assert plan['bc_plan']['output_root'] == str(args.output_root / 'bc2')
    cluster = FakeCluster(args.output_root, gpu_cap=4)
    comparison.submit(plan, run=cluster)
    assert len(cluster.submissions) == 11
    assert all(not job['dependency'] for job in plan['bc_plan']['jobs'])
    for bc_job, bon_job in zip(plan['bc_plan']['jobs'], plan['bon_plan']['jobs']):
        assert bon_job['dependency'] == 'afterany:' + bc_job['job_id']


@pytest.mark.parametrize('ambiguous', [False, True])
def test_partial_failure_is_durable_and_never_retried(args, ambiguous):
    plan = comparison.build_plan(args)
    cluster = FakeCluster(args.output_root, **({'ambiguous_at': 7} if ambiguous else {'fail_at': 7}))
    with pytest.raises(RuntimeError):
        comparison.submit(plan, run=cluster)
    saved = json.loads((args.output_root / 'manifest.json').read_text())
    bon_saved = json.loads((args.output_root / 'bon/manifest.json').read_text())
    assert saved['status'] == bon_saved['status'] == 'partial_failure'
    assert sum(job['job_id'] is not None for job in saved['jobs']) == 5
    assert saved['bon_plan']['jobs'][1]['submission_state'] == ('unknown' if ambiguous else 'rejected')
    assert saved['bon_plan'] == bon_saved
    assert len(cluster.submissions) == 7
    with pytest.raises(FileExistsError):
        comparison.submit(comparison.build_plan(args), run=lambda *_a, **_k: pytest.fail('must not retry'))


def test_cli_serial_bc1_is_dry_by_default(args, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [str(SCRIPTS / 'submit_bc_bon_comparison.py'),
        '--actor', str(args.actor), '--critic', str(args.critic), '--output-root', str(args.output_root),
        '--eval-seeds', '0', '--bc-time', '01:00:00', '--bon-time', '01:00:00',
        '--bc-label', 'bc1', '--serial'])
    monkeypatch.setattr(comparison.subprocess, 'run', lambda *_a, **_k: pytest.fail('dry-run called cluster'))
    comparison.main()
    output = json.loads(capsys.readouterr().out)
    assert output['bc_label'] == 'bc1' and output['serial']
    assert output['episodes_per_method'] == 200
    assert len(output['jobs']) == 8
    assert not args.output_root.exists()
