"""Validate sweep ordering and submission bookkeeping without any cluster calls."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('svf_sweep_test', REPO / 'scripts/submit_svf_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def config():
    return json.loads((REPO / 'configs/svf_sweep_seed42.json').read_text())


def plan(**kwargs):
    return sweep.build_plan(config(), REPO / 'output/svf-joint/sweeps/test-only', **kwargs)


def test_exact_grid_stage_and_future_actor_export():
    p = plan()
    expected = [(k, g) for k in [0.4, 0.6, 0.8, 1.0] for g in [0.25, 0.5, 0.8]]
    assert [(a['kappa'], a['g']) for a in p['arms']] == expected
    assert len(p['jobs']) == 72
    assert p['unresolved'] == ['train_time', 'eval_time', 'eval_episodes']
    baseline = p['arms'][0]['training_config']
    for arm in p['arms']:
        cfg = arm['training_config']
        assert {k: v for k, v in cfg.items() if k not in ('kappa', 'g')} == {
            k: v for k, v in baseline.items() if k not in ('kappa', 'g')}
        assert (cfg['steps'], cfg['stop_after_steps'], cfg['seed']) == (10000, 5000, 42)
        assert (cfg['soft_value_loss'], cfg['soft_value_init'], cfg['lr_scheduler']) == ('mse', 'critic-trunk', 'cosine')
        for job in arm['evaluation']['jobs']:
            assert job['actor'].endswith('/train/checkpoint-5000/actor')
            position = job['command'].index(str(REPO / 'slurm/svf_robocasa_eval.sbatch'))
            args = job['command'][position + 1:]
            assert len(args) == 16
            assert args[1] == '5000' and args[2] == job['actor']
            assert args[14] == '0'  # no BC Trainer step check on an SVF actor
            assert args[10].startswith(arm['key'])


def test_all_phases_have_correct_slurm_dependencies_and_resources():
    p = plan(eval_episodes=10, train_time='12:00:00', eval_time='01:00:00')
    accepted = {}
    for index, job in enumerate(p['jobs']):
        command, dep = sweep.resolved_command(job, accepted)
        phase = index % 6
        if phase == 0:
            assert job['kind'] == 'train'
            assert dep == (f'afterok:{1000 + index - 1}' if index else '')
            assert '--gres=gpu:4' in command
            assert command[command.index('--steps') + 1] == '10000'
            assert command[command.index('--stop-after-steps') + 1] == '5000'
        elif phase < 5:
            assert job['kind'] == 'eval'
            assert dep == f'afterok:{1000 + index - phase}'
            assert '--gres=gpu:1' in command and '--mem=96G' in command
        else:
            assert job['kind'] == 'summary'
            assert dep == 'afterany:' + ':'.join(str(1000 + i) for i in range(index - 4, index))
            assert not any(x.startswith('--gres=') for x in command)
        assert '--export=NONE' in command and '--kill-on-invalid-dep=yes' in command
        assert '<' not in ' '.join(command)
        accepted[job['key']] = str(1000 + index)
    assert len(accepted) == 72


def test_default_preview_does_not_write_or_call_cluster(monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail('Dry-run must not write or execute a command')
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    monkeypatch.setattr(Path, 'write_text', forbidden)
    monkeypatch.setattr(sweep.subprocess, 'run', forbidden)
    monkeypatch.setattr(sweep.subprocess, 'check_output', forbidden)
    assert sweep.main([]) == 0
    assert '12 arms / 72 jobs' in capsys.readouterr().out
    with pytest.raises(ValueError, match='Set these'):
        sweep.submit_plan(plan())


@pytest.mark.parametrize('field,value', [('kappas', [0.4, 0.4]), ('gs', [0.0]), ('gs', [float('nan')])])
def test_invalid_grids_rejected(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        sweep.build_plan(cfg, REPO / 'output/svf-joint/sweeps/test-only')


def test_invalid_eval_and_time_rejected():
    for kwargs in ({'eval_episodes': 0}, {'eval_episodes': True}, {'eval_time': '00:00:00'},
                   {'train_time': '12:70:00'}):
        with pytest.raises(ValueError):
            plan(**kwargs)
    with pytest.raises(ValueError, match='Dependency is not accepted'):
        sweep.resolved_command(plan()['jobs'][1], {})


@pytest.fixture
def submission_plan(tmp_path, monkeypatch):
    # tmp_path is rooted in this repository for the existing personal-path gates.
    monkeypatch.setattr(sweep, 'REPO', tmp_path)
    monkeypatch.setattr(sweep.train, 'REPO', tmp_path)
    monkeypatch.setattr(sweep.evaluation, 'REPO_ROOT', tmp_path)
    (tmp_path / 'configs').mkdir()
    cfg = config()
    base = json.loads((REPO / cfg['training_config']).read_text())
    (tmp_path / cfg['training_config']).write_text(json.dumps(base))
    return sweep.build_plan(cfg, tmp_path / 'output/svf-joint/test', train_time='12:00:00',
                            eval_time='01:00:00', eval_episodes=10)


def cluster_mock(p, calls, *, fail_at=None):
    def run(command, **kwargs):
        calls.append(command)
        if '--validate-config' in command:
            return subprocess.CompletedProcess(command, 0, '', '')
        if command[:2] == ['snode', '--json']:
            cap = {'gpu': 4, 'cpu': 56, 'mem_mib': 900000}
            state = {'accounts': {'sub': {'per_user_own_cap': cap, 'users': {}, 'avail': cap}},
                     'cluster': {'avail': cap}}
            return subprocess.CompletedProcess(command, 0, json.dumps(state), '')
        assert command[0] == 'sbatch'
        number = sum(c[0] == 'sbatch' for c in calls)
        saved = json.loads((Path(p['output_root']) / 'manifest.json').read_text())
        assert sum(j['submission_state'] == 'accepted' for j in saved['jobs']) == number - 1
        if number == fail_at:
            return subprocess.CompletedProcess(command, 1, '', 'mock rejection')
        return subprocess.CompletedProcess(command, 0, f'{1000 + number};cluster\n', '')
    return run


def test_mock_submission_saves_ids_and_refuses_duplicate(submission_plan, monkeypatch):
    p, calls = submission_plan, []
    monkeypatch.setattr(sweep.train, 'require_finished_teacher', lambda cfg: None)
    run = cluster_mock(p, calls)
    sweep.submit_plan(p, run=run)
    assert len([c for c in calls if c[0] == 'sbatch']) == 72
    assert p['status'] == 'submitted'
    for arm in p['arms']:
        saved = json.loads((Path(arm['evaluation']['output_root']) / 'manifest.json').read_text())
        assert len(saved['jobs']) == 4
        assert all(j['job_id'] for j in saved['jobs'])
        assert saved['aggregator']['job_id']
    before = len(calls)
    with pytest.raises(FileExistsError):
        sweep.submit_plan(p, run=run)
    assert len(calls) == before


def test_rejection_stops_submission_and_preserves_accepted_ids(submission_plan, monkeypatch):
    p, calls = submission_plan, []
    monkeypatch.setattr(sweep.train, 'require_finished_teacher', lambda cfg: None)
    with pytest.raises(RuntimeError, match='mock rejection'):
        sweep.submit_plan(p, run=cluster_mock(p, calls, fail_at=3))
    saved = json.loads((Path(p['output_root']) / 'manifest.json').read_text())
    assert saved['status'] == 'partial_failure'
    assert [j['submission_state'] for j in saved['jobs'][:4]] == ['accepted', 'accepted', 'rejected', 'planned']
    assert len([c for c in calls if c[0] == 'sbatch']) == 3
