"""CPU-only checks for BoN sample-count planning; no cluster commands."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts' / 'robocasa'
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location('bon_submission_count_test', SCRIPTS / 'submit_bon_evaluations.py')
    bon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bon)
finally:
    sys.path.pop(0)


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(bon.common, 'USER_ROOT', tmp_path)
    for name in ('bc2', 'critic'):
        checkpoint = tmp_path / name
        (checkpoint / 'experiment_cfg').mkdir(parents=True)
        (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 10000}))
        (checkpoint / 'config.json').write_text('{}')
        (checkpoint / 'experiment_cfg' / 'metadata.json').write_text('{"new_embodiment": {}}')
        (checkpoint / 'model.safetensors').write_bytes(b'fake checkpoint; never loaded')
    return argparse.Namespace(actor=tmp_path / 'bc2', critic=tmp_path / 'critic',
                              output_root=tmp_path / 'evaluation', eval_seeds=[0, 1, 2],
                              tasks=list(bon.common.TASKS), episodes=50,
                              execute_horizon=16, time='12:00:00')


def assert_count(plan, expected, report_to='wandb'):
    assert plan['config']['num_samples'] == expected
    assert plan['config']['report_to'] == report_to
    assert plan['config']['wandb_group'].startswith(f'robocasa-bon{expected}-')
    assert len(plan['jobs']) == 12
    assert sum(job['expected_episodes'] for job in plan['jobs']) == 600
    for job in plan['jobs']:
        assert f'-bon{expected}-exec16' in job['key']
        assert job['command'][-1] == report_to
        assert job['command'][-2] == str(expected)
        assert job['command'][-3] == '16'
        assert f'--comment={plan["config"]["wandb_group"]}:{job["key"]}' in job['command']
        assert job['job_id'] is None
        assert job['submission_state'] == 'planned'


def test_requested_50_reaches_worker_and_manifest(args):
    args.num_samples = 50
    assert_count(bon.build_plan(args), 50)
    assert not args.output_root.exists()


def test_legacy_namespace_defaults_to_10(args):
    assert not hasattr(args, 'num_samples')
    assert_count(bon.build_plan(args), 10)


@pytest.mark.parametrize('invalid', [0, -1, True, False, 1.5, '50', None])
def test_nonpositive_or_noninteger_counts_rejected(args, invalid):
    args.num_samples = invalid
    with pytest.raises(ValueError, match='positive integer'):
        bon.build_plan(args)


@pytest.mark.parametrize('walltime', ['00:00:00', '1:60:00', 'bad'])
def test_invalid_walltime_rejected(args, walltime):
    args.time = walltime
    with pytest.raises(ValueError, match='positive HH:MM:SS'):
        bon.build_plan(args)


@pytest.mark.parametrize('count_args, expected', [([], 10), (['--num-samples', '50'], 50)])
def test_cli_default_and_requested_value(args, monkeypatch, capsys, count_args, expected):
    monkeypatch.setattr(sys, 'argv', [str(SCRIPTS / 'submit_bon_evaluations.py'),
                                    '--actor', str(args.actor), '--critic', str(args.critic),
                                    '--output-root', str(args.output_root), '--time', args.time,
                                    *count_args])
    bon.main()
    assert_count(json.loads(capsys.readouterr().out), expected)
    assert not args.output_root.exists()


def test_worker_accepts_legacy_nine_args_and_forwards_count():
    worker = (REPO / 'slurm' / 'robocasa_bon_eval.sbatch').read_text()
    assert '( $# -eq 9 || $# -eq 10 || $# -eq 11 )' in worker
    assert 'NUM_SAMPLES=${10:-10}' in worker
    assert '[[ $NUM_SAMPLES =~ ^[1-9][0-9]*$ ]] || exit 2' in worker
    assert '--num_samples "$NUM_SAMPLES"' in worker
    assert 'deas-bon${NUM_SAMPLES}-train42' in worker


def test_no_wandb_flag_reaches_manifest_and_worker(args):
    args.num_samples = 50
    args.report_to = 'none'
    assert_count(bon.build_plan(args), 50, report_to='none')


@pytest.mark.parametrize('invalid', ['offline', 'online', '', None, True])
def test_invalid_reporting_rejected(args, invalid):
    args.report_to = invalid
    with pytest.raises(ValueError, match='report_to must be none or wandb'):
        bon.build_plan(args)


def test_cli_no_external_reporting(args, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [str(SCRIPTS / 'submit_bon_evaluations.py'),
                                    '--actor', str(args.actor), '--critic', str(args.critic),
                                    '--output-root', str(args.output_root), '--time', args.time,
                                    '--num-samples', '50', '--report-to', 'none'])
    bon.main()
    assert_count(json.loads(capsys.readouterr().out), 50, report_to='none')
    assert not args.output_root.exists()


def test_worker_disables_wandb_for_none():
    worker = (REPO / 'slurm' / 'robocasa_bon_eval.sbatch').read_text()
    assert 'REPORT_TO=${11:-wandb}' in worker
    assert 'case "$REPORT_TO" in none|wandb)' in worker
    assert 'export WANDB_MODE=disabled' in worker
    assert '--report_to "$REPORT_TO"' in worker
