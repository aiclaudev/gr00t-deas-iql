"""SVF evaluation planning and completion checks stay CPU-only and fail closed."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = load_script('svf_eval_submit_test', 'scripts/robocasa/submit_evaluations.py')
validator = load_script('svf_eval_checkpoint_test', 'scripts/robocasa/validate_svf_checkpoint.py')


def planned_args(tmp_path):
    return submit.arguments(['--actor', str(tmp_path / 'train/checkpoint-5000/actor'),
                             '--seed', '42', '--output-root', str(tmp_path / 'evaluation')])


def test_existing_actor_cli_still_rejects_future_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, 'USER_ROOT', tmp_path)
    with pytest.raises(ValueError, match='Missing actor file'):
        submit.build_plan(planned_args(tmp_path))


def test_future_checkpoint_planning_is_explicit_and_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, 'USER_ROOT', tmp_path)
    args = planned_args(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Planning may not create directories or run cluster commands')
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    monkeypatch.setattr(submit.subprocess, 'run', forbidden)
    plan = submit.build_plan(args, allow_future_actor=True)
    assert len(plan['jobs']) == 4
    assert {job['task'] for job in plan['jobs']} == set(submit.TASKS)
    assert all(job['actor'] == str(args.actor) for job in plan['jobs'])
    assert all(job['job_id'] is None for job in plan['jobs'])
    assert not args.output_root.exists()
    assert not args.actor.exists()


def test_no_public_future_checkpoint_switch(tmp_path):
    with pytest.raises(SystemExit):
        submit.arguments(['--actor', str(tmp_path / 'actor'), '--seed', '42',
                          '--output-root', str(tmp_path / 'eval'), '--allow-future-actor'])


def completed(tmp_path, monkeypatch, *, step=5000):
    monkeypatch.setattr(validator, 'USER_ROOT', tmp_path)
    checkpoint = tmp_path / 'train/checkpoint-5000'
    actor = checkpoint / 'actor'
    actor.mkdir(parents=True)
    (checkpoint / 'complete.json').write_text(json.dumps({'step': step, 'world_size': 4,
                                                         'microbatches': 40000}))
    return checkpoint, actor


def test_completed_checkpoint_validates_without_torch_weights(tmp_path, monkeypatch):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    assert validator.validate_checkpoint(checkpoint, 5000, actor) == (checkpoint, actor)
    assert not (actor / 'trainer_state.json').exists()
    assert not (checkpoint / 'training_state.pt').exists()


def test_missing_completion_marker_rejected(tmp_path, monkeypatch):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    (checkpoint / 'complete.json').unlink()
    with pytest.raises(ValueError, match='completion marker is missing'):
        validator.validate_checkpoint(checkpoint, 5000, actor)


@pytest.mark.parametrize('marker_step', [4999, 10000, 0])
def test_wrong_completed_step_rejected(tmp_path, monkeypatch, marker_step):
    checkpoint, actor = completed(tmp_path, monkeypatch, step=marker_step)
    with pytest.raises(ValueError, match='Expected SVF step 5000'):
        validator.validate_checkpoint(checkpoint, 5000, actor)


@pytest.mark.parametrize('marker', [[], {}, {'step': '5000'}, {'step': True}])
def test_invalid_completion_marker_rejected(tmp_path, monkeypatch, marker):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    (checkpoint / 'complete.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='integer step'):
        validator.validate_checkpoint(checkpoint, 5000, actor)


@pytest.mark.parametrize('step', [0, -1, True, '5000'])
def test_invalid_expected_step_rejected(tmp_path, monkeypatch, step):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='positive integer'):
        validator.validate_checkpoint(checkpoint, step, actor)


def test_different_actor_rejected(tmp_path, monkeypatch):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    other_actor = tmp_path / 'other-actor'
    other_actor.mkdir()
    with pytest.raises(ValueError, match='exactly CHECKPOINT/actor'):
        validator.validate_checkpoint(checkpoint, 5000, other_actor)


def test_actor_symlink_to_unrelated_export_rejected(tmp_path, monkeypatch):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    actor.rmdir()
    unrelated = tmp_path / 'unrelated-actor'
    unrelated.mkdir()
    actor.symlink_to(unrelated, target_is_directory=True)
    with pytest.raises(ValueError, match='exactly CHECKPOINT/actor'):
        validator.validate_checkpoint(checkpoint, 5000, actor)


def test_incomplete_directory_rejected(tmp_path, monkeypatch):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    partial = checkpoint.with_name('.checkpoint-5000.incomplete')
    checkpoint.rename(partial)
    with pytest.raises(ValueError, match='incomplete checkpoint'):
        validator.validate_checkpoint(partial, 5000, partial / 'actor')


def test_checkpoint_outside_personal_scope_rejected(tmp_path, monkeypatch):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    monkeypatch.setattr(validator, 'USER_ROOT', tmp_path / 'other-home')
    with pytest.raises(ValueError, match='must stay inside'):
        validator.validate_checkpoint(checkpoint, 5000, actor)


def test_symlinked_completion_marker_rejected(tmp_path, monkeypatch):
    checkpoint, actor = completed(tmp_path, monkeypatch)
    marker = checkpoint / 'complete.json'
    alternative = tmp_path / 'copied-marker.json'
    marker.rename(alternative)
    marker.symlink_to(alternative)
    with pytest.raises(ValueError, match='must not be a symlink'):
        validator.validate_checkpoint(checkpoint, 5000, actor)


def test_wrapper_rejects_login_execution_without_importing_evaluator():
    wrapper = REPO / 'slurm/svf_robocasa_eval.sbatch'
    result = subprocess.run(['bash', str(wrapper)], env={'PATH': '/usr/bin:/bin'},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'in an sbatch worker' in result.stderr


def test_wrapper_requires_svf_marker_step_and_disables_legacy_step_check():
    wrapper = REPO / 'slurm/svf_robocasa_eval.sbatch'
    args = ['checkpoint', '5000'] + ['placeholder'] * 14
    result = subprocess.run(['bash', str(wrapper), *args],
                            env={'PATH': '/usr/bin:/bin', 'SLURM_JOB_ID': '12345'},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'EXPECTED_TRAINING_STEPS must be 0' in result.stderr
