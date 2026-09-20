"""Read-only planning and final-teacher gates must never submit accidentally."""
import importlib.util
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('svf_submission_test', REPO / 'scripts/submit_svf.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def config():
    return json.loads((REPO / 'configs/svf_joint_seed42.json').read_text())


def test_default_planner_neither_creates_files_nor_calls_external_commands(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Dry-run must not create directories or call external commands')
    monkeypatch.setattr(submit.subprocess, 'run', forbidden)
    monkeypatch.setattr(submit.subprocess, 'check_output', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    assert submit.main([]) == 0
    assert submit.main(['--preflight-steps', '20']) == 0


def test_submit_requires_explicit_walltime_before_external_calls(monkeypatch):
    monkeypatch.setattr(submit.subprocess, 'check_output', lambda *a, **k: pytest.fail('Unexpected external call'))
    with pytest.raises(SystemExit):
        submit.main(['--submit'])


def test_missing_final_critic_does_not_fall_back_to_intermediate(tmp_path, monkeypatch):
    cfg = config()
    cfg['critic'] = str(tmp_path / 'unfinished_critic')
    monkeypatch.setattr(submit.subprocess, 'check_output', lambda *a, **k: pytest.fail('Unexpected external call'))
    with pytest.raises(ValueError, match='Final critic is not ready'):
        submit.require_finished_teacher(cfg)


def test_saved_final_model_still_requires_completed_slurm_job(tmp_path, monkeypatch):
    cfg = config()
    cfg['critic'] = str(tmp_path)
    (tmp_path / 'experiment_cfg').mkdir()
    for name in ('config.json', 'experiment_cfg/metadata.json'):
        (tmp_path / name).write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'mock')
    (tmp_path / 'trainer_state.json').write_text('{"global_step":10000}')
    monkeypatch.setattr(submit.subprocess, 'check_output', lambda *a, **k: '136451|RUNNING|\n')
    with pytest.raises(ValueError, match='COMPLETED'):
        submit.require_finished_teacher(cfg)
    monkeypatch.setattr(submit.subprocess, 'check_output', lambda *a, **k: '136451|COMPLETED|\n')
    submit.require_finished_teacher(cfg)


def test_lora_config_and_legacy_full_default_reach_the_worker():
    cfg = config()
    args = submit.training_args(cfg, Path("/example/train"))
    assert args[args.index("--actor-tuning") + 1] == "full"
    lora = json.loads((REPO / "configs/svf_joint_seed42_lora.json").read_text())
    submit.validate_config(lora)
    args = submit.training_args(lora, Path("/example/train"))
    for flag, value in (("--actor-tuning", "dit-lora"), ("--lora-rank", "16"),
                        ("--lora-alpha", "16.0"), ("--lora-dropout", "0.0")):
        assert args[args.index(flag) + 1] == value
    assert lora["actor_lr"] == cfg["actor_lr"]


@pytest.mark.parametrize("key,value", [("actor_tuning", "wrong"), ("lora_rank", 0),
                                       ("lora_rank", 1.5), ("lora_alpha", float("inf")),
                                       ("lora_dropout", 1.0)])
def test_invalid_lora_config_is_rejected_before_submission(key, value):
    with pytest.raises(ValueError):
        submit.validate_config({**config(), key: value})


def test_soft_value_initialization_defaults_and_selection_reach_the_worker():
    cfg = config()
    assert cfg["soft_value_init"] == "random"
    cfg.pop("soft_value_init")
    submit.validate_config(cfg)
    args = submit.training_args(cfg, Path("/example/train"))
    assert args[args.index("--soft-value-init") + 1] == "random"
    cfg["soft_value_init"] = "critic-trunk"
    submit.validate_config(cfg)
    args = submit.training_args(cfg, Path("/example/train"))
    assert args[args.index("--soft-value-init") + 1] == "critic-trunk"
    with pytest.raises(ValueError, match="soft_value_init"):
        submit.validate_config({**cfg, "soft_value_init": "unsupported"})


def test_soft_value_initialization_cli_override_remains_a_dry_run(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not execute or create directories")
    monkeypatch.setattr(submit.subprocess, "run", forbidden)
    monkeypatch.setattr(submit.subprocess, "check_output", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert submit.main(["--soft-value-init", "critic-trunk"]) == 0
    assert "--soft-value-init critic-trunk" in capsys.readouterr().out


def test_soft_value_loss_defaults_and_allowed_initialization_combinations():
    cfg = config()
    assert cfg["soft_value_loss"] == "mse"
    cfg.pop("soft_value_loss")
    submit.validate_config(cfg)
    args = submit.training_args(cfg, Path("/example/train"))
    assert args[args.index("--soft-value-loss") + 1] == "mse"
    for loss, init in (("mse", "random"), ("mse", "critic-trunk"),
                       ("hl-gauss", "random"), ("hl-gauss", "critic-trunk"),
                       ("hl-gauss", "critic-full")):
        selected = {**cfg, "soft_value_loss": loss, "soft_value_init": init}
        submit.validate_config(selected)
        args = submit.training_args(selected, Path("/example/train"))
        assert args[args.index("--soft-value-loss") + 1] == loss
        assert args[args.index("--soft-value-init") + 1] == init
    with pytest.raises(ValueError, match="requires.*hl-gauss"):
        submit.validate_config({**cfg, "soft_value_init": "critic-full"})
    with pytest.raises(ValueError, match="soft_value_loss"):
        submit.validate_config({**cfg, "soft_value_loss": "unsupported"})


def test_soft_value_loss_cli_override_does_not_implicitly_initialize_from_critic(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not execute or create directories")
    monkeypatch.setattr(submit.subprocess, "run", forbidden)
    monkeypatch.setattr(submit.subprocess, "check_output", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert submit.main(["--soft-value-loss", "hl-gauss"]) == 0
    output = capsys.readouterr().out
    assert "--soft-value-loss hl-gauss" in output
    assert "--soft-value-init random" in output
    assert submit.main(["--soft-value-loss", "hl-gauss", "--soft-value-init", "critic-full"]) == 0
    assert "--soft-value-init critic-full" in capsys.readouterr().out


def test_legacy_configuration_keeps_full_horizon_constant_actor_lr_and_wandb():
    cfg = config()
    for key in ("lr_scheduler", "warmup_steps", "min_lr_ratio", "stop_after_steps"):
        cfg.pop(key, None)
    submit.validate_config(cfg)
    args = submit.training_args(cfg, Path("/example/train"))
    for flag, expected in (("--steps", "10000"), ("--lr-scheduler", "constant"),
                           ("--warmup-steps", "0"), ("--min-lr-ratio", "0.0"),
                           ("--report-to", "wandb")):
        assert args[args.index(flag) + 1] == expected
    assert "--stop-after-steps" not in args


def test_staged_actor_schedule_keeps_full_horizon_and_logging_on_resume():
    cfg = {**config(), "lr_scheduler": "cosine", "warmup_steps": 0,
           "min_lr_ratio": 0.0, "stop_after_steps": 5000}
    submit.validate_config(cfg)
    args = submit.training_args(cfg, Path("/example/train"),
                                resume=Path("/example/train/checkpoint-5000"))
    for flag, expected in (("--steps", "10000"), ("--stop-after-steps", "5000"),
                           ("--lr-scheduler", "cosine"), ("--warmup-steps", "0"),
                           ("--min-lr-ratio", "0.0"), ("--report-to", "wandb"),
                           ("--resume", "/example/train/checkpoint-5000")):
        assert args[args.index(flag) + 1] == expected
    assert args[args.index("--value-lr") + 1] == str(cfg["value_lr"])
    override = submit.training_args(cfg, Path("/example/train"), stop_after_steps=2500)
    assert override[override.index("--stop-after-steps") + 1] == "2500"
    assert cfg["steps"] == 10000 and cfg["stop_after_steps"] == 5000


def test_preflight_overrides_staged_config_but_keeps_original_horizon():
    cfg = {**config(), "lr_scheduler": "cosine", "stop_after_steps": 5000}
    args = submit.training_args(cfg, Path("/example/train"), preflight_steps=20)
    assert args[args.index("--stop-after-steps") + 1] == "20"
    assert args[args.index("--steps") + 1] == "10000"
    assert args[args.index("--report-to") + 1] == "none"
    assert args.count("--stop-after-steps") == 1
    assert cfg["stop_after_steps"] == 5000


@pytest.mark.parametrize("key,value", [
    ("lr_scheduler", "linear"), ("lr_scheduler", None),
    ("warmup_steps", -1), ("warmup_steps", 1.5), ("warmup_steps", True),
    ("warmup_steps", "0"), ("warmup_steps", 10000),
    ("min_lr_ratio", -0.1), ("min_lr_ratio", 1.1),
    ("min_lr_ratio", float("nan")), ("min_lr_ratio", float("inf")),
    ("min_lr_ratio", True), ("min_lr_ratio", "0"), ("min_lr_ratio", None),
    ("stop_after_steps", 0), ("stop_after_steps", -1),
    ("stop_after_steps", 1.5), ("stop_after_steps", True),
    ("stop_after_steps", "5000"),
])
def test_invalid_stage_and_schedule_configuration_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        submit.validate_config({**config(), key: value})


@pytest.mark.parametrize("kwargs", [
    {"preflight_steps": 20, "stop_after_steps": 5000},
    {"preflight_steps": 101}, {"preflight_steps": 0},
    {"preflight_steps": True}, {"stop_after_steps": 0},
    {"stop_after_steps": True}, {"stop_after_steps": 1.5},
])
def test_direct_training_argument_builder_rejects_unsafe_stop_options(kwargs):
    with pytest.raises(ValueError):
        submit.training_args(config(), Path("/example/train"), **kwargs)


def test_stage_cli_remains_read_only_and_can_override_config(tmp_path, monkeypatch, capsys):
    cfg = {**config(), "lr_scheduler": "cosine", "stop_after_steps": 5000}
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(cfg))
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not execute or create directories")
    monkeypatch.setattr(submit.subprocess, "run", forbidden)
    monkeypatch.setattr(submit.subprocess, "check_output", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert submit.main(["--config", str(path)]) == 0
    output = capsys.readouterr().out
    assert "--steps 10000" in output and "--stop-after-steps 5000" in output
    assert "--report-to wandb" in output and "--lr-scheduler cosine" in output
    assert submit.main(["--config", str(path), "--stop-after-steps", "2500"]) == 0
    output = capsys.readouterr().out
    assert "--steps 10000" in output and "--stop-after-steps 2500" in output
    assert "--report-to wandb" in output
    assert submit.main(["--config", str(path), "--preflight-steps", "20"]) == 0
    output = capsys.readouterr().out
    assert "--steps 10000" in output and "--stop-after-steps 20" in output
    assert "--report-to none" in output
    with pytest.raises(SystemExit):
        submit.main(["--preflight-steps", "20", "--stop-after-steps", "5000"])
    with pytest.raises(SystemExit):
        submit.main(["--preflight-steps", "101"])


@pytest.mark.parametrize("extra", [{"warmup_steps": 100}, {"min_lr_ratio": 0.1}])
def test_constant_schedule_rejects_cosine_only_options(extra):
    cfg = {**config(), "lr_scheduler": "constant", **extra}
    with pytest.raises(ValueError, match="require the cosine actor scheduler"):
        submit.validate_config(cfg)
    submit.validate_config({**cfg, "lr_scheduler": "cosine"})
