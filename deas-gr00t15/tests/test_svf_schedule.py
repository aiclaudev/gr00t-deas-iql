"""CPU regression: full-horizon actor LR and exact checkpoint continuation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_svf.py"
spec = importlib.util.spec_from_file_location("svf_schedule_under_test", SCRIPT)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def args(*extra):
    return train.parser().parse_args([
        "--actor", "/actor", "--critic", "/critic", "--dataset-path", "/dataset",
        "--output", "/home/nas_main/dohyunlee/svf-schedule-test",
        "--steps", "10000", "--lr-scheduler", "cosine", *extra])


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = torch.nn.Linear(2, 1)

    def save_pretrained(self, destination, **kwargs):
        torch.save(self.state_dict(), Path(destination) / "weights.pt")


def components(config):
    model = SimpleNamespace(actor=Actor(), soft_value=torch.nn.Linear(2, 1))
    optimizer = torch.optim.AdamW([
        {"params": list(model.actor.parameters()), "name": "actor", "lr": config.actor_lr},
        {"params": list(model.soft_value.parameters()), "name": "soft_value", "lr": config.value_lr},
    ], weight_decay=config.weight_decay)
    return model, optimizer, train.build_lr_scheduler(optimizer, config)


def advance(optimizer, scheduler, count):
    history = []
    for _ in range(count):
        history.append(tuple(group["lr"] for group in optimizer.param_groups))
        # Parameter-dependent stochastic gradients make missing optimizer/RNG
        # restoration observable without loading GR00T or using a GPU.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = parameter.detach() * 0.1 + torch.randn_like(parameter)
        optimizer.step()
        scheduler.step()
    return history


def test_actor_only_cosine_boundaries_and_stop_is_not_schedule_horizon():
    config = args("--stop-after-steps", "5000")
    schedule = train.lr_schedule_values(config)
    assert train.actor_lr_multiplier(0, total_steps=10000, config=schedule) == 1
    assert train.actor_lr_multiplier(5000, total_steps=10000, config=schedule) == pytest.approx(0.5)
    assert train.actor_lr_multiplier(10000, total_steps=10000, config=schedule) == 0
    batch = train.batch_schedule(128, 4, 1)
    assert train.run_identity(config, batch) == train.run_identity(args(), batch)
    assert train.run_identity(config, batch)["lr_schedule"]["soft_value"] == "constant"


def test_10000_updates_equal_5000_checkpoint_plus_5000_resume(tmp_path, monkeypatch):
    config = args("--soft-value-init", "critic-trunk", "--soft-value-loss", "mse")
    source = tmp_path / "source"
    (source / "experiment_cfg").mkdir(parents=True)
    (source / "experiment_cfg/metadata.json").write_text('{"new_embodiment": {}}')
    config.actor = str(source)
    batch = train.batch_schedule(128, 4, 1)
    identity = train.run_identity(config, batch)
    torch.manual_seed(123)
    continuous, continuous_opt, continuous_schedule = components(config)
    full_history = advance(continuous_opt, continuous_schedule, 10000)

    torch.manual_seed(123)
    staged, staged_opt, staged_schedule = components(config)
    history = advance(staged_opt, staged_schedule, 5000)
    assert staged_schedule.get_last_lr() == pytest.approx([5e-6, 3e-4])
    output = tmp_path / "training"
    output.mkdir()
    monkeypatch.setattr(train, "capture_rng", lambda: {"torch": torch.random.get_rng_state()})
    checkpoint = train.save_checkpoint(
        staged, staged_opt, output=output, step=5000, microbatches=5000 * 32,
        identity=identity, rank=0, world_size=1, elapsed=0, wandb_id="same-run",
        scheduler=staged_schedule)
    saved = torch.load(checkpoint / "training_state.pt", weights_only=False)
    assert saved["lr_scheduler"]["last_epoch"] == 5000
    assert saved["microbatches"] == 160000
    assert saved["identity"]["steps"] == 10000
    assert saved["wandb_run_id"] == "same-run"

    torch.manual_seed(999)
    resumed, resumed_opt, resumed_schedule = components(config)
    train.assert_resume_compatible(saved["identity"], identity)
    resumed.actor.action_head.load_state_dict(saved["actor_head"])
    resumed.soft_value.load_state_dict(saved["soft_value"])
    train.restore_optimizer_and_scheduler(resumed_opt, resumed_schedule, saved, config)
    torch.random.set_rng_state(torch.load(checkpoint / "rng-rank-0.pt", weights_only=False)["torch"])
    history += advance(resumed_opt, resumed_schedule, 5000)

    assert history == full_history
    assert all(value_lr == config.value_lr for _, value_lr in history)
    assert resumed_schedule.last_epoch == continuous_schedule.last_epoch == 10000
    assert resumed_schedule.get_last_lr() == [0.0, config.value_lr]
    for original, restored in zip(
            list(continuous.actor.parameters()) + list(continuous.soft_value.parameters()),
            list(resumed.actor.parameters()) + list(resumed.soft_value.parameters())):
        assert torch.equal(original, restored)
    expected_opt, actual_opt = continuous_opt.state_dict(), resumed_opt.state_dict()
    assert expected_opt["param_groups"] == actual_opt["param_groups"]
    for pid, state in expected_opt["state"].items():
        for key, value in state.items():
            assert torch.equal(value, actual_opt["state"][pid][key])


@pytest.mark.parametrize("change", ["horizon", "kind", "warmup", "floor"])
def test_resume_rejects_changed_schedule(change):
    original = args()
    altered = {"horizon": ["--steps", "20000"], "kind": ["--lr-scheduler", "constant"],
               "warmup": ["--warmup-steps", "100"], "floor": ["--min-lr-ratio", "0.1"]}[change]
    batch = train.batch_schedule(128, 4, 1)
    with pytest.raises(ValueError, match="steps|lr_schedule"):
        train.assert_resume_compatible(train.run_identity(original, batch),
                                       train.run_identity(args(*altered), batch))


@pytest.mark.parametrize("damage", ["missing", "counter", "lr"])
def test_resume_rejects_missing_or_inconsistent_scheduler(damage):
    config = args()
    model, optimizer, scheduler = components(config)
    advance(optimizer, scheduler, 3)
    saved = {"step": 3, "optimizer": copy.deepcopy(optimizer.state_dict()),
             "lr_scheduler": copy.deepcopy(scheduler.state_dict())}
    if damage == "missing": saved.pop("lr_scheduler")
    if damage == "counter": saved["lr_scheduler"]["last_epoch"] = 0
    if damage == "lr": saved["optimizer"]["param_groups"][0]["lr"] = config.actor_lr
    _, new_optimizer, new_scheduler = components(config)
    with pytest.raises(ValueError, match="scheduler|LR"):
        train.restore_optimizer_and_scheduler(new_optimizer, new_scheduler, saved, config)


def test_legacy_fixed_lr_checkpoint_can_continue_without_scheduler_state():
    config = args("--lr-scheduler", "constant")
    _, optimizer, scheduler = components(config)
    advance(optimizer, scheduler, 3)
    saved = {"step": 3, "optimizer": copy.deepcopy(optimizer.state_dict())}
    _, new_optimizer, new_scheduler = components(config)
    train.restore_optimizer_and_scheduler(new_optimizer, new_scheduler, saved, config)
    assert new_scheduler.last_epoch == 3
    assert advance(new_optimizer, new_scheduler, 1) == [(config.actor_lr, config.value_lr)]
    identity = train.run_identity(config, train.batch_schedule(128, 4, 1))
    legacy = {key: value for key, value in identity.items() if key != "lr_schedule"}
    train.assert_resume_compatible(legacy, identity)


@pytest.mark.parametrize("flags", [
    ["--warmup-steps", "-1"], ["--warmup-steps", "10000"],
    ["--min-lr-ratio", "nan"], ["--min-lr-ratio", "1.1"],
    ["--lr-scheduler", "constant", "--warmup-steps", "100"],
])
def test_invalid_schedule_fails_before_dataset_or_model_loading(flags):
    with pytest.raises(ValueError, match="warmup|ratio|scheduler"):
        train.validate_args(args(*flags), world_size=1)
