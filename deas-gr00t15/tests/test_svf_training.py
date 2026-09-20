"""CPU checks for distributed batch accounting, deterministic resume, and artifacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_svf.py"
spec = importlib.util.spec_from_file_location("svf_train_under_test", SCRIPT)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_four_rank_accumulation_preserves_global_128():
    config = train.batch_schedule(128, 4, 4)
    assert config["gradient_accumulation_steps"] == 8
    draws = [position for rank in range(4)
             for batch in train.MicrobatchSampler(start=0, stop=8, rank=rank, world_size=4, size=4)
             for position in batch]
    assert sorted(draws) == list(range(128))
    with pytest.raises(ValueError, match="divisible"):
        train.batch_schedule(130, 4, 4)


def test_resume_sampler_continues_without_repeating_or_skipping():
    kwargs = dict(rank=2, world_size=4, size=4)
    full = list(train.MicrobatchSampler(start=0, stop=32, **kwargs))
    before = list(train.MicrobatchSampler(start=0, stop=16, **kwargs))
    after = list(train.MicrobatchSampler(start=16, stop=32, **kwargs))
    assert before + after == full
    assert len(list(train.MicrobatchSampler(start=32, stop=32, **kwargs))) == 0


class AugmentedMixture:
    def __len__(self):
        return 7

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample_step(self, index):
        return self.epoch, index

    def __getitem__(self, index):
        return self.epoch, index, random.random(), np.random.rand(), torch.rand(1).item()


def test_worker_prefetch_and_resume_preserve_augmentation_and_model_rng():
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    expected_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    original = train.DeterministicSamples(AugmentedMixture(), seed=42)
    full = [original[i] for i in range(20)]
    # Simulate a restart and different prefetch order, with unrelated global seeds.
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    resumed = train.DeterministicSamples(AugmentedMixture(), seed=42)
    second = {i: resumed[i] for i in reversed(range(10, 20))}
    assert [second[i] for i in range(10, 20)] == full[10:]
    random.setstate(expected_rng[0])
    np.random.set_state(expected_rng[1])
    torch.random.set_rng_state(expected_rng[2])
    resumed[999]
    assert random.getstate() == expected_rng[0]
    assert np.array_equal(np.random.get_state()[1], expected_rng[1][1])
    assert torch.equal(torch.random.get_rng_state(), expected_rng[2])
    assert full[7][0:2] == (1, 0)


def test_resume_rejects_changed_batch_or_teacher():
    expected = {"critic": "/critic/final", "batch": train.batch_schedule(128, 4, 4)}
    train.assert_resume_compatible(dict(expected), expected)
    with pytest.raises(ValueError, match="critic"):
        train.assert_resume_compatible({**expected, "critic": "/other"}, expected)
    with pytest.raises(ValueError, match="batch"):
        train.assert_resume_compatible({**expected, "batch": train.batch_schedule(128, 4, 2)}, expected)


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = torch.nn.Linear(3, 2)

    def save_pretrained(self, destination, **kwargs):
        (Path(destination) / "config.json").write_text('{"model_type":"test"}')
        torch.save(kwargs.get("state_dict", self.state_dict()), Path(destination) / "weights.pt")


def test_checkpoint_has_optimizer_rank_rng_export_and_exact_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "experiment_cfg").mkdir(parents=True)
    metadata = '{"new_embodiment": {"statistics": {"opaque": 1.23456}}}\n'
    (source / "experiment_cfg/metadata.json").write_text(metadata)
    output = tmp_path / "training"
    output.mkdir()
    model = SimpleNamespace(actor=TinyActor(), soft_value=torch.nn.Linear(3, 1),
                            soft_value_initialization={"mode": "random", "copied_parameters": 0})
    value_config = {"loss_type": "mse", "num_bins": 1, "value_min": None, "value_max": None, "sigma": None}
    model.soft_value.loss_configuration = lambda: dict(value_config)
    params = list(model.actor.parameters()) + list(model.soft_value.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    loss = model.actor.action_head(torch.ones(2, 3)).square().sum() + model.soft_value(torch.ones(2, 3)).square().sum()
    loss.backward()
    optimizer.step()
    monkeypatch.setattr(train, "capture_rng", lambda: {"rank_marker": "correct"})
    identity = {"actor": str(source), "svf": {"K": 8}, "batch": train.batch_schedule(128, 4, 4)}
    path = train.save_checkpoint(model, optimizer, output=output, step=10,
        microbatches=80, identity=identity, rank=0, world_size=1, elapsed=12.5, wandb_id="run1")
    saved = torch.load(path / "training_state.pt", weights_only=False)
    assert saved["step"] == 10 and saved["microbatches"] == 80
    assert saved["identity"] == identity
    assert json.loads((path / "soft_value_initialization.json").read_text()) == model.soft_value_initialization
    assert json.loads((path / "soft_value_config.json").read_text()) == value_config
    assert saved["optimizer"]["state"]
    assert saved["wandb_run_id"] == "run1"
    assert torch.load(path / "rng-rank-0.pt", weights_only=False)["rank_marker"] == "correct"
    assert (path / "actor/experiment_cfg/metadata.json").read_text() == metadata
    assert (path / "complete.json").is_file()
    assert not (output / ".checkpoint-10.incomplete").exists()
    assert json.loads((output / "latest_checkpoint.json").read_text())["path"] == str(path)
    assert "teacher" not in saved and "reference_head" not in saved
    with pytest.raises(FileExistsError):
        train.save_checkpoint(model, optimizer, output=output, step=10,
            microbatches=80, identity=identity, rank=0, world_size=1, elapsed=12.5, wandb_id="run1")


def test_atomic_json_rejects_nan_without_overwriting_previous_status(tmp_path):
    path = tmp_path / "status.json"
    train.atomic_json(path, {"step": 5})
    with pytest.raises(ValueError):
        train.atomic_json(path, {"step": float("nan")})
    assert json.loads(path.read_text()) == {"step": 5}


def test_metadata_validation_fails_before_any_data_scan(tmp_path, monkeypatch):
    path = tmp_path / "dataset"
    (path / "meta").mkdir(parents=True)
    for name in ("modality.json", "info.json", "episodes.jsonl", "tasks.jsonl"):
        (path / "meta" / name).write_text('{}\n')
    monkeypatch.setattr(Path, "glob", lambda *a, **k: pytest.fail("Dataset tree must not be scanned"))
    with pytest.raises(ValueError, match="automatic statistics scans are disabled"):
        train.validate_dataset_metadata(path)
    values = {k: [0.0, 1.0] for k in ("min", "max", "mean", "std", "q01", "q99")}
    (path / "meta/stats.json").write_text(json.dumps({"observation.state": values}))
    assert train.validate_dataset_metadata(path)["observation.state"]["q01"] == [0.0, 1.0]
    del values["q99"]
    (path / "meta/stats.json").write_text(json.dumps({"observation.state": values}))
    with pytest.raises(ValueError, match="q99"):
        train.validate_dataset_metadata(path)


def _cli_args(*extra):
    return train.parser().parse_args([
        "--actor", "/actor", "--critic", "/critic", "--output", "/home/nas_main/dohyunlee/test-output",
        "--dataset-path", "/dataset", *extra])


def test_lora_structure_and_scaling_are_part_of_strict_resume_identity():
    args = _cli_args("--actor-tuning", "dit-lora", "--lora-rank", "4", "--lora-alpha", "8")
    identity = train.run_identity(args, train.batch_schedule(128, 4, 4))
    assert identity["actor_tuning"] == {"mode": "dit-lora", "rank": 4, "alpha": 8.0, "dropout": 0.0}
    train.assert_resume_compatible(identity, dict(identity))
    for key, value in (("mode", "full"), ("rank", 8), ("alpha", 4.0), ("dropout", 0.1)):
        changed = {**identity, "actor_tuning": {**identity["actor_tuning"], key: value}}
        with pytest.raises(ValueError, match="actor_tuning"):
            train.assert_resume_compatible(identity, changed)
    old_full = {key: value for key, value in identity.items() if key != "actor_tuning"}
    full = {**old_full, "actor_tuning": train.actor_tuning_values()}
    train.assert_resume_compatible(old_full, full)
    with pytest.raises(ValueError, match="actor_tuning"):
        train.assert_resume_compatible(old_full, identity)


@pytest.mark.parametrize("flag,value", [("--lora-rank", "0"), ("--lora-alpha", "nan"),
                                       ("--lora-alpha", "0"), ("--lora-dropout", "1"),
                                       ("--lora-dropout", "-0.1")])
def test_invalid_lora_configuration_fails_before_model_loading(flag, value):
    args = _cli_args(flag, value)
    with pytest.raises(ValueError, match="lora-"):
        train.validate_args(args, world_size=4)


def _tiny_pretrained_actor_class():
    from transformers import PretrainedConfig, PreTrainedModel

    class TinyPretrainedActor(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.action_head = torch.nn.Module()
            self.action_head.model = torch.nn.Module()
            block = torch.nn.Module()
            block.attn1 = torch.nn.Module()
            for name in ("to_q", "to_k", "to_v"):
                setattr(block.attn1, name, torch.nn.Linear(3, 3))
            block.attn1.to_out = torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.Dropout(0)])
            self.action_head.model.transformer_blocks = torch.nn.ModuleList([block])
            self.action_head.projector = torch.nn.Linear(3, 3)

    return TinyPretrainedActor


def test_lora_checkpoint_keeps_adapters_but_export_loads_as_plain_pretrained_actor(tmp_path, monkeypatch):
    from transformers import PretrainedConfig
    from gr00t.model.svf.lora import ActorTuningConfig, apply_dit_lora, is_lora_parameter

    actor_class = _tiny_pretrained_actor_class()
    actor = actor_class(PretrainedConfig())
    tuning = ActorTuningConfig(mode="dit-lora", rank=2, alpha=4.0)
    targets = apply_dit_lora(actor.action_head, tuning)
    # Exercise a nonzero merged update, not merely the identity initialization.
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if name.endswith("lora_B"):
                param.fill_(0.125)
    actor.eval()
    inputs = torch.randn(2, 3)
    expected = [actor.action_head.get_submodule(name)(inputs).detach() for name in targets]
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    source = tmp_path / "source"
    (source / "experiment_cfg").mkdir(parents=True)
    metadata = '{"new_embodiment": {"opaque": "unchanged metadata"}}\n'
    (source / "experiment_cfg/metadata.json").write_text(metadata)
    output = tmp_path / "training"
    output.mkdir()
    model = SimpleNamespace(actor=actor, soft_value=torch.nn.Linear(3, 1), actor_tuning=tuning,
                            lora_target_modules=targets)
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad])
    monkeypatch.setattr(train, "capture_rng", lambda: {"mock": "rank-zero"})
    identity = {"actor": str(source), "svf": {"K": 8},
                "actor_tuning": {"mode": "dit-lora", "rank": 2, "alpha": 4.0, "dropout": 0.0}}
    path = train.save_checkpoint(model, optimizer, output=output, step=1, microbatches=1,
        identity=identity, rank=0, world_size=1, elapsed=0.0, wandb_id=None)
    saved = torch.load(path / "training_state.pt", weights_only=False)
    assert any(is_lora_parameter(name) for name in saved["actor_head"])
    assert any(".base.weight" in name for name in saved["actor_head"])
    assert saved["identity"]["actor_tuning"] == identity["actor_tuning"]
    exported = actor_class.from_pretrained(path / "actor", local_files_only=True).eval()
    assert not any(is_lora_parameter(name) or ".base." in name for name in exported.state_dict())
    for name, value in zip(targets, expected):
        torch.testing.assert_close(exported.action_head.get_submodule(name)(inputs), value)
    assert set(actor.state_dict()) == set(before)
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert (path / "actor/experiment_cfg/metadata.json").read_text() == metadata
    provenance = json.loads((path / "actor/svf_actor_tuning.json").read_text())
    assert provenance["format"] == "merged_vanilla_gr00t"
    assert provenance["target_modules"] == targets
    assert json.loads((path / "actor_tuning.json").read_text())["rank"] == 2
    resumed = actor_class(PretrainedConfig())
    apply_dit_lora(resumed.action_head, tuning)
    resumed.action_head.load_state_dict(saved["actor_head"], strict=True)
    resumed.eval()
    for name, value in zip(targets, expected):
        torch.testing.assert_close(resumed.action_head.get_submodule(name)(inputs), value)


def test_soft_value_initialization_is_part_of_strict_resume_identity():
    schedule = train.batch_schedule(128, 4, 4)
    random_init = train.run_identity(_cli_args(), schedule)
    critic_init = train.run_identity(_cli_args("--soft-value-init", "critic-trunk"), schedule)
    assert random_init["soft_value_init"] == "random"
    assert critic_init["soft_value_init"] == "critic-trunk"
    train.assert_resume_compatible(critic_init, dict(critic_init))
    for saved, expected in ((random_init, critic_init), (critic_init, random_init)):
        with pytest.raises(ValueError, match="soft_value_init"):
            train.assert_resume_compatible(saved, expected)
    legacy = {key: value for key, value in random_init.items() if key != "soft_value_init"}
    train.assert_resume_compatible(legacy, random_init)
    with pytest.raises(ValueError, match="soft_value_init"):
        train.assert_resume_compatible(legacy, critic_init)


def test_invalid_soft_value_initialization_is_rejected_by_cli():
    with pytest.raises(SystemExit):
        _cli_args("--soft-value-init", "unsupported")


def test_soft_value_loss_is_separate_from_initialization_and_strict_on_resume():
    schedule = train.batch_schedule(128, 4, 4)
    mse = train.run_identity(_cli_args(), schedule)
    distributional = train.run_identity(_cli_args("--soft-value-loss", "hl-gauss"), schedule)
    assert mse["soft_value_loss"] == "mse"
    assert distributional["soft_value_loss"] == "hl-gauss"
    assert distributional["soft_value_init"] == "random"
    full = train.run_identity(_cli_args("--soft-value-loss", "hl-gauss", "--soft-value-init", "critic-full"), schedule)
    assert full["soft_value_init"] == "critic-full"
    train.assert_resume_compatible(distributional, dict(distributional))
    for saved, expected in ((mse, distributional), (distributional, mse)):
        with pytest.raises(ValueError, match="soft_value_loss"):
            train.assert_resume_compatible(saved, expected)
    legacy = {key: value for key, value in mse.items() if key != "soft_value_loss"}
    train.assert_resume_compatible(legacy, mse)
    with pytest.raises(ValueError, match="soft_value_loss"):
        train.assert_resume_compatible(legacy, distributional)
    with pytest.raises(ValueError, match="soft_value_init"):
        train.assert_resume_compatible(distributional, full)


def test_critic_full_requires_distributional_loss_before_dataset_or_model_loading():
    args = _cli_args("--soft-value-init", "critic-full")
    with pytest.raises(ValueError, match="requires.*hl-gauss"):
        train.validate_args(args, world_size=4)
    with pytest.raises(SystemExit):
        _cli_args("--soft-value-loss", "unsupported")


def _hl_gauss_configuration():
    return {"loss_type": "hl-gauss", "num_bins": 101,
            "value_min": -100.0, "value_max": 0.0, "sigma": 0.7425742574257426}


@pytest.mark.parametrize("key,value", [("num_bins", 51), ("value_min", -99.0),
                                       ("value_max", 1.0), ("sigma", 0.5),
                                       ("loss_type", "mse")])
def test_soft_value_resume_rejects_changed_distribution_support(tmp_path, key, value):
    saved = _hl_gauss_configuration()
    (tmp_path / "soft_value_config.json").write_text(json.dumps(saved))
    train.assert_soft_value_resume_compatible(tmp_path, dict(saved))
    with pytest.raises(ValueError, match=key):
        train.assert_soft_value_resume_compatible(tmp_path, {**saved, key: value})


def test_soft_value_resume_only_allows_legacy_missing_config_for_mse(tmp_path):
    train.assert_soft_value_resume_compatible(tmp_path, {"loss_type": "mse"})
    with pytest.raises(ValueError, match="requires.*soft_value_config"):
        train.assert_soft_value_resume_compatible(tmp_path, _hl_gauss_configuration())
    (tmp_path / "soft_value_config.json").write_text('{"loss_type":"mse"}')
    train.assert_soft_value_resume_compatible(tmp_path, {"loss_type": "mse"})
    with pytest.raises(ValueError, match="soft-value configuration differs"):
        train.assert_soft_value_resume_compatible(tmp_path, _hl_gauss_configuration())


def test_soft_value_resume_rejects_invalid_config_object(tmp_path):
    (tmp_path / "soft_value_config.json").write_text('[]')
    with pytest.raises(ValueError, match="expected an object"):
        train.assert_soft_value_resume_compatible(tmp_path, {"loss_type": "mse"})
