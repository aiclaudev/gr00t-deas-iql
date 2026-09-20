"""CPU-only checks for fair GPU comparison and worker-only benchmark gating."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_deas_throughput.py"
spec = importlib.util.spec_from_file_location("deas_benchmark_under_test", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_layout_equal_batch_workers_and_prefetch_budget():
    one = bench.layout(128, 16, 1)
    four = bench.layout(128, 16, 4)
    assert (one["batch_per_rank"], four["batch_per_rank"]) == (128, 32)
    assert (one["workers_per_rank"], four["workers_per_rank"]) == (16, 4)
    assert one["prefetched_samples_limit"] == four["prefetched_samples_limit"] == 2048
    for config in (one, four):
        assert config["world_size"] * config["batch_per_rank"] == 128
        assert config["workers_per_rank"] * config["world_size"] == 16
        assert config["workers_per_rank"] * config["world_size"] * config["prefetch_factor"] * config["batch_per_rank"] == 2048
    assert bench.layout(128, 0, 1)["prefetch_factor"] is None
    with pytest.raises(ValueError, match="divisible"):
        bench.layout(127, 16, 4)
    with pytest.raises(ValueError, match="global-workers"):
        bench.layout(128, 15, 4)
    with pytest.raises(ValueError, match="exactly"):
        bench.layout(128, 16, 2)


def test_same_global_draw_positions_for_one_and_four_gpus():
    for step in (0, 4, 5, 24):
        single = list(bench.MicrobatchSampler(start=step, stop=step + 1, rank=0, world_size=1, size=128))[0]
        distributed = [draw for rank in range(4) for batch in bench.MicrobatchSampler(
            start=step, stop=step + 1, rank=rank, world_size=4, size=32) for draw in batch]
        assert single == distributed == list(range(step * 128, (step + 1) * 128))


def test_login_gate_runs_before_path_validation_or_heavy_imports(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bench, "validate_args", lambda *a: pytest.fail("Must reject login first"))
    with pytest.raises(RuntimeError, match="login benchmarking is prohibited"):
        bench.main(["--checkpoint", "/not-loaded", "--output", "/not-created", "--dataset-path", "/not-read"])


def test_summary_uses_elapsed_interval_not_sum_or_average_of_ranks():
    reports = []
    for rank in range(4):
        reports.append({"rank": rank, "measured_elapsed_seconds": 8 + rank,
            "peak_allocated_bytes": 100 + rank, "peak_reserved_bytes": 200 + rank,
            "steps": [{"wall_seconds": 1 + rank / 10, "data_wait_seconds": 0.1,
                       "cuda_update_milliseconds": 900 + rank} for _ in range(2)]})
    result = bench.summarize(reports, global_batch=128)
    assert result["updates_per_second"] == 2 / 11
    assert result["samples_per_second"] == 256 / 11
    assert result["peak_allocated_bytes_max_rank"] == 103
    assert result["peak_reserved_bytes_max_rank"] == 203
    assert result["step_seconds_median_max_rank"] == 1.3
    assert result["estimated_10000_updates_gpu_hours"] == result["estimated_10000_updates_wall_hours"] * 4
    assert "not pure compute" in result["timing_notes"]
    reports[0]["steps"].append(dict(reports[0]["steps"][0]))
    with pytest.raises(ValueError, match="same measured"):
        bench.summarize(reports, global_batch=128)


def test_percentile_interpolates_and_rejects_empty_input():
    assert bench.percentile([4, 1, 3, 2], 0.9) == pytest.approx(3.7)
    with pytest.raises(ValueError, match="empty"):
        bench.percentile([], 0.9)


def test_validation_rejects_missing_checkpoint_before_dataset_scan(tmp_path, monkeypatch):
    args = bench.parser().parse_args(["--checkpoint", str(tmp_path / "missing"),
        "--output", "/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/unused-benchmark-test",
        "--dataset-path", "/not-scanned", "--validate-config"])
    monkeypatch.setattr(bench, "validate_dataset_metadata", lambda *a: pytest.fail("No dataset scan"))
    with pytest.raises(ValueError, match="Missing checkpoint artifact"):
        bench.validate_args(args, 1)


def test_validation_only_does_not_need_slurm_or_torch(monkeypatch, capsys):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bench, "validate_args", lambda args, world: bench.layout(128, 16, world))
    bench.main(["--checkpoint", "/mock-checkpoint", "--output", "/mock-output",
                "--dataset-path", "/mock-data", "--validate-config", "--world-size", "4"])
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["layout"]["batch_per_rank"] == 32


def test_saved_precision_restoration_uses_header_and_preserves_head_fp32(tmp_path):
    import struct
    import torch
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    model.critic_head = torch.nn.Linear(2, 1).bfloat16()
    header = {name: {"dtype": "BF16" if name.startswith("backbone.") else "F32",
                     "shape": list(value.shape), "data_offsets": [0, 0]}
              for name, value in model.state_dict().items()}
    encoded = json.dumps(header).encode()
    # A body is unnecessary: the helper must only inspect the small header.
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    precision = bench.restore_saved_precision(model, tmp_path)
    assert model.backbone.weight.dtype == torch.bfloat16
    assert model.critic_head.weight.dtype == torch.float32
    assert precision["optimizer_state"] == "float32"
    header["backbone.weight"]["dtype"] = "F32"
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    with pytest.raises(ValueError, match="Unexpected saved precision"):
        bench.restore_saved_precision(model, tmp_path)


def _minimal_rl_checkpoint_metadata():
    def stats(width, offset):
        return {"min": [offset] * width, "max": [offset + 2.0] * width,
                "mean": [offset + 1.0] * width, "std": [0.5] * width,
                "q01": [offset + 0.1] * width, "q99": [offset + 1.9] * width}
    state = {"eef": {"absolute": True, "rotation_type": None,
                      "shape": [3], "continuous": True}}
    video = {"front": {"resolution": [256, 256], "channels": 3, "fps": 20.0}}
    return {"new_embodiment": {
        "embodiment_tag": "new_embodiment",
        "statistics": {"state": {"eef": stats(3, -3.0)},
                       "action": {"control": stats(2, -1.0)}},
        "modalities": {
            "state": state, "next_state": state,
            "video": video, "next_video": video,
            "action": {"control": {"absolute": False, "rotation_type": None,
                                      "shape": [2], "continuous": True}},
            "reward": {"shape": [0, 1], "dtype": "float64"},
            "done": {"shape": [0, 1], "dtype": "bool"},
        },
    }}


def _write_rl_checkpoint_metadata(checkpoint, metadata):
    cfg = checkpoint / "experiment_cfg"
    cfg.mkdir(parents=True)
    (cfg / "metadata.json").write_text(json.dumps(metadata))


def test_critic_metadata_retains_rl_fields_and_pinned_normalization(tmp_path):
    from gr00t.data.schema import RLDatasetMetadata
    from gr00t.data.transform.concat import RLConcatTransform

    checkpoint = tmp_path / "checkpoint"
    source = _minimal_rl_checkpoint_metadata()
    _write_rl_checkpoint_metadata(checkpoint, source)
    pinned = bench.load_critic_metadata(checkpoint)
    assert isinstance(pinned, RLDatasetMetadata)
    assert pinned.model_dump(mode="json") == source["new_embodiment"]
    # Exercise the exact metadata consumer that failed, without video data,
    # model weights, a pretrained processor, or any GPU operations.
    transform = RLConcatTransform(
        video_concat_order=["video.front"], next_video_concat_order=["next_video.front"],
        state_concat_order=["state.eef"], next_state_concat_order=["next_state.eef"],
        action_concat_order=["action.control"],
    )
    transform.set_metadata(pinned)
    assert transform.next_state_dims == {"next_state.eef": 3}
    assert transform.get_modality_metadata("next_state.eef").shape == (3,)
    assert pinned.modalities.next_video["front"].channels == 3
    assert pinned.modalities.reward.dtype == "float64"
    assert pinned.modalities.done.dtype == "bool"
    assert pinned.statistics.state["eef"].min.tolist() == [-3.0] * 3
    assert pinned.statistics.action["control"].max.tolist() == [1.0] * 2


@pytest.mark.parametrize("missing", ["next_state", "next_video", "reward", "done"])
def test_preflight_rejects_missing_rl_modality_before_dataset_access(tmp_path, monkeypatch, missing):
    checkpoint = tmp_path / "checkpoint"
    source = _minimal_rl_checkpoint_metadata()
    del source["new_embodiment"]["modalities"][missing]
    _write_rl_checkpoint_metadata(checkpoint, source)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "gr00t_n1_5_deass_critic"}))
    # Empty placeholders are enough: validation must reject metadata before
    # loading weights/optimizer state or enumerating any dataset.
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "optimizer.pt").touch()
    monkeypatch.setattr(bench, "validate_dataset_metadata", lambda *a: pytest.fail("No dataset access"))
    args = bench.parser().parse_args([
        "--checkpoint", str(checkpoint),
        "--output", "/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/unused-benchmark-test",
        "--dataset-path", "/not-scanned", "--validate-config",
    ])
    with pytest.raises(ValueError, match=missing):
        bench.validate_args(args, 1)


def _tiny_optimizer_model(*, tied):
    import torch
    from torch import nn

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(5, 3)
            self.lm_head = nn.Linear(3, 5, bias=False)
            if tied:
                self.tie_weights()
            self.requires_grad_(False)

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def tie_weights(self):
            self.lm_head.weight = self.model.embed_tokens.weight

    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.eagle_model = nn.Module()
    model.backbone.eagle_model.language_model = LanguageModel()
    model.critic_head = nn.Linear(3, 2)
    return model


def _save_tiny_optimizer_checkpoint(path):
    import struct
    import torch
    original = _tiny_optimizer_model(tied=True)
    optimizer = torch.optim.Adam(original.parameters(), lr=0.004, betas=(0.85, 0.98), eps=3e-7)
    original.critic_head(torch.tensor([[1., 2., 3.]])).square().sum().backward()
    optimizer.step()
    saved = optimizer.state_dict()
    torch.save(saved, path / "optimizer.pt")
    header = {name: {"dtype": "F32", "shape": list(value.shape), "data_offsets": [0, 0]}
              for name, value in original.named_parameters()}
    encoded = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    return saved


def test_optimizer_restores_verified_frozen_embedding_alias_and_moments(tmp_path):
    import torch
    saved = _save_tiny_optimizer_checkpoint(tmp_path)
    restored = _tiny_optimizer_model(tied=False)
    assert len(list(restored.parameters())) == len(saved["param_groups"][0]["params"]) + 1
    optimizer, report = bench.restore_checkpoint_optimizer(restored, tmp_path)
    lm = restored.backbone.eagle_model.language_model
    assert lm.lm_head.weight is lm.model.embed_tokens.weight
    assert not lm.lm_head.weight.requires_grad
    assert report == {"parameter_count": 3, "state_count": 2, "restored_frozen_embedding_alias": True}
    result = optimizer.state_dict()
    assert result["param_groups"] == saved["param_groups"]
    assert set(result["state"]) == set(saved["state"])
    for pid, state in result["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, saved["state"][pid][name], rtol=0, atol=0)
    # An already-tied model remains valid and does not claim a repair.
    _, second_report = bench.restore_checkpoint_optimizer(_tiny_optimizer_model(tied=True), tmp_path)
    assert second_report["restored_frozen_embedding_alias"] is False


@pytest.mark.parametrize("mismatch", ["extra_parameter", "tie_disabled", "trainable_embedding"])
def test_optimizer_refuses_unverified_layout_or_alias_repair(tmp_path, mismatch):
    import torch
    _save_tiny_optimizer_checkpoint(tmp_path)
    model = _tiny_optimizer_model(tied=False)
    lm = model.backbone.eagle_model.language_model
    if mismatch == "extra_parameter":
        model.extra_parameter = torch.nn.Parameter(torch.ones(2))
    elif mismatch == "tie_disabled":
        lm.config.tie_word_embeddings = False
    else:
        lm.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="verified embedding alias repair|compatible frozen"):
        bench.restore_checkpoint_optimizer(model, tmp_path)
    assert lm.lm_head.weight is not lm.model.embed_tokens.weight


def test_optimizer_refuses_moment_shape_mismatch_before_loading(tmp_path):
    import torch
    saved = _save_tiny_optimizer_checkpoint(tmp_path)
    pid = next(iter(saved["state"]))
    saved["state"][pid]["exp_avg"] = torch.zeros(9)
    torch.save(saved, tmp_path / "optimizer.pt")
    with pytest.raises(ValueError, match="exp_avg mismatch"):
        bench.restore_checkpoint_optimizer(_tiny_optimizer_model(tied=True), tmp_path)
