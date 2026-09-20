"""CPU-only checks for durable partial results and opt-in telemetry with a fake client."""
import csv
import json

import pytest

from gr00t.eval.results import EvaluationRecorder


def config():
    return {"actor_model_path": "/user/checkpoint", "critic_model_path": None,
            "env_name": "CoffeeSetupMug", "model_type": "gr00tn15", "n_envs": 1,
            "action_horizon": 16, "denoising_steps": 4, "noise": 0.0}


def episode(index, success=True):
    return {"episode": index, "env_index": 0, "success": success,
            "length": 16 + index, "elapsed_seconds": float(index + 1)}


def snapshot(path):
    return json.loads((path / "result.json").read_text())


class FakeRun:
    def __init__(self, fail_finish=False):
        self.logs = []
        self.summary = {}
        self.exit_codes = []
        self.fail_finish = fail_finish

    def log(self, metrics, step):
        self.logs.append((step, metrics))

    def finish(self, exit_code):
        self.exit_codes.append(exit_code)
        if self.fail_finish:
            raise RuntimeError("telemetry finish failed")


class FakeWandb:
    def __init__(self, fail_finish=False):
        self.run = FakeRun(fail_finish)
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def test_incremental_files_and_final_success_are_consistent(tmp_path):
    with EvaluationRecorder(tmp_path, 2, config=config(), protocol={"object_instance_split": "B"},
                            training_seed=42, evaluation_seed=123) as recorder:
        assert snapshot(tmp_path)["status"] == "running"
        recorder.record_episode(episode(0))
        partial = snapshot(tmp_path)
        assert partial["status"] == "running"
        assert partial["completed_episodes"] == 1
        assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 1
        with (tmp_path / "eval.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 1
        recorder.record_episode(episode(1, False))
    result = snapshot(tmp_path)
    assert result["status"] == "completed"
    assert result["expected_episodes"] == result["completed_episodes"] == 2
    assert result["success_count"] == 1
    assert result["success_rate"] == 0.5
    assert result["mean_episode_length"] == 16.5
    assert result["seeds"] == {"training": 42, "evaluation": 123}
    assert result["walltime_seconds"] >= 0
    assert result["finished_at_utc"] is not None
    assert (tmp_path / "success.txt").read_text() == "Success Rate: 0.5000\n"


def test_partial_failure_keeps_completed_episodes_and_original_error(tmp_path):
    client = FakeWandb(fail_finish=True)
    with pytest.warns(UserWarning, match="cleanup also failed"):
        with pytest.raises(ValueError, match="simulator failed"):
            with EvaluationRecorder(tmp_path, 3, config=config(), protocol={},
                                    report_to="wandb", wandb_client=client) as recorder:
                recorder.record_episode(episode(0))
                raise ValueError("simulator failed")
    result = snapshot(tmp_path)
    assert result["status"] == "failed"
    assert result["completed_episodes"] == 1
    assert result["error"] == {"type": "ValueError", "message": "simulator failed"}
    assert client.run.summary["eval/status"] == "failed"
    assert client.run.summary["eval/success_rate"] is None
    assert client.run.exit_codes == [1]
    assert not (tmp_path / "success.txt").exists()


def test_early_return_cannot_mark_partial_run_complete(tmp_path):
    with pytest.raises(RuntimeError, match="before all requested episodes"):
        with EvaluationRecorder(tmp_path, 2, config=config(), protocol={}) as recorder:
            recorder.record_episode(episode(0))
    assert snapshot(tmp_path)["status"] == "failed"
    assert snapshot(tmp_path)["completed_episodes"] == 1


def test_wandb_opt_in_honors_env_and_uploads_only_scalar_records(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "project override")
    monkeypatch.setenv("WANDB_ENTITY", "entity override")
    monkeypatch.setenv("WANDB_MODE", "offline")
    client = FakeWandb()
    with EvaluationRecorder(tmp_path, 1, config=config(), protocol={"noise": 0},
                            report_to="wandb", wandb_client=client,
                            run_name="evaluation", wandb_group="seed-42") as recorder:
        recorder.record_episode(episode(0))
    assert client.init_kwargs["project"] == "project override"
    assert client.init_kwargs["entity"] == "entity override"
    assert client.init_kwargs["mode"] == "offline"
    assert client.init_kwargs["group"] == "seed-42"
    assert client.init_kwargs["settings"] == {"disable_code": True, "save_code": False}
    assert client.run.logs[0][0] == 1
    assert all(isinstance(value, (int, float)) for value in client.run.logs[0][1].values())
    assert client.run.summary["eval/success_rate"] == 1.0
    assert client.run.exit_codes == [0]


def test_default_does_not_initialize_wandb_and_preserves_existing_results(tmp_path):
    client = FakeWandb()
    with EvaluationRecorder(tmp_path, 1, config=config(), protocol={}, wandb_client=client) as recorder:
        recorder.record_episode(episode(0))
    original = (tmp_path / "result.json").read_text()
    assert client.init_kwargs is None
    with pytest.raises(FileExistsError, match="already exist"):
        EvaluationRecorder(tmp_path, 1, config=config(), protocol={})
    assert (tmp_path / "result.json").read_text() == original


def test_telemetry_finish_failure_does_not_leave_final_success_file(tmp_path):
    client = FakeWandb(fail_finish=True)
    with pytest.warns(UserWarning, match="cleanup also failed"):
        with pytest.raises(RuntimeError, match="telemetry finish failed"):
            with EvaluationRecorder(tmp_path, 1, config=config(), protocol={},
                                    report_to="wandb", wandb_client=client) as recorder:
                recorder.record_episode(episode(0))
    assert snapshot(tmp_path)["status"] == "failed"
    assert not (tmp_path / "success.txt").exists()


def test_evaluator_normal_return_completes_recorder_and_closes_env(tmp_path):
    # Execute the real evaluator function with a toy simulator/client. Extract only
    # its definitions so the test cannot import robotics packages or touch CUDA.
    import ast
    import contextlib
    from pathlib import Path
    import sys
    import time
    from types import SimpleNamespace
    import warnings

    import numpy as np

    source = Path(__file__).resolve().parents[1] / "scripts/eval_policy_robocasa.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"run_evaluation", "evaluation_protocol"}]

    class FakeEnv:
        num_envs = 1
        closed = False

        def reset(self):
            return {}, {}

        def step(self, actions):
            return {}, np.zeros(1), np.array([True]), np.array([False]), {
                "success": np.array([True]), "num_executed_steps": np.array([16])}

        def close(self):
            self.closed = True

    env = FakeEnv()
    policy = SimpleNamespace(get_modality_config=lambda: {}, get_action=lambda obs: {})
    namespace = {"np": np, "sys": sys, "time": time, "warnings": warnings,
                 "json": json, "Path": Path, "control_seed": lambda seed: None,
                 "DATA_CONFIG_MAP": {"single_panda_gripper_rl_inference": lambda **kw: None},
                 "RobotInferenceClient": lambda **kw: policy,
                 "load_composite_controller_config": lambda **kw: {},
                 "load_robocasa_gym_env": lambda *a, **kw: env,
                 "tqdm": lambda **kw: contextlib.nullcontext(SimpleNamespace(update=lambda n: None))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    args = SimpleNamespace(**{**config(), "actor_model_path": None,
        "seed": 42, "data_config": "single_panda_gripper_rl_inference",
        "generative_textures": False, "noise_smoothing": .3, "robots": "PandaOmron",
        "num_samples": 1, "temperature": 0., "reward_shaping": False,
        "host": "localhost", "port": 5555, "collect_data": False,
        "data_collection_path": "", "save_video": False, "output_path": str(tmp_path),
        "num_episodes": 1, "execute_horizon": 16, "save_inference_inputs": False, "deas_backend": "legacy"})
    with EvaluationRecorder(tmp_path, 1, config=vars(args),
                            protocol=namespace["evaluation_protocol"](args)) as recorder:
        namespace["run_evaluation"](args, recorder)
    assert env.closed
    assert snapshot(tmp_path)["status"] == "completed"
    assert snapshot(tmp_path)["completed_episodes"] == 1
    assert snapshot(tmp_path)["success_rate"] == 1.0


def test_video_manifest_matches_counted_episodes_and_rejects_missing_files(tmp_path):
    from gr00t.eval.results import recorded_episode_videos
    episodes = [{"episode": 0, "env_index": 1, "success": False, "length": 16},
                {"episode": 1, "env_index": 0, "success": True, "length": 32},
                {"episode": 2, "env_index": 1, "success": True, "length": 16}]
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n")
    for env_index, local_index in ((1, 0), (0, 0), (1, 1)):
        path = tmp_path / "videos" / f"env_{env_index}" / f"rl-video-episode-{local_index}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock-video")
    videos = recorded_episode_videos(tmp_path)
    assert [row["episode"] for row in videos] == [0, 1, 2]
    assert videos[2]["path"].endswith("env_1/rl-video-episode-1.mp4")
    assert (tmp_path / "videos.json").is_file()
    (tmp_path / "videos/env_1/rl-video-episode-1.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="episode 2"):
        recorded_episode_videos(tmp_path)
