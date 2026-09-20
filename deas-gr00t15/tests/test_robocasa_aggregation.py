import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/robocasa/aggregate_results.py"
spec = importlib.util.spec_from_file_location("robocasa_aggregation", SCRIPT)
aggregation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregation)


def make_manifest(tmp_path, seeds=(42, 43, 44)):
    jobs = []
    for seed in seeds:
        jobs.append({
            "training_seed": seed, "eval_seed": seed, "task": "CoffeeSetupMug",
            "method": "gr00tn15", "job_id": str(seed + 1000),
            "expected_episodes": 50, "actor": str(tmp_path / f"actor-{seed}"),
            "result_path": str(tmp_path / f"seed-{seed}" / "result.json"),
        })
    return {"schema_version": 1, "output_root": str(tmp_path), "jobs": jobs,
            "config": {"n_envs": 1, "action_horizon": 16, "denoising_steps": 4}}


def write_result(job, *, successes=25, episodes=50, status="completed", **overrides):
    result = {
        "schema_version": 1, "status": status, "expected_episodes": 50,
        "completed_episodes": episodes, "success_count": successes,
        "success_rate": successes / episodes if episodes else None,
        "walltime_seconds": 100.0,
        "config": {"env_name": job["task"], "model_type": job["method"],
                   "n_envs": 1, "action_horizon": 16, "denoising_steps": 4, "noise": 0.0},
        "seeds": {"training": job["training_seed"], "evaluation": job["eval_seed"]},
        "checkpoints": {"actor": job["actor"], "critic": None},
    }
    result.update(overrides)
    path = Path(job["result_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return path


def test_completed_seeds_mean_std_and_output(tmp_path):
    manifest = make_manifest(tmp_path)
    for job, successes in zip(manifest["jobs"], (10, 25, 40)):
        write_result(job, successes=successes)
    result = aggregation.aggregate(manifest)
    assert result["status"] == "completed"
    group = result["groups"][0]
    assert group["pooled_success_rate"] == pytest.approx(0.5)
    assert group["seed_mean_success_rate"] == pytest.approx(0.5)
    assert group["seed_std_success_rate"] == pytest.approx(0.3)
    aggregation.write_outputs(tmp_path / "aggregate", result)
    assert (tmp_path / "aggregate/runs.csv").is_file()
    assert "75/150" in (tmp_path / "aggregate/summary.md").read_text()


def test_failed_and_missing_runs_are_not_zero_success_scores(tmp_path):
    manifest = make_manifest(tmp_path)
    write_result(manifest["jobs"][0], successes=40)
    write_result(manifest["jobs"][1], successes=1, episodes=7, status="failed")
    result = aggregation.aggregate(manifest)
    assert result["status"] == "incomplete"
    assert [row["status"] for row in result["runs"]] == ["completed", "failed", "missing"]
    assert result["groups"][0]["pooled_success_rate"] == 0.8
    assert result["groups"][0]["evaluated_episodes"] == 50
    assert result["runs"][1]["completed_episodes"] == 7
    assert result["runs"][1]["success_rate"] is None


@pytest.mark.parametrize("change", [
    {"completed_episodes": 49}, {"success_rate": 0.99},
    {"seeds": {"training": 43, "evaluation": 42}},
    {"checkpoints": {"actor": "/wrong/checkpoint"}},
    {"walltime_seconds": float("nan")},
    {"config": {"env_name": "CoffeeSetupMug", "model_type": "gr00tn15", "n_envs": 5,
                "action_horizon": 16, "denoising_steps": 4, "noise": 0.0}},
])
def test_mismatched_result_never_contributes(tmp_path, change):
    manifest = make_manifest(tmp_path, seeds=(42,))
    write_result(manifest["jobs"][0], **change)
    result = aggregation.aggregate(manifest)
    assert result["runs"][0]["status"] == "invalid"
    assert result["groups"][0]["pooled_success_rate"] is None


def test_duplicate_seed_result_is_rejected(tmp_path):
    manifest = make_manifest(tmp_path, seeds=(42, 42))
    with pytest.raises(ValueError, match="Duplicate"):
        aggregation.aggregate(manifest)


def test_result_outside_manifest_root_is_rejected(tmp_path):
    manifest = make_manifest(tmp_path, seeds=(42,))
    manifest["jobs"][0]["result_path"] = str(tmp_path.parent / "another-run/result.json")
    with pytest.raises(ValueError, match="escapes"):
        aggregation.aggregate(manifest)


def test_real_recorder_outputs_are_accepted_by_aggregator(tmp_path):
    from gr00t.eval.results import EvaluationRecorder

    manifest = make_manifest(tmp_path, seeds=(42,))
    job = manifest["jobs"][0]
    job["expected_episodes"] = 2
    config = {"env_name": job["task"], "model_type": job["method"],
              "actor_model_path": job["actor"], "critic_model_path": None,
              "n_envs": 1, "action_horizon": 16, "denoising_steps": 4, "noise": 0.0}
    output = Path(job["result_path"]).parent
    with EvaluationRecorder(output, 2, config=config, protocol={},
                            training_seed=job["training_seed"], evaluation_seed=job["eval_seed"]) as recorder:
        for episode, success in enumerate((True, False)):
            recorder.record_episode({"episode": episode, "env_index": 0, "success": success,
                                     "length": 16, "elapsed_seconds": float(episode + 1)})
    result = aggregation.aggregate(manifest)
    assert result["status"] == "completed"
    assert result["groups"][0]["pooled_success_rate"] == 0.5


def test_requested_videos_are_required_for_completed_aggregation(tmp_path):
    manifest = make_manifest(tmp_path, seeds=(42,))
    manifest["config"]["save_video"] = True
    job = manifest["jobs"][0]
    path = write_result(job)
    result = json.loads(path.read_text())
    result["config"]["save_video"] = True
    path.write_text(json.dumps(result))
    report = aggregation.aggregate(manifest)
    assert report["runs"][0]["status"] == "invalid"
    assert report["groups"][0]["pooled_success_rate"] is None
    result["videos"] = []
    for index in range(50):
        video = path.parent / "videos" / f"episode-{index}.mp4"
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(b"mock encoded video")
        result["videos"].append({"episode": index, "path": str(video)})
    path.write_text(json.dumps(result))
    assert aggregation.aggregate(manifest)["status"] == "completed"


def test_missing_recorded_input_invalidates_completed_evaluation(tmp_path):
    manifest = make_manifest(tmp_path, seeds=(42,))
    manifest['config']['save_inference_inputs'] = True
    job = manifest['jobs'][0]
    path = write_result(job)
    result = json.loads(path.read_text())
    result['config']['save_inference_inputs'] = True
    directory = path.parent / 'inference'
    directory.mkdir()
    (directory / 'index.jsonl').write_text(json.dumps({'call': 0, 'file': 'call-000000.npz'}) + '\n')
    result['inference_trace'] = {'directory': str(directory), 'calls': 1}
    path.write_text(json.dumps(result))
    assert aggregation.aggregate(manifest)['runs'][0]['status'] == 'invalid'
    (directory / 'call-000000.npz').write_bytes(b'test trace placeholder')
    assert aggregation.aggregate(manifest)['status'] == 'completed'


def test_different_execute_horizons_cannot_be_mixed(tmp_path):
    manifest = make_manifest(tmp_path, seeds=(42,))
    manifest['config']['execute_horizon'] = 8
    path = write_result(manifest['jobs'][0])
    assert aggregation.aggregate(manifest)['runs'][0]['status'] == 'invalid'
    result = json.loads(path.read_text())
    result['config']['execute_horizon'] = 8
    path.write_text(json.dumps(result))
    assert aggregation.aggregate(manifest)['status'] == 'completed'
