"""Incremental evaluation records; importing this module needs only the standard library."""

import csv
import datetime
import importlib
import json
import math
import os
from pathlib import Path
import time
import warnings


EPISODE_FIELDS = ("episode", "env_index", "success", "length", "elapsed_seconds")


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_write_json(path, data):
    """Replace a JSON snapshot on the same filesystem; readers never see partial JSON."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class EvaluationRecorder:
    """Record counted episodes, including partial progress if evaluation raises.

    Episode ``elapsed_seconds`` is measured from rollout start, excluding model and
    simulator setup. Result ``walltime_seconds`` includes that setup and cleanup.
    A killed process may leave status=running; aggregators must accept completed only.
    """

    def __init__(self, output_path, expected_episodes, *, config, protocol,
                 report_to="none", training_seed=None, evaluation_seed=None,
                 run_name=None, wandb_group=None, wandb_client=None):
        if expected_episodes < 1:
            raise ValueError("expected_episodes must be positive")
        if report_to not in ("none", "wandb"):
            raise ValueError("report_to must be none or wandb")
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.result_path = self.output_path / "result.json"
        if any((self.output_path / name).exists() for name in
               ("result.json", "episodes.jsonl", "eval.csv", "success.txt")):
            raise FileExistsError(f"Evaluation results already exist in {self.output_path}; use a new output path")
        self._started = time.monotonic()
        self._run = None
        self._jsonl = None
        self._csv = None
        self._client = wandb_client
        self._report_to = report_to
        self._run_name = run_name
        self._wandb_group = wandb_group
        self.result = {
            "schema_version": 1,
            "status": "running",
            "expected_episodes": int(expected_episodes),
            "completed_episodes": 0,
            "success_count": 0,
            "success_rate": None,
            "mean_episode_length": None,
            "walltime_seconds": 0.0,
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "episode_elapsed_origin": "rollout_start",
            "config": dict(config),
            "protocol": dict(protocol),
            "checkpoints": {
                "actor": config.get("actor_model_path"),
                "critic": config.get("critic_model_path"),
            },
            "seeds": {"training": training_seed, "evaluation": evaluation_seed},
        }
        self._length_sum = 0
        self._write_snapshot()

    def _write_snapshot(self):
        self.result["walltime_seconds"] = time.monotonic() - self._started
        atomic_write_json(self.result_path, self.result)

    def __enter__(self):
        try:
            self._jsonl = (self.output_path / "episodes.jsonl").open("x", encoding="utf-8")
            self._csv = (self.output_path / "eval.csv").open("x", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv, fieldnames=EPISODE_FIELDS)
            self._writer.writeheader()
            self._csv.flush()
            if self._report_to == "wandb":
                client = self._client or importlib.import_module("wandb")
                self._run = client.init(
                    project=os.environ.get("WANDB_PROJECT", "gr00t1.5 finetune"),
                    entity=os.environ.get("WANDB_ENTITY", "aiclaudev"),
                    mode=os.environ.get("WANDB_MODE", "online"),
                    name=self._run_name,
                    group=self._wandb_group,
                    job_type="robocasa-eval",
                    dir=str(self.output_path),
                    config={"evaluation": self.result["config"],
                            "protocol": self.result["protocol"],
                            "checkpoints": self.result["checkpoints"],
                            "seeds": self.result["seeds"]},
                    settings={"disable_code": True, "save_code": False},
                )
            return self
        except BaseException as error:
            self._fail(error)
            self._close(preserve_error=True)
            raise

    def record_episode(self, record):
        if self.result["status"] != "running":
            raise RuntimeError("Cannot add episodes after evaluation finished")
        expected_index = self.result["completed_episodes"]
        if int(record["episode"]) != expected_index or expected_index >= self.result["expected_episodes"]:
            raise ValueError("Episode records must be sequential and cannot exceed expected_episodes")
        episode = {"episode": expected_index, "env_index": int(record["env_index"]),
                   "success": bool(record["success"]), "length": int(record["length"]),
                   "elapsed_seconds": float(record["elapsed_seconds"])}
        if episode["env_index"] < 0 or episode["length"] < 0 or not math.isfinite(episode["elapsed_seconds"]) or episode["elapsed_seconds"] < 0:
            raise ValueError("Invalid episode environment, length, or elapsed_seconds")
        self._jsonl.write(json.dumps(episode, allow_nan=False) + "\n")
        self._jsonl.flush()
        self._writer.writerow({**episode, "success": int(episode["success"])})
        self._csv.flush()
        self.result["completed_episodes"] += 1
        self.result["success_count"] += int(episode["success"])
        self._length_sum += episode["length"]
        count = self.result["completed_episodes"]
        self.result["success_rate"] = self.result["success_count"] / count
        self.result["mean_episode_length"] = self._length_sum / count
        self._write_snapshot()
        if self._run is not None:
            self._run.log({
                "eval/episodes_completed": count,
                "eval/episode_success": int(episode["success"]),
                "eval/episode_length": episode["length"],
                "eval/running_success_rate": self.result["success_rate"],
                "eval/rollout_elapsed_seconds": episode["elapsed_seconds"],
                "eval/walltime_seconds": self.result["walltime_seconds"],
            }, step=count)

    def complete(self):
        if self.result["completed_episodes"] != self.result["expected_episodes"]:
            raise RuntimeError("Evaluation ended before all requested episodes completed")
        self.result["status"] = "completed"
        self.result["finished_at_utc"] = _utc_now()
        self._write_snapshot()
        (self.output_path / "success.txt").write_text(
            f"Success Rate: {self.result['success_rate']:.4f}\n", encoding="utf-8")
        self._wandb_summary()

    def _wandb_summary(self):
        if self._run is not None:
            self._run.summary.update({
                "eval/status": self.result["status"],
                "eval/expected_episodes": self.result["expected_episodes"],
                "eval/completed_episodes": self.result["completed_episodes"],
                "eval/success_count": self.result["success_count"],
                "eval/success_rate": self.result["success_rate"] if self.result["status"] == "completed" else None,
                "eval/mean_episode_length": self.result["mean_episode_length"],
                "eval/walltime_seconds": self.result["walltime_seconds"],
            })

    def _fail(self, error):
        self.result["status"] = "failed"
        self.result["finished_at_utc"] = _utc_now()
        self.result["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            (self.output_path / "success.txt").unlink(missing_ok=True)
            self._write_snapshot()
            self._wandb_summary()
        except BaseException as recording_error:
            warnings.warn(f"Could not save evaluation failure details: {recording_error}")

    def _close(self, preserve_error=False):
        first_error = None
        for stream in (self._jsonl, self._csv):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as error:
                    first_error = first_error or error
        if self._run is not None:
            try:
                self._run.finish(exit_code=0 if self.result["status"] == "completed" else 1)
            except BaseException as error:
                first_error = first_error or error
        if first_error is not None:
            if preserve_error:
                warnings.warn(f"Evaluation cleanup also failed: {first_error}")
            else:
                raise first_error

    def __exit__(self, exc_type, error, traceback):
        if error is not None:
            self._fail(error)
            self._close(preserve_error=True)
            return False
        try:
            self.complete()
            self._close()
        except BaseException as completion_error:
            self._fail(completion_error)
            self._close(preserve_error=True)
            raise
        return False


def recorded_episode_videos(output_path):
    """Map counted episodes to finalized MP4s, including vector env-local indices."""
    output = Path(output_path)
    counts = {}
    videos = []
    with (output / "episodes.jsonl").open() as stream:
        for line in stream:
            episode = json.loads(line)
            env_index = episode["env_index"]
            local_index = counts.get(env_index, 0)
            path = output / "videos" / f"env_{env_index}" / f"rl-video-episode-{local_index}.mp4"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing video for evaluated episode {episode['episode']}: {path}")
            videos.append({"episode": episode["episode"], "env_index": env_index,
                           "success": episode["success"], "length": episode["length"],
                           "path": str(path.resolve()), "bytes": path.stat().st_size})
            counts[env_index] = local_index + 1
    atomic_write_json(output / "videos.json", {"videos": videos})
    return videos
