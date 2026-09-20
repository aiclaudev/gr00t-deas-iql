#!/usr/bin/env python3
"""Aggregate exactly the RoboCasa evaluations listed in a submission manifest."""
import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def positive_int(value, label, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def same_path(left, right):
    return isinstance(left, str) and Path(left).resolve() == Path(right).resolve()


def inspect_result(job, root, planned_config):
    path = Path(job["result_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Result path escapes output_root: {path}")
    row = {
        "training_seed": job["training_seed"], "eval_seed": job["eval_seed"],
        "task": job["task"], "method": job["method"], "job_id": job.get("job_id"),
        "expected_episodes": job["expected_episodes"], "completed_episodes": 0,
        "success_count": None, "success_rate": None, "walltime_seconds": None,
        "status": "missing", "result_path": str(path), "error": "",
    }
    if not path.is_file():
        return row
    try:
        result = read_json(path)
        if result.get("schema_version") != 1:
            raise ValueError("Unsupported result schema_version")
        status = result.get("status")
        if status not in ("running", "completed", "failed"):
            raise ValueError("Unexpected result status")
        config, seeds = result.get("config", {}), result.get("seeds", {})
        expected = positive_int(result.get("expected_episodes"), "expected_episodes")
        completed = positive_int(result.get("completed_episodes"), "completed_episodes", True)
        successes = positive_int(result.get("success_count"), "success_count", True)
        if expected != job["expected_episodes"] or not 0 <= successes <= completed <= expected:
            raise ValueError("Episode counts do not match the planned evaluation")
        if config.get("env_name") != job["task"] or config.get("model_type") != job["method"]:
            raise ValueError("Task/method does not match the planned evaluation")
        if seeds.get("training") != job["training_seed"] or seeds.get("evaluation") != job["eval_seed"]:
            raise ValueError("Seeds do not match the planned evaluation")
        for name in ("n_envs", "action_horizon", "denoising_steps"):
            if config.get(name) != planned_config.get(name):
                raise ValueError(f"{name} does not match the planned evaluation")
        # Older full-horizon manifests omitted this setting.
        actual_execute = config.get("execute_horizon", config.get("action_horizon"))
        expected_execute = planned_config.get("execute_horizon", planned_config.get("action_horizon"))
        if actual_execute != expected_execute:
            raise ValueError("execute_horizon does not match the planned evaluation")
        if config.get("noise") != 0.0:
            raise ValueError("Unexpected action noise in the evaluation")
        if bool(config.get("save_video", False)) != bool(planned_config.get("save_video", False)):
            raise ValueError("Video setting does not match the planned evaluation")
        if not same_path(result.get("checkpoints", {}).get("actor"), job["actor"]):
            raise ValueError("Actor checkpoint does not match the planned evaluation")
        if job["method"] == "deas":
            if not same_path(result.get("checkpoints", {}).get("critic"), job["critic"]):
                raise ValueError("Critic checkpoint does not match the planned evaluation")
            for name in ("num_samples", "temperature", "deas_backend"):
                if config.get(name) != planned_config.get(name):
                    raise ValueError(f"{name} does not match the planned BoN evaluation")
        if planned_config.get("save_inference_inputs", False):
            if not config.get("save_inference_inputs", False):
                raise ValueError("Inference input recording was not enabled")
            if status == "completed":
                trace = result.get("inference_trace", {})
                directory = Path(trace.get("directory", "")).resolve()
                if not directory.is_relative_to(path.parent) or trace.get("calls", 0) < 1:
                    raise ValueError("Missing inference trace summary")
                index = directory / "index.jsonl"
                records = [json.loads(line) for line in index.read_text().splitlines()]
                if len(records) != trace["calls"]:
                    raise ValueError("Incomplete inference trace index")
                for entry in records:
                    item = (directory / entry["file"]).resolve()
                    if not item.is_relative_to(directory) or not item.is_file() or item.stat().st_size == 0:
                        raise ValueError("Missing inference input archive")
        walltime = result.get("walltime_seconds")
        if isinstance(walltime, bool) or not isinstance(walltime, (int, float)) or not math.isfinite(walltime) or walltime < 0:
            raise ValueError("Invalid walltime_seconds")
        row.update(status=status, completed_episodes=completed, success_count=successes,
                   walltime_seconds=walltime)
        if status == "completed":
            if completed != expected:
                raise ValueError("Completed result does not have all expected episodes")
            rate = result.get("success_rate")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not math.isclose(rate, successes / completed, abs_tol=1e-12):
                raise ValueError("Success rate disagrees with success/episode counts")
            if planned_config.get("save_video", False):
                videos = result.get("videos", [])
                if len(videos) != expected or {video.get("episode") for video in videos} != set(range(expected)):
                    raise ValueError("Completed evaluation is missing requested episode videos")
                for video in videos:
                    video_path = Path(video["path"]).resolve()
                    if not video_path.is_relative_to(path.parent) or not video_path.is_file() or video_path.stat().st_size <= 0:
                        raise ValueError("Missing or invalid episode video file")
            row["success_rate"] = successes / completed
        else:
            row["error"] = str(result.get("error") or "Evaluation has not completed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(status="invalid", success_rate=None, error=str(exc))
    return row


def aggregate(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    root = Path(manifest["output_root"]).resolve()
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest must contain evaluation jobs")
    seen = set()
    rows = []
    for job in jobs:
        key = (job["training_seed"], job["eval_seed"], job["task"], job["method"])
        if key in seen:
            raise ValueError(f"Duplicate evaluation in manifest: {key}")
        seen.add(key)
        positive_int(job["expected_episodes"], "planned episode count")
        rows.append(inspect_result(job, root, manifest["config"]))
    groups = []
    for task, method in sorted({(row["task"], row["method"]) for row in rows}):
        planned = [row for row in rows if (row["task"], row["method"]) == (task, method)]
        complete = [row for row in planned if row["status"] == "completed"]
        rates = [row["success_rate"] for row in complete]
        episodes = sum(row["completed_episodes"] for row in complete)
        successes = sum(row["success_count"] for row in complete)
        groups.append({
            "task": task, "method": method, "planned_runs": len(planned),
            "completed_runs": len(complete), "complete": len(complete) == len(planned),
            "successful_episodes": successes, "evaluated_episodes": episodes,
            "pooled_success_rate": successes / episodes if episodes else None,
            "seed_mean_success_rate": statistics.mean(rates) if rates else None,
            "seed_std_success_rate": statistics.stdev(rates) if len(rates) > 1 else None,
            "completed_training_seeds": [row["training_seed"] for row in complete],
            "completed_evaluation_seeds": [row["eval_seed"] for row in complete],
        })
    done = sum(row["status"] == "completed" for row in rows)
    return {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if done == len(rows) else "incomplete",
        "planned_runs": len(rows), "completed_runs": done,
        "planned_episodes": sum(row["expected_episodes"] for row in rows),
        "aggregation_policy": "Only completed, validated runs contribute to success rates; missing/failed/running/invalid runs are excluded, never counted as zero successes.",
        "runs": rows, "groups": groups,
    }


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_outputs(output, report):
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", report)
    for filename, rows in (("runs.csv", report["runs"]), ("by_task.csv", report["groups"])):
        path = output / filename
        temporary = path.with_name(path.name + ".pending")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    lines = ["# RoboCasa evaluation results", "", f"Status: **{report['status']}**; {report['completed_runs']}/{report['planned_runs']} runs completed.", "",
             "Only completed and validated runs are included below. Missing or failed evaluations are not zero-success results.", "",
             "| Task | Method | Completed runs | Successes / episodes | Success rate |", "|---|---|---:|---:|---:|"]
    for group in report["groups"]:
        rate = "—" if group["pooled_success_rate"] is None else f"{100 * group['pooled_success_rate']:.1f}%"
        lines.append(f"| {group['task']} | {group['method']} | {group['completed_runs']}/{group['planned_runs']} | {group['successful_episodes']}/{group['evaluated_episodes']} | {rate} |")
    incomplete = [row for row in report["runs"] if row["status"] != "completed"]
    if incomplete:
        lines.extend(["", "## Incomplete evaluations", ""])
        lines.extend(f"- Seed {row['training_seed']}, {row['task']}: {row['status']} ({row['error'] or 'result.json missing'})" for row in incomplete)
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    report = aggregate(manifest)
    output = args.output_dir or Path(manifest["output_root"]) / "aggregate"
    write_outputs(output, report)
    print(json.dumps({"status": report["status"], "completed_runs": report["completed_runs"], "planned_runs": report["planned_runs"], "summary": str((output / "summary.json").resolve())}))
    return 0 if report["status"] == "completed" or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
