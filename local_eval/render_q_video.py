#!/usr/bin/env python3
"""Render each episode beside the Q the critic gave the action it executed.

The repository's scripts/robocasa/render_q_comparison.py refuses anything but
n_envs=1 — every context lookup in it is hardcoded to index 0 — so it cannot
read a run that evaluated environments in parallel. This reads the same
artifacts with the environment index carried through, and writes one video per
episode instead of a success/failure pair.

Alignment, which is the part worth stating:

- `videos/env_{i}/rl-video-episode-{n}.mp4` holds one frame per simulator step
  plus the reset frame, where `n` counts episodes within environment `i`.
- Each trace call is one action chunk. `context.env_episodes[i]` says which
  episode environment `i` was in, `context.episode_steps[i]` the step it
  started from, and `context.pending_reset[i]` marks the vector step a finished
  environment sits out.
- `q_scores` is (num_samples, n_envs). With BoN the executed action is the
  argmax, so the plotted value is the column max; at num_samples=1 that is
  simply the critic's score for the action the policy chose.

No model is loaded; Q comes from the recorded traces.

    render_q_video.py --task-dir <output>/CoffeeSetupMug --output-dir q_videos
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 720, 600
VIDEO_TOP, VIDEO_HEIGHT = 46, 360
FFMPEG = "ffmpeg"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def episode_video(task_dir, env_index, env_episode):
    return task_dir / f"videos/env_{env_index}" / f"rl-video-episode-{env_episode}.mp4"


def per_env_episode_numbers(episodes):
    """Map each episode record to its index within its own environment."""
    seen = {}
    numbers = []
    for record in episodes:
        env = record.get("env_index", 0)
        numbers.append(seen.get(env, 0))
        seen[env] = seen.get(env, 0) + 1
    return numbers


def q_curve(task_dir, index, env_index, env_episode):
    rows = []
    for entry in index:
        context = entry["context"]
        if context["env_episodes"][env_index] != env_episode:
            continue
        if context["pending_reset"][env_index]:
            continue
        with np.load(task_dir / "inference" / entry["file"], allow_pickle=False) as archive:
            if "q_scores" not in archive:
                raise ValueError(f"{entry['file']} has no q_scores; was the critic attached?")
            scores = archive["q_scores"][:, env_index].astype(float)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite Q scores")
        rows.append((context["episode_steps"][env_index], float(scores.max())))
    return np.array(rows) if rows else None


def render(task_dir, record, env_episode, index, output, title, span):
    curve = q_curve(task_dir, index, record.get("env_index", 0), env_episode)
    if curve is None:
        raise ValueError(f"No Q trace for env {record.get('env_index', 0)} episode {env_episode}")
    source = episode_video(task_dir, record.get("env_index", 0), env_episode)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {source}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"{source} reports no frame rate")

    low, high = span
    success = record["success"]
    colour = (90, 225, 90) if success else (90, 90, 250)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE)

    last = None
    length = record["length"]
    try:
        for frame in range(frames):
            ok, image = capture.read()
            if ok:
                last = cv2.resize(image, (WIDTH - 40, VIDEO_HEIGHT))
            canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            if last is not None:
                canvas[VIDEO_TOP:VIDEO_TOP + VIDEO_HEIGHT, 20:WIDTH - 20] = last

            def text(value, position, colour=(230, 230, 230), scale=0.52):
                cv2.putText(canvas, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                            colour, 1, cv2.LINE_AA)

            step = min(frame, length)
            cursor = max(0, np.searchsorted(curve[:, 0], step, side="right") - 1)
            value = curve[cursor, 1]
            text(f"{title} | {'SUCCESS' if success else 'FAILURE'} | "
                 f"env {record.get('env_index', 0)} ep {env_episode}", (16, 30), colour)
            text(f"step {step}/{length}   critic Q = {value:.4f}", (16, VIDEO_TOP + VIDEO_HEIGHT + 26))

            x0, y0, width, height = 70, VIDEO_TOP + VIDEO_HEIGHT + 50, WIDTH - 110, 110
            cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (80, 80, 80), 1)
            text(f"{high:.3f}", (6, y0 + 12), scale=0.4)
            text(f"{low:.3f}", (6, y0 + height), scale=0.4)
            points = np.array([(x0 + int(s / max(length, 1) * width),
                                y0 + height - int((v - low) / (high - low) * height))
                               for s, v in curve], np.int32)
            cv2.polylines(canvas, [points], False, (100, 100, 100), 1)
            cv2.polylines(canvas, [points[:cursor + 1]], False, colour, 2)
            marker = x0 + int(step / max(length, 1) * width)
            cv2.line(canvas, (marker, y0), (marker, y0 + height), (220, 220, 220), 1)
            text("simulation step   (Q scale shared across this task)",
                 (x0, HEIGHT - 12), scale=0.42)
            encoder.stdin.write(canvas.tobytes())
    finally:
        capture.release()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    return {"env_index": record.get("env_index", 0), "env_episode": env_episode,
            "success": success, "length": length, "frames": frames,
            "q_min": float(curve[:, 1].min()), "q_max": float(curve[:, 1].max()),
            "file": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True,
                        help="One task's evaluation output (result.json, videos/, inference/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Global episode numbers from episodes.jsonl; default is all")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    global FFMPEG
    FFMPEG = shutil.which("ffmpeg")
    if FFMPEG is None:
        # imageio-ffmpeg ships a binary, and the environment already depends on
        # it for writing episode videos.
        try:
            import imageio_ffmpeg

            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            parser.error("ffmpeg is not on PATH and imageio-ffmpeg is unavailable")
    cv2.setNumThreads(1)

    task_dir = args.task_dir.resolve()
    title = args.title or task_dir.name
    episodes = load_jsonl(task_dir / "episodes.jsonl")
    index = load_jsonl(task_dir / "inference" / "index.jsonl")
    numbers = per_env_episode_numbers(episodes)

    chosen = [(record, number) for record, number in zip(episodes, numbers)
              if args.episodes is None or record["episode"] in args.episodes]
    if not chosen:
        parser.error("No matching episodes")

    # One Q scale for the whole task, so episodes are visually comparable.
    curves = [q_curve(task_dir, index, record.get("env_index", 0), number)
              for record, number in chosen]
    values = np.concatenate([c[:, 1] for c in curves if c is not None])
    low, high = values.min(), values.max()
    pad = max((high - low) * 0.1, 1e-3)
    span = (low - pad, high + pad)

    manifest = []
    for record, number in chosen:
        output = args.output_dir / f"ep{record['episode']:03d}_env{record.get('env_index', 0)}_" \
                                   f"{'success' if record['success'] else 'failure'}.mp4"
        if output.exists():
            output.unlink()
        manifest.append(render(task_dir, record, number, index, output, title, span))
        print(f"  {output}  Q [{manifest[-1]['q_min']:.4f}, {manifest[-1]['q_max']:.4f}]",
              flush=True)

    summary = args.output_dir / "manifest.json"
    summary.write_text(json.dumps(
        {"task": task_dir.name, "q_scale": [float(span[0]), float(span[1])],
         "q_definition": "max over candidates of min(Q1,Q2); with num_samples=1 this is the "
                         "critic's score for the executed action",
         "alignment": "frame 0 is the reset; Q is held from each action selection step",
         "episodes": manifest}, indent=2) + "\n")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
