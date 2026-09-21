#!/usr/bin/env python3
"""Download a fine-tuned GR00T checkpoint from the Hugging Face Hub.

Prints the resolved local checkpoint directory on the last stdout line, so a
shell caller can do:

    ACTOR=$(local_eval/fetch_weights.py --repo-id ORG/NAME | tail -1)

A checkpoint directory is what scripts/eval_policy_robocasa.py expects for
--actor_model_path / --critic_model_path: it must contain config.json and
experiment_cfg/metadata.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REQUIRED_FILES = ("config.json", "experiment_cfg/metadata.json")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True,
                        help="Hugging Face repository, e.g. my-org/gr00t-n15-robocasa")
    parser.add_argument("--subfolder", default=None,
                        help="Checkpoint directory inside the repository, e.g. checkpoint-20000")
    parser.add_argument("--revision", default=None, help="Branch, tag or commit SHA")
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset"))
    parser.add_argument("--local-dir", type=Path, default=None,
                        help="Materialise the files here instead of using the shared HF cache")
    parser.add_argument("--token", default=None,
                        help="Access token for a private repository; defaults to $HF_TOKEN")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Skip the checkpoint layout check")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    from huggingface_hub import snapshot_download

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Restricting the download to the requested subfolder avoids pulling every
    # checkpoint in a repository that holds several.
    allow_patterns = None
    if args.subfolder:
        allow_patterns = [f"{args.subfolder.strip('/')}/**"]

    root = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        token=token,
        allow_patterns=allow_patterns,
        local_dir=str(args.local_dir) if args.local_dir else None,
    ))
    checkpoint = root / args.subfolder.strip("/") if args.subfolder else root

    if not args.allow_incomplete:
        missing = [name for name in REQUIRED_FILES if not (checkpoint / name).is_file()]
        if missing:
            candidates = sorted(
                str(path.relative_to(checkpoint))
                for path in checkpoint.glob("*/config.json")
            )
            hint = f"  Directories that do look like checkpoints: {candidates}" if candidates else ""
            print(
                f"{checkpoint} is missing {missing}; it is not a GR00T checkpoint directory.\n"
                f"  Pass --subfolder to point at the checkpoint inside the repository.\n{hint}",
                file=sys.stderr,
            )
            return 2

    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
