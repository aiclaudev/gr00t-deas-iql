#!/usr/bin/env python3
"""Make a GR00T N1.7 checkpoint loadable on a machine it was not trained on.

Training records the backbone VLM's location as an absolute path on the training
host, in `processor/processor_config.json` under `processor_kwargs.model_name`.
Loading elsewhere then fails in Qwen3VLProcessor.from_pretrained with
"Can't load image processor for /home/.../nvidia/Cosmos-Reason2-2B".

This builds a thin directory that symlinks the checkpoint's weights and config
and carries a corrected copy of `processor/`, so the original download stays
untouched and nothing large is duplicated. The replacement backbone is taken
from the checkpoint's own config.json `model_name` when that is already a Hub
id, which is the usual case.

    prepare_checkpoint.py --source <hf snapshot dir> --output ./ckpt
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROCESSOR_DIR = "processor"
CONFIG_NAME = "config.json"
DEFAULT_BACKBONE = "nvidia/Cosmos-Reason2-2B"


def resolve_backbone(source: Path, override: str | None) -> str:
    if override:
        return override
    config_path = source / CONFIG_NAME
    if config_path.is_file():
        name = json.loads(config_path.read_text()).get("model_name")
        # A Hub id has no leading slash; an absolute path is the stale form.
        if isinstance(name, str) and name and not name.startswith("/"):
            return name
    return DEFAULT_BACKBONE


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="Downloaded checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default=None,
                        help=f"Override the backbone id (default: from config.json, else {DEFAULT_BACKBONE})")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / PROCESSOR_DIR).is_dir():
        raise SystemExit(f"{source} has no {PROCESSOR_DIR}/ directory")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    backbone = resolve_backbone(source, args.backbone)

    for entry in sorted(source.iterdir()):
        if entry.name == PROCESSOR_DIR:
            continue
        (output / entry.name).symlink_to(entry.resolve())

    shutil.copytree(source / PROCESSOR_DIR, output / PROCESSOR_DIR)
    processor_config = output / PROCESSOR_DIR / "processor_config.json"
    config = json.loads(processor_config.read_text())
    kwargs = config.get("processor_kwargs", {})
    previous = kwargs.get("model_name")
    if previous and Path(previous).is_absolute() and not Path(previous).exists():
        kwargs["model_name"] = backbone
        processor_config.write_text(json.dumps(config, indent=2) + "\n")
        print(f"backbone model_name: {previous}\n                  -> {backbone}")
    else:
        print(f"backbone model_name left as {previous!r} (it resolves here)")

    print(f"Prepared {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
