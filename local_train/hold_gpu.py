#!/usr/bin/env python3
"""Reserve a GPU by holding memory on it, optionally keeping utilisation non-zero.

For when a GPU needs to stay claimed between real jobs. Prefer running the
actual workload when there is one; this is for the gap.

    local_train/hold_gpu.py --gpu 0 --gib 60
    local_train/hold_gpu.py --gpu 0 --gib 60 --busy      # also keeps util high
    local_train/hold_gpu.py --gpu 0 --gib 60 --hours 8   # release automatically

Ctrl-C, SIGTERM or the --hours deadline releases the memory.
"""
from __future__ import annotations

import argparse
import signal
import time


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gib", type=float, default=40.0, help="Memory to hold, in GiB")
    parser.add_argument("--busy", action="store_true",
                        help="Run a small matmul in a loop so utilisation is not 0%%")
    parser.add_argument("--hours", type=float, default=None,
                        help="Release after this long; default is until killed")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Seconds between status lines")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible")
    device = torch.device(f"cuda:{args.gpu}")
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(device)}, {total:.1f} GiB total")

    # One big block would fail on a fragmented device; many smaller ones will not.
    block_gib = 1.0
    elements = int(block_gib * 2**30 // 2)  # float16
    blocks = []
    try:
        while len(blocks) * block_gib < args.gib:
            blocks.append(torch.empty(elements, dtype=torch.float16, device=device))
    except torch.cuda.OutOfMemoryError:
        print(f"Stopped early at {len(blocks) * block_gib:.0f} GiB; the rest was unavailable")
    held = len(blocks) * block_gib
    print(f"Holding {held:.0f} GiB" + (", busy loop on" if args.busy else ", idle"))

    stop = False

    def release(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, release)
    signal.signal(signal.SIGTERM, release)

    started = time.monotonic()
    deadline = started + args.hours * 3600 if args.hours else None
    spinner = torch.randn(4096, 4096, device=device, dtype=torch.float16) if args.busy else None
    last_report = 0.0
    while not stop and (deadline is None or time.monotonic() < deadline):
        if args.busy:
            spinner = (spinner @ spinner).clamp_(-1, 1)
            torch.cuda.synchronize(device)
        else:
            time.sleep(1.0)
        elapsed = time.monotonic() - started
        if elapsed - last_report >= args.interval:
            print(f"held {held:.0f} GiB for {elapsed / 60:.0f} min", flush=True)
            last_report = elapsed

    print(f"Releasing after {(time.monotonic() - started) / 60:.0f} min")


if __name__ == "__main__":
    raise SystemExit(main())
