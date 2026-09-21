#!/usr/bin/env python3
"""List LIBERO tasks per suite, as the gymnasium ids the evaluator uses.

Runs inside the LIBERO island environment.
"""
from __future__ import annotations

import argparse
import json

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def tasks_by_suite(suites=SUITES):
    # Instantiating a benchmark prints "[info] using task orders ..." to stdout,
    # which would corrupt --json output; send it to stderr instead.
    import contextlib
    import sys

    from libero.libero import benchmark

    with contextlib.redirect_stdout(sys.stderr):
        catalog = benchmark.get_benchmark_dict()
        result = {}
        for suite in suites:
            if suite not in catalog:
                raise SystemExit(f"Unknown suite {suite}; available: {sorted(catalog)}")
            result[suite] = list(catalog[suite]().get_task_names())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=list(SUITES))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    catalog = tasks_by_suite(args.suites)
    if args.json:
        print(json.dumps(catalog, indent=2))
        return
    total = 0
    for suite, names in catalog.items():
        print(f"{suite} ({len(names)} tasks)")
        for name in names:
            print(f"  libero_sim/{name}")
        total += len(names)
    print(f"\n{total} tasks across {len(catalog)} suites")


if __name__ == "__main__":
    raise SystemExit(main())
