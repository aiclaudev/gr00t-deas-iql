#!/usr/bin/env python3
"""Read episode progress from the exact local files listed in a manifest.

No GPU, external logging, Slurm submission, or recursive directory scan.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / 'output/robocasa-comparison/seed42-bc2-vs-bon50-eval012-50ep-20260919/manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest['output_root']).resolve()
    totals = defaultdict(lambda: [0, 0, 0])
    print(datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST'))
    print(f"{'Method':<8} {'Seed':>4} {'Task':<23} {'Episodes':>10} {'Success':>8} {'State':<14} Job")
    print('-' * 92)
    for job in manifest['jobs']:
        path = Path(job['result_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Result path outside manifest root: {path}')
        result = json.loads(path.read_text()) if path.is_file() else {}
        done = result.get('completed_episodes', 0)
        expected = job['expected_episodes']
        successes = result.get('success_count', 0)
        method = (manifest['config'].get('bc_label', 'bc2').upper() if job['method'] == 'gr00tn15'
                  else 'BoN' + str(manifest['config']['num_samples']))
        totals[method][0] += done
        totals[method][1] += expected
        totals[method][2] += successes
        # Absence of result.json cannot distinguish queued from worker setup.
        status = result.get('status', 'no-result-yet')
        count = f'{done}/{expected}'
        print(f"{method:<8} {job['eval_seed']:>4} {job['task']:<23} {count:>10} {successes:>8} {status:<14} {job['job_id']}")
    print()
    for method, (done, total, successes) in totals.items():
        print(f'{method}: {done}/{total} episodes ({100 * done / total:.1f}%), {successes} successes so far')
    done = sum(t[0] for t in totals.values())
    total = sum(t[1] for t in totals.values())
    print(f'Total: {done}/{total} episodes ({100 * done / total:.1f}%)')
    print('no-result-yet = queued or setting up; use sjob <jobid> for live job state.')


if __name__ == '__main__':
    main()
