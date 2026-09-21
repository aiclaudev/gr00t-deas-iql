#!/usr/bin/env python3
"""Copy Git-visible source files from sibling workspaces; default is a dry run.
Never deletes destination files: review upstream deletions separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
mapping = {'deas-gr00t15': 'DEAS-Isaac-GR00T', 'gr00t17': 'Isaac-GR00T'}
blocked = {'demo_data', '.github', '.git', '.cache', '.env', 'output', 'outputs', 'wandb', 'checkpoints', '__pycache__', 'node_modules'}
extensions = {'.py', '.sh', '.md', '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.txt', '.lock', '.ipynb', '.xml', '.html', '.css', '.js', '.ts', '.rst', '.dockerignore', '.gitignore', '.gitattributes'}
names = {'LICENSE', 'NOTICE', 'Dockerfile', 'Makefile', 'uv.lock'}
provenance = {}
for target, source in mapping.items():
    src = root.parent / source
    listed = subprocess.check_output(['git', '-C', str(src), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
    digest = hashlib.sha256()
    count = 0
    for name in sorted(set(filter(None, listed))):
        rel = Path(name)
        if any(x in blocked or x.startswith('.venv') for x in rel.parts):
            continue
        p = src / rel
        if p.is_symlink() or not p.is_file():
            continue
        if p.suffix not in extensions and p.name not in names:
            continue
        if p.stat().st_size > 10 * 1024**2:
            continue
        data = p.read_bytes()
        digest.update(name.encode() + b'\0' + hashlib.sha256(data).digest())
        count += 1
        dst = root / target / rel
        if not dst.exists() or dst.read_bytes() != data:
            print(f'{target}/{rel}')
            if args.apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    provenance[target] = {
        'source_commit': subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip(),
        'snapshot': 'working tree including local modifications',
        'synced_utc': datetime.now(timezone.utc).isoformat(),
        'source_files': count,
        'source_digest_sha256': digest.hexdigest(),
    }
if args.apply:
    (root / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
