# Combined GR00T source management

This repository is the shared Git publication point for GR00T 1.5 and 1.7.
Keep separate environments: both projects expose the `gr00t` Python package.

The sibling workspaces `../DEAS-Isaac-GR00T` and `../Isaac-GR00T` remain the
runtime/editable-install locations. After editing them, synchronize source here:

```bash
python tools/sync_workspace.py
python tools/sync_workspace.py --apply
```

Review the diff, run relevant checks, and commit both components here. Push when
authorized. Do not edit both copies independently. The sync script never deletes
files; review source deletions and renames explicitly. Update provenance with it.
Do not publish tokens, datasets, weights, environments, or experiment outputs.
Do not change running job snapshots when synchronizing code.
GPU operations follow the user's b200-cluster skill and server instructions.
