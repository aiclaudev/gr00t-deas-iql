
## Shared Git publication

Publish local research source changes through `../gr00t-deas-iql`, which combines
GR00T 1.5 and 1.7. After edits, run
`python ../gr00t-deas-iql/tools/sync_workspace.py --apply`, review the combined
Git diff and commit there. Push when authorized. This workspace remains the
runtime/editable-install source; do not independently edit its mirrored copy.
Never copy weights, data, tokens, or running output snapshots into Git.
