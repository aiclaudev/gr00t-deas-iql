import json

from libero_shards_to_lerobot import load_shards


def test_vector_overflow_and_partial_episodes_are_excluded(tmp_path):
    task = tmp_path / "libero_spatial" / "task_a"
    for name, complete in [("env000_ep00000", True), ("env001_ep00000", True),
                           ("env002_ep00000", False)]:
        shard = task / name
        shard.mkdir(parents=True)
        (shard / "frames.parquet").touch()
        (shard / "shard.json").write_text(json.dumps({
            "task_name": "task_a", "complete": complete}))
    # Without a selection file, older datasets still include all complete shards.
    assert len(load_shards(tmp_path, "suite")["libero_spatial"]) == 2
    (task / "accepted_episodes.json").write_text(json.dumps([
        "env001_ep00000", "env002_ep00000"]))
    selected = load_shards(tmp_path, "suite")["libero_spatial"]
    assert [path.name for path, _ in selected] == ["env001_ep00000"]
