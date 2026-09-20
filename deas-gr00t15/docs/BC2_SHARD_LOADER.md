# BC2: N1.7-style shard loading

Status: prepared, NOT submitted. Replaces cancelled BC2 job139104.

- BC model remains DEAS GR00T N1.5. Initialize from BC1 job139103 final30000 weights.
- 1GPU, global batch32, seed42, 30000 steps, save every10000.
- Original eight BC2 datasets (four tasks, demos + successful rollouts), mixture normalization and transforms retained.
- `--bc-loader shard` opts in; default random loader unchanged.
- Four worker processes, each with four sampled episodes per shard and256 uniform timestep draws per episode, shuffled across the shard. Original mixture/trajectory weighting is retained in expectation; sample order and correlation change.
- Each worker prefetches one next shard in a single CPU thread while transforming current samples. Raw video is bulk-decoded with Decord, keeping the original nearest-frame-start timestamp mapping and padding. This ports the N1.7 caching strategy, not its model or dataset API.
- Each raw shard budget4GiB, at most current+next per worker (32GiB across4workers), plus decode temporaries, parquet, prefetched batches and model memory. Oversize shard fails explicitly before allocating the excess video array.
- Prepared resources CPU12, RAM192GiB, own QOS; recheck capacity before submission.
- W&B retains BC settings: aiclaudev / gr00t1.5 finetune.
- Three CPU tests pass, including exact synthetic video frame/padding equivalence. Shell and Python syntax checked. No GPU smoke or throughput comparison has been run; 0.319 seconds/step from N1.7 is not a measured speed for this loader on N1.5.
- No exact dataloader resume guarantee: iterable sampling restarts its seeded stream on a fresh process.

Prepared manifest:
`/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/deas-training/20260920T113754Z-bc2-shard-bs32-30k-seed42/manifest.json`

The manifest stores the unexecuted sbatch command, BC1 dependency, source snapshot and hashes.
