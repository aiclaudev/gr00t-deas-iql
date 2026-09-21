#!/usr/bin/env python3
"""Check that the SVF BoN loader reproduces the Q values the training class computes.

The deployment loader rebuilds PolicyTDSVF's scoring path from the published
export. That reconstruction is only trustworthy if, on identical inputs, it
returns the same numbers as PolicyTDSVF itself. This runs both and compares.

Reference path (gr00t/model/svf/policy_td.py, PolicyTDSVF.forward):

    raw      = actor.backbone(batch)
    context  = prepare_head_context(reference_head, raw, batch)
    pooled   = context.backbone_features.mean(1, keepdim=True).float()
    features = project(pooled, embodiment_id)            # tanh
    q1, q2   = q(features, state*state_mask, action*action_mask)

Deployment path (gr00t/model/svf_bon_policy.py, SVFCriticScorer):

    pooled   = encode(backbone_input)
    features = project(pooled, embodiment_id)
    scores   = score(features, states, actions)

Both are driven from one batch, so any divergence is in the reconstruction
rather than in the data.

    verify_svf_q.py --critic <export>/critic --reference-actor <bc2> \\
        --dataset ~/data/fake_robocasa_rl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def build_batch(dataset_path, reference_actor, horizon, batch_size, device):
    """One collated batch, normalized exactly as SVF training normalizes."""
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.transforms import DefaultDataCollator

    config = DATA_CONFIG_MAP["single_panda_gripper_rl"](AS=1)
    transforms = config.transform()
    metadata = json.loads((Path(reference_actor) / "experiment_cfg" / "metadata.json").read_text())
    dataset = LeRobotSingleDataset(
        dataset_path, config.modality_config(), transforms=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, video_backend="decord", use_rl=True,
    )
    samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    batch = DefaultDataCollator()(samples)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}, metadata


def reference_q(reference_actor, export, batch, horizon):
    """Q as PolicyTDSVF computes it, with the exported weights loaded in."""
    from safetensors.torch import load_file

    from gr00t.model.svf.adapters import prepare_head_context
    from gr00t.model.svf.policy_td import PolicyTDSVF

    model = PolicyTDSVF(reference_actor).to("cuda").eval()
    weights = load_file(str(export))
    model.projection.load_state_dict(
        {k[len("projection."):]: v for k, v in weights.items() if k.startswith("projection.")},
        strict=True)
    model.q.load_state_dict(
        {k[len("q."):]: v for k, v in weights.items() if k.startswith("q.")}, strict=True)
    model.to("cuda").eval()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model.actor.backbone(batch)
        context = prepare_head_context(model.reference_head, raw, batch)
    pooled = context.backbone_features.mean(1, keepdim=True).float()
    features = model.project(pooled, batch["embodiment_id"])
    states = batch["state"].float() * batch["state_mask"].float()
    actions = batch["action"].float() * batch["action_mask"].float()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q1, q2 = model.q(features, states, actions[:, :horizon])
    del model
    torch.cuda.empty_cache()
    return pooled.float(), features.float(), torch.minimum(q1.float(), q2.float())


def deployment_q(reference_actor, export, batch, horizon):
    """Q as the BoN loader computes it."""
    from gr00t.model.svf_bon_policy import SVFCriticScorer

    scorer = SVFCriticScorer(reference_actor, horizon=horizon, device="cuda")
    scorer.load_export(export)
    pooled = scorer.encode(batch)
    features = scorer.project(pooled, batch["embodiment_id"])
    states = batch["state"].float() * batch["state_mask"].float()
    actions = batch["action"].float() * batch["action_mask"].float()
    scores = scorer.score(features, states, actions[:, :horizon])
    del scorer
    torch.cuda.empty_cache()
    return pooled.float(), features.float(), scores


def report(name, a, b, tolerance):
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape:
        print(f"  {name:10s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    gap = (a - b).abs()
    scale = b.abs().max().item() or 1.0
    ok = gap.max().item() <= tolerance * max(scale, 1.0)
    print(f"  {name:10s} max|diff| {gap.max().item():.3e}  mean|diff| {gap.mean().item():.3e}"
          f"  (|value| up to {scale:.3f})  {'OK' if ok else 'DIFFERS'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--critic", type=Path, required=True,
                        help="The export's critic/ directory")
    parser.add_argument("--reference-actor", type=Path, required=True,
                        help="Checkpoint the SVF run started from (BC2)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Any LeRobot dataset the RL config can load")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-2,
                        help="Both paths run under bfloat16 autocast, so exact equality "
                             "is not expected; this bounds the relative gap")
    args = parser.parse_args()

    config = json.loads((args.critic / "config.json").read_text())
    horizon = config["horizon"]
    export = args.critic / "q_projection.safetensors"
    print(f"algorithm: {config.get('algorithm')!r}")
    print(f"horizon: {horizon}, trained steps: {config.get('steps')}")

    batch, _ = build_batch(args.dataset, args.reference_actor, horizon,
                           args.batch_size, "cuda")
    print(f"batch: state {tuple(batch['state'].shape)}, action {tuple(batch['action'].shape)}")

    print("\nreference (PolicyTDSVF)")
    ref_pooled, ref_features, ref_q = reference_q(args.reference_actor, export, batch, horizon)
    print(f"  Q min {ref_q.min().item():.4f}  mean {ref_q.mean().item():.4f}  max {ref_q.max().item():.4f}")

    print("\ndeployment (SVFCriticScorer)")
    dep_pooled, dep_features, dep_q = deployment_q(args.reference_actor, export, batch, horizon)
    print(f"  Q min {dep_q.min().item():.4f}  mean {dep_q.mean().item():.4f}  max {dep_q.max().item():.4f}")

    print("\ncomparison")
    results = [
        report("pooled", ref_pooled, dep_pooled, args.tolerance),
        report("features", ref_features, dep_features, args.tolerance),
        report("Q", ref_q, dep_q, args.tolerance),
    ]
    if all(results):
        print("\nPASS: the deployment loader reproduces the training-time Q path.")
        return 0
    print("\nFAIL: the paths disagree; do not evaluate with this loader.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
