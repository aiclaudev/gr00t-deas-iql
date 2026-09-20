"""CPU/Gloo regression for input-only autograd inside the joint SVF DDP forward.

No checkpoints, GPU, environment imports or /tmp files are needed. Rendezvous
and the small result record live inside this repository's output/code-checks.
"""

from contextlib import nullcontext
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel


REPO = Path(__file__).resolve().parents[1]


def _load_math(name, filename):
    # gr00t.model.__init__ eagerly imports full VLM dependencies. The tested
    # source files themselves require torch only; load those exact files here.
    spec = importlib.util.spec_from_file_location(name, REPO / 'gr00t/model/svf' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


objective = _load_math('_svf_ddp_objective', 'objective.py')
networks = _load_math('_svf_ddp_networks', 'networks.py')


class TinyJointModel(nn.Module):
    """Same single-DDP-wrapper graph as JointSVFModel, with small CPU heads."""

    def __init__(self, loss_type="mse"):
        super().__init__()
        self.actor = nn.Linear(6, 6)
        self.soft_value = networks.DoubleSoftValue(4, 3, 2, 3, time_dim=4, hidden_dim=16, depth=2,
            loss_type=loss_type, num_bins=17, value_min=-4., value_max=4., sigma=.4)
        self.reference_weight = nn.Parameter(torch.tensor(.2), requires_grad=False)
        self.teacher_weight = nn.Parameter(torch.tensor(.7), requires_grad=False)
        self.config = objective.SVFConfig(K=3, flow_steps=2)
        self.register_buffer('mask', torch.tensor([1., 1., 0.]))

    def forward(self, inputs, seed):
        features, states, actions = inputs
        def actor_velocity(x, t):
            return self.actor(x.flatten(1)).reshape_as(x)
        def reference_velocity(x, t):
            return self.reference_weight * x
        def teacher_score(endpoints):
            return self.teacher_weight * (endpoints * self.mask).sum((-1, -2))
        loss, metrics = objective.joint_losses(
            self.soft_value, actor_velocity, reference_velocity, teacher_score,
            features, states, actions, self.mask, self.config,
            torch.Generator().manual_seed(seed))
        return {'loss': loss, 'metrics': metrics}


def _flatten_parameters(model):
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])


def _worker(rank, size, rendezvous, results, loss_type):
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=size, timeout=timedelta(seconds=45))
    try:
        config = objective.SVFConfig()
        # Rank 0 spreads [1,2], rank 1 spreads [3,4] but marks the latter invalid.
        # Pooled spread is (1+2+3)/3 = 2, so lambda = .64*2 = 1.28.
        qs = torch.tensor([[0., 0.], [2., 4.]]) if rank == 0 else torch.tensor([[0., 0.], [6., 8.]])
        valid = torch.tensor([True, True]) if rank == 0 else torch.tensor([True, False])
        lam = objective.estimate_lambda(qs, config, valid)
        torch.testing.assert_close(lam, torch.tensor(1.28))
        lambdas = [torch.empty_like(lam) for _ in range(size)]
        dist.all_gather(lambdas, lam)
        for other in lambdas:
            torch.testing.assert_close(other, lam, rtol=0, atol=0)

        torch.manual_seed(71)
        model = TinyJointModel(loss_type)
        initial_actor = model.actor.weight.detach().clone()
        initial_heads = [head[0].weight.detach().clone() for head in model.soft_value.heads]
        frozen = [model.reference_weight.detach().clone(), model.teacher_weight.detach().clone()]
        ddp = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.02)
        observed_losses = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                rng = torch.Generator().manual_seed(1000 + 100 * step + 10 * micro + rank)
                inputs = (torch.randn(2, 1, 4, generator=rng),
                          torch.randn(2, 1, 3, generator=rng),
                          torch.randn(2, 2, 3, generator=rng))
                sync = ddp.no_sync() if micro == 0 else nullcontext()
                with sync:
                    output = ddp(inputs, 9000 + 100 * step + 10 * micro + rank)
                    assert torch.isfinite(output['loss']), 'nonfinite joint loss'
                    (output['loss'] / 2).backward()
                observed_losses.append(float(output['loss'].detach()))
                gathered = [torch.empty_like(output['metrics']['lambda']) for _ in range(size)]
                dist.all_gather(gathered, output['metrics']['lambda'])
                for value in gathered:
                    torch.testing.assert_close(value, gathered[0], rtol=0, atol=0)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f'missing gradient: {name}'
                    assert torch.isfinite(param.grad).all(), f'nonfinite gradient: {name}'
                else:
                    assert param.grad is None, f'frozen parameter got gradient: {name}'
            optimizer.step()
            vector = _flatten_parameters(model)
            replicas = [torch.empty_like(vector) for _ in range(size)]
            dist.all_gather(replicas, vector)
            for replica in replicas:
                torch.testing.assert_close(replica, vector, rtol=1e-6, atol=1e-7)
        assert not torch.equal(initial_actor, model.actor.weight), 'actor did not update'
        for initial, head in zip(initial_heads, model.soft_value.heads):
            assert not torch.equal(initial, head[0].weight), 'value head did not update'
        torch.testing.assert_close(frozen[0], model.reference_weight, rtol=0, atol=0)
        torch.testing.assert_close(frozen[1], model.teacher_weight, rtol=0, atol=0)
        if rank == 0:
            Path(results).write_text(json.dumps({'ranks': size, 'optimizer_steps': 3,
                'accumulation_steps': 2, 'pooled_lambda': float(lam),
                'losses': observed_losses, 'all_replicas_match': True,
                'actor_and_both_value_heads_updated': True, 'teachers_unchanged': True}))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo is unavailable')
class SVFDistributedTests(unittest.TestCase):
    def test_two_rank_joint_input_gradient_and_accumulation(self):
        self._check_two_rank_joint("mse")

    def test_two_rank_ce_input_gradient_and_accumulation(self):
        self._check_two_rank_joint("hl-gauss")

    def _check_two_rank_joint(self, loss_type):
        scratch = REPO / 'output/code-checks/svf-ddp'
        scratch.mkdir(parents=True, exist_ok=True)
        # tempfile is explicitly confined to this project-owned directory.
        with tempfile.TemporaryDirectory(prefix='cpu-gloo-', dir=scratch) as directory:
            rendezvous = str(Path(directory) / 'rendezvous')
            results = str(Path(directory) / 'result.json')
            # torch.multiprocessing itself uses NamedTemporaryFile for child
            # error reports. Confine that internal file too, including child
            # interpreters whose tempfile cache has not been initialized yet.
            previous_tempdir = tempfile.tempdir
            previous_tmpdir_env = os.environ.get('TMPDIR')
            tempfile.tempdir = directory
            os.environ['TMPDIR'] = directory
            try:
                mp.spawn(_worker, args=(2, rendezvous, results, loss_type), nprocs=2, join=True)
            finally:
                tempfile.tempdir = previous_tempdir
                if previous_tmpdir_env is None:
                    os.environ.pop('TMPDIR', None)
                else:
                    os.environ['TMPDIR'] = previous_tmpdir_env
            record = json.loads(Path(results).read_text())
            self.assertEqual(record['optimizer_steps'], 3)
            self.assertEqual(record['accumulation_steps'], 2)
            self.assertTrue(record['all_replicas_match'])
            self.assertTrue(record['actor_and_both_value_heads_updated'])
            self.assertTrue(record['teachers_unchanged'])
            self.assertAlmostEqual(record['pooled_lambda'], 1.28, places=6)


if __name__ == '__main__':
    unittest.main()
