"""CPU checks of SVF mathematics, masking and optimizer separation."""

import math
import unittest

import torch
from torch import nn

from gr00t.model.svf import (DoubleSoftValue, SVFConfig, actor_mse, estimate_lambda,
                             fourier_time_embedding, guidance_from_value,
                             joint_losses, reference_endpoints, soft_value_target)


class SVFMathTests(unittest.TestCase):
    def test_fourier_angular_frequency_and_double_head_shapes(self):
        t = torch.tensor([0.0, 0.25])
        embedded = fourier_time_embedding(t, 4)
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0],
                                 [math.sin(.25), math.sin(64.0), math.cos(.25), math.cos(64.0)]])
        torch.testing.assert_close(embedded, expected, rtol=1e-5, atol=1e-5)
        value = DoubleSoftValue(4, 3, 2, 3, time_dim=4, hidden_dim=8, depth=2)
        result = value(torch.zeros(2, 1, 4), torch.zeros(2, 1, 3), torch.zeros(2, 2, 3), t)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(set(map(id, value.heads[0].parameters())).isdisjoint(
            set(map(id, value.heads[1].parameters()))))

    def test_soft_value_constant_shift_and_temperature_scale(self):
        config = SVFConfig(kappa=.4, g=.25)
        self.assertAlmostEqual(config.c, .64)
        qs = torch.tensor([[1.0, 4.0], [3.0, 8.0]], requires_grad=True)
        lam = estimate_lambda(qs, config)
        torch.testing.assert_close(lam, torch.tensor(.64 * 1.5))
        self.assertFalse(lam.requires_grad)
        target = soft_value_target(qs, lam)
        torch.testing.assert_close(soft_value_target(qs + 10, lam), target + 10)
        torch.testing.assert_close(soft_value_target(torch.full((8, 2), 1234.), .001), torch.full((2,), 1234.))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(estimate_lambda(torch.ones(2, 2), config), torch.tensor(.00064))

    def test_temperature_ignores_invalid_samples(self):
        qs = torch.tensor([[0., 0.], [2., 10000.]])
        lam = estimate_lambda(qs, SVFConfig(), torch.tensor([True, False]))
        torch.testing.assert_close(lam, torch.tensor(.64))

    def test_vectorized_reference_is_euler_maruyama_and_preserves_padding(self):
        config = SVFConfig(K=3, flow_steps=1)
        x = torch.tensor([[[.2, 5.]], [[-.3, 6.]]])
        t = torch.tensor([.5, .75])
        mask = torch.tensor([1., 0.])
        seen = []
        def velocity(latent, time):
            seen.append((latent.clone(), time.clone(), torch.is_grad_enabled()))
            return torch.full_like(latent, .1)
        actual = reference_endpoints(velocity, x, t, mask, config, torch.Generator().manual_seed(11))
        initial = x.repeat(3, 1, 1)
        s = t.repeat(3)[:, None, None]
        dt = 1 - s
        noise = torch.randn(initial.shape, generator=torch.Generator().manual_seed(11))
        expected = initial + (.1 - config.kappa ** 2 * (initial - s * .1) / s) * dt
        expected += config.kappa * (2 * (1 - s) / s * dt).sqrt() * noise
        expected = expected.reshape(3, 2, 1, 2)
        expected[..., 0].clamp_(-1, 1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0][0], initial)
        torch.testing.assert_close(seen[0][1], t.repeat(3))
        self.assertFalse(seen[0][2])
        self.assertTrue((actual[..., 1] > 1).all())

    def test_guidance_direction_schedule_padding_and_no_parameter_grad(self):
        weight = nn.Parameter(torch.tensor(2.0))
        x = torch.tensor([[[1., 100.]], [[2., 200.]]], requires_grad=True)
        def value(latent):
            scalar = weight * latent[:, 0, 0] + 100 * latent[:, 0, 1]
            return torch.stack((scalar, scalar), dim=-1)
        config = SVFConfig(guidance_clip=None)
        guidance, info = guidance_from_value(value, x, torch.tensor([.5, .05]),
                                             torch.ones_like(x), torch.tensor([1., 0.]), config, 2.)
        torch.testing.assert_close(guidance, torch.tensor([[[.16, 0.]], [[0., 0.]]]))
        self.assertFalse(guidance.requires_grad)
        self.assertIsNone(weight.grad)
        prediction = nn.Parameter(torch.zeros_like(x))
        actor_mse(prediction, guidance, torch.tensor([1., 0.])).backward()
        self.assertIsNone(weight.grad)
        self.assertIsNone(x.grad)
        self.assertTrue((prediction.grad[..., 1] == 0).all())
        self.assertGreater(float(info['guidance_norm']), 0)

    def test_guidance_clips_by_valid_base_norm(self):
        x = torch.zeros(2, 1, 2)
        guidance, info = guidance_from_value(
            lambda latent: 100 * latent[:, 0, 0], x, torch.full((2,), .5),
            torch.tensor([[[1., 10000.]], [[1., 10000.]]]), torch.tensor([1., 0.]),
            SVFConfig(guidance_clip=.1), 1.)
        torch.testing.assert_close(guidance[:, 0, 0], torch.full((2,), .1))
        torch.testing.assert_close(info['clip_fraction'], torch.tensor(1.))
        torch.testing.assert_close(info['base_velocity_norm'], torch.tensor(1.))

    def test_actor_loss_mask_and_all_invalid_batch(self):
        prediction = nn.Parameter(torch.tensor([[[1., 100.], [3., 200.]]]))
        loss = actor_mse(prediction, torch.zeros_like(prediction), torch.tensor([1., 0.]))
        torch.testing.assert_close(loss, torch.tensor(5.))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[[1., 0.], [3., 0.]]]))
        zero = actor_mse(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction))
        torch.testing.assert_close(zero, torch.tensor(0.))

    def test_joint_update_trains_actor_and_both_value_heads_only(self):
        torch.manual_seed(12)
        batch, horizon, dim = 4, 2, 3
        config = SVFConfig(K=3, flow_steps=2)
        features = torch.randn(batch, 1, 4, requires_grad=True)
        states = torch.randn(batch, 1, 3, requires_grad=True)
        actions = torch.randn(batch, horizon, dim)
        mask = torch.tensor([1., 1., 0.])
        soft_value = DoubleSoftValue(4, 3, horizon, dim, time_dim=4, hidden_dim=16, depth=2)
        actor = nn.Linear(horizon * dim, horizon * dim)
        reference_weight = nn.Parameter(torch.tensor(.2))
        q_weight = nn.Parameter(torch.tensor(.7))
        q_inputs, reference_calls, value_inputs = [], [], []
        soft_value.register_forward_pre_hook(lambda module, args: value_inputs.append(args[2].detach().clone()))
        def actor_velocity(x, t):
            return actor(x.flatten(1)).reshape_as(x)
        def reference_velocity(x, t):
            reference_calls.append((x.shape, torch.is_grad_enabled()))
            return reference_weight * x
        def teacher_score(endpoints):
            q_inputs.append(endpoints.clone())
            return q_weight * (endpoints * mask).sum(dim=(-1, -2))
        loss, metrics = joint_losses(soft_value, actor_velocity, reference_velocity, teacher_score,
                                     features, states, actions, mask, config,
                                     torch.Generator().manual_seed(123))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(actor.weight.grad.abs().sum()), 0.)
        for head in soft_value.heads:
            self.assertGreater(float(head[0].weight.grad.abs().sum()), 0.)
        self.assertIsNone(reference_weight.grad)
        self.assertIsNone(q_weight.grad)
        self.assertIsNone(features.grad)
        self.assertIsNone(states.grad)
        self.assertEqual(len(q_inputs), 2)
        self.assertFalse(torch.equal(q_inputs[0], q_inputs[1]))
        self.assertEqual(len(reference_calls), 2 * config.flow_steps)
        self.assertTrue(all(shape[0] == batch * config.K and not grad for shape, grad in reference_calls))
        self.assertTrue(all((value[..., -1] == 0).all() for value in value_inputs))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))


if __name__ == '__main__':
    unittest.main()
