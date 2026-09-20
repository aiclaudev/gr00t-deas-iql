"""CPU tests for categorical soft values and differentiable expectation guidance."""
import copy
import io
import unittest

import torch
from torch import nn

from gr00t.model.critic.hlg import HLGaussLoss
from gr00t.model.svf.networks import DoubleSoftValue
from gr00t.model.svf.objective import SVFConfig, joint_losses


def network(**kwargs):
    return DoubleSoftValue(feature_dim=3, state_dim=2, action_horizon=2,
                            action_dim=3, time_dim=4, hidden_dim=12, depth=2, **kwargs)


class SoftValueHLGaussTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        torch.set_num_threads(1)

    def test_mse_preserves_old_parameter_schema_and_direct_scalar_behavior(self):
        implicit, explicit = network(), network(loss_type="mse")
        self.assertEqual(set(implicit.state_dict()), set(explicit.state_dict()))
        self.assertTrue(all(name.startswith("heads.") for name in implicit.state_dict()))
        explicit.load_state_dict(implicit.state_dict(), strict=True)
        features, states, actions, time = torch.randn(2, 3), torch.randn(2, 2), torch.randn(2, 2, 3), torch.rand(2)
        expected = implicit(features, states, actions, time)
        logits = explicit.forward_logits(features, states, actions, time)
        torch.testing.assert_close(expected, logits, rtol=0, atol=0)
        torch.testing.assert_close(explicit.values_from_logits(logits), logits, rtol=0, atol=0)
        target = torch.tensor([1., 2.], requires_grad=True)
        torch.testing.assert_close(explicit.loss_from_logits(logits, target), (expected - target.detach()[:, None]).square())
        self.assertEqual(implicit.loss_configuration(), {"loss_type": "mse"})

    def test_projection_is_finite_normalized_at_support_edges_and_matches_teacher(self):
        value = network(loss_type="hl-gauss", sigma=.75 * 100 / 101)
        targets = torch.tensor([-100., -99.5, -50., -.5, 0.], requires_grad=True)
        probabilities = value.target_probabilities(targets)
        self.assertEqual(probabilities.shape, (5, 101))
        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertFalse(probabilities.requires_grad)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all())
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(5), rtol=0, atol=2e-7)
        teacher = HLGaussLoss(-100., 0., 101, value.sigma)
        torch.testing.assert_close(probabilities, teacher.transform_to_probs(targets.detach()), rtol=2e-5, atol=2e-7)
        means = (probabilities * value.bin_centers).sum(-1)
        self.assertGreater(means[0].item(), -100.)
        self.assertLess(means[-1].item(), 0.)
        tiny_sigma = network(loss_type="hl-gauss", sigma=1e-5)
        projected = tiny_sigma.target_probabilities(torch.tensor([-100., 0., -50.]))
        self.assertTrue(torch.isfinite(projected).all())
        torch.testing.assert_close(projected.sum(-1), torch.ones(3), rtol=0, atol=1e-7)

    def test_projection_is_continuous_across_bin_boundaries(self):
        value = network(loss_type="hl-gauss")
        boundary = value.support_edges[50]
        targets = torch.stack((boundary - 1e-4, boundary, boundary + 1e-4))
        probabilities = value.target_probabilities(targets)
        self.assertLess((probabilities[0] - probabilities[1]).abs().sum().item(), .001)
        self.assertLess((probabilities[2] - probabilities[1]).abs().sum().item(), .001)
        for target in (torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                       torch.tensor([-100.01]), torch.tensor([.01]), torch.zeros(2, 1)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                value.target_probabilities(target)

    def test_ce_is_soft_label_cross_entropy_and_target_is_detached(self):
        value = network(loss_type="hl-gauss", num_bins=9, value_min=-4., value_max=0., sigma=.2)
        logits = torch.randn(3, 2, 9, requires_grad=True)
        targets = torch.tensor([-3., -2., -1.], requires_grad=True)
        loss = value.loss_from_logits(logits, targets)
        probabilities = value.target_probabilities(targets)
        expected = -(probabilities[:, None] * logits.log_softmax(-1)).sum(-1)
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        torch.testing.assert_close(logits.grad, (logits.detach().softmax(-1) - probabilities[:, None]) / 6, rtol=1e-5, atol=1e-7)
        self.assertIsNone(targets.grad)

    def test_forward_expectation_gradient_matches_finite_difference(self):
        value = network(loss_type="hl-gauss", num_bins=9, value_min=-4., value_max=0., sigma=.2)
        features, states, time = torch.randn(1, 3), torch.randn(1, 2), torch.tensor([.4])
        actions = torch.randn(1, 2, 3, requires_grad=True)
        prediction = value(features, states, actions, time)
        logits = value.forward_logits(features, states, actions, time)
        self.assertEqual(prediction.shape, (1, 2))
        self.assertEqual(logits.shape, (1, 2, 9))
        torch.testing.assert_close(prediction, (logits.softmax(-1) * value.bin_centers).sum(-1))
        derivative = torch.autograd.grad(prediction.mean(), actions)[0]
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertGreater(derivative.abs().sum().item(), 0)
        for index in ((0, 0, 0), (0, 1, 2)):
            delta = torch.zeros_like(actions)
            delta[index] = .002
            with torch.no_grad():
                finite_difference = (value(features, states, actions + delta, time).mean() -
                                     value(features, states, actions - delta, time).mean()) / .004
            torch.testing.assert_close(derivative[index], finite_difference, rtol=.02, atol=1e-4)
        self.assertTrue(all(p.grad is None for p in value.parameters()))

    def test_state_dict_roundtrip_preserves_support_and_predictions(self):
        value = network(loss_type="hl-gauss", num_bins=9, value_min=-4., value_max=0., sigma=.2)
        config = value.loss_configuration()
        clone = network(**config)
        stream = io.BytesIO()
        torch.save(value.state_dict(), stream)
        stream.seek(0)
        clone.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        for name, tensor in value.state_dict().items():
            torch.testing.assert_close(tensor, clone.state_dict()[name], rtol=0, atol=0)
        self.assertIn("support_edges", value.state_dict())
        self.assertIn("bin_centers", value.state_dict())
        with self.assertRaises(RuntimeError):
            network().load_state_dict(value.state_dict(), strict=True)

    def test_joint_ce_trains_actor_and_value_with_teacher_gradient_isolation(self):
        value = network(loss_type="hl-gauss", num_bins=11, value_min=-10., value_max=0., sigma=.5)
        actor = nn.Linear(6, 6)
        ref_weight, q_weight = nn.Parameter(torch.tensor(.2)), nn.Parameter(torch.tensor(.3))
        features = torch.randn(2, 3, requires_grad=True)
        states = torch.randn(2, 2, requires_grad=True)
        actions = torch.randn(2, 2, 3, requires_grad=True)
        mask = torch.tensor([1., 1., 0.])
        loss, metrics = joint_losses(
            value, lambda x, t: actor(x.flatten(1)).reshape_as(x),
            lambda x, t: ref_weight * x,
            lambda ends: -5 + q_weight * (ends * mask).sum((-1, -2)),
            features, states, actions, mask, SVFConfig(K=3, flow_steps=2),
            torch.Generator().manual_seed(171))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(actor.weight.grad.abs().sum().item(), 0)
        for head in value.heads:
            self.assertGreater(head[-1].weight.grad.abs().sum().item(), 0)
        for parameter in (features, states, actions, ref_weight, q_weight):
            self.assertIsNone(parameter.grad)
        for key in ("value_target_mae", "value_target_rmse", "value_projection_mae",
                    "value_projection_bias", "value_prediction_entropy", "value_target_entropy"):
            self.assertIn(key, metrics)
        self.assertTrue(all(torch.isfinite(item).all() and not item.requires_grad for item in metrics.values()))
        self.assertEqual(torch.count_nonzero(actor.weight.grad[[2, 5]]).item(), 0)

    def test_invalid_sample_features_and_all_invalid_masks_do_not_train_value(self):
        reference = network(loss_type="hl-gauss", num_bins=11, value_min=-10., value_max=0., sigma=.5)
        features, states, actions = torch.randn(2, 3), torch.randn(2, 2), torch.randn(2, 2, 3)
        mask = torch.ones_like(actions)
        mask[1] = 0
        gradients, outputs = [], []
        for changed in (False, True):
            value = copy.deepcopy(reference)
            actor = nn.Linear(6, 6)
            sample_features = features.clone()
            if changed:
                sample_features[1] += 1000
            loss, metrics = joint_losses(value,
                lambda x, t: actor(x.flatten(1)).reshape_as(x),
                lambda x, t: .2 * x, lambda ends: -5 + .2 * ends.sum((-1, -2)),
                sample_features, states, actions, mask, SVFConfig(K=3, flow_steps=2),
                torch.Generator().manual_seed(741))
            loss.backward()
            gradients.append([p.grad.clone() for p in value.parameters()])
            outputs.append(metrics["soft_value_loss"])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for first, second in zip(*gradients):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        value = copy.deepcopy(reference)
        actor = nn.Linear(6, 6)
        loss, metrics = joint_losses(value,
            lambda x, t: actor(x.flatten(1)).reshape_as(x), lambda x, t: .2 * x,
            lambda ends: torch.full(ends.shape[:2], 100.),  # invalid rows bypass support projection
            features, states, actions, torch.zeros_like(actions), SVFConfig(K=3, flow_steps=2),
            torch.Generator().manual_seed(74))
        torch.testing.assert_close(loss, torch.tensor(0.))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad).item() == 0 for p in value.parameters()))
        torch.testing.assert_close(metrics["value_target_mae"], torch.tensor(0.))

    def test_invalid_loss_configuration_is_rejected(self):
        for options in ({"loss_type": "ce"}, {"loss_type": "hl-gauss", "num_bins": 1},
                        {"loss_type": "hl-gauss", "sigma": 0},
                        {"loss_type": "hl-gauss", "sigma": float("nan")},
                        {"loss_type": "hl-gauss", "value_min": 0, "value_max": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                network(**options)


if __name__ == "__main__":
    unittest.main()
