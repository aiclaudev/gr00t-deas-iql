"""CPU validation of critic-trunk transfer and its deliberate limitations."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from gr00t.model.critic.networks import DoubleCritic
from gr00t.model.critic.hlg import HLGaussLoss
from gr00t.model.svf.adapters import ActionCoordinateAdapter
from gr00t.model.svf.initialization import initialize_soft_value_from_critic
from gr00t.model.svf.networks import DoubleSoftValue, fourier_time_embedding


def example(loss_type="mse"):
    options = {} if loss_type == "mse" else {
        "loss_type": "hl-gauss", "num_bins": 101, "value_min": -100., "value_max": 0., "sigma": 0.75 * 100 / 101,
    }
    value = DoubleSoftValue(feature_dim=2, state_dim=3, action_horizon=2,
                            action_dim=4, time_dim=4, hidden_dim=8, depth=2, **options)
    teacher = SimpleNamespace(critic=DoubleCritic(13, [8, 8], output_dim=101),
                              hlg=HLGaussLoss(-100., 0., 101, 0.75 * 100 / 101))
    teacher.critic.requires_grad_(False)
    coordinates = ActionCoordinateAdapter(
        torch.ones(3), torch.zeros(3), torch.ones(3, dtype=torch.bool),
        torch.tensor([-1.5, 0., 1., 9.]), torch.tensor([.3, -.8, 0., 8.]),
        torch.tensor([True, True, True, False]),
        torch.tensor([False, False, True, False]),
    )
    return value, teacher, coordinates


def snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


class SVFInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        torch.set_num_threads(1)

    def assert_snapshot(self, module, saved):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

    def test_continuous_trunk_matches_affine_critic_coordinates(self):
        value, teacher, coordinates = example()
        initialize_soft_value_from_critic(value, teacher, coordinates)
        features, states, action = torch.randn(3, 2), torch.randn(3, 3), torch.randn(3, 2, 4)
        action[:, :, 3] = 10000  # padding remains invisible to both paths
        time = torch.tensor([.1, .5, .8])
        inputs = torch.cat((features, states, action.flatten(1), fourier_time_embedding(time, 4)), -1)
        critic_inputs = torch.cat((features, states, coordinates.convert_action(action, threshold_binary=False).flatten(1)), -1)
        for destination, source in zip(value.heads, (teacher.critic.Q1.mlp, teacher.critic.Q2.mlp)):
            torch.testing.assert_close(destination[:-1](inputs), source[:-1](critic_inputs), rtol=2e-5, atol=2e-6)
            self.assertEqual(torch.count_nonzero(destination[0].weight[:, 13:]).item(), 0)
            for index in (8, 12):  # action channel 3 in each 4-wide timestep
                self.assertEqual(torch.count_nonzero(destination[0].weight[:, index]).item(), 0)

    def test_binary_endpoints_match_teacher_but_intermediate_extension_is_smooth(self):
        value, teacher, coordinates = example()
        initialize_soft_value_from_critic(value, teacher, coordinates)
        action = torch.randn(2, 2, 4)
        action[:, :, 2] = torch.tensor([[0., 1.], [1., 0.]])
        features, states, time = torch.randn(2, 2), torch.randn(2, 3), torch.tensor([.2, .9])
        inputs = torch.cat((features, states, action.flatten(1), fourier_time_embedding(time, 4)), -1)
        critic_inputs = torch.cat((features, states, coordinates.convert_action(action).flatten(1)), -1)
        for destination, source in zip(value.heads, (teacher.critic.Q1.mlp, teacher.critic.Q2.mlp)):
            torch.testing.assert_close(destination[:-1](inputs), source[:-1](critic_inputs), rtol=2e-5, atol=2e-6)
        action[:, :, 2] = .25
        action.requires_grad_()
        value(features, states, action, time).sum().backward()
        self.assertGreater(action.grad[:, :, 2].abs().sum().item(), 0)
        self.assertFalse(torch.equal(coordinates.convert_action(action),
                                     coordinates.convert_action(action, threshold_binary=False)))

    def test_initial_time_invariance_and_time_columns_can_learn(self):
        value, teacher, coordinates = example()
        initialize_soft_value_from_critic(value, teacher, coordinates)
        features, states, action = torch.randn(3, 2), torch.randn(3, 3), torch.randn(3, 2, 4)
        early = value(features, states, action, torch.tensor([.1, .2, .3]))
        late = value(features, states, action, torch.tensor([.6, .7, .8]))
        torch.testing.assert_close(early, late, rtol=0, atol=0)
        (early - 1).square().mean().backward()
        for head in value.heads:
            self.assertGreater(head[0].weight.grad[:, 13:].abs().sum().item(), 0)

    def test_teacher_immutable_scalar_readout_unchanged_and_independent_storage(self):
        value, teacher, coordinates = example()
        source_before = snapshot(teacher.critic)
        final_before = [snapshot(head[-1]) for head in value.heads]
        provenance = initialize_soft_value_from_critic(value, teacher, coordinates)
        self.assert_snapshot(teacher.critic, source_before)
        for head, saved in zip(value.heads, final_before):
            self.assert_snapshot(head[-1], saved)
        self.assertFalse(provenance["preserves_q_output"])
        self.assertEqual(provenance["mode"], "critic-trunk")
        self.assertEqual(provenance["binary_channels"], [2])
        self.assertEqual(provenance["scalar_readout"], "original_random_initialization_unchanged")
        self.assertTrue(all(not p.requires_grad for p in teacher.critic.parameters()))
        self.assertTrue(all(p.requires_grad for p in value.parameters()))
        with torch.no_grad():
            value.heads[0][0].weight.add_(1)
            value.heads[1][4].weight.add_(1)
        self.assert_snapshot(teacher.critic, source_before)

    def test_bfloat16_source_is_copied_into_fp32_destination(self):
        value, teacher, coordinates = example()
        teacher.critic.bfloat16()
        initialize_soft_value_from_critic(value, teacher, coordinates)
        self.assertTrue(all(p.dtype == torch.float32 for p in value.parameters()))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in teacher.critic.parameters()))
        for destination, source in zip(value.heads, (teacher.critic.Q1.mlp, teacher.critic.Q2.mlp)):
            torch.testing.assert_close(destination[3].weight, source[3].weight.float(), rtol=0, atol=0)

    def test_invalid_second_head_or_layernorm_is_atomic(self):
        for failure in ("second_head", "layernorm", "scalar_output", "activation"):
            with self.subTest(failure=failure):
                value, teacher, coordinates = example()
                if failure == "second_head":
                    teacher.critic.Q2.mlp[3] = nn.Linear(8, 9)
                elif failure == "layernorm":
                    teacher.critic.Q2.mlp[4].eps = .1
                elif failure == "scalar_output":
                    value.heads[1][-1] = nn.Linear(8, 2)
                else:
                    teacher.critic.Q2.mlp[5] = nn.ReLU()
                before = snapshot(value)
                with self.assertRaises(ValueError):
                    initialize_soft_value_from_critic(value, teacher, coordinates)
                self.assert_snapshot(value, before)

    def test_nonfinite_source_or_coordinate_is_atomic(self):
        for failure in ("source", "coordinate", "fold_overflow"):
            with self.subTest(failure=failure):
                value, teacher, coordinates = example()
                with torch.no_grad():
                    if failure == "source":
                        teacher.critic.Q2.mlp[3].weight[0, 0] = float("nan")
                    elif failure == "coordinate":
                        coordinates.action_bias[0] = float("inf")
                    else:
                        teacher.critic.Q2.mlp[0].weight[:, 5] = 1e30
                        coordinates.action_scale[0] = 1e30
                before = snapshot(value)
                with self.assertRaisesRegex(ValueError, "finite"):
                    initialize_soft_value_from_critic(value, teacher, coordinates)
                self.assert_snapshot(value, before)

    def test_hlgauss_trunk_only_keeps_random_categorical_readout(self):
        value, teacher, coordinates = example("hl-gauss")
        teacher.hlg.sigma *= 2  # Trunk transfer does not promise output equivalence.
        final_before = [snapshot(head[-1]) for head in value.heads]
        provenance = initialize_soft_value_from_critic(value, teacher, coordinates)
        self.assertEqual(provenance["mode"], "critic-trunk")
        self.assertEqual(provenance["loss_type"], "hl-gauss")
        self.assertFalse(provenance["preserves_each_continuous_q_head_at_initialization"])
        for head, saved in zip(value.heads, final_before):
            self.assert_snapshot(head[-1], saved)

    def test_full_hlgauss_logits_and_expectations_match_continuous_q_heads(self):
        value, teacher, coordinates = example("hl-gauss")
        teacher_before = snapshot(teacher.critic)
        provenance = initialize_soft_value_from_critic(value, teacher, coordinates, copy_output=True)
        features, states = torch.randn(4, 2), torch.randn(4, 3)
        action, time = torch.randn(4, 2, 4), torch.tensor([.1, .4, .7, 1.])
        action[:, :, 2] = .25  # Deliberately tests smooth binary-coordinate extension.
        action[:, :, 3] = 9999  # Padding must remain invisible.
        critic_actions = coordinates.convert_action(action, threshold_binary=False)
        source_logits = torch.stack(teacher.critic(features, states, critic_actions), dim=1)
        actual_logits = value.forward_logits(features, states, action, time)
        torch.testing.assert_close(actual_logits, source_logits, rtol=3e-5, atol=3e-6)
        expectations = teacher.hlg.transform_from_probs(torch.softmax(source_logits.float(), dim=-1))
        torch.testing.assert_close(value(features, states, action, time), expectations, rtol=2e-6, atol=1e-5)
        for destination, source in zip(value.heads, (teacher.critic.Q1.mlp, teacher.critic.Q2.mlp)):
            torch.testing.assert_close(destination[-1].weight, source[-1].weight, rtol=0, atol=0)
            torch.testing.assert_close(destination[-1].bias, source[-1].bias, rtol=0, atol=0)
        self.assertEqual(provenance["mode"], "critic-full")
        self.assertEqual(provenance["hl_gauss_configuration"], value.loss_configuration())
        self.assertTrue(provenance["preserves_each_continuous_q_head_at_initialization"])
        self.assertFalse(provenance["preserves_q_output"])
        self.assertEqual(len(provenance["q_equivalence_limitations"]), 2)
        self.assert_snapshot(teacher.critic, teacher_before)
        with torch.no_grad():
            value.heads[0][-1].weight.add_(1.)
        self.assert_snapshot(teacher.critic, teacher_before)

    def test_full_hlgauss_time_columns_and_binary_guidance_have_gradients(self):
        value, teacher, coordinates = example("hl-gauss")
        initialize_soft_value_from_critic(value, teacher, coordinates, copy_output=True)
        features, states = torch.randn(3, 2), torch.randn(3, 3)
        action = torch.randn(3, 2, 4, requires_grad=True)
        early = value(features, states, action, torch.tensor([.1, .2, .3]))
        late = value(features, states, action, torch.tensor([.6, .7, .8]))
        torch.testing.assert_close(early, late, rtol=0, atol=0)
        (early + 10).square().mean().backward()
        for head in value.heads:
            self.assertGreater(head[0].weight.grad[:, 13:].abs().sum().item(), 0)
        self.assertGreater(action.grad[:, :, 2].abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(action.grad[:, :, 3]).item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.critic.parameters()))

    def test_full_requires_hlgauss_and_rejects_support_mismatch_atomically(self):
        for failure in ("mse", "num_bins", "value_min", "value_max", "sigma", "missing_hlg",
                        "support_edges", "bin_centers", "source_final", "nonfinite_final"):
            with self.subTest(failure=failure):
                value, teacher, coordinates = example("mse" if failure == "mse" else "hl-gauss")
                if failure in {"num_bins", "value_min", "value_max", "sigma"}:
                    key = {"value_min": "min_value", "value_max": "max_value"}.get(failure, failure)
                    setattr(teacher.hlg, key, getattr(teacher.hlg, key) + 1)
                elif failure == "missing_hlg":
                    del teacher.hlg
                elif failure in {"support_edges", "bin_centers"}:
                    with torch.no_grad():
                        getattr(value, failure)[0] += .25
                elif failure == "source_final":
                    teacher.critic.Q2.mlp[-1] = nn.Linear(8, 100)
                elif failure == "nonfinite_final":
                    with torch.no_grad():
                        teacher.critic.Q2.mlp[-1].bias[0] = float("nan")
                before = snapshot(value)
                with self.assertRaises(ValueError):
                    initialize_soft_value_from_critic(value, teacher, coordinates, copy_output=True)
                self.assert_snapshot(value, before)

    def test_full_support_validation_matches_overflow_safe_center_construction(self):
        _, teacher, coordinates = example("hl-gauss")
        value = DoubleSoftValue(feature_dim=2, state_dim=3, action_horizon=2,
            action_dim=4, time_dim=4, hidden_dim=8, depth=2, loss_type="hl-gauss",
            num_bins=101, value_min=-3e38, value_max=-2e38, sigma=1e36)
        teacher.hlg = HLGaussLoss(-3e38, -2e38, 101, 1e36)
        provenance = initialize_soft_value_from_critic(value, teacher, coordinates, copy_output=True)
        self.assertEqual(provenance["mode"], "critic-full")
        self.assertTrue(torch.isfinite(value.bin_centers).all())

    def test_full_hlgauss_readout_alias_is_rejected_atomically(self):
        value, teacher, coordinates = example("hl-gauss")
        value.heads[1][-1].bias = teacher.critic.Q2.mlp[-1].bias
        before = snapshot(value)
        with self.assertRaisesRegex(ValueError, "independent parameter storage"):
            initialize_soft_value_from_critic(value, teacher, coordinates, copy_output=True)
        self.assert_snapshot(value, before)

    def test_shared_storage_is_rejected_before_transfer(self):
        value, teacher, coordinates = example()
        value.heads[0][1].weight = teacher.critic.Q1.mlp[1].weight
        before = snapshot(value)
        with self.assertRaisesRegex(ValueError, "independent parameter storage"):
            initialize_soft_value_from_critic(value, teacher, coordinates)
        self.assert_snapshot(value, before)


if __name__ == "__main__":
    unittest.main()
