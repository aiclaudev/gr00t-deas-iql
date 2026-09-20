"""CPU contract tests for SVF adapters; no checkpoints, VLMs or environments."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from gr00t.model.svf.adapters import (
    ActionCoordinateAdapter, prepare_head_context, velocity_from_head,
    prepare_critic_features, score_critic,
)


def metadata():
    state_shapes = {
        "end_effector_position_relative": 3, "end_effector_rotation_relative": 4,
        "gripper_qpos": 2, "base_position": 3, "base_rotation": 4,
    }
    action_shapes = {
        "end_effector_position": 3, "end_effector_rotation": 3,
        "gripper_close": 1, "base_motion": 4, "control_mode": 1,
    }
    result = {"statistics": {}, "modalities": {}}
    for modality, shapes in (("state", state_shapes), ("action", action_shapes)):
        result["statistics"][modality] = {}
        result["modalities"][modality] = {}
        for key, width in shapes.items():
            binary = modality == "action" and key in {"gripper_close", "control_mode"}
            result["statistics"][modality][key] = {
                "min": [0.0 if binary else -1.0] * width, "max": [1.0] * width,
            }
            result["modalities"][modality][key] = {
                "shape": [width], "continuous": not binary, "absolute": True,
                "rotation_type": "quaternion" if modality == "state" and "rotation" in key else None,
            }
    return result


class CoordinateTests(unittest.TestCase):
    def test_roundtrip_separate_ranges_and_padding(self):
        actor, critic = metadata(), metadata()
        critic["statistics"]["action"]["end_effector_position"] = {
            "min": [-3.0] * 3, "max": [5.0] * 3,
        }
        critic["statistics"]["state"]["base_position"] = {
            "min": [-4.0] * 3, "max": [2.0] * 3,
        }
        forward = ActionCoordinateAdapter.from_metadata({"new_embodiment": actor}, critic)
        reverse = ActionCoordinateAdapter.from_metadata(critic, actor)
        x = torch.randn(2, 16, 32, requires_grad=True)
        y = forward.convert_action(x, threshold_binary=False)
        self.assertTrue(torch.allclose(y[..., :3], x[..., :3] / 4 - .25))
        self.assertTrue(torch.equal(y[..., 12:], torch.zeros_like(y[..., 12:])))
        recovered = reverse.convert_action(y, threshold_binary=False)
        self.assertTrue(torch.allclose(recovered[..., :12], x[..., :12], atol=1e-6))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad[..., 12:], torch.zeros_like(x.grad[..., 12:])))
        state = torch.randn(2, 1, 64)
        state_roundtrip = reverse.convert_state(forward.convert_state(state))
        self.assertTrue(torch.allclose(state_roundtrip[..., :20], state[..., :20], atol=1e-6))
        self.assertEqual(int(forward.action_mask.sum()), 12)
        self.assertEqual(int(forward.state_mask.sum()), 20)

    def test_binary_threshold_and_unclipped_minmax(self):
        adapter = ActionCoordinateAdapter.from_metadata(metadata(), metadata())
        action = torch.full((2, 16, 32), 4.0)
        action[0, :, 6] = .5
        action[1, :, 6] = .5001
        y = adapter.convert_action(action)
        self.assertTrue(torch.equal(y[..., :3], action[..., :3]))  # min_max does not saturate
        self.assertTrue((y[0, :, 6] == 0).all())
        self.assertTrue((y[1, :, 6] == 1).all())
        self.assertTrue((y[..., 11] == 1).all())
        self.assertTrue((y[..., 12:] == 0).all())

    def test_rotation6d_uses_fixed_bounds_and_constants_zero(self):
        actor, critic = metadata(), metadata()
        critic["statistics"]["state"]["base_rotation"] = {"min": [0.] * 4, "max": [.2] * 4}
        critic["statistics"]["action"]["base_motion"] = {"min": [2.] * 4, "max": [2.] * 4}
        adapter = ActionCoordinateAdapter.from_metadata(actor, critic)
        state = torch.randn(1, 1, 64)
        self.assertTrue(torch.equal(adapter.convert_state(state)[..., 14:20], state[..., 14:20]))
        action = torch.randn(1, 16, 32)
        self.assertTrue((adapter.convert_action(action)[..., 7:11] == 0).all())

    def test_layout_mismatch_fails(self):
        actor, critic = metadata(), metadata()
        critic["statistics"]["action"]["base_motion"] = {"min": [-1.] * 3, "max": [1.] * 3}
        critic["modalities"]["action"]["base_motion"]["shape"] = [3]
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            ActionCoordinateAdapter.from_metadata(actor, critic)


class CategoryIdentity(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, x, ids):
        return self.linear(x)


class ActionEncoder(CategoryIdentity):
    def forward(self, x, times, ids):
        self.times = times.detach().clone()
        return self.linear(x) + times[:, None, None] / 1000


class ToyDiT(nn.Module):
    def forward(self, **kwargs):
        self.arguments = kwargs
        h = kwargs["hidden_states"]
        features = kwargs["encoder_hidden_states"]
        mask = kwargs["encoder_attention_mask"].unsqueeze(-1)
        return h + h.mean(1, keepdim=True) + (features * mask).sum(1, keepdim=True)


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_dim=3, add_pos_embed=True)
        self.num_timestep_buckets = 1000
        self.vlln = nn.Linear(3, 3, bias=False)
        self.vl_self_attention = nn.Linear(3, 3, bias=False)
        self.state_encoder = CategoryIdentity()
        self.action_encoder = ActionEncoder()
        self.action_decoder = CategoryIdentity()
        self.future_tokens = nn.Embedding(2, 3)
        self.position_embedding = nn.Embedding(16, 3)
        self.model = ToyDiT()
        with torch.no_grad():
            nn.init.eye_(self.vlln.weight)
            nn.init.eye_(self.vl_self_attention.weight)
            self.future_tokens.weight.fill_(2.)
            self.position_embedding.weight.fill_(.25)


class VelocityTests(unittest.TestCase):
    def test_velocity_uses_future_tokens_mask_positions_and_gradients(self):
        head = ToyHead()
        raw = {"backbone_features": torch.tensor([[[1., 2., 3.], [90., 90., 90.]]]),
               "backbone_attention_mask": torch.tensor([[1., 0.]])}
        original = raw["backbone_features"].clone()
        inputs = {"state": torch.ones(1, 1, 3), "embodiment_id": torch.tensor([0])}
        context = prepare_head_context(head, raw, inputs)
        x = torch.zeros(1, 2, 3, requires_grad=True)
        velocity = velocity_from_head(head, context, x, .2519)
        action_embedding = torch.full((1, 2, 3), .501)
        hidden = torch.cat((inputs["state"], head.future_tokens.weight[None], action_embedding), 1)
        expected = action_embedding + hidden.mean(1, keepdim=True) + original[:, :1]
        self.assertTrue(torch.allclose(velocity, expected))
        self.assertTrue(torch.equal(head.action_encoder.times, torch.tensor([251])))
        self.assertIs(head.model.arguments["encoder_attention_mask"], raw["backbone_attention_mask"])
        self.assertFalse(head.model.arguments["return_all_hidden_states"])
        self.assertTrue(torch.equal(raw["backbone_features"], original))
        velocity.sum().backward()
        for parameter in (head.future_tokens.weight, head.state_encoder.linear.weight,
                          head.vlln.weight, head.vl_self_attention.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_block_repeat_slice_and_mixed_dtype(self):
        head = ToyHead().double()
        raw = {"backbone_features": torch.ones(2, 1, 3),
               "backbone_attention_mask": torch.ones(2, 1)}
        inputs = {"state": torch.zeros(2, 1, 3), "embodiment_id": torch.tensor([4, 9])}
        context = prepare_head_context(head, raw, inputs)
        repeated = context.repeat_batches(3)
        self.assertEqual(repeated.embodiment_id.tolist(), [4, 9, 4, 9, 4, 9])
        self.assertEqual(repeated.slice_batch(1, 4).embodiment_id.tolist(), [9, 4, 9])
        x = torch.ones(6, 2, 3, requires_grad=True)
        y = velocity_from_head(head, repeated, x, torch.arange(6) / 10)
        self.assertEqual(y.dtype, torch.float64)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        with self.assertRaisesRegex(ValueError, "one value"):
            velocity_from_head(head, repeated, x, torch.ones(2))


class FakeDoubleQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))

    def forward(self, features, state, action):
        q = action.flatten(1).sum(1) * self.weight
        zeros = torch.zeros_like(q)
        return torch.stack((zeros, q), -1), torch.stack((zeros, q + 1), -1)


class FakeHLG:
    def transform_from_probs(self, probs):
        return (probs * torch.tensor([-2., -1.], device=probs.device)).sum(-1)


class ToyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlln = nn.Linear(3, 3)
        self.vl_self_attention = nn.Linear(3, 3)
        with torch.no_grad():
            self.vlln.weight.copy_(torch.eye(3) * 2)
            self.vlln.bias.fill_(1.)
            self.vl_self_attention.weight.copy_(torch.eye(3) * .5)
            self.vl_self_attention.bias.fill_(.5)
        self.backbone_encoder = CategoryIdentity()
        self.critic = FakeDoubleQ()
        self.hlg = FakeHLG()
        self.critic_action_horizon = 2


class CriticTests(unittest.TestCase):
    def test_two_pass_training_parity_and_no_mutation(self):
        head = ToyCritic()
        full = nn.Module()
        full.add_module("critic_head", head)
        raw = {"backbone_features": torch.tensor([[[.1, .2, .3], [.3, .4, .5]]], requires_grad=True)}
        original = raw["backbone_features"].detach().clone()
        ids = torch.tensor([0])
        # Both layers compose to x + 1; legacy online-Q training applies it twice.
        expected = torch.tanh((original + 2).mean(1, keepdim=True))
        actual = prepare_critic_features(full, raw, ids)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertFalse(actual.requires_grad)
        self.assertTrue(torch.equal(raw["backbone_features"], original))
        self.assertFalse(head.training)
        once = prepare_critic_features(head, raw, ids, passes=1)
        self.assertFalse(torch.allclose(once, actual))

    def test_online_min_expectation_and_action_gradient(self):
        head = ToyCritic().double().requires_grad_(False)
        actions = torch.zeros(2, 3, 3, requires_grad=True)
        q = score_critic(head, torch.zeros(2, 1, 3), torch.zeros(2, 1, 3), actions)
        self.assertTrue(torch.allclose(q, torch.full((2,), -1.5)))
        q.sum().backward()
        self.assertGreater(float(actions.grad[:, :2].abs().sum()), 0)
        self.assertEqual(float(actions.grad[:, 2:].abs().sum()), 0)
        self.assertIsNone(head.critic.weight.grad)
        with self.assertRaisesRegex(ValueError, "trained horizon"):
            score_critic(head, torch.zeros(2, 1, 3), torch.zeros(2, 1, 3), actions[:, :1])


if __name__ == "__main__":
    unittest.main()
