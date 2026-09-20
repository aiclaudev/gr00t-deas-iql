"""CPU integration tests: real JointSVFModel.forward with tiny module stand-ins."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from gr00t.model.svf.adapters import ActionCoordinateAdapter
from gr00t.model.svf.model import JointSVFModel
from gr00t.model.svf.networks import DoubleSoftValue
from gr00t.model.svf.objective import SVFConfig


class Batch(dict):
    def __getattr__(self, name):
        return self[name]


class CategoryLinear(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x, *unused):
        return self.linear(x)


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask,
                timestep, return_all_hidden_states):
        return self.linear(hidden_states + hidden_states.mean(1, keepdim=True)
                           + encoder_hidden_states.mean(1, keepdim=True))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_dim=3, add_pos_embed=False)
        self.num_timestep_buckets = 1000
        self.vlln, self.vl_self_attention = nn.LayerNorm(3), nn.Linear(3, 3)
        self.state_encoder, self.action_encoder, self.action_decoder = [CategoryLinear() for _ in range(3)]
        self.future_tokens = nn.Embedding(2, 3)
        self.model = TinyDiT()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, inputs):
        return Batch(backbone_features=self.linear(inputs["observations"].to(self.linear.weight)),
                     backbone_attention_mask=inputs["attention_mask"])


class TinyQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 2)

    def forward(self, features, states, actions):
        batch = actions.shape[0]
        x = torch.cat([v.reshape(batch, -1) for v in (features, states, actions)], -1)
        q = self.linear(x)
        return q, q + q.flip(-1) * .1


class TinyHLG:
    def transform_from_probs(self, probs):
        return (probs * torch.tensor([-2., -1.], device=probs.device)).sum(-1)


class TinyModel(nn.Module):
    def __init__(self, critic=False):
        super().__init__()
        self.backbone = TinyBackbone()
        if critic:
            head = nn.Module()
            head.vlln, head.vl_self_attention = nn.LayerNorm(3), nn.Linear(3, 3)
            head.backbone_encoder, head.critic = CategoryLinear(), TinyQ()
            head.hlg, head.critic_action_horizon = TinyHLG(), 2
            self.critic_head = head
        else:
            self.action_head = TinyHead()

    def prepare_input(self, inputs):
        return inputs, Batch(inputs)


def tiny_joint(loss_type="mse"):
    model = JointSVFModel.__new__(JointSVFModel)
    nn.Module.__init__(model)
    model.actor = TinyModel()
    model.actor.backbone.requires_grad_(False).to(dtype=torch.bfloat16)
    model.reference_head = copy.deepcopy(model.actor.action_head).requires_grad_(False).to(dtype=torch.bfloat16)
    model.teacher = TinyModel(critic=True).requires_grad_(False).to(dtype=torch.bfloat16)
    model.coordinates = ActionCoordinateAdapter(
        torch.ones(3), torch.zeros(3), torch.tensor([True, True, True]),
        torch.ones(3), torch.zeros(3), torch.tensor([True, True, False]),
        torch.tensor([False, False, False]),
    )
    model.soft_value = DoubleSoftValue(feature_dim=3, state_dim=3, action_horizon=2,
                                       action_dim=3, time_dim=4, hidden_dim=16, depth=1,
                                       loss_type=loss_type, num_bins=11, value_min=-3., value_max=0., sigma=.2)
    model.svf_config = SVFConfig(K=2, flow_steps=2)
    model.teacher_feature_passes, model.reference_microbatch_size = 2, 3
    return model.train(True)


class ModelIntegrationTests(unittest.TestCase):
    def test_forward_backward_only_updates_actor_head_and_soft_value(self):
        self._check_forward_backward("mse")

    def test_ce_forward_backward_only_updates_actor_head_and_soft_value(self):
        self._check_forward_backward("hl-gauss")

    def _check_forward_backward(self, loss_type):
        torch.manual_seed(3)
        model = tiny_joint(loss_type)
        inputs = {
            "observations": torch.randn(2, 2, 3), "attention_mask": torch.ones(2, 2),
            "state": torch.randn(2, 1, 3), "embodiment_id": torch.zeros(2, dtype=torch.long),
            "action": torch.randn(2, 2, 3),
            "action_mask": torch.tensor([1., 1., 0.]).expand(2, 2, -1),
        }
        original = {k: v.clone() for k, v in inputs.items()}
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        # CPU modules exercise real adapters/objective; only CUDA autocast is disabled.
        with patch("gr00t.model.svf.model.torch.autocast", side_effect=lambda **kwargs: nullcontext()):
            result = model(inputs)
            self.assertTrue(torch.isfinite(result["loss"]))
            result["loss"].backward()
        for group in (model.actor.backbone, model.reference_head, model.teacher):
            self.assertTrue(all(p.grad is None for p in group.parameters()))
        for group in (model.actor.action_head, model.soft_value):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in group.parameters()))
        optimizer.step()
        changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p.detach())]
        self.assertTrue(any(name.startswith("actor.action_head") for name in changed))
        self.assertTrue(any(name.startswith("soft_value") for name in changed))
        self.assertTrue(all(name.startswith(("actor.action_head", "soft_value")) for name in changed))
        for name in inputs:
            self.assertTrue(torch.equal(inputs[name], original[name]), name)

    def test_train_calls_keep_frozen_modules_in_eval(self):
        model = tiny_joint()
        model.eval()
        model.train()
        self.assertTrue(model.actor.action_head.training)
        self.assertTrue(model.soft_value.training)
        for group in (model.actor.backbone, model.reference_head, model.teacher):
            self.assertFalse(group.training)
            self.assertTrue(all(not p.requires_grad for p in group.parameters()))


if __name__ == "__main__":
    unittest.main()
