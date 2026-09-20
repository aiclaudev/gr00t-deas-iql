"""CPU LoRA invariants: BC initialization, trainable scope, resume and export."""
import copy
import unittest

import torch
from torch import nn

from gr00t.model.svf.lora import (
    ActorTuningConfig, LoRALinear, apply_dit_lora, is_lora_parameter,
    merged_action_head_state_dict, merged_actor_state_dict,
)


class MockAttention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.ModuleList([nn.Linear(width, width), nn.Dropout(0)])

    def forward(self, x):
        return self.to_out[1](self.to_out[0](
            torch.tanh(self.to_q(x) + self.to_k(x)) * self.to_v(x)))


class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = MockAttention()
        self.ff = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        return self.ff(self.norm(x + self.attn1(x))) + x


class MockHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(8, 8)
        self.vl_self_attention = MockAttention()
        self.model = nn.Module()
        self.model.transformer_blocks = nn.ModuleList([MockBlock(), MockBlock()])
        self.model.proj_out = nn.Linear(8, 8)

    def forward(self, x):
        x = self.projector(x) + self.vl_self_attention(x)
        for block in self.model.transformer_blocks:
            x = block(x)
        return self.model.proj_out(x)


class MockActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.action_head = MockHead()

    def forward(self, x):
        return self.action_head(self.backbone(x))


class SVFLoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)
        torch.set_num_threads(1)

    def test_config_rejects_invalid_values(self):
        for kwargs in ({"mode": "lora"}, {"rank": 0}, {"rank": True}, {"rank": 2.5},
                       {"alpha": 0}, {"alpha": float("nan")}, {"alpha": float("inf")},
                       {"dropout": -0.1}, {"dropout": 1.0}, {"dropout": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActorTuningConfig(**kwargs)

    def test_zero_initialization_preserves_bc_and_strict_target_scope(self):
        head = MockHead().eval()
        original = copy.deepcopy(head)
        inputs = torch.randn(3, 4, 8)
        targets = apply_dit_lora(head, ActorTuningConfig("dit-lora", 3, 6))
        self.assertEqual(len(targets), 8)
        self.assertTrue(all(path.startswith("model.transformer_blocks.") and ".attn1." in path
                            for path in targets))
        self.assertIsInstance(head.vl_self_attention.to_q, nn.Linear)
        self.assertIsInstance(head.projector, nn.Linear)
        torch.testing.assert_close(head(inputs), original(inputs), rtol=0, atol=0)
        for name, parameter in head.named_parameters():
            self.assertEqual(parameter.requires_grad, is_lora_parameter(name), name)
            if is_lora_parameter(name):
                self.assertEqual(parameter.dtype, torch.float32)
        self.assertFalse(head.model.transformer_blocks[0].attn1.to_q.training)
        with self.assertRaisesRegex(ValueError, "already installed"):
            apply_dit_lora(head, ActorTuningConfig("dit-lora"))

    def test_only_adapter_weights_change_and_input_gradient_survives(self):
        head = MockHead()
        original = {key: tensor.clone() for key, tensor in head.state_dict().items()}
        apply_dit_lora(head, ActorTuningConfig("dit-lora", 3, 3))
        optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=.02)
        inputs = torch.randn(2, 4, 8, requires_grad=True)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (head(inputs) - .5).square().mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(inputs.grad.abs().sum().item(), 0)
        for name, parameter in head.named_parameters():
            if is_lora_parameter(name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            else:
                self.assertIsNone(parameter.grad)
                vanilla_name = name.replace(".base.", ".")
                torch.testing.assert_close(parameter, original[vanilla_name], rtol=0, atol=0)

    def test_dropout_is_only_enabled_in_train(self):
        layer = LoRALinear(nn.Linear(8, 8, bias=False), 4, 4, .5)
        with torch.no_grad():
            layer.lora_A.fill_(.1)
            layer.lora_B.fill_(.2)
        inputs = torch.ones(16, 8)
        layer.train()
        first, second = layer(inputs), layer(inputs)
        self.assertFalse(torch.equal(first, second))
        layer.eval()
        torch.testing.assert_close(layer(inputs), layer(inputs), rtol=0, atol=0)
        expected = layer.base(inputs) + inputs @ layer.lora_A.T @ layer.lora_B.T
        torch.testing.assert_close(layer(inputs), expected)

    def test_fp32_adapters_allow_bfloat16_base_and_autocast_gradients(self):
        layer = LoRALinear(nn.Linear(8, 8).to(torch.bfloat16), 2, 2)
        inputs = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(inputs)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(layer.lora_A.dtype, torch.float32)
        output.float().square().mean().backward()
        self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0)
        self.assertGreater(inputs.grad.abs().sum().item(), 0)

    def test_merged_actor_loads_in_vanilla_model_without_live_mutation(self):
        actor = MockActor().eval()
        vanilla = copy.deepcopy(actor)
        before_keys = set(actor.state_dict())
        apply_dit_lora(actor.action_head, ActorTuningConfig("dit-lora", 3, 7))
        for module in actor.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_B, std=.2)
        live_state = {key: tensor.clone() for key, tensor in actor.state_dict().items()}
        merged = merged_actor_state_dict(actor)
        self.assertEqual(set(merged), before_keys)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in merged.values()))
        self.assertTrue(all(".base" not in key and "lora_" not in key for key in merged._metadata))
        vanilla.load_state_dict(merged, strict=True)
        inputs = torch.randn(3, 4, 8)
        torch.testing.assert_close(vanilla(inputs), actor(inputs), rtol=2e-5, atol=1e-6)
        for key, tensor in actor.state_dict().items():
            torch.testing.assert_close(tensor, live_state[key], rtol=0, atol=0)
        head_only = merged_action_head_state_dict(actor.action_head)
        self.assertEqual(set(head_only), set(vanilla.action_head.state_dict()))
        # Export tensors must not alias live CPU model storage either.
        with torch.no_grad():
            merged["backbone.weight"].zero_()
        torch.testing.assert_close(actor.backbone.weight, live_state["backbone.weight"], rtol=0, atol=0)

    def test_unmerged_checkpoint_reload_is_exact(self):
        original = MockHead().eval()
        restored = copy.deepcopy(original)
        config = ActorTuningConfig("dit-lora", 3, 7, .2)
        apply_dit_lora(original, config)
        apply_dit_lora(restored, config)
        for name, parameter in original.named_parameters():
            if is_lora_parameter(name):
                with torch.no_grad():
                    parameter.normal_()
        restored.load_state_dict(original.state_dict(), strict=True)
        inputs = torch.randn(2, 3, 8)
        torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)

    def test_full_mode_is_noop_and_invalid_target_is_atomic(self):
        head = MockHead()
        head.projector.requires_grad_(False)
        trainable = {name: param.requires_grad for name, param in head.named_parameters()}
        self.assertEqual(apply_dit_lora(head, ActorTuningConfig()), [])
        self.assertEqual(trainable, {name: p.requires_grad for name, p in head.named_parameters()})
        plain = merged_action_head_state_dict(head)
        for key, tensor in head.state_dict().items():
            torch.testing.assert_close(plain[key], tensor, rtol=0, atol=0)
        head.model.transformer_blocks[-1].attn1.to_v = nn.Identity()
        trainable = {name: p.requires_grad for name, p in head.named_parameters()}
        with self.assertRaisesRegex(TypeError, "not Linear"):
            apply_dit_lora(head, ActorTuningConfig("dit-lora"))
        self.assertEqual(trainable, {name: p.requires_grad for name, p in head.named_parameters()})
        self.assertFalse(any(isinstance(module, LoRALinear) for module in head.modules()))


if __name__ == "__main__":
    unittest.main()
