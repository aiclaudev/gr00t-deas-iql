"""Exercise real loading control flow with tiny CPU modules and mocked checkpoints."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("requested_horizon", [8, 16])
def test_actor_horizon_loading_preserves_weights_and_inference_setup(monkeypatch, requested_horizon):
    # Extract the real method to avoid importing the VLM stack or triggering downloads.
    source = Path(__file__).resolve().parents[1] / "gr00t/model/policy.py"
    tree = ast.parse(source.read_text())
    policy = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "Gr00tPolicy")
    method = next(node for node in policy.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_load_model")

    class TinyHead(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.tensor([-7.0, -3.0]))

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = TinyHead(SimpleNamespace(action_horizon=16))
            self.backbone = torch.nn.Linear(2, 2)
            self.config = SimpleNamespace(action_horizon=16, action_head_cfg={"action_horizon": 16})
            self.action_horizon = 16
            self.moves = []

        def to(self, *args, **kwargs):
            self.moves.append((self.action_head, dict(kwargs)))
            return super().to(*args, **kwargs)

    model = TinyModel()
    model.action_head.weight.data.copy_(torch.tensor([1.25, 0.5]))
    model.to(dtype=torch.bfloat16)
    model.moves.clear()
    original_head = model.action_head
    original_weights = original_head.weight.detach().clone()
    loads = []

    def load_checkpoint(path, **kwargs):
        loads.append((path, kwargs))
        return model

    fake_head_module = ModuleType("gr00t.model.action_head.flow_matching_action_head")
    fake_head_module.FlowmatchingActionHead = TinyHead
    monkeypatch.setitem(sys.modules, fake_head_module.__name__, fake_head_module)
    namespace = {"GR00T_N1_5": SimpleNamespace(from_pretrained=load_checkpoint),
                 "COMPUTE_DTYPE": torch.bfloat16}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    instance = SimpleNamespace(device="cpu", _modality_config={
        "action": SimpleNamespace(delta_indices=list(range(requested_horizon)))})
    namespace["_load_model"](instance, "/mock/actor")

    assert loads == [("/mock/actor", {"torch_dtype": torch.bfloat16})]
    assert instance.model is model
    assert (model.action_head is original_head) == (requested_horizon == 16)
    assert model.moves == [(model.action_head, {"device": "cpu", "dtype": torch.bfloat16})]
    assert not model.training and not model.action_head.training and not model.backbone.training
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert torch.equal(model.action_head.weight, original_weights)
    assert model.action_horizon == requested_horizon
    assert model.action_head.config.action_horizon == requested_horizon
    assert model.config.action_head_cfg["action_horizon"] == requested_horizon
