"""CPU checks for direct feature IQL without loading a BC checkpoint."""
import io
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.mark.parametrize('encoder,expected_dim', [('deas', 64), ('none', 2048)])
def test_encoder_modes_update_and_checkpoint(monkeypatch, encoder, expected_dim):
    from gr00t.model.iql.model import ChunkIQLCritic, GR00T_N1_5
    from gr00t.model.action_head import deas_critic

    def fake_actor(*args, **kwargs):
        return SimpleNamespace(backbone=nn.Linear(1, 1), action_dim=2,
            action_head=SimpleNamespace(vlln=nn.Linear(1, 1), vl_self_attention=nn.Linear(1, 1),
                config=SimpleNamespace(backbone_embedding_dim=2048, max_state_dim=3)))

    class SmallCategoryMLP(nn.Module):
        def __init__(self, categories, input_dim, hidden, output_dim):
            super().__init__()
            self.layer4 = nn.Linear(input_dim, output_dim)
        def forward(self, features, ids):
            return self.layer4(features)

    monkeypatch.setattr(GR00T_N1_5, 'from_pretrained', fake_actor)
    monkeypatch.setattr(deas_critic, 'CategorySpecificMLP', SmallCategoryMLP)
    model = ChunkIQLCritic('unused', hidden=8, depth=1, horizon=2, critic_encoder=encoder)
    features = torch.randn(3, 1, 2048) * 4
    ids = torch.zeros(3, dtype=torch.long)
    projected = model.project(features, ids)
    assert projected.shape == (3, 1, expected_dim)
    if encoder == 'none':
        torch.testing.assert_close(projected, features, rtol=0, atol=0)
        assert projected.abs().max() > 1
        assert not list(model.head.backbone_encoder.parameters())
    else:
        assert projected.abs().max() <= 1
    states, actions = torch.randn(3, 1, 3), torch.randn(3, 2, 2)
    q1, q2 = model.head.q(projected, states, actions)
    v = model.head.v(projected, states)
    assert q1.shape == q2.shape == v.shape == (3,)
    loss = (q1.square() + q2.square() + v.square()).mean()
    loss.backward()
    for network in (model.head.critic, model.head.value):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in network.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
    optimizer = torch.optim.Adam(model.head.parameters(), lr=3e-4)
    optimizer.step()
    model.head.update_target(.005)
    saved = io.BytesIO()
    torch.save(model.head.state_dict(), saved)
    saved.seek(0)
    restored = ChunkIQLCritic('unused', hidden=8, depth=1, horizon=2, critic_encoder=encoder)
    restored.head.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored.score_actions(restored.project(features, ids), states, actions)
        expected = model.score_actions(model.project(features, ids), states, actions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
