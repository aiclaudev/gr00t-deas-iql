"""Frozen BC2 representation plus freshly initialized scalar IQL Q/V."""
import torch
from torch import nn
from gr00t.model.gr00t_n1 import GR00T_N1_5
from .core import ScalarIQL


class ChunkIQLCritic(nn.Module):
    def __init__(self, actor_path, hidden=512, depth=4, horizon=16):
        super().__init__()
        actor = GR00T_N1_5.from_pretrained(str(actor_path), torch_dtype=torch.bfloat16,
                                         tune_visual=False, tune_llm=False)
        self.backbone = actor.backbone.requires_grad_(False).eval()
        self.vlln = actor.action_head.vlln.requires_grad_(False).eval()
        self.vl_self_attention = actor.action_head.vl_self_attention.requires_grad_(False).eval()
        self.feature_dim = actor.action_head.config.backbone_embedding_dim
        from gr00t.model.action_head.deas_critic import CategorySpecificMLP
        self.head = ScalarIQL(feature_dim=64,
                             state_dim=actor.action_head.config.max_state_dim,
                             action_dim=actor.action_dim, horizon=horizon, hidden=hidden, depth=depth)
        self.head.backbone_encoder = CategorySpecificMLP(32, self.feature_dim, 1024, 64)
        del actor

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval(); self.vlln.eval(); self.vl_self_attention.eval()
        return self

    @torch.no_grad()
    def encode(self, batch, prefix='eagle_'):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            raw = self.backbone(batch, eagle_prefix=prefix)
            features = self.vl_self_attention(self.vlln(raw['backbone_features']))
            return features.mean(dim=1, keepdim=True).float()

    def project(self, features, embodiment_id):
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16, enabled=features.is_cuda):
            projected = self.head.backbone_encoder(features, embodiment_id)
        return torch.tanh(projected.float())

    def score_actions(self, features, states, actions):
        q1,q2 = self.head.q(features,states,actions)
        return torch.minimum(q1,q2)
