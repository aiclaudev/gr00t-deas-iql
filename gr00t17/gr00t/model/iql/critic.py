"""Frozen N1.7 actor features and the DEAS-shaped scalar IQL critic.

Inputs must come from the actor checkpoint's N1.7 processor/collator. Never feed
N1.5-normalized tensors into this adapter. The actor's native padded dimensions
are retained; only the first `horizon` actions are evaluated.
"""
from pathlib import Path
import json
import torch
from torch import nn
from .core import ScalarIQL, expectile_loss, masked_mean, td_target


class CategoryProjection(nn.Module):
    """Same four category-specific layers as the existing DEAS critic."""
    def __init__(self, categories, input_dim, hidden=1024, output_dim=64):
        super().__init__()
        from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear
        self.layer1 = CategorySpecificLinear(categories, input_dim, hidden)
        self.layer2 = CategorySpecificLinear(categories, hidden, hidden)
        self.layer3 = CategorySpecificLinear(categories, hidden, hidden)
        self.layer4 = CategorySpecificLinear(categories, hidden, output_dim)

    def forward(self, x, ids):
        for layer in (self.layer1, self.layer2, self.layer3):
            x = torch.nn.functional.silu(layer(x, ids))
        return self.layer4(x, ids)


class N17IQLCritic(nn.Module):
    def __init__(self, actor, *, actor_checkpoint=None, horizon=16):
        super().__init__()
        cfg = actor.config
        if not 1 <= horizon <= cfg.action_horizon:
            raise ValueError('Critic horizon must fit the actor horizon')
        self.spec = dict(actor_checkpoint=str(Path(actor_checkpoint).resolve()) if actor_checkpoint else None,
                         horizon=horizon, feature_dim=cfg.backbone_embedding_dim,
                         state_dim=cfg.max_state_dim * cfg.state_history_length,
                         action_dim=cfg.max_action_dim, categories=cfg.max_num_embodiments)
        self.backbone = actor.backbone.requires_grad_(False).eval()
        self.vlln = actor.action_head.vlln.requires_grad_(False).eval()
        self.vl_self_attention = actor.action_head.vl_self_attention.requires_grad_(False).eval()
        self.head = ScalarIQL(feature_dim=64, state_dim=self.spec['state_dim'],
                              action_dim=self.spec['action_dim'], horizon=horizon)
        self.head.backbone_encoder = CategoryProjection(cfg.max_num_embodiments,
                                                         cfg.backbone_embedding_dim)

    @classmethod
    def from_actor_checkpoint(cls, checkpoint, horizon=16):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
        actor, info = Gr00tN1d7.from_pretrained(
            str(checkpoint), local_files_only=True, output_loading_info=True,
            transformers_loading_kwargs={'local_files_only': True}, load_bf16=True,
            tune_llm=False, tune_visual=False, tune_projector=False,
            tune_diffusion_model=False, tune_vlln=False)
        if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys']):
            raise ValueError(f'Actor checkpoint mismatch: {info}')
        actor.backbone.to(torch.bfloat16)
        return cls(actor, actor_checkpoint=checkpoint, horizon=horizon)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.backbone, self.vlln, self.vl_self_attention):
            module.eval()
        self.head.target_critic.eval()
        return self

    @torch.no_grad()
    def encode(self, inputs):
        device = next(self.head.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            raw = self.backbone(self.backbone.prepare_input(batch))
            features = self.vl_self_attention(self.vlln(raw['backbone_features']))
        # Same mean pooling as the existing 1.5 IQL critic.
        return features.mean(dim=1, keepdim=True).float()

    def project(self, features, ids):
        with torch.autocast(features.device.type, dtype=torch.bfloat16, enabled=features.is_cuda):
            projected = self.head.backbone_encoder(features, ids)
        return projected.float().tanh()

    def lowdim(self, inputs):
        device = next(self.head.parameters()).device
        state = inputs['state'].to(device).float()
        action = inputs['action'].to(device).float()
        if state.flatten(1).shape[-1] != self.spec['state_dim']:
            raise ValueError('Use the N1.7 checkpoint processor state layout')
        if action.shape[-1] != self.spec['action_dim'] or action.shape[1] < self.spec['horizon']:
            raise ValueError('Invalid N1.7 action dimensions/horizon')
        if 'state_mask' in inputs: state = state * inputs['state_mask'].to(device)
        if 'action_mask' in inputs: action = action * inputs['action_mask'].to(device)
        return state, action[:, :self.spec['horizon']], inputs['embodiment_id'].to(device).long()

    def forward(self, current, next_inputs, chunk_return, bootstrap_mask, chunk_valid,
                *, discount=.99, expectile=.7):
        """One IQL loss; QC masks/returns are supplied by the episode sampler.

        chunk_valid excludes incomplete chunks. bootstrap_mask is separate:
        successful terminal chunks are valid but do not bootstrap. Unknown final
        boundaries without a real next observation must have chunk_valid=0.
        """
        states, actions, ids = self.lowdim(current)
        next_states = next_inputs['state'].to(states.device).float()
        if 'state_mask' in next_inputs:
            next_states = next_states * next_inputs['state_mask'].to(states.device)
        features = self.project(self.encode(current), ids)
        valid = torch.as_tensor(chunk_valid, device=states.device, dtype=torch.float32).reshape(-1)
        returns = torch.as_tensor(chunk_return, device=states.device, dtype=torch.float32).reshape(-1)
        bootstrap = torch.as_tensor(bootstrap_mask, device=states.device).reshape(-1)
        if valid.shape != (states.shape[0],) or returns.shape != valid.shape or bootstrap.shape != valid.shape:
            raise ValueError('QC fields must contain one scalar per sample')
        with torch.no_grad():
            next_features = self.project(self.encode(next_inputs), ids)
            tq1, tq2 = self.head.q(features, states, actions, target=True)
            nextv = self.head.v(next_features, next_states)
            target = td_target(returns, bootstrap, nextv, discount, self.spec['horizon'])
        value = self.head.v(features, states)
        vloss = expectile_loss(torch.minimum(tq1,tq2)-value, expectile, valid)
        q1, q2 = self.head.q(features, states, actions)
        qloss = masked_mean(((q1-target).square()+(q2-target).square())/2, valid)
        return dict(loss=qloss+vloss, q_loss=qloss, v_loss=vloss,
                    q=torch.minimum(q1,q2), value=value, target=target)

    @torch.no_grad()
    def score_actions(self, inputs, actions=None, action_mask=None):
        """Score one batch of normalized candidate chunks using min(Q1,Q2)."""
        batch = dict(inputs)
        if actions is not None:
            batch['action'] = actions
            if action_mask is not None: batch['action_mask'] = action_mask
            elif 'action_mask' in batch:
                batch['action_mask'] = batch['action_mask'][:, :actions.shape[1]]
            else:
                raise ValueError('Candidate scoring requires the processor action mask')
        states, actions, ids = self.lowdim(batch)
        features = self.project(self.encode(batch), ids)
        q1,q2 = self.head.q(features, states, actions)
        return torch.minimum(q1,q2)

    def enable_soft_value(self):
        """Attach the existing SVF double scalar head without changing IQL Q/V."""
        if not hasattr(self.head, 'soft_value'):
            from .soft_value import DoubleSoftValue
            device = next(self.head.parameters()).device
            self.head.soft_value = DoubleSoftValue(
                feature_dim=64, state_dim=self.spec['state_dim'],
                action_horizon=self.spec['horizon'], action_dim=self.spec['action_dim'],
                time_dim=64, hidden_dim=512, depth=4, loss_type='mse').to(device)
            self.head.soft_value.train(self.training)
            self.spec['soft_value'] = dict(loss_type='mse', time_dim=64, hidden_dim=512, depth=4)
        return self.head.soft_value

    def soft_values(self, inputs, x_t, t, action_mask=None):
        """Return [B,2] SVF values; preserve gradients to x_t and soft head only.

        x_t is in the actor processor's normalized action coordinates. Values
        and input gradients use FP32, matching the existing SVF implementation.
        """
        if not hasattr(self.head, 'soft_value'):
            raise RuntimeError('Call enable_soft_value before creating the optimizer')
        states, _, ids = self.lowdim(inputs)
        device = states.device
        x = x_t.to(device).float()
        h = self.spec['horizon']
        if x.ndim != 3 or x.shape[0] != states.shape[0] or x.shape[1] < h or x.shape[2] != self.spec['action_dim']:
            raise ValueError('x_t must use the N1.7 [B,H>=16,132] action layout')
        mask = action_mask if action_mask is not None else inputs.get('action_mask')
        if mask is None:
            raise ValueError('SVF requires an action mask for padded coordinates')
        mask = torch.as_tensor(mask, device=device).detach()
        if mask.ndim != 3 or mask.shape[1] < h:
            raise ValueError('action_mask must have shape [B,H,D]')
        mask = mask[:, :h].to(x)
        if mask.shape != x[:, :h].shape:
            raise ValueError('action_mask must match x_t batch and action dimensions')
        time = torch.as_tensor(t, device=device).float().reshape(-1)
        if time.shape != (x.shape[0],) or not torch.isfinite(time).all() or ((time<0)|(time>1)).any():
            raise ValueError('t must contain one finite value in [0,1] per sample')
        with torch.no_grad():
            features = self.project(self.encode(inputs), ids)
        with torch.autocast(device.type, enabled=False):
            return self.head.soft_value(features.detach().float(), states.detach().float(),
                                        x[:, :h] * mask, time.detach())

    def soft_value_loss(self, inputs, x_t, t, teacher_qs, temperature,
                        *, action_mask=None, chunk_valid=None):
        """SVF regression to detached lambda*logmeanexp(Q/lambda).

        teacher_qs are [K,B] scores for frozen-BC continuations from this x_t,t,
        not unrelated action samples. This does not generate continuations or
        update the actor. Use the SVF temperature, not IQL's expectile kappa.
        """
        import math
        values = self.soft_values(inputs, x_t, t, action_mask)
        qs = torch.as_tensor(teacher_qs, device=values.device).detach().float()
        lam = torch.as_tensor(temperature, device=values.device).detach().float()
        if qs.ndim != 2 or qs.shape[0] < 1 or qs.shape[1] != values.shape[0] or not torch.isfinite(qs).all():
            raise ValueError('teacher_qs must be finite [K,B]')
        if lam.numel()!=1 or not torch.isfinite(lam).all() or not (lam>0).all():
            raise ValueError('SVF temperature must be a finite positive scalar')
        target = (lam.reshape(()) * (torch.logsumexp(qs / lam.reshape(()), dim=0)-math.log(qs.shape[0]))).detach()
        mask = action_mask if action_mask is not None else inputs['action_mask']
        valid = torch.as_tensor(mask, device=values.device)[:, :self.spec['horizon']].flatten(1).bool().any(-1)
        if chunk_valid is not None:
            chunk_valid = torch.as_tensor(chunk_valid, device=values.device).reshape(-1)
            if chunk_valid.shape != valid.shape: raise ValueError('chunk_valid must be [B]')
            valid = valid & chunk_valid.bool()
        loss = masked_mean((values.float()-target[:,None]).square().mean(-1), valid)
        return dict(loss=loss, soft_value_loss=loss, soft_values=values, soft_value_target=target)

    def save_head(self, directory):
        """Save trainable/target critic weights; frozen features reference actor checkpoint."""
        if self.spec['actor_checkpoint'] is None:
            raise ValueError('An actor checkpoint reference is required for a reloadable critic')
        path = Path(directory);path.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path/'critic_head.pt')
        (path/'critic_config.json').write_text(json.dumps(self.spec, indent=2)+'\n')

    @classmethod
    def load_head(cls, directory):
        path = Path(directory);spec=json.loads((path/'critic_config.json').read_text())
        model=cls.from_actor_checkpoint(spec['actor_checkpoint'], horizon=spec['horizon'])
        if 'soft_value' in spec:
            model.enable_soft_value()
        if model.spec != spec: raise ValueError('Actor configuration changed since critic save')
        model.head.load_state_dict(torch.load(path/'critic_head.pt', map_location='cpu', weights_only=True))
        return model
