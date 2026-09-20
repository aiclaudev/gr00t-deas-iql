"""Best-of-N evaluation with each checkpoint's own features and normalization.

The actor generates N action chunks from one backbone pass. Candidates are
converted to robot actions, then normalized with the critic's training metadata.
The critic retains its own frozen visual feature processing and Q networks.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.model.gr00t_n1_deas_critic import GR00T_N1_5_DEAS_Critic
from gr00t.model.policy import COMPUTE_DTYPE, Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values


def repeat_batch(features, count):
    return BatchFeature({key: value.repeat((count,) + (1,) * (value.ndim - 1))
                         for key, value in features.items()})


def select_candidates(actions, scores, count, batch_size, temperature=0.0):
    scores = scores.reshape(count, batch_size)
    if not torch.isfinite(scores).all():
        raise ValueError('Critic returned nonfinite BoN scores')
    if temperature > 0:
        indices = torch.distributions.Categorical(logits=(scores / temperature).T).sample()
    else:
        indices = scores.argmax(dim=0)
    flat = (indices * batch_size + torch.arange(batch_size, device=indices.device)).cpu().numpy()
    return {key: np.asarray(value)[flat] for key, value in actions.items()}


class CheckpointDEASBoNPolicy:
    def __init__(self, actor_model_path, critic_model_path, embodiment_tag,
                 modality_config, modality_transform, denoising_steps=4,
                 num_samples=10, temperature=0.0, device='cuda:0'):
        if num_samples < 1 or temperature < 0:
            raise ValueError('Invalid BoN sample count or temperature')
        critic_transform = copy.deepcopy(modality_transform)
        self.actor = Gr00tPolicy(actor_model_path, embodiment_tag, modality_config,
                                 modality_transform, denoising_steps, device)
        self.critic = GR00T_N1_5_DEAS_Critic.from_pretrained(
            str(critic_model_path), torch_dtype=COMPUTE_DTYPE,
            tune_visual=False, tune_llm=False, tune_critic=False, tune_value=False)
        # Unmarked checkpoints were trained with the value-loss mutation that
        # made online Q consume two feature-transform passes. Fixed checkpoints
        # explicitly record one pass; never silently change a saved Q's inputs.
        self.critic_feature_passes = self.critic.config.critic_cfg.get('online_q_feature_passes', 2)
        if type(self.critic_feature_passes) is not int or self.critic_feature_passes not in (1, 2):
            raise ValueError('online_q_feature_passes must be 1 or 2')
        self.critic.to(device=device, dtype=COMPUTE_DTYPE).eval()
        self.critic.requires_grad_(False)
        self.actor.model.requires_grad_(False)
        self.model = self.critic  # evaluator checks critic_action_horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.device = torch.device(device)
        self.critic_transform = critic_transform
        metadata = json.loads((Path(critic_model_path) / 'experiment_cfg/metadata.json').read_text())
        self.critic_transform.set_metadata(DatasetMetadata.model_validate(metadata[self.actor.embodiment_tag.value]))
        self.critic_transform.eval()
        self.critic_action_horizon = self.critic.critic_head.critic_action_horizon
        if self.critic_action_horizon > self.actor.model.action_horizon:
            raise ValueError('Critic action horizon exceeds actor horizon')
        self.last_scores = None
        self.last_candidates = None
        print(f'Checkpoint DEAS BoN: N={num_samples}, temperature={temperature}; '
              'separate weights and normalization; '
              f'saved Q uses {self.critic_feature_passes} feature-transform pass(es)', flush=True)

    def get_modality_config(self):
        return self.actor.get_modality_config()

    def _normalize_candidates_for_critic(self, actions):
        data = {key: np.array(value, copy=True) for key, value in actions.items()}
        for transform in self.critic_transform.transforms:
            if isinstance(transform, (StateActionToTensor, StateActionTransform, ConcatTransform)):
                data = transform(data)
        action = torch.as_tensor(data['action'])
        width = self.critic.critic_head.config.action_dim
        if action.shape[-1] > width:
            raise ValueError('Robot action dimension exceeds critic input')
        # Training pads unused action dimensions with zeros, never diffusion noise.
        return F.pad(action, (0, width - action.shape[-1])).to(self.device, dtype=COMPUTE_DTYPE)

    @torch.inference_mode()
    def get_action(self, observations):
        is_batch = self.actor._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)
        observations = {key: np.asarray(value) for key, value in observations.items()}
        actor_input = self.actor.apply_transforms(dict(observations))
        critic_input = self.critic_transform(dict(observations))
        with torch.autocast(device_type=self.device.type, dtype=COMPUTE_DTYPE):
            backbone_input, action_input = self.actor.model.prepare_input(actor_input)
            backbone_output = self.actor.model.backbone(backbone_input)
            batch_size = action_input['state'].shape[0]
            predicted = self.actor.model.action_head.get_action(
                repeat_batch(backbone_output, self.num_samples),
                repeat_batch(action_input, self.num_samples))['action_pred']
            # Keep physical actions to return exactly the candidate that was scored.
            actions = self.actor._get_unnormalized_action(predicted.float())
            normalized_actions = self._normalize_candidates_for_critic(actions)
            critic_backbone_input, critic_action_input = self.critic.prepare_input(critic_input)
            critic_features = self.critic.backbone(critic_backbone_input)
            head = self.critic.critic_head
            for _ in range(self.critic_feature_passes):
                critic_features = head.process_backbone_output(critic_features)
            critic_features = critic_features['backbone_features']
            embedding = torch.tanh(head.backbone_encoder(
                critic_features.mean(dim=1, keepdim=True), critic_action_input['embodiment_id']))
            state = critic_action_input['state']
            q1_logits, q2_logits = head.critic(
                embedding.repeat(self.num_samples, 1, 1),
                state.repeat(self.num_samples, 1, 1),
                normalized_actions[:, :self.critic_action_horizon])
            q1 = head.hlg.transform_from_probs(torch.softmax(q1_logits.float(), dim=-1))
            q2 = head.hlg.transform_from_probs(torch.softmax(q2_logits.float(), dim=-1))
            scores = torch.minimum(q1, q2)
        self.last_candidates = actions
        self.last_scores = scores.reshape(self.num_samples, batch_size).cpu()
        selected = select_candidates(actions, scores, self.num_samples, batch_size, self.temperature)
        if not all(np.isfinite(value).all() for value in selected.values()):
            raise ValueError('Actor returned nonfinite actions')
        return selected if is_batch else squeeze_dict_values(selected)
