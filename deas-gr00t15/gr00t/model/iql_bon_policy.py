"""BoN actor evaluation against the independently trained scalar IQL critic."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.model.policy import Gr00tPolicy, COMPUTE_DTYPE, squeeze_dict_values, unsqueeze_dict_values
from gr00t.model.checkpoint_bon_policy import repeat_batch, select_candidates
from gr00t.model.iql.model import ChunkIQLCritic

class CheckpointIQLBoNPolicy:
    def __init__(self,actor_model_path,critic_model_path,embodiment_tag,
                 modality_config,modality_transform,denoising_steps=4,
                 num_samples=50,temperature=0.,device='cuda:0'):
        if num_samples<1 or temperature<0:raise ValueError('Invalid BoN settings')
        path=Path(critic_model_path);cfg=json.loads((path/'config.json').read_text())
        if cfg.get('algorithm')!='scalar IQL + QC complete chunks':raise ValueError('Not a scalar IQL checkpoint')
        root=path.parent
        payload=torch.load(path/'training.pt',map_location='cpu',weights_only=False)
        if payload['step']!=10000:raise ValueError('Expected requested IQL step10000')
        self.critic=ChunkIQLCritic(cfg['actor'],horizon=cfg['horizon'])
        self.critic.head.load_state_dict(payload['head'],strict=True)
        del payload
        frozen=load_file(str(root/'frozen_features.safetensors'))
        missing,unexpected=self.critic.load_state_dict(frozen,strict=False)
        if unexpected or any(not key.startswith('head.') for key in missing):raise ValueError('Frozen critic features mismatch')
        del frozen
        self.critic.to(device).eval().requires_grad_(False)
        self.actor=Gr00tPolicy(actor_model_path,embodiment_tag,modality_config,
                              modality_transform,denoising_steps,device)
        self.actor.model.requires_grad_(False)
        self.critic_transform=copy.deepcopy(modality_transform)
        metadata=json.loads((root/'normalization_metadata.json').read_text())
        self.critic_transform.set_metadata(DatasetMetadata.model_validate(metadata[self.actor.embodiment_tag.value]))
        self.critic_transform.eval()
        self.critic_action_horizon=cfg['horizon']
        self.action_dim=self.actor.model.action_dim
        if self.critic_action_horizon>self.actor.model.action_horizon:raise ValueError('Actor horizon too short')
        self.model=SimpleNamespace(critic_action_horizon=self.critic_action_horizon)
        self.device=torch.device(device);self.num_samples=num_samples;self.temperature=temperature
        self.last_scores=None;self.last_candidates=None
        print(f'Scalar IQL BoN N={num_samples}; independent actor/critic feature passes; min(Q1,Q2)',flush=True)

    def get_modality_config(self):return self.actor.get_modality_config()

    def _normalize_candidates_for_critic(self,actions):
        data={k:np.array(v,copy=True) for k,v in actions.items()}
        for transform in self.critic_transform.transforms:
            if isinstance(transform,(StateActionToTensor,StateActionTransform,ConcatTransform)):
                data=transform(data)
        action=torch.as_tensor(data['action'])
        if action.shape[-1]>self.action_dim:raise ValueError('Action width mismatch')
        return F.pad(action,(0,self.action_dim-action.shape[-1])).to(self.device,dtype=torch.float32)

    @torch.inference_mode()
    def get_action(self,observations):
        is_batch=self.actor._check_state_is_batched(observations)
        if not is_batch:observations=unsqueeze_dict_values(observations)
        observations={k:np.asarray(v) for k,v in observations.items()}
        actor_input=self.actor.apply_transforms(dict(observations))
        critic_input=self.critic_transform(dict(observations))
        with torch.autocast(self.device.type,dtype=COMPUTE_DTYPE):
            backbone_input,action_input=self.actor.model.prepare_input(actor_input)
            raw=self.actor.model.backbone(backbone_input)
            batch=action_input['state'].shape[0]
            predicted=self.actor.model.action_head.get_action(
                repeat_batch(raw,self.num_samples),repeat_batch(action_input,self.num_samples))['action_pred']
        actions=self.actor._get_unnormalized_action(predicted.float())
        normalized=self._normalize_candidates_for_critic(actions)
        critic_backbone,critic_action=self.actor.model.prepare_input(critic_input)
        features=self.critic.encode(critic_backbone)
        embedding=self.critic.project(features,critic_action['embodiment_id'])
        state=critic_action['state'].float()
        if 'state_mask' in critic_action:state=state*critic_action['state_mask']
        q1,q2=self.critic.head.q(embedding.repeat(self.num_samples,1,1),
                               state.repeat(self.num_samples,1,1),normalized[:,:self.critic_action_horizon])
        scores=torch.minimum(q1,q2)
        self.last_candidates=actions;self.last_scores=scores.reshape(self.num_samples,batch).cpu()
        selected=select_candidates(actions,scores,self.num_samples,batch,self.temperature)
        if not all(np.isfinite(v).all() for v in selected.values()):raise ValueError('Nonfinite actions')
        return selected if is_batch else squeeze_dict_values(selected)
