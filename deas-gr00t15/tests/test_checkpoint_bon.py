"""CPU checks for BoN candidate ordering and checkpoint-specific normalization."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[1] / 'gr00t/model/checkpoint_bon_policy.py'


def extracted(name, namespace):
    tree=ast.parse(SOURCE.read_text())
    nodes=tree.body
    if name.startswith('_'):
        nodes=next(n for n in tree.body if isinstance(n,ast.ClassDef)).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace[name]


def test_argmax_uses_correct_candidate_for_each_environment():
    select=extracted('select_candidates',dict(np=np,torch=torch))
    # Candidate-major order: (candidate0-env0, candidate0-env1, candidate1-env0, ...).
    actions={'action.a':np.arange(6).reshape(6,1,1)}
    scores=torch.tensor([0.,9.,8.,1.,2.,3.])
    result=select(actions,scores,3,2)
    assert result['action.a'].reshape(-1).tolist()==[2,1]
    with pytest.raises(ValueError,match='nonfinite'):
        select(actions,torch.tensor([float('nan')]*6),3,2)


def test_critic_actions_are_renormalized_and_unused_dimensions_zeroed():
    class ToTensor:
        def __call__(self,data):
            return {k:torch.as_tensor(v) for k,v in data.items()}
    class CriticNormalization:
        def __call__(self,data):
            # Physical range [10,30] -> critic [-1,1], differs from actor range.
            return {k:(v-10)/10-1 for k,v in data.items()}
    class Concat:
        def __call__(self,data):
            return {'action':torch.cat(list(data.values()),dim=-1)}
    method=extracted('_normalize_candidates_for_critic',dict(
        np=np,torch=torch,F=F,COMPUTE_DTYPE=torch.float32,
        StateActionToTensor=ToTensor,StateActionTransform=CriticNormalization,ConcatTransform=Concat))
    instance=SimpleNamespace(critic_transform=SimpleNamespace(transforms=[ToTensor(),CriticNormalization(),Concat()]),
                             critic=SimpleNamespace(critic_head=SimpleNamespace(config=SimpleNamespace(action_dim=4))),device='cpu')
    physical={'action.a':np.array([[[10.]],[[30.]]],dtype=np.float32)}
    actual=method(instance,physical)
    assert actual[:,:,0].flatten().tolist()==[-1.,1.]
    assert torch.count_nonzero(actual[:,:,1:])==0
    assert physical['action.a'].flatten().tolist()==[10.,30.]


@pytest.mark.parametrize('passes', [1, 2])
def test_training_feature_path_matches_checkpoint_contract(passes):
    # Execute the actual training methods with small fake networks to identify
    # which feature representation reaches the online Q loss.
    from transformers import BatchFeature
    source = SOURCE.parent / 'action_head/deas_critic.py'
    tree = ast.parse(source.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DEASCritic')
    names = {'forward', 'process_backbone_output', 'compute_value_loss', 'compute_critic_loss'}
    methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(torch=torch, F=F, BatchFeature=BatchFeature)
    module = ast.parse('from __future__ import annotations')
    module.body += methods
    exec(compile(module, str(source), 'exec'), namespace)
    class Harness:
        pass
    for name in names:
        setattr(Harness, name, namespace[name])
    head = Harness()
    head.set_frozen_modules_to_eval_mode = lambda: None
    head.vlln = lambda x: x + 1
    head.vl_self_attention = lambda x: x
    head.backbone_encoder = lambda x, tag: x
    head.config = SimpleNamespace(expand_batch=None, online_q_feature_passes=passes)
    head.rl_config = SimpleNamespace(q_agg='min', expectile=0.7, negative_reward=False,
                                    discount1=0.99, discount2=0.99, nstep=1)
    head.critic_action_horizon = 16
    class HLG:
        def transform_from_probs(self, p): return p.sum(dim=-1)
        def __call__(self, logits, target): return logits.sum() * 0
    head.hlg = HLG()
    seen = {'value': [], 'target_critic': [], 'critic': []}
    def recorder(name, double=False):
        def network(features, *args):
            seen[name].append(features.clone())
            result = torch.zeros(2, 3)
            return (result, result) if double else result
        return network
    head.value = recorder('value')
    head.target_critic = recorder('target_critic', double=True)
    head.critic = recorder('critic', double=True)
    mask = torch.ones(2, 2, dtype=torch.long)
    current = BatchFeature(data={'backbone_features': torch.ones(2, 2, 3),
                                 'backbone_attention_mask': mask})
    future = BatchFeature(data={'backbone_features': torch.full((2, 2, 3), 4.0)})
    inputs = BatchFeature(data=dict(embodiment_id=torch.zeros(2, dtype=torch.long),
        state=torch.zeros(2, 1, 3), next_state=torch.zeros(2, 1, 3),
        action=torch.zeros(2, 16, 3), reward=torch.zeros(2, 16), done=torch.zeros(2, 16)))
    head.forward(current, future, inputs)
    expected_online_q = torch.full((2, 1, 3), 1.0 + passes).tanh()
    assert torch.allclose(seen['critic'][0], expected_online_q)
    assert torch.allclose(seen['target_critic'][0], torch.full((2, 1, 3), 2.0).tanh())
    assert torch.allclose(seen['value'][0], torch.full((2, 1, 3), 2.0).tanh())
    assert torch.allclose(seen['value'][1], torch.full((2, 1, 3), 5.0).tanh())
    # One-pass checkpoints keep shared VLM output intact. Missing/legacy markers
    # retain the old mutation so resuming them does not change their semantics.
    assert torch.equal(current.backbone_features, torch.full((2, 2, 3), 1.0 if passes == 1 else 3.0))
    assert torch.equal(future.backbone_features, torch.full((2, 2, 3), 4.0 if passes == 1 else 5.0))
    assert current.backbone_attention_mask is mask


@pytest.mark.parametrize('marker,expected', [(None, 2), (1, 1), (2, 2)])
def test_bon_reads_checkpoint_feature_contract(tmp_path, marker, expected):
    instance = initialize_mock_bon(tmp_path, {} if marker is None else {'online_q_feature_passes': marker})
    assert instance.critic_feature_passes == expected


@pytest.mark.parametrize('marker', [0, 3, True, '1', None])
def test_bon_rejects_invalid_checkpoint_feature_contract(tmp_path, marker):
    with pytest.raises(ValueError, match='online_q_feature_passes'):
        initialize_mock_bon(tmp_path, {'online_q_feature_passes': marker})


def initialize_mock_bon(tmp_path, critic_cfg):
    """Run the real constructor without loading GR00T or allocating a GPU."""
    import copy
    import json
    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(critic_cfg=critic_cfg)
            self.critic_head = SimpleNamespace(critic_action_horizon=16)
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
    class Transform:
        def set_metadata(self, metadata):
            self.metadata = metadata
        def eval(self):
            return self
    actor_model = torch.nn.Module()
    actor_model.action_horizon = 16
    actor = SimpleNamespace(model=actor_model, embodiment_tag=SimpleNamespace(value='new_embodiment'))
    metadata_dir = tmp_path / 'experiment_cfg'
    metadata_dir.mkdir(exist_ok=True)
    (metadata_dir / 'metadata.json').write_text(json.dumps({'new_embodiment': {}}))
    constructor = extracted('__init__', dict(
        copy=copy, json=json, torch=torch, Path=Path, COMPUTE_DTYPE=torch.float32,
        GR00T_N1_5_DEAS_Critic=Critic, Gr00tPolicy=lambda *args: actor,
        DatasetMetadata=SimpleNamespace(model_validate=lambda value: value),
    ))
    instance = SimpleNamespace()
    constructor(instance, tmp_path, tmp_path, 'new_embodiment', {}, Transform(), device='cpu')
    return instance
