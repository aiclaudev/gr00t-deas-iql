from types import SimpleNamespace
import torch
from torch import nn
from gr00t.model.iql.critic import N17IQLCritic
from gr00t.model.iql.core import chunk_fields

class Backbone(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(8,8)
    def prepare_input(self,x): return x
    def forward(self,x):return {'backbone_features':self.linear(x['tokens'])}

def make_model():
    actor=nn.Module();actor.config=SimpleNamespace(action_horizon=40,backbone_embedding_dim=8,
        max_state_dim=4,state_history_length=1,max_action_dim=3,max_num_embodiments=2)
    actor.backbone=Backbone();actor.action_head=nn.Module();actor.action_head.vlln=nn.LayerNorm(8)
    actor.action_head.vl_self_attention=nn.Identity()
    return N17IQLCritic(actor).train()

def batch():
    return dict(tokens=torch.randn(2,3,8),state=torch.randn(2,1,4),
                action=torch.randn(2,40,3),action_mask=torch.ones(2,40,3),
                embodiment_id=torch.zeros(2,dtype=torch.long))

def test_loss_freezing_and_terminal_target():
    torch.set_num_threads(2)
    model=make_model();a=batch();b=batch()
    result=model(a,b,[-2.,-4.],[0.,1.],[1.,0.])
    torch.testing.assert_close(result['target'][0],torch.tensor(-2.))
    result['loss'].backward()
    assert all(p.grad is None for m in [model.backbone,model.vlln,model.head.target_critic] for p in m.parameters())
    assert any(p.grad is not None for p in model.head.backbone_encoder.parameters())
    assert any(p.grad is not None for p in model.head.value.parameters())
    assert not model.backbone.training and not model.vlln.training

def test_action_horizon_and_padding():
    model=make_model();a=batch();a['action_mask'][:,:,2]=0
    expected=model.score_actions(a)
    a['action'][:,16:]=1e6;a['action'][:,:,2]=1e6
    torch.testing.assert_close(model.score_actions(a),expected)
    torch.testing.assert_close(model.score_actions(a, actions=a['action'][:, :16]), expected)

def test_qc_validity_separate_from_bootstrap():
    rewards=[-1.]*32;ends=[False]*31+[True]
    terminal=[False]*32
    assert chunk_fields(rewards,ends,terminal,16,16,.99)['chunk_valid']==0
    terminal[-1]=True
    fields=chunk_fields(rewards,ends,terminal,16,16,.99)
    assert fields['chunk_valid']==1 and fields['bootstrap_mask']==0


def test_soft_value_target_mask_and_gradient_isolation():
    torch.set_num_threads(2)
    model=make_model();a=batch();before=model.score_actions(a)
    soft=model.enable_soft_value()
    torch.testing.assert_close(model.score_actions(a),before)
    assert model.enable_soft_value() is soft
    a['action_mask'][:,:,2]=0
    x=torch.randn(2,40,3,requires_grad=True)
    teacher=torch.tensor([[-3.,-2.],[-3.,-4.]],requires_grad=True)
    out=model.soft_value_loss(a,x,torch.tensor([.2,.8]),teacher,.5,chunk_valid=[1,0])
    torch.testing.assert_close(out['soft_value_target'][0],torch.tensor(-3.))
    assert out['soft_values'].shape==(2,2)
    out['loss'].backward()
    assert teacher.grad is None
    assert torch.count_nonzero(x.grad[:,:,2])==0
    assert torch.count_nonzero(x.grad[:,16:])==0
    assert torch.count_nonzero(x.grad[1])==0
    assert any(p.grad is not None for p in soft.parameters())
    for name,p in model.named_parameters():
        if not name.startswith('head.soft_value.'):assert p.grad is None,name
    state=model.head.state_dict();other=make_model();other.enable_soft_value();other.head.load_state_dict(state)
    for k,v in state.items():torch.testing.assert_close(other.head.state_dict()[k],v)


def test_soft_value_save_load_metadata():
    import tempfile
    from unittest.mock import patch
    model=make_model();model.spec['actor_checkpoint']='/home/nas_main/dohyunlee/mock-actor'
    model.enable_soft_value()
    def actor_factory(path,horizon):
        fresh=make_model();fresh.spec['actor_checkpoint']=path
        return fresh
    with tempfile.TemporaryDirectory() as directory:
        model.save_head(directory)
        with patch.object(N17IQLCritic,'from_actor_checkpoint',side_effect=actor_factory):
            restored=N17IQLCritic.load_head(directory)
        assert restored.spec==model.spec
        for k,v in model.head.state_dict().items():
            torch.testing.assert_close(restored.head.state_dict()[k],v)
