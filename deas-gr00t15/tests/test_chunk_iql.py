import numpy as np
import torch
from gr00t.model.iql.core import chunk_fields,expectile_loss,td_target,ScalarIQL


def test_complete_terminal_and_partial_chunk():
    r=np.zeros(20);r[-1]=1
    ends=np.zeros(20,bool);ends[-1]=True
    f=chunk_fields(r,ends,ends,4,16,.99)
    assert f['chunk_valid']==1 and f['bootstrap_mask']==0
    assert abs(f['chunk_return']-.99**15)<1e-6
    f=chunk_fields(r,ends,ends,12,16,.99)
    assert f['chunk_valid']==0 and f['valid'].sum()==8
    assert abs(f['chunk_return']-.99**7)<1e-6


def test_timeout_not_terminal_and_missing_next_excluded():
    r=np.zeros(40);ends=np.zeros(40,bool);ends[-1]=True
    term=np.zeros(40,bool)
    f=chunk_fields(r,ends,term,0,16,.99)
    assert f['chunk_valid']==1 and f['bootstrap_mask']==1 and f['next_index']==16
    f=chunk_fields(r,ends,term,24,16,.99)
    assert f['valid'].all() and f['chunk_valid']==0 and f['bootstrap_mask']==1
    ends[15]=True
    f=chunk_fields(r,ends,term,0,16,.99)
    assert f['chunk_valid']==1 and f['bootstrap_mask']==1


def test_first_transition_terminal_and_zero_padding():
    r=np.array([1.,100.,100.]);ends=np.array([True,False,True])
    f=chunk_fields(r,ends,ends,0,16,.99)
    assert f['chunk_valid']==0 and f['valid'].sum()==1 and f['chunk_return']==1


def test_expectile_direction_mask_and_target_gradient():
    diff=torch.tensor([2.,-2.,100.],requires_grad=True)
    loss=expectile_loss(diff,.7,torch.tensor([1.,1.,0.]))
    torch.testing.assert_close(loss,torch.tensor(2.))
    loss.backward();torch.testing.assert_close(diff.grad,torch.tensor([1.4,-.6,0.]))
    result=td_target(torch.tensor([1.,0.]),torch.tensor([0.,1.]),torch.tensor([float('nan'),2.]),.99,16)
    torch.testing.assert_close(result,torch.tensor([1.,2*.99**16]))


def test_scalar_shapes_and_ema():
    h=ScalarIQL(feature_dim=4,state_dim=2,action_dim=1,horizon=2,hidden=8,depth=1)
    f=torch.randn(3,4);s=torch.randn(3,1,2);a=torch.randn(3,2,1)
    q1,q2=h.q(f,s,a)
    assert q1.shape==q2.shape==h.v(f,s).shape==(3,)
    before=next(h.target_q1.parameters()).detach().clone()
    with torch.no_grad():next(h.q1.parameters()).add_(2)
    h.update_target(.1)
    torch.testing.assert_close(next(h.target_q1.parameters()),before+.2)
    assert all(not p.requires_grad for p in h.target_q1.parameters())
