import torch
from gr00t.model.svf.policy_td import policy_q_loss


def test_policy_backup_terminal_and_invalid_masks():
    q1=torch.tensor([1.,2.,float('nan')],requires_grad=True)
    q2=torch.tensor([3.,4.,float('nan')],requires_grad=True)
    n1=torch.tensor([5.,float('nan'),7.],requires_grad=True)
    n2=torch.tensor([6.,float('nan'),8.],requires_grad=True)
    loss,target=policy_q_loss(q1,q2,torch.tensor([-2.,0.,0.]),torch.tensor([1.,0.,1.]),n1,n2,torch.tensor([1.,1.,0.]))
    torch.testing.assert_close(target[:2],torch.tensor([-2.+.99**16*5.,0.]))
    assert torch.isfinite(loss) and not target.requires_grad
    loss.backward()
    assert n1.grad is None and n2.grad is None


def test_backup_depends_on_next_policy_q():
    z=torch.zeros(2);ones=torch.ones(2)
    _,a=policy_q_loss(z,z,z,ones,ones,ones,ones)
    _,b=policy_q_loss(z,z,z,ones,3*ones,2*ones,ones)
    torch.testing.assert_close(b-a,ones*.99**16)
