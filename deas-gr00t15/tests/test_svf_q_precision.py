import unittest
import torch
from gr00t.model.svf.precision import q_forward_fp32

class QPrecisionTest(unittest.TestCase):
    def test_outer_autocast_and_tf32_do_not_change_q_precision(self):
        class Q(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer=torch.nn.Linear(6,2)
            def forward(self,f,s,a):
                assert not torch.backends.cuda.matmul.allow_tf32
                return self.layer(torch.cat((f,s,a),-1))
        q=Q()
        x=torch.randn(3,2).bfloat16()
        old=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            with torch.autocast('cpu',dtype=torch.bfloat16):
                result=q_forward_fp32(q,x,x,x)
            self.assertEqual(result.dtype,torch.float32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            result.sum().backward()
            self.assertTrue(torch.isfinite(q.layer.weight.grad).all())
        finally:torch.backends.cuda.matmul.allow_tf32=old


class ProjectionPrecisionTest(unittest.TestCase):
    def test_projection_avoids_outer_autocast_and_keeps_gradient(self):
        from gr00t.model.svf.precision import projection_forward_fp32
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer=torch.nn.Linear(2,2)
            def forward(self,x,ids):
                assert x.dtype == torch.float32
                assert ids.dtype == torch.long
                assert not torch.backends.cuda.matmul.allow_tf32
                return self.layer(x)
        layer=Projection()
        x=torch.randn(3,2,requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            y=projection_forward_fp32(layer,x,torch.zeros(3,dtype=torch.long))
        self.assertEqual(y.dtype,torch.float32)
        torch.testing.assert_close(y,layer.layer(x).tanh())
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
