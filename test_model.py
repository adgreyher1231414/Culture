import unittest
import torch
from model import CAMTF
from losses import objective
class TestModel(unittest.TestCase):
    def test_forward_backward(self):
        # Tensor-only unit test, not benchmark observations or experimental results.
        n=4; model=CAMTF(8,10,[3]*4)
        t=torch.ones(n,8); v=torch.ones(n,10); e=torch.ones(n,22); m=torch.ones(n,6); ids=torch.ones(n,4,dtype=torch.long); q=torch.ones(n,5); present=torch.ones(n,3,dtype=torch.bool); present[0,0]=False
        out=model(t,v,e,m,ids,q,present); self.assertEqual(out[2][0,0].item(),0); self.assertTrue(torch.allclose(out[2].sum(1),torch.ones(n)))
        loss=objective(out,torch.ones(n),torch.tensor([0.,1.,0.,1.]),torch.tensor([0.,.3,.6,.9]),torch.tensor(1.)); self.assertTrue(torch.isfinite(loss)); loss.backward()
if __name__=='__main__': unittest.main()
