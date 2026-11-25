import torch
import torch.nn.functional as F

def objective(out,y,r,score,weight,rule=True):
    v,z,a=out; p=z.sigmoid()
    loss=F.huber_loss(v,y,delta=5.)+.55*F.binary_cross_entropy_with_logits(z,r,pos_weight=weight)
    entropy=-(a*a.clamp_min(1e-12).log()).sum(1).mean()
    if rule:
        d=score[:,None]-score[None,:]; valid=torch.isfinite(d)&(d.abs()>.1)
        if valid.any(): loss=loss+.1*(.05-(p[:,None]-p[None,:])[valid]*d[valid]).relu().mean()
    return loss-.05*entropy
