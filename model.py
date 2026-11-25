import torch
from torch import nn

def mlp(a,h,b):
    return nn.Sequential(nn.Linear(a,h),nn.ReLU(),nn.Linear(h,b))

class CAMTF(nn.Module):
    def __init__(self, text_dim, image_dim, sizes, rule=True, drop=None):
        super().__init__(); self.rule=rule; self.drop=drop
        self.text=nn.Sequential(mlp(text_dim,32,32),nn.LayerNorm(32))
        self.image=nn.Sequential(mlp(image_dim,32,32),nn.LayerNorm(32))
        self.econ=mlp(22,24,24); self.project=nn.Linear(24,32)
        self.gate=nn.Linear(96+6+3+1,3)
        self.effects=nn.ModuleList(nn.Embedding(n,16,padding_idx=0) for n in sizes)
        self.value=nn.Linear(32,1); self.risk=nn.Linear(72,1); self.rule_net=mlp(5,16,1)
    def forward(self,t,v,e,m,ids,q,present):
        present=present.clone(); m=m.clone()
        if self.drop is not None: present[:,self.drop]=False
        m[:,3]=m[:,3]*present[:,1]; m[:,0]=m[:,0]*present[:,0]
        ht=self.text(t)*present[:,0,None]; hv=self.image(v)*present[:,1,None]
        he=self.econ(e)*present[:,2,None]; hp=self.project(he)*present[:,2,None]
        interaction=(ht*hv).sum(1,keepdim=True)/32**0.5
        logits=self.gate(torch.cat([ht,hv,hp,m,present.float(),interaction],1))
        if not present.any(1).all(): raise ValueError('Record has no active modality')
        a=logits.masked_fill(~present,-torch.inf).softmax(1)
        f=(torch.stack([ht,hv,hp],1)*a[:,:,None]).sum(1)
        g=sum(emb(ids[:,j]) for j,emb in enumerate(self.effects))
        risk=self.risk(torch.cat([f,g,he],1)).squeeze(1)
        if self.rule: risk=risk+self.rule_net(q).squeeze(1)
        return self.value(f).squeeze(1),risk,a
