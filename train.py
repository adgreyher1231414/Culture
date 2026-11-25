import copy, random
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from model import CAMTF
from losses import objective
from metrics import evaluate

def predict(model,ds,ix,device,missing=0,seed=0):
    model.eval(); out=[]; rng=np.random.default_rng(seed)
    masks=rng.random((len(ix),2))<missing; offset=0
    with torch.no_grad():
        for batch in DataLoader(Subset(ds,ix),batch_size=512):
            x=[z.to(device) for z in batch[:7]]; n=len(x[0]); x[6]=x[6].clone(); x[6][:,:2]&=~torch.from_numpy(masks[offset:offset+n]).to(device); offset+=n
            v,r,_=model(*x); out.append(torch.stack([v,r.sigmoid()],1).cpu().numpy())
    return np.concatenate(out).T

def fit(ds,ix,sizes,seed,device,epochs=100,rule=True,drop=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr,va,_=ix; model=CAMTF(ds.tensors[0].shape[1],ds.tensors[1].shape[1],sizes,rule,drop).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.0001)
    r=ds.tensors[8][tr]; pos=r.sum().item()
    if pos==0 or pos==len(r): raise ValueError('Training requires both risk classes')
    weight=torch.tensor((len(r)-pos)/pos,device=device); history=[]; best_mae=float('inf'); stale=0
    for epoch in range(epochs):
        model.train()
        for batch in DataLoader(Subset(ds,tr),batch_size=512,shuffle=True):
            *x,y,r,s=[z.to(device) for z in batch]; opt.zero_grad(); loss=objective(model(*x),y,r,s,weight,rule); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
        v,p=predict(model,ds,va,device); met=evaluate(ds.tensors[7][va].numpy(),ds.tensors[8][va].numpy(),v,p)
        history.append((met,copy.deepcopy(model.state_dict())))
        if met['MAE']<best_mae: best_mae=met['MAE']; stale=0
        else: stale+=1
        if stale>=4: break
    candidates=[x for x in history if x[0]['MAE']<=best_mae*1.01]
    best=max(candidates,key=lambda x:(np.nan_to_num(x[0]['AUROC'],nan=-1),x[0]['Recall'],-x[0]['MAE']))
    model.load_state_dict(best[1]); return model
