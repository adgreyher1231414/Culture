import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier

def features(ds,mode,ix,missing=0,seed=0):
    t,v,e,m,ids,q,p=[x[ix].numpy().copy() for x in ds.tensors[:7]]
    p[:,:2]&=~(np.random.default_rng(seed).random((len(ix),2))<missing)
    t[~p[:,0]]=0; v[~p[:,1]]=0; m[:,0]=p[:,0]; m[:,3]=p[:,1]
    return e if mode=='structured' else np.c_[e,t,m,p] if mode=='text' else np.c_[e,t,v,m,p]

def fit(ds,tr,mode,seed):
    x=features(ds,mode,tr); y,r=[z[tr].numpy() for z in ds.tensors[7:9]]
    a,b=(Ridge(),LogisticRegression(max_iter=1000,class_weight='balanced')) if mode=='structured' else (HistGradientBoostingRegressor(random_state=seed,early_stopping=False),HistGradientBoostingClassifier(random_state=seed,early_stopping=False,class_weight='balanced'))
    return a.fit(x,y),b.fit(x,r)

def predict(models,ds,ix,mode,missing=0,seed=0):
    x=features(ds,mode,ix,missing,seed); return models[0].predict(x),models[1].predict_proba(x)[:,1]
