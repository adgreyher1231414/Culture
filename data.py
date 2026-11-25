import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
FEATURES=['rating','review_count','helpfulness','price','trade_growth','tariff_volatility','trademark_status','sanctions_similarity','origin_documentation','content_restriction','year_offset']
GROUPS=['brand','category','region','year']
REL=['text_present','review_volume_proxy','rating_consistency','image_present','metadata_completeness','provenance_weight']
Q=[6,5,7,8,9]

def load(path,embeddings,train_end,val_end):
    df=pd.read_csv(path); ts=pd.to_datetime(df.timestamp,utc=True); outcome=pd.to_datetime(df.outcome_timestamp,utc=True)
    if df.record_id.duplicated().any() or (outcome<=ts).any(): raise ValueError('Invalid IDs or outcome timestamps')
    a,b=pd.to_datetime(train_end,utc=True),pd.to_datetime(val_end,utc=True)
    if a>=b: raise ValueError('Invalid split boundaries')
    ix=[np.flatnonzero((ts<a)&(outcome<a)),np.flatnonzero((ts>=a)&(ts<b)&(outcome<b)),np.flatnonzero(ts>=b)]
    if min(map(len,ix))==0: raise ValueError('Empty chronological partition')
    tr=ix[0]; df['year']=ts.dt.year
    raw=df[FEATURES].to_numpy(float); miss=~np.isfinite(raw)
    for j in [1,3]:
        if np.any(raw[:,j]<0): raise ValueError('Negative count or price')
        raw[:,j]=np.log1p(raw[:,j])
    median=np.nanmedian(np.where(miss,np.nan,raw)[tr],axis=0); median=np.nan_to_num(median)
    filled=np.where(miss,median,raw); scaler=StandardScaler().fit(filled[tr]); e=np.c_[scaler.transform(filled),miss].astype('float32')
    q=raw[:,Q]; known=np.isfinite(q[:,[0,1,2,4]]).all(1)
    score=np.where(known,.35*q[:,2]+.25*(1-q[:,0])+.2*q[:,4]+.2*np.maximum(q[:,1],0),np.nan)
    q=np.where(np.isfinite(q),q,median[Q]); ids=[]; sizes=[]
    for col in GROUPS:
        vals=df[col].fillna('UNKNOWN').astype(str); vocab={x:i+1 for i,x in enumerate(sorted(set(vals.iloc[tr])))}
        ids.append(vals.map(vocab).fillna(0).to_numpy(int)); sizes.append(len(vocab)+1)
    with np.load(embeddings,allow_pickle=False) as f:
        if not np.array_equal(f['record_id'].astype(str),df.record_id.astype(str).to_numpy()): raise ValueError('Embedding IDs not aligned')
        t,v=f['text'].astype('float32'),f['image'].astype('float32')
    m=df[REL].to_numpy('float32'); present=np.c_[m[:,0]>0,m[:,3]>0,np.ones(len(df),bool)]
    if not np.isfinite(m).all() or not np.isfinite(t[present[:,0]]).all() or not np.isfinite(v[present[:,1]]).all(): raise ValueError('Invalid reliability or embedding values')
    t=np.where(present[:,0,None],t,0); v=np.where(present[:,1,None],v,0)
    y=df.brand_value.to_numpy('float32'); r=df.risk_label.to_numpy('float32')
    if not np.isfinite(y).all() or not ((y>=0)&(y<=100)).all() or not np.isin(r,[0,1]).all(): raise ValueError('Invalid observed labels')
    arrays=[t,v,e,m,np.stack(ids,1),q.astype('float32'),present,y,r,score.astype('float32')]
    ds=TensorDataset(*map(torch.from_numpy,arrays))
    return df,ds,ix,sizes
