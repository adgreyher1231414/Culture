import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
import train,baselines
from data import load
from metrics import evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--records',required=True); p.add_argument('--embeddings',required=True); p.add_argument('--train-end',required=True); p.add_argument('--val-end',required=True); p.add_argument('--out',default='results'); p.add_argument('--epochs',type=int,default=100); p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46]); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--bootstrap',type=int,default=1000); a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df,ds,ix,sizes=load(a.records,a.embeddings,a.train_end,a.val_end); tr,va,te=ix; y,r=[x[te].numpy() for x in ds.tensors[7:9]]; overall=[]; groups=[]; missing=[]; abl=[]; boot=[]
    pd.concat([df.iloc[j][['record_id','timestamp','outcome_timestamp','brand','risk_label']].assign(split=k) for j,k in zip(ix,['train','validation','test'])]).to_csv(out/'splits.csv',index=False)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    for seed in a.seeds:
        def report(rows,name,v,p,ind=None,**kw):
            ind=np.arange(len(y)) if ind is None else ind; rows.append(dict(seed=seed,model=name,**kw,**evaluate(y[ind],r[ind],v[ind],p[ind])))
        base={m:baselines.fit(ds,tr,m,seed) for m in ['structured','text','early']}; preds={m:baselines.predict(b,ds,te,m) for m,b in base.items()}
        for m,(v,p) in preds.items(): report(overall,m,v,p)
        for rule,name in [(False,'gated'),(True,'backbone')]:
            net=train.fit(ds,ix,sizes,seed,a.device,a.epochs,rule); v,p=train.predict(net,ds,te,a.device); report(overall,name,v,p); torch.save(net.state_dict(),out/f'{name}_{seed}.pt')
        joblib.dump(base,out/f'baselines_{seed}.joblib')
        def blended(level=0):
            v,p=train.predict(net,ds,te,a.device,level,seed); ev,_=baselines.predict(base['early'],ds,te,'early',level,seed); _,sp=preds['structured']; return .65*v+.35*ev,.85*p+.15*sp
        bv,bp=blended(); report(overall,'CA-MTF blended',bv,bp); report(abl,'full backbone',v,p)
        for col in ['category','region']:
            for key,group in df.iloc[te].reset_index(drop=True).groupby(col,dropna=False): report(groups,'CA-MTF blended',bv,bp,group.index.to_numpy(),group_type=col,group=str(key))
        for level in np.arange(6)/10:
            mv,mp=blended(level); report(missing,'CA-MTF blended',mv,mp,missing_percent=int(round(level*100)))
        for name,drop,rule in [('no text',0,True),('no image',1,True),('no structured economics',2,True),('no compliance rule',None,False)]:
            # Without economics, rows with neither semantic modality cannot be evaluated; report the eligible subset.
            subix=ix
            if drop==2:
                subix=[j[ds.tensors[6][j,:2].any(1).numpy()] for j in ix]
                if min(map(len,subix))==0: continue
            n=train.fit(ds,subix,sizes,seed,a.device,a.epochs,rule,drop); av,ap=train.predict(n,ds,subix[2],a.device); ind=np.flatnonzero(np.isin(te,subix[2])); report(abl,name,av,ap,ind=ind)
        pd.DataFrame(dict(record_id=df.iloc[te].record_id.to_numpy(),brand_value=y,risk_label=r,value=bv,risk_probability=bp)).to_csv(out/f'predictions_{seed}.csv',index=False)
        rng=np.random.default_rng(seed); brands=df.iloc[te].brand.fillna('UNKNOWN').to_numpy(); unique=np.unique(brands); clusters=[np.flatnonzero(brands==b) for b in unique]
        for k in range(a.bootstrap):
            j=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]); met=evaluate(y[j],r[j],bv[j],bp[j]); boot.append(dict(seed=seed,replicate=k,**met))
        if seed==a.seeds[0]:
            fig,ax=plt.subplots(); ax.plot([0,1],[0,1],'--'); bins=np.minimum((bp*10).astype(int),9)
            for k in range(10):
                m=bins==k
                if m.any(): ax.scatter(bp[m].mean(),r[m].mean(),s=20+100*m.mean())
            ax.set(xlabel='Predicted probability',ylabel='Observed frequency'); fig.savefig(out/'calibration.png',dpi=200); plt.close(fig)
            if len(np.unique(r))==2:
                for name,fn in [('PR',precision_recall_curve),('ROC',roc_curve)]:
                    x,z,_=fn(r,bp); fig,ax=plt.subplots(); ax.plot(z,x) if name=='PR' else ax.plot(x,z); ax.set(xlabel='Recall' if name=='PR' else 'FPR',ylabel='Precision' if name=='PR' else 'TPR'); fig.savefig(out/f'{name}.png',dpi=200); plt.close(fig)
    for name,rows,keys in [('overall',overall,['model']),('subgroups',groups,['group_type','group']),('missing',missing,['missing_percent']),('ablation',abl,['model']),('bootstrap',boot,['seed'])]:
        if not rows: continue
        tab=pd.DataFrame(rows); tab.to_csv(out/f'{name}.csv',index=False); cols=['MAE','RMSE','R2','F1','AUROC','ECE','Brier']; agg=tab.groupby(keys)[cols]
        (agg.quantile(.025).join(agg.quantile(.975),lsuffix='_lower',rsuffix='_upper') if name=='bootstrap' else agg.agg(['mean','std'])).to_csv(out/f'{name}_summary.csv')
        if name!='bootstrap':
            fig,ax=plt.subplots(figsize=(10,5)); agg.mean()[['F1','AUROC']].plot(kind='bar',ax=ax); ax.set_ylabel('Score'); fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=200); plt.close(fig)
if __name__=='__main__': main()
