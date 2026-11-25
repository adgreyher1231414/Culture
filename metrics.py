import numpy as np
from sklearn import metrics as m

def evaluate(y,r,v,p,threshold=.5):
    pred=p>=threshold; both=len(np.unique(r))==2
    ece=0.
    for k in range(10):
        mask=np.minimum((p*10).astype(int),9)==k
        if mask.any(): ece+=mask.mean()*abs(p[mask].mean()-r[mask].mean())
    fpr,tpr,_=m.roc_curve(r,p) if both else (np.array([0]),np.array([np.nan]),None)
    return dict(N=len(y),Positive=int(r.sum()),Negative=int(len(r)-r.sum()),MAE=m.mean_absolute_error(y,v),RMSE=np.sqrt(m.mean_squared_error(y,v)),R2=m.r2_score(y,v) if len(y)>1 else np.nan,Accuracy=m.accuracy_score(r,pred),Precision=m.precision_score(r,pred,zero_division=0),Recall=m.recall_score(r,pred,zero_division=0),F1=m.f1_score(r,pred,zero_division=0),AUROC=m.roc_auc_score(r,p) if both else np.nan,PR_AUC=m.average_precision_score(r,p) if both else np.nan,Recall_at_FPR20=max(tpr[fpr<=.2]),ECE=ece,Brier=m.brier_score_loss(r,p))
