"""Unlabeled-target CORAL + PCA-HGB control for the confirmatory protocol.

The adaptation covariance and mean are computed only from the unlabeled target
adaptation half.  Target evaluation rows and labels are never used to fit the
transform or the regressor.
"""
from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed_tmb'; OUT=ROOT/'results/coral_cpu'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(11,23,47)

def met(y,p):
    return {'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),
            'spearman':float(spearmanr(y,p).statistic),'r2':float(r2_score(y,p))}

def coral_fit_transform(xs, xt):
    """Map source PCA scores into the unlabeled target covariance frame."""
    ms=xs.mean(0); mt=xt.mean(0)
    cs=np.cov(xs-ms,rowvar=False)+1e-3*np.eye(xs.shape[1])
    ct=np.cov(xt-mt,rowvar=False)+1e-3*np.eye(xt.shape[1])
    ws=fractional_matrix_power(cs,-0.5).real
    wt=fractional_matrix_power(ct,0.5).real
    A=ws@wt
    return lambda z: (z-ms)@A+mt

def main():
    X=np.load(DATA/'expression_log1p_tpm.npy',mmap_mode='r'); meta=pd.read_csv(DATA/'sample_metadata.csv'); y=meta.log1p_tmb.to_numpy(float)
    rows=[]; preds=[]
    for source,target in [('TCGA','CPTAC'),('CPTAC','TCGA')]:
        si=np.flatnonzero(meta.cohort.to_numpy()==source); ti=np.flatnonzero(meta.cohort.to_numpy()==target)
        for seed in SEEDS:
            tr,va=train_test_split(si,test_size=.2,random_state=seed)
            adapt,ext=train_test_split(ti,test_size=.5,random_state=seed+9000)
            mu=np.nanmean(np.asarray(X[tr],np.float32),axis=0); sd=np.nanstd(np.asarray(X[tr],np.float32),axis=0); sd[~np.isfinite(sd)|(sd<1e-5)]=1.
            def norm(idx): return (np.nan_to_num(np.asarray(X[idx],np.float32),nan=0)-mu)/sd
            xtr,xall,xa,xe=norm(tr),norm(si),norm(adapt),norm(ext)
            pca=PCA(n_components=128,whiten=True,random_state=seed); ztr=pca.fit_transform(xtr); za=pca.transform(xa); ze=pca.transform(xe); zall=pca.transform(xall)
            transform=coral_fit_transform(zall,za)
            zfit=transform(zall); ze_aligned=ze # target rows already in target PCA frame
            model=HistGradientBoostingRegressor(max_iter=220,learning_rate=.06,max_leaf_nodes=31,l2_regularization=1.,random_state=seed)
            model.fit(zfit,y[si]); pred=model.predict(ze_aligned); direction=f'{source}_to_{target}'
            rec={'direction':direction,'seed':seed,'model':'coral_pca_hgb','split':'external_holdout','source_train_n':len(tr),'source_refit_n':len(si),'target_adaptation_n':len(adapt),'target_external_n':len(ext),**met(y[ext],pred)}; rows.append(rec)
            for cid,lin,truth,pv in zip(meta.iloc[ext].case_id,meta.iloc[ext].label,y[ext],pred): preds.append({'direction':direction,'seed':seed,'model':'coral_pca_hgb','case_id':cid,'lineage':lin,'true':float(truth),'pred':float(pv)})
            print(json.dumps(rec),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame(preds).to_csv(OUT/'external_predictions.csv',index=False)
    (OUT/'run_manifest.json').write_text(json.dumps({'method':'PCA-HGB with CORAL source-to-unlabeled-target covariance alignment','target_split':'50/50 adaptation/evaluation random_state=seed+9000','preprocessing':'source-development-train mean/std and source-only PCA','target_labels_used_for_fit':False,'seeds':list(SEEDS)},indent=2))
    print(pd.DataFrame(rows).groupby('direction')[['rmse','mae','spearman','r2']].agg(['mean','std']).round(4).to_string())
if __name__=='__main__': main()
