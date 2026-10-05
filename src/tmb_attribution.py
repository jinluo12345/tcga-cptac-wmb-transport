"""Gradient-times-input attribution for fixed external TMB predictions."""
from pathlib import Path
import json, copy
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tmb_gpu_experiment import AssayAwareHetero

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed_tmb'; CK=ROOT/'results/tmb_confirmatory_gpu/checkpoints'; OUT=ROOT/'results/tmb_attribution'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(11,23,47)

def one(model,x,device):
    model.eval(); sums_abs=[]; sums_signed=[]; n=0
    for i in range(0,len(x),64):
        xb=torch.from_numpy(x[i:i+64]).to(device).requires_grad_(True); model.zero_grad(set_to_none=True); mean,_,_=model(xb,0.0); mean.sum().backward(); g=xb.grad.detach().cpu().numpy(); z=x[i:i+64]; a=g*z; sums_abs.append(np.abs(a).sum(0)); sums_signed.append(a.sum(0)); n+=len(z)
    return np.sum(sums_abs,0)/n,np.sum(sums_signed,0)/n

def main():
    X=np.load(DATA/'expression_log1p_tpm.npy',mmap_mode='r'); meta=pd.read_csv(DATA/'sample_metadata.csv'); panel=np.asarray((DATA/'gene_panel.txt').read_text().splitlines()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out_json={}
    for source,target in (('TCGA','CPTAC'),('CPTAC','TCGA')):
        direction=f'{source}_to_{target}'
        si=np.flatnonzero(meta.cohort.to_numpy()==source); ti=np.flatnonzero(meta.cohort.to_numpy()==target); abs_seed=[]; signed_seed=[]; sample_counts=[]
        for seed in SEEDS:
            tr,cal=train_test_split(si,test_size=.2,random_state=seed); xtr=np.asarray(X[tr],np.float32); mu=np.nanmean(xtr,0); sd=np.nanstd(xtr,0); sd[~np.isfinite(sd)|(sd<1e-5)]=1.
            pred_ids=pd.read_csv(ROOT/'results/tmb_confirmatory_gpu/external_predictions.csv').query('direction==@direction and seed==@seed and split=="external_holdout"').case_id.astype(str).tolist()
            if not pred_ids:
                raise RuntimeError(f'no held-out external IDs for {direction} seed {seed}')
            idx=meta.index[meta.case_id.astype(str).isin(pred_ids)].to_numpy(); xt=(np.nan_to_num(np.asarray(X[idx],np.float32),nan=0)-mu)/sd
            if len(idx) != len(pred_ids):
                raise RuntimeError(f'held-out ID mismatch for {direction} seed {seed}: {len(idx)} vs {len(pred_ids)}')
            rng=np.random.RandomState(seed+900); sel=rng.choice(len(xt),size=min(512,len(xt)),replace=False); sample_counts.append(len(sel)); model=AssayAwareHetero(X.shape[1]); ck=torch.load(CK/f'{source}_to_{target}_seed{seed}.pt',map_location=device); model.load_state_dict(ck['state_dict']); model.to(device); a,s=one(model,xt[sel],device); abs_seed.append(a); signed_seed.append(s)
        aa=np.stack(abs_seed); ss=np.stack(signed_seed); mean_abs=aa.mean(0); mean_signed=ss.mean(0); sign_cons=np.abs(np.mean(np.sign(ss),0)); order=np.argsort(-mean_abs); df=pd.DataFrame({'gene_id':panel,'mean_abs_attribution':mean_abs,'mean_signed_attribution':mean_signed,'sign_consistency':sign_cons,'rank':0}); df.loc[order,'rank']=np.arange(1,len(panel)+1); df=df.sort_values('rank'); df.to_csv(OUT/f'{source}_to_{target}_gene_importance.csv',index=False); top=df.head(50); out_json[f'{source}_to_{target}']={'n_external':int(sum(sample_counts)),'n_attribution_samples_per_seed':sample_counts,'top_abs':top[['gene_id','mean_abs_attribution','mean_signed_attribution','sign_consistency']].to_dict('records'),'top_positive':df.sort_values('mean_signed_attribution',ascending=False).head(20)[['gene_id','mean_signed_attribution','sign_consistency']].to_dict('records'),'top_negative':df.sort_values('mean_signed_attribution').head(20)[['gene_id','mean_signed_attribution','sign_consistency']].to_dict('records')}
    (OUT/'summary.json').write_text(json.dumps(out_json,indent=2))
    (OUT/'run_manifest.json').write_text(json.dumps({'checkpoint_source':'results/tmb_confirmatory_gpu/checkpoints','prediction_source':'results/tmb_confirmatory_gpu/external_predictions.csv','external_split':'held-out target evaluation only','preprocessing':'source development-train mean/std for each direction and seed','method':'gradient-times-input on final refit model','seeds':list(SEEDS)},indent=2))
    print(json.dumps({k:{'top_abs':[z['gene_id'] for z in v['top_abs'][:10]]} for k,v in out_json.items()}),flush=True)
if __name__=='__main__': main()
