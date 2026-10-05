"""Confirmatory all-source refit for cross-cohort TMB regression.

For each direction and seed, all source patients are used for the supervised
refit with fixed 50 epochs. Target patients are split 50/50 using a declared
seed; only the unlabeled adaptation half is seen by the domain head. External
metrics use the disjoint held-out half. No target labels enter fitting or
calibration.
"""
from pathlib import Path
import argparse, json, math, random, copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed_tmb'
OUT = ROOT / 'results' / 'tmb_confirmatory_gpu'
(OUT / 'checkpoints').mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class GR(Function):
    @staticmethod
    def forward(ctx, x, c): ctx.c = float(c); return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.c * g, None


def gr(x, c): return GR.apply(x, c)


class AssayAwareHetero(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(p, 512), nn.LayerNorm(512), nn.GELU(),
                                     nn.Dropout(.15), nn.Linear(512, 128), nn.GELU())
        self.mean = nn.Linear(128, 1)
        self.logvar = nn.Linear(128, 1)
        self.domain = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, x, domain_coeff=0.0):
        h = self.encoder(x)
        return self.mean(h).squeeze(1), self.logvar(h).squeeze(1), self.domain(gr(h, domain_coeff)).squeeze(1)


def predict(model, x, device):
    model.eval(); means=[]; scales=[]
    with torch.no_grad():
        for i in range(0, len(x), 512):
            m, lv, _ = model(torch.from_numpy(x[i:i+512]).to(device), 0.0)
            means.append(m.cpu().numpy())
            scales.append(torch.exp(0.5 * torch.clamp(lv, -6, 4)).cpu().numpy())
    return np.concatenate(means), np.concatenate(scales)


def metric(y, mean, scale):
    scale=np.maximum(np.asarray(scale,float),1e-5)
    return {
        'n': int(len(y)),
        'rmse': float(mean_squared_error(y,mean)**.5),
        'mae': float(mean_absolute_error(y,mean)),
        'spearman': float(spearmanr(y,mean).statistic),
        'r2': float(r2_score(y,mean)),
        'coverage_50': float(np.mean(np.abs(y-mean)<=.67449*scale)),
        'coverage_90': float(np.mean(np.abs(y-mean)<=1.64485*scale)),
        'coverage_95': float(np.mean(np.abs(y-mean)<=1.95996*scale)),
        'width_90': float(np.mean(2*1.64485*scale)),
    }


def fit_all_source(xsrc, ysrc, xtarget_adapt, epochs, batch_size, seed):
    seed_all(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AssayAwareHetero(xsrc.shape[1]).to(device)
    src=DataLoader(TensorDataset(torch.from_numpy(xsrc), torch.from_numpy(ysrc.astype(np.float32))), batch_size=batch_size, shuffle=True)
    tgt=DataLoader(torch.from_numpy(xtarget_adapt), batch_size=batch_size, shuffle=True)
    ti=iter(tgt); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-4); bce=nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train(); coeff=.35*(2/(1+math.exp(-8*epoch/max(1,epochs-1)))-1)
        for xb,yb in src:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True)
            mean,lv,_=model(xb,0.0); lv=torch.clamp(lv,-6,4)
            loss=.5*(torch.exp(-lv)*(mean-yb).pow(2)+lv).mean()
            try: xu=next(ti).to(device)
            except StopIteration: ti=iter(tgt); xu=next(ti).to(device)
            z=torch.cat([xb,xu]); dom_y=torch.cat([torch.zeros(len(xb),device=device),torch.ones(len(xu),device=device)])
            _,_,dlogit=model(z,coeff); loss=loss+bce(dlogit,dom_y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    return model, device


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--seeds',default='11,23,47'); args=ap.parse_args()
    seeds=[int(x) for x in args.seeds.split(',') if x]
    print({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'},flush=True)
    X=np.load(DATA/'expression_log1p_tpm.npy',mmap_mode='r'); meta=pd.read_csv(DATA/'sample_metadata.csv'); y=meta.log1p_tmb.to_numpy(np.float32)
    rows=[]; preds=[]
    for source,target in (('TCGA','CPTAC'),('CPTAC','TCGA')):
        si=np.flatnonzero(meta.cohort.to_numpy()==source); ti=np.flatnonzero(meta.cohort.to_numpy()==target)
        for seed in seeds:
            # Target split is independent of model fitting; only adaptation expression is exposed.
            adapt, ext = train_test_split(ti, test_size=.5, random_state=seed+9000)
            # Freeze interval scale on a development checkpoint's held-out
            # source calibration split. The final point predictor is then
            # refit on all source patients with the same fixed settings.
            train_idx, calib = train_test_split(si, test_size=.2, random_state=seed)
            xsrc=np.asarray(X[si],np.float32); xtrain=np.asarray(X[train_idx],np.float32)
            xadapt=np.asarray(X[adapt],np.float32); xext=np.asarray(X[ext],np.float32); xcal=np.asarray(X[calib],np.float32)
            # Preprocessing statistics are frozen from development train and
            # reused for development calibration and the final all-source fit.
            mu=np.nanmean(xtrain,0); sd=np.nanstd(xtrain,0); sd[~np.isfinite(sd)|(sd<1e-5)]=1
            xsrc,xtrain,xadapt,xext,xcal=tuple((np.nan_to_num(z,nan=0)-mu)/sd for z in (xsrc,xtrain,xadapt,xext,xcal))
            dev_model,dev_device=fit_all_source(xtrain,y[train_idx],xadapt,args.epochs,args.batch_size,seed+10000)
            cm,cs=predict(dev_model,xcal,dev_device)
            factor=float(np.clip(np.std(y[calib]-cm)/(np.mean(cs)+1e-6),.25,5.0))
            # Final all-source model is used for point predictions only.
            model,device=fit_all_source(xsrc,y[si],xadapt,args.epochs,args.batch_size,seed+10000)
            em,es=predict(model,xext,device); es*=factor; cs*=factor
            direction=f'{source}_to_{target}'
            z=metric(y[ext],em,es); z.update({'direction':direction,'source':source,'target':target,'seed':seed,'model':'assay_aware_refit','split':'external_holdout','epochs':args.epochs,'scale_factor':factor,'source_train_n':int(len(train_idx)),'source_calibration_n':int(len(calib)),'target_adaptation_n':int(len(adapt)),'target_external_n':int(len(ext))})
            rows.append(z)
            cz=metric(y[calib],cm,cs); cz.update({'direction':direction,'source':source,'target':target,'seed':seed,'model':'assay_aware_refit','split':'source_calibration_dev_checkpoint','epochs':args.epochs,'scale_factor':factor,'source_train_n':int(len(train_idx)),'source_calibration_n':int(len(calib))})
            rows.append(cz)
            ck=OUT/'checkpoints'/f'{direction}_seed{seed}.pt'; torch.save({'state_dict':model.state_dict(),'source':source,'target':target,'seed':seed,'epochs':args.epochs,'scale_factor':factor,'development_train_n':len(train_idx),'calibration_n':len(calib),'adaptation_n':len(adapt),'external_n':len(ext)},ck)
            for ii,truth,pred,scale in zip(ext,y[ext],em,es):
                preds.append({'direction':direction,'seed':seed,'case_id':meta.iloc[ii].case_id,'cohort':target,'lineage':meta.iloc[ii].label,'true':float(truth),'pred':float(pred),'scale':float(scale),'lo90':float(pred-1.64485*scale),'hi90':float(pred+1.64485*scale),'split':'external_holdout'})
            print(json.dumps(z,sort_keys=True),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame(preds).to_csv(OUT/'external_predictions.csv',index=False)
    (OUT/'run_manifest.json').write_text(json.dumps({'seeds':seeds,'epochs':args.epochs,'batch_size':args.batch_size,'directions':[['TCGA','CPTAC'],['CPTAC','TCGA']],'model':'assay_aware_heteroscedastic','target_split':'50/50 adaptation/evaluation, random_state=seed+9000','source_refit':'all source patients for final point predictor','development_calibration':'20% source calibration held out from development checkpoint; frozen scale reused for final all-source refit','preprocessing':'development-train mean/std frozen for both development and final refit','target_labels_used_for_fit':False},indent=2))

if __name__=='__main__': main()
