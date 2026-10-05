"""Confirmatory ablations under the held-out target protocol.

Each variant gets a development fit (source 80% + unlabeled target adaptation)
for source-only calibration, then a fixed-epoch all-source refit for point
predictions. Target labels are used only for the disjoint evaluation half.
"""
from pathlib import Path
import argparse, copy, json, math, random
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from tmb_gpu_ablation_corrected import MLPRegressor, AssayAwareHetero, HeteroNoDomain, AssayAwareHomo, gr

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed_tmb'; OUT=ROOT/'results/ablation'; (OUT/'checkpoints').mkdir(parents=True,exist_ok=True)
SEEDS=(11,23,47); VARIANTS=('mlp_no_domain','hetero_no_domain','hetero_domain','homo_domain')

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def pred(model,x,device):
 model.eval(); means=[]; scales=[]
 with torch.no_grad():
  for i in range(0,len(x),512):
   m,lv,_=model(torch.from_numpy(x[i:i+512]).to(device),0.); means.append(m.cpu().numpy()); scales.append(np.ones(len(m),np.float32) if lv is None else torch.exp(.5*torch.clamp(lv,-6,4)).cpu().numpy())
 return np.concatenate(means),np.concatenate(scales)

def met(y,p,s):
 s=np.maximum(np.asarray(s,float),1e-5)
 return {'n':int(len(y)),'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),'spearman':float(spearmanr(y,p).statistic),'r2':float(r2_score(y,p)),'coverage_50':float(np.mean(np.abs(y-p)<=.67449*s)),'coverage_90':float(np.mean(np.abs(y-p)<=1.64485*s)),'coverage_95':float(np.mean(np.abs(y-p)<=1.95996*s)),'width_90':float(np.mean(2*1.64485*s))}

def make_model(variant,p):
 return {'mlp_no_domain':MLPRegressor,'hetero_no_domain':HeteroNoDomain,'hetero_domain':AssayAwareHetero,'homo_domain':AssayAwareHomo}[variant](p)

def fit_fixed(variant,xsrc,ysrc,xtarget,epochs,batch_size,seed):
 seed_all(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=make_model(variant,xsrc.shape[1]).to(device)
 src=DataLoader(TensorDataset(torch.from_numpy(xsrc),torch.from_numpy(ysrc.astype(np.float32))),batch_size=batch_size,shuffle=True); tgt=DataLoader(torch.from_numpy(xtarget),batch_size=batch_size,shuffle=True); ti=iter(tgt); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-4); bce=nn.BCEWithLogitsLoss(); use_domain=variant in ('hetero_domain','homo_domain')
 for epoch in range(epochs):
  model.train(); coeff=.35*(2/(1+math.exp(-8*epoch/max(1,epochs-1)))-1)
  for xb,yb in src:
   xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); mean,lv,_=model(xb,0.0)
   if variant=='mlp_no_domain': loss=.5*(mean-yb).pow(2).mean()
   else:
    lv=torch.clamp(lv,-6,4); loss=.5*(torch.exp(-lv)*(mean-yb).pow(2)+lv).mean()
   if use_domain:
    try: xu=next(ti).to(device)
    except StopIteration: ti=iter(tgt); xu=next(ti).to(device)
    z=torch.cat([xb,xu]); dy=torch.cat([torch.zeros(len(xb),device=device),torch.ones(len(xu),device=device)]); _,_,dl=model(z,coeff); loss=loss+bce(dl,dy)
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
 return model,device

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--seeds',default='11,23,47'); a=ap.parse_args(); seeds=[int(x) for x in a.seeds.split(',') if x]
 X=np.load(DATA/'expression_log1p_tpm.npy',mmap_mode='r'); meta=pd.read_csv(DATA/'sample_metadata.csv'); y=meta.log1p_tmb.to_numpy(np.float32); rows=[]; preds=[]
 print({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'},flush=True)
 for source,target in (('TCGA','CPTAC'),('CPTAC','TCGA')):
  si=np.flatnonzero(meta.cohort.to_numpy()==source); ti=np.flatnonzero(meta.cohort.to_numpy()==target)
  for seed in seeds:
   adapt,ext=train_test_split(ti,test_size=.5,random_state=seed+9000); train_idx,calib=train_test_split(si,test_size=.2,random_state=seed)
   xsrc=np.asarray(X[si],np.float32); xtrain=np.asarray(X[train_idx],np.float32); xa=np.asarray(X[adapt],np.float32); xe=np.asarray(X[ext],np.float32); xc=np.asarray(X[calib],np.float32); mu=np.nanmean(xtrain,0); sd=np.nanstd(xtrain,0); sd[~np.isfinite(sd)|(sd<1e-5)]=1.; xsrc,xtrain,xa,xe,xc=tuple((np.nan_to_num(z,nan=0)-mu)/sd for z in (xsrc,xtrain,xa,xe,xc))
   for vi,var in enumerate(VARIANTS):
    offset=seed+10000+vi*1000; dev,dd=fit_fixed(var,xtrain,y[train_idx],xa,a.epochs,a.batch_size,offset); cm,cs=pred(dev,xc,dd); factor=float(np.clip(np.std(y[calib]-cm)/(np.mean(cs)+1e-6),.25,5.)); final,fd=fit_fixed(var,xsrc,y[si],xa,a.epochs,a.batch_size,offset); em,es=pred(final,xe,fd); es*=factor; direction=f'{source}_to_{target}'
    cz=met(y[calib],cm,cs*factor); cz.update({'direction':direction,'seed':seed,'variant':var,'evaluation':'source_calibration_dev_checkpoint','epochs':a.epochs,'scale_factor':factor,'source_train_n':len(train_idx),'source_calibration_n':len(calib)}); rows.append(cz)
    rng=np.random.RandomState(seed+777); mask=rng.rand(*xe.shape)<.10; xem=xe.copy(); xem[mask]=0.; xen=xe+rng.normal(0,.1,xe.shape).astype(np.float32)
    for ev,zx in [('clean',xe),('mask10',xem),('noise01',xen)]:
     pm,ps=pred(final,zx,fd); ps*=factor; z=met(y[ext],pm,ps); z.update({'direction':direction,'seed':seed,'variant':var,'evaluation':ev,'epochs':a.epochs,'scale_factor':factor,'source_train_n':len(train_idx),'source_calibration_n':len(calib),'target_adaptation_n':len(adapt),'target_external_n':len(ext)}); rows.append(z)
     if ev=='clean':
      for ii,tv,pv,sc in zip(ext,y[ext],pm,ps): preds.append({'direction':direction,'seed':seed,'variant':var,'case_id':meta.iloc[ii].case_id,'cohort':target,'lineage':meta.iloc[ii].label,'true':float(tv),'pred':float(pv),'scale':float(sc),'lo90':float(pv-1.64485*sc),'hi90':float(pv+1.64485*sc),'split':'external_holdout'})
    torch.save({'state_dict':final.state_dict(),'variant':var,'source':source,'target':target,'seed':seed,'epochs':a.epochs,'scale_factor':factor,'source_train_n':len(train_idx),'source_calibration_n':len(calib),'target_adaptation_n':len(adapt),'target_external_n':len(ext)},OUT/'checkpoints'/f'{direction}_{var}_seed{seed}.pt')
    print(json.dumps(rows[-3],sort_keys=True),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame(preds).to_csv(OUT/'external_predictions.csv',index=False); (OUT/'run_manifest.json').write_text(json.dumps({'seeds':seeds,'variants':list(VARIANTS),'epochs':a.epochs,'batch_size':a.batch_size,'target_split':'50/50 adaptation/evaluation random_state=seed+9000','source_development':'80/20 train/calibration random_state=seed','source_refit':'all source patients fixed epochs','preprocessing':'development-train mean/std frozen','calibration':'development checkpoint held-out source calibration only','target_labels_used_for_fit':False},indent=2))
if __name__=='__main__': main()
