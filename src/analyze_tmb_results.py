from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'results/analysis'; out.mkdir(parents=True,exist_ok=True)
gpu=pd.read_csv(ROOT/'results/confirmatory_gpu/external_predictions.csv')
gpu['model']='assay_aware_refit'
cpu_path=ROOT/'results/coral_cpu/external_predictions.csv'
cpu_all=pd.read_csv(cpu_path)
# Restrict every CPU baseline to exactly the held-out target IDs used by the confirmatory run.
ids=gpu[['direction','seed','case_id']]
cpu=cpu_all.merge(ids,on=['direction','seed','case_id'],how='inner')
def collapse_unique(g):
    """Average repeated seed predictions per patient for descriptive summaries."""
    keys=['direction','case_id','lineage']
    agg={'true':('true','first'),'pred':('pred','mean')}
    if 'scale' in g.columns:
        agg['scale']=('scale','mean')
    z=g.groupby(keys,as_index=False).agg(**agg)
    if 'scale' in z.columns:
        z['lo90']=z.pred-1.64485*z.scale; z['hi90']=z.pred+1.64485*z.scale
    z['model']=g.model.iloc[0]
    return z
def metrics(g):
 y=g.true.to_numpy(float); p=g.pred.to_numpy(float)
 z={'n':len(g),'rmse':float(np.sqrt(mean_squared_error(y,p))),'mae':float(mean_absolute_error(y,p)),
    'spearman':float(spearmanr(y,p).statistic),'r2':float(r2_score(y,p)),'bias':float(np.mean(p-y))}
 if len(g)>3:
  lr=LinearRegression().fit(p[:,None],y); z.update({'calibration_intercept':float(lr.intercept_),'calibration_slope':float(lr.coef_[0])})
 if {'lo90','hi90','scale'}.issubset(g.columns):
  z.update({'coverage90':float(np.mean((y>=g.lo90)&(y<=g.hi90))),
            'coverage95':float(np.mean((y>=g.pred-1.96*g.scale)&(y<=g.pred+1.96*g.scale))),
            'mean_width90':float(np.mean(g.hi90-g.lo90))})
 return z
rows=[]
for (d,m),g in gpu.groupby(['direction','model']):
 gu=collapse_unique(g); rows.append({'source':'gpu','direction':d,'model':m,'stratum':'overall',**metrics(gu)})
for (d,m),g in cpu.groupby(['direction','model']):
 gu=collapse_unique(g); rows.append({'source':'cpu','direction':d,'model':m,'stratum':'overall',**metrics(gu)})
for (d,m,lin),g in gpu.groupby(['direction','model','lineage']):
 gu=collapse_unique(g); rows.append({'source':'gpu','direction':d,'model':m,'stratum':lin,**metrics(gu)})
for (d,m,lin),g in cpu.groupby(['direction','model','lineage']):
 gu=collapse_unique(g); rows.append({'source':'cpu','direction':d,'model':m,'stratum':lin,**metrics(gu)})
pd.DataFrame(rows).to_csv(out/'stratified_metrics.csv',index=False)
q=[]
for (d,m),g in gpu.groupby(['direction','model']):
 g=collapse_unique(g); g['abs_error']=(g.pred-g.true).abs(); g['true_q']=pd.qcut(g.true,4,duplicates='drop'); g['scale_q']=pd.qcut(g.scale,4,duplicates='drop')
 for col in ['true_q','scale_q']:
  for lev,h in g.groupby(col,observed=True): q.append({'direction':d,'model':m,'stratum':col,'level':str(lev),'n':len(h),'mae':float(h.abs_error.mean()),'coverage90':float(((h.true>=h.lo90)&(h.true<=h.hi90)).mean()),'mean_scale':float(h.scale.mean())})
pd.DataFrame(q).to_csv(out/'error_uncertainty_bins.csv',index=False)
# Per-seed paired deltas on the same held-out patients.
rows=[]
for (d,s),a in gpu.groupby(['direction','seed']):
 b=cpu.query('direction==@d and seed==@s and model=="coral_pca_hgb"')
 z=a[['case_id','true','pred']].merge(b[['case_id','pred']],on='case_id',suffixes=('_candidate','_baseline'))
 y=z.true.to_numpy(float); pc=z.pred_candidate.to_numpy(float); pb=z.pred_baseline.to_numpy(float)
 rows.append({'direction':d,'seed':int(s),'n':len(z),'rmse_candidate':np.sqrt(mean_squared_error(y,pc)),'rmse_baseline':np.sqrt(mean_squared_error(y,pb)),'mae_candidate':mean_absolute_error(y,pc),'mae_baseline':mean_absolute_error(y,pb),'spearman_candidate':spearmanr(y,pc).statistic,'spearman_baseline':spearmanr(y,pb).statistic,'r2_candidate':r2_score(y,pc),'r2_baseline':r2_score(y,pb)})
delta=pd.DataFrame(rows); delta['rmse_improvement']=delta.rmse_baseline-delta.rmse_candidate; delta['mae_improvement']=delta.mae_baseline-delta.mae_candidate; delta['spearman_improvement']=delta.spearman_candidate-delta.spearman_baseline; delta['r2_improvement']=delta.r2_candidate-delta.r2_baseline
delta.to_csv(out/'candidate_vs_coral_by_seed.csv',index=False)
(out/'analysis_manifest.json').write_text(json.dumps({'external_predictions':['results/confirmatory_gpu/external_predictions.csv','results/coral_cpu/external_predictions.csv'],'case_matching':'CPU rows restricted to confirmatory held-out case IDs','descriptive_summaries':'Repeated seed predictions collapsed to one mean prediction per unique case before lineage/error summaries','metrics':['stratified_metrics.csv','error_uncertainty_bins.csv','candidate_vs_coral_by_seed.csv','bootstrap_comparison.json'],'bootstrap':'2000 patient resamples within each fixed direction-seed evaluation set; candidate minus CORAL-PCA-HGB'},indent=2))
print(pd.read_csv(out/'stratified_metrics.csv').query("stratum=='overall'").to_string(index=False))
