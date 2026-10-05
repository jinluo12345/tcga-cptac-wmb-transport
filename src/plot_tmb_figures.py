from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        return {"status": "skipped", "reason": "optional nature-figure audit package unavailable"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})
BLUE, ORANGE, TEAL, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#666666", "#CC79A7"

def save(fig, name, axes):
    fig.canvas.draw()
    try:
        # Explicitly pass the primary panel axes.  Calibration adds twin axes;
        # including those auxiliary axes makes the auditor infer invalid groups.
        panel_ids = [chr(ord("a") + i) for i in range(len(axes))]
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=panel_ids,
            json_out=OUT/(name+".alignment.json"), strict=False)
    except Exception as exc:
        (OUT/(name+".alignment_error.txt")).write_text(str(exc))
    fig.savefig(OUT/(name+".pdf"), bbox_inches=None)
    fig.savefig(OUT/(name+".svg"), bbox_inches=None)
    plt.close(fig)

def fig_mechanism():
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    def box(x,y,w,h,text,fc,ec=BLUE,fs=9):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=1.3)
        ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color="#1f2937",multialignment="center")
    def arrow(x1,y1,x2,y2,label=None,style="-"):
        a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,lw=1.2,color="#374151",connectionstyle="arc3,rad=0",linestyle=style)
        ax.add_patch(a)
        if label: ax.text((x1+x2)/2,(y1+y2)/2+0.15,label,ha="center",va="center",fontsize=7,color="#374151")
    box(.25,3.15,1.7,1.0,"Gene expression\n(4,096 genes)","#E8F1FA")
    box(.25,1.0,1.7,1.0,"Cohort label\n(domain target only)","#FDE9D9",ec=ORANGE,fs=8)
    box(2.55,2.35,2.0,1.4,"Shared encoder\n512 → 128\nLayerNorm + GELU", "#E9F5EF",ec=TEAL)
    box(5.25,3.05,1.8,1.0,"Mean head\nμ(x)","#E8F1FA")
    box(5.25,1.75,1.8,1.0,"Variance head\nlog σ²(x)","#E8F1FA")
    box(7.7,2.35,1.9,1.4,"Predicted\nlog1p WMB\n+ 90% interval","#EAF2F8",ec=BLUE)
    box(5.25,.35,1.8,0.85,"Gradient reversal\n+ domain head", "#FDE9D9",ec=ORANGE,fs=8)
    arrow(1.95,3.65,2.55,3.05)
    arrow(1.95,1.5,5.25,.75,"domain label", "--")
    arrow(4.55,3.05,5.25,3.55)
    arrow(4.55,2.75,5.25,2.25)
    arrow(7.05,3.55,7.7,3.25)
    arrow(7.05,2.25,7.7,2.85)
    arrow(4.25,2.35,5.25,0.75,None, "--")
    ax.text(5.9,4.55,"Supervised heteroscedastic NLL on source patients",ha="center",fontsize=9,fontweight="bold",color="#111827")
    ax.text(5.9,.02,"Target adaptation batch is unlabeled; evaluation batch is disjoint and receives no gradient",ha="center",fontsize=7.2,color="#4b5563")
    fig.savefig(OUT/"figure1_mechanism.pdf",bbox_inches=None); fig.savefig(OUT/"figure1_mechanism.svg",bbox_inches=None); plt.close(fig)

def fig_main():
    d = pd.read_csv(ROOT/"results/confirmatory_gpu/metrics.csv")
    d = d[d.split.eq("external_holdout")].copy(); d["model"]="Assay-aware"
    h = pd.read_csv(ROOT/"results/confirmatory_gpu/cpu_comparison.csv")
    fig = plt.figure(figsize=(7.2,5.4), constrained_layout=True)
    gs = GridSpec(2,2, figure=fig, hspace=.35, wspace=.28)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]
    metrics = [("rmse","RMSE (log1p WMB)"),("mae","MAE (log1p WMB)"),("spearman","Spearman ρ"),("r2","R²")]
    dirs=["TCGA_to_CPTAC","CPTAC_to_TCGA"]; labels=["TCGA→CPTAC","CPTAC→TCGA"]
    for a,(m,lab) in zip(ax,metrics):
        x=np.arange(2); w=.34
        for j,(model,col,label) in enumerate([("Assay-aware",BLUE,"Assay-aware"),("hgb",ORANGE,"PCA-HGB"),("coral",TEAL,"CORAL-PCA-HGB")]):
            vals=[]; errs=[]
            for dr in dirs:
                if model=="Assay-aware": q=d[d.direction==dr][m]
                elif model=="hgb": q=h[h.direction.eq(dr)&h.model.eq("hgb")][m]
                else: q=pd.read_csv(ROOT/"results/coral_cpu/metrics.csv").query("direction==@dr")[m]
                vals.append(q.mean()); errs.append(q.std())
            a.bar(x+(j-1)*w, vals, w, yerr=errs, capsize=2, color=col, label=label, edgecolor="white", linewidth=.5)
        a.set_xticks(x,labels); a.set_ylabel(lab); a.grid(axis="y",alpha=.2); a.set_title(lab)
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05,1.28), ncol=2)
    fig.suptitle("Confirmatory held-out performance across fixed seeds (mean ± SD)", fontsize=11, fontweight="bold")
    save(fig,"figure2_main",ax)

def fig_calibration():
    p=pd.read_csv(ROOT/"results/confirmatory_gpu/external_predictions.csv")
    # Collapse the repeated seed predictions to one descriptive estimate per
    # held-out case before plotting calibration or error strata.
    p=p.groupby(["direction","case_id","lineage"],as_index=False).agg(
        true=("true","first"), pred=("pred","mean"), scale=("scale","mean"))
    fig=plt.figure(figsize=(7.2,5.4), constrained_layout=True); gs=GridSpec(2,2,figure=fig,hspace=.35,wspace=.28)
    ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    for a,(direction,label) in zip(ax[:2],[("TCGA_to_CPTAC","TCGA→CPTAC"),("CPTAC_to_TCGA","CPTAC→TCGA")]):
        q=p[p.direction.eq(direction)].copy(); q["bin"]=pd.qcut(q["true"],4,duplicates="drop")
        g=q.groupby("bin",observed=True).agg(y=("true","mean"),pred=("pred","mean"),scale=("scale","mean"),n=("true","size")).reset_index()
        a.errorbar(g.y,g.pred,yerr=1.64485*g.scale,fmt="o",color=BLUE,capsize=2)
        lo=min(g.y.min(),g.pred.min()); hi=max(g.y.max(),g.pred.max()); a.plot([lo,hi],[lo,hi],"--",color=GREY,lw=1)
        a.set(xlabel="Observed log1p WMB (bin mean)",ylabel="Predicted log1p WMB",title=f"{label}: calibration bins"); a.grid(alpha=.2)
    for a,(direction,label) in zip(ax[2:],[("TCGA_to_CPTAC","TCGA→CPTAC"),("CPTAC_to_TCGA","CPTAC→TCGA")]):
        q=p[p.direction.eq(direction)].copy(); q["abs_err"]=(q.true-q.pred).abs(); q["q"]=pd.qcut(q.true,4,labels=["Q1","Q2","Q3","Q4"])
        g=q.groupby("q",observed=False).agg(mae=("abs_err","mean"),coverage=("abs_err",lambda x: np.nan)).reset_index()
        # coverage is evaluated directly with each row's scale
        cov=q.groupby("q",observed=False).apply(lambda z: np.mean(np.abs(z.true-z.pred)<=1.64485*z.scale),include_groups=False)
        a2=a.twinx(); a.bar(g.q,g.mae,color=ORANGE,alpha=.85,label="MAE"); a2.plot(g.q,cov.values,"o-",color=TEAL,lw=1.8,label="90% coverage")
        a.set(xlabel="Observed WMB quartile",ylabel="MAE",title=f"{label}: error and coverage"); a2.set_ylabel("90% interval coverage"); a2.set_ylim(0,1); a.grid(axis="y",alpha=.2)
    fig.suptitle("Reliability depends on cohort direction and WMB stratum",fontsize=11,fontweight="bold")
    save(fig,"figure4_calibration",ax)

def fig_attribution():
    fig=plt.figure(figsize=(7.2,5.4), constrained_layout=True); gs=GridSpec(2,2,figure=fig,hspace=.38,wspace=.34)
    ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    for a,(direction,label) in zip(ax[:2],[("TCGA_to_CPTAC","TCGA→CPTAC"),("CPTAC_to_TCGA","CPTAC→TCGA")]):
        d=pd.read_csv(ROOT/f"results/attribution/{direction}_gene_importance.csv").head(10).sort_values("mean_abs_attribution")
        gene_col="gene_symbol" if "gene_symbol" in d.columns else "gene_id"
        a.barh(d[gene_col],d.mean_abs_attribution,color=BLUE); a.set_xlabel("Mean |gradient × input|"); a.set_title(label); a.grid(axis="x",alpha=.2)
    sm=pd.read_json(ROOT/"results/attribution/summary.json",orient="index") if False else None
    # Directional rank concordance is computed from saved tables, retaining only shared genes.
    tabs=[]
    for i,x in enumerate(['TCGA_to_CPTAC','CPTAC_to_TCGA']):
        z=pd.read_csv(ROOT/f"results/attribution/{x}_gene_importance.csv")
        gene_col="gene_symbol" if "gene_symbol" in z.columns else "gene_id"
        tabs.append(z[[gene_col,'rank']].rename(columns={gene_col:'gene', 'rank':f'rank_{i}'}))
    a,b=tabs
    m=a.merge(b,on='gene'); ax[2].scatter(m.rank_0,m.rank_1,s=8,alpha=.35,color=TEAL); ax[2].set(xlabel="TCGA→CPTAC rank",ylabel="CPTAC→TCGA rank",title=f"Gene-rank concordance (n={len(m):,})"); ax[2].grid(alpha=.2)
    strat=pd.read_csv(ROOT/"results/analysis/error_uncertainty_bins.csv"); strat=strat[strat.model.eq("assay_aware_refit")]
    labels_added=[]
    for direction,color,label in [("TCGA_to_CPTAC",BLUE,"TCGA→CPTAC"),("CPTAC_to_TCGA",ORANGE,"CPTAC→TCGA")]:
        q=strat[(strat.direction.eq(direction)) & (strat.stratum.eq("true_q"))].copy(); q=q.sort_values("level"); ax[3].plot(np.arange(len(q)),q.mae,"o-",color=color)
        # Direction is encoded by the fixed blue/orange palette and described in the caption.
    ax[3].set_xticks(np.arange(4),["Q1","Q2","Q3","Q4"]); ax[3].set(xlabel="Observed WMB quartile",ylabel="MAE",title="Error stratification",xlim=(-.15,3.75)); ax[3].grid(alpha=.2)
    fig.suptitle("Attribution is assay-specific and errors concentrate in WMB extremes",fontsize=11,fontweight="bold")
    save(fig,"figure5_attribution",ax)

def fig_ablation():
    d=pd.read_csv(ROOT/"results/ablation/metrics.csv")
    g=d.groupby(["evaluation","direction","variant"])[["rmse","mae","spearman","r2"]].mean().reset_index()
    fig=plt.figure(figsize=(7.2,5.4), constrained_layout=True); gs=GridSpec(2,2,figure=fig,hspace=.35,wspace=.30); ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    variants=["hetero_domain","hetero_no_domain","homo_domain","mlp_no_domain"]
    short=["full","no-align","homo","MLP"]
    dirs=["TCGA_to_CPTAC","CPTAC_to_TCGA"]
    dir_labels=["TCGA→CPTAC","CPTAC→TCGA"]
    colors=[BLUE,ORANGE]
    for a,(metric,label) in zip(ax,[("rmse","RMSE"),("mae","MAE"),("spearman","Spearman ρ"),("r2","R²")]):
        clean=g[g.evaluation.eq("clean")].pivot(index="variant",columns="direction",values=metric).loc[variants]
        x=np.arange(len(variants)); w=.34
        for j,(dr,col,dl) in enumerate(zip(dirs,colors,dir_labels)):
            a.bar(x+(j-.5)*w,clean[dr].to_numpy(),w,color=col,edgecolor="white",label=dl)
            # Evaluation-only perturbations are shown as points at the same
            # bar centers, making their small shifts visible without hiding
            # the primary clean comparison.
            for ev,marker in [("mask10","^"),("noise01","v")]:
                vals=g[(g.evaluation==ev)&(g.direction==dr)].set_index("variant").loc[variants,metric].to_numpy()
                a.plot(x+(j-.5)*w,vals,marker=marker,ms=3.2,mec="white",mew=.35,
                       linestyle="none",color="#222222",alpha=.9,
                       label=f"{ev} ({dl})" if (metric=="rmse" and j==0) else None)
        a.set_xticks(x,short); a.set_ylabel(label); a.set_xlabel(""); a.tick_params(axis="x",labelsize=7); a.grid(axis="y",alpha=.2)
        if metric=="rmse":
            a.legend(frameon=False,fontsize=6,ncol=2,loc="upper left")
    fig.suptitle("Ablation on clean target cases with evaluation-only perturbations",fontsize=11,fontweight="bold")
    save(fig,"figure3_ablation",ax)

if __name__=="__main__":
    fig_mechanism(); fig_main(); fig_calibration(); fig_attribution(); fig_ablation()
