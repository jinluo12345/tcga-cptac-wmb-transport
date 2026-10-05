from pathlib import Path
import argparse, gzip, json, math
import pandas as pd
QUAL={"Frame_Shift_Del","Frame_Shift_Ins","In_Frame_Del","In_Frame_Ins","Missense_Mutation","Nonsense_Mutation","Nonstop_Mutation","Splice_Site","Translation_Start_Site","Multi_Hit"}
def parse(path):
    variants=set(); header=None
    with gzip.open(path,"rt",errors="replace") as f:
        for line in f:
            if line.startswith("#"): continue
            z=line.rstrip("\n").split("\t")
            if header is None: header={x:i for i,x in enumerate(z)}; continue
            vc=z[header["Variant_Classification"]] if "Variant_Classification" in header else ""
            if vc not in QUAL: continue
            vals=[]
            for k in ("Chromosome","Start_Position","End_Position","Reference_Allele","Tumor_Seq_Allele2"):
                i=header.get(k,-1); vals.append(z[i] if i>=0 and i<len(z) else "")
            if all(vals): variants.add(tuple(vals))
    return variants
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",default="metadata/wxs_maf_manifest.json"); ap.add_argument("--maf-root",required=True); ap.add_argument("--expression-manifest",default="data/tcga_cptac_manifest.json"); ap.add_argument("--out",default="data/tmb_labels.csv"); a=ap.parse_args()
    rows=[]
    for r in json.loads(Path(a.manifest).read_text()):
        p=Path(a.maf_root)/(r["file_id"]+".maf.gz")
        if not p.exists(): continue
        rows.append((r,parse(p)))
    by={}
    for r,vs in rows: by.setdefault(r["case_id"],set()).update(vs)
    expr=json.loads(Path(a.expression_manifest).read_text()); out=[]
    for r in expr:
        if r["case_id"] in by:
            n=len(by[r["case_id"]]); out.append({"case_id":r["case_id"],"cohort":r["cohort"],"label":r.get("label"),"file_id":r.get("file_id"),"tmb_count":n,"log1p_tmb":math.log1p(n)})
    pd.DataFrame(out).to_csv(a.out,index=False); print(f"wrote {len(out)} cases to {a.out}")
if __name__=="__main__": main()
