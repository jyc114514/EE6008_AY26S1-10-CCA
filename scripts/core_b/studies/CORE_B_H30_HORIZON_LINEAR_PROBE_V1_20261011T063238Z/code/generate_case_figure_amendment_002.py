from __future__ import annotations
import csv,hashlib,json,datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIELDS={}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(path):
 with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 cases=read(ROOT/"figure_data/LOCKED_CASE_IDS.csv")
 if len(cases)!=12:raise AssertionError("locked case count mismatch")
 systems=["Ground truth","Persistence","Ridge-CPU",
  "Q0_TS11-D_S01","Q0_TS11-D_S02","Q0_TS11-D_S03",
  "Q1_TS11-D_S01","Q1_TS11-D_S02","Q1_TS11-D_S03",
  "Q0_STATE-CTRL_S01","Q0_STATE-CTRL_S02","Q0_STATE-CTRL_S03",
  "Q1_STATE-CTRL_S01","Q1_STATE-CTRL_S02","Q1_STATE-CTRL_S03"]
 labels={"Ground truth":"GT","Persistence":"Persistence","Ridge-CPU":"Ridge CPU",
  "Q0_TS11-D_S01":"Q0 TS11-D S01","Q0_TS11-D_S02":"Q0 TS11-D S02","Q0_TS11-D_S03":"Q0 TS11-D S03",
  "Q1_TS11-D_S01":"Q1 TS11-D S01","Q1_TS11-D_S02":"Q1 TS11-D S02","Q1_TS11-D_S03":"Q1 TS11-D S03",
  "Q0_STATE-CTRL_S01":"Q0 State S01","Q0_STATE-CTRL_S02":"Q0 State S02","Q0_STATE-CTRL_S03":"Q0 State S03",
  "Q1_STATE-CTRL_S01":"Q1 State S01","Q1_STATE-CTRL_S02":"Q1 State S02","Q1_STATE-CTRL_S03":"Q1 State S03"}
 csv_path=ROOT/"figure_data/CASE_PATHS_H30_STANDARDIZED.csv"
 data={(c["case_id"],s):np.full((30,28),np.nan,dtype=np.float32) for c in cases for s in systems}
 with csv_path.open(encoding="utf-8",newline="") as f:
  for r in csv.DictReader(f):
   data[(r["case_id"],r["system"])][int(r["step"])-1,int(r["action_dimension"])]=float(r["prediction_residual_standardized"])
 if any(not np.isfinite(a).all() for a in data.values()):raise AssertionError("case heatmap data has missing cells")
 limit=max(1.0,max(float(np.max(np.abs(a))) for a in data.values()))
 pdf_path=ROOT/"figures/FIG05_FIXED_H30_CASE_PATHS_AMEND002.pdf"
 if pdf_path.exists():raise FileExistsError(pdf_path)
 png_paths=[]
 with PdfPages(pdf_path,metadata={"Title":"Fixed H30 cases with corrected compact row labels",
      "Subject":"Prelocked dev cases; standardized by primary-train target sigma"}) as pages:
  for c in cases:
   fig,axes=plt.subplots(15,1,figsize=(11.7,14.0),sharex=True,layout="constrained")
   for ax,system in zip(axes,systems):
    im=ax.imshow(data[(c["case_id"],system)].T,origin="lower",aspect="auto",interpolation="nearest",
       cmap="RdBu_r",vmin=-limit,vmax=limit)
    ax.set_ylabel(labels[system],rotation=0,ha="right",va="center",labelpad=8,fontsize=6.5)
    ax.set_yticks([0,13,27],labels=["0","13","27"])
    ax.tick_params(axis="y",labelsize=6,length=2)
   axes[-1].set_xticks([0,14,29],labels=["1","15","30"])
   axes[-1].set_xlabel("Future step after anchor (H30)")
   fig.colorbar(im,ax=list(axes),location="right",shrink=.68,
       label="(prediction − anchor) / primary-train target sigma")
   fig.suptitle("%s | %s | fixed prior dev case | action[0:28]"%(c["case_id"],c["stratum"]))
   pages.savefig(fig,facecolor="white")
   png=ROOT/"figures"/("FIG05_CASE_"+c["case_id"]+"_AMEND002.png")
   if png.exists():raise FileExistsError(png)
   fig.savefig(png,dpi=150,bbox_inches=None,facecolor="white")
   png_paths.append(png)
   plt.close(fig)
 files=[pdf_path]+png_paths
 manifest={"status":"PASS_CASE_FIGURE_LABEL_AMENDMENT_002","supersedes":"FIG05_FIXED_H30_CASE_PATHS.pdf and FIG05_CASE_*.png",
  "reason":"shortened per-system row labels so labels do not collide or clip","case_count":12,"pdf_pages":12,
  "global_color_limit_standardized":limit,"data_source":str(csv_path),"data_source_sha256":sha(csv_path),
  "target_values_read":False,"human_review":"pending","files":[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in files]}
 out=ROOT/"qa/FIGURE_CASE_AMENDMENT_002.json"
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"status":manifest["status"],"case_count":12,"pdf_pages":12,"png_count":12,"color_limit":limit},sort_keys=True))
if __name__=="__main__":main()

