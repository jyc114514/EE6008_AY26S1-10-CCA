from __future__ import annotations
import os
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
QUERY = PROJECT / "reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
INPUTS = V3 / "data/V3_MAIN_INPUTS.npz"
COLORS = {"Q0_TS11-D":"#0072B2","Q1_TS11-D":"#56B4E9","Q0_STATE-CTRL":"#009E73","Q1_STATE-CTRL":"#E69F00","Ridge-CPU":"#D55E00","Persistence":"#333333"}
MARKERS = {"Q0_TS11-D":"o","Q1_TS11-D":"s","Q0_STATE-CTRL":"^","Q1_STATE-CTRL":"D","Ridge-CPU":"P","Persistence":"X"}
SHORT = {"Q0_TS11-D":"Q0 TS11-D","Q1_TS11-D":"Q1 TS11-D","Q0_STATE-CTRL":"Q0 STATE","Q1_STATE-CTRL":"Q1 STATE","Ridge-CPU":"Ridge CPU","Persistence":"Persistence"}

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def read_csv(p):
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))

def write_csv(p,fields,rows):
 if p.exists(): raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="raise");w.writeheader();w.writerows(rows)

def save_figure(fig,stem):
 pp=ROOT/"figures"/(stem+".png"); pf=ROOT/"figures"/(stem+".pdf")
 if pp.exists() or pf.exists(): raise FileExistsError(stem)
 fig.savefig(pp,dpi=180,bbox_inches=None,facecolor="white")
 fig.savefig(pf,bbox_inches=None,facecolor="white")
 plt.close(fig)

def fig_query_gains():
 rows=read_csv(ROOT/"figure_data/QUERY_GAIN_POINTS.csv")
 fig,axes=plt.subplots(1,2,figsize=(11.6,4.4),layout="constrained")
 for ax,metric,title in zip(axes,("position_standardized_mae","safe_fd_normalized"),
    ("H30 position gain","H30 safe-FD gain")):
  for family,color,marker,shift in (("TS11-D","#0072B2","o",-0.07),("STATE-CTRL","#009E73","s",0.07)):
   data=[r for r in rows if r["metric"]==metric and r["family"]==family]
   vals=[next(r for r in data if int(r["seed"])==seed) for seed in (20261001,20261002,20261003)]
   x=np.arange(3)+shift; y=[float(r["recomputed_old_minus_had"]) for r in vals]
   ax.scatter(x,y,color=color,marker=marker,s=60,label=family,zorder=3)
   for xi,yi,seed in zip(x,y,(20261001,20261002,20261003)):
    ax.annotate(str(seed)[-2:],(xi,yi),xytext=(0,6),textcoords="offset points",ha="center",fontsize=8)
  ax.axhline(0,color="#555555",linestyle="--",linewidth=1)
  ax.set_xticks(np.arange(3),["seed 01","seed 02","seed 03"])
  ax.set_title(title);ax.set_ylabel("old error − Hadamard error; positive favors Hadamard")
  ax.grid(axis="y",alpha=.22)
 axes[0].legend(frameon=False)
 fig.suptitle("Previously exposed H30 query contrast by fixed seed; no p-values")
 save_figure(fig,"FIG01_H30_QUERY_GAINS_BY_SEED")

def fig_horizon(split):
 rows=[r for r in read_csv(ROOT/"figure_data/HORIZON_MEAN_BY_SYSTEM.csv") if r["split"]==split]
 systems=list(COLORS)
 panels=[
  ("position_prefix_skill_vs_persistence_mean","position_prefix_skill_vs_persistence_seed_sample_sd","Prefix position skill vs Persistence","1 − Lmodel/Lpersistence"),
  ("future_internal_fd_safe_skill_vs_persistence_mean","future_internal_fd_safe_skill_vs_persistence_seed_sample_sd","Future-internal safe-FD skill","1 − Lmodel/Lpersistence"),
  ("endpoint_position_standardized_mae_mean","endpoint_position_standardized_mae_seed_sample_sd","Single-step endpoint position error","standardized MAE at H"),
 ]
 fig,axes=plt.subplots(1,3,figsize=(15.4,4.8),layout="constrained")
 for ax,(metric,sdmetric,title,ylabel) in zip(axes,panels):
  for system in systems:
   data=sorted([r for r in rows if r["system"]==system],key=lambda r:int(r["horizon"]))
   y=np.array([float(r[metric]) for r in data]); x=np.array([float(r["actual_target_delta_ms_mean"]) for r in data])
   finite=np.isfinite(y)
   if not finite.any():continue
   if int(data[0]["seed_count"])==3:
    err=np.array([float(r[sdmetric]) for r in data])[finite]
    ax.errorbar(x[finite],y[finite],yerr=err,color=COLORS[system],marker=MARKERS[system],markersize=4.5,capsize=2,linewidth=1.4,label=SHORT[system])
   else:
    ax.plot(x[finite],y[finite],color=COLORS[system],marker=MARKERS[system],markersize=4.5,linewidth=1.5,label=SHORT[system])
  if "skill" in metric:ax.axhline(0,color="#555555",linestyle="--",linewidth=1)
  ax.set_title(title);ax.set_xlabel("Actual target time − anchor (ms)");ax.set_ylabel(ylabel);ax.grid(alpha=.22)
 axes[0].legend(frameon=False,fontsize=8)
 fig.suptitle(split.capitalize()+" horizon profile; seed error bars are sample SD, not CI")
 save_figure(fig,"FIG02_TRAIN_HORIZON_PROFILE" if split=="train" else "FIG03_DEV_HORIZON_PROFILE")

def fig_h30():
 rows=read_csv(ROOT/"figure_data/HORIZON_MEAN_BY_SYSTEM.csv")
 systems=list(COLORS)
 panels=[
  ("prefix_position_standardized_mae_mean","prefix_position_standardized_mae_seed_sample_sd","Position","standardized prefix MAE"),
  ("future_internal_fd_safe_normalized_mean","future_internal_fd_safe_normalized_seed_sample_sd","Dynamic error","safe-FD normalized MAE"),
  ("tv_ratio_aggregate_mean","tv_ratio_aggregate_seed_sample_sd","Movement amplitude","episode-equal TVpred / TVGT"),
  ("balanced_direction_accuracy_episode_equal_mean","balanced_direction_accuracy_episode_equal_seed_sample_sd","Direction","balanced direction accuracy"),
 ]
 fig,axes=plt.subplots(2,4,figsize=(16.4,7.6),layout="constrained")
 for i,split in enumerate(("train","dev")):
  for j,(metric,sdmetric,title,ylabel) in enumerate(panels):
   ax=axes[i,j]
   for k,system in enumerate(systems):
    rs=[r for r in rows if r["split"]==split and r["system"]==system and int(r["horizon"])==30]
    if not rs:continue
    r=rs[0];value=float(r[metric])
    if not np.isfinite(value):continue
    if int(r["seed_count"])==3 and np.isfinite(float(r[sdmetric])):
     ax.errorbar(k,value,yerr=float(r[sdmetric]),color=COLORS[system],marker=MARKERS[system],capsize=3,markersize=6,linestyle="none")
    else:ax.scatter(k,value,color=COLORS[system],marker=MARKERS[system],s=55)
   ax.set_xticks(range(len(systems)),[SHORT[s] for s in systems],rotation=35,ha="right")
   ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.22)
   if metric=="tv_ratio_aggregate_mean":ax.axhline(1,color="#555555",linestyle="--",linewidth=1)
   if metric=="balanced_direction_accuracy_episode_equal_mean":ax.set_ylim(0,1)
  axes[i,0].text(-.22,.5,split.upper(),transform=axes[i,0].transAxes,rotation=90,va="center",ha="center",fontweight="bold")
 handles=[plt.Line2D([0],[0],marker=MARKERS[s],color=COLORS[s],label=SHORT[s],linestyle="none") for s in systems]
 fig.legend(handles=handles,loc="lower center",ncol=6,frameon=False,bbox_to_anchor=(.5,-.02))
 fig.suptitle("H30 position, dynamic error, movement amplitude and direction; dev was previously exposed")
 save_figure(fig,"FIG04_H30_FOUR_METRICS_TRAIN_AND_DEV")



def build_case_paths():
 cases=read_csv(ROOT/"figure_data/LOCKED_CASE_IDS.csv")
 case_ids=[r["anchor_id"] for r in cases]
 if len(case_ids)!=12:raise AssertionError("expected 12 old locked cases")
 with np.load(INPUTS,allow_pickle=False) as z:
  ids=z["anchor_id"].astype("U64",copy=True)
  split=z["split_code"].astype(np.int8,copy=True)
  anchors_all=z["anchor_command_native"].astype(np.float32,copy=True)
  archive_targets=z["targets_native"]
  index={str(a):i for i,a in enumerate(ids.tolist())}
  ix=np.asarray([index[a] for a in case_ids],dtype=np.int64)
  if not np.all(split[ix]==1):raise AssertionError("locked qualitative cases are not dev")
  anchors=anchors_all[ix].astype(np.float64)
  targets=archive_targets[ix,:30,:].astype(np.float32)
 norms=json.loads((V3/"data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
 sigma=np.asarray(norms["target_sigma"],dtype=np.float64)
 systems={
  "Ground truth":targets.astype(np.float64),
  "Persistence":np.repeat(anchors[:,None,:],30,axis=1).astype(np.float64),
 }
 ridge_path=ROOT/"predictions/dev/RIDGE_FULL_DEV.npz"
 with np.load(ridge_path,allow_pickle=False) as z:
  rid=z["anchor_id"].astype("U64",copy=True);rp=z["predictions_native"].astype(np.float32,copy=True)
 ridx={str(a):i for i,a in enumerate(rid.tolist())}
 systems["Ridge-CPU"]=rp[np.asarray([ridx[a] for a in case_ids],dtype=np.int64)].astype(np.float64)
 refs=json.loads((ROOT/"protocol/REFERENCE_MODELS.json").read_text(encoding="utf-8"))["bundles"]
 for rec in refs:
  if rec["split"]!="dev":continue
  path=Path(rec["prediction_path"])
  if sha(path)!=rec["prediction_sha256"]:raise AssertionError("accepted dev prediction hash changed")
  with np.load(path,allow_pickle=False) as z:
   old_ids=z["anchor_id"].astype("U64",copy=True)
   old=z["predictions_native"].astype(np.float32,copy=True)
  oi={str(a):i for i,a in enumerate(old_ids.tolist())}
  ci=np.asarray([oi[a] for a in case_ids],dtype=np.int64)
  tag="%02d"%(int(rec["seed"])%100)
  label=str(rec["query"])+"_"+str(rec["system"])+"_S"+tag
  systems[label]=old[ci].astype(np.float64)
 order=["Ground truth","Persistence","Ridge-CPU",
  "Q0_TS11-D_S01","Q0_TS11-D_S02","Q0_TS11-D_S03",
  "Q1_TS11-D_S01","Q1_TS11-D_S02","Q1_TS11-D_S03",
  "Q0_STATE-CTRL_S01","Q0_STATE-CTRL_S02","Q0_STATE-CTRL_S03",
  "Q1_STATE-CTRL_S01","Q1_STATE-CTRL_S02","Q1_STATE-CTRL_S03"]
 if set(order)!=set(systems):raise AssertionError("fixed case model list mismatch")
 resid={}
 global_limit=0.0
 for name in order:
  a=(systems[name]-anchors[:,None,:])/sigma[None,None,:]
  resid[name]=a
  global_limit=max(global_limit,float(np.max(np.abs(a))))
 out=ROOT/"figure_data/CASE_PATHS_H30_STANDARDIZED.csv"
 if out.exists():raise FileExistsError(out)
 with out.open("w",encoding="utf-8",newline="") as f:
  fields=["case_id","stratum","source_group_id","anchor_id","system","step","action_dimension",
          "anchor_native","target_native","prediction_native","target_residual_standardized",
          "prediction_residual_standardized","final_test_access"]
  wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
  for ci,case in enumerate(cases):
   for name in order:
    for step in range(30):
     for dim in range(28):
      wr.writerow({"case_id":case["case_id"],"stratum":case["stratum"],"source_group_id":case["source_group_id"],
       "anchor_id":case["anchor_id"],"system":name,"step":step+1,"action_dimension":dim,
       "anchor_native":float(anchors[ci,dim]),"target_native":float(targets[ci,step,dim]),
       "prediction_native":float(systems[name][ci,step,dim]),
       "target_residual_standardized":float((targets[ci,step,dim]-anchors[ci,dim])/sigma[dim]),
       "prediction_residual_standardized":float(resid[name][ci,step,dim]),"final_test_access":0})
 return cases,systems,resid,order,global_limit


def case_figures(cases,resid,order,limit):
 pdf=ROOT/"figures/FIG05_FIXED_H30_CASE_PATHS.pdf"
 if pdf.exists():raise FileExistsError(pdf)
 with PdfPages(pdf,metadata={"Title":"Fixed H30 cases across accepted models and CPU Ridge",
   "Subject":"Prelocked development cases; channel residuals standardized by primary-train target sigma"}) as pages:
  for ci,case in enumerate(cases):
   fig,axes=plt.subplots(len(order),1,figsize=(11.7,14.0),sharex=True,layout="constrained")
   for ax,name in zip(axes,order):
    im=ax.imshow(resid[name][ci].T,origin="lower",aspect="auto",interpolation="nearest",
       cmap="RdBu_r",vmin=-limit,vmax=limit)
    ax.set_ylabel(name.replace("_","\\n"),fontsize=7)
    ax.set_yticks([0,13,27],labels=["0","13","27"])
    ax.tick_params(axis="y",labelsize=6,length=2)
   axes[-1].set_xticks([0,14,29],labels=["1","15","30"])
   axes[-1].set_xlabel("Future step after anchor (H30)")
   fig.colorbar(im,ax=list(axes),location="right",shrink=.68,
       label="(prediction − anchor) / primary-train target sigma")
   fig.suptitle("%s | %s | fixed prior dev case | source-native action[0:28]"%(case["case_id"],case["stratum"]))
   pages.savefig(fig,facecolor="white")
   png=ROOT/"figures"/("FIG05_CASE_"+case["case_id"]+".png")
   if png.exists():raise FileExistsError(png)
   fig.savefig(png,dpi=150,bbox_inches=None,facecolor="white")
   plt.close(fig)
 return pdf


def main():
 for p in [ROOT/"qa/HORIZON_ANALYSIS_ACCEPTANCE.json",ROOT/"results/HORIZON_SCORE_SUMMARY.json",
           ROOT/"figure_data/HORIZON_MEAN_BY_SYSTEM.csv",ROOT/"figure_data/LOCKED_CASE_IDS.csv"]:
  if not p.exists():raise FileNotFoundError(p)
 with plt.rc_context({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,
    "legend.fontsize":7,"figure.titlesize":11,"pdf.fonttype":42,"ps.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False}):
  fig_query_gains()
  fig_horizon("train")
  fig_horizon("dev")
  fig_h30()
  cases,systems,resid,order,limit=build_case_paths()
  case_pdf=case_figures(cases,resid,order,limit)
 access={"status":"PASS_FIXED_CASE_H30_PATHS_RENDERED_AFTER_MODEL_LOCK",
  "case_count":len(cases),"case_source":str(QUERY/"qualitative/CASE_MANIFEST.csv"),
  "case_manifest_sha256":sha(QUERY/"qualitative/CASE_MANIFEST.csv"),
  "all_cases_prelocked_before_new_ridge_scores":True,"ridge_score_used_for_case_selection":False,
  "target_slice_retained":"12 fixed dev anchors x H30 x Full28; only the first 30 target steps were used",
  "underlying_archive_stores_60_steps":True,"steps_31_to_60_used":False,
  "final_test_access":0,"heldout_access":0}
 access_path=ROOT/"protocol/FIGURE_TARGET_ACCESS.json"
 if access_path.exists():raise FileExistsError(access_path)
 access_path.write_text(json.dumps(access,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
 caption="""# Figure captions and alt text

## Figure 1 — Previously exposed H30 query gains by seed
Points are accepted development error under old query minus error under Hadamard. Positive values favor Hadamard. All fixed seeds are shown and no p-values are reported. Alt text: TS11-D improvement is concentrated in seed 02; seeds 01 and 03 do not improve. STATE-CTRL position gains are negative for all three seeds.

## Figures 2 and 3 — Train and dev horizon profiles
Each curve slices the same H30 prediction at H=1,3,6,10,15,30; no short-horizon model was trained. The x-axis shows raw target timestamp minus anchor, averaged across anchors. Panels show prefix position skill versus Persistence, internal first-difference skill versus Persistence, and endpoint position error at H. Error bars for the four accepted neural systems are sample SD across the three fixed seeds, not CIs. Ridge and Persistence are single-fit points. Train Ridge is in-sample. Dev was previously exposed. H=1 internal FD is NA. Alt text: Ridge shows no positive position or dynamic skill across the dev horizons; state baselines retain positive position skill.

## Figure 4 — H30 position, dynamic error, movement amplitude, and direction
Rows separate train and previously exposed dev. Columns show prefix Full28 position MAE, future-internal safe-FD error, aggregate TVpred/TVGT, and balanced direction accuracy. Seed error bars are sample SD; Ridge and Persistence have no seed SD. TV=1 is magnitude parity and does not establish correct motion. Alt text: Ridge predicts more movement amplitude than the accepted neural systems but has worse position and FD errors than Persistence and STATE-CTRL on dev.

## Figure 5 — Twelve fixed H30 cases
The multipage PDF has one page per case from the old low/high activity lock. Rows show GT, Persistence, Ridge, and each accepted query/system/seed separately; no prediction ensemble is formed. Heatmaps show (value minus anchor) divided by primary-train target sigma for all 28 source-native command dimensions. The CSV preserves raw native command values. Physical units and anatomical mappings remain unresolved. Alt text: Every page compares full 28-dimensional future paths over 30 steps for one preselected case.
"""
 cap=ROOT/"captions/FIGURE_CAPTIONS_AND_ALT.zh-CN.md"
 if cap.exists():raise FileExistsError(cap)
 cap.write_text(caption,encoding="utf-8")
 pngs=sorted((ROOT/"figures").glob("*.png"));pdfs=sorted((ROOT/"figures").glob("*.pdf"))
 manifest={"status":"PASS_GENERAL_SCIENTIFIC_FIGURE_EXPORT","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "png_count":len(pngs),"pdf_count":len(pdfs),"case_count":len(cases),"case_pdf_pages":len(cases),
  "case_global_color_limit_standardized":limit,"target_journal":"unspecified; publisher compliance not assessed",
  "human_review_status":"pending","submission_ready":False,
  "files":[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in pngs+pdfs]}
 mp=ROOT/"figures/FIGURE_MANIFEST.json"
 if mp.exists():raise FileExistsError(mp)
 mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
 print(json.dumps({"status":manifest["status"],"png_count":len(pngs),"pdf_count":len(pdfs),"case_count":len(cases)},sort_keys=True))


if __name__=="__main__":
 main()

