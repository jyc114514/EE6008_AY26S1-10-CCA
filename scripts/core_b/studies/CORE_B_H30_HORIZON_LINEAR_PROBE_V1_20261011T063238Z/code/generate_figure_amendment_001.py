from pathlib import Path
import csv,json,hashlib,datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
COLORS={"Q0_TS11-D":"#0072B2","Q1_TS11-D":"#56B4E9","Q0_STATE-CTRL":"#009E73","Q1_STATE-CTRL":"#E69F00","Ridge-CPU":"#D55E00","Persistence":"#333333"}
MARKERS={"Q0_TS11-D":"o","Q1_TS11-D":"s","Q0_STATE-CTRL":"^","Q1_STATE-CTRL":"D","Ridge-CPU":"P","Persistence":"X"}
SHORT={"Q0_TS11-D":"Q0 TS11-D","Q1_TS11-D":"Q1 TS11-D","Q0_STATE-CTRL":"Q0 STATE","Q1_STATE-CTRL":"Q1 STATE","Ridge-CPU":"Ridge CPU","Persistence":"Persistence"}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(path):
 with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 rows=read(ROOT/"figure_data/HORIZON_MEAN_BY_SYSTEM.csv")
 metrics=[
  ("prefix_position_standardized_mae_mean","prefix_position_standardized_mae_seed_sample_sd","Position","standardized prefix MAE"),
  ("future_internal_fd_safe_normalized_mean","future_internal_fd_safe_normalized_seed_sample_sd","Dynamic error","safe-FD normalized MAE"),
  ("tv_ratio_aggregate_mean","tv_ratio_aggregate_seed_sample_sd","Movement amplitude","episode-equal TVpred / TVGT"),
  ("balanced_direction_accuracy_episode_equal_mean","balanced_direction_accuracy_episode_equal_seed_sample_sd","Direction","balanced direction accuracy"),
 ]
 systems=list(COLORS)
 fig,axes=plt.subplots(2,4,figsize=(16.4,7.6),layout="constrained")
 for ri,split in enumerate(("train","dev")):
  for ci,(metric,sdmetric,title,ylabel) in enumerate(metrics):
   ax=axes[ri,ci]
   for k,system in enumerate(systems):
    rr=[r for r in rows if r["split"]==split and r["system"]==system and int(r["horizon"])==30]
    if not rr:continue
    r=rr[0];y=float(r[metric])
    if not (y==y and abs(y)<1e100):continue
    sd=float(r[sdmetric]) if r[sdmetric] not in ("nan","") else float("nan")
    if int(r["seed_count"])==3 and sd==sd:
     ax.errorbar(k,y,yerr=sd,color=COLORS[system],marker=MARKERS[system],capsize=3,markersize=6,linestyle="none")
    else:
     ax.scatter(k,y,color=COLORS[system],marker=MARKERS[system],s=55)
   ax.set_xticks(range(len(systems)),[SHORT[s] for s in systems],rotation=35,ha="right")
   ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.22)
   if metric=="tv_ratio_aggregate_mean":ax.axhline(1,color="#555555",linestyle="--",linewidth=1)
   if metric=="balanced_direction_accuracy_episode_equal_mean":ax.set_ylim(0,1)
  axes[ri,0].text(-.22,.5,split.upper(),transform=axes[ri,0].transAxes,rotation=90,va="center",ha="center",fontweight="bold")
 fig.suptitle("H30 position, dynamic error, movement amplitude and direction; dev was previously exposed")
 stem="FIG04_H30_FOUR_METRICS_TRAIN_AND_DEV_AMEND001"
 png=ROOT/"figures"/(stem+".png");pdf=ROOT/"figures"/(stem+".pdf")
 if png.exists() or pdf.exists():raise FileExistsError(stem)
 fig.savefig(png,dpi=180,bbox_inches=None,facecolor="white")
 fig.savefig(pdf,bbox_inches=None,facecolor="white")
 plt.close(fig)
 o={"status":"PASS_FIG04_LAYOUT_AMENDMENT_001","supersedes":"FIG04_H30_FOUR_METRICS_TRAIN_AND_DEV.png and .pdf","reason":"the original bottom figure legend was clipped; category labels already identify each marker, so the redundant legend is removed","png":{"path":str(png.relative_to(ROOT)),"sha256":sha(png),"bytes":png.stat().st_size},"pdf":{"path":str(pdf.relative_to(ROOT)),"sha256":sha(pdf),"bytes":pdf.stat().st_size},"data_source":"figure_data/HORIZON_MEAN_BY_SYSTEM.csv","data_source_sha256":sha(ROOT/"figure_data/HORIZON_MEAN_BY_SYSTEM.csv"),"publisher_compliance":"not assessed","human_review":"pending"}
 am=ROOT/"qa/FIGURE_MANIFEST_AMENDMENT_001.json"
 if am.exists():raise FileExistsError(am)
 am.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps(o,sort_keys=True))
if __name__=="__main__":main()

