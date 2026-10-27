#!/usr/bin/env python3
"""Generate three source-backed Core A V2 figures and their data files."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/"core_a"
FIG=ROOT/"figures"
FIG.mkdir(parents=True,exist_ok=True)
(FIG/"captions").mkdir(parents=True,exist_ok=True)
(FIG/"figure_data").mkdir(parents=True,exist_ok=True)
DYN="#007F87"
STA="#C8793E"
NAVY="#19324A"
MUTED="#526579"
GRID="#D9E2E7"


def read_csv(path):
 with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields):
 path.parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save(fig,stem):
 FIG.mkdir(parents=True,exist_ok=True)
 fig.savefig(FIG/(stem+".svg"),bbox_inches="tight",metadata={"Title":stem,"Description":"V2 Core A final score, 602 primary rows, corrected train-target class mapping."})
 fig.savefig(FIG/(stem+".png"),dpi=220,bbox_inches="tight",metadata={"Title":stem})
 plt.close(fig)

def main():
 per=read_csv(CORE/"FULL602_PER_SEED.csv")
 summary=read_csv(CORE/"FULL602_SUMMARY.csv")
 contrast=read_csv(CORE/"PAIRED_CONTRASTS_AND_CI.csv")
 detail=read_csv(CORE/"CONFUSION_AND_PER_CLASS.csv")
 boot=json.loads((CORE/"BOOTSTRAP_DRAW_MANIFEST.json").read_text(encoding="utf-8"))
 zcount=boot["estimands"]["A-FULL602"]["macro_f1_zero_support_count_by_task"]
 conditions=["A_DYNAMIC","A_STATIC"]
 metrics=[("top1_accuracy","Top-1 accuracy"),("top5_accuracy","Top-5 accuracy"),("macro_f1_true_classes","Macro-F1")]
 colors={"A_DYNAMIC":DYN,"A_STATIC":STA}
 name={"A_DYNAMIC":"Dynamic clip","A_STATIC":"Repeated first frame"}
 # Figure 1: condition means and three fitted-seed points; SD bars are across seeds, not CIs.
 fig,axes=plt.subplots(1,3,figsize=(14,4.9),sharey=True)
 for ax,(metric,title) in zip(axes,metrics):
  for y,program in enumerate(conditions):
   row=next(r for r in summary if r["program"]==program and r["metric"]==metric)
   mean=float(row["mean"])*100;sd=float(row["sample_sd"])*100
   seeds=[float(r[metric])*100 for r in per if r["program"]==program]
   seeds=[float(next(r[metric] for r in per if r["program"]==program and int(r["seed"])==seed))*100 for seed in (20260825,20260826,20260827)]
   ax.errorbar(mean,y,xerr=sd,fmt="o",ms=8,capsize=4,lw=2,color=colors[program],zorder=3)
   jitter=np.linspace(-0.11,0.11,len(seeds))
   ax.scatter(seeds,np.full(3,y)+jitter,s=24,color=colors[program],edgecolor="white",linewidth=0.5,zorder=4)
   if mean+sd>96:
    ax.text(mean-sd-0.4,y-0.14,f"{mean:.2f}%",fontsize=9,color=colors[program],ha="right",va="center")
   else:
    ax.text(mean+sd+0.4,y-0.14,f"{mean:.2f}%",fontsize=9,color=colors[program],ha="left",va="center")
  ax.set_title(title,color=NAVY,fontsize=12,fontweight="bold")
  ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.grid(axis="x",color=GRID,linewidth=.8)
  ax.set_axisbelow(True);ax.spines[["top","right","left"]].set_visible(False)
  ax.set_yticks([0,1],labels=[name[x] for x in conditions]);ax.invert_yaxis()
  ax.tick_params(axis="y",length=0,labelcolor=NAVY);ax.tick_params(axis="x",labelcolor=MUTED)
  ax.set_xlabel("Percent")
 fig.suptitle("Core A corrected final evaluation · A-FULL602",x=.08,y=1.04,ha="left",fontsize=17,color=NAVY,fontweight="bold")
 fig.text(.08,-.02,"Dots are the three fitted seeds; center and whisker show mean ± sample SD across seeds. This is not a confidence interval.",fontsize=9,color=MUTED)
 fig.tight_layout(w_pad=2)
 save(fig,"CORE_A_FINAL_CONDITIONS_V2")

 # Figure 2: paired dynamic-minus-static conditional percentile intervals.
 metric_names={"top1_accuracy":"Top-1 accuracy","top5_accuracy":"Top-5 accuracy","macro_f1_true_classes":"Macro-F1"}
 rows=[]
 for r in contrast:
  rows.append({"metric":r["metric"],"metric_label":metric_names[r["metric"]],"difference_pp":float(r["mean_difference"])*100,
   "ci_low_pp":float(r["conditional_grouped_bootstrap_ci_low"])*100,"ci_high_pp":float(r["conditional_grouped_bootstrap_ci_high"])*100,
   "bootstrap_draws":int(r["bootstrap_draws"]),"bootstrap_seed":int(r["bootstrap_seed"]),"cluster_count":int(r["cluster_count"])})
 write_csv(FIG/"figure_data/CORE_A_PAIRED_CONTRASTS_V2.csv",rows,list(rows[0]))
 fig,ax=plt.subplots(figsize=(9.2,4.5))
 y=np.arange(len(rows))[::-1]
 for yi,r in zip(y,rows):
  color=DYN if r["difference_pp"]>=0 else STA
  ax.errorbar(r["difference_pp"],yi,xerr=[[r["difference_pp"]-r["ci_low_pp"]],[r["ci_high_pp"]-r["difference_pp"]]],fmt="o",ms=8,capsize=5,lw=2,color=color,zorder=3)
  ax.text(r["ci_high_pp"]+0.12,yi,f'{r["difference_pp"]:+.2f} pp',va="center",fontsize=9,color=NAVY)
 ax.axvline(0,color=MUTED,linewidth=1,linestyle="--")
 ax.set_yticks(y,labels=[r["metric_label"] for r in rows]);ax.set_xlim(-4,4.2);ax.set_xlabel("Dynamic − repeated-first-frame (percentage points)")
 ax.set_title("Paired Core A contrasts · 5,000 shared episode draws",loc="left",fontsize=15,color=NAVY,fontweight="bold")
 ax.grid(axis="x",color=GRID,linewidth=.8);ax.set_axisbelow(True);ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0,labelcolor=NAVY);ax.tick_params(axis="x",labelcolor=MUTED)
 fig.text(.14,-.02,"Whiskers: 95% percentile interval conditional on the six fitted checkpoints; no p-values. Positive favors dynamic.",fontsize=9,color=MUTED)
 fig.tight_layout()
 save(fig,"CORE_A_PAIRED_CONTRASTS_V2")

 # Figure 3: why fixed-label macro-F1 bootstrap draws can omit rare classes.
 class_rows=[r for r in detail if r["record_type"]=="per_class" and r["program"]=="A_DYNAMIC" and int(r["seed"])==20260825]
 class_rows.sort(key=lambda x:int(x["true_task_id"]))
 support=np.asarray([int(r["support"]) for r in class_rows],dtype=int)
 zero_pct=np.asarray([int(zcount[str(int(r["true_task_id"]))])/50.0 for r in class_rows],dtype=float)
 support_dist=[{"support":int(k),"class_count":int(np.sum(support==k))} for k in range(1,int(support.max())+1)]
 write_csv(FIG/"figure_data/CORE_A_SUPPORT_DISTRIBUTION_V2.csv",support_dist,["support","class_count"])
 fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8))
 counts=np.bincount(support,minlength=int(support.max())+1)
 xs=np.arange(1,len(counts))
 ax1.bar(xs,counts[1:],color=DYN,width=.82)
 ax1.axvline(4.5,color=STA,linestyle="--",linewidth=1.5)
 ax1.text(4.65,max(counts[1:])*.82,"27 classes have\nsupport < 5",color=STA,fontsize=9)
 ax1.set_xlabel("Observed final rows per class");ax1.set_ylabel("Number of classes");ax1.set_title("Observed support (120 fixed labels)",loc="left",color=NAVY,fontweight="bold")
 ax1.grid(axis="y",color=GRID,linewidth=.8);ax1.set_axisbelow(True)
 zero_sorted=np.sort(zero_pct)
 ax2.plot(np.arange(1,121),zero_sorted,color=STA,linewidth=2)
 ax2.scatter([1,120],[zero_sorted[0],zero_sorted[-1]],color=STA,s=22,zorder=3)
 ax2.axhline(100*13/5000,color=MUTED,linestyle=":",linewidth=1)
 ax2.set_xlabel("Class rank by bootstrap omission rate");ax2.set_ylabel("Draws with zero support (%)");ax2.set_ylim(-1,42)
 ax2.set_title("Fixed-label bootstrap omissions",loc="left",color=NAVY,fontweight="bold")
 ax2.grid(axis="y",color=GRID,linewidth=.8);ax2.set_axisbelow(True)
 fig.suptitle("Macro-F1 keeps all 120 labels; zero support contributes F1 = 0",x=.07,y=1.04,ha="left",fontsize=15,color=NAVY,fontweight="bold")
 fig.text(.07,-.02,"The 5,000 episode-cluster resamples may omit rare classes; 118/120 classes are absent in at least one draw. This drives the conditional percentile interval.",fontsize=9,color=MUTED)
 fig.tight_layout(w_pad=2)
 save(fig,"CORE_A_RARE_CLASS_BOOTSTRAP_V2")

 # Shared source data for all condition means and uncertainty.
 condition_data=[]
 for r in summary:
  condition_data.append({"estimand":r["estimand"],"program":r["program"],"metric":r["metric"],"n_rows":int(r["n_rows"]),"n_seeds":int(r["n_seeds"]),"mean":float(r["mean"]),"sample_sd":float(r["sample_sd"]),"bootstrap_ci_low":float(r["conditional_grouped_bootstrap_ci_low"]),"bootstrap_ci_high":float(r["conditional_grouped_bootstrap_ci_high"]),"bootstrap_draws":int(r["bootstrap_draws"]),"bootstrap_seed":int(r["bootstrap_seed"]),"cluster_count":int(r["cluster_count"])})
 write_csv(FIG/"figure_data/CORE_A_FINAL_CONDITIONS_V2.csv",condition_data,list(condition_data[0]))
 for stem,alt in [
  ("CORE_A_FINAL_CONDITIONS_V2","Three side-by-side dot plots show dynamic and repeated-first-frame Core A means for Top-1, Top-5, and Macro-F1 on 602 final rows; seed points and mean plus sample SD are shown."),
  ("CORE_A_PAIRED_CONTRASTS_V2","A forest plot of dynamic minus static Top-1, Top-5, and Macro-F1 in percentage points, with 95 percent conditional grouped percentile intervals and a zero reference line."),
  ("CORE_A_RARE_CLASS_BOOTSTRAP_V2","Two panels show observed class support counts and sorted bootstrap omission rates for 120 fixed labels; zero support contributes zero to Macro-F1.")]:
  (FIG/"captions"/(stem+".md")).write_text(f"# {stem}\n\n{alt}\n\nData and interval definitions are in the matching figure_data CSV.\n",encoding="utf-8")
 print(json.dumps({"status":"PASS","figures":["CORE_A_FINAL_CONDITIONS_V2","CORE_A_PAIRED_CONTRASTS_V2","CORE_A_RARE_CLASS_BOOTSTRAP_V2"],"output":str(FIG)},ensure_ascii=False))
if __name__=="__main__":main()
