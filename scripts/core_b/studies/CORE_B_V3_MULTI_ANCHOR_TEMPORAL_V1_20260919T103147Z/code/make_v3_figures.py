#!/usr/bin/env python3
"""Generate the V3 eight-group figure package."""

from __future__ import annotations
import os
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
AN=EXP/"analysis"; FIG=EXP/"figures"; FD=EXP/"figure_data"; FM=EXP/"figure_manifests"; CELLS=("S-HIST","F-POOL-D","F-TOK-D","F-TOK-S","V-TOK-D-A","V-TOK-S-A"); COLORS={"S-HIST":"#6b7280","F-POOL-D":"#2563eb","F-TOK-D":"#b91c1c","F-TOK-S":"#f97316","V-TOK-D-A":"#059669","V-TOK-S-A":"#7c3aed"}

def read(p):
 with p.open(encoding="utf-8",newline="") as h:return list(csv.DictReader(h))
def pair(fig,stem):
 FIG.mkdir(exist_ok=True); fig.savefig(FIG/f"{stem}.png",dpi=200);fig.savefig(FIG/f"{stem}.pdf");plt.close(fig)
def cap(stem,text):
 FM.mkdir(exist_ok=True);(FM/f"{stem}.caption.md").write_text(text+"\n",encoding="utf-8")

def main()->int:
 FIG.mkdir(exist_ok=True);FD.mkdir(exist_ok=True);FM.mkdir(exist_ok=True); figs=[]; agg=read(AN/"AGGREGATE_RESULTS.csv"); paired=read(AN/"PAIRED_EFFECTS_AGGREGATE.csv"); interactions=read(AN/"HORIZON_INTERACTIONS.csv"); resources=read(AN/"RESOURCE_COSTS.csv")
 # 1 coverage
 fig,ax=plt.subplots(figsize=(8,4.8),layout="constrained"); labels=["primary train\nepisodes","primary train\nwindows","primary dev\nepisodes","primary dev\nwindows","supp 1s\nepisodes","supp 1s\nwindows"]; vals=[2712,17757,603,2322,64,64]; ax.bar(np.arange(6),vals,color=["#374151","#4b5563","#2563eb","#60a5fa","#059669","#6ee7b7"]);ax.set_xticks(np.arange(6),labels);ax.set_ylabel("count");ax.set_title("V3 locked anchor/horizon coverage");ax.grid(axis="y",alpha=.25)
 for i,v in enumerate(vals):ax.text(i,v,f"{v:,}",ha="center",va="bottom",fontsize=8)
 pair(fig,"FIG01_ANCHOR_HORIZON_COVERAGE");(FD/"FIG01_ANCHOR_HORIZON_COVERAGE.csv").write_text("population,count\n"+"\n".join(f"{l.replace(chr(10),' ')},{v}" for l,v in zip(labels,vals))+"\n");cap("FIG01_ANCHOR_HORIZON_COVERAGE","The locked primary and supplementary populations. Primary 1s and 2s use the same 2s-legal anchor list; supplementary 1s rows are separate.");figs.append("FIG01_ANCHOR_HORIZON_COVERAGE")
 # 2 metrics by horizon
 fig,axs=plt.subplots(1,3,figsize=(16,5),layout="constrained"); x=np.arange(len(CELLS));
 for ax,key,title in zip(axs,("arm_mae_std","hand_mae_std","full_mae_std"),("Arm-14 standardized MAE","Hand-14 standardized MAE","Full-28 standardized MAE")):
  for j,h in enumerate((30,60)):
   rs=[r for r in agg if int(r["horizon_steps"])==h]; vals=[float(next(r for r in rs if r["cell"]==c)[key+"_mean"]) for c in CELLS]; sd=[float(next(r for r in rs if r["cell"]==c)[key+"_sd"]) for c in CELLS]; ax.errorbar(x+(j-.5)*.12,vals,yerr=sd,fmt="o-",capsize=3,label=f"{h/30:.0f}s")
  ax.set_xticks(x,CELLS,rotation=35,ha="right");ax.set_title(title);ax.grid(axis="y",alpha=.25);ax.legend()
 pair(fig,"FIG02_ARM_HAND_FULL_BY_HORIZON");(FD/"FIG02_ARM_HAND_FULL_BY_HORIZON.csv").write_text("\n".join(",".join(str(r.get(k,"")) for k in ("horizon_steps","cell","arm_mae_std_mean","arm_mae_std_sd","hand_mae_std_mean","hand_mae_std_sd","full_mae_std_mean","full_mae_std_sd")) for r in agg)+"\n");cap("FIG02_ARM_HAND_FULL_BY_HORIZON","Aggregate mean and between-seed SD for Arm, Hand and Full standardized MAE. Error bars describe the three pre-registered training seeds; they are not confidence intervals.");figs.append("FIG02_ARM_HAND_FULL_BY_HORIZON")
 # 3 feature forest
 fig,axs=plt.subplots(1,2,figsize=(13,6),layout="constrained")
 for ax,group,title in zip(axs,("arm_mae_std","hand_mae_std"),("Arm paired effect","Hand paired effect")):
  rs=[r for r in paired if r["group"]==group]; labels=[r["contrast"]+" H"+r["horizon_steps"] for r in rs]; y=np.arange(len(rs)); v=np.array([float(r["effect"]) for r in rs]); lo=np.array([float(r["ci_low"]) for r in rs]); hi=np.array([float(r["ci_high"]) for r in rs]); ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt="o",capsize=3);ax.axvline(0,color="#555",lw=.8);ax.set_yticks(y,labels,fontsize=8);ax.set_title(title);ax.grid(axis="x",alpha=.25)
 pair(fig,"FIG03_FEATURE_PAIRED_EFFECTS");(FD/"FIG03_FEATURE_PAIRED_EFFECTS.csv").write_text("\n".join(",".join(str(r.get(k,"")) for k in ("contrast","horizon_steps","group","effect","ci_low","ci_high","p_two_sided","holm_p")) for r in paired)+"\n");cap("FIG03_FEATURE_PAIRED_EFFECTS","Primary feature contrasts with crossed seed×episode 95% intervals. Positive control-minus-candidate effects favor the candidate; aggregate rows are development evidence.");figs.append("FIG03_FEATURE_PAIRED_EFFECTS")
 # 4 interactions
 fig,axs=plt.subplots(1,2,figsize=(10,5),layout="constrained")
 for ax,group,title in zip(axs,("arm_mae_std","hand_mae_std"),("Arm horizon interaction","Hand horizon interaction")):
  rs=[r for r in interactions if r["group"]==group]; x2=np.arange(len(rs));v=np.array([float(r["effect"]) for r in rs]);lo=np.array([float(r["ci_low"]) for r in rs]);hi=np.array([float(r["ci_high"]) for r in rs]);ax.errorbar(x2,v,yerr=[v-lo,hi-v],fmt="o",capsize=4);ax.axhline(0,color="#555",lw=.8);ax.set_xticks(x2,[r["contrast"] for r in rs],rotation=35,ha="right");ax.set_title(title);ax.grid(axis="y",alpha=.25)
 pair(fig,"FIG04_HORIZON_INTERACTIONS");(FD/"FIG04_HORIZON_INTERACTIONS.csv").write_text("\n".join(",".join(str(r.get(k,"")) for k in ("contrast","group","effect","ci_low","ci_high","p_two_sided")) for r in interactions)+"\n");cap("FIG04_HORIZON_INTERACTIONS","The pre-registered 2s-minus-1s difference in each feature contrast on the shared primary episode population.");figs.append("FIG04_HORIZON_INTERACTIONS")
 # 5 Pareto
 fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
 for h,marker in ((30,"o"),(60,"s")):
  base=np.mean([float(r["hand_fd_native_mean"]) for r in []]) if False else None
  for c in CELLS:
   r=next(x for x in agg if int(x["horizon_steps"])==h and x["cell"]==c);ax.scatter(float(r["hand_mae_std_mean"]),float(r["hand_fd_native_mean"]),marker=marker,color=COLORS[c],s=65,label=f"{c} {h/30:.0f}s")
 ax.set(xlabel="Hand standardized MAE",ylabel="Hand native FD MAE",title="V3 development MAE–FD Pareto");ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
 pair(fig,"FIG05_MAE_FD_PARETO");(FD/"FIG05_MAE_FD_PARETO.csv").write_text("\n".join(",".join(str(r.get(k,"")) for k in ("horizon_steps","cell","hand_mae_std_mean","hand_fd_native_mean")) for r in agg)+"\n");cap("FIG05_MAE_FD_PARETO","Aggregate Hand level error versus native adjacent-difference error. Lower-left is favorable; old fixed-anchor margins are not applied as a gate to V3 training.");figs.append("FIG05_MAE_FD_PARETO")
 # 6 task/joint support
 task=read(AN/"TASK_JOINT_RESULTS.csv"); fig,axs=plt.subplots(1,2,figsize=(11,5),layout="constrained")
 for ax,level in zip(axs,("task","joint")):
  rs=[r for r in task if r["level"]==level and r["cell"]=="F-TOK-D" and r["horizon_steps"]=="30"]; vals=np.array([float(r["hand_mae_std"]) for r in rs]); ax.hist(vals,bins=20,color="#b91c1c",alpha=.75);ax.set_title(f"F-TOK-D Hand {level} support");ax.set_xlabel("standardized MAE");ax.set_ylabel("count");ax.grid(alpha=.2)
 pair(fig,"FIG06_TASK_JOINT_SUPPORT");(FD/"FIG06_TASK_JOINT_SUPPORT.csv").write_text("\n".join(",".join(str(r.get(k,"")) for k in ("level","index","hand_mae_std","N_window")) for r in task if r["cell"]=="F-TOK-D" and r["horizon_steps"]=="30")+"\n");cap("FIG06_TASK_JOINT_SUPPORT","Distribution of task/joint Hand errors for one pre-registered cell; full task-level claims require the complete table, not the histogram alone.");figs.append("FIG06_TASK_JOINT_SUPPORT")
 # 7 trajectories
 fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout="constrained"); data=load=None
 with np.load(EXP/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z: inp={k:z[k].copy() for k in z.files}
 dev=np.flatnonzero(inp["split_code"]==1); positions=[0,300,600];
 for ax,h in zip(axs,(30,60)):
  pred=np.load(EXP/f"predictions/V3-F-TOK-D-H{h}-S20260950/primary.npz",allow_pickle=False)["predictions_native"]; target=inp["targets_native"][dev,:h,0]; base=inp["anchor_command_native"][dev,0];
  for j,pos in enumerate(positions): ax.plot(np.arange(1,h+1),target[pos],alpha=.65,label="GT" if j==0 else None);ax.plot(np.arange(1,h+1),pred[pos,:,0],alpha=.8,label="F-TOK-D" if j==0 else None);ax.axhline(base[pos],ls="--",color="#777",alpha=.5,label="Persistence" if j==0 else None)
  ax.set_title(f"F-TOK-D {h/30:.0f}s; Hand joint0");ax.set_xlabel("future step");ax.grid(alpha=.25)
 axs[0].legend(fontsize=8);pair(fig,"FIG07_FIXED_TRAJECTORIES_1S_2S");(FD/"FIG07_FIXED_TRAJECTORIES_1S_2S.csv").write_text("fixed_positions,0,300,600\n");cap("FIG07_FIXED_TRAJECTORIES_1S_2S","Fixed representative primary-dev trajectories for both horizons. Selection is index-fixed, not outcome-selected; command values are not physical units.");figs.append("FIG07_FIXED_TRAJECTORIES_1S_2S")
 # 8 cost
 res=read(AN/"RESOURCE_COSTS.csv"); groups=[]
 for kind in ("feature","stage0","formal","inference"):
  vals=[float(r["wall_gpu_hours"]) for r in res if r["kind"]==kind and r["status"]=="complete" and r["wall_gpu_hours"] not in ("nan","")];groups.append((kind,sum(vals)))
 fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained");ax.bar(np.arange(len(groups)),[v for _,v in groups],color=["#374151","#2563eb","#b91c1c","#059669"]);ax.set_xticks(np.arange(len(groups)),[g for g,_ in groups]);ax.set_ylabel("GPU-hours");ax.set_title("V3 measured compute ledger");ax.grid(axis="y",alpha=.25)
 for i,(_,v) in enumerate(groups):ax.text(i,v,f"{v:.3f}",ha="center",va="bottom")
 pair(fig,"FIG08_RESOURCE_COST");(FD/"FIG08_RESOURCE_COST.csv").write_text("kind,gpu_hours\n"+"\n".join(f"{g},{v}" for g,v in groups)+"\n");cap("FIG08_RESOURCE_COST","Measured feature extraction, Stage0, formal training and inference GPU-hours. Waiting time is not counted as GPU compute.");figs.append("FIG08_RESOURCE_COST")
 (FM/"FIGURE_MANIFEST.json").write_text(json.dumps({"status":"PASS","figure_count":len(figs),"figures":figs,"final_test_access":0},indent=2)+"\n",encoding="utf-8");print(json.dumps({"status":"PASS","figure_count":len(figs)},sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
