from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V8=Path(__file__).resolve().parents[1]
COLORS={"blue":"#0072B2","orange":"#D55E00","green":"#009E73","purple":"#CC79A7",
        "black":"#222222","gray":"#666666"}
METHOD_STYLE={
 "G":(COLORS["black"],"o","-"),
 "U":(COLORS["blue"],"s","--"),
 "M":(COLORS["orange"],"^","-."),
 "A":(COLORS["blue"],"o","-"),
 "B":(COLORS["green"],"s","--"),
 "E":(COLORS["orange"],"^","-."),
}


def now():
 return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")


def read_csv(path):
 with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def write_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()


def save(fig,name):
 p=V8/"figures"/name
 p.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(p,dpi=180,format="png",facecolor="white",bbox_inches="tight")
 plt.close(fig)
 from PIL import Image
 with Image.open(p) as im:
  return {"path":str(p),"sha256":sha(p),"format":"PNG","pixels":list(im.size),
          "dpi":list(im.info.get("dpi",(None,None)))}


def render_task_gains():
 rows=[r for r in read_csv(V8/"results/DECISION_TABLE.csv")
       if r["record_type"]=="accepted_paired_gain" and r["contrast"]=="MIX_VS_GT"
       and r["scope"]=="task"]
 tasks=[119,94,90,141,230]
 seeds=[202610131,202610132,202610133]
 specs=[("position_mae","Position MAE gain (standardized)"),
        ("safe_fd_l1","Safe-FD gain (standardized)"),
        ("seam_safe_fd_l1","Seam gain (standardized)")]
 fig,axes=plt.subplots(3,1,figsize=(10.5,9.0),layout="constrained",sharex=True)
 offsets={seeds[0]:-0.10,seeds[1]:0.0,seeds[2]:0.10}
 for ax,(metric,ylabel) in zip(axes,specs):
  values=[]
  for seed,color,marker in ((seeds[0],COLORS["blue"],"o"),(seeds[1],COLORS["green"],"s"),(seeds[2],COLORS["orange"],"^")):
   y=[]
   for task in tasks:
    row=next(r for r in rows if r["metric"]==metric and int(r["task_id"])==task and int(r["seed"])==seed)
    y.append(float(row["gain"]))
   values.extend(y)
   ax.plot(np.asarray(tasks)+offsets[seed],y,color=color,marker=marker,linestyle="None",
           linewidth=0,markersize=5,label=str(seed))
  means=[np.mean([float(next(r for r in rows if r["metric"]==metric and int(r["task_id"])==t and int(r["seed"])==s)["gain"]) for s in seeds]) for t in tasks]
  ax.plot(tasks,means,color=COLORS["black"],marker="D",linestyle="None",linewidth=0,markersize=6,label="3-seed mean")
  ax.axhline(0,color=COLORS["gray"],linewidth=0.8)
  lo=min(0,min(values));hi=max(0,max(values));pad=max((hi-lo)*0.12,0.00003)
  ax.set_ylim(lo-pad,hi+pad)
  ax.set_ylabel(ylabel)
  ax.grid(axis="y",color="#dddddd",linewidth=0.6)
 axes[-1].set_xticks(tasks);axes[-1].set_xlabel("Task ID")
 axes[0].legend(ncol=4,frameon=False,fontsize=8,loc="best")
 fig.suptitle("Accepted V7 MIX_SOFT vs GT_RETRAIN task gains\npositive values favor MIX_SOFT; each point is one seed",fontsize=12)
 return save(fig,"DEV_TASK_GAINS_MIX_VS_GT.png")


def render_teacher_distances():
 rows=read_csv(V8/"results/TRAIN_DEV_IMITATION.csv")
 specs=[("position_distance","Position distance to T"),
        ("safe_fd_distance","Internal safe-FD distance to T"),
        ("seam_distance","Seam distance to T")]
 fig,axes=plt.subplots(1,3,figsize=(13.2,4.4),layout="constrained")
 x=np.array([0,1],dtype=float)
 offset={"G":-0.08,"U":0.0,"M":0.08}
 for ax,(metric,label) in zip(axes,specs):
  for method in ("G","U","M"):
   vals=[];errs=[]
   for split in ("train","dev"):
    subset=[r for r in rows if r["scope"]=="equal_task_macro" and r["split"]==split and r["method"]==method]
    seedvals=[float(r[metric]) for r in subset]
    vals.append(float(np.mean(seedvals)));errs.append(float(np.std(seedvals,ddof=1)))
   color,marker,linestyle=METHOD_STYLE[method]
   ax.errorbar(x+offset[method],vals,yerr=errs,color=color,marker=marker,linestyle=linestyle,
               capsize=3,linewidth=1.4,markersize=5,label=method)
  ax.set_xticks(x,["Train","Dev"])
  ax.set_ylabel(label)
  ax.grid(axis="y",color="#dddddd",linewidth=0.6)
  ax.set_xlim(-0.35,1.35)
 axes[0].legend(frameon=False,title="Method")
 fig.suptitle("Frozen student-to-teacher distance, mean and sample SD across 3 seeds",fontsize=12)
 return save(fig,"TRAIN_DEV_TEACHER_DISTANCE.png")


def render_geometry():
 rows=read_csv(V8/"results/CORRECTION_GEOMETRY.csv")
 specs=[("position","weighted_cosine","Position cosine"),
        ("position","projection_k","Position projection k"),
        ("internal_fd","weighted_cosine","Internal-FD cosine"),
        ("internal_fd","projection_k","Internal-FD projection k")]
 fig,axes=plt.subplots(2,2,figsize=(10.8,7.4),layout="constrained")
 x=np.array([0,1],dtype=float)
 for ax,(space,field,label) in zip(axes.flat,specs):
  for vec in ("A","B","E"):
   vals=[]
   for split in ("train","dev"):
    r=next(q for q in rows if q["scope"]=="three_seed_macro" and q["split"]==split
           and q["record_type"]=="projection" and q["space"]==space and q["vector"]==vec)
    vals.append(float(r[field]))
   color,marker,linestyle=METHOD_STYLE[vec]
   ax.plot(x,vals,color=color,marker=marker,linestyle=linestyle,linewidth=1.5,markersize=6,label=vec)
  ax.axhline(0,color=COLORS["gray"],linewidth=0.8)
  ax.set_xticks(x,["Train","Dev"]);ax.set_ylabel(label);ax.grid(axis="y",color="#dddddd",linewidth=0.6)
  if field=="weighted_cosine":ax.set_ylim(-1.05,1.05)
  else:
   vals=[float(r[field]) for r in rows if r["scope"]=="three_seed_macro" and r["space"]==space and r["record_type"]=="projection"]
   lo=min(0,min(vals));hi=max(0,max(vals));pad=max((hi-lo)*0.12,0.05)
   ax.set_ylim(lo-pad,hi+pad)
 axes[0,0].legend(frameon=False,title="Student change")
 fig.suptitle("Correction geometry relative to teacher correction C=T-G\nA=M-G, B=U-G, E=M-U; k is descriptive, not a deployment coefficient",fontsize=11)
 return save(fig,"CORRECTION_GEOMETRY.png")


def render_help_harm():
 rows=read_csv(V8/"results/TEACHER_HELP_HARM.csv")
 metrics=[("position_mae","Position MAE"),("safe_fd_l1","Internal safe-FD"),("seam_safe_fd_l1","Seam")]
 categories=[("teacher_help","Teacher helps",COLORS["blue"]),("teacher_harm","Teacher harms",COLORS["orange"])]
 tasks=[119,94,90,141,230]
 fig,axes=plt.subplots(3,1,figsize=(11,8.2),layout="constrained",sharex=True)
 width=0.34;x=np.arange(len(tasks))
 for ax,(metric,label) in zip(axes,metrics):
  for j,(category,catlabel,color) in enumerate(categories):
   vals=[]
   for task in tasks:
    z=[float(r["teacher_gap_weighted_contribution_G_minus_T"]) for r in rows
       if int(r["task_id"])==task and r["metric"]==metric and r["teacher_stratum"]==category]
    vals.append(float(np.mean(z)))
   ax.bar(x+(j-0.5)*width,vals,width,label=catlabel,color=color,edgecolor="black",linewidth=0.4)
  ax.axhline(0,color=COLORS["black"],linewidth=0.8)
  ax.set_ylabel(label+" contribution")
  ax.grid(axis="y",color="#dddddd",linewidth=0.6)
 axes[0].legend(frameon=False,ncol=2)
 axes[-1].set_xticks(x,tasks);axes[-1].set_xlabel("Task ID")
 fig.suptitle("Accepted V7 teacher-help/harm strata\nweighted contributions of L(G,Y)-L(T,Y); groups/tasks weighted, windows are not independent N",fontsize=11)
 return save(fig,"TEACHER_HELP_HARM_CONTRIBUTIONS.png")


def main():
 out=[
  render_task_gains(),
  render_teacher_distances(),
  render_geometry(),
  render_help_harm(),
 ]
 inputs=["results/DECISION_TABLE.csv","results/TRAIN_DEV_IMITATION.csv",
         "results/CORRECTION_GEOMETRY.csv","results/TEACHER_HELP_HARM.csv"]
 manifest={
  "schema_version":1,"created_utc":now(),"status":"PASS_WITH_LIMITS",
  "figure_destination":"provisional internal report; no publisher/journal specified",
  "matplotlib_version":matplotlib.__version__,"dpi":180,
  "raw_target_values_loaded_by_renderer":False,"prediction_arrays_loaded_by_renderer":False,
  "input_tables":[{"path":str(V8/p),"sha256":sha(V8/p)} for p in inputs],
  "outputs":out,
  "review":{"agent_visual_inspection":"PENDING","human_review":"PENDING"},
  "limitations":["dev horizon/axis values unavailable; no target-bearing case trajectory figure",
                 "all error bars in teacher-distance plot are sample SD across three seeds",
                 "task-gain plot shows seed points and mean; no new CI or p-values"],
 }
 write_json(V8/"figure_data/FIGURE_EXPORT_METADATA.json",manifest)
 print(json.dumps({"status":"PASS_WITH_LIMITS","figures":[x["path"] for x in out],
                   "renderer_target_loads":0,"renderer_prediction_tensor_loads":0},sort_keys=True))


if __name__=="__main__":
 main()
