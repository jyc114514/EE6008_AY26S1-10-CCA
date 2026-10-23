#!/usr/bin/env python3
"""Generate six source-linked figures for observation-delay robustness."""
from __future__ import annotations
import os
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
V3=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EXP=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
sys.path.insert(0,str(EXP/"code"))
from delay_contract import DELAYS, delay_transform_rows
SEEDS=(20260956,20260957,20260958)
CELLS=("CLEAN-N","CLEAN-D","CLEAN-S","ROBUST-N","ROBUST-D","ROBUST-S")
COLORS={"CLEAN-N":"#777777","CLEAN-D":"#0072B2","CLEAN-S":"#E69F00","ROBUST-N":"#555555","ROBUST-D":"#009E73","ROBUST-S":"#D55E00"}
MARKERS={"CLEAN-N":"o","CLEAN-D":"s","CLEAN-S":"^","ROBUST-N":"o","ROBUST-D":"s","ROBUST-S":"^"}
def rcsv(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def wcsv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save(fig,name):
    fig.savefig(EXP/f"figures/{name}.png",dpi=180,facecolor="white")
    fig.savefig(EXP/f"figures/{name}.pdf",facecolor="white");plt.close(fig)
def cap(name,text):
    (EXP/f"figure_manifests/{name}.caption.md").write_text(text+"\n",encoding="utf-8")
def main():
    for d in ("figures","figure_data","figure_manifests"):(EXP/d).mkdir(parents=True,exist_ok=True)
    curves=rcsv(EXP/"analysis/AGGREGATE_CURVES.csv");primary=rcsv(EXP/"analysis/PRIMARY_ROBUSTNESS_EFFECTS.csv")
    results=rcsv(EXP/"analysis/RESULTS_BY_SEED_AND_DELAY.csv");ages=rcsv(EXP/"analysis/OBSERVATION_AGE_DISTRIBUTION.csv")
    support=rcsv(EXP/"analysis/TASK_JOINT_SUPPORT.csv")
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with (V3/"data/V3_NORMALIZERS.json").open() as f:norm=json.load(f)
    dev=np.flatnonzero(data["split_code"]==1);row=int(dev[0]);t0=float(data["anchor_timestamp"][row]);qrel=data["history_query_timestamps"][row].astype(float)-t0
    # FIG01: real query/sample times, cutoffs, and effective age.
    fig,ax=plt.subplots(1,2,figsize=(11.2,3.7),layout="constrained");timeline=[]
    for di,delay in enumerate(DELAYS):
        _,age,selected,_=delay_transform_rows(data,norm,np.asarray([row]),di);selected=selected[0].astype(float)-t0;y=3-di
        y=3.5-di
        ax[0].scatter(selected,np.full(32,y),s=12,color="#0072B2",label="selected observation" if di==0 else None)
        ax[0].scatter(qrel,np.full(32,y-.18),s=15,marker="|",color="#999999",label="query time" if di==0 else None)
        ax[0].axvline(-float(delay),color="#D55E00",linestyle="--",linewidth=.9)
        for j in range(32):timeline.append({"anchor_id":str(data["anchor_id"][row]),"delay_seconds":float(delay),"query_index":j,"query_relative":float(qrel[j]),"selected_observation_relative":float(selected[j]),"age_seconds":float(age[0,j]),"cutoff_relative":-float(delay)})
    ax[0].axvline(0,color="black",linewidth=1,label="t0");ax[0].set(xlim=(-4.25,.15),ylim=(.2,3.8),yticks=[.5,1.5,2.5,3.5],yticklabels=["1.0s","0.5s","0.25s","0s"],xlabel="seconds relative to t0",title="Actual sample-and-hold timeline");ax[0].grid(axis="x",alpha=.2);ax[0].legend(frameon=False,fontsize=7,ncol=3)
    age_summary=[]
    for di,delay in enumerate(DELAYS):
        rows=[r for r in ages if int(r["delay_index"])==di and r["cell"]=="ROBUST-D"]
        mean=float(np.mean([float(r["episode_equal_mean_age_seconds"]) for r in rows]));p95=float(np.mean([float(r["age_p95_seconds"]) for r in rows]))
        age_summary.append({"delay_seconds":float(delay),"mean_age":mean,"mean_window_p95":p95})
        ax[1].plot([delay,delay],[mean,p95],color="#0072B2",linewidth=2);ax[1].scatter(delay,mean,color="#009E73",label="episode-equal mean" if di==0 else None);ax[1].scatter(delay,p95,color="#D55E00",marker="s",label="mean window p95" if di==0 else None)
    ax[1].plot([0,1],[0,1],":",color="#777777",label="requested delay");ax[1].set(xlabel="requested delay (s)",ylabel="effective age (s)",title="Saved observation age",xticks=DELAYS);ax[1].grid(alpha=.2);ax[1].legend(frameon=False,fontsize=7)
    wcsv(EXP/"figure_data/FIG01_TIMELINE.csv",timeline);wcsv(EXP/"figure_data/FIG01_AGE.csv",age_summary);save(fig,"FIG01_DELAY_TIMELINE_AND_AGE")
    cap("FIG01_DELAY_TIMELINE_AND_AGE","Actual source observation times selected for one fixed primary anchor; dashed lines mark t0-delta. The age summary uses saved effective ages. This is a deterministic simulation, not measured robot-network latency.")
    # FIG02: six-model MAE curves.
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.3),layout="constrained")
    base=json.loads((EXP/"analysis/BASELINE_RECONCILIATION.json").read_text())
    for a,metric,title in zip(axes,("arm_mae_std","hand_mae_std"),("Arm MAE","Hand MAE")):
        for cell in CELLS:
            rs=sorted([r for r in curves if r["cell"]==cell],key=lambda r:int(r["delay_index"]))
            a.errorbar([float(r["delay_seconds"]) for r in rs],[float(r[metric+"_mean"]) for r in rs],yerr=[float(r[metric+"_sd"]) for r in rs],color=COLORS[cell],marker=MARKERS[cell],linestyle="-" if cell.startswith("CLEAN") else "--",capsize=2,label=cell)
        a.axhline(float(base[metric]),color="black",linestyle=":",label="Persistence");a.set(title=title,xlabel="observation delay (s)",ylabel="episode-equal standardized MAE",xticks=DELAYS);a.grid(alpha=.2)
    axes[1].legend(frameon=False,fontsize=7,ncol=2);wcsv(EXP/"figure_data/FIG02_CURVES.csv",curves);save(fig,"FIG02_ARM_HAND_ERROR_VS_DELAY")
    cap("FIG02_ARM_HAND_ERROR_VS_DELAY","Six configurations over four delays. Points are three-seed means and error bars are seed SD; dotted line is unchanged Persistence.")
    # FIG03: primary effects and crossed bootstrap intervals.
    fig,a=plt.subplots(figsize=(9,4.8),layout="constrained")
    for i,r in enumerate(primary):
        e=float(r["effect"]);lo=float(r["ci_low"]);hi=float(r["ci_high"]);color="#0072B2" if r["group"]=="arm_mae_std" else "#D55E00"
        a.errorbar(i,e,yerr=[[e-lo],[hi-e]],fmt="o",capsize=4,color=color)
    a.axhline(0,color="black",linewidth=.9);a.set_xticks(range(len(primary)),[f"{r['contrast']}\n{r['group'].split('_')[0].title()}" for r in primary],rotation=22,ha="right");a.set(ylabel="control MAE - candidate MAE (positive favors candidate)",title="Stress outcome: equal mean over 0.25, 0.50, 1.00 s");a.grid(axis="y",alpha=.2)
    wcsv(EXP/"figure_data/FIG03_PRIMARY.csv",primary);save(fig,"FIG03_PRIMARY_ROBUSTNESS_EFFECTS")
    cap("FIG03_PRIMARY_ROBUSTNESS_EFFECTS","Six prespecified Arm/Hand comparisons with unadjusted crossed seed-by-episode 95% intervals. Holm values are in the CSV; positive effects favor the candidate.")
    # FIG04: clean/stress Pareto.
    pareto=[];fig,axes=plt.subplots(1,2,figsize=(11.2,4.8),layout="constrained")
    offsets_arm=[(-42,17),(9,-22),(-42,4),(-43,-17),(9,17),(-43,-31)]
    offsets_hand=[(8,-18),(9,17),(-43,17),(9,31),(-43,-19),(-43,-2)]
    for a,metric,title in zip(axes,("arm_mae_std","hand_mae_std"),("Arm","Hand")):
        for ci,cell in enumerate(CELLS):
            clean=[];stress=[]
            for seed in SEEDS:
                rs=[r for r in results if int(r["seed"])==seed and r["cell"]==cell]
                clean.append(float(next(r[metric] for r in rs if float(r["delay_seconds"])==0)))
                stress.append(float(np.mean([float(r[metric]) for r in rs if float(r["delay_seconds"]) in (.25,.5,1.)])))
            x=float(np.mean(clean));y=float(np.mean(stress));xs=float(np.std(clean,ddof=1));ys=float(np.std(stress,ddof=1))
            short={"CLEAN-N":"C-N","CLEAN-D":"C-D","CLEAN-S":"C-S","ROBUST-N":"R-N","ROBUST-D":"R-D","ROBUST-S":"R-S"}[cell]
            offsets=offsets_arm if metric=="arm_mae_std" else offsets_hand
            a.errorbar(x,y,xerr=xs,yerr=ys,marker=MARKERS[cell],color=COLORS[cell],capsize=2,linestyle="none");a.annotate(short,(x,y),xytext=offsets[ci],textcoords="offset points",fontsize=7,arrowprops={"arrowstyle":"-","lw":.5,"color":COLORS[cell]})
            pareto.append({"cell":cell,"group":metric,"clean_mean":x,"clean_seed_sd":xs,"stress_mean":y,"stress_seed_sd":ys})
        a.set(title=title,xlabel="MAE at delta=0",ylabel="mean MAE over delayed conditions");a.grid(alpha=.2)
    wcsv(EXP/"figure_data/FIG04_PARETO.csv",pareto);save(fig,"FIG04_CLEAN_DELAY_PARETO")
    cap("FIG04_CLEAN_DELAY_PARETO","Each point is a configuration; x is normal-input MAE and y is equal mean over three delayed inputs. Error bars are across-seed SD. C denotes clean-only training, R denotes mixed-delay training; suffix N/D/S denotes no video/dynamic/static video.")
    # FIG05: fixed future curves with input cutoffs.
    t=np.arange(1,61)/30;target=data["targets_native"][row,:60];fig=plt.figure(figsize=(12,7.2),layout="constrained");gs=fig.add_gridspec(2,2,height_ratios=(.65,1))
    ax=fig.add_subplot(gs[0,:])
    for di,delay in enumerate(DELAYS):
        run=f"ODR-ROBUST-D-H60-S{SEEDS[0]}"
        with np.load(EXP/f"predictions/{run}/delay_{di}.npz",allow_pickle=False) as z:selected=z["observation_selected_timestamps"][0].astype(float)-t0
        y=3.5-di;ax.scatter(selected,np.full(32,y),s=11,color="#0072B2");ax.scatter(qrel,np.full(32,y-.18),s=13,marker="|",color="#999999");ax.axvline(-float(delay),color="#D55E00",linestyle="--",linewidth=.8)
    ax.axvline(0,color="black");ax.set(xlim=(-4.2,.12),ylim=(.2,3.8),yticks=[.5,1.5,2.5,3.5],yticklabels=["1.0s","0.5s","0.25s","0s"],xlabel="seconds relative to t0",title=f"Fixed primary anchor {data['anchor_id'][row]}");ax.grid(axis="x",alpha=.2)
    trajectory=[]
    for a,j,title in ((fig.add_subplot(gs[1,0]),0,"Arm dim 0"),(fig.add_subplot(gs[1,1]),14,"Hand dim 0")):
        a.plot(t,target[:,j],color="black",lw=1.8,label="GT");a.plot(t,np.repeat(data["anchor_command_native"][row,j],60),color="#777777",ls=":",label="Persistence")
        for cell,di in (("CLEAN-D",0),("CLEAN-D",3),("ROBUST-D",0),("ROBUST-D",2),("ROBUST-D",3)):
            run=f"ODR-{cell}-H60-S{SEEDS[0]}"
            with np.load(EXP/f"predictions/{run}/delay_{di}.npz",allow_pickle=False) as z:vals=z["predictions_native"][0,:,j]
            color="#0072B2" if cell=="CLEAN-D" else "#009E73";style="-" if cell=="CLEAN-D" else "--"
            a.plot(t,vals,color=color,ls=style,lw=1.1,label=f"{cell} delta={DELAYS[di]:g}")
            for k,v in enumerate(vals):trajectory.append({"joint":j,"future_seconds":float(t[k]),"series":f"{cell}-D{di}","prediction_native":float(v)})
        a.set(title=title,xlabel="future seconds",ylabel="native command");a.grid(alpha=.2)
    handles,labels=fig.axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False,fontsize=7)
    wcsv(EXP/"figure_data/FIG05_TRAJECTORY.csv",trajectory);save(fig,"FIG05_FIXED_TRAJECTORIES_AND_CUTOFF")
    cap("FIG05_FIXED_TRAJECTORIES_AND_CUTOFF","Fixed primary anchor, observed query/sample timeline and feedback cutoffs above, future GT/Persistence and CLEAN-D/ROBUST-D outputs below. This example is descriptive; inference uses all primary episodes.")
    # FIG06: joint and task support with individual seed directions.
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout="constrained");jm=np.zeros((28,3))
    for j in range(28):
        col="arm_mae_std" if j<14 else "hand_mae_std";rs=[r for r in support if r["level"]=="joint" and int(r["index"])==j and float(r["delay_seconds"]) in (.25,.5,1.) and r["cell"] in ("ROBUST-N","ROBUST-D")]
        for si,seed in enumerate(SEEDS):
            n=np.mean([float(r[col]) for r in rs if int(r["seed"])==seed and r["cell"]=="ROBUST-N"]);d=np.mean([float(r[col]) for r in rs if int(r["seed"])==seed and r["cell"]=="ROBUST-D"]);jm[j,si]=n-d
    vmax=max(float(np.max(np.abs(jm))),1e-9);im=axes[0].imshow(jm,aspect="auto",cmap="RdBu",vmin=-vmax,vmax=vmax,interpolation="nearest")
    axes[0].set(title="ROBUST-D vs ROBUST-N by joint",xlabel="seed",ylabel="joint: 0-13 Arm; 14-27 Hand",xticks=range(3),xticklabels=[str(s) for s in SEEDS],yticks=range(28));fig.colorbar(im,ax=axes[0],label="N MAE - D MAE (positive favors D)")
    tr=[r for r in support if r["level"]=="task" and float(r["delay_seconds"]) in (.25,.5,1.) and r["cell"] in ("ROBUST-N","ROBUST-D")];te=[]
    for task in sorted({int(r["index"]) for r in tr}):
        vals=[]
        for seed in SEEDS:
            n=np.mean([float(r["full_mae_std"]) for r in tr if int(r["index"])==task and int(r["seed"])==seed and r["cell"]=="ROBUST-N"]);d=np.mean([float(r["full_mae_std"]) for r in tr if int(r["index"])==task and int(r["seed"])==seed and r["cell"]=="ROBUST-D"]);vals.append(n-d)
        te.append({"task_index":task,"effect_mean":float(np.mean(vals)),"seed_positive_count":int(sum(v>0 for v in vals)),"seed_effects":json.dumps(vals)})
    fraction=sum(r["effect_mean"]>0 for r in te)/max(1,len(te));axes[1].hist([r["effect_mean"] for r in te],bins=20,color="#009E73",alpha=.85);axes[1].axvline(0,color="black",lw=.8);axes[1].set(title=f"Task support: {fraction:.1%} favor ROBUST-D",xlabel="ROBUST-N MAE - ROBUST-D MAE",ylabel="task count");axes[1].grid(axis="y",alpha=.2)
    wcsv(EXP/"figure_data/FIG06_TASK_EFFECTS.csv",te)
    wcsv(EXP/"figure_data/FIG06_JOINT_SEED_EFFECTS.csv",[{"joint_index":j,"seed":seed,"robust_n_minus_d_stress_mae":float(jm[j,i])} for j in range(28) for i,seed in enumerate(SEEDS)])
    save(fig,"FIG06_JOINT_AND_TASK_SEED_SUPPORT")
    cap("FIG06_JOINT_AND_TASK_SEED_SUPPORT","Joint effects are shown per seed; the task histogram uses episode-equal Full MAE differences averaged over three delayed conditions and seeds. Positive effects favor ROBUST-D; this support analysis supplements the primary family.")
    names=["FIG01_DELAY_TIMELINE_AND_AGE","FIG02_ARM_HAND_ERROR_VS_DELAY","FIG03_PRIMARY_ROBUSTNESS_EFFECTS","FIG04_CLEAN_DELAY_PARETO","FIG05_FIXED_TRAJECTORIES_AND_CUTOFF","FIG06_JOINT_AND_TASK_SEED_SUPPORT"]
    manifest={"status":"PASS","figures":names,"formats":["png","pdf"],"csv_sources":True,"captions":True,"final_test_access":0}
    (EXP/"figure_manifests/FIGURE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
