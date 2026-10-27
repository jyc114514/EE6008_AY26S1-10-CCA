#!/usr/bin/env python3
"""Create six figure groups with source CSVs and captions for the audit."""
from __future__ import annotations
import os
import csv,hashlib,json
import textwrap
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z'))
V3=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
FIG=ROOT/"figures";DATA=ROOT/"figure_data";CAP=ROOT/"figures/captions"
for d in (FIG,DATA,CAP):d.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.size":9,"axes.titlesize":11,"axes.labelsize":9,"legend.fontsize":8,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
COLORS={"CLEAN-N":"#4C78A8","CLEAN-D":"#F58518","CLEAN-S":"#54A24B","ROBUST-N":"#9D755D","ROBUST-D":"#E45756","ROBUST-S":"#72B7B2"}
COND_COLOR={"MATCHED-D":"#1B6CA8","MATCHED-S":"#777777","SAME-TASK-DONOR-D-M1":"#F2A541","CROSS-TASK-DONOR-D-M1":"#C44E52","PERSISTENCE":"#222222"}

MAP_SEEDS=(202609701,202609702,202609703)
def rd(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def wc(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def save(fig,name,caption):
    fig.savefig(FIG/f"{name}.png",dpi=180,bbox_inches="tight")
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)
    (CAP/f"{name}.md").write_text(caption.strip()+"\n",encoding="utf-8")
def format_condition(cond):
    return {"MATCHED-D":"Matched dynamic","MATCHED-S":"Matched static","SAME-TASK":"Same-task donor","CROSS-TASK":"Cross-task donor"}.get(cond,cond)

def figure1():
    table=rd(ROOT/"analysis/DELAY_RESULTS_RECONCILIATION_TABLE.csv")
    guard=json.loads((ROOT/"DELAY_RESULT_RECONCILIATION.zh-CN.json").read_text(encoding="utf-8"))
    guards=guard["normal_input_guardrails"]
    wc(DATA/"FIG01_DELAY_AND_GUARDRAILS.csv",table+guards)
    fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(2,2,height_ratios=(1,0.78))
    axa=fig.add_subplot(gs[0,0]);axh=fig.add_subplot(gs[0,1]);axg=fig.add_subplot(gs[1,:])
    for cell in ("CLEAN-N","CLEAN-D","CLEAN-S","ROBUST-N","ROBUST-D","ROBUST-S"):
        rows=sorted([r for r in table if r["cell"]==cell],key=lambda x:float(x["delay_seconds"]))
        x=np.asarray([float(r["delay_seconds"]) for r in rows])
        for ax,key,sdkey,label in ((axa,"arm_mae_std_mean","arm_mae_std_seed_sd","Arm"),(axh,"hand_mae_std_mean","hand_mae_std_seed_sd","Hand")):
            y=np.asarray([float(r[key]) for r in rows]);e=np.asarray([float(r[sdkey]) for r in rows])
            ax.errorbar(x,y,yerr=e,color=COLORS[cell],marker="o",linewidth=1.5,capsize=2,label=cell)
    axa.set(title="Arm standardized MAE",xlabel="Requested observation delay (s)",ylabel="MAE")
    axh.set(title="Hand standardized MAE",xlabel="Requested observation delay (s)",ylabel="MAE")
    for ax in (axa,axh):ax.set_xticks([0,.25,.5,1]);ax.grid(alpha=.25)
    axh.legend(ncol=2,loc="upper left",frameon=False)
    names=["arm_mae_std","hand_mae_std","arm_fd_native","hand_fd_native"]
    x=np.arange(len(names));y=np.asarray([100*float(next(g for g in guards if g["group"]==n)["relative_change"]) for n in names])
    lo=np.asarray([100*float(next(g for g in guards if g["group"]==n)["relative_ci_low"]) for n in names]);hi=np.asarray([100*float(next(g for g in guards if g["group"]==n)["relative_ci_high"]) for n in names])
    axg.axhline(0,color="#555555",linewidth=.8);axg.axhline(1,color="#C44E52",linestyle="--",linewidth=1,label="+1% practical-harm limit")
    axg.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt="o",capsize=4,color="#4C78A8")
    axg.set_xticks(x,["Arm MAE","Hand MAE","Arm FD","Hand FD"]);axg.set_ylabel("ROBUST-D − CLEAN-D at delay 0 (%)");axg.set_title("Normal-input cost guardrail (crossed 95% CI)")
    axg.grid(axis="y",alpha=.25);axg.legend(frameon=False,loc="upper left")
    fig.suptitle("Delay robustness and normal-input trade-off",y=.995,fontweight="bold");fig.tight_layout()
    save(fig,"FIG01_DELAY_AND_NORMAL_GUARDRAILS","Development-only delay summary. Top: six accepted model cells, mean ± training-seed SD over the 603-episode source population. Bottom: the locked ROBUST-D minus CLEAN-D zero-delay relative guardrails; Arm FD remains compatible with harm above the +1% margin.")
def figure2():
    quality=rd(ROOT/"analysis/DONOR_SUPPORT_SUMMARY.csv");div=rd(ROOT/"MAP_DIVERSITY.csv")
    rows=[]
    for q in quality:
        related=[x for x in div if x["pairing_type"]==q["pairing_type"] and str(q["map_seed"]) in (x["map_seed_a"],x["map_seed_b"])]
        if not related:raise RuntimeError("map diversity summary is missing pairwise comparisons")
        rows.append({**q,"different_pair_fraction_mean":float(np.mean([float(x["different_pair_fraction"]) for x in related]))})
    wc(DATA/"FIG02_DONOR_MATCHING_QUALITY.csv",rows)
    fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    labels=[f"{'same-task' if r['pairing_type']=='same_task' else 'cross-task'} M{int(str(r['map_seed'])[-1])}" for r in rows]
    colors=["#F2A541" if r["pairing_type"]=="same_task" else "#C44E52" for r in rows]
    x=np.arange(len(rows))
    axs[0].bar(x,[float(r["dynamic_token_distance_mean"]) for r in rows],color=colors)
    axs[0].scatter(x,[float(r["dynamic_token_distance_p95"]) for r in rows],color="#222222",marker="_",s=180,label="p95")
    axs[0].set(title="Dynamic-token distance",ylabel="Mean absolute feature difference",xticks=x,xticklabels=labels);axs[0].tick_params(axis="x",rotation=35);axs[0].legend(frameon=False)
    axs[1].bar(x,[float(r["different_pair_fraction_mean"]) for r in rows],color=colors)
    axs[1].set(title="Mapping diversity",ylabel="Fraction of different donor pairs",ylim=(0,1.05),xticks=x,xticklabels=labels);axs[1].tick_params(axis="x",rotation=35)
    axs[2].bar(x,[float(r["donor_window_reuse_fraction"]) for r in rows],color=colors)
    axs[2].set(title="Donor-window reuse",ylabel="Repeated use beyond first (%)",ylim=(0,1),xticks=x,xticklabels=labels);axs[2].tick_params(axis="x",rotation=35)
    for ax in axs:ax.grid(axis="y",alpha=.2)
    fig.suptitle("Outcome-blind donor support and matching quality",fontweight="bold");fig.tight_layout()
    save(fig,"FIG02_DONOR_MATCHING_QUALITY","All maps are one-to-one at the episode level, each covering 578 recipients and 578 distinct donors; same-task maps preserve task, cross-task maps change it, and all donors cross duplicate groups. Token distance is descriptive; the larger cross-task distance leaves out-of-distribution sensitivity as a competing explanation.")
def MAP_SEEDS_NEXT(seed):
    seeds=(202609701,202609702,202609703);i=seeds.index(int(seed));return seeds[(i+1)%3]
def figure3():
    rows=rd(ROOT/"analysis/PRIMARY_CORRESPONDENCE_EFFECTS.csv");wc(DATA/"FIG03_PRIMARY_EFFECTS.csv",rows)
    order=[]
    for context in ("ANCHOR","ROBUST_STRESS"):
        for contrast in ("MATCHED-D_vs_SAME-TASK","MATCHED-D_vs_MATCHED-S"):
            for group in ("Arm","Hand"):
                order.append(next(x for x in rows if x["context_group"]==context and x["contrast"]==contrast and x["group"]==group))
    fig,ax=plt.subplots(figsize=(10,5.5));y=np.arange(len(order))[::-1]
    for yi,r in zip(y,order):
        e=float(r["effect"]);lo=float(r["crossed_ci_low"]);hi=float(r["crossed_ci_high"])
        color="#4C78A8" if r["context_group"]=="ANCHOR" else "#E45756"
        ax.errorbar(e,yi,xerr=[[e-lo],[hi-e]],fmt="o",color=color,capsize=3)
    ax.axvline(0,color="#333333",linewidth=.9)
    labels=[f"{r['context_group']} · {r['contrast'].replace('MATCHED-D_vs_','')} · {r['group']}" for r in order]
    ax.set_yticks(y,labels);ax.set_xlabel("Alternative loss − MATCHED-D loss (standardized MAE; positive favors matched dynamic)")
    ax.set_title("Primary episode-correspondence effects: crossed seed × episode 95% CIs",fontweight="bold");ax.grid(axis="x",alpha=.25)
    fig.tight_layout();save(fig,"FIG03_PRIMARY_CORRESPONDENCE_EFFECTS","Eight locked primary Arm/Hand effects. Each crossed interval resamples the three model seeds and 578 recipient episodes; Holm correction is applied to conditional seed sign-flip p-values in the CSV. These are fixed-model development input interventions, not physical causal effects.")
def figure4():
    rows=rd(ROOT/"analysis/SEED_MAP_SENSITIVITY.csv");wc(DATA/"FIG04_SEED_MAP_SENSITIVITY.csv",rows)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharey=False)
    contexts=("ANCHOR","ROBUST_STRESS");contrasts=("MATCHED-D_vs_SAME-TASK","MATCHED-D_vs_MATCHED-S")
    seed_colors={20260953:"#4C78A8",20260954:"#F58518",20260955:"#54A24B",20260956:"#4C78A8",20260957:"#F58518",20260958:"#54A24B"}
    for ax,(ctx,contrast) in zip(axs.flat,[(c,k) for c in contexts for k in contrasts]):
        sub=[r for r in rows if r["context_group"]==ctx and r["contrast"]==contrast]
        maps=sorted(set(int(r["map_seed"]) for r in sub));mapping={m:i for i,m in enumerate(maps)}
        for r in sub:
            pos=mapping[int(r["map_seed"])]
            if r["group"]=="Hand":pos+=.12
            else:pos-=.12
            ax.scatter(pos,float(r["episode_mean_effect"]),color=seed_colors[int(r["seed"])],marker="o" if r["group"]=="Arm" else "s",s=40,alpha=.85)
        ax.axhline(0,color="#333333",linewidth=.8);ax.set_xticks(range(len(maps)),["Static" if m==-1 else f"M{MAP_SEEDS.index(m)+1}" for m in maps])
        ax.set_title(f"{ctx}: {contrast.replace('MATCHED-D_vs_','')}");ax.set_ylabel("Alternative − MATCHED-D MAE");ax.grid(axis="y",alpha=.2)
    fig.suptitle("Per-seed and per-map sensitivity (points are episode means)",fontweight="bold")
    from matplotlib.lines import Line2D
    legend=[Line2D([],[],marker="o",linestyle="None",color="#444444",label="Arm"),Line2D([],[],marker="s",linestyle="None",color="#444444",label="Hand")]
    legend += [Line2D([],[],marker="o",linestyle="None",color=seed_colors[s],label=f"Seed rank {i+1}") for i,s in enumerate((20260953,20260954,20260955))]
    fig.legend(handles=legend,ncol=5,loc="lower center",bbox_to_anchor=(.5,-.01),frameon=False)
    fig.text(.5,.035,"Seed rank is within family (SIV 53–55; ROBUST 56–58); maps are fixed diagnostics, not independent replicates.",ha="center")
    fig.tight_layout(rect=(0,.07,1,.96));save(fig,"FIG04_SEED_MAP_SENSITIVITY","Per-seed and per-map episode-mean effects; positive means the alternative video worsens MAE relative to correctly matched dynamic video. Maps are shown separately for sensitivity and are never counted as independent samples.")
def select_window_episode_data():
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:
        ids=z["anchor_id"].copy();eps=z["episode_index"].copy();split=z["split_code"].copy();frames=z["anchor_frame"].copy();targets=z["targets_native"].copy();anchors=z["anchor_command_native"].copy()
    eligible={int(r["episode_index"]) for r in rd(ROOT/"RECIPIENT_ELIGIBILITY.csv") if r["eligible_common_subset"]=="True"}
    subset=np.asarray([i for i in np.flatnonzero(split==1) if int(eps[i]) in eligible],dtype=np.int64)
    recipient_ids=ids[subset];lookup={str(a):j for j,a in enumerate(recipient_ids)}
    eps_sorted=sorted(eligible)
    fixed=sorted(eps_sorted,key=lambda e:hashlib.sha256(f"VCA_FIXED_EXAMPLE_20260923:{e}".encode()).hexdigest())[:3]
    chosen=[]
    for ep in fixed:
        ix=[i for i in subset if int(eps[i])==ep];row=min(ix,key=lambda i:int(frames[i]));chosen.append((ep,row,lookup[str(ids[row])],"hash_selected"))
    # Explicit post-hoc failure illustration: worst episode-level Hand MAE for one frozen seed.
    outcomes=rd(ROOT/"analysis/PER_EPISODE_RESULTS.csv")
    scored=[r for r in outcomes if r["context_id"]=="SIV-ANCHOR-D-H60-S20260953" and r["condition"]=="MATCHED-D"]
    worst=sorted(scored,key=lambda r:(-float(r["hand_mae_std"]),int(r["episode_index"])))[:3]
    failures=[]
    for r in worst:
        ep=int(r["episode_index"]);ix=[i for i in subset if int(eps[i])==ep];row=min(ix,key=lambda i:int(frames[i]));failures.append((ep,row,lookup[str(ids[row])],"posthoc_high_hand_error"))
    return ids,eps,subset,recipient_ids,lookup,targets,anchors,chosen,failures
def figure5():
    ids,eps,subset,recipient_ids,local,targets,anchors,fixed,failures=select_window_episode_data()
    context="SIV-ANCHOR-D-H60-S20260953"
    conditions=("MATCHED-D","MATCHED-S","SAME-TASK-DONOR-D-M1","CROSS-TASK-DONOR-D-M1")
    colors={"MATCHED-D":"#1B6CA8","MATCHED-S":"#777777","SAME-TASK-DONOR-D-M1":"#F2A541","CROSS-TASK-DONOR-D-M1":"#C44E52","PERSISTENCE":"#222222"}
    loaded={}
    for c in conditions:
        with np.load(ROOT/f"predictions/{context}/{c}.npz",allow_pickle=False) as z:loaded[c]=z["predictions_native"].copy()
    rows=[];fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for col,(ep,row,local_idx,selection) in enumerate(fixed):
        aid=str(ids[row]);bundle_idx=local[aid];target=targets[row];persist=anchors[row]
        for ri,joint in enumerate((0,14)):
            ax=axs[ri,col]
            series=[("GT",target[:,joint],"#111111","-"),("Persistence",np.full(60,persist[joint]),colors["PERSISTENCE"],":")]
            for c in conditions:series.append((format_condition(c),loaded[c][bundle_idx,:,joint],colors[c],"--" if c=="MATCHED-S" else "-"))
            for label,val,color,style in series:
                ax.plot(np.arange(60),val,label=label,color=color,linestyle=style,linewidth=1.15 if label!="GT" else 1.8)
                for t,v in enumerate(val):rows.append({"selection":selection,"episode_index":ep,"anchor_id":aid,"joint_index":joint,"step":t,"series":label,"command_value":float(v)})
            ax.set_title(f"Episode {ep} · {'Arm j0' if joint==0 else 'Hand j14'}")
            ax.grid(alpha=.2)
            if col==0:ax.set_ylabel("Dataset command value")
            if ri==1:ax.set_xlabel("Future step")
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=6,loc="upper center",bbox_to_anchor=(.5,.99),frameon=False)
    fig.suptitle("Outcome-blind fixed episode examples · frozen SIV seed 20260953",y=1.05,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.93));save(fig,"FIG05A_FIXED_TRAJECTORIES","Three recipient episodes were chosen deterministically by hashing episode IDs before selecting trajectories. Curves compare the same frozen model and unchanged recipient state/anchor with matched dynamic, matched static, same-task donor, cross-task donor, and Persistence. Command physical units are not asserted.")
    wc(DATA/"FIG05A_FIXED_TRAJECTORIES.csv",rows)
    # Post-hoc largest episode-average Hand errors; illustrative failure cases, not inferential examples.
    rows=[];fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for col,(ep,row,local_idx,selection) in enumerate(failures):
        aid=str(ids[row]);bundle_idx=local[aid];target=targets[row];persist=anchors[row]
        best_joint=14+int(np.argmax(np.abs(loaded["MATCHED-D"][bundle_idx,:,14:]-target[:,14:]).mean(axis=0)))
        series=[("GT",target[:,best_joint],"#111111","-"),("Persistence",np.full(60,persist[best_joint]),colors["PERSISTENCE"],":")]
        for c in conditions:series.append((format_condition(c),loaded[c][bundle_idx,:,best_joint],colors[c],"--" if c=="MATCHED-S" else "-"))
        for label,val,color,style in series:
            axs[0,col].plot(np.arange(60),val,label=label,color=color,linestyle=style,linewidth=1.1 if label!="GT" else 1.8)
            if label!="GT":axs[1,col].plot(np.arange(60),val,label=label,color=color,linestyle=style,linewidth=1.1 if label!="Persistence" else 1.4)
            for t,v in enumerate(val):rows.append({"selection":selection,"episode_index":ep,"anchor_id":aid,"joint_index":best_joint,"step":t,"series":label,"command_value":float(v)})
        pred_values=np.concatenate([np.asarray(v) for label,v,_,_ in series if label!="GT"]);span=max(float(np.ptp(pred_values)),.01);center=float(np.mean(pred_values))
        axs[1,col].set_ylim(center-.8*span,center+.8*span)
        axs[0,col].set_title(f"Post-hoc high Hand error · ep {ep} · j{best_joint}");axs[1,col].set_xlabel("Future step")
        axs[0,col].grid(alpha=.2);axs[1,col].grid(alpha=.2);axs[1,col].text(.03,.95,"GT outside zoom",transform=axs[1,col].transAxes,va="top",fontsize=7)
    axs[0,0].set_ylabel("Full command range");axs[1,0].set_ylabel("Prediction-only zoom")
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=6,loc="upper center",bbox_to_anchor=(.5,1.02),frameon=False)
    fig.suptitle("Failure illustrations selected after scoring (not confirmatory)",y=1.06,fontweight="bold");fig.tight_layout(rect=(0,0,1,.92));save(fig,"FIG05B_POSTHOC_FAILURE_EXAMPLES","These three examples were selected post hoc as the highest episode-average Hand MAE under matched dynamic video for one development seed. Top row shows targets and predictions on the full range; bottom row zooms to prediction variation only so overlapping alternatives remain visible and omits the out-of-range target. Illustrative only, not inferential.")
    wc(DATA/"FIG05B_POSTHOC_FAILURE_EXAMPLES.csv",rows)
    wc(DATA/"FIG05_EXAMPLE_SELECTION.csv",[{"selection":x[3],"episode_index":x[0],"anchor_id":str(ids[x[1]]),"anchor_frame":int(np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False)["anchor_frame"][x[1]])} for x in fixed+failures])
def figure6():
    primary=rd(ROOT/"analysis/PRIMARY_CORRESPONDENCE_EFFECTS.csv");tasks=rd(ROOT/"analysis/TASK_SUPPORT_SUMMARY.csv");joints=rd(ROOT/"analysis/JOINT_SUPPORT_SUMMARY.csv")
    rows=[]
    rows.append({"claim":"Episode-specific correspondence","evidence":"All same-task-donor vs MATCHED-D Arm/Hand effects are positive; crossed CIs exclude zero; conditional Holm p < 0.001.","boundary":"Conditioned on locked dev donor maps; not a physical causal intervention."})
    rows.append({"claim":"Dynamic multi-frame video beyond repeated-last-frame static","evidence":"All MATCHED-S vs MATCHED-D Arm/Hand MAE comparisons are positive in ANCHOR and ROBUST_STRESS; conditional Holm p < 0.001.","boundary":"Supports a multi-frame signal; does not prove raw-frame temporal-order understanding."})
    rows.append({"claim":"Breadth across tasks/joints","evidence":"93 tasks and 14 Arm/14 Hand joints are tabulated; per-task outcomes are descriptive, not individually tested.","boundary":"Task imbalance and development reuse remain."})
    rows.append({"claim":"Cross-task donor degradation","evidence":"Cross-task donors have larger feature distance and can reduce baseline-relative skill.","boundary":"OOD feature shift is a competing explanation; not a clean task-causal contrast."})
    rows.append({"claim":"Real network failure recovery / final generalization","evidence":"Not tested.","boundary":"No physical intervention, held-out/final access, or real network delay."})
    wc(DATA/"FIG06_CLAIM_EVIDENCE.csv",rows)
    fig,ax=plt.subplots(figsize=(13,6.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    from matplotlib.patches import Rectangle
    widths=(.23,.42,.35);xs=(0,.23,.65);headers=("Claim","Evidence","Interpretation boundary")
    fig.suptitle("Evidence ledger: what the fixed-checkpoint intervention does and does not show",y=.985,fontweight="bold")
    header_y=.88;header_h=.08
    for j,title in enumerate(headers):
        ax.add_patch(Rectangle((xs[j],header_y),widths[j],header_h,facecolor="#1B6CA8",edgecolor="#D0D7DE"))
        ax.text(xs[j]+.012,header_y+header_h/2,title,color="white",weight="bold",va="center",fontsize=8.2)
    y=header_y
    for i,r in enumerate(rows):
        wrapped=(textwrap.fill(r["claim"],24),textwrap.fill(r["evidence"],52),textwrap.fill(r["boundary"],45))
        nlines=max(x.count("\n")+1 for x in wrapped);height=.035*nlines+.045;y-=height
        face="#F1F5F8" if i%2 else "#FFFFFF"
        for j,value in enumerate(wrapped):
            ax.add_patch(Rectangle((xs[j],y),widths[j],height,facecolor=face,edgecolor="#D0D7DE"))
            ax.text(xs[j]+.012,y+height-.018,value,va="top",fontsize=7.8,linespacing=1.2)
    save(fig,"FIG06_CLAIM_EVIDENCE","Summary of the predeclared claim boundaries. The primary evidence supports fixed-model use of correctly matched dynamic episode video over same-task donor and repeated-last-frame static tokens on this development population; it does not establish physical causation, raw-frame order understanding, network robustness, or held-out generalization.")

def main():
    if json.loads((ROOT/"analysis/ANALYSIS_STATUS.json").read_text())["status"]!="PASS":raise RuntimeError("analysis gate is not PASS")
    figure1();figure2();figure3();figure4();figure5();figure6()
    manifest={"status":"GENERATED_PENDING_VISUAL_QA","figure_groups":6,"figure_files":14,"formats":["PNG","PDF"],"captions":str(CAP),"data_tables":str(DATA),"final_test_access":0}
    (ROOT/"figure_manifests/FIGURE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
