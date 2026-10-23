#!/usr/bin/env python3
"""Accept all fixed-inference bundles, then compute locked development metrics."""
from __future__ import annotations
import os
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
ROOT=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
V3=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SIV=REPO/"reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
DELAY=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
CONDITIONS=("MATCHED-D","MATCHED-S","SAME-TASK-DONOR-D-M1","SAME-TASK-DONOR-D-M2","SAME-TASK-DONOR-D-M3","CROSS-TASK-DONOR-D-M1","CROSS-TASK-DONOR-D-M2","CROSS-TASK-DONOR-D-M3")
MAP_SEEDS=(202609701,202609702,202609703)
DELAYS=(0.0,.25,.5,1.0)
PRIMARY_BOOT=10000;CROSSED_BOOT=20000;SIGNFLIP_N=10000

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def readcsv(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def writecsv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise RuntimeError(f"refusing empty table: {path}")
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def writejson(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def scalar(z,key):return np.asarray(z[key]).item()
def delay_context(family,seed,delay):
    if family=="SIV":return f"SIV-ANCHOR-D-H60-S{seed}"
    suffix={0.0:"D0",.25:"D0.25",.5:"D0.5",1.0:"D1"}[float(delay)]
    return f"ODR-ROBUST-D-H60-S{seed}-{suffix}"
def condition_pair_type(condition):
    if condition.startswith("SAME-TASK"):return "same_task"
    if condition.startswith("CROSS-TASK"):return "cross_task"
    return None
def expected_map_seed(condition):
    if condition in ("MATCHED-D","MATCHED-S"):return -1
    return MAP_SEEDS[int(condition.rsplit("M",1)[1])-1]

def accept_all():
    q=json.loads((ROOT/"logs/FORMAL_QUEUE_STATUS.json").read_text(encoding="utf-8"))
    if q.get("status")!="PASS" or q.get("completed_count")!=15 or q.get("final_test_access")!=0 or q.get("max_active_watchers",0)>2:
        raise RuntimeError("formal watcher queue has not passed 15/15; targets remain unopened")
    manifest=json.loads((ROOT/"configs/FORMAL_JOBS.json").read_text(encoding="utf-8"))
    models={(r["family"],int(r["seed"])):r for r in readcsv(ROOT/"MODEL_MANIFEST.csv")}
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:
        anchor_ids=z["anchor_id"].copy();episodes=z["episode_index"].copy();tasks=z["original_task_index"].copy();split=z["split_code"].copy();frames=z["anchor_frame"].copy();times=z["anchor_timestamp"].copy()
    dev_rows=np.flatnonzero(split==1)
    eligible={int(r["episode_index"]) for r in readcsv(ROOT/"RECIPIENT_ELIGIBILITY.csv") if r["eligible_common_subset"]=="True"}
    subset=np.asarray([i for i in dev_rows if int(episodes[i]) in eligible],dtype=np.int64)
    if len(eligible)!=578 or len(subset)!=2222:raise RuntimeError("locked eligible recipient population mismatch")
    expected_recipient_ids=anchor_ids[subset]
    row_by_id={str(a):int(i) for i,a in enumerate(anchor_ids)}
    map_by={(r["pairing_type"],int(r["map_seed"]),r["recipient_anchor_id"]):r for r in readcsv(ROOT/"DONOR_WINDOW_MAPS.csv")}
    timeline={r["anchor_id"]:r for r in readcsv(V3/"OBSERVATION_TARGET_TIMELINE.csv")}
    groups={int(r["episode_index"]):r["split_assignment_key"] for r in readcsv(V3/"DATA_ALLOWLIST.csv")}
    feature_manifest=json.loads((V3/"cache/CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    extractor=(V3/"code/extract_v3_features.py").read_text(encoding="utf-8")
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:n_main=len(z["anchor_id"])
    with np.load(V3/"data/V3_SUPPLEMENTARY_1S_INPUTS.npz",allow_pickle=False) as z:n_supp=len(z["anchor_id"])
    if feature_manifest["row_count"]!=n_main+n_supp or feature_manifest["shape"][0]!=n_main+n_supp:
        raise RuntimeError("frozen cache primary/supplementary row offset mismatch")
    if 'for name in ("V3_MAIN_INPUTS.npz", "V3_SUPPLEMENTARY_1S_INPUTS.npz")' not in extractor or 'mm[int(row["cache_index"]), 0] = d_values[j]; mm[int(row["cache_index"]), 1] = s_values[j]' not in extractor:
        raise RuntimeError("frozen dynamic/static cache slot or main-first offset contract changed")
    accepted=[];bundle_predictions={}
    contexts={r["context_id"]:r for r in readcsv(ROOT/"EVAL_MATRIX_120.csv")}
    for job in manifest["jobs"]:
        context=job["context_id"];ctx=contexts[context];family=ctx["source_family"];seed=int(ctx["model_seed"])
        run=Path(job["output_dir"]);replay=json.loads((run/"CONTEXT_REPLAY_ACCEPTED.json").read_text(encoding="utf-8"))
        model=models[(family,seed)]
        if replay.get("status")!="PASS" or replay.get("allclose_at_1e-6") is not True or replay.get("rows")!=2322 or replay.get("shape")!=[2322,60,28] or replay.get("state_unchanged") is not True:
            raise RuntimeError(f"full matched-D source replay not accepted: {context}")
        if replay.get("checkpoint_sha256")!=model["checkpoint_sha256"] or replay.get("optimizer_steps")!=0 or replay.get("encoder_forward")!=0 or replay.get("final_test_access")!=0:
            raise RuntimeError(f"replay provenance/zero-work guard failed: {context}")
        delay=ctx["delay_seconds"]
        delay_value=None if delay in ("","None","null") else float(delay)
        if replay.get("delay_seconds")!=delay_value:raise RuntimeError(f"replay delay identity mismatch: {context}")
        for cond in CONDITIONS:
            base=ROOT/f"predictions/{context}";npz=base/f"{cond}.npz";meta_path=base/f"{cond}.json"
            meta=json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status")!="PASS" or meta.get("context_id")!=context or int(meta.get("model_seed",-1))!=seed or meta.get("condition")!=cond or meta.get("checkpoint_sha256")!=model["checkpoint_sha256"] or meta.get("final_test_access")!=0 or meta.get("optimizer_steps")!=0 or meta.get("encoder_forward")!=0:
                raise RuntimeError(f"bundle metadata/provenance failed: {context}/{cond}")
            if meta.get("bundle_sha256")!=digest(npz):raise RuntimeError(f"bundle hash mismatch: {context}/{cond}")
            with np.load(npz,allow_pickle=False) as z:
                pred=z["predictions_native"]
                if pred.shape!=(2222,60,28) or not np.isfinite(pred).all():raise RuntimeError(f"bundle shape/finite failure: {context}/{cond}")
                if not np.array_equal(z["recipient_anchor_id"],expected_recipient_ids):raise RuntimeError(f"recipient order differs: {context}/{cond}")
                if str(scalar(z,"condition"))!=cond:raise RuntimeError(f"condition field differs: {context}/{cond}")
                mseed=expected_map_seed(cond)
                if int(scalar(z,"map_seed"))!=mseed or not np.all(z["donor_map_seed"]==mseed):raise RuntimeError(f"donor map seed differs: {context}/{cond}")
                if float(scalar(z,"requested_delay_seconds"))!=(-1.0 if delay_value is None else delay_value):raise RuntimeError(f"requested delay field differs: {context}/{cond}")
                donor_ids=z["donor_anchor_id"].astype(str)
                if len(donor_ids)!=len(subset):raise RuntimeError(f"donor id length differs: {context}/{cond}")
                if cond in ("MATCHED-D","MATCHED-S"):
                    if not np.array_equal(donor_ids,expected_recipient_ids.astype(str)):raise RuntimeError(f"matched condition is not identity: {context}/{cond}")
                else:
                    kind=condition_pair_type(cond);mseed=expected_map_seed(cond);expected=[]
                    for aid in expected_recipient_ids.astype(str):
                        maprow=map_by[(kind,mseed,aid)];expected.append(maprow["donor_anchor_id"])
                        if maprow["recipient_episode"]==maprow["donor_episode"] or maprow["recipient_group_key"]==maprow["donor_group_key"]:
                            raise RuntimeError(f"same-episode/group donor violation: {context}/{cond}/{aid}")
                        tr=timeline[maprow["donor_anchor_id"]]
                        if tr["split"]!="dev" or tr["population"]!="primary" or tr["past_le_anchor"]!="True" or int(tr["past_end_frame"])>int(tr["anchor_frame"]):
                            raise RuntimeError(f"donor source includes non-past/non-primary frame: {context}/{cond}/{aid}")
                        if int(maprow["donor_row"])>=n_main:raise RuntimeError("donor row points into supplementary cache offset")
                    if not np.array_equal(donor_ids,np.asarray(expected,dtype=str)):raise RuntimeError(f"saved donor IDs differ from locked map: {context}/{cond}")
                bundle_predictions[(context,cond)]=npz
            accepted.append({"bundle_id":meta["bundle_id"],"context_id":context,"family":family,"model_seed":seed,"delay_seconds":delay_value,"condition":cond,"map_seed":expected_map_seed(cond),"shape":"2222x60x28","bundle_path":str(npz),"bundle_sha256":meta["bundle_sha256"],"checkpoint_sha256":meta["checkpoint_sha256"],"final_test_access":0})
    if len(accepted)!=120 or len(bundle_predictions)!=120:raise RuntimeError(f"expected 120 accepted bundles, got {len(accepted)}")
    acceptance={"status":"PASS","accepted_bundles":120,"expected_bundles":120,"contexts":15,"source_checkpoints":6,"source_replays_allclose_at_1e-6":True,"eligible_episodes":578,"eligible_windows":2222,"donor_same_episode_violations":0,"donor_duplicate_group_violations":0,"donor_future_frame_violations":0,"primary_cache_rows":n_main,"supplementary_cache_rows":n_supp,"dynamic_cache_slot":0,"static_cache_slot":1,"optimizer_steps_total":0,"encoder_forward_total":0,"final_test_access":0,"bundle_index":accepted}
    writejson(ROOT/"BUNDLE_ACCEPTANCE.json",acceptance);writecsv(ROOT/"analysis/ACCEPTED_BUNDLE_INDEX.csv",accepted)
    return manifest,contexts,subset,expected_recipient_ids,episodes[subset],tasks[subset],bundle_predictions,acceptance

def compute_episode_metrics(pred,target,sigma,row_eps,episode_ids):
    abs_std=np.abs(pred-target)/sigma.reshape(1,1,-1)
    delta_pred=np.diff(pred,axis=1);delta_target=np.diff(target,axis=1)
    abs_fd=np.abs(delta_pred-delta_target)
    tv_pred=np.abs(delta_pred).sum(axis=(1,2));tv_target=np.abs(delta_target).sum(axis=(1,2))
    window={"arm_mae_std":abs_std[:,:,:14].mean(axis=(1,2)),"hand_mae_std":abs_std[:,:,14:].mean(axis=(1,2)),"full_mae_std":abs_std.mean(axis=(1,2)),
        "arm_fd_native":abs_fd[:,:,:14].mean(axis=(1,2)),"hand_fd_native":abs_fd[:,:,14:].mean(axis=(1,2)),"full_fd_native":abs_fd.mean(axis=(1,2)),
        "arm_endpoint_mae_std":(np.abs(pred[:,-1,:14]-target[:,-1,:14])/sigma[:14]).mean(axis=1),
        "hand_endpoint_mae_std":(np.abs(pred[:,-1,14:]-target[:,-1,14:])/sigma[14:]).mean(axis=1)}
    results={k:np.empty(len(episode_ids),dtype=np.float64) for k in window}
    tv_ratio=np.empty(len(episode_ids),dtype=np.float64);joint=np.empty((len(episode_ids),28),dtype=np.float64);joint_fd=np.empty((len(episode_ids),28),dtype=np.float64)
    for j,ep in enumerate(episode_ids):
        ix=np.flatnonzero(row_eps==ep)
        for k,v in window.items():results[k][j]=float(v[ix].mean())
        tv_ratio[j]=float(tv_pred[ix].sum()/max(tv_target[ix].sum(),1e-12))
        joint[j]=abs_std[ix].mean(axis=(0,1))
        joint_fd[j]=abs_fd[ix].mean(axis=(0,1))
    results["full_tv_ratio"]=tv_ratio
    return results,joint,joint_fd

def holm(pvalues):
    p=np.asarray(pvalues,dtype=np.float64);order=np.argsort(p);adj=np.empty_like(p);running=0.0;m=len(p)
    for rank,i in enumerate(order):
        running=max(running,min(1.0,(m-rank)*p[i]));adj[i]=running
    return adj

def bootstrap_stats(matrix,denominator,seed_offset):
    # matrix and denominator are seed×episode paired arrays.
    matrix=np.asarray(matrix,dtype=np.float64);denominator=np.asarray(denominator,dtype=np.float64)
    nseed,nep=matrix.shape
    rel=matrix/np.maximum(denominator,1e-12)
    ep=matrix.mean(axis=0);ep_rel=rel.mean(axis=0)
    rng=np.random.default_rng(202609801+seed_offset)
    eb=np.empty(PRIMARY_BOOT);erb=np.empty(PRIMARY_BOOT)
    for b in range(PRIMARY_BOOT):
        ix=rng.integers(0,nep,size=nep);eb[b]=ep[ix].mean();erb[b]=ep_rel[ix].mean()
    rng=np.random.default_rng(202609802+seed_offset)
    cb=np.empty(CROSSED_BOOT);crb=np.empty(CROSSED_BOOT);batch=256
    for start in range(0,CROSSED_BOOT,batch):
        k=min(batch,CROSSED_BOOT-start);si=rng.integers(0,nseed,size=(k,nseed));ei=rng.integers(0,nep,size=(k,nep))
        cb[start:start+k]=matrix[si[:,:,None],ei[:,None,:]].mean(axis=(1,2))
        crb[start:start+k]=rel[si[:,:,None],ei[:,None,:]].mean(axis=(1,2))
    rng=np.random.default_rng(202609803+seed_offset)
    observed=float(ep.mean());sign=np.empty(SIGNFLIP_N);chunk=512
    for start in range(0,SIGNFLIP_N,chunk):
        k=min(chunk,SIGNFLIP_N-start);sgn=rng.integers(0,2,size=(k,nep),dtype=np.int8)*2-1
        sign[start:start+k]=(sgn*ep).mean(axis=1)
    p_sign=(np.sum(np.abs(sign)>=abs(observed))+1)/(SIGNFLIP_N+1)
    p_cross=min(1.0,2*min((np.sum(cb<=0)+1)/(CROSSED_BOOT+1),(np.sum(cb>=0)+1)/(CROSSED_BOOT+1)))
    return {"effect":float(matrix.mean()),"relative_effect":float(rel.mean()),
        "episode_ci_low":float(np.quantile(eb,.025)),"episode_ci_high":float(np.quantile(eb,.975)),
        "episode_relative_ci_low":float(np.quantile(erb,.025)),"episode_relative_ci_high":float(np.quantile(erb,.975)),
        "crossed_ci_low":float(np.quantile(cb,.025)),"crossed_ci_high":float(np.quantile(cb,.975)),
        "crossed_relative_ci_low":float(np.quantile(crb,.025)),"crossed_relative_ci_high":float(np.quantile(crb,.975)),
        "p_crossed_bootstrap":float(p_cross),"p_signflip_conditional":float(p_sign),
        "episode_bootstrap_replicates":PRIMARY_BOOT,"crossed_bootstrap_replicates":CROSSED_BOOT,"signflip_replicates":SIGNFLIP_N,
        "n_seeds":nseed,"n_episodes":nep}

def main():
    manifest,contexts,subset,recipient_ids,sub_eps,sub_tasks,bundle_paths,acceptance=accept_all()
    # This is the first point in the workflow at which development targets are opened.
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:
        target=z["targets_native"][subset].astype(np.float32);anchor=z["anchor_command_native"][subset].astype(np.float32)
        sigma=np.asarray(json.loads((V3/"data/V3_NORMALIZERS.json").read_text())["target_sigma"],dtype=np.float32)
    if target.shape!=(2222,60,28) or anchor.shape!=(2222,28):raise RuntimeError("development target/anchor shape mismatch after acceptance")
    episode_ids=sorted(set(map(int,sub_eps)));episode_windows={ep:int(np.sum(sub_eps==ep)) for ep in episode_ids}
    context_info={r["context_id"]:r for r in readcsv(ROOT/"EVAL_MATRIX_120.csv")}
    values={};joint_mae={};joint_fd={}
    per_episode_rows=[]
    # Fixed persistence baseline, repeated at each episode's unchanged recipient anchor.
    persistence=np.broadcast_to(anchor[:,None,:],(len(anchor),60,28)).copy()
    pmetrics,pjoint,pjfd=compute_episode_metrics(persistence,target,sigma,sub_eps,episode_ids)
    for context in sorted(context_info):
        ctx=context_info[context];family=ctx["source_family"];seed=int(ctx["model_seed"]);delay=ctx["delay_seconds"]
        for cond in CONDITIONS:
            path=bundle_paths[(context,cond)]
            map_seed=expected_map_seed(cond)
            with np.load(path,allow_pickle=False) as z:pred=z["predictions_native"].astype(np.float32)
            met,jm,jf=compute_episode_metrics(pred,target,sigma,sub_eps,episode_ids)
            values[(context,cond,map_seed)]=met;joint_mae[(context,cond,map_seed)]=jm;joint_fd[(context,cond,map_seed)]=jf
            for i,ep in enumerate(episode_ids):
                per_episode_rows.append({"context_id":context,"family":family,"model_seed":seed,"delay_seconds":delay,"condition":cond,"map_seed":map_seed,"episode_index":ep,"task_index":int(next(sub_tasks[k] for k,e in enumerate(sub_eps) if int(e)==ep)),"window_count":episode_windows[ep],
                    **{k:float(v[i]) for k,v in met.items()},"persistence_arm_mae_std":float(pmetrics["arm_mae_std"][i]),"persistence_hand_mae_std":float(pmetrics["hand_mae_std"][i]),"persistence_full_mae_std":float(pmetrics["full_mae_std"][i])})
    writecsv(ROOT/"analysis/PER_EPISODE_RESULTS.csv",per_episode_rows)

    # Summary table keeps each donor map separate; maps are sensitivity rows, not replicates.
    summary=[]
    for (context,cond,mseed),met in values.items():
        ctx=context_info[context];row={"context_id":context,"family":ctx["source_family"],"model_seed":int(ctx["model_seed"]),"delay_seconds":ctx["delay_seconds"],"condition":cond,"map_seed":mseed,"episode_count":len(episode_ids),"window_count":2222}
        for key,val in met.items():row[key+"_episode_equal_mean"]=float(np.mean(val))
        for group,pkey in (("arm","arm_mae_std"),("hand","hand_mae_std"),("full","full_mae_std")):
            row[group+"_persistence_skill"]=float(1-np.mean(met[pkey])/np.mean(pmetrics[pkey]))
        summary.append(row)
    writecsv(ROOT/"analysis/RESULTS_BY_MODEL_SEED_MAP_DELAY.csv",summary)

    def context_keys(kind,seed):
        if kind=="ANCHOR":return [delay_context("SIV",seed,None)]
        return [delay_context("DELAY",seed,d) for d in (.25,.5,1.0)]
    def cond_values(kind,seed,condition_base,metric,map_seed=None):
        keys=context_keys(kind,seed)
        if condition_base=="MATCHED-D":conds=["MATCHED-D"];maps=[-1]
        elif condition_base=="MATCHED-S":conds=["MATCHED-S"];maps=[-1]
        elif condition_base=="SAME-TASK":conds=[f"SAME-TASK-DONOR-D-M{i}" for i in (1,2,3)];maps=list(MAP_SEEDS)
        else:conds=[f"CROSS-TASK-DONOR-D-M{i}" for i in (1,2,3)];maps=list(MAP_SEEDS)
        if map_seed is not None:
            conds=[f"{condition_base}-DONOR-D-M{MAP_SEEDS.index(map_seed)+1}"];maps=[map_seed]
        arr=[]
        for ctx in keys:
            a=[]
            for c,m in zip(conds,maps):a.append(values[(ctx,c,m)][metric])
            arr.append(np.mean(np.stack(a),axis=0))
        return np.mean(np.stack(arr),axis=0)
    def condition_joint(kind,seed,condition_base,group,map_seed=None):
        keys=context_keys(kind,seed)
        if condition_base=="MATCHED-D":conds=["MATCHED-D"];maps=[-1]
        elif condition_base=="MATCHED-S":conds=["MATCHED-S"];maps=[-1]
        elif condition_base=="SAME-TASK":conds=[f"SAME-TASK-DONOR-D-M{i}" for i in (1,2,3)];maps=list(MAP_SEEDS)
        else:conds=[f"CROSS-TASK-DONOR-D-M{i}" for i in (1,2,3)];maps=list(MAP_SEEDS)
        if map_seed is not None:conds=[f"{condition_base}-DONOR-D-M{MAP_SEEDS.index(map_seed)+1}"];maps=[map_seed]
        store=joint_mae if group=="mae" else joint_fd
        bydelay=[]
        for ctx in keys:
            vals=[store[(ctx,c,m)] for c,m in zip(conds,maps)]
            bydelay.append(np.mean(np.stack(vals),axis=0))
        return np.mean(np.stack(bydelay),axis=0)

    seed_family={"ANCHOR":("SIV",(20260953,20260954,20260955)),"ROBUST_STRESS":("DELAY",(20260956,20260957,20260958))}
    effect_results=[];seed_map_rows=[];task_rows=[];joint_rows=[]
    primary_specs=[];secondary_specs=[]
    for kind,(family,seeds) in seed_family.items():
        for alt,contrast in (("SAME-TASK","MATCHED-D_vs_SAME-TASK"),("MATCHED-S","MATCHED-D_vs_MATCHED-S")):
            for group,metric in (("Arm","arm_mae_std"),("Hand","hand_mae_std")):
                mats=[];dens=[]
                for seed in seeds:
                    matched=cond_values(kind,seed,"MATCHED-D",metric)
                    alternative=cond_values(kind,seed,alt,metric)
                    mats.append(alternative-matched);dens.append(matched)
                result={"context_group":kind,"contrast":contrast,"alternative":alt,"group":group,"metric":metric,**bootstrap_stats(np.stack(mats),np.stack(dens),len(primary_specs))}
                primary_specs.append(result)
            for group,metric in (("Arm","arm_fd_native"),("Hand","hand_fd_native")):
                mats=[];dens=[]
                for seed in seeds:
                    matched=cond_values(kind,seed,"MATCHED-D",metric);alternative=cond_values(kind,seed,alt,metric)
                    mats.append(alternative-matched);dens.append(matched)
                secondary_specs.append({"context_group":kind,"contrast":contrast,"alternative":alt,"group":group,"metric":metric,**bootstrap_stats(np.stack(mats),np.stack(dens),8+len(secondary_specs))})
            # Per-seed/map effect sensitivity. No map is treated as a seed or independent observation.
            for seed in seeds:
                matched=cond_values(kind,seed,"MATCHED-D","arm_mae_std")
                for group,metric in (("Arm","arm_mae_std"),("Hand","hand_mae_std")):
                    matched_metric=cond_values(kind,seed,"MATCHED-D",metric)
                    if alt=="MATCHED-S":
                        alternatives=[("MATCHED-S",-1,cond_values(kind,seed,"MATCHED-S",metric))]
                    else:
                        alternatives=[(alt,m,cond_values(kind,seed,alt,metric,map_seed=m)) for m in MAP_SEEDS]
                    for label,mseed,alternative in alternatives:
                        diff=alternative-matched_metric
                        seed_map_rows.append({"context_group":kind,"seed":seed,"contrast":contrast,"alternative":label,"group":group,"map_seed":mseed,"episode_mean_effect":float(diff.mean()),"episode_median_effect":float(np.median(diff)),"episode_fraction_alternative_worse":float(np.mean(diff>0)),"episode_count":len(diff)})
        # Task support for primary Arm/Hand outcomes uses same episode-paired estimand.
    primary_by_key={(r["context_group"],r["contrast"],r["group"]):r for r in primary_specs}
    for result in primary_specs:
        kind=result["context_group"];contrast=result["contrast"];alt=result["alternative"];metric=result["metric"]
        seeds=seed_family[kind][1]
        mats=[];altj=[];matchj=[]
        for seed in seeds:
            matched=cond_values(kind,seed,"MATCHED-D",metric);alternative=cond_values(kind,seed,alt,metric)
            mats.append(alternative-matched)
            altj.append(condition_joint(kind,seed,alt,"mae"));matchj.append(condition_joint(kind,seed,"MATCHED-D","mae"))
        diff=np.stack(mats)
        for task in sorted(set(map(int,sub_tasks))):
            eix=np.asarray([i for i,ep in enumerate(episode_ids) if int(next(sub_tasks[k] for k,e in enumerate(sub_eps) if int(e)==ep))==task],dtype=np.int64)
            if len(eix)==0:continue
            task_effect=diff[:,eix].mean()
            task_rows.append({"context_group":kind,"contrast":contrast,"alternative":alt,"group":result["group"],"task_index":task,"episode_count":len(eix),"window_count":int(sum(episode_windows[episode_ids[i]] for i in eix)),"mean_effect_alt_minus_matched":float(task_effect),"fraction_episodes_alt_worse":float(np.mean(diff[:,eix]>0))})
        joint_ids=range(14) if result["group"]=="Arm" else range(14,28)
        for j in joint_ids:
            dj=np.stack([altj[s][:,j]-matchj[s][:,j] for s in range(3)])
            joint_rows.append({"context_group":kind,"contrast":contrast,"alternative":alt,"group":result["group"],"joint_index":j,"mean_effect_alt_minus_matched":float(dj.mean()),"fraction_episodes_alt_worse":float(np.mean(dj>0)),"episode_count":len(episode_ids)})
    for family_rows in (primary_specs,secondary_specs):
        adj=holm([r["p_signflip_conditional"] for r in family_rows])
        for i,r in enumerate(family_rows):r["holm_p_signflip_conditional"]=float(adj[i]);r["holm_rank_signflip_conditional"]=int(np.argsort([x["p_signflip_conditional"] for x in family_rows]).tolist().index(i)+1)
    writecsv(ROOT/"analysis/PRIMARY_CORRESPONDENCE_EFFECTS.csv",primary_specs)
    writecsv(ROOT/"analysis/FD_EFFECTS.csv",secondary_specs)
    writecsv(ROOT/"analysis/SEED_MAP_SENSITIVITY.csv",seed_map_rows)
    writecsv(ROOT/"analysis/PER_TASK_SUPPORT.csv",task_rows)
    writecsv(ROOT/"analysis/PER_JOINT_SUPPORT.csv",joint_rows)

    # Cross-task is diagnostic only; no Holm claim.
    cross_rows=[]
    for kind,(family,seeds) in seed_family.items():
        for metric in ("arm_mae_std","hand_mae_std","full_mae_std","arm_fd_native","hand_fd_native","full_fd_native","full_tv_ratio"):
            for seed in seeds:
                mat=cond_values(kind,seed,"CROSS-TASK",metric)-cond_values(kind,seed,"MATCHED-D",metric)
                cross_rows.append({"context_group":kind,"metric":metric,"seed":seed,"map_averaged_effect_alt_minus_matched":float(mat.mean()),"episode_fraction_alt_worse":float(np.mean(mat>0)),"episode_count":len(mat),"multiplicity":"diagnostic_unadjusted"})
    writecsv(ROOT/"analysis/CROSS_TASK_DIAGNOSTIC.csv",cross_rows)
    status={"status":"PASS","analysis_lock":str(ROOT/"ANALYSIS_LOCK.json"),"bundle_acceptance":"120/120","primary_family":len(primary_specs),"fd_secondary_family":len(secondary_specs),"holm_applied_to":"conditional seed sign-flip p within each locked family","per_episode_rows":len(per_episode_rows),"seed_map_rows":len(seed_map_rows),"task_support_rows":len(task_rows),"joint_support_rows":len(joint_rows),"training_started":False,"optimizer_steps_total":0,"encoder_forward_total":0,"final_test_access":0}
    writejson(ROOT/"analysis/ANALYSIS_STATUS.json",status)
    print(json.dumps(status,ensure_ascii=False,sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
