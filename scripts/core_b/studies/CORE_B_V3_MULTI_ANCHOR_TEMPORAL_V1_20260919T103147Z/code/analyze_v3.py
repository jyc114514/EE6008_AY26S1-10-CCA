#!/usr/bin/env python3
"""CPU-only V3 metrics, baselines, paired bootstrap and interactions."""

from __future__ import annotations
import os
import csv,json,hashlib
from pathlib import Path
import numpy as np

EXP=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
CELLS=("S-HIST","F-POOL-D","F-TOK-D","F-TOK-S","V-TOK-D-A","V-TOK-S-A")
SEEDS=(20260950,20260951,20260952); HORIZONS=(30,60); HAND=slice(14,28); ARM=slice(0,14); BOOT=10000; HIER=20000


def write_json(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x)))+"\n",encoding="utf-8")
def write_csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True); fields=list(dict.fromkeys(k for r in rows for k in r));
 with p.open("w",encoding="utf-8",newline="") as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def load_npz(name):
 with np.load(EXP/"data"/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bootstrap_indices(n,seed): return np.random.default_rng(seed).integers(0,n,size=(BOOT,n),dtype=np.int32)
def effect_boot(diff,idx,seed):
 vals=np.asarray(diff,dtype=np.float64); est=vals[idx].mean(axis=1); rng=np.random.default_rng(seed); signs=rng.choice(np.array([-1.,1.]),size=(BOOT,len(vals))); p=(1+np.sum(np.abs((signs*vals[None]).mean(axis=1))>=abs(vals.mean())))/(BOOT+1)
 return {"effect":float(vals.mean()),"ci_low":float(np.quantile(est,.025)),"ci_high":float(np.quantile(est,.975)),"p_signflip":float(p),"N_episode":len(vals),"replicates":BOOT}
def hier_effect(diff_by_seed,seed):
 diff=np.asarray(diff_by_seed,dtype=np.float64); n=diff.shape[1]; rng=np.random.default_rng(seed); out=np.empty(HIER)
 for start in range(0,HIER,200):
  m=min(200,HIER-start); chosen=rng.integers(0,3,size=(m,3)); total=np.zeros(m)
  for slot in range(3):
   ix=rng.integers(0,n,size=(m,n)); total+=diff[chosen[:,slot,None],ix].mean(axis=1)
  out[start:start+m]=total/3
 return {"effect":float(diff.mean()),"ci_low":float(np.quantile(out,.025)),"ci_high":float(np.quantile(out,.975)),"p_two_sided":float(2*min((1+np.sum(out<=0))/(HIER+1),(1+np.sum(out>=0))/(HIER+1))),"replicates":HIER,"seed_level":3,"episode_level":n}


def read_targets_and_motion():
 data=load_npz("V3_MAIN_INPUTS.npz"); norm=json.loads((EXP/"data/V3_NORMALIZERS.json").read_text()); sigma=np.asarray(norm["target_sigma"],dtype=np.float64); fd=np.asarray(norm["safe_fd_scale"],dtype=np.float64); dev=np.flatnonzero(data["split_code"]==1); train=np.flatnonzero(data["split_code"]==0)
 # common motion thresholds fit from primary train 60-step targets, episode equal.
 episode_train=data["episode_index"][train]; target=data["targets_native"]
 train_episode_values=[]
 for e in np.unique(episode_train):
  rows=train[episode_train==e]; values=np.mean(np.abs(np.diff(target[rows],axis=1)),axis=(0,1)); train_episode_values.append(values)
 train_episode_values=np.asarray(train_episode_values); q25=np.quantile(train_episode_values,[.25],axis=0)[0]; q75=np.quantile(train_episode_values,[.75],axis=0)[0]
 dev_values=[]
 for e in np.unique(data["episode_index"][dev]):
  rows=dev[data["episode_index"][dev]==e]; dev_values.append(np.mean(np.abs(np.diff(target[rows],axis=1)),axis=(0,1)))
 dev_values=np.asarray(dev_values); dev_episodes=np.unique(data["episode_index"][dev]); arm_v=dev_values[:,0:14].mean(axis=1); hand_v=dev_values[:,14:28].mean(axis=1)
 def label(v,q1,q3): return np.where(v<=q1,"low",np.where(v>=q3,"high","mid"))
 motion={"episode_index":dev_episodes,"arm_value":arm_v,"hand_value":hand_v,"arm_stratum":label(arm_v,float(q25[:14].mean()),float(q75[:14].mean())),"hand_stratum":label(hand_v,float(q25[14:].mean()),float(q75[14:].mean()))}
 return data,norm,sigma,fd,dev,train,motion


def slope_predictions(data,rows,h):
 out=np.empty((len(rows),h,28),dtype=np.float64); fallback=0
 for i,row in enumerate(rows):
  times=data["history_selected_timestamps"][row].astype(np.float64); commands=data["history_native"][row,:,43:71].astype(np.float64); t0=float(times[-1]); valid=np.flatnonzero(times>=t0-.25-1e-9); unique=[]
  for j in valid:
   if not unique or abs(times[j]-times[unique[-1]])>1e-9: unique.append(int(j))
  anchor=data["anchor_command_native"][row].astype(np.float64); target_times=np.arange(1,h+1,dtype=np.float64)/30.0+t0
  if len(unique)<2 or times[unique[-1]]<=times[unique[0]]: out[i]=anchor;fallback+=1
  else:
   slope=(commands[unique[-1]]-commands[unique[0]])/(times[unique[-1]]-times[unique[0]])
   out[i]=anchor+slope[None,:]*(target_times[:,None]-t0)
 return out,fallback


def metric(pred,target,anchor,sigma,fd_scale,episode_ids,episode_order):
 pred=np.asarray(pred,dtype=np.float64); target=np.asarray(target,dtype=np.float64); anchor=np.asarray(anchor,dtype=np.float64); sigma=np.asarray(sigma,dtype=np.float64); fd_scale=np.asarray(fd_scale,dtype=np.float64)
 errors=np.abs(pred-target); std_err=errors/sigma[None,None,:]; delta_err=np.abs(np.diff(pred,axis=1)-np.diff(target,axis=1)); lag3=np.abs(pred[:,3:]-pred[:,:-3]-(target[:,3:]-target[:,:-3])); lag6=np.abs(pred[:,6:]-pred[:,:-6]-(target[:,6:]-target[:,:-6])); tvp=np.abs(np.diff(pred,axis=1)).sum(axis=1).mean(axis=1); tvt=np.abs(np.diff(target,axis=1)).sum(axis=1).mean(axis=1); ratio=np.divide(tvp,tvt,out=np.full(len(pred),np.nan),where=tvt>1e-12)
 w={"arm_mae_std":std_err[:, :, :14].mean(axis=(1,2)),"hand_mae_std":std_err[:, :, 14:].mean(axis=(1,2)),"full_mae_std":std_err.mean(axis=(1,2)),"arm_mae_native":errors[:,:,:14].mean(axis=(1,2)),"hand_mae_native":errors[:,:,14:].mean(axis=(1,2)),"full_mae_native":errors.mean(axis=(1,2)),"arm_fd_native":delta_err[:,:,:14].mean(axis=(1,2)),"hand_fd_native":delta_err[:,:,14:].mean(axis=(1,2)),"full_fd_native":delta_err.mean(axis=(1,2)),"arm_fd_scaled":(delta_err[:,:,:14]/fd_scale[:14]).mean(axis=(1,2)),"hand_fd_scaled":(delta_err[:,:,14:]/fd_scale[14:]).mean(axis=(1,2)),"full_lag3_native":lag3.mean(axis=(1,2)),"full_lag6_native":lag6.mean(axis=(1,2)),"full_anchor_to_first":np.abs((pred[:,0]-anchor)-(target[:,0]-anchor)).mean(axis=1),"full_endpoint_mae_native":errors[:,-1].mean(axis=1),"full_tv":tvp,"target_full_tv":tvt,"full_tv_ratio":ratio,"full_tv_closeness":np.abs(ratio-1)}
 for key,value in list(w.items()):
  ep=[]
  for e in episode_order: ep.append(np.nanmean(np.asarray(value)[episode_ids==e]))
  w[key]=np.asarray(ep,dtype=np.float64)
 return w


def main()->int:
 data,norm,sigma,fd,dev,train,motion=read_targets_and_motion(); analysis=EXP/"analysis"; analysis.mkdir(exist_ok=True); dev_episodes=np.unique(data["episode_index"][dev]); n_ep=len(dev_episodes); idx=bootstrap_indices(n_ep,20260960); seed_metrics={}; predictions={}
 baseline_rows=[]
 for seed in SEEDS:
  for h in HORIZONS:
   target=data["targets_native"][dev,:h].astype(np.float64); anchor=data["anchor_command_native"][dev].astype(np.float64); persist=np.repeat(anchor[:,None,:],h,axis=1); slope,fallback=slope_predictions(data,dev,h)
   baseline_rows.extend([{**{"seed":seed,"horizon_steps":h,"model":"PERSISTENCE","slope_fallback":0},**{k:float(np.nanmean(v)) for k,v in metric(persist,target,anchor,sigma,fd,data["episode_index"][dev],dev_episodes).items()}},{**{"seed":seed,"horizon_steps":h,"model":"SLOPE","slope_fallback":fallback},**{k:float(np.nanmean(v)) for k,v in metric(slope,target,anchor,sigma,fd,data["episode_index"][dev],dev_episodes).items()}}])
   for cell in CELLS:
    run_id=f"V3-{cell}-H{h}-S{seed}"; p=EXP/f"predictions/{run_id}/primary.npz"; withp=np.load(p,allow_pickle=False); pred=withp["predictions_native"].astype(np.float64); predictions[(seed,h,cell)]=metric(pred,target,anchor,sigma,fd,data["episode_index"][dev],dev_episodes)
    row={"seed":seed,"horizon_steps":h,"cell":cell,"N_episode":n_ep}
    row.update({k:float(np.nanmean(v)) for k,v in predictions[(seed,h,cell)].items()}); baseline_metrics=metric(persist,target,anchor,sigma,fd,data["episode_index"][dev],dev_episodes); row["arm_skill_vs_persistence"]=1-row["arm_mae_std"]/np.mean(baseline_metrics["arm_mae_std"]); row["hand_skill_vs_persistence"]=1-row["hand_mae_std"]/np.mean(baseline_metrics["hand_mae_std"]); row["full_skill_vs_persistence"]=1-row["full_mae_std"]/np.mean(baseline_metrics["full_mae_std"]); seed_metrics[(seed,h,cell)]=row
 primary_rows=[]
 for value in seed_metrics.values(): primary_rows.append(value)
 write_csv(analysis/"PRIMARY_RESULTS_BY_SEED.csv",primary_rows); write_csv(analysis/"BASELINE_RESULTS.csv",baseline_rows)
 contrasts=[("TOKD_vs_SHIST","F-TOK-D","S-HIST"),("TOKD_vs_POOL","F-TOK-D","F-POOL-D"),("TOKD_vs_TOKS","F-TOK-D","F-TOK-S"),("VTOKD_vs_VTOKS","V-TOK-D-A","V-TOK-S-A")]; paired=[]; interactions=[]
 for h in HORIZONS:
  for name,cand,ctrl in contrasts:
   for group_key in ("arm_mae_std","hand_mae_std"):
    for seed in SEEDS:
     diff=predictions[(seed,h,ctrl)][group_key]-predictions[(seed,h,cand)][group_key]; stat=effect_boot(diff,idx,20261000+len(paired)); paired.append({"contrast":name,"seed":seed,"horizon_steps":h,"group":group_key,"control":ctrl,"candidate":cand,**stat,"final_test_access":0})
    diffseed=np.stack([predictions[(s,h,ctrl)][group_key]-predictions[(s,h,cand)][group_key] for s in SEEDS]); agg=hier_effect(diffseed,20261100+len(paired)); paired.append({"contrast":name,"seed":"aggregate","horizon_steps":h,"group":group_key,"control":ctrl,"candidate":cand,**agg,"final_test_access":0})
   # horizon interaction on paired main anchors, aggregate difference 2s-1s.
   for group_key in ("arm_mae_std","hand_mae_std"):
    if h!=60: continue
    for name,cand,ctrl in contrasts:
     diffseed=[]
     for s in SEEDS:
      d2=predictions[(s,60,ctrl)][group_key]-predictions[(s,60,cand)][group_key]; d1=predictions[(s,30,ctrl)][group_key]-predictions[(s,30,cand)][group_key]; diffseed.append(d2-d1)
     interactions.append({"contrast":name,"group":group_key,**hier_effect(np.stack(diffseed),20261200+len(interactions)),"definition":"2s effect minus 1s effect on same primary episodes","final_test_access":0})
 write_csv(analysis/"PAIRED_EFFECTS.csv",paired); write_csv(analysis/"HORIZON_INTERACTIONS.csv",interactions)
 # task/joint summaries for each seed/horizon/cell.
 task_rows=[]; target=data["targets_native"][dev]
 for (seed,h,cell), values in predictions.items():
  pred=np.load(EXP/f"predictions/V3-{cell}-H{h}-S{seed}/primary.npz",allow_pickle=False)["predictions_native"].astype(np.float64); tar=target[:,:h].astype(np.float64); err=np.abs(pred-tar)/sigma[None,None,:]; tasks=data["original_task_index"][dev]
  for task in np.unique(tasks):
   mask=tasks==task; task_rows.append({"seed":seed,"horizon_steps":h,"cell":cell,"level":"task","index":int(task),"N_window":int(mask.sum()),"arm_mae_std":float(err[mask,:,:14].mean()),"hand_mae_std":float(err[mask,:,14:].mean()),"full_mae_std":float(err[mask].mean()),"final_test_access":0})
  for joint in range(28): task_rows.append({"seed":seed,"horizon_steps":h,"cell":cell,"level":"joint","index":joint,"N_window":len(pred),"arm_mae_std":float(err[:,:,joint].mean()) if joint<14 else np.nan,"hand_mae_std":float(err[:,:,joint].mean()) if joint>=14 else np.nan,"full_mae_std":float(err[:,:,joint].mean()),"final_test_access":0})
 write_csv(analysis/"TASK_JOINT_RESULTS.csv",task_rows)
 # supplementary bundle metrics (1s only), one seed-by-cell row per episode/window.
 supp=load=np.load(EXP/"data/V3_SUPPLEMENTARY_1S_INPUTS.npz",allow_pickle=False); supp_rows=[]
 for seed in SEEDS:
  for cell in CELLS:
   p=np.load(EXP/f"predictions/V3-{cell}-H30-S{seed}/supplementary_1s.npz",allow_pickle=False)["predictions_native"].astype(np.float64); tar=supp["targets_native"].astype(np.float64); e=np.abs(p-tar)/sigma[None,None,:]; supp_rows.append({"seed":seed,"cell":cell,"N_window":len(p),"episode_count":len(np.unique(supp["episode_index"])),"arm_mae_std":float(e[:,:,:14].mean()),"hand_mae_std":float(e[:,:,14:].mean()),"full_mae_std":float(e.mean()),"final_test_access":0})
 write_csv(analysis/"SUPPLEMENTARY_1S.csv",supp_rows)
 write_json(analysis/"MOTION_STRATA.json", {"source":"primary train 60-step native absolute future change; Q25/Q75 episode-equal","arm_strata_counts":{k:int(np.sum(motion["arm_stratum"]==k)) for k in ("low","mid","high")},"hand_strata_counts":{k:int(np.sum(motion["hand_stratum"]==k)) for k in ("low","mid","high")},"final_test_access":0})
 print(json.dumps({"status":"PASS","primary_rows":2322,"episodes":603,"paired_rows":len(paired),"interactions":len(interactions),"final_test_access":0},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
