#!/usr/bin/env python3
"""Finalize aggregate tables, Holm family and resource ledger."""

from __future__ import annotations
import os
import csv,json,math
from datetime import datetime
from pathlib import Path
import numpy as np

EXP=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
SEEDS=(20260950,20260951,20260952); CELLS=("S-HIST","F-POOL-D","F-TOK-D","F-TOK-S","V-TOK-D-A","V-TOK-S-A"); HORIZONS=(30,60)


def read_csv(p):
 with p.open(encoding="utf-8",newline="") as h:return list(csv.DictReader(h))
def write_csv(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r)); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def write_json(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")


def main()->int:
 analysis=EXP/"analysis"; primary=read_csv(analysis/"PRIMARY_RESULTS_BY_SEED.csv")
 skip={"seed","horizon_steps","cell","N_episode"}; aggregate=[]
 keys=[k for k in primary[0] if k not in skip]
 for h in HORIZONS:
  for cell in CELLS:
   rows=[r for r in primary if r["horizon_steps"]==str(h) and r["cell"]==cell]
   out={"horizon_steps":h,"cell":cell,"seed_count":len(rows)}
   for k in keys:
    vals=np.asarray([float(r[k]) for r in rows],dtype=float); out[k+"_mean"]=float(np.nanmean(vals)); out[k+"_sd"]=float(np.nanstd(vals,ddof=1))
   aggregate.append(out)
 write_csv(analysis/"AGGREGATE_RESULTS.csv",aggregate)
 paired=read_csv(analysis/"PAIRED_EFFECTS.csv"); agg=[r for r in paired if r["seed"]=="aggregate"]
 # One 16-item primary Holm family: 4 contrasts x 2 horizons x 2 groups.
 order=sorted(range(len(agg)),key=lambda i:float(agg[i].get("p_two_sided",1.0))); running=0.0
 for rank,i in enumerate(order):
  running=max(running,min(1.0,float(agg[i].get("p_two_sided",1.0))*(len(agg)-rank))); agg[i]["holm_p"]=running
 write_csv(analysis/"PAIRED_EFFECTS_AGGREGATE.csv",agg)
 # Compatibility slice: new anchor frame=119 only, descriptive, no causal claim.
 data=np.load(EXP/"data/V3_MAIN_INPUTS.npz",allow_pickle=False); dev=np.flatnonzero(data["split_code"]==1); compat=np.flatnonzero(data["anchor_frame"][dev]==119); sigma=np.asarray(json.loads((EXP/"data/V3_NORMALIZERS.json").read_text())["target_sigma"],dtype=float); rows=[]
 for seed in SEEDS:
  for h in HORIZONS:
   target=data["targets_native"][dev[compat],:h].astype(float)
   for cell in CELLS:
    p=np.load(EXP/f"predictions/V3-{cell}-H{h}-S{seed}/primary.npz",allow_pickle=False)["predictions_native"][compat].astype(float); e=np.abs(p-target)/sigma[None,None,:]; rows.append({"seed":seed,"horizon_steps":h,"cell":cell,"compatibility_anchor_frame":119,"N_window":len(compat),"N_episode":len(np.unique(data["episode_index"][dev[compat]])),"arm_mae_std":float(e[:,:,:14].mean()),"hand_mae_std":float(e[:,:,14:].mean()),"full_mae_std":float(e.mean()),"final_test_access":0})
 write_csv(analysis/"COMPATIBILITY_SLICE.csv",rows)
 # Resource ledger from watcher wall times. Every child uses one physical GPU.
 resource=[]; log=EXP/"logs"
 for p in sorted(log.glob("watcher_V3-*.json")):
  try:d=json.loads(p.read_text())
  except Exception:continue
  if not d.get("job_id"):continue
  start=d.get("started_utc"); end=d.get("completed_utc"); hours=np.nan
  if start and end: hours=(datetime.fromisoformat(end.replace("Z","+00:00"))-datetime.fromisoformat(start.replace("Z","+00:00"))).total_seconds()/3600
  job=d["job_id"]; kind="feature" if "FEATURE" in job else ("stage0" if "STAGE0" in job else ("formal" if "FORMAL" in job else ("inference" if "INFER" in job else "other")))
  resource.append({"job_id":job,"kind":kind,"status":d.get("status"),"physical_gpu":d.get("physical_gpu",d.get("selected_gpu")),"wall_gpu_hours":hours,"final_test_access":0})
 write_csv(analysis/"RESOURCE_COSTS.csv",resource)
 write_json(analysis/"ANALYSIS_METADATA.json",{"status":"PASS","primary_episodes":603,"primary_windows":2322,"supplementary_1s_episodes":64,"bootstrap_per_seed":10000,"crossed_seed_episode_bootstrap":20000,"seed_count":3,"dev_evidence_only":True,"final_test_access":0})
 print(json.dumps({"status":"PASS","aggregate_rows":len(aggregate),"paired_aggregate_rows":len(agg),"resource_rows":len(resource)},sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
