#!/usr/bin/env python3
"""Create condition, donor, and support summaries from accepted analysis tables."""
from __future__ import annotations
import os
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z'))
AN=ROOT/"analysis"
def rd(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def wc(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
def wj(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def avg_sd(xs):return float(np.mean(xs)),float(np.std(xs,ddof=1)) if len(xs)>1 else 0.0

def main():
    results=rd(AN/"RESULTS_BY_MODEL_SEED_MAP_DELAY.csv")
    metrics=("arm_mae_std_episode_equal_mean","hand_mae_std_episode_equal_mean","full_mae_std_episode_equal_mean","arm_fd_native_episode_equal_mean","hand_fd_native_episode_equal_mean","full_fd_native_episode_equal_mean","full_tv_ratio_episode_equal_mean","arm_persistence_skill","hand_persistence_skill","full_persistence_skill")
    condition_types={"MATCHED-D":["MATCHED-D"],"MATCHED-S":["MATCHED-S"],"SAME-TASK":["SAME-TASK-DONOR-D-M1","SAME-TASK-DONOR-D-M2","SAME-TASK-DONOR-D-M3"],"CROSS-TASK":["CROSS-TASK-DONOR-D-M1","CROSS-TASK-DONOR-D-M2","CROSS-TASK-DONOR-D-M3"]}
    specs=(("ANCHOR","SIV",(20260953,20260954,20260955),(None,)),("ROBUST_STRESS","DELAY",(20260956,20260957,20260958),(.25,.5,1.0)))
    condition_summary=[];delay_summary=[]
    for kind,family,seeds,delays in specs:
        for label,conds in condition_types.items():
            seed_values=[]
            for seed in seeds:
                rows=[r for r in results if r["family"]==family and int(r["model_seed"])==seed and r["condition"] in conds and ((delays==(None,) and r["family"]=="SIV") or (delays!=(None,) and float(r["delay_seconds"]) in delays))]
                if not rows:raise RuntimeError(f"missing summary rows {kind}/{label}/{seed}")
                seed_values.append({m:float(np.mean([float(r[m]) for r in rows])) for m in metrics})
            out={"context_group":kind,"condition":label,"seed_count":len(seeds)}
            for m in metrics:
                mu,sd=avg_sd([x[m] for x in seed_values]);out[m+"_mean"]=mu;out[m+"_seed_sd"]=sd
            condition_summary.append(out)
        if kind=="ROBUST_STRESS":
            for delay in (0.0,.25,.5,1.0):
                for label,conds in condition_types.items():
                    seed_values=[]
                    for seed in seeds:
                        rows=[r for r in results if r["family"]=="DELAY" and int(r["model_seed"])==seed and float(r["delay_seconds"])==delay and r["condition"] in conds]
                        seed_values.append({m:float(np.mean([float(r[m]) for r in rows])) for m in metrics})
                    out={"context_group":"ROBUST_SINGLE_DELAY","delay_seconds":delay,"condition":label,"seed_count":3}
                    for m in metrics:
                        mu,sd=avg_sd([x[m] for x in seed_values]);out[m+"_mean"]=mu;out[m+"_seed_sd"]=sd
                    delay_summary.append(out)
    wc(AN/"CONDITION_SUMMARY.csv",condition_summary);wc(AN/"CONDITION_BY_DELAY.csv",delay_summary)

    donor_maps=rd(ROOT/"DONOR_WINDOW_MAPS.csv");quality=[]
    for kind in ("same_task","cross_task"):
        for seed in (202609701,202609702,202609703):
            rows=[r for r in donor_maps if r["pairing_type"]==kind and int(r["map_seed"])==seed]
            unique_anchor=len({r["donor_anchor_id"] for r in rows});use=defaultdict(int)
            for r in rows:use[r["donor_anchor_id"]]+=1
            ts=np.asarray([float(r["absolute_anchor_timestamp_diff"]) for r in rows]);cmd=np.asarray([float(r["past_command_distance_scaled_mae"]) for r in rows]);tok=np.asarray([float(r["matched_dynamic_token_mean_abs_diff"]) for r in rows])
            quality.append({"pairing_type":kind,"map_seed":seed,"recipient_episodes":len({r["recipient_episode"] for r in rows}),"recipient_windows":len(rows),"unique_donor_episodes":len({r["donor_episode"] for r in rows}),"unique_donor_windows":unique_anchor,"donor_window_reuse_fraction":float(1-unique_anchor/len(rows)),"max_donor_window_reuse":max(use.values()),"absolute_anchor_time_diff_median":float(np.median(ts)),"absolute_anchor_time_diff_p95":float(np.quantile(ts,.95)),"past_command_distance_mean":float(cmd.mean()),"past_command_distance_p95":float(np.quantile(cmd,.95)),"dynamic_token_distance_mean":float(tok.mean()),"dynamic_token_distance_median":float(np.median(tok)),"dynamic_token_distance_p95":float(np.quantile(tok,.95))})
    wc(AN/"DONOR_SUPPORT_SUMMARY.csv",quality)

    seedmap=rd(AN/"SEED_MAP_SENSITIVITY.csv");support=[]
    groups=defaultdict(list)
    for r in seedmap:groups[(r["context_group"],r["contrast"],r["group"],r["alternative"])].append(r)
    for key,rows in groups.items():
        effects=[float(r["episode_mean_effect"]) for r in rows]
        support.append({"context_group":key[0],"contrast":key[1],"group":key[2],"alternative":key[3],"seed_map_rows":len(rows),"positive_seed_map_effect_rows":sum(x>0 for x in effects),"effect_min":min(effects),"effect_max":max(effects),"episode_fraction_alt_worse_min":min(float(r["episode_fraction_alternative_worse"]) for r in rows),"episode_fraction_alt_worse_max":max(float(r["episode_fraction_alternative_worse"]) for r in rows)})
    wc(AN/"SEED_MAP_SUPPORT_SUMMARY.csv",support)
    tasks=rd(AN/"PER_TASK_SUPPORT.csv");task_summary=[]
    groups=defaultdict(list)
    for r in tasks:groups[(r["context_group"],r["contrast"],r["group"],r["alternative"])].append(r)
    for key,rows in groups.items():
        effects=[float(r["mean_effect_alt_minus_matched"]) for r in rows]
        task_summary.append({"context_group":key[0],"contrast":key[1],"group":key[2],"alternative":key[3],"task_count":len(rows),"positive_task_count":sum(x>0 for x in effects),"positive_task_fraction":float(np.mean(np.asarray(effects)>0)),"task_effect_min":min(effects),"task_effect_max":max(effects)})
    wc(AN/"TASK_SUPPORT_SUMMARY.csv",task_summary)
    joints=rd(AN/"PER_JOINT_SUPPORT.csv");joint_summary=[]
    groups=defaultdict(list)
    for r in joints:groups[(r["context_group"],r["contrast"],r["group"],r["alternative"])].append(r)
    for key,rows in groups.items():
        js=[int(r["joint_index"]) for r in rows];effects=[float(r["mean_effect_alt_minus_matched"]) for r in rows]
        joint_summary.append({"context_group":key[0],"contrast":key[1],"group":key[2],"alternative":key[3],"joint_count":len(rows),"joint_index_min":min(js),"joint_index_max":max(js),"positive_joint_count":sum(x>0 for x in effects),"joint_effect_min":min(effects),"joint_effect_max":max(effects)})
    wc(AN/"JOINT_SUPPORT_SUMMARY.csv",joint_summary)

    primary=rd(AN/"PRIMARY_CORRESPONDENCE_EFFECTS.csv");fd=rd(AN/"FD_EFFECTS.csv");cross=rd(AN/"CROSS_TASK_DIAGNOSTIC.csv")
    distinct=rd(AN/"PER_EPISODE_RESULTS.csv")
    persistence={}
    for group,metric in (("Arm","arm_mae_std"),("Hand","hand_mae_std"),("Full","full_mae_std")):
        seen={}
        for r in distinct:
            seen[int(r["episode_index"])]=float(r["persistence_"+metric])
        persistence[group]={"mean":float(np.mean(list(seen.values()))),"episodes":len(seen)}
    payload={"status":"PASS","condition_summary_rows":condition_summary,"single_delay_rows":delay_summary,"donor_support":quality,"seed_map_support":support,"task_support":task_summary,"joint_support":joint_summary,"persistence_subset":persistence,"primary_effects":primary,"fd_effects":fd,"cross_task_diagnostic":cross,"final_test_access":0}
    wj(ROOT/"INTERPRETATION_DATA.json",payload)
    print(json.dumps({"status":"PASS","condition_rows":len(condition_summary),"delay_rows":len(delay_summary),"donor_map_rows":len(quality),"task_support_groups":len(task_summary),"joint_support_groups":len(joint_summary),"persistence_episodes":persistence["Full"]["episodes"],"final_test_access":0},ensure_ascii=False,sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
