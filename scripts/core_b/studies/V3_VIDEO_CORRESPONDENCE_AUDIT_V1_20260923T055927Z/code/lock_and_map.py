#!/usr/bin/env python3
"""Reconcile the delay study and lock outcome-blind episode donor maps."""
from __future__ import annotations
import os
import csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
V3=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SIV=REPO/"reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
DELAY=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
EXP=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
MAP_SEEDS=(202609701,202609702,202609703)
MODEL_SEEDS=(20260953,20260954,20260955)
DELAY_SEEDS=(20260956,20260957,20260958)
DELAYS=(0.0,.25,.5,1.0)
CONDITIONS=("MATCHED-D","MATCHED-S","SAME-TASK-DONOR-D-M1","SAME-TASK-DONOR-D-M2","SAME-TASK-DONOR-D-M3","CROSS-TASK-DONOR-D-M1","CROSS-TASK-DONOR-D-M2","CROSS-TASK-DONOR-D-M3")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def readcsv(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def writecsv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def writejson(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x)))+"\n",encoding="utf-8")
def assignment(ids,tasks,groups,allowed,seed):
    ids=list(map(int,ids)); n=len(ids)
    if n<2:raise RuntimeError("derangement requires at least two episodes")
    rng=np.random.default_rng(seed); cost=rng.random((n,n),dtype=np.float64)
    valid=np.zeros((n,n),dtype=bool)
    for i,recipient in enumerate(ids):
        for j,donor in enumerate(ids):
            valid[i,j]=(recipient!=donor and groups[recipient]!=groups[donor] and allowed(recipient,donor))
    cost[~valid]=1e6
    rr,cc=linear_sum_assignment(cost)
    if len(rr)!=n or not valid[rr,cc].all():raise RuntimeError(f"no perfect donor assignment n={n}")
    return {ids[int(i)]:ids[int(j)] for i,j in zip(rr,cc)}

def main():
    for d in ("analysis","data","configs","figure_data","figure_manifests","logs","predictions","qa","tests"):
        (EXP/d).mkdir(parents=True,exist_ok=True)
    # The old delayed study is read-only; correction is recorded in this new root.
    delay_status=(DELAY/"FINAL_STATUS.ini").read_text(encoding="utf-8")
    delay_rows=readcsv(DELAY/"analysis/AGGREGATE_CURVES.csv")
    primary=readcsv(DELAY/"analysis/PRIMARY_ROBUSTNESS_EFFECTS.csv")
    fd=readcsv(DELAY/"analysis/FD_EFFECTS.csv")
    guard=readcsv(DELAY/"analysis/CLEAN_GUARDRAILS.csv")
    inter=readcsv(DELAY/"analysis/VIDEO_DELAY_INTERACTIONS.csv")
    age=readcsv(DELAY/"analysis/OBSERVATION_AGE_DISTRIBUTION.csv")
    rec_rows=[]
    for r in delay_rows:
        rec_rows.append({"cell":r["cell"],"delay_seconds":r["delay_seconds"],
          "arm_mae_mean":r["arm_mae_std_mean"],"arm_mae_seed_sd":r["arm_mae_std_sd"],
          "hand_mae_mean":r["hand_mae_std_mean"],"hand_mae_seed_sd":r["hand_mae_std_sd"],
          "full_mae_mean":r["full_mae_std_mean"],"full_mae_seed_sd":r["full_mae_std_sd"],
          "arm_fd_native_mean":r["arm_fd_native_mean"],"hand_fd_native_mean":r["hand_fd_native_mean"],
          "full_fd_native_mean":r["full_fd_native_mean"],
          "arm_skill_vs_persistence":1-float(r["arm_mae_std_mean"])/.3638559265323989,
          "hand_skill_vs_persistence":1-float(r["hand_mae_std_mean"])/.2933058894944345})
    writecsv(EXP/"analysis/DELAY_RESULT_RECONCILIATION.csv",rec_rows)
    status_key="VIDEO_DELAY_INTERACTION="
    old_line=next((line for line in delay_status.splitlines() if line.startswith(status_key)),"UNKNOWN")
    corrected="hand interaction is a small unadjusted diagnostic (effect 0.000133934, crossed CI [0.000003903,0.000295013], p=0.044798); the interaction family was not Holm-corrected. Arm interaction effect 0.000116907, CI [-0.000132351,0.000363685], p=0.369082. Do not use the old after_multiplicity wording."
    writejson(EXP/"DELAY_RESULT_RECONCILIATION.json",{"status":"PASS","source_delay_root":str(DELAY),"baseline_full_mae_std":.3285809080134167,"baseline_full_fd_native":.01358671196908519,"primary_family_items":6,"primary_holm_significant":sum(float(r["holm_p"])<.05 for r in primary),"fd_family_items":6,"fd_holm_significant":sum(float(r["holm_p"])<.05 for r in fd),"clean_guardrails":guard,"video_delay_interactions":inter,"effective_age_summary":age,"old_status_line":old_line,"correct_interaction_interpretation":corrected,"final_test_access":0})
    md=["# DELAY study result reconciliation","",
        "Source old root is read-only. Correct interpretation uses the numeric analysis tables, not the erroneous compact status label.","",
        f"Old FINAL_STATUS line: {old_line}",
        "",
        "Corrected interpretation: the Hand interaction is a small unadjusted diagnostic; no Holm correction was specified/applied to this 2-item diagnostic interaction table. Hand effect=0.000133934, crossed CI [0.000003903,0.000295013], p=0.044798. Arm effect=0.000116907, CI [-0.000132351,0.000363685], p=0.369082. This is not a multiplicity-adjusted interaction claim.","",
        "| Cell | Delay(s) | Arm MAE mean±seed SD | Hand MAE mean±seed SD | Arm native FD | Hand native FD | Arm skill | Hand skill |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rec_rows:md.append(f"| {r['cell']} | {r['delay_seconds']} | {float(r['arm_mae_mean']):.5f}±{float(r['arm_mae_seed_sd']):.5f} | {float(r['hand_mae_mean']):.5f}±{float(r['hand_mae_seed_sd']):.5f} | {float(r['arm_fd_native_mean']):.7f} | {float(r['hand_fd_native_mean']):.7f} | {float(r['arm_skill_vs_persistence']):.3f} | {float(r['hand_skill_vs_persistence']):.3f} |")
    md += ["","Full Holm effects and all guardrails are in the JSON and source analysis CSVs. The ROBUST-D normal-input Arm FD relative CI still crosses the +1% practical margin; do not state that every normal-input fidelity guardrail passed.",""]
    (EXP/"DELAY_RESULT_RECONCILIATION.zh-CN.md").write_text("\n".join(md),encoding="utf-8")
    addendum={"status":"CORRECTED_IN_NEW_ROOT","old_root_modified":False,"old_field":old_line,"replacement":"VIDEO_DELAY_INTERACTION=hand_small_positive_unadjusted;arm_null","interpretation":corrected,"source_numeric_file":str(DELAY/"analysis/VIDEO_DELAY_INTERACTIONS.csv"),"final_test_access":0}
    writejson(EXP/"STATUS_INTERPRETATION_ADDENDUM.json",addendum)
    (EXP/"STATUS_INTERPRETATION_ADDENDUM.md").write_text("# Status interpretation addendum\n\nThe sealed DELAY root contains the compact field:\n\n"+old_line+"\n\nThis field is corrected here without changing the sealed root. The Hand interaction is small, positive, and unadjusted; the file has two diagnostic rows and no Holm adjustment. The Arm interval crosses zero. Use the exact values in DELAY_RESULT_RECONCILIATION.md/JSON and the original VIDEO_DELAY_INTERACTIONS.csv.\n",encoding="utf-8")
    # Read only primary development rows/metadata. Targets are never opened during donor-map construction.
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:
        anchor_id=z["anchor_id"].copy(); episode=z["episode_index"].copy(); task=z["original_task_index"].copy()
        split=z["split_code"].copy(); anchor_frame=z["anchor_frame"].copy(); anchor_ts=z["anchor_timestamp"].copy()
        anchor_cmd=z["anchor_command_native"].copy()
    allow={int(x["episode_index"]):x for x in readcsv(V3/"DATA_ALLOWLIST.csv")}
    dev_rows=np.flatnonzero(split==1)
    primary_manifest=[x for x in readcsv(V3/"ANCHOR_MANIFEST.csv") if x["population"]=="primary" and x["split"]=="dev"]
    if len(dev_rows)!=2322 or len({int(episode[r]) for r in dev_rows})!=603 or len(primary_manifest)!=2322:raise RuntimeError("source primary-dev population contract failed")
    eps=sorted(set(int(episode[r]) for r in dev_rows))
    ep_task={e:int(next(task[r] for r in dev_rows if int(episode[r])==e)) for e in eps}
    ep_group={e:str(allow[e]["split_assignment_key"]) for e in eps}
    ep_windows={e:int(np.sum(episode[dev_rows]==e)) for e in eps}
    task_eps=defaultdict(list)
    for e in eps:task_eps[ep_task[e]].append(e)
    singleton_tasks=sorted(t for t,es in task_eps.items() if len(es)<2)
    eligible_tasks=sorted(t for t,es in task_eps.items() if len(es)>=2)
    eligible_eps=sorted(e for t in eligible_tasks for e in task_eps[t])
    elig_set=set(eligible_eps)
    eligible_rows=np.asarray([r for r in dev_rows if int(episode[r]) in elig_set],dtype=np.int64)
    repeated_group=[g for g,n in Counter(ep_group[e] for e in eligible_eps).items() if n>1]
    if repeated_group:raise RuntimeError(f"duplicate/split groups repeat inside eligible dev: {len(repeated_group)}")
    # Build 3 seeded, one-to-one episode derangements of both relation types.
    episode_maps=[];mapping_summaries=[]
    for mi,mseed in enumerate(MAP_SEEDS):
        same={}
        for t in eligible_tasks:
            ids=task_eps[t]
            part=assignment(ids,ep_task,ep_group,lambda a,b:ep_task[a]==ep_task[b],mseed+int(t)*101)
            same.update(part)
        cross=assignment(eligible_eps,ep_task,ep_group,lambda a,b:ep_task[a]!=ep_task[b],mseed+900001)
        for kind,m in (("same_task",same),("cross_task",cross)):
            if len(m)!=len(eligible_eps) or any(a==b or ep_group[a]==ep_group[b] for a,b in m.items()):raise RuntimeError(f"{kind} map invariant failed")
            if kind=="same_task" and any(ep_task[a]!=ep_task[b] for a,b in m.items()):raise RuntimeError("same task map task mismatch")
            if kind=="cross_task" and any(ep_task[a]==ep_task[b] for a,b in m.items()):raise RuntimeError("cross task map task match")
            for recipient,donor in m.items():episode_maps.append({"map_seed":mseed,"map_id":f"M{mi+1}","pairing_type":kind,"recipient_episode":recipient,"donor_episode":donor,"recipient_task":ep_task[recipient],"donor_task":ep_task[donor],"recipient_group_key":ep_group[recipient],"donor_group_key":ep_group[donor],"same_episode":False,"same_duplicate_group":False})
            mapping_summaries.append({"map_seed":mseed,"map_id":f"M{mi+1}","pairing_type":kind,"recipient_episodes":len(m),"unique_donor_episodes":len(set(m.values())),"one_to_one":len(set(m.values()))==len(m),"same_task_fraction":float(np.mean([ep_task[a]==ep_task[b] for a,b in m.items()])),"cross_group_fraction":float(np.mean([ep_group[a]!=ep_group[b] for a,b in m.items()]))})
    writecsv(EXP/"DONOR_MAPS.csv",episode_maps)
    # Per-window donor anchor selection uses only the last known command and time tie-break.
    row_by_ep=defaultdict(list)
    for r in dev_rows:row_by_ep[int(episode[r])].append(int(r))
    sigma=np.asarray(json.loads((V3/"data/V3_NORMALIZERS.json").read_text())["target_sigma"],dtype=np.float64)
    window_maps=[];reuse_summaries=[]
    features=np.load(V3/"cache/V3_TEMPORAL_FEATURES.npy",mmap_mode="r")
    cache_l1={}
    for pair in episode_maps:
        rec=pair["recipient_episode"];don=pair["donor_episode"];mapseed=pair["map_seed"];typ=pair["pairing_type"]
        for rr in row_by_ep[rec]:
            candidates=row_by_ep[don]
            distances=np.mean(np.abs((anchor_cmd[candidates].astype(np.float64)-anchor_cmd[rr].astype(np.float64))/sigma[None,:]),axis=1)
            best=float(np.min(distances));ties=[j for j,v in enumerate(distances) if abs(float(v)-best)<=1e-12]
            if len(ties)>1:
                best_time=min(abs(float(anchor_ts[candidates[j]])-float(anchor_ts[rr])) for j in ties)
                ties=[j for j in ties if abs(abs(float(anchor_ts[candidates[j]])-float(anchor_ts[rr]))-best_time)<=1e-12]
            chosen=min(ties,key=lambda j:str(anchor_id[candidates[j]]));dr=candidates[chosen]
            # Difference is descriptive only; it is not used to choose or reject donors.
            feature_diff=float(np.mean(np.abs(np.asarray(features[rr,0],dtype=np.float32)-np.asarray(features[dr,0],dtype=np.float32))))
            cache_l1.setdefault((mapseed,typ),[]).append(feature_diff)
            window_maps.append({**pair,"recipient_anchor_id":str(anchor_id[rr]),"recipient_row":rr,"recipient_anchor_frame":int(anchor_frame[rr]),"recipient_anchor_timestamp":float(anchor_ts[rr]),"donor_anchor_id":str(anchor_id[dr]),"donor_row":dr,"donor_anchor_frame":int(anchor_frame[dr]),"donor_anchor_timestamp":float(anchor_ts[dr]),"signed_anchor_timestamp_diff":float(anchor_ts[dr]-anchor_ts[rr]),"absolute_anchor_timestamp_diff":float(abs(float(anchor_ts[dr])-float(anchor_ts[rr]))),"past_command_distance_scaled_mae":best,"matched_dynamic_token_mean_abs_diff":feature_diff})
    writecsv(EXP/"DONOR_WINDOW_MAPS.csv",window_maps)
    for summary in mapping_summaries:
        vals=cache_l1[(summary["map_seed"],summary["pairing_type"])]
        summary.update({"recipient_windows":len(vals),"dynamic_token_mean_abs_diff_mean":float(np.mean(vals)),"dynamic_token_mean_abs_diff_median":float(np.median(vals)),"dynamic_token_mean_abs_diff_p95":float(np.quantile(vals,.95))})
    writecsv(EXP/"DONOR_MATCH_QUALITY.csv",mapping_summaries)
    diversity=[]
    for typ in ("same_task","cross_task"):
        maps={seed:{int(r["recipient_episode"]):int(r["donor_episode"]) for r in episode_maps if int(r["map_seed"])==seed and r["pairing_type"]==typ} for seed in MAP_SEEDS}
        for i in range(len(MAP_SEEDS)):
            for j in range(i+1,len(MAP_SEEDS)):
                left,right=MAP_SEEDS[i],MAP_SEEDS[j]
                agreement=float(np.mean([maps[left][e]==maps[right][e] for e in eligible_eps]))
                diversity.append({"pairing_type":typ,"map_seed_a":left,"map_seed_b":right,"recipient_episode_count":len(eligible_eps),"pair_agreement_fraction":agreement,"different_pair_fraction":1.0-agreement})
    writecsv(EXP/"MAP_DIVERSITY.csv",diversity)
    # Episode/window eligibility and explicit task-level exclusion reasons.
    eligible_rows_csv=[]
    for e in eps:
        reason="" if e in elig_set else "task has only one primary-dev episode; no same-task donor"
        eligible_rows_csv.append({"episode_index":e,"task_index":ep_task[e],"split_assignment_key":ep_group[e],"primary_dev_windows":ep_windows[e],"eligible_common_subset":e in elig_set,"exclusion_reason":reason,"final_test_access":0})
    writecsv(EXP/"RECIPIENT_ELIGIBILITY.csv",eligible_rows_csv)
    evalrows=[]
    contexts=[]
    for seed in MODEL_SEEDS:
        contexts.append({"context_id":f"SIV-ANCHOR-D-H60-S{seed}","source_family":"SIV","source_cell":"ANCHOR-D","model_seed":seed,"delay_seconds":None})
    for seed in DELAY_SEEDS:
        for delay in DELAYS:
            contexts.append({"context_id":f"ODR-ROBUST-D-H60-S{seed}-D{delay:g}","source_family":"DELAY","source_cell":"ROBUST-D","model_seed":seed,"delay_seconds":delay})
    for c in contexts:
        for cond in CONDITIONS:
            mseed=int(cond.rsplit("M",1)[1]) if ("M" in cond and cond.startswith(("SAME","CROSS"))) else None
            ptype="same_task" if cond.startswith("SAME") else "cross_task" if cond.startswith("CROSS") else None
            evalrows.append({"bundle_id":f"{c['context_id']}-{cond}","context_id":c["context_id"],**c,"condition":cond,"map_seed":MAP_SEEDS[mseed-1] if mseed else None,"pairing_type":ptype,"recipient_population":"common_eligible_primary_dev","recipient_episodes":len(eligible_eps),"recipient_windows":len(eligible_rows),"status":"LOCKED","final_test_access":0})
    writecsv(EXP/"EVAL_MATRIX_120.csv",evalrows)
    # Model/checkpoint manifest and strict source config validation.
    cache=json.loads((V3/"cache/CACHE_MANIFEST.json").read_text())
    modelrows=[]
    for family,seeds,cells,codepath,root in (
      ("SIV",MODEL_SEEDS,("ANCHOR-D",),SIV/"code/state_video_model.py",SIV),
      ("DELAY",DELAY_SEEDS,("ROBUST-D",),DELAY/"code/delay_model.py",DELAY)):
        for seed in seeds:
            cell=cells[0];runid=f"SIV-{cell}-H60-S{seed}" if family=="SIV" else f"ODR-{cell}-H60-S{seed}"
            ckpt=(root/f"runs/{runid}/formal/checkpoint_last.pt")
            config=(root/f"runs/{runid}/formal/RUN_CONFIG_LOCKED.json")
            complete=(root/f"runs/{runid}/formal/RUN_COMPLETE.json")
            cfg=json.loads(config.read_text());done=json.loads(complete.read_text())
            source_input=V3/"data/V3_MAIN_INPUTS.npz"
            expected_input=cfg.get("v3_input_sha256",cfg.get("input_sha256"))
            expected_cache=cfg.get("v3_cache_sha256",cfg.get("cache_sha256"))
            modelrows.append({"family":family,"cell":cell,"seed":seed,"run_id":runid,"checkpoint_path":str(ckpt),"checkpoint_sha256":digest(ckpt),"run_config_sha256":digest(config),"model_code_sha256":digest(codepath),"epoch":done.get("epoch"),"optimizer_steps":done.get("optimizer_steps"),"input_hash_matches_current":expected_input==digest(source_input),"cache_hash_matches_locked":expected_cache==cache["feature_cache_sha256"],"status":"PASS" if done.get("status")=="PASS" and done.get("epoch")==40 and done.get("optimizer_steps")==11120 and expected_input==digest(source_input) and expected_cache==cache["feature_cache_sha256"] else "FAIL","cache_sha256":expected_cache,"input_sha256":expected_input,"final_test_access":0})
    writecsv(EXP/"MODEL_MANIFEST.csv",modelrows)
    if not all(r["status"]=="PASS" for r in modelrows):raise RuntimeError("source checkpoint acceptance failed")
    # Source registry includes sealed root manifests and verified hashes.
    sourcefiles=[V3/"PROTOCOL_LOCKED.json",V3/"ANCHOR_MANIFEST.csv",V3/"DATA_ALLOWLIST.csv",V3/"data/V3_MAIN_INPUTS.npz",V3/"data/V3_NORMALIZERS.json",V3/"data/V3_VIDEO_NORMALIZER.json",V3/"cache/CACHE_MANIFEST.json",SIV/"PROTOCOL_LOCKED.json",SIV/"qa/CHECKSUMS.sha256",DELAY/"FINAL_STATUS.ini",DELAY/"qa/CHECKSUMS.sha256"]
    writejson(EXP/"SOURCE_REGISTRY.json",{"status":"LOCKED","v3_root":str(V3),"siv_root":str(SIV),"delay_root":str(DELAY),"files":{str(p.relative_to(REPO)):digest(p) for p in sourcefiles},"feature_cache_path":str(V3/"cache/V3_TEMPORAL_FEATURES.npy"),"feature_cache_sha256":cache["feature_cache_sha256"],"encoder_updated":False,"new_video_extraction_started":False,"final_test_access":0})
    # Hash-lock eligibility and donor maps before any new dev prediction is computed.
    map_hashes={name:digest(EXP/name) for name in ("RECIPIENT_ELIGIBILITY.csv","DONOR_MAPS.csv","DONOR_WINDOW_MAPS.csv","DONOR_MATCH_QUALITY.csv","MAP_DIVERSITY.csv","EVAL_MATRIX_120.csv","MODEL_MANIFEST.csv")}
    lock={"status":"LOCKED_BEFORE_NEW_INFERENCE","map_seeds":list(MAP_SEEDS),"eligible_episodes":len(eligible_eps),"eligible_windows":len(eligible_rows),"excluded_singleton_tasks":singleton_tasks,"source_map_hashes":map_hashes,"same_task_maps_feasible":True,"cross_task_maps_feasible":True,"same_and_cross_share_recipient_subset":True,"mapping_uses_targets":False,"mapping_uses_model_scores":False,"mapping_uses_video_features_for_selection":False,"final_test_access":0}
    writejson(EXP/"DONOR_MAP_LOCK.json",lock)
    conditions=["MATCHED-D","MATCHED-S","SAME-TASK-DONOR-D-M1","SAME-TASK-DONOR-D-M2","SAME-TASK-DONOR-D-M3","CROSS-TASK-DONOR-D-M1","CROSS-TASK-DONOR-D-M2","CROSS-TASK-DONOR-D-M3"]
    writejson(EXP/"PROTOCOL_LOCKED.json",{"status":"LOCKED","protocol_id":"V3_VIDEO_CORRESPONDENCE_AUDIT_V1","source_roots":{"v3":str(V3),"siv":str(SIV),"delay":str(DELAY)},"model_families":["SIV ANCHOR-D H60","DELAY ROBUST-D H60"],"model_seeds":{"SIV":list(MODEL_SEEDS),"DELAY":list(DELAY_SEEDS)},"contexts":contexts,"conditions":conditions,"mapping_seeds":list(MAP_SEEDS),"eligible_episodes":len(eligible_eps),"eligible_windows":len(eligible_rows),"prediction_bundles":120,"full_matched_d_reference_replays":"6 model checkpoints; ODR also replay each of four delay contexts; compare with sealed source predictions","optimizer_steps_total":0,"encoder_forward_total":0,"new_video_extraction_started":False,"final_test_access":0})
    writejson(EXP/"RESOURCE_ESTIMATE.json",{"status":"PENDING_GPU_SMOKE","gpu_hours_upper_bound":6,"additional_disk_gib_upper_bound":12,"expected_gpu_context_jobs":15,"expected_counterfactual_bundles":120,"max_project_gpu_children":2,"dynamic_all_visible":True,"final_test_access":0})
    (EXP/"OUTCOME_EXPOSURE.md").write_text("# Outcome exposure\n\nAll 603-episode eligibility, task exclusions, mapping RNG seeds, one-to-one donor maps, nearest past-command donor-window matches and the 120-row evaluation matrix were generated from task/episode/group/timestamp/anchor-command metadata before any new prediction was produced. Future targets, model errors and prediction similarity were not used for donor assignment.\n\nThe map and eligibility artifacts are SHA-256 locked in DONOR_MAP_LOCK.json. This is a development counterfactual, not a physical intervention.\n",encoding="utf-8")
    (EXP/"CLAIM_LEDGER.md").write_text("# Claim ledger\n\n- MATCHED-D/S and donor substitutions are test-time input interventions on fixed checkpoints.\n- Episode-specific visual correspondence can be supported only conditional on these fixed donor maps and the eligible development subset.\n- Same-task donors retain some scene/task imbalance; matching is not a perfect counterfactual.\n- Donor maps reuse a finite development donor pool; inference is conditional on maps and must not treat maps or windows as independent replicates.\n- Static-vs-dynamic differences do not establish temporal-order understanding; no raw-frame order intervention is performed.\n- No training, encoder update, video extraction, final-test access or professor contact is authorized in this audit.\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","primary_dev":603,"eligible_episodes":len(eligible_eps),"eligible_windows":len(eligible_rows),"excluded_tasks":len(singleton_tasks),"tasks_retained":len(eligible_tasks),"maps":len(episode_maps),"window_maps":len(window_maps),"eval_bundles":len(evalrows),"checkpoints":len(modelrows),"final_test_access":0},sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
