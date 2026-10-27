"""Freeze source-group-correct D1 and group-aware D2/D4 reanalysis contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pathlibs import BENCH_ROOT, NEXT_ROOT, PROJECT_ROOT, REPO_ROOT, V3_ROOT


RAW_ROOT=PROJECT_ROOT/"data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
SPLIT=REPO_ROOT/"manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
ANCHORS=V3_ROOT/"ANCHOR_MANIFEST.csv"
SIGMA=np.asarray(json.loads((V3_ROOT/"data/V3_NORMALIZERS.json").read_text())["target_sigma"],dtype=np.float64)


def read_csv(path:Path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


def atomic_json(path:Path,data):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name("."+path.name+".part");tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");tmp.replace(path)


def sha256(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def q_signal(action):
    delta=np.abs(np.diff(action,axis=0))/SIGMA[None,:]
    return np.stack((delta[:,:14].mean(axis=1),delta[:,14:].mean(axis=1)),axis=1)


def weighted_q(per_episode, q):
    vals=[]; weights=[]; n_ep=len(per_episode)
    for arr in per_episode:
        if len(arr):
            arr=np.asarray(arr,dtype=np.float64)
            vals.append(arr); weights.append(np.full(len(arr),1.0/(n_ep*len(arr)),dtype=np.float64))
    v=np.concatenate(vals); w=np.concatenate(weights); order=np.argsort(v,kind="mergesort"); cum=np.cumsum(w[order])
    pos=min(int(np.searchsorted(cum,q,side="left")),len(order)-1)
    return float(v[order[pos]])


def main():
    # Source groups use the label-free train/dev duplicate-safe split manifest.
    manifest=pq.read_table(SPLIT,columns=["episode_index","split","video_sha256_fingerprint"],filters=[("split","in",["train","validation"])])
    group_map={int(r["episode_index"]):(r["video_sha256_fingerprint"] or f"episode:{r['episode_index']}") for r in manifest.to_pylist()}
    anchor_rows=[r for r in read_csv(ANCHORS) if r["population"]=="primary" and r["split"] in {"train","dev"} and r["final_test_access"]=="0"]
    by_episode=defaultdict(list)
    for row in anchor_rows:by_episode[int(row["episode_index"])].append(row)
    train_episode_values=[[] for _ in range(2)]; dt_deviation=[]; source_group_metrics=Counter()
    for episode in sorted({int(r["episode_index"]) for r in anchor_rows if r["split"]=="train"}):
        row=by_episode[episode][0]
        table=pq.read_table(RAW_ROOT/row["parquet_path"],columns=["timestamp","action"])
        ts=table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        action=np.asarray(table["action"].to_pylist(),dtype=np.float64)
        dt_deviation.extend(np.abs(np.diff(ts)-1.0/30.0).tolist())
        signal=q_signal(action)
        legal=np.zeros(len(action),dtype=bool)
        for ar in by_episode[episode]:
            a=int(ar["anchor_frame"])
            # q at endpoint f needs both command frames f-1 and f within
            # the legal [a-119,a+60] observation/target range.
            left=max(1,a-118); right=min(len(action)-1,a+60)
            if right>=left:legal[left:right+1]=True
        per_group=[[],[]]
        for g in range(2): per_group[g]=signal[legal[1:],g] if len(legal)>1 else np.asarray([])
        train_episode_values[0].append(per_group[0]);train_episode_values[1].append(per_group[1])
    theta=np.zeros((2,2),dtype=np.float64)
    for g in range(2):
        on=max(weighted_q(train_episode_values[g],.75),1e-4)
        off=min(weighted_q(train_episode_values[g],.25),.5*on)
        theta[g]=[on,off]
    threshold={"protocol_id":"CORE_B_D2_EVENT_REANALYSIS_V2","parent_protocol_id":"CORE_B_BENCHMARK_D2_COMMAND_CHANGE_V1","group_order":["Hand","Arm"],"thresholds_train_only":True,"theta_on":theta[:,0].tolist(),"theta_off":theta[:,1].tolist(),"threshold_population":"unique legal command-step endpoints covered by union of primary train anchor windows [a-119,a+60]; each train episode equal, overlap timestamps deduplicated","state_machine":"For each Hand/Arm group independently: q<=off for 6 consecutive steps arms; q>on for 3 consecutive steps emits onset at first high step; intermediate q breaks low/high consecutive counters but preserves armed/disarmed categorical status; initial unknown until a complete quiet run is observed.","prediction_prefix":"Only the legal 4s command prefix ending at each query initializes prediction event state; carry armed/known state and streak counters into predicted future; unknown queries excluded from event scoring.","group_matching":"Separate forecast clustering and one-to-one onset matching by Hand/Arm group, episode, model, and horizon.","dense_main_denominator":"unique GT events with >= nominal H seconds of legal dense query history in the fixed block, occurring before block last query and with at least one state-known eligible query; per-lead denominator further requires an actual eligible query with that lead and effective onset horizon.","false_alarm_exposure":"sum over blocks of last_query_timestamp-first_query_timestamp; overlap windows do not add exposure; Hand and Arm false-alarm counts share elapsed wall exposure and are also reported per group.","match_tolerance_seconds":0.2,"forecast_cluster_rule":"Within event group, merge alarms whose predicted-onset is within 0.2s of the first alarm in a cluster; cluster center remains first predicted onset; global one-to-one matching maximizes hit count then minimizes onset timing error.","effective_onset_horizon":"actual query timestamp to timestamp of last forecast step that can complete the 3-step onset confirmation; nominal H minus 2 raw steps on stable 30Hz input.","timestamp_check_max_abs_dt_error_seconds":float(max(dt_deviation) if dt_deviation else float("nan")),"train_episodes":len(train_episode_values[0]),"final_test_access":0}
    threshold["protocol_hash"]=hashlib.sha256(json.dumps(threshold,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    atomic_json(NEXT_ROOT/"protocol/D2_EVENT_REANALYSIS_V2.json",threshold)
    groups_by_fp=Counter(group_map.values())
    dev_primary_episodes=sorted({int(r["episode_index"]) for r in anchor_rows if r["split"]=="dev"})
    group_summary={"v3_dev_primary_episodes":len(dev_primary_episodes),"v3_dev_source_groups":len({group_map[e] for e in dev_primary_episodes}),"v3_dev_primary_rows":sum(len(by_episode[e]) for e in dev_primary_episodes)}
    d1={"protocol_id":"CORE_B_D1_SOURCE_GROUP_REANALYSIS_V2","parent_protocol_id":"CORE_B_BENCHMARK_D1_NEIGHBOR_DISPERSION_V1","distance":"mean squared Euclidean distance on the same 32x71 standardized state array","K":20,"bank":"V3 primary train rows, first up to 8 anchors per episode in locked anchor order","neighbor_independence":"one nearest anchor per distinct video_sha256_fingerprint source group; exclude query episode and entire source group","calibration":"same train query anchor rule as source run; leave-source-group-out; episode-equal Q25/Q75 A and Q95 kth-neighbor distance; no dev outcome thresholds","dev_strata":"train-only 2s A Q25/Q75 and k20 distance Q95; keep natural overall and distant-neighbor rows","source_group_field":"split_manifest.video_sha256_fingerprint; fallback episode-index only when fingerprint is empty","source_split_manifest_sha256":sha256(SPLIT),"source_groups_train_dev":len(groups_by_fp),"repeated_source_groups_train_dev":sum(v>1 for v in groups_by_fp.values()),"v3_split_name_mapping":{"anchor_manifest":"dev","split_manifest":"validation"},"training_neighbor_build_status":"locked before source-group-aware recomputation","final_test_access":0}
    d1["primary_contrasts"]={"visual_gain":"MAE_control-MAE_F-TOK-D for S-HIST, F-POOL-D, and F-TOK-S controls; high/low supported strata defined by train-only Q75/Q25 of 2s neighbor residual dispersion","interaction":"gain_high_supported-gain_low_supported; Arm and Hand primary, Full28 contextual","reported_metrics":["standardized_MAE","native_FD"],"bootstrap":{"unit":"video_sha256_fingerprint source group","replicates":10000,"seed":20260923,"shared_group_resamples_across_seed_cell_body_and_strata":True}}
    d1["protocol_hash"]=hashlib.sha256(json.dumps(d1,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    atomic_json(NEXT_ROOT/"protocol/D1_SOURCE_GROUP_REANALYSIS_V2.json",d1)
    d4={"protocol_id":"CORE_B_D4_ROLLING_ABSOLUTE_TIME_V2","status":"LOCKED_BEFORE_D4_OUTCOME_COMPUTATION","source":"accepted D2 dense input panel and 36 sealed V3 primary prediction bundles","pair_rule":"Adjacent dense queries within same episode and dense block, sorted by actual timestamp; no pair crosses block/episode.","common_future_support":"For old anchor a and new b, join raw target by exact episode frame/timestamp on T={t:b<t<=a+H} that exists in both native forecast horizons. Require timestamp arrays from Parquet to match source frame joins; no index-only assumption.","estimands":{"L":"mean standardized absolute error on identical common future points, by Hand[0:14], Arm[14:28], Full28","U":"L_old-L_new; positive means lower common-point error after the later observation","R":"mean absolute standardized revision between old and new forecasts on T; not a utility score","FD":"native first-difference forecast error over adjacent timestamps inside T","endpoint":"native MAE at the last timestamp in T","transition":"pair share with U>0, U<0, tie; no unregistered binary correct/wrong threshold"},"primary_contrasts":["DeltaU_video=U_F-TOK-D-U_S-HIST; Arm/Hand x H30/H60"],"secondary_contrasts":["DeltaU_temporal=U_F-TOK-D-U_F-POOL-D","DeltaU_dynamic=U_F-TOK-D-U_F-TOK-S"],"event_slices":"GT-defined diagnostic only; use D2_EVENT_REANALYSIS_V2 groups/state and only next onset confirmed inside common T; bin onset-b: (0,.5s],(.5,1s],(1s,2s],none_within_T; exclude pairs where either query is unknown or disarmed/ongoing from slices but retain in natural D4.","event_layer":"D2 v2 grouped events; summarize same-group event metrics/lead/FA exposure at new anchor; no event labels enter inputs.","resampling":"aggregate pairs within episode, episodes within video-hash source group, then source groups equally; 10000 paired cluster bootstrap per seed/horizon, shared resamples across contrasts; also report each seed and mean+SD.","outcome_exposure":"D1/D2/D3 development outcomes and V3_STATE 54-cell development study were already exposed before this D4 run; D4 utility itself not previously computed. This is exploratory development evidence.","v3state_compatibility":"54-cell V3_STATE shares primary dev anchor IDs/Full28/native target/horizons but separate state-mask/video factorial and fresh seeds; optional dense extension only if prediction/checkpoint/input contracts validate.","D3_extension_rule":"Only if complete D3 formal bundles and same fixed D2 dense query IDs all 36 configs pass IDs/shape/hash/target checks and preflight budget <=24 GPU-hours <=25GiB; inference only, no training.","final_test_access":0}
    d4["execution_details"]={"pair_manifest":"For each seed x horizon x contrast, include only common adjacent pairs with accepted old/new/candidate/control bundles and exact frame/timestamp/GT support; old input-only manifest is retained separately.","common_frames":"Integer frame join t=b+1...a+H; require raw Parquet timestamp identity and target support in both forecasts; verify source interval is actual 7/8-frame gap, not a fixed 0.25s assumption.","primary_effects":["DeltaU_video=U_F-TOK-D-U_S-HIST","DeltaU_temporal=U_F-TOK-D-U_F-POOL-D","DeltaU_dynamic=U_F-TOK-D-U_F-TOK-S"],"extra_metrics":["native_MAE","FD_error","endpoint_error","coordinate_error_lower_share","coordinate_error_higher_share","coordinate_error_tie_share"],"bootstrap":{"unit":"highest independent video_sha256_fingerprint source group","replicates":10000,"seed":20260926,"shared_resamples_across_cells_and_contrasts":True,"p_values":"not reported; percentile paired cluster intervals only"},"event_slice":"Per Hand/Arm separately; both old and new query must be state-known and armed under D2 v2; next event must be fully confirmed in T; bins by onset minus new anchor: (0,.5],(.5,1],(1,2],none_within_T; beyond new native horizon is NA.","representative_pair_selection":"Outcome-blind: choose the H60 input pair with minimum SHA256(pair_id|CORE_B_D4_REP_V2) from fixed D2 panel; seed 20260950 F-TOK-D; plot Hand coordinate 0 and Arm coordinate 14 plus GT.","failure_pair_selection":"After scoring, diagnostic only: maximal negative DeltaU_video pair among natural common pairs; not a confirmatory estimate.","prior_v1_controls":"stale/fresh-video, latency, missing-frame and timestamp-gap interventions require new controlled inference; deferred and excluded from D4 rolling-core completion claim.","final_test_access":0}
    d4["protocol_hash"]=hashlib.sha256(json.dumps(d4,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    atomic_json(NEXT_ROOT/"protocol/D4_PROTOCOL_LOCKED.json",d4)
    (NEXT_ROOT/"OUTCOME_EXPOSURE.md").write_text("""# Outcome exposure\n\nD1, D2 and D3 development results, the 54-cell V3_STATE results, and their analysis summaries were already visible in the prior sealed benchmark roots and previous handoff. Their raw numbers are not preregistration-blind. This NEXT_ROOT recomputes and audits those values; all claims remain exploratory development claims.\n\nThe D4 rolling-update estimand itself (same absolute future timestamps, paired old/new, U and R definitions, source-group resampling, event-slice bins) was frozen in `protocol/D4_PROTOCOL_LOCKED.json` before D4 pair-level prediction errors were calculated. Its source predictions had been used in prior per-anchor D1/D2/D3 evaluation, so D4 is a post-hoc secondary development analysis, not an independent confirmation.\n\nD1 source-group-aware reanalysis is versioned because the previous D1 neighbor builder excluded only the same episode even though the label-free split manifest provides exact video hash groups. D2 event results are versioned because prior sparse/dense evaluators merged Hand and Arm events and used an invalid dense event denominator. Existing thresholds are preserved; the corrected threshold computation is train-only over the union of legal primary train time ranges.\n""",encoding="utf-8")
    # Dense pair manifest is input-only and is built before reading D4 forecast arrays.
    panel=read_csv(BENCH_ROOT/"d2/D2_DENSE_PANEL.csv")
    by_ep=defaultdict(list)
    for row in panel:by_ep[int(row["episode_index"])].append(row)
    pairs=[]
    for ep,rows in sorted(by_ep.items()):
        rows=sorted(rows,key=lambda r:int(r["query_index"]))
        for old,new in zip(rows,rows[1:]):
            if old["block_id"]!=new["block_id"] or int(new["query_index"])!=int(old["query_index"])+1:continue
            a=int(old["anchor_frame"]);b=int(new["anchor_frame"]);dt=float(new["anchor_timestamp"])-float(old["anchor_timestamp"])
            source_group=group_map.get(ep,f"episode:{ep}")
            for h in (30,60):
                lo=b+1;hi=a+h;n=max(0,hi-lo+1)
                pairs.append({"pair_id":f"e{ep:06d}_a{a:06d}_b{b:06d}_H{h}","episode_index":ep,"task":int(old["task"]),"source_group_id":source_group,"block_id":old["block_id"],"old_dense_id":f"dense_e{ep:06d}_a{a:06d}","new_dense_id":f"dense_e{ep:06d}_a{b:06d}","old_anchor_frame":a,"new_anchor_frame":b,"old_anchor_timestamp":float(old["anchor_timestamp"]),"new_anchor_timestamp":float(new["anchor_timestamp"]),"actual_anchor_interval_seconds":dt,"horizon_steps":h,"common_start_frame":lo if n else "","common_end_frame":hi if n else "","common_target_steps":n,"status":"locked_input_only" if n else "empty_common_support","final_test_access":0})
    pair_fields=["pair_id","episode_index","task","source_group_id","block_id","old_dense_id","new_dense_id","old_anchor_frame","new_anchor_frame","old_anchor_timestamp","new_anchor_timestamp","actual_anchor_interval_seconds","horizon_steps","common_start_frame","common_end_frame","common_target_steps","status","final_test_access"]
    with (NEXT_ROOT/"d4/D4_PAIR_MANIFEST.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=pair_fields);writer.writeheader();writer.writerows(pairs)
    print(json.dumps({"D1_protocol_hash":d1["protocol_hash"],"D2_event_protocol_hash":threshold["protocol_hash"],"D4_protocol_hash":d4["protocol_hash"],"D4_pairs":len(pairs),"source_groups":group_summary["v3_dev_source_groups"],"theta_on":threshold["theta_on"],"theta_off":threshold["theta_off"],"dt_max_abs_deviation":threshold["timestamp_check_max_abs_dt_error_seconds"]}))


if __name__=="__main__":main()
