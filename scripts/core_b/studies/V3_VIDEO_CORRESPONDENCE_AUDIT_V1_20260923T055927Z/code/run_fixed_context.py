#!/usr/bin/env python3
"""Replay one already-trained checkpoint on fixed matched/donor video conditions."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
V3=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SIV=REPO/"reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
DELAY=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
EXP=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
DELAY_CODE=DELAY/"code"
if str(DELAY_CODE) not in sys.path:sys.path.insert(0,str(DELAY_CODE))
from delay_contract import delay_transform_rows
ACTION_DIM=28; BATCH=64
CONDITIONS=("MATCHED-D","MATCHED-S","SAME-TASK-DONOR-D-M1","SAME-TASK-DONOR-D-M2","SAME-TASK-DONOR-D-M3","CROSS-TASK-DONOR-D-M1","CROSS-TASK-DONOR-D-M2","CROSS-TASK-DONOR-D-M3")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def readcsv(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x)))+"\n",encoding="utf-8");os.replace(tmp,path)
def atomic_npz(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("wb") as f:np.savez_compressed(f,**values)
    os.replace(tmp,path)
def load_inputs():
    # Deliberately excludes targets_native; model forward never receives future GT.
    needed=("anchor_id","episode_index","original_task_index","split_code","anchor_frame","anchor_timestamp",
            "history_native","history_standardized","history_mask","history_query_timestamps",
            "history_selected_timestamps","history_selected_frame_indices",
            "history_relative_delta_standardized","anchor_command_native")
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:
        return {k:z[k].copy() for k in needed}
def load_model(family,seed,device):
    if family=="SIV":
        code=SIV/"code"
        if str(code) not in sys.path:sys.path.insert(0,str(code))
        from state_video_model import StateVideoPredictor, state_hash
        model_path=SIV/f"runs/SIV-ANCHOR-D-H60-S{seed}/formal/checkpoint_last.pt"
        model=StateVideoPredictor(horizon=60,seed=seed).to(device)
    elif family=="DELAY":
        code=DELAY/"code"
        if str(code) not in sys.path:sys.path.insert(0,str(code))
        from delay_model import DelayAwarePredictor, state_hash
        model_path=DELAY/f"runs/ODR-ROBUST-D-H60-S{seed}/formal/checkpoint_last.pt"
        model=DelayAwarePredictor(horizon=60,seed=seed).to(device)
    else:raise ValueError(family)
    payload=torch.load(model_path,map_location="cpu",weights_only=False)
    model.load_state_dict(payload["model_state_dict"],strict=True);model.eval()
    return model,model_path,state_hash
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--context-id",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for fixed-checkpoint inference")
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();device=torch.device("cuda:0")
    ctx=next(r for r in readcsv(EXP/"EVAL_MATRIX_120.csv") if r["context_id"]==args.context_id)
    family=ctx["source_family"];seed=int(ctx["model_seed"]);delay=ctx["delay_seconds"]
    delay_index=None if delay in ("","None","null") else min(range(4),key=lambda i:abs([0,.25,.5,1][i]-float(delay)))
    data=load_inputs();features=np.load(V3/"cache/V3_TEMPORAL_FEATURES.npy",mmap_mode="r")
    normalizer=json.loads((V3/"data/V3_NORMALIZERS.json").read_text());vnorm=json.loads((V3/"data/V3_VIDEO_NORMALIZER.json").read_text())
    vmean=np.asarray(vnorm["mean"],dtype=np.float32);vstd=np.asarray(vnorm["safe_std"],dtype=np.float32)
    sigma=torch.tensor(normalizer["target_sigma"],dtype=torch.float32,device=device)
    model,checkpoint,state_hash_fn=load_model(family,seed,device);before=state_hash_fn(model)
    dev_rows=np.flatnonzero(data["split_code"]==1)
    eligible=readcsv(EXP/"RECIPIENT_ELIGIBILITY.csv")
    eligible_eps={int(r["episode_index"]) for r in eligible if r["eligible_common_subset"]=="True"}
    subset_rows=np.asarray([r for r in dev_rows if int(data["episode_index"][r]) in eligible_eps],dtype=np.int64)
    # The locked window map provides donor rows; each condition reuses the same recipient order.
    window_maps=readcsv(EXP/"DONOR_WINDOW_MAPS.csv")
    if family=="SIV":
        src_pred=SIV/f"predictions/SIV-ANCHOR-D-H60-S{seed}/primary.npz"
    else:
        src_pred=DELAY/f"predictions/ODR-ROBUST-D-H60-S{seed}/delay_{delay_index}.npz"
    with np.load(src_pred,allow_pickle=False) as z:
        source_prediction=z["predictions_native"].copy();source_ids=z["anchor_id"].copy()
    if not np.array_equal(source_ids,data["anchor_id"][dev_rows]):raise RuntimeError("source matched-D prediction IDs do not match V3 primary dev")
    if source_prediction.shape!=(len(dev_rows),60,ACTION_DIM):raise RuntimeError("source matched-D prediction shape mismatch")
    video_mean=torch.tensor(vmean,dtype=torch.float32,device=device);video_std=torch.tensor(vstd,dtype=torch.float32,device=device)
    def input_arrays(rows):
        if family=="SIV":
            state=data["history_standardized"][rows]
            age=np.zeros((len(rows),32),dtype=np.float32)
        else:
            state,age,sel_times,sel_ix=delay_transform_rows(data,normalizer,rows,np.full(len(rows),delay_index,dtype=np.int64))
            # Validate the reused ODR input transform against the saved source bundle for this delay.
            with np.load(DELAY/f"predictions/ODR-ROBUST-D-H60-S{seed}/delay_{delay_index}.npz",allow_pickle=False) as z:
                source_age=z["observation_age_seconds"]
                source_times=z["observation_selected_timestamps"]
            source_positions=np.searchsorted(dev_rows,rows)
            if not np.array_equal(dev_rows[source_positions],rows):
                raise RuntimeError("requested ODR rows are not in the primary dev source bundle")
            if not np.allclose(age,source_age[source_positions],atol=1e-6) or not np.allclose(sel_times,source_times[source_positions],atol=1e-6):
                raise RuntimeError("ODR delayed-input transform does not match saved source ages/timestamps")
        return state.astype(np.float32),age.astype(np.float32)
    def forward_rows(rows,video_rows,token_slot,state_age):
        state_np,age_np=state_age
        outs=[]
        for start in range(0,len(rows),BATCH):
            rr=rows[start:start+BATCH];vr=video_rows[start:start+BATCH]
            state=torch.from_numpy(state_np[start:start+BATCH]).to(device)
            mask=torch.from_numpy(data["history_mask"][rr]).to(device)
            rel=torch.from_numpy(data["history_relative_delta_standardized"][rr]).to(device)
            anchor=torch.from_numpy(data["anchor_command_native"][rr]).to(device)
            token=np.asarray(features[vr,token_slot],dtype=np.float32)
            video=(torch.from_numpy(token).to(device)-video_mean)/video_std
            with torch.inference_mode():
                if family=="SIV":
                    residual=model.forward_locked(state,mask,rel,video,anchor,state_level="ANCHOR",video_level="D")
                else:
                    age=torch.from_numpy(age_np[start:start+BATCH]).to(device)
                    residual=model.forward_delay(state,mask,rel,age,video,anchor)
                pred=anchor.unsqueeze(1)+residual*sigma.view(1,1,ACTION_DIM)
            outs.append(pred.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs)
    replay_rows=dev_rows[:64] if args.smoke else dev_rows
    replay_positions=np.searchsorted(dev_rows,replay_rows)
    replay_state_age=input_arrays(replay_rows)
    replay_pred=forward_rows(replay_rows,replay_rows,0,replay_state_age)
    replay_reference=source_prediction[replay_positions]
    replay={"context_id":args.context_id,"family":family,"model_seed":seed,"delay_seconds":None if delay_index is None else float([0,.25,.5,1][delay_index]),"rows":len(replay_rows),"shape":list(replay_pred.shape),"source_shape":list(source_prediction.shape),"max_abs_diff":float(np.max(np.abs(replay_pred-replay_reference))),"mean_abs_diff":float(np.mean(np.abs(replay_pred-replay_reference))),"allclose_at_1e-6":bool(np.allclose(replay_pred,replay_reference,rtol=0,atol=1e-6)),"checkpoint":str(checkpoint),"checkpoint_sha256":digest(checkpoint),"source_prediction_sha256":digest(src_pred),"parameter_hash_before":before,"physical_gpu":os.environ.get("GPU_WATCHER_PHYSICAL_GPU","unknown"),"final_test_access":0}
    if replay_pred.shape!=(len(replay_rows),60,ACTION_DIM):raise RuntimeError("source replay shape mismatch")
    if args.smoke:
        replay["smoke_rows"]=len(replay_rows)
        replay["status"]="PASS" if replay["allclose_at_1e-6"] else "FAIL"
        if replay["status"]!="PASS":raise RuntimeError(f"matched-D smoke mismatch {replay['max_abs_diff']}")
        out.mkdir(parents=True,exist_ok=True);atomic_json(out/"STAGE0_REPLAY.json",replay)
        peak=int(torch.cuda.max_memory_reserved()//(1024*1024));replay["peak_memory_reserved_mib"]=peak;atomic_json(out/"STAGE0_REPLAY.json",replay)
        print(json.dumps(replay,sort_keys=True));return 0
    if not replay["allclose_at_1e-6"]:raise RuntimeError(f"full matched-D replay mismatch max_abs={replay['max_abs_diff']}")
    state_age=input_arrays(subset_rows);conditions=[]
    for condition in CONDITIONS:
        if condition=="MATCHED-D":donor_rows=subset_rows.copy();slot=0;mapseed=-1;pair_type="matched"
        elif condition=="MATCHED-S":donor_rows=subset_rows.copy();slot=1;mapseed=-1;pair_type="matched_static"
        else:
            if condition.startswith("SAME"):
                pair_type="same_task";mi=int(condition.rsplit("M",1)[1])-1
            else:
                pair_type="cross_task";mi=int(condition.rsplit("M",1)[1])-1
            mapseed=(202609701,202609702,202609703)[mi]
            byrow={int(r["recipient_row"]):r for r in window_maps if r["pairing_type"]==pair_type and int(r["map_seed"])==mapseed}
            try:donor_rows=np.asarray([int(byrow[int(r)]["donor_row"]) for r in subset_rows],dtype=np.int64)
            except KeyError as e:raise RuntimeError(f"missing donor row {e}")
            slot=0
        bundle_id=f"{args.context_id}-{condition}"
        bundle_dir=EXP/f"predictions/{args.context_id}"
        bundle_path=bundle_dir/f"{condition}.npz"
        metadata_path=bundle_dir/f"{condition}.json"
        if bundle_path.exists() or metadata_path.exists():
            if not bundle_path.is_file() or not metadata_path.is_file():
                raise RuntimeError(f"partial existing shard preserved; refusing overwrite: {bundle_id}")
            existing=json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("bundle_id")!=bundle_id or existing.get("context_id")!=args.context_id or existing.get("condition")!=condition or int(existing.get("model_seed",-1))!=seed or existing.get("checkpoint_sha256")!=replay["checkpoint_sha256"] or existing.get("bundle_sha256")!=digest(bundle_path):
                raise RuntimeError(f"existing shard provenance/hash mismatch; refusing overwrite: {bundle_id}")
            with np.load(bundle_path,allow_pickle=False) as z:
                if z["predictions_native"].shape!=(len(subset_rows),60,ACTION_DIM) or not np.isfinite(z["predictions_native"]).all():
                    raise RuntimeError(f"existing shard prediction schema invalid: {bundle_id}")
                if not np.array_equal(z["recipient_anchor_id"],data["anchor_id"][subset_rows]) or not np.array_equal(z["donor_anchor_id"],data["anchor_id"][donor_rows]) or not np.array_equal(z["donor_map_seed"],np.full(len(subset_rows),mapseed,dtype=np.int64)) or str(z["condition"].item())!=condition:
                    raise RuntimeError(f"existing shard IDs/condition mismatch; refusing overwrite: {bundle_id}")
            conditions.append(existing)
            continue
        pred=forward_rows(subset_rows,donor_rows,slot,state_age)
        if pred.shape!=(len(subset_rows),60,28) or not np.isfinite(pred).all():raise RuntimeError(f"bundle shape/finite failed {condition}: {pred.shape}")
        donor_meta=[]
        if mapseed<0:
            for rr,dr in zip(subset_rows,donor_rows):
                donor_meta.append({"donor_episode_index":int(data["episode_index"][dr]),"donor_anchor_id":str(data["anchor_id"][dr]),"donor_task_index":int(data["original_task_index"][dr]),"donor_anchor_frame":int(data["anchor_frame"][dr]),"donor_anchor_timestamp":float(data["anchor_timestamp"][dr]),"donor_command_distance":0.0})
        else:
            byrow={(r["pairing_type"],int(r["map_seed"]),int(r["recipient_row"])):r for r in window_maps}
            for rr,dr in zip(subset_rows,donor_rows):
                donor_meta.append({k:byrow[(pair_type,mapseed,int(rr))][k] for k in ("donor_episode","donor_anchor_id","donor_task","donor_anchor_frame","donor_anchor_timestamp","signed_anchor_timestamp_diff","absolute_anchor_timestamp_diff","past_command_distance_scaled_mae")})
        bundle_id=f"{args.context_id}-{condition}"
        bundle_dir=EXP/f"predictions/{args.context_id}";bundle_path=bundle_dir/f"{condition}.npz"
        metadata={k:np.asarray([m[k] for m in donor_meta]) for k in donor_meta[0]}
        atomic_npz(bundle_path,recipient_anchor_id=data["anchor_id"][subset_rows],recipient_episode_index=data["episode_index"][subset_rows],recipient_task_index=data["original_task_index"][subset_rows],recipient_anchor_frame=data["anchor_frame"][subset_rows],recipient_anchor_timestamp=data["anchor_timestamp"][subset_rows],donor_map_seed=np.full(len(subset_rows),mapseed,dtype=np.int64),donor_episode_index=metadata.get("donor_episode_index",metadata.get("donor_episode")),donor_anchor_id=metadata.get("donor_anchor_id"),donor_task_index=metadata.get("donor_task_index",metadata.get("donor_task")),donor_anchor_frame=metadata.get("donor_anchor_frame"),donor_anchor_timestamp=metadata.get("donor_anchor_timestamp"),donor_command_distance=metadata.get("donor_command_distance",metadata.get("past_command_distance_scaled_mae")),requested_delay_seconds=np.asarray(-1.0 if delay_index is None else [0,.25,.5,1][delay_index],dtype=np.float32),condition=np.asarray(condition),map_seed=np.asarray(mapseed,dtype=np.int64),predictions_native=pred)
        condition_meta={"status":"PASS","bundle_id":bundle_id,"context_id":args.context_id,"family":family,"model_seed":seed,"delay_seconds":None if delay_index is None else [0,.25,.5,1][delay_index],"condition":condition,"map_seed":mapseed,"pairing_type":pair_type,"shape":list(pred.shape),"bundle_sha256":digest(bundle_path),"checkpoint_sha256":replay["checkpoint_sha256"],"model_hash_before":before,"final_test_access":0,"optimizer_steps":0,"encoder_forward":0}
        atomic_json(bundle_dir/f"{condition}.json",condition_meta)
        conditions.append(condition_meta)
    after=state_hash_fn(model)
    if after!=before:raise RuntimeError("model parameters/buffers changed during inference")
    replay.update({"status":"PASS","eligible_episodes":len(eligible_eps),"eligible_windows":len(subset_rows),"parameter_hash_after":after,"state_unchanged":True,"condition_count":len(conditions),"condition_ids":[x["bundle_id"] for x in conditions],"peak_memory_allocated_mib":int(torch.cuda.max_memory_allocated()//(1024*1024)),"peak_memory_reserved_mib":int(torch.cuda.max_memory_reserved()//(1024*1024)),"encoder_forward":0,"optimizer_steps":0})
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/"CONTEXT_REPLAY_ACCEPTED.json",replay);print(json.dumps(replay,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
