#!/usr/bin/env python3
"""Build only the authorized observed-prefix tensors for fixed Core B final anchors."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT=PROJECT/"reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
RAW_ROOT=PROJECT/"data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
INPUTS=OUT/"core_b/FINAL_CURRENT_PROFILE_INPUTS.csv"
POP=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z/protocol/CORE_B_FINAL_POPULATION_METADATA.csv"
LOCK=OUT/"core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json"
FREEZE=OUT/"core_b/CORE_B_V2_FINAL_FREEZE.json"
ACCESS=OUT/"protocol/FINAL_ACCESS_OPENED.json"
OUT_PREFIX=RUN/"replay/core_b/FINAL_PREFIXES_CURRENT_PROFILE_V2.npz"
OUT_RECEIPT=OUT/"core_b/FINAL_PREFIX_EXTRACTION_RECEIPT.json"
TASKS=(119,94,90,141,230)
TASK_RANK={t:i for i,t in enumerate(TASKS)}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def utc(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f".{p.name}.{os.getpid()}.part")
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,p)

def verify_gates(lock_sha,freeze_sha,access_sha):
 if sha(LOCK)!=lock_sha or sha(FREEZE)!=freeze_sha: raise ValueError("final profile/freeze hash mismatch")
 from core_b_predict_v2 import verify_final_access_receipt
 verify_final_access_receipt(ACCESS,access_sha)
 lock=json.loads(LOCK.read_text(encoding="utf-8"));freeze=json.loads(FREEZE.read_text(encoding="utf-8"))
 if lock.get("status")!="LOCKED_BEFORE_CURRENT_PROFILE_DEV_FEATURES" or lock.get("route")!="R2": raise ValueError("CURRENT_PROFILE_V2 is not locked")
 if sha(Path(__file__))!=lock.get("final_inputs",{}).get("final_prefix_builder_code_sha256"): raise ValueError("final prefix builder source hash differs from the pre-access lock")
 if freeze.get("status")!="FROZEN_BEFORE_FINAL_ACCESS" or freeze.get("profile_lock_sha256")!=lock_sha: raise ValueError("Core B V2 final freeze is incomplete or mismatched")
 return lock,freeze

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--expected-lock-sha256",required=True);ap.add_argument("--expected-final-freeze-sha256",required=True);ap.add_argument("--expected-final-access-sha256",required=True);ap.add_argument("--dry-run",action="store_true");args=ap.parse_args()
 # Receipt and freeze are verified before parsing any final input manifest or opening a parquet/video.
 lock,freeze=verify_gates(args.expected_lock_sha256,args.expected_final_freeze_sha256,args.expected_final_access_sha256)
 if sha(INPUTS)!=lock["inputs"]["final_input_manifest_sha256"]: raise ValueError("final input manifest differs from locked metadata")
 if sha(POP)!=lock["final_inputs"]["final_population_metadata_sha256"]: raise ValueError("final population metadata differs from lock")
 with INPUTS.open(encoding="utf-8-sig",newline="") as stream:
  rows=list(csv.DictReader(stream))
 with POP.open(encoding="utf-8-sig",newline="") as stream:
  pop=list(csv.DictReader(stream))
 if len(rows)!=162 or len(pop)!=162: raise ValueError("fixed final Core B population must have 162 anchors")
 pop_by={r["anchor_id"]:r for r in pop}
 if len(pop_by)!=162: raise ValueError("duplicate final metadata anchor")
 for r in rows:
  p=pop_by.get(r["anchor_id"])
  if not p or r["dataset_role"]!="final_test" or p["dataset_role"]!="final_test": raise ValueError("final prefix input identity missing/non-final")
  for key in ("episode_id","task_id","anchor_frame","parquet_path"):
   target="literal_source_key" if key=="parquet_path" else key
   if str(r[key])!=str(p[target]): raise ValueError(f"final map/population mismatch {key}: {r['anchor_id']}")
  if int(p["target_h30_row_start_inclusive"])!=int(r["anchor_frame"])+1 or int(p["target_h30_row_end_inclusive"])!=int(r["anchor_frame"])+30:
   raise ValueError(f"locked future target interval differs from H30 contract: {r['anchor_id']}")
 if args.dry_run:
  return print(json.dumps({"status":"DRY_RUN_PASS_FINAL_METADATA_ONLY","rows":162,"final_payload_read":False,"future_targets_materialized":0,"locked_input_manifest_sha256":sha(INPUTS)})) or 0
 if OUT_PREFIX.exists() or OUT_RECEIPT.exists(): raise FileExistsError("refusing to overwrite final prefix artifacts")

 by_parquet=defaultdict(list)
 for i,r in enumerate(rows): by_parquet[str(r["parquet_path"])].append(i)
 history=np.empty((162,32,71),dtype=np.float32)
 mask=np.ones((162,32),dtype=np.float32)
 rel=np.empty((162,32),dtype=np.float32)
 anchor_native=np.empty((162,28),dtype=np.float32)
 parquet_receipts=[]
 for relative,indices in sorted(by_parquet.items()):
  parquet=(RAW_ROOT/relative).resolve()
  if not parquet.is_relative_to(RAW_ROOT.resolve()) or not parquet.is_file(): raise FileNotFoundError(f"final prefix parquet absent/outside root: {relative}")
  requested=set()
  for ix in indices:
   a=int(rows[ix]["anchor_frame"])
   if a<119: raise ValueError(f"final anchor lacks 120-row observation window: {rows[ix]['anchor_id']}")
   requested.update(range(a-119,a+1))
  frames_requested=sorted(requested)
  # Read only the observed history fields and rows through the maximum permitted anchor.
  table=ds.dataset(str(parquet),format="parquet").to_table(
   columns=["frame_index","timestamp","action","observation.arm_joints","observation.hand_joints","observation.leg_joints"],
   filter=ds.field("frame_index").isin(frames_requested))
  frame_index=table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)
  order=np.argsort(frame_index,kind="stable")
  frame_index=frame_index[order]
  timestamp=table["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)[order]
  action=np.asarray(table["action"].to_pylist(),dtype=np.float32)[order]
  arm=np.asarray(table["observation.arm_joints"].to_pylist(),dtype=np.float32)[order]
  hand=np.asarray(table["observation.hand_joints"].to_pylist(),dtype=np.float32)[order]
  leg=np.asarray(table["observation.leg_joints"].to_pylist(),dtype=np.float32)[order]
  if len(frame_index)!=len(frames_requested) or set(frame_index.tolist())!=set(frames_requested): raise ValueError(f"history frame selection mismatch: {relative}")
  if action.shape!=(len(frame_index),28) or arm.ndim!=2 or hand.ndim!=2 or leg.ndim!=2 or not all(np.isfinite(x).all() for x in (timestamp,action,arm,hand,leg)):
   raise ValueError(f"final observed history schema/finite failure: {relative}")
  frame_to_pos={int(f):i for i,f in enumerate(frame_index.tolist())}
  for ix in indices:
   row=rows[ix]; a=int(row["anchor_frame"]); anchor_id=row["anchor_id"]
   if a not in frame_to_pos: raise ValueError(f"final observed anchor row is missing: {anchor_id}")
   t0=float(timestamp[frame_to_pos[a]])
   query=t0+np.arange(-31,1,dtype=np.float64)*0.125
   selected=np.searchsorted(timestamp,query,side="right")-1
   if np.any(selected<0) or np.any(frame_index[selected]<a-119) or np.any(frame_index[selected]>a) or np.any(timestamp[selected]>query+1e-9):
    raise ValueError(f"final historical query alignment failure: {anchor_id}")
   history[ix]=np.concatenate((arm[selected],hand[selected],leg[selected],action[selected]),axis=1).astype(np.float32)
   rel[ix]=(query-t0).astype(np.float32)
   anchor_native[ix]=action[frame_to_pos[a]].astype(np.float32)
   if history[ix].shape!=(32,71) or not np.isfinite(history[ix]).all(): raise ValueError(f"final state prefix shape/finite failure: {anchor_id}")
  parquet_receipts.append({"parquet_path":relative,"file_size_bytes_metadata_only":int(parquet.stat().st_size),"history_source_rows_read":len(frame_index),"anchors":len(indices),"maximum_frame_read":max(frames_requested)})
 episodes=np.asarray([int(r["episode_id"]) for r in rows],dtype=np.int64)
 tasks=np.asarray([int(r["task_id"]) for r in rows],dtype=np.int64)
 anchors=np.asarray([r["anchor_id"] for r in rows],dtype="U64")
 canonical=np.asarray([r["canonical_id"] for r in rows],dtype="U768")
 source=np.asarray([r["source_key"] for r in rows],dtype="U256")
 literal=np.asarray([r["literal_source_key"] for r in rows],dtype="U512")
 cache=np.asarray([int(r["cache_index"]) for r in rows],dtype=np.int64)
 anchor_frames=np.asarray([int(r["anchor_frame"]) for r in rows],dtype=np.int64)
 cluster=np.asarray([pop_by[r["anchor_id"]]["source_group_key"] for r in rows],dtype="U256")
 if len(set(canonical.tolist()))!=162 or len(set(anchors.tolist()))!=162 or len(set(cache.tolist()))!=162: raise ValueError("final prefix identities/cache indexes are not unique")
 arrays={"anchor_id":anchors,"episode_id":episodes,"task_id":tasks,
  "task_rank":np.asarray([TASK_RANK[int(t)] for t in tasks],dtype=np.int8),"source_cluster":cluster,
  "cache_index":cache,"anchor_frame":anchor_frames,"history_native":history,"history_mask":mask,
  "relative_delta":rel,"anchor_native":anchor_native,"canonical_id":canonical,
  "dataset_revision":np.full(162,"6b6b599def75e3558b2736062e79bdbeec6cd224",dtype="U64"),
  "source_key":source,"literal_source_key":literal,"split":np.full(162,"final_test",dtype="U16")}
 OUT_PREFIX.parent.mkdir(parents=True,exist_ok=True);tmp=OUT_PREFIX.with_name(f".{OUT_PREFIX.name}.{os.getpid()}.part")
 with tmp.open("wb") as f: np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(tmp,OUT_PREFIX)
 receipt={"schema_version":1,"status":"PASS_FINAL_PREFIX_ONLY_AFTER_ACCESS","created_utc":utc(),
  "current_profile_lock_sha256":args.expected_lock_sha256,"core_b_v2_final_freeze_sha256":args.expected_final_freeze_sha256,
  "final_access_opened_sha256":args.expected_final_access_sha256,"final_input_manifest_sha256":sha(INPUTS),
  "rows":162,"episode_support":len(set(episodes.tolist())),"task_support":{str(t):int(np.sum(tasks==t)) for t in TASKS},
  "canonical_id_order_sha256":hashlib.sha256("\n".join(canonical.tolist()).encode()).hexdigest(),
  "prefix_path":str(OUT_PREFIX),"prefix_sha256":sha(OUT_PREFIX),"prefix_shape":{"history_native":list(history.shape),"anchor_native":list(anchor_native.shape)},
  "raw_prefix_action_values_materialized":int(history.shape[0]*history.shape[1]*history.shape[2]),
  "future_target_array_materialized":False,"target_h30_rows_materialized":0,"final_payload_access":True,
  "parquet_prefix_source_receipts":parquet_receipts,"history_query":"anchor timestamp + arange(-31,1)*0.125; original V9 input semantics",
  "future_target_read_rule":"none; this builder queried only rows <= each anchor's legal history window"}
 atomic_json(OUT_RECEIPT,receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
