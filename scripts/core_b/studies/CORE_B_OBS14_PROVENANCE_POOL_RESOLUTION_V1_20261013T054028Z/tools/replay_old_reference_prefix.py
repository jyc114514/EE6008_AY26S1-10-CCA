#!/usr/bin/env python3
"""Compare a few old-train raw records with accepted prefix inputs; never reads targets."""
import os
from pathlib import Path
import argparse,csv,hashlib,json,time,resource,shutil,zipfile
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OBS=PROJECT/'reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z'
RAW=PROJECT/'data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d'
ASSET_SUMS=PROJECT/'repo/ee6008-humanoid-vjepa/manifests/asset_acquisition/G1_SHA256SUMS_20260825T094953Z.txt'
EXPECTED={'arm':14,'hand':14,'leg':15,'action':28}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
 t0=time.perf_counter(); c0=time.process_time(); threadpool_limits(limits=8)
 inp_path=OBS/'data/train/inputs_prefix.npz'
 with np.load(inp_path,allow_pickle=False) as z:
  ids=z['anchor_id'].astype(str).copy(); eps=z['episode_index'].astype(np.int64).copy(); frames=z['anchor_frame'].astype(np.int64).copy()
  groups=z['source_group_id'].astype(str).copy(); timestamps=z['anchor_timestamp'].astype(np.float64).copy()
  s_prev=z['s_prev'].copy(); s0=z['s0'].copy(); u0=z['u0'].copy()
 selected=[]
 for ep in (0,1,3):
  ix=np.where(eps==ep)[0]
  if not len(ix): raise RuntimeError(f'old train fixture episode {ep} absent from locked input arrays')
  i=int(ix[np.argmin(frames[ix])]); selected.append(i)
 raw_manifest=OBS/'source/RAW_PARQUET_READ_MANIFEST.csv'
 with raw_manifest.open(encoding='utf-8',newline='') as f: raw_by_ep={int(r['episode_index']):r for r in csv.DictReader(f)}
 sums={}
 for line in ASSET_SUMS.read_text().splitlines():
  if line.startswith('#') or not line.strip():continue
  h,name=line.split(None,1); sums[name.strip().lstrip('./')]=h
 rows=[]
 for i in selected:
  ep=int(eps[i]); a=int(frames[i]); record=raw_by_ep[ep]
  if record['split']!='train': raise RuntimeError('selected reference episode is not old train')
  path=(RAW/record['parquet_path']).resolve()
  if not path.is_relative_to(RAW.resolve()): raise RuntimeError('raw path escaped known G1 snapshot')
  table=pq.read_table(path,columns=['frame_index','timestamp','action','observation.arm_joints','observation.hand_joints','observation.leg_joints'],use_threads=False)
  frame=table['frame_index'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
  ts=table['timestamp'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
  act=np.asarray(table['action'].to_pylist(),dtype=np.float32)
  arm=np.asarray(table['observation.arm_joints'].to_pylist(),dtype=np.float32)
  hand=np.asarray(table['observation.hand_joints'].to_pylist(),dtype=np.float32)
  leg=np.asarray(table['observation.leg_joints'].to_pylist(),dtype=np.float32)
  if (act.shape[1:]!=(EXPECTED['action'],) or arm.shape[1:]!=(EXPECTED['arm'],) or hand.shape[1:]!=(EXPECTED['hand'],) or leg.shape[1:]!=(EXPECTED['leg'],)):
   raise RuntimeError(f'field shape mismatch at old train episode {ep}')
  obs=np.concatenate((arm,hand,leg),axis=1)
  fs=[float(np.max(np.abs(s_prev[i]-obs[a-1]))),float(np.max(np.abs(s0[i]-obs[a]))),float(np.max(np.abs(u0[i]-act[a])))]
  path_rel=record['parquet_path']; expected_sha=sums.get(path_rel)
  actual_sha=sha(path)
  row={'episode_index':ep,'source_group_id_sha256':hashlib.sha256(groups[i].encode()).hexdigest(),'anchor_id_sha256':hashlib.sha256(ids[i].encode()).hexdigest(),'anchor_frame':a,'raw_relative_path':path_rel,'raw_file_bytes':path.stat().st_size,'raw_file_sha256':actual_sha,'import_manifest_sha256':expected_sha,'raw_hash_matches_import_manifest':actual_sha==expected_sha,'columns_read':['frame_index','timestamp','action','observation.arm_joints','observation.hand_joints','observation.leg_joints'],'shape':{'action':list(act.shape[1:]),'arm':list(arm.shape[1:]),'hand':list(hand.shape[1:]),'leg':list(leg.shape[1:])},'frame_grid_contiguous':bool(np.array_equal(frame,np.arange(len(frame)))),'timestamp_finite_monotone':bool(np.isfinite(ts).all() and np.all(np.diff(ts)>0)),'max_abs_timestamp_minus_frame_over_30_s':float(np.max(np.abs(ts-frame.astype(np.float64)/30.0))),'max_abs_timestamp_interval_minus_1_over_30_s':float(np.max(np.abs(np.diff(ts)-1/30))),'stored_prefix_max_abs_diffs_s_prev_s0_u0':fs,'stored_anchor_timestamp_abs_diff_s':abs(float(ts[a])-float(timestamps[i])),'exact_prefix_match':all(x==0.0 for x in fs)}
  rows.append(row)
 if not all(r['raw_hash_matches_import_manifest'] and r['exact_prefix_match'] for r in rows): raise RuntimeError('old train source/prefix parity failure')
 report={'status':'PASS_OLD_TRAIN_PREFIX_SOURCE_PARITY','scope':'three old primary-train episodes only; not prospective data','source_dataset_revision':'USC-PSI-Lab/Humanoid-Everyday-G1@6b6b599def75e3558b2736062e79bdbeec6cd224','accepted_extractor_hash_only':'31e8f2d153d5c48592883069266471213d5631ecfeb64d45dccd9edea058bc77','accepted_extractor_code_bytes_preserved':False,'current_final_locked_pipeline_sha256':'514c7ad155e177b1a54f5fe1766a44057d1913314bc699ccea1c8ba3bf9f8a33','field_transform_replayed':'arm14 + hand14 + leg15; action28 passed through as stored; float32 matrix cast; no semantic channel names inferred','reference_rows':rows,'source_prefix_array_sha256':sha(inp_path),'new_candidate_payloads_read':0,'new_labels_read':0,'future_observations_read':False,'final_or_heldout_payload_access':0,'gpu_actions':0,'new_model_fits':0,'wall_seconds':time.perf_counter()-t0,'process_cpu_seconds':time.process_time()-c0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),'disk_free_bytes':shutil.disk_usage(Path(a.out).parent).free}
 Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
