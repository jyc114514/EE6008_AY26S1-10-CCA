#!/usr/bin/env python3
"""Accept all Q0-reused and Q1-new prediction bundles without opening targets."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SCHED=PROJECT/'reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z'
INPUTS=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz'

def now():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sha_query(values):
 a=np.ascontiguousarray(values,dtype=np.float32);h=hashlib.sha256();h.update(a.dtype.str.encode('ascii'));h.update(b'\0');h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes());return h.hexdigest()
def sha_ids(values):return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_once(path,obj):
 p=Path(path)
 if p.exists():raise FileExistsError(f'preserving existing acceptance output: {p}')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def load_metadata():
 # Explicitly read metadata/input keys only; never access targets_native or any target cache.
 with np.load(INPUTS,allow_pickle=False) as z:
  split=z['split_code'].astype(np.int8,copy=True)
  ids=z['anchor_id'].astype('U96',copy=True)
  eps=z['episode_index'].astype(np.int64,copy=True)
  tasks=z['original_task_index'].astype(np.int64,copy=True)
  frames=z['anchor_frame'].astype(np.int64,copy=True)
  times=z['anchor_timestamp'].astype(np.float64,copy=True)
  anchors=z['anchor_command_native'].astype(np.float32,copy=True)
 if len(split)!=20079 or not np.array_equal(np.unique(split),np.array([0,1],dtype=np.int8)):raise AssertionError('input metadata split schema mismatch')
 group_map={}
 with (ROOT/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):
   ep=int(row['episode_index']);value=(row['split'],row['source_group_id'])
   if ep in group_map and group_map[ep]!=value:raise AssertionError(f'conflicting source group for episode {ep}')
   group_map[ep]=value
 meta={}
 for i,aid in enumerate(ids.tolist()):
  split_name='train' if int(split[i])==0 else 'dev'
  mapped,group=group_map[int(eps[i])]
  if mapped!=split_name:raise AssertionError(f'source group split mismatch for {aid}')
  meta[aid]={'split':split_name,'episode_index':int(eps[i]),'original_task_index':int(tasks[i]),'anchor_frame':int(frames[i]),'anchor_timestamp':float(times[i]),'anchor_command_native':anchors[i].copy(),'source_group_id':group}
 expected={'train':17757,'dev':2322}
 for sp,n in expected.items():
  ids_sp=[a for a in ids.tolist() if meta[a]['split']==sp]
  if len(ids_sp)!=n or len(set(ids_sp))!=n:raise AssertionError(f'{sp} input anchor IDs mismatch')
 return meta,expected
def validate_arrays(path,meta,expected_rows,config,seed,split,query_sha,new_format):
 with np.load(path,allow_pickle=False) as z:
  keys=set(z.files)
  forbidden={'targets_native','target_native','targets_standardized','target_standardized'}
  if keys & forbidden:raise AssertionError(f'prediction bundle contains target arrays: {sorted(keys & forbidden)}')
  ids=z['anchor_id'].astype('U96');eps=z['episode_index'].astype(np.int64);tasks=z['original_task_index'].astype(np.int64)
  frames=z['anchor_frame'].astype(np.int64);times=z['anchor_timestamp'].astype(np.float64);pred=z['predictions_native'].astype(np.float32,copy=True)
  if len(ids)!=expected_rows or len(set(ids.tolist()))!=expected_rows or pred.shape!=(expected_rows,30,28) or not np.isfinite(pred).all():raise AssertionError(f'prediction array count/shape/finite mismatch {path}')
  if set(ids.tolist())!={aid for aid,v in meta.items() if v['split']==split}:raise AssertionError(f'prediction anchor IDs do not cover {split} manifest')
  steps=np.arange(1,31,dtype=np.int64);expected_frames=frames[:,None]+steps[None,:];expected_times=times[:,None]+steps[None,:]/30.0
  if new_format:
   groups=z['source_group_id'].astype('U128');stored_frames=z['target_frame_indices'].astype(np.int64);stored_times=z['target_timestamp_seconds'].astype(np.float64);stored_anchors=z['anchor_command_native'].astype(np.float32)
   if not np.array_equal(stored_frames,expected_frames) or not np.allclose(stored_times,expected_times,atol=1e-9,rtol=0):raise AssertionError(f'target frame/time formula mismatch {path}')
  else:
   groups=np.asarray([meta[x]['source_group_id'] for x in ids.tolist()],dtype='U128');stored_anchors=np.asarray([meta[x]['anchor_command_native'] for x in ids.tolist()],dtype=np.float32)
  for i,aid in enumerate(ids.tolist()):
   m=meta[aid]
   if int(eps[i])!=m['episode_index'] or int(tasks[i])!=m['original_task_index'] or int(frames[i])!=m['anchor_frame'] or abs(float(times[i])-m['anchor_timestamp'])>1e-8:raise AssertionError(f'anchor metadata mismatch {aid}')
   if str(groups[i])!=m['source_group_id'] or not np.array_equal(stored_anchors[i],m['anchor_command_native']):raise AssertionError(f'group/anchor native metadata mismatch {aid}')
  if int(z['horizon_steps'].item())!=30 or str(z['config'].item())!=config or int(z['seed'].item())!=seed or str(z['split'].item())!=split:raise AssertionError(f'prediction scalar metadata mismatch {path}')
  if new_format and str(z['query_table_sha256'].item())!=query_sha:raise AssertionError(f'new prediction query hash mismatch {path}')
 return {'rows':expected_rows,'prediction_shape':[expected_rows,30,28],'ids_sha256':sha_ids(ids),'anchor_ids_unique':True,'source_group_metadata_joined':True,'target_frame_time_formula':'anchor_frame+step; anchor_timestamp+step/30 for step=1..30','native_anchor_metadata_validated':True,'target_arrays_present':False,'prediction_finite':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('q0-only','full'),default='full');args=ap.parse_args()
 lock=readj(ROOT/'protocol/PROTOCOL_LOCKED.json');lock_sha=sha(ROOT/'protocol/PROTOCOL_LOCKED.json')
 amendment1=readj(ROOT/'protocol/AMENDMENT_001_QUERY_ARCHIVE.json');amendment1_sha=sha(ROOT/'protocol/AMENDMENT_001_QUERY_ARCHIVE.json')
 amendment2_path=ROOT/'protocol/AMENDMENT_002_PREDICTION_ACCEPTANCE.json'
 if not amendment2_path.is_file():raise RuntimeError('prediction acceptance amendment is absent')
 amendment2=readj(amendment2_path);amendment2_sha=sha(amendment2_path);receipt=readj(ROOT/'coordination/AMENDMENT_002_RECEIPT.json')
 own_sha=sha(Path(__file__).resolve())
 if (amendment2.get('status')!='LOCKED_IMPLEMENTATION_AMENDMENT'
     or amendment2.get('parent_protocol_sha256')!=lock_sha
     or amendment2.get('parent_amendment_001_sha256')!=amendment1_sha
     or amendment2.get('task_code_sha256_overrides',{}).get('code/accept_predictions.py')!=own_sha
     or receipt.get('amendment_sha256')!=amendment2_sha):raise AssertionError('prediction acceptance amendment/receipt/hash mismatch')
 query=readj(ROOT/'protocol/QUERY_TABLES_AND_HASHES.json')
 with np.load(ROOT/'protocol/HADAMARD_QUERY_TABLES.npz',allow_pickle=False) as z:old_table=np.asarray(z['old_future_h30'],dtype=np.float32).copy();candidate=np.asarray(z['candidate_h30'],dtype=np.float32).copy()
 old_sha=sha_query(old_table);new_sha=sha_query(candidate)
 if new_sha!=lock['query_table_sha256_h30'] or old_sha==new_sha:raise AssertionError('old/candidate query identity mismatch')
 meta,counts=load_metadata()
 with (ROOT/'protocol/MATRIX12.csv').open(newline='',encoding='utf-8') as f:matrix=list(csv.DictReader(f))
 with (ROOT/'protocol/REUSE_ELIGIBILITY.csv').open(newline='',encoding='utf-8') as f:reuse=list(csv.DictReader(f))
 reuse_by_q0={x['q0_run_id']:x for x in reuse}
 old_accept=readj(SCHED/'PREDICTION_ACCEPTANCE.json');old_index={(x['run_id'],x['split']):x for x in old_accept['bundles']}
 bundle_records=[];replay_records=[]
 for row in matrix:
  q0=row['action']=='reuse';config=row['config'];seed=int(row['seed']);cell_id=row['run_id']
  if args.mode=='q0-only' and not q0:continue
  if q0:
   ref=reuse_by_q0[cell_id];source_run=ref['source_run_id'];run_dir=SCHED/'runs/formal'/source_run;complete=readj(run_dir/'RUN_COMPLETE.json');checkpoint_sha=ref['checkpoint_sha256'];query_sha=old_sha;source='reused_control'
   if complete.get('status')!='PASS' or sha(run_dir/'checkpoint_last.pt')!=checkpoint_sha or int(complete.get('optimizer_steps',-1))!=11120:raise AssertionError(f'Q0 checkpoint/source mismatch {source_run}')
  else:
   source_run=cell_id;run_dir=ROOT/'runs/formal'/cell_id;complete=readj(run_dir/'RUN_COMPLETE.json');checkpoint_sha=sha(run_dir/'checkpoint_last.pt');query_sha=new_sha;source='new'
   if (complete.get('status')!='PASS' or int(complete.get('epochs',-1))!=40 or int(complete.get('optimizer_steps',-1))!=11120 or complete.get('study_protocol_hash')!=lock_sha or complete.get('query_table_sha256')!=new_sha or complete.get('checkpoint_last_sha256')!=checkpoint_sha):raise AssertionError(f'Q1 run/checkpoint/protocol mismatch {cell_id}')
   amend=readj(ROOT/'protocol/AMENDMENT_001_QUERY_ARCHIVE.json');amend_sha=sha(ROOT/'protocol/AMENDMENT_001_QUERY_ARCHIVE.json')
   if complete.get('protocol_amendment_sha256')!=amend_sha:raise AssertionError(f'Q1 checkpoint amendment provenance mismatch {cell_id}')
   replay=readj(run_dir/'RUN_RELOAD_ACCEPTANCE.json')
   if replay.get('status')!='PASS' or not replay.get('prediction_allclose') or abs(float(replay.get('max_abs_diff',float('inf'))))>1e-5:raise AssertionError(f'fresh-process replay not accepted {cell_id}')
   replay_records.append({'run_id':cell_id,'checkpoint_sha256':checkpoint_sha,'query_table_sha256':new_sha,'status':'PASS','max_abs_diff':replay['max_abs_diff'],'max_rel_diff':replay['max_rel_diff'],'path':str(run_dir/'RUN_RELOAD_ACCEPTANCE.json')})
  for split in ('train','dev'):
   expected=counts[split]
   if q0:
    pred_path=SCHED/'predictions'/split/f'{source_run}.npz';side_path=SCHED/'predictions'/split/f'{source_run}.json';official=old_index.get((source_run,split));side=readj(side_path)
    if (not official or official.get('status')!='PASS' or int(official.get('rows',-1))!=expected or sha(pred_path)!=ref[f'{split}_prediction_sha256'] or sha(pred_path)!=official['prediction_sha256']):raise AssertionError(f'Q0 prediction acceptance/hash failed {source_run}/{split}')
    if side.get('status')!='PASS' or side.get('checkpoint_sha256')!=checkpoint_sha or side.get('prediction_sha256')!=sha(pred_path) or side.get('target_arrays_loaded') is not False:raise AssertionError(f'Q0 sidecar mismatch {source_run}/{split}')
    array_meta=validate_arrays(pred_path,meta,expected,config,seed,split,query_sha,False)
   else:
    pred_root=ROOT/'predictions'/split/cell_id
    accepted=[]
    for side_path in sorted(pred_root.glob('attempt_*.json')):
     side=readj(side_path);pred_path=pred_root/side.get('prediction_file','')
     if side.get('status')=='PASS' and side.get('run_id')==cell_id and side.get('split')==split and pred_path.is_file() and sha(pred_path)==side.get('prediction_sha256'):accepted.append((side_path,pred_path,side))
    if not accepted:raise RuntimeError(f'missing accepted Q1 bundle {cell_id}/{split}')
    side_path,pred_path,side=accepted[-1]
    if (int(side.get('rows',-1))!=expected or side.get('config')!=config or int(side.get('seed',-1))!=seed or int(side.get('horizon_steps',-1))!=30 or side.get('checkpoint_sha256')!=checkpoint_sha or side.get('query_table_sha256')!=new_sha or side.get('study_protocol_sha256')!=lock_sha or side.get('target_arrays_loaded') is not False or side.get('dev_targets_opened_by_inference') is not False or side.get('optimizer_created') is not False or side.get('backward_called') is not False or int(side.get('optimizer_steps',-1))!=0):raise AssertionError(f'Q1 sidecar provenance/target-blind gate failed {cell_id}/{split}')
    array_meta=validate_arrays(pred_path,meta,expected,config,seed,split,new_sha,True)
   bundle_records.append({'run_id':cell_id,'source_run_id':source_run,'source':source,'config':config,'query_variant':'old_fixed_sinusoidal_time_query' if q0 else 'canonical_Hadamard_H30_RMS_matched','query_table_sha256':query_sha,'seed':seed,'horizon_steps':30,'split':split,'rows':expected,'prediction_shape':array_meta['prediction_shape'],'prediction_path':str(pred_path),'sidecar_path':str(side_path),'prediction_sha256':sha(pred_path),'checkpoint_sha256':checkpoint_sha,'input_archive_sha256':readj(ROOT/'protocol/INPUT_IDENTITIES.json')['V3_MAIN_INPUTS']['sha256'],'source_group_map_sha256':sha(ROOT/'inventory/SOURCE_GROUP_MAP.csv'),'feature_cache_manifest_sha256':readj(ROOT/'protocol/INPUT_IDENTITIES.json')['feature_cache_manifest_sha256'],'video_normalizer_sha256':readj(ROOT/'protocol/INPUT_IDENTITIES.json')['video_normalizer_sha256'],'target_normalizer_sha256':readj(ROOT/'protocol/INPUT_IDENTITIES.json')['target_normalizer_sha256'],'array_metadata_validation':array_meta,'target_arrays_loaded_during_inference':False,'final_test_access':0,'heldout_access':0})
 if args.mode=='q0-only':
  result={'schema_version':1,'status':'PASS_6_OF_6_Q0_REUSED_BUNDLES','created_utc':now(),'models_accepted':6,'train_bundles_accepted':'6/6','dev_bundles_accepted':'6/6','bundle_records':bundle_records,'acceptance_amendment_id':amendment2['amendment_id'],'acceptance_amendment_sha256':amendment2_sha,'acceptance_code_sha256':own_sha,'target_arrays_loaded':False,'dev_targets_opened':False,'final_test_access':0,'heldout_access':0}
  write_once(ROOT/'qa/Q0_REUSE_BUNDLE_AUDIT_AMENDMENT_002.json',result)
  print(json.dumps({'status':result['status'],'bundle_records':len(bundle_records)},indent=2));return 0
 if len(bundle_records)!=24:raise AssertionError(f'expected 24 prediction bundles, found {len(bundle_records)}')
 train_n=sum(1 for x in bundle_records if x['split']=='train');dev_n=sum(1 for x in bundle_records if x['split']=='dev')
 new_n=sum(1 for x in bundle_records if x['source']=='new');q0_n=sum(1 for x in bundle_records if x['source']=='reused_control')
 if (train_n,dev_n,new_n,q0_n)!=(12,12,12,12):raise AssertionError('accepted bundle coverage differs from locked 12/12 train/dev design')
 result={'schema_version':1,'status':'PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES','created_utc':now(),'train_bundles_accepted':'12/12','dev_bundles_accepted':'12/12','total_bundles_accepted':'24/24','q0_reused_bundles':'12/12','q1_new_bundles':'12/12','bundle_records':bundle_records,'replay_acceptance_records':replay_records,'replay_acceptance':'PASS_6_OF_6_FRESH_PROCESS_REPLAY','query_table_sha256_h30':new_sha,'old_query_table_sha256_h30':old_sha,'input_archive_metadata_read':True,'acceptance_amendment_id':amendment2['amendment_id'],'acceptance_amendment_sha256':amendment2_sha,'acceptance_code_sha256':own_sha,'target_arrays_loaded_during_inference':False,'dev_targets_opened_by_inference':False,'dev_targets_opened_by_acceptance':False,'final_test_access':0,'heldout_access':0}
 write_once(ROOT/'qa/PREDICTION_ACCEPTANCE.json',result)
 replay={'schema_version':1,'status':'PASS_6_OF_6_FRESH_PROCESS_REPLAY','created_utc':now(),'run_records':replay_records,'atol':1e-5,'rtol':1e-5,'cuda_initialization_by_acceptance':False,'target_arrays_loaded':False,'final_test_access':0}
 write_once(ROOT/'qa/REPLAY_ACCEPTANCE.json',replay)
 print(json.dumps({'status':result['status'],'train_bundles':train_n,'dev_bundles':dev_n,'new_q1_bundles':new_n,'reused_q0_bundles':q0_n,'replays':len(replay_records),'dev_targets_opened':False},indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
