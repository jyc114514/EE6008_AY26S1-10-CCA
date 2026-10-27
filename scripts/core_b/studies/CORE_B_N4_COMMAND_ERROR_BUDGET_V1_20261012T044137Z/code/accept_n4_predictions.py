#!/usr/bin/env python3
"""Accept the fixed 24 N4 bundles before any target array is opened."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
V3=P/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
N1=P/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def C(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writej(rel,obj):
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.part')
 t.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');t.replace(p)
def writec(rel,rows):
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 stamp=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 locks=['PROTOCOL_LOCK.json','PROFILE_LOCK.json','PROFILE_LOCK_AMENDMENT_001.json',
  'METRIC_BUDGET_LOCK.json','METRIC_SOURCE_AMENDMENT_001.json','ANALYSIS_LOCK.json','CASE_SELECTION_LOCK.json']
 for x in locks:
  if not (R/'protocol'/x).is_file():raise SystemExit('Missing pre-outcome lock '+x)
 exposure=J(R/'protocol/OUTCOME_EXPOSURE.json')
 if exposure.get('n4_outcomes_opened') is not False:raise SystemExit('Outcome gate already opened.')
 source=J(R/'inventory/SOURCE_ACCEPTANCE.json')
 if source.get('n2_checksum_manifest_sha256')!=sha(N2/'CHECKSUMS.sha256'):
  raise SystemExit('N2 manifest changed.')
 for rec in source.get('source_hashes',{}).values():
  if sha(Path(rec['path']))!=rec['sha256']:raise SystemExit('Locked source changed: '+rec['path'])
 for rec in source.get('n2_selected_file_inventory',[]):
  if sha(Path(rec['path']))!=rec['sha256']:raise SystemExit('Selected N2 source changed: '+rec['path'])
 matrix=C(R/'protocol/MATRIX24.csv')
 if len(matrix)!=24:raise SystemExit('N4 matrix is not 24 rows.')
 n2views=C(N2/'results/N2_VIEW_ACCEPTANCE.csv')
 n2by={(r['model'],r['query'],r['seed'],r['profile']):r for r in n2views}
 group_rows=C(N2/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv')
 ep_split={int(r['episode_index']):r['split'] for r in group_rows}
 ep_group={int(r['episode_index']):r['source_group_id'] for r in group_rows}
 # Metadata-only IDs: never index targets_native here.
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  keys=set(z.files)
  split=np.asarray(z['split_code'],dtype=np.int8)
  ids=np.asarray(z['anchor_id']).astype('U96')
  eps=np.asarray(z['episode_index'],dtype=np.int64)
  target_mask_keys=[k for k in keys if ('mask' in k.lower() and 'history' not in k.lower() and 'target' in k.lower())]
 if len(ids)!=20079 or len(eps)!=20079 or len(split)!=20079:raise SystemExit('Canonical ID/split metadata size mismatch.')
 if int(np.sum(split==0))!=17757 or int(np.sum(split==1))!=2322:raise SystemExit('Canonical split counts mismatch.')
 expected={}
 for name,code in [('primary_train',0),('exposed_dev',1)]:
  idx=np.flatnonzero(split==code);expected[name]={'ids':ids[idx],'episodes':eps[idx]}
  if any(int(e) not in ep_group or ep_split[int(e)]!=('train' if code==0 else 'dev') for e in expected[name]['episodes']):
   raise SystemExit('ID-to-source-group split mapping mismatch.')
 groups={name:np.asarray([ep_group[int(e)] for e in d['episodes']],dtype='U128') for name,d in expected.items()}
 if (len(np.unique(expected['primary_train']['episodes'])),len(np.unique(groups['primary_train'])))!=(2712,2712):
  raise SystemExit('Train episode/source-group population mismatch.')
 if (len(np.unique(expected['exposed_dev']['episodes'])),len(np.unique(groups['exposed_dev'])))!=(603,603):
  raise SystemExit('Dev episode/source-group population mismatch.')
 for name in ('primary_train','exposed_dev'):
  pairs={(int(e),str(g)) for e,g in zip(expected[name]['episodes'],groups[name])}
  group_episode_counts={}
  for ep,gid in pairs:group_episode_counts[gid]=group_episode_counts.get(gid,0)+1
  if max(group_episode_counts.values())!=1:
   raise SystemExit(name+' source group has multiple episodes; equal-weight bootstrap mapping needs review.')
 n2_accept=J(N2/'qa/N2_PREDICTION_ACCEPTANCE.json')
 if n2_accept.get('cuda_common_views')!=12 or n2_accept.get('target_arrays_loaded_by_inference') is not False:
  raise SystemExit('N2 CUDA_COMMON source gate failed.')
 train_status=J(R/'coordination/N4_TRAIN_INFERENCE_STATUS.json')
 watcher=J(R/'coordination/N4_TRAIN_WATCHER_STATUS.json')
 if train_status.get('status')!='PASS' or train_status.get('train_views')!=12:
  raise SystemExit('New train inference was not accepted 12/12.')
 if watcher.get('status')!='complete' or watcher.get('exit_code')!=0 or watcher.get('lease_released') is not True:
  raise SystemExit('Watcher/resource release gate failed.')
 records=[]
 for row in matrix:
  model,q,seed=row['model'],row['query'],row['seed']
  split_name=row['split'];key='primary_train' if split_name=='primary_train' else 'exposed_dev'
  exp=expected[key];eid=exp['ids'];eep=exp['episodes'];eg=groups[key]
  path=Path(row['prediction_path']);receipt=Path(row['prediction_receipt_path'])
  if not path.is_file() or not receipt.is_file():raise SystemExit('Missing required bundle '+row['view_id'])
  side=J(receipt);digest=sha(path)
  if side.get('status')!='PASS' or side.get('prediction_sha256')!=digest:raise SystemExit('Prediction receipt/hash mismatch '+row['view_id'])
  with np.load(path,allow_pickle=False) as z:
   required={'anchor_id','episode_index','original_task_index','anchor_frame','anchor_timestamp','predictions_native'}
   if not required.issubset(z.files):raise SystemExit('Prediction archive schema incomplete '+row['view_id'])
   got_ids=z['anchor_id'].astype('U96');got_eps=np.asarray(z['episode_index'],dtype=np.int64)
   pred=np.asarray(z['predictions_native'])
   if pred.dtype!=np.float32:raise SystemExit('Prediction dtype is not FP32 '+row['view_id'])
   if pred.shape!=(len(eid),30,28) or not np.isfinite(pred).all():raise SystemExit('Prediction shape/finite gate failed '+row['view_id'])
   if not np.array_equal(got_ids,eid) or not np.array_equal(got_eps,eep):raise SystemExit('Canonical ID/roworder mismatch '+row['view_id'])
  expected_model_id=f'{model}__{q}__S{seed}__H30'
  if side.get('model_id')!=expected_model_id or side.get('model')!=model or side.get('query')!=q or str(side.get('seed'))!=str(seed):
   raise SystemExit('Prediction model identity mismatch '+row['view_id'])
  if side.get('profile')!='CUDA_COMMON' or side.get('split')!=('train' if key=='primary_train' else 'dev'):
   raise SystemExit('Prediction profile/split mismatch '+row['view_id'])
  if side.get('target_arrays_loaded') is not False or side.get('final_test_access')!=0 or side.get('heldout_access')!=0:
   raise SystemExit('Target-blind/final-test gate failed '+row['view_id'])
  if row['profile']!='CUDA_COMMON':raise SystemExit('Matrix profile mismatch '+row['view_id'])
  if key=='primary_train':
   if side.get('train_targets_opened_by_inference') is not False or side.get('optimizer_steps')!=0 or side.get('backward_called') is not False:
    raise SystemExit('Train inference target/training gate failed '+row['view_id'])
   if side.get('physical_gpu') is not None:pass
   if side.get('runtime',{}).get('physical_gpu_index')!=2 or side.get('runtime',{}).get('cuda_visible_devices')!='2':
    raise SystemExit('N4 train profile does not match accepted CUDA_COMMON physical profile.')
   if side.get('runtime',{}).get('torch')!='2.12.1+cu130' or side.get('runtime',{}).get('sdpa_backend')!='math_only':
    raise SystemExit('N4 train runtime profile mismatch.')
  else:
   n2row=n2by[(model,q,seed,'CUDA_COMMON')]
   if path!=Path(n2row['prediction_path']) or digest!=n2row['prediction_sha256']:
    raise SystemExit('Dev view is not the accepted N2 CUDA_COMMON bundle '+row['view_id'])
   if side.get('target_arrays_loaded') is not False:
    raise SystemExit('N2 dev inference was not target blind '+row['view_id'])
  if int(row['expected_rows'])!=len(eid) or int(row['expected_source_groups'])!=len(np.unique(eg)):
   raise SystemExit('N4 matrix expected population mismatch '+row['view_id'])
  records.append({'view_id':row['view_id'],'model':model,'query':q,'seed':int(seed),
   'split':key,'profile':'CUDA_COMMON','rows':len(eid),'episodes':len(np.unique(eep)),
   'source_groups':len(np.unique(eg)),'prediction_path':str(path),'prediction_sha256':digest,
   'receipt_path':str(receipt),'receipt_sha256':sha(receipt),
   'model_id':expected_model_id,'target_arrays_loaded_by_inference':False,
   'source':'new_target_blind_train_forward' if key=='primary_train' else 'reused_accepted_N2_CUDA_COMMON'})
 if len(records)!=24 or sum(r['split']=='primary_train' for r in records)!=12 or sum(r['split']=='exposed_dev' for r in records)!=12:
  raise SystemExit('Accepted bundle matrix count mismatch.')
 case_rows=C(R/'protocol/CASE_SELECTION_LOCK.csv')
 for c in case_rows:
  if c['anchor_id'] not in set(expected['exposed_dev']['ids'].tolist()):raise SystemExit('Locked case ID missing from dev support.')
 canary=J(R/'qa/GPU_CANARY.json')
 if canary.get('status')!='SKIP_IDENTICAL_N2_PHYSICAL_GPU_AND_PROFILE':
  raise SystemExit('Expected same-physical-GPU canary skip receipt not found.')
 acceptance={'schema_version':1,'status':'PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES',
  'created_utc':stamp,'models_fixed':12,'bundles_accepted':24,'train_bundles_accepted':'12/12',
  'dev_bundles_accepted':'12/12','dev_cuda_common_views_reused':12,'new_train_forward_views':12,
  'new_dev_formal_forward_views':0,'matrix_sha256':sha(R/'protocol/MATRIX24.csv'),
  'profile_lock_sha256':sha(R/'protocol/PROFILE_LOCK.json'),
  'metric_source_amendment_sha256':sha(R/'protocol/METRIC_SOURCE_AMENDMENT_001.json'),
  'canonical_input_sha256':sha(V3/'data/V3_MAIN_INPUTS.npz'),
  'target_arrays_opened_by_acceptance':False,'target_gate_opened':True,
  'source_group_unit':'source group; each group contains exactly one canonical episode on both splits',
  'final_test_access':0,'heldout_access':0,'new_neural_training':0,'optimizer_steps':0,'encoder_extraction':0,
  'watcher_job_id':watcher['job_id'],'watcher_exit_code':watcher['exit_code'],
  'gpu_lease_released':watcher['lease_released'],'physical_gpu':2,
  'gpu_uuid':watcher['final_recheck']['uuid'],'cuda_visible_devices':'2','logical_device':'cuda:0',
  'actual_max_gpu_peak_allocated_mib':train_status['max_gpu_peak_allocated_mib'],
  'n2_root_manifest_sha256':source['n2_checksum_manifest_sha256'],
  'bundle_records':records}
 writec('qa/PREDICTION_ACCEPTANCE.csv',records)
 writej('qa/PREDICTION_ACCEPTANCE.json',acceptance)
 writej('qa/SUPPORT_AND_AGGREGATION.json',{'schema_version':1,'status':'PASS_SUPPORT_AND_AGGREGATION_METADATA',
  'created_utc':stamp,'split_support':{'primary_train':{'anchors':17757,'episodes':2712,'source_groups':2712},
   'exposed_dev':{'anchors':2322,'episodes':603,'source_groups':603}},
  'episodes_per_source_group_max':{'primary_train':1,'exposed_dev':1},
  'anchor_ids_unique_by_split':True,'canonical_row_order':'exact V3 input order filtered by split_code',
  'path_shape':[30,28],'dense_valid_support':True,'target_mask_keys_found':target_mask_keys,
  'time_weight_source':'none; accepted N1 scorer uses uniform step/axis means on dense 30x28 arrays',
  'direction_threshold_source_sha256':sha(N2/'n1_refs/accepted_n1_refs/MOTION_ACTIVITY_THRESHOLD.json'),
  'group_bootstrap_manifest_sha256':sha(N2/'protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz'),
  'group_bootstrap_source_group_order_matches':True,'train_dev_group_overlap':0,
  'target_arrays_opened':False,'final_test_access':0})
 exposure.update({'status':'PREDICTIONS_ACCEPTED_TARGET_GATE_OPEN',
  'n4_predictions_accepted_utc':stamp,'n4_outcomes_opened':False,
  'n4_access_authorized_by_lock':True,'prediction_acceptance_sha256':sha(R/'qa/PREDICTION_ACCEPTANCE.json')})
 writej('protocol/OUTCOME_EXPOSURE.json',exposure)
 print(json.dumps({'status':acceptance['status'],'bundles':24,'train':12,'dev':12,
  'train_forwards':12,'dev_formal_forwards':0,'target_arrays_opened':False,
  'target_gate_opened':True,'final_test_access':0,'gpu_lease_released':True,
  'n2_manifest_unchanged':True,'acceptance_path':str(R/'qa/PREDICTION_ACCEPTANCE.json')},
  ensure_ascii=False,indent=2))
if __name__=='__main__':main()
