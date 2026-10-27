#!/usr/bin/env python3
"""Target-blind acceptance gate for all locked N2 prediction views."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
MATRIX=ROOT/'protocol/N2_MATRIX24.csv'
REQUEST=ROOT/'MODEL_EXPORT_REQUEST.csv'
INPUTS=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz'
GROUP_MAP=ROOT/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv'
CACHE=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json'
VNORM=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_VIDEO_NORMALIZER.json'
TNORM=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_NORMALIZERS.json'
LOCKED_MATRIX_SHA='70e81b236727c617aeadc21e67dbb0412d6a8c56a2ab3ec03151e077897a6ffd'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_json(p:Path,o)->None:p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def main()->int:
 if sha(MATRIX)!=LOCKED_MATRIX_SHA:raise RuntimeError('locked matrix SHA mismatch')
 with MATRIX.open(newline='',encoding='utf-8') as f:matrix=list(csv.DictReader(f))
 with REQUEST.open(newline='',encoding='utf-8') as f:requests=list(csv.DictReader(f))
 if len(matrix)!=24 or len({r['view_id'] for r in matrix})!=24:raise RuntimeError('matrix row/ID count mismatch')
 req_by={(r['model'],r['query'],r['seed']):r for r in requests}
 with np.load(INPUTS,allow_pickle=False) as archive:
  split=archive['split_code'].astype(np.int8,copy=True)
  ids_all=archive['anchor_id'].astype('U96',copy=True)
  episodes_all=archive['episode_index'].astype(np.int64,copy=True)
 idx=np.flatnonzero(split==1)
 if len(idx)!=2322:raise RuntimeError('canonical dev split count mismatch')
 expected_ids=ids_all[idx]
 if len(set(expected_ids.tolist()))!=2322:raise RuntimeError('canonical dev IDs are not unique')
 group_map={}
 with GROUP_MAP.open(newline='',encoding='utf-8') as f:
  for x in csv.DictReader(f):group_map[int(x['episode_index'])]=(x['split'],x['source_group_id'])
 dev_eps=episodes_all[idx]
 dev_groups=np.asarray([group_map[int(e)][1] for e in dev_eps],dtype='U128')
 if len(set(dev_groups.tolist()))!=603:raise RuntimeError('accepted dev source-group count mismatch')
 input_sha=sha(INPUTS);cache_sha=sha(CACHE);vnorm_sha=sha(VNORM);tnorm_sha=sha(TNORM)
 safe_reload=json.loads((ROOT/'qa/SAFE_ONLY_RELOAD.json').read_text())
 if safe_reload.get('status')!='PASS_SAFE_TENSORS_FRESH_PROCESS_RELOAD_AND_PROMOTION' or int(safe_reload.get('promoted_count',-1))!=12:
  raise RuntimeError('12/12 fresh-process safe tensor acceptance missing')
 matrix_rows=[];seen_ids=set();cuda_uuids=set();cuda_phys=set();max_peak=0
 for row in matrix:
  key=(row['model'],row['query'],row['seed'])
  req=req_by.get(key)
  if req is None:raise RuntimeError('matrix row has no export request '+row['view_id'])
  if row['checkpoint_sha256']!=req['checkpoint_sha256'] or row['checkpoint_path']!=req['checkpoint_path']:
   raise RuntimeError('matrix row checkpoint identity mismatch '+row['view_id'])
  out=ROOT/'predictions/dev'/row['view_id']
  pred_path=out/'primary.npz';side_path=out/'PREDICTION_COMPLETE.json'
  if not pred_path.is_file() or not side_path.is_file():raise RuntimeError('missing N2 output '+row['view_id'])
  side=json.loads(side_path.read_text())
  if side.get('status')!='PASS' or side.get('view_id')!=row['view_id'] or side.get('profile')!=row['profile']:
   raise RuntimeError('N2 sidecar status/view/profile mismatch '+row['view_id'])
  if int(side.get('rows',-1))!=2322 or int(side.get('expected_rows',-1))!=2322 or int(side.get('source_groups_expected',-1))!=603:
   raise RuntimeError('N2 sidecar support mismatch '+row['view_id'])
  if side.get('prediction_sha256')!=sha(pred_path):raise RuntimeError('N2 prediction SHA mismatch '+row['view_id'])
  if side.get('checkpoint_sha256')!=req['checkpoint_sha256'] or side.get('safe_model_sha256')!=sha(ROOT/'safe_models'/req['model_id']/'model.safetensors'):
   raise RuntimeError('N2 checkpoint/safe model identity mismatch '+row['view_id'])
  if side.get('input_sha256')!=input_sha or side.get('feature_cache_manifest_sha256')!=cache_sha or side.get('video_normalizer_sha256')!=vnorm_sha or side.get('target_normalizer_sha256')!=tnorm_sha:
   raise RuntimeError('N2 input/cache/normalizer identity mismatch '+row['view_id'])
  if side.get('target_arrays_loaded') is not False or side.get('dev_targets_opened_by_inference') is not False or int(side.get('final_test_access',-1))!=0:
   raise RuntimeError('target access boundary mismatch '+row['view_id'])
  if not side.get('strict_load') or not side.get('fresh_process_safe_tensor_reload'):
   raise RuntimeError('safe model load acceptance absent '+row['view_id'])
  with np.load(pred_path,allow_pickle=False) as z:
   ids=z['anchor_id'].astype('U96')
   pred=z['predictions_native']
   if pred.dtype!=np.float32 or pred.shape!=(2322,30,28) or not np.isfinite(pred).all():
    raise RuntimeError('prediction tensor dtype/shape/finite check failed '+row['view_id'])
   if not np.array_equal(ids,expected_ids):raise RuntimeError('prediction IDs or row order mismatch '+row['view_id'])
   if str(z['split'].item())!='dev' or int(z['horizon_steps'].item())!=30:
    raise RuntimeError('prediction metadata mismatch '+row['view_id'])
  runtime=side.get('runtime',{})
  if row['profile']=='CPU_COMMON':
   if runtime.get('backend')!='cpu' or runtime.get('device')!='cpu' or runtime.get('physical_gpu_index') is not None:
    raise RuntimeError('CPU profile device mismatch '+row['view_id'])
  else:
   if runtime.get('backend')!='cuda' or runtime.get('device')!='cuda:0' or runtime.get('cuda_visible_devices')!=str(runtime.get('physical_gpu_index')):
    raise RuntimeError('CUDA profile physical/logical mapping mismatch '+row['view_id'])
   cuda_uuids.add(runtime.get('gpu_uuid'));cuda_phys.add(runtime.get('physical_gpu_index'))
   peak=int(runtime.get('gpu_peak_allocated_mib',0));max_peak=max(max_peak,peak)
  if row['view_id'] in seen_ids:raise RuntimeError('duplicate view id')
  seen_ids.add(row['view_id'])
  matrix_rows.append({'view_id':row['view_id'],'model':row['model'],'query':row['query'],'seed':row['seed'],
   'profile':row['profile'],'rows':2322,'source_groups':603,'checkpoint_sha256':req['checkpoint_sha256'],
   'safe_model_sha256':side['safe_model_sha256'],'prediction_path':str(pred_path),'prediction_sha256':side['prediction_sha256'],
   'sidecar_path':str(side_path),'sidecar_sha256':sha(side_path),'status':'PASS','target_arrays_loaded':'false','final_test_access':0})
 if len(seen_ids)!=24 or len([x for x in matrix_rows if x['profile']=='CPU_COMMON'])!=12 or len([x for x in matrix_rows if x['profile']=='CUDA_COMMON'])!=12:
  raise RuntimeError('accepted N2 views do not cover 12 CPU + 12 CUDA')
 if len(cuda_uuids)!=1 or len(cuda_phys)!=1:raise RuntimeError('CUDA views did not use one physical GPU/UUID')
 p=ROOT/'results/N2_VIEW_ACCEPTANCE.csv'
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(matrix_rows[0]));w.writeheader();w.writerows(matrix_rows)
 result={'schema_version':1,'status':'PASS_24_OF_24_TARGET_BLIND_PREDICTIONS',
  'accepted_views':24,'cpu_common_views':12,'cuda_common_views':12,'dev_rows_per_view':2322,'dev_source_groups':603,
  'cpu_rows_in_canonical_order':True,'cuda_rows_in_canonical_order':True,'cuda_physical_gpu':next(iter(cuda_phys)),
  'cuda_uuid':next(iter(cuda_uuids)),'cuda_views_same_physical_gpu':True,'max_gpu_peak_allocated_mib':max_peak,
  'pre_registered_resource_peak_mib':86,'new_training':0,'optimizer_steps':0,'encoder_extraction':0,
  'target_arrays_loaded_by_inference':False,'development_targets_scored_yet':False,'final_test_access':0,
  'matrix_sha256':sha(MATRIX),'safe_only_reload_sha256':sha(ROOT/'qa/SAFE_ONLY_RELOAD.json'),
  'view_acceptance_csv_sha256':sha(p),'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
 write_json(ROOT/'qa/N2_PREDICTION_ACCEPTANCE.json',result)
 print(json.dumps(result,sort_keys=True),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
