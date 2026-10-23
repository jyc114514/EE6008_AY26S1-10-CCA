#!/usr/bin/env python3
"""Target-blind N2 common-profile inference from accepted safetensors only."""
from __future__ import annotations
import os
import argparse,bisect,csv,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
BENCH=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z'
REQUEST=ROOT/'MODEL_EXPORT_REQUEST.csv'
MATRIX=ROOT/'protocol/N2_MATRIX24.csv'
PROFILE_LOCK=ROOT/'protocol/N2_PROFILE_LOCK.json'
ANALYSIS_LOCK=ROOT/'protocol/ANALYSIS_LOCKED.json'
QUERY_ARCHIVE=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z/protocol/HADAMARD_QUERY_TABLES.npz'))
INPUTS=V3/'data/V3_MAIN_INPUTS.npz'
CACHE_MANIFEST=BENCH/'d3/cache/D3_CACHE_MANIFEST.json'
VIDEO_NORMALIZER=BENCH/'d3/cache/D3_VIDEO_NORMALIZER.json'
TARGET_NORMALIZER=V3/'data/V3_NORMALIZERS.json'
N2_MATRIX24_SHA256='70e81b236727c617aeadc21e67dbb0412d6a8c56a2ab3ec03151e077897a6ffd'
TARGET_BLIND_KEYS=('anchor_id','episode_index','original_task_index','anchor_frame','anchor_timestamp',
 'history_standardized','history_mask','history_relative_delta_standardized','anchor_command_native')

def utc_now()->str:return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sha_array(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha_query(a:np.ndarray)->str:
 x=np.ascontiguousarray(a,dtype=np.float32);h=hashlib.sha256()
 h.update(x.dtype.str.encode('ascii'));h.update(b'\0');h.update(np.asarray(x.shape,dtype=np.int64).tobytes());h.update(x.tobytes())
 return h.hexdigest()
def tensor_sha(t:torch.Tensor)->str:return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write_json(path:Path,obj:dict[str,Any])->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 part=path.with_name('.'+path.name+'.'+str(os.getpid())+'.part')
 part.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 os.replace(part,path)
def append_jsonl(path:Path,obj:dict[str,Any])->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a',encoding='utf-8') as f:f.write(json.dumps(obj,sort_keys=True,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def select_inputs(archive:Any,indices:np.ndarray)->dict[str,np.ndarray]:
 return {key:archive[key][indices].copy() for key in TARGET_BLIND_KEYS}

class GridStore:
 def __init__(self,manifest:dict[str,Any])->None:
  self.shards=sorted(manifest['shards'],key=lambda x:int(x['start_row']))
  self.starts=[int(x['start_row']) for x in self.shards]
 def get(self,indices:np.ndarray)->tuple[np.ndarray,np.ndarray]:
  dynamic=np.empty((len(indices),16,4,768),dtype=np.float32);static=np.empty_like(dynamic)
  grouped:dict[int,list[tuple[int,int]]]={}
  for out_i,row_i in enumerate(indices.astype(np.int64)):
   si=bisect.bisect_right(self.starts,int(row_i))-1;sh=self.shards[si]
   if not int(sh['start_row'])<=int(row_i)<int(sh['end_row']):raise IndexError('feature cache row gap')
   grouped.setdefault(si,[]).append((out_i,int(row_i)-int(sh['start_row'])))
  for si,pairs in grouped.items():
   values=np.load(self.shards[si]['array'],mmap_mode='r',allow_pickle=False)
   for out_i,local_i in pairs:
    dynamic[out_i]=values[local_i,0];static[out_i]=values[local_i,1]
  return dynamic,static

def gpu_uuid(physical:int)->str:
 result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
  capture_output=True,text=True,check=True)
 mapping={}
 for line in result.stdout.splitlines():
  parts=[x.strip() for x in line.split(',')]
  if len(parts)==2:mapping[int(parts[0])]=parts[1]
 if physical not in mapping:raise RuntimeError('watcher-assigned GPU index has no UUID in aggregate nvidia-smi query')
 return mapping[physical]

def setup_profile(backend:str)->tuple[torch.device,int|None,str|None,str|None]:
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
 if backend=='cpu':
  if os.environ.get('GPU_WATCHER_PHYSICAL_GPU') is not None:
   raise RuntimeError('CPU profile refuses inherited GPU watcher assignment')
  return torch.device('cpu'),None,None,None
 physical=int(os.environ.get('GPU_WATCHER_PHYSICAL_GPU','-1'))
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if physical<0 or visible!=str(physical):raise RuntimeError('CUDA child lacks watcher physical GPU/CUDA_VISIBLE_DEVICES binding')
 if not torch.cuda.is_available():raise RuntimeError('watcher-assigned CUDA device is unavailable')
 torch.cuda.set_device(0)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 return torch.device('cuda:0'),physical,visible,gpu_uuid(physical)

def model_for(row:dict[str,str],query_table:np.ndarray):
 source=ROOT/'source'
 if sha_file(source/'d3_model.py')!=row['model_code_sha256']:raise RuntimeError('model code identity mismatch')
 sys.path.insert(0,str(source))
 from d3_model import D3TimeSpacePredictor,parameter_count
 base=D3TimeSpacePredictor(horizon=30,seed=int(row['seed']),video_mode=row['model'])
 if row['query']=='Q0':model=base
 else:
  if sha_file(source/'fixed_future_query_model.py')!=row['query_wrapper_code_sha256']:raise RuntimeError('Q1 wrapper identity mismatch')
  from fixed_future_query_model import FixedFutureQueryOverride
  model=FixedFutureQueryOverride(base,torch.from_numpy(query_table))
 return model,parameter_count(model)

def main()->int:
 ap=argparse.ArgumentParser()
 ap.add_argument('--backend',choices=('cpu','cuda'),required=True)
 args=ap.parse_args()
 backend=args.backend;profile='CPU_COMMON' if backend=='cpu' else 'CUDA_COMMON'
 profile_lock=json.loads(PROFILE_LOCK.read_text(encoding='utf-8'))
 if sha_file(MATRIX)!=N2_MATRIX24_SHA256:raise RuntimeError('locked N2 matrix SHA mismatch')
 if (int(profile_lock.get('batch_size',-1))!=64 or profile_lock.get('amp') is not False
     or profile_lock.get('compile') is not False or profile_lock.get('model_dtype')!='float32'):
  raise RuntimeError('N2 common-profile lock fields differ from runner')
 with REQUEST.open(newline='',encoding='utf-8') as f:request_rows=list(csv.DictReader(f))
 with MATRIX.open(newline='',encoding='utf-8') as f:matrix=list(csv.DictReader(f))
 if len(request_rows)!=12 or len(matrix)!=24:raise RuntimeError('locked model/matrix population mismatch')
 request_by_id={r['model_id']:r for r in request_rows}
 selected=[r for r in matrix if r['profile']==profile]
 if len(selected)!=12:raise RuntimeError('backend profile must select exactly 12 locked views')
 device,physical,cvd,uuid=setup_profile(backend)
 input_sha=sha_file(INPUTS);cache_sha=sha_file(CACHE_MANIFEST);video_sha=sha_file(VIDEO_NORMALIZER);target_sha=sha_file(TARGET_NORMALIZER)
 if any(r['input_sha256']!=input_sha or r['feature_cache_manifest_sha256']!=cache_sha or
        r['video_normalizer_sha256']!=video_sha or r['target_normalizer_sha256']!=target_sha for r in request_rows):
  raise RuntimeError('input/cache/normalizer hashes differ from the export request')
 cache=json.loads(CACHE_MANIFEST.read_text(encoding='utf-8'))
 if cache.get('status')!='PASS' or int(cache.get('covered_rows',-1))!=20079:raise RuntimeError('accepted feature cache manifest status/rows mismatch')
 with np.load(INPUTS,allow_pickle=False) as archive:
  split=archive['split_code'].astype(np.int8,copy=True)
  if len(split)!=20079 or set(np.unique(split).tolist())!={0,1}:raise RuntimeError('train/dev split-code population mismatch')
  split_idx=np.flatnonzero(split==1).astype(np.int64)
  if len(split_idx)!=2322:raise RuntimeError('dev split row count mismatch')
  data=select_inputs(archive,split_idx)
 ids=data['anchor_id'].astype('U96')
 if len(ids)!=2322 or len(set(ids.tolist()))!=2322:raise RuntimeError('dev IDs are not unique/complete')
 id_hash=sha_array(ids.astype('U96'))
 with np.load(QUERY_ARCHIVE,allow_pickle=False) as z:query_table=np.asarray(z['candidate_h30'],dtype=np.float32).copy()
 qsha=sha_query(query_table)
 qlocked=json.loads((PROJECT/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z/protocol/QUERY_TABLES_AND_HASHES.json').read_text())
 if query_table.shape!=(30,128) or qsha!=qlocked['candidate_table_sha256_h30']:raise RuntimeError('accepted query archive/table mismatch')
 video_norm=json.loads(VIDEO_NORMALIZER.read_text());target_norm=json.loads(TARGET_NORMALIZER.read_text())
 mean=np.asarray(video_norm['mean'],dtype=np.float32);std=np.asarray(video_norm['safe_std'],dtype=np.float32)
 if mean.shape!=(768,) or std.shape!=(768,) or np.any(std<=0):raise RuntimeError('video normalizer shape/range invalid')
 sigma=torch.as_tensor(np.asarray(target_norm['target_sigma'],dtype=np.float32),device=device).view(1,1,28)
 if sigma.shape!=(1,1,28):raise RuntimeError('target normalizer shape invalid')
 grid=GridStore(cache)
 state=torch.from_numpy(data['history_standardized'].astype(np.float32,copy=False))
 mask=torch.from_numpy(data['history_mask'].astype(np.float32,copy=False))
 relative=torch.from_numpy(data['history_relative_delta_standardized'].astype(np.float32,copy=False))
 anchor=torch.from_numpy(data['anchor_command_native'].astype(np.float32,copy=False))
 if state.shape!=(2322,32,71) or mask.shape!=(2322,32) or relative.shape!=(2322,32) or anchor.shape!=(2322,28):
  raise RuntimeError('target-blind input array shapes differ from the locked contract')
 runtime={
  'torch':torch.__version__,'numpy':np.__version__,'backend':backend,'device':str(device),
  'physical_gpu_index':physical,'gpu_uuid':uuid,'cuda_visible_devices':cvd,
  'dtype':'float32','batch_size':64,'amp':False,'compile':False,'cpu_threads':torch.get_num_threads(),
  'cpu_interop_threads':torch.get_num_interop_threads(),'tf32_matmul':False if backend=='cuda' else None,
  'tf32_cudnn':False if backend=='cuda' else None,'cudnn_benchmark':False if backend=='cuda' else None,
  'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
  'sdpa_backend':'math_only' if backend=='cuda' else 'not_applicable',
  'n2_profile_lock_sha256':sha_file(PROFILE_LOCK),'n2_analysis_lock_sha256':sha_file(ANALYSIS_LOCK),
  'n2_matrix24_sha256':sha_file(MATRIX),
  'feature_cache_manifest_sha256':cache_sha,'video_normalizer_sha256':video_sha,
  'target_normalizer_sha256':target_sha,'input_sha256':input_sha,'query_archive_sha256':sha_file(QUERY_ARCHIVE),
  'query_table_sha256':qsha,'row_order_ids_sha256':id_hash
 }
 if backend=='cuda':torch.cuda.reset_peak_memory_stats(device)
 start_all=time.perf_counter()
 for matrix_row in selected:
  view_id=matrix_row['view_id']
  model_id=f"{matrix_row['model']}__{matrix_row['query']}__S{matrix_row['seed']}__H{matrix_row['horizon_steps']}"
  row=request_by_id.get(model_id)
  if row is None:raise RuntimeError('matrix view has no exact model export request: '+view_id)
  if matrix_row['checkpoint_sha256']!=row['checkpoint_sha256'] or matrix_row['checkpoint_path']!=row['checkpoint_path']:
   raise RuntimeError('N2 matrix checkpoint identity differs from export request: '+view_id)
  outdir=ROOT/'predictions'/'dev'/view_id
  pred_path=outdir/'primary.npz';side_path=outdir/'PREDICTION_COMPLETE.json'
  if outdir.exists():raise FileExistsError('prediction output already exists; preserve and inspect it: '+str(outdir))
  safe_accept=json.loads((ROOT/'qa/SAFE_ONLY_RELOAD.json').read_text(encoding='utf-8'))
  if safe_accept.get('status')!='PASS_SAFE_TENSORS_FRESH_PROCESS_RELOAD_AND_PROMOTION' or safe_accept.get('promoted_count')!=12:
   raise RuntimeError('12/12 safe-only reload acceptance is not present')
  safe_dir=ROOT/'safe_models'/model_id
  manifest_path=safe_dir/'EXPORT_MANIFEST.json';tensor_path=safe_dir/'model.safetensors'
  manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
  if manifest.get('model_file_sha256')!=sha_file(tensor_path) or manifest.get('source_checkpoint_sha256')!=row['checkpoint_sha256']:
   raise RuntimeError('safe-model manifest/file/source mismatch: '+model_id)
  if manifest.get('model_code_sha256')!=row['model_code_sha256'] or manifest.get('query_table_sha256')!=row['accepted_query_table_sha256']:
   raise RuntimeError('safe-model model/query identity mismatch: '+model_id)
  state_dict=load_file(str(tensor_path),device='cpu')
  model,nparams=model_for(row,query_table)
  if nparams!=890780:raise RuntimeError('unexpected parameter count: '+model_id)
  incompatible=model.load_state_dict(state_dict,strict=True)
  if incompatible.missing_keys or incompatible.unexpected_keys:raise RuntimeError('strict safe-model load key mismatch: '+model_id)
  model.to(device);model.eval()
  if backend=='cuda':torch.cuda.reset_peak_memory_stats(device)
  predictions=np.empty((2322,30,28),dtype=np.float32)
  t0=time.perf_counter()
  sdpa_context=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if backend=='cuda' else nullcontext()
  with sdpa_context,torch.inference_mode():
   for begin in range(0,2322,64):
    end=min(begin+64,2322)
    dynamic,static=grid.get(split_idx[begin:end])
    dynamic=((dynamic-mean[None,None,None,:])/std[None,None,None,:]).astype(np.float32)
    static=((static-mean[None,None,None,:])/std[None,None,None,:]).astype(np.float32)
    state_b=state[begin:end].to(device=device,dtype=torch.float32)
    mask_b=mask[begin:end].to(device=device,dtype=torch.float32)
    rel_b=relative[begin:end].to(device=device,dtype=torch.float32)
    anchor_b=anchor[begin:end].to(device=device,dtype=torch.float32)
    dyn_b=torch.from_numpy(dynamic).to(device=device,dtype=torch.float32)
    sta_b=torch.from_numpy(static).to(device=device,dtype=torch.float32)
    residual=model(state_b,mask_b,rel_b,dyn_b,sta_b,anchor_b)
    native=anchor_b[:,None,:]+residual*sigma
    predictions[begin:end]=native.detach().cpu().numpy().astype(np.float32,copy=False)
  if predictions.shape!=(2322,30,28) or not np.isfinite(predictions).all():raise RuntimeError('prediction shape/finite check failed: '+view_id)
  outdir.mkdir(parents=True,exist_ok=False)
  tmp=outdir/('.primary.npz.'+str(os.getpid())+'.part')
  with tmp.open('wb') as f:
   np.savez_compressed(f,anchor_id=ids,episode_index=data['episode_index'],
    original_task_index=data['original_task_index'],anchor_frame=data['anchor_frame'],
    anchor_timestamp=data['anchor_timestamp'],predictions_native=predictions,
    horizon_steps=np.asarray(30),model=np.asarray(row['model']),query=np.asarray(row['query']),
    seed=np.asarray(int(row['seed'])),profile=np.asarray(profile),split=np.asarray('dev'))
   f.flush();os.fsync(f.fileno())
  os.replace(tmp,pred_path)
  peak=int(torch.cuda.max_memory_allocated(device)/1024/1024) if backend=='cuda' else 0
  view_runtime={**runtime,'gpu_peak_allocated_mib':peak,'view_elapsed_seconds':round(time.perf_counter()-t0,6)}
  side={
   'schema_version':1,'status':'PASS','view_id':view_id,'model_id':model_id,'run_id':view_id,
   'model':row['model'],'query':row['query'],'query_variant':row['query_variant'],'seed':int(row['seed']),
   'horizon_steps':30,'profile':profile,'profile_role':matrix_row['profile_role'],'split':'dev',
   'rows':2322,'expected_rows':2322,'source_groups_expected':603,'prediction_shape':[2322,30,28],
   'ids_sha256':id_hash,'prediction_path':str(pred_path),'prediction_sha256':sha_file(pred_path),
   'checkpoint_path':row['checkpoint_path'],'checkpoint_sha256':row['checkpoint_sha256'],
   'checkpoint_epoch40_sha256':row['checkpoint_epoch40_sha256'],'safe_model_path':str(tensor_path),
   'safe_model_sha256':sha_file(tensor_path),'tensor_catalogue_sha256':manifest['tensor_catalogue_sha256'],
   'model_code_sha256':row['model_code_sha256'],'query_wrapper_code_sha256':row['query_wrapper_code_sha256'],
   'training_protocol_lock_sha256':row['training_protocol_lock_sha256'],'query_table_sha256':row['accepted_query_table_sha256'],
   'n2_profile_lock_sha256':sha_file(PROFILE_LOCK),'n2_analysis_lock_sha256':sha_file(ANALYSIS_LOCK),
   'n2_matrix24_sha256':sha_file(MATRIX),'safe_only_reload_acceptance_sha256':sha_file(ROOT/'qa/SAFE_ONLY_RELOAD.json'),
   'query_archive_sha256':sha_file(QUERY_ARCHIVE),'input_sha256':input_sha,'feature_cache_manifest_sha256':cache_sha,
   'video_normalizer_sha256':video_sha,'target_normalizer_sha256':target_sha,
   'native_inverse_formula':'native = anchor_command_native + target_sigma * residual; all operands float32 on the selected backend',
   'target_blind_input_keys':list(TARGET_BLIND_KEYS),'target_arrays_loaded':False,'dev_targets_opened_by_inference':False,
   'optimizer_created':False,'optimizer_steps':0,'backward_called':False,'encoder_updated':False,
   'fresh_process_safe_tensor_reload':True,'strict_load':True,'parameter_count':nparams,
   'final_test_access':0,'heldout_access':0,'runtime':view_runtime,'created_utc':utc_now()}
  write_json(side_path,side)
  if backend=='cuda':torch.cuda.empty_cache()
  print(json.dumps({'status':'view_complete','view_id':view_id,'rows':2322,
   'seconds':view_runtime['view_elapsed_seconds'],'gpu_peak_allocated_mib':peak},sort_keys=True),flush=True)
  append_jsonl(ROOT/'coordination/JOB_LEDGER.jsonl',{'utc':utc_now(),'event':'view_complete','view_id':view_id,
   'backend':backend,'prediction_sha256':side['prediction_sha256'],'status':'PASS','physical_gpu':physical,'gpu_uuid':uuid})
  del model,state_dict,predictions
 total=time.perf_counter()-start_all
 result={'schema_version':1,'status':'PASS','backend':backend,'profile':profile,'views':len(selected),
  'started_utc':utc_now(),'elapsed_seconds':round(total,3),'runtime':runtime,
  'prediction_root':str(ROOT/'predictions/dev'),'target_arrays_loaded':False,'final_test_access':0}
 write_json(ROOT/'coordination'/f'{backend.upper()}_INFERENCE_STATUS.json',result)
 print(json.dumps({'status':'backend_complete','backend':backend,'views':len(selected),
  'elapsed_seconds':round(total,3),'physical_gpu':physical,'gpu_uuid':uuid},sort_keys=True),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
