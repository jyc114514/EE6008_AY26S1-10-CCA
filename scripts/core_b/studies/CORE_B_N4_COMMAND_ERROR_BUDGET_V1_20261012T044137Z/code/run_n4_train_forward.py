#!/usr/bin/env python3
"""Target-blind CUDA_COMMON inference for only missing N4 primary-train views."""
from __future__ import annotations
import os
import argparse, bisect, csv, hashlib, json, os, subprocess, sys, time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT=Path(__file__).resolve().parents[1]
N2=PROJECT/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
N1=PROJECT/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
BENCH=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z'
INPUTS=V3/'data/V3_MAIN_INPUTS.npz'
CACHE_MANIFEST=BENCH/'d3/cache/D3_CACHE_MANIFEST.json'
VIDEO_NORMALIZER=BENCH/'d3/cache/D3_VIDEO_NORMALIZER.json'
TARGET_NORMALIZER=V3/'data/V3_NORMALIZERS.json'
QUERY_ARCHIVE=N1/'protocol/HADAMARD_QUERY_TABLES.npz'
N2_MATRIX=N2/'protocol/N2_MATRIX24.csv'
N2_VIEW_ACCEPTANCE=N2/'results/N2_VIEW_ACCEPTANCE.csv'
N2_PROFILE_ACCEPTANCE=N2/'qa/N2_PROFILE_ACCEPTANCE.json'
SAFE_ACCEPTANCE=N2/'qa/SAFE_ONLY_RELOAD.json'
REQUEST=N2/'MODEL_EXPORT_REQUEST.csv'
TARGET_BLIND_KEYS=('anchor_id','episode_index','original_task_index','anchor_frame','anchor_timestamp',
 'history_standardized','history_mask','history_relative_delta_standardized','anchor_command_native')
EXPECTED_N2_GPU_UUID='GPU-00465dc6-e96e-67f1-285a-84ed61565ac2'
CANARY_TOLERANCE_ABS=0.0

def utc()->str:return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
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
def read_json(p:Path):return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p:Path):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_json(p:Path,obj:dict[str,Any])->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name('.'+p.name+'.'+str(os.getpid())+'.part')
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 os.replace(tmp,p)
def append_jsonl(p:Path,obj:dict[str,Any])->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:
  f.write(json.dumps(obj,sort_keys=True,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
class GridStore:
 def __init__(self,manifest):
  self.shards=sorted(manifest['shards'],key=lambda x:int(x['start_row']))
  self.starts=[int(x['start_row']) for x in self.shards]
 def get(self,indices):
  dynamic=np.empty((len(indices),16,4,768),dtype=np.float32);static=np.empty_like(dynamic)
  grouped={}
  for oi,row_i in enumerate(indices.astype(np.int64)):
   si=bisect.bisect_right(self.starts,int(row_i))-1
   sh=self.shards[si]
   if not int(sh['start_row'])<=int(row_i)<int(sh['end_row']):raise IndexError('feature cache row gap')
   grouped.setdefault(si,[]).append((oi,int(row_i)-int(sh['start_row'])))
  for si,pairs in grouped.items():
   arr=np.load(self.shards[si]['array'],mmap_mode='r',allow_pickle=False)
   for oi,li in pairs:dynamic[oi]=arr[li,0];static[oi]=arr[li,1]
  return dynamic,static
def gpu_uuid(physical:int)->str:
 r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 mapping={}
 for line in r.stdout.splitlines():
  parts=[x.strip() for x in line.split(',')]
  if len(parts)==2:mapping[int(parts[0])]=parts[1]
 if physical not in mapping:raise RuntimeError('watcher assigned physical GPU missing from aggregate UUID query')
 return mapping[physical]
def setup_cuda(torch):
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
 physical=int(os.environ.get('GPU_WATCHER_PHYSICAL_GPU','-1'))
 cvd=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if physical<0 or cvd!=str(physical):raise RuntimeError('missing shared-watcher physical/CUDA_VISIBLE_DEVICES binding')
 if not torch.cuda.is_available():raise RuntimeError('watcher-assigned CUDA device unavailable')
 torch.cuda.set_device(0)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 return torch.device('cuda:0'),physical,cvd,gpu_uuid(physical)
def model_for(req,query_table,torch):
 sys.path.insert(0,str(N2/'source'))
 from d3_model import D3TimeSpacePredictor,parameter_count
 base=D3TimeSpacePredictor(horizon=30,seed=int(req['seed']),video_mode=req['model'])
 if req['query']=='Q0':model=base
 else:
  from fixed_future_query_model import FixedFutureQueryOverride
  model=FixedFutureQueryOverride(base,torch.from_numpy(query_table))
 return model,parameter_count(model)
def get_inputs(split_code):
 with np.load(INPUTS,allow_pickle=False) as z:
  split=np.asarray(z['split_code'],dtype=np.int8)
  indices=np.flatnonzero(split==split_code).astype(np.int64)
  if len(indices)==0:raise RuntimeError('requested canonical split has no rows')
  data={k:z[k][indices].copy() for k in TARGET_BLIND_KEYS}
 return indices,data
def load_safe_model(req,query_table,torch,load_file):
 model_id=req['model_id'];safe_dir=N2/'safe_models'/model_id
 manifest=read_json(safe_dir/'EXPORT_MANIFEST.json');tensor=safe_dir/'model.safetensors'
 if manifest.get('model_file_sha256')!=sha_file(tensor):raise RuntimeError('safe model hash mismatch '+model_id)
 if manifest.get('source_checkpoint_sha256')!=req['checkpoint_sha256']:raise RuntimeError('source checkpoint identity mismatch '+model_id)
 if manifest.get('model_code_sha256')!=req['model_code_sha256']:raise RuntimeError('model code identity mismatch '+model_id)
 if manifest.get('query_table_sha256')!=req['accepted_query_table_sha256']:raise RuntimeError('query hash identity mismatch '+model_id)
 sd=load_file(str(tensor),device='cpu')
 model,nparams=model_for(req,query_table,torch)
 if nparams!=890780:raise RuntimeError('parameter count mismatch '+model_id)
 incompatible=model.load_state_dict(sd,strict=True)
 if incompatible.missing_keys or incompatible.unexpected_keys:raise RuntimeError('strict model load mismatch '+model_id)
 model.to(torch.device('cuda:0'));model.eval()
 return model,sd,manifest,tensor,nparams
def forward_batch(model,indices,data,grid,mean,std,sigma,device,torch):
 dyn,sta=grid.get(indices)
 dyn=((dyn-mean[None,None,None,:])/std[None,None,None,:]).astype(np.float32)
 sta=((sta-mean[None,None,None,:])/std[None,None,None,:]).astype(np.float32)
 state=torch.from_numpy(data['history_standardized'].astype(np.float32,copy=False)).to(device=device,dtype=torch.float32)
 mask=torch.from_numpy(data['history_mask'].astype(np.float32,copy=False)).to(device=device,dtype=torch.float32)
 rel=torch.from_numpy(data['history_relative_delta_standardized'].astype(np.float32,copy=False)).to(device=device,dtype=torch.float32)
 anchor=torch.from_numpy(data['anchor_command_native'].astype(np.float32,copy=False)).to(device=device,dtype=torch.float32)
 db=torch.from_numpy(dyn).to(device=device,dtype=torch.float32)
 sb=torch.from_numpy(sta).to(device=device,dtype=torch.float32)
 residual=model(state,mask,rel,db,sb,anchor)
 native=anchor[:,None,:]+residual*sigma
 return native.detach().cpu().numpy().astype(np.float32,copy=False)
def validate_source_and_plan():
 if not (ROOT/'protocol/PROTOCOL_LOCK.json').is_file():raise RuntimeError('N4 protocol is not locked')
 lock=read_json(ROOT/'protocol/PROTOCOL_LOCK.json')
 if lock.get('lock_status')!='LOCKED_BEFORE_N4_OUTCOME_READ':raise RuntimeError('N4 protocol lock status invalid')
 source_accept=read_json(ROOT/'inventory/SOURCE_ACCEPTANCE.json')
 if source_accept.get('n2_checksum_manifest_sha256')!=sha_file(N2/'CHECKSUMS.sha256'):
  raise RuntimeError('sealed N2 checksum manifest changed since N4 lock')
 for rec in source_accept.get('n2_selected_file_inventory',[]):
  if sha_file(Path(rec['path']))!=rec['sha256']:raise RuntimeError('selected N2 checksum changed: '+rec['path'])
 for rec in source_accept.get('source_hashes',{}).values():
  if sha_file(Path(rec['path']))!=rec['sha256']:raise RuntimeError('locked source hash changed: '+rec['path'])
 amendment=read_json(ROOT/'protocol/PROFILE_LOCK_AMENDMENT_001.json')
 if amendment.get('canary_tolerance_abs')!=0.0 or amendment.get('tolerance_widened') is not False:
  raise RuntimeError('N4 conditional canary tolerance is not the locked exact tolerance')
 safe=read_json(SAFE_ACCEPTANCE)
 if safe.get('status')!='PASS_SAFE_TENSORS_FRESH_PROCESS_RELOAD_AND_PROMOTION' or safe.get('promoted_count')!=12:
  raise RuntimeError('12 accepted safe model exports required')
 matrix=read_csv(ROOT/'protocol/MATRIX24.csv')
 train=[r for r in matrix if r['split']=='primary_train']
 if len(train)!=12:raise RuntimeError('N4 primary-train matrix must contain 12 fixed models')
 if sha_file(N2_MATRIX)!='70e81b236727c617aeadc21e67dbb0412d6a8c56a2ab3ec03151e077897a6ffd':
  raise RuntimeError('accepted N2 matrix hash changed')
 reqs=read_csv(REQUEST)
 if len(reqs)!=12:raise RuntimeError('accepted safe model request count mismatch')
 expected={'input_sha256':sha_file(INPUTS),'feature_cache_manifest_sha256':sha_file(CACHE_MANIFEST),
  'video_normalizer_sha256':sha_file(VIDEO_NORMALIZER),'target_normalizer_sha256':sha_file(TARGET_NORMALIZER)}
 for req in reqs:
  for key,value in expected.items():
   if req.get(key)!=value:raise RuntimeError('locked request/source mismatch: '+key)
  source=N2/'source'
  if sha_file(source/'d3_model.py')!=req['model_code_sha256']:
   raise RuntimeError('accepted model source hash mismatch: '+req['model_id'])
  if req['query']=='Q1':
   if sha_file(source/'fixed_future_query_model.py')!=req['query_wrapper_code_sha256']:
    raise RuntimeError('accepted Q1 wrapper hash mismatch: '+req['model_id'])
   if req['query_archive_sha256']!=sha_file(QUERY_ARCHIVE):
    raise RuntimeError('accepted Q1 archive hash mismatch: '+req['model_id'])
  if req['query']=='Q0' and req.get('accepted_query_table_sha256') not in ('',None):
   raise RuntimeError('Q0 unexpectedly binds a Q1 table hash')
 for r in train:
  if Path(r['prediction_path']).exists() or Path(r['prediction_receipt_path']).exists():
   raise FileExistsError('N4 train output already exists; inspect its receipt instead of overwriting: '+r['view_id'])
 for path in (INPUTS,CACHE_MANIFEST,VIDEO_NORMALIZER,TARGET_NORMALIZER,QUERY_ARCHIVE):
  if not path.is_file():raise FileNotFoundError(str(path))
 return train
def dry_run():
 train=validate_source_and_plan()
 reqs={r['model_id']:r for r in read_csv(REQUEST)}
 ids=[]
 for r in train:
  mid=f"{r['model']}__{r['query']}__S{r['seed']}__H30"
  q=reqs[mid]
  tensor=N2/'safe_models'/mid/'model.safetensors'
  if sha_file(tensor)!=read_json(tensor.parent/'EXPORT_MANIFEST.json').get('model_file_sha256'):
   raise RuntimeError('dry-run safe-model byte/hash failure '+mid)
  ids.append({'model_id':mid,'model':r['model'],'query':r['query'],'seed':r['seed'],
   'safe_model_sha256':sha_file(tensor),'train_output':r['prediction_path'],'expected_rows':17757})
 print(json.dumps({'status':'PASS_DRY_RUN','protocol_lock_sha256':sha_file(ROOT/'protocol/PROTOCOL_LOCK.json'),
  'models':len(ids),'plan':ids,'model_loaded':False,'optimizer_created':False,
  'targets_opened':False,'gpu_accessed':False,'old_root_writes':False},ensure_ascii=False,indent=2))
 return 0
def main():
 ap=argparse.ArgumentParser(description='N4 locked primary-train target-blind inference using accepted safetensors only')
 ap.add_argument('--dry-run',action='store_true',help='validate source and output plan without loading model or touching CUDA')
 args=ap.parse_args()
 if args.dry_run:return dry_run()
 train=validate_source_and_plan()
 # Delay Torch/CUDA import and every model load until after the N4 lock is verified.
 import torch
 from safetensors.torch import load_file
 device,physical,cvd,uuid=setup_cuda(torch)
 prof=read_json(N2_PROFILE_ACCEPTANCE)['CUDA_COMMON']
 if prof.get('status')!='PASS' or prof.get('uuid')!=EXPECTED_N2_GPU_UUID:
  raise RuntimeError('N2 accepted CUDA_COMMON profile identity missing')
 if str(torch.__version__)!=str(read_json(N2/'protocol/N2_PROFILE_LOCK.json')['torch']):
  raise RuntimeError('installed torch version differs from N2 locked profile')
 runtime={'torch':str(torch.__version__),'torch_cuda':torch.version.cuda,'device':str(device),
  'physical_gpu_index':physical,'gpu_uuid':uuid,'cuda_visible_devices':cvd,'dtype':'float32',
  'batch_size':64,'amp':False,'compile':False,'tf32_matmul':False,'tf32_cudnn':False,
  'cudnn_benchmark':False,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
  'sdpa_backend':'math_only','cpu_threads':torch.get_num_threads(),
  'cpu_interop_threads':torch.get_num_interop_threads(),'n2_input_sha256':sha_file(INPUTS),
  'n2_feature_cache_manifest_sha256':sha_file(CACHE_MANIFEST),
  'n2_video_normalizer_sha256':sha_file(VIDEO_NORMALIZER),
  'n2_target_normalizer_sha256':sha_file(TARGET_NORMALIZER),
  'n2_query_archive_sha256':sha_file(QUERY_ARCHIVE),
  'n4_profile_lock_sha256':sha_file(ROOT/'protocol/PROFILE_LOCK.json'),
  'n4_protocol_lock_sha256':sha_file(ROOT/'protocol/PROTOCOL_LOCK.json')}
 if uuid!=EXPECTED_N2_GPU_UUID:
  # No tolerance widening: the accepted source-profile replay recorded exact equality.
  canary=run_canary(train,device,physical,cvd,uuid,runtime,torch,load_file)
  if canary['status']!='PASS_EXACT_32_UNIQUE_IDS_ALL_12_MODELS':
   write_json(ROOT/'qa/GPU_CANARY.json',canary)
   raise RuntimeError('non-original physical GPU canary failed at locked abs tolerance 0.0')
  write_json(ROOT/'qa/GPU_CANARY.json',canary)
 else:
  write_json(ROOT/'qa/GPU_CANARY.json',{'status':'SKIP_IDENTICAL_N2_PHYSICAL_GPU_AND_PROFILE',
   'physical_gpu':physical,'gpu_uuid':uuid,'profile_flags_match':True,'canary_required':False,
   'target_arrays_loaded':False,'formal_dev_views_added':0,'created_utc':utc()})
 write_json(ROOT/'coordination/GPU_RUNTIME_PREFLIGHT.json',runtime)
 n2req={r['model_id']:r for r in read_csv(REQUEST)}
 n2views={(r['model'],r['query'],r['seed'],r['profile']):r for r in read_csv(N2_VIEW_ACCEPTANCE)}
 cache=read_json(CACHE_MANIFEST)
 if cache.get('status')!='PASS' or int(cache.get('covered_rows',-1))!=20079:raise RuntimeError('frozen feature cache acceptance failed')
 with np.load(QUERY_ARCHIVE,allow_pickle=False) as z:
  query_table=np.asarray(z['candidate_h30'],dtype=np.float32).copy()
 query_hash=sha_query(query_table)
 query_lock=read_json(N1/'protocol/QUERY_TABLES_AND_HASHES.json')
 if query_table.shape!=(30,128) or query_hash!=query_lock['candidate_table_sha256_h30']:
  raise RuntimeError('accepted Q1 query table mismatch')
 video=read_json(VIDEO_NORMALIZER);target_norm=read_json(TARGET_NORMALIZER)
 mean=np.asarray(video['mean'],dtype=np.float32);std=np.asarray(video['safe_std'],dtype=np.float32)
 sigma_np=np.asarray(target_norm['target_sigma'],dtype=np.float32)
 if mean.shape!=(768,) or std.shape!=(768,) or np.any(std<=0) or sigma_np.shape!=(28,):
  raise RuntimeError('accepted normalizer shape/range mismatch')
 sigma=torch.as_tensor(sigma_np,device=device,dtype=torch.float32).view(1,1,28)
 train_idx,train_data=get_inputs(0)
 if len(train_idx)!=17757:raise RuntimeError('canonical primary train row count mismatch')
 ids=train_data['anchor_id'].astype('U96')
 if len(np.unique(ids))!=17757:raise RuntimeError('canonical train IDs are not unique')
 id_hash=sha_array(ids)
 group_map={int(r['episode_index']):r for r in read_csv(N2/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv') if r['split']=='train'}
 episodes=train_data['episode_index'].astype(np.int64)
 unique_eps=np.unique(episodes)
 if len(unique_eps)!=2712 or any(int(e) not in group_map for e in unique_eps):
  raise RuntimeError('train episode/source-group mapping mismatch')
 unique_groups={group_map[int(e)]['source_group_id'] for e in unique_eps}
 if len(unique_groups)!=2712:raise RuntimeError('train source group count mismatch')
 grid=GridStore(cache)
 state=train_data['history_standardized'];mask=train_data['history_mask']
 relative=train_data['history_relative_delta_standardized'];anchor=train_data['anchor_command_native']
 if state.shape!=(17757,32,71) or mask.shape!=(17757,32) or relative.shape!=(17757,32) or anchor.shape!=(17757,28):
  raise RuntimeError('train target-blind input shape mismatch')
 owned=[];all_start=time.perf_counter();max_peak=0
 for matrix_row in train:
  model_id=f"{matrix_row['model']}__{matrix_row['query']}__S{matrix_row['seed']}__H30"
  req=n2req[model_id]
  outdir=ROOT/'predictions/train'/model_id
  pred_path=outdir/'primary.npz';side_path=outdir/'PREDICTION_COMPLETE.json'
  if outdir.exists():
   if side_path.is_file() and pred_path.is_file():
    side=read_json(side_path)
    if side.get('status')=='PASS' and side.get('prediction_sha256')==sha_file(pred_path) and side.get('model_id')==model_id and side.get('split')=='train':
     owned.append({'view_id':matrix_row['view_id'],'status':'REUSED_OWN_ACCEPTED_OUTPUT','prediction_sha256':side['prediction_sha256']})
     continue
   raise FileExistsError('Preserving incomplete prior N4 output; inspect before resuming: '+str(outdir))
  model,sd,export,tensor,nparams=load_safe_model(req,query_table,torch,load_file)
  torch.cuda.reset_peak_memory_stats(device);t0=time.perf_counter()
  predictions=np.empty((17757,30,28),dtype=np.float32)
  context=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
  with context,torch.inference_mode():
   for begin in range(0,17757,64):
    end=min(begin+64,17757)
    native=forward_batch(model,train_idx[begin:end],{k:v[begin:end] for k,v in train_data.items()},
     grid,mean,std,sigma,device,torch)
    predictions[begin:end]=native
  if predictions.shape!=(17757,30,28) or not np.isfinite(predictions).all():
   raise RuntimeError('train prediction shape/finite failure '+model_id)
  peak=int(torch.cuda.max_memory_allocated(device)/1024/1024);max_peak=max(max_peak,peak)
  elapsed=round(time.perf_counter()-t0,6)
  outdir.mkdir(parents=True,exist_ok=False)
  tmp=outdir/('.primary.npz.'+str(os.getpid())+'.part')
  with tmp.open('wb') as f:
   np.savez_compressed(f,anchor_id=ids,episode_index=episodes,
    original_task_index=train_data['original_task_index'],anchor_frame=train_data['anchor_frame'],
    anchor_timestamp=train_data['anchor_timestamp'],predictions_native=predictions,
    horizon_steps=np.asarray(30),model=np.asarray(req['model']),query=np.asarray(req['query']),
    seed=np.asarray(int(req['seed'])),profile=np.asarray('CUDA_COMMON'),split=np.asarray('train'))
   f.flush();os.fsync(f.fileno())
  os.replace(tmp,pred_path)
  side={'schema_version':1,'status':'PASS','view_id':matrix_row['view_id'],'model_id':model_id,
   'model':req['model'],'query':req['query'],'query_variant':req['query_variant'],'seed':int(req['seed']),
   'horizon_steps':30,'split':'train','profile':'CUDA_COMMON','rows':17757,'expected_rows':17757,
   'episodes':2712,'source_groups':2712,'prediction_shape':[17757,30,28],'ids_sha256':id_hash,
   'prediction_path':str(pred_path),'prediction_sha256':sha_file(pred_path),
   'checkpoint_path':req['checkpoint_path'],'checkpoint_sha256':req['checkpoint_sha256'],
   'checkpoint_epoch40_sha256':req['checkpoint_epoch40_sha256'],'safe_model_path':str(tensor),
   'safe_model_sha256':sha_file(tensor),'tensor_catalogue_sha256':export['tensor_catalogue_sha256'],
   'model_code_sha256':req['model_code_sha256'],'query_wrapper_code_sha256':req['query_wrapper_code_sha256'],
   'training_protocol_lock_sha256':req['training_protocol_lock_sha256'],
   'query_table_sha256':req['accepted_query_table_sha256'],'query_archive_sha256':sha_file(QUERY_ARCHIVE),
   'input_sha256':sha_file(INPUTS),'feature_cache_manifest_sha256':sha_file(CACHE_MANIFEST),
   'video_normalizer_sha256':sha_file(VIDEO_NORMALIZER),'target_normalizer_sha256':sha_file(TARGET_NORMALIZER),
   'native_inverse_formula':'native = anchor_command_native + target_sigma * residual; float32 on selected backend',
   'target_blind_input_keys':list(TARGET_BLIND_KEYS),'target_arrays_loaded':False,'train_targets_opened_by_inference':False,
   'optimizer_created':False,'optimizer_steps':0,'backward_called':False,'encoder_updated':False,
   'fresh_process_safe_tensor_reload':True,'strict_load':True,'parameter_count':nparams,
   'runtime':{**runtime,'gpu_peak_allocated_mib':peak,'view_elapsed_seconds':elapsed},
   'final_test_access':0,'heldout_access':0,'created_utc':utc()}
  write_json(side_path,side)
  owned.append({'view_id':matrix_row['view_id'],'status':'PASS','prediction_sha256':side['prediction_sha256'],
   'gpu_peak_allocated_mib':peak,'elapsed_seconds':elapsed})
  append_jsonl(ROOT/'coordination/INFERENCE_EVENTS.jsonl',{'utc':utc(),'event':'train_view_complete',
   'view_id':matrix_row['view_id'],'prediction_sha256':side['prediction_sha256'],'physical_gpu':physical,'gpu_uuid':uuid})
  write_json(ROOT/'coordination/INFERENCE_PROGRESS.json',{'status':'RUNNING','completed_train_views':len(owned),
   'expected_train_views':12,'views':owned,'physical_gpu':physical,'gpu_uuid':uuid,'target_arrays_loaded':False})
  print(json.dumps({'status':'train_view_complete','view_id':matrix_row['view_id'],'rows':17757,
   'gpu_peak_allocated_mib':peak,'elapsed_seconds':elapsed},sort_keys=True),flush=True)
  del model,sd,predictions
  torch.cuda.empty_cache()
 status={'schema_version':1,'status':'PASS','split':'train','profile':'CUDA_COMMON',
  'train_views':len(owned),'expected_train_views':12,'started_utc':utc(),
  'elapsed_seconds':round(time.perf_counter()-all_start,3),'max_gpu_peak_allocated_mib':max_peak,
  'physical_gpu':physical,'gpu_uuid':uuid,'cuda_visible_devices':cvd,'runtime':runtime,
  'target_arrays_loaded':False,'optimizer_steps':0,'encoder_extraction':0,'final_test_access':0,'views':owned}
 write_json(ROOT/'coordination/N4_TRAIN_INFERENCE_STATUS.json',status)
 print(json.dumps({'status':'PASS_TRAIN_FORWARD_12','views':len(owned),'physical_gpu':physical,
  'gpu_uuid':uuid,'max_peak_mib':max_peak,'elapsed_seconds':status['elapsed_seconds']},sort_keys=True),flush=True)
 return 0
def run_canary(train,device,physical,cvd,uuid,runtime,torch,load_file):
 # Fixed first 32 canonical dev IDs, duplicated once only to retain the locked batch-64 profile.
 dev_idx,dev_data=get_inputs(1)
 if len(dev_idx)!=2322:raise RuntimeError('dev canary input count mismatch')
 refs={(r['model'],r['query'],r['seed'],r['profile']):r for r in read_csv(N2_VIEW_ACCEPTANCE)}
 q=np.asarray(np.load(QUERY_ARCHIVE,allow_pickle=False)['candidate_h30'],dtype=np.float32)
 video=read_json(VIDEO_NORMALIZER);target=read_json(TARGET_NORMALIZER)
 mean=np.asarray(video['mean'],dtype=np.float32);std=np.asarray(video['safe_std'],dtype=np.float32)
 sigma=torch.as_tensor(np.asarray(target['target_sigma'],dtype=np.float32),device=device).view(1,1,28)
 cache=read_json(CACHE_MANIFEST);grid=GridStore(cache)
 source_ids=dev_data['anchor_id'].astype('U96')[:32]
 selected_indices=np.concatenate([dev_idx[:32],dev_idx[:32]])
 batch_data={k:np.concatenate([v[:32],v[:32]],axis=0) for k,v in dev_data.items()}
 result={'status':'RUNNING','physical_gpu':physical,'gpu_uuid':uuid,'cuda_visible_devices':cvd,
  'unique_ids':32,'batch_size':64,'duplicate_padding':'each fixed dev ID repeated once to retain N2 batch size',
  'tolerance_abs':CANARY_TOLERANCE_ABS,'results':[],'target_arrays_loaded':False,'formal_dev_views_added':0}
 reqs={r['model_id']:r for r in read_csv(REQUEST)}
 for mr in train:
  mid=f"{mr['model']}__{mr['query']}__S{mr['seed']}__H30";req=reqs[mid]
  refrow=refs[(mr['model'],mr['query'],mr['seed'],'CUDA_COMMON')]
  refpath=Path(refrow['prediction_path'])
  if sha_file(refpath)!=refrow['prediction_sha256']:raise RuntimeError('accepted N2 canary reference hash mismatch '+mid)
  with np.load(refpath,allow_pickle=False) as z:
   ref_ids=z['anchor_id'].astype('U96')[:32]
   ref_pred=np.asarray(z['predictions_native'][:32],dtype=np.float32)
  if not np.array_equal(ref_ids,source_ids):raise RuntimeError('N2 canary ID order mismatch '+mid)
  model,sd,export,tensor,nparams=load_safe_model(req,q,torch,load_file)
  torch.cuda.reset_peak_memory_stats(device)
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),torch.inference_mode():
   got=forward_batch(model,selected_indices,batch_data,grid,mean,std,sigma,device,torch)[:32]
  diff=np.abs(got-ref_pred);mx=float(diff.max(initial=0.0))
  row={'model_id':mid,'ids_sha256':sha_array(source_ids),'max_abs_difference':mx,
   'mean_abs_difference':float(diff.mean()),'tolerance_abs':CANARY_TOLERANCE_ABS,
   'status':'PASS' if mx<=CANARY_TOLERANCE_ABS else 'FAIL'}
  result['results'].append(row)
  append_jsonl(ROOT/'qa/GPU_CANARY_PROGRESS.jsonl',row)
  del model,sd
  torch.cuda.empty_cache()
  if row['status']!='PASS':
   result['status']='FAIL_CANARY_TOLERANCE';result['failed_model_id']=mid
   result['completed_models']=len(result['results'])
   return result
 result['status']='PASS_EXACT_32_UNIQUE_IDS_ALL_12_MODELS'
 result['completed_models']=12;result['created_utc']=utc()
 return result
if __name__=='__main__':raise SystemExit(main())
