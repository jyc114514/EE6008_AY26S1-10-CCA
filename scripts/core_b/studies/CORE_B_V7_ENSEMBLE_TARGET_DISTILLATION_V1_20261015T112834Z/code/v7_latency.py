#!/usr/bin/env python3
"""Measure cached-feature Q0 head latency for one fixed seed on the watcher GPU."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'code';LOCKED=CODE/'locked_source'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(LOCKED))
from prepare_v2_data import Q0ShardedGrid
from v2_model import V2TaskConditionedPredictor,state_hash
TASKS=(119,94,90,141,230)
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p:Path,obj:dict[str,Any])->None:
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def load_prefix(path:Path)->dict[str,np.ndarray]:
 fields=('anchor_id','episode_id','task_id','cache_index','history_native','history_mask','relative_delta','anchor_native')
 with np.load(path,allow_pickle=False) as z:
  if 'target_native_h30' in z.files:raise RuntimeError('latency job received GT-bearing archive')
  return {k:z[k].copy() for k in fields}
def make_batch(data:dict[str,np.ndarray],norm:dict[str,Any],device:torch.device,mode:str,grid:Q0ShardedGrid|None)->tuple[dict[str,torch.Tensor],torch.Tensor]:
 ids=np.arange(32,dtype=np.int64)
 state=(data['history_native'][ids]-np.asarray(norm['state_mean'],dtype=np.float32)[None,None])/np.asarray(norm['state_safe_std'],dtype=np.float32)[None,None]
 rel=(data['relative_delta'][ids]-np.float32(norm['relative_delta_mean']))/np.float32(norm['relative_delta_safe_std'])
 prefix={'state':torch.as_tensor(state,device=device,dtype=torch.float32),'history_mask':torch.as_tensor(data['history_mask'][ids],device=device,dtype=torch.float32),
         'relative_delta':torch.as_tensor(rel,device=device,dtype=torch.float32),'anchor_command':torch.as_tensor(data['anchor_native'][ids],device=device,dtype=torch.float32)}
 task=torch.as_tensor([TASKS.index(int(data['task_id'][i])) for i in ids],device=device,dtype=torch.long)
 if mode!='STATE-CTRL':
  raw=np.stack([grid.row(int(data['cache_index'][i]),str(data['anchor_id'][i]),str(data['episode_id'][i])) for i in ids]).astype(np.float32,copy=False)
  vm=np.asarray(norm['video_mean'],dtype=np.float32)[None,None,None,:];vs=np.asarray(norm['video_safe_std'],dtype=np.float32)[None,None,None,:]
  prefix['dynamic']=torch.as_tensor((raw[:,0]-vm)/vs,device=device,dtype=torch.float32);prefix['static']=torch.as_tensor((raw[:,1]-vm)/vs,device=device,dtype=torch.float32)
 return prefix,task
def run(job_path:Path)->int:
 job=json.loads(job_path.read_text());out=Path(job['output_root']);out.mkdir(parents=True,exist_ok=True);status=out/'RUN_STATUS.json'
 try:
  if not torch.cuda.is_available():raise RuntimeError('latency measurement requires watcher-authorized CUDA')
  device=torch.device('cuda:0');seed=int(job['seed']);norm_path=Path(job['normalizer_path']);norm=json.loads(norm_path.read_text())
  data=load_prefix(Path(job['train_prefix_path']));resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text());cache_path=Path(resolved['feature_cache'])
  if sha(norm_path)!=resolved['formal_normalizer_sha256']:raise RuntimeError('latency normalizer hash mismatch')
  if sha(cache_path)!=resolved['feature_cache_manifest_sha256']:raise RuntimeError('latency Q0 cache manifest hash mismatch')
  grid=Q0ShardedGrid(json.loads(cache_path.read_text()))
  torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  rows=[]
  for item in job['models']:
   mode=item['video_mode'];model_path=Path(item['model_path']);manifest_path=Path(item['model_manifest_path']);manifest=json.loads(manifest_path.read_text())
   if sha(model_path)!=manifest.get('model_file_sha256'):raise RuntimeError(f'model SHA mismatch: {item["name"]}')
   if manifest.get('normalizer_sha256')!=resolved['formal_normalizer_sha256']:raise RuntimeError(f'model normalizer mismatch: {item["name"]}')
   model=V2TaskConditionedPredictor(seed=seed,video_mode=mode,waypoint_mode=None);model.load_state_dict(load_file(str(model_path),device='cpu'),strict=True)
   before=state_hash(model);model.eval().to(device)
   prefix,task=make_batch(data,norm,device,mode,grid if mode!='STATE-CTRL' else None)
   with torch.no_grad():
    for _ in range(int(job['warmup'])):model(prefix,task)
    torch.cuda.synchronize(device);times=[]
    for _ in range(int(job['repeats'])):
     torch.cuda.synchronize(device);t0=time.perf_counter();model(prefix,task);torch.cuda.synchronize(device);times.append((time.perf_counter()-t0)*1000.0)
   if state_hash(model)!=before:raise RuntimeError(f'model changed during latency measurement: {item["name"]}')
   rows.append({'head':item['name'],'role':item['role'],'program_id':item.get('program_id'),'video_mode':mode,'seed':seed,
                'batch_size':32,'warmup_iterations':int(job['warmup']),'timed_iterations':int(job['repeats']),
                'median_ms':float(np.median(times)),'p25_ms':float(np.percentile(times,25)),'p75_ms':float(np.percentile(times,75)),
                'model_sha256':sha(model_path),'parameter_count':sum(int(x.numel()) for x in model.parameters()),
                'input':'cached state/history/task/anchor and accepted Q0 grid where required'})
  result={'schema_version':1,'status':'PASS','created_utc':utc(),'job_id':job['job_id'],'seed':seed,'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),
          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':str(device),'cached_feature_head_cost_only':True,
          'encoder_cost_included':False,'models':rows,'peak_memory_allocated_mib':float(torch.cuda.max_memory_allocated(device)/(1024*1024)),
          'peak_memory_reserved_mib':float(torch.cuda.max_memory_reserved(device)/(1024*1024)),'final_heldout_payload_access':0}
  atomic_json(out/'LATENCY.json',result);atomic_json(status,{'status':'complete','job_id':job['job_id'],'completed_utc':utc()})
  print(json.dumps({'status':'PASS','job_id':job['job_id'],'models':len(rows),'physical_gpu':result['physical_gpu']},sort_keys=True));return 0
 except Exception as exc:
  atomic_json(status,{'status':'failed','job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown')})
  print(json.dumps({'status':'FAILED','job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc)},sort_keys=True),flush=True);return 1
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);return run(ap.parse_args().job_json)
if __name__=='__main__':raise SystemExit(main())
