#!/usr/bin/env python3
"""Extract only frozen V2 hidden/output tensors, from prefix-only inputs."""
from __future__ import annotations
import argparse,csv,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from v3_common import ROOT,V2,read_json,atomic_json,sha,load_prefix_archive,make_prefix,select_base,load_base,capture_forward,Q0ShardedGrid

def write_rows(path,data):
 fields=['row','anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame','inner_role']
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for i in range(len(data['anchor_id'])):
   w.writerow({'row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),
    'task_id':int(data['task_id'][i]),'task_rank':int(data['task_rank'][i]),'source_cluster':str(data['source_cluster'][i]),
    'anchor_frame':int(data['anchor_frame'][i]),'inner_role':str(data.get('inner_role',np.asarray(['NA']*len(data['anchor_id'])))[i])})

def atomic_safe(path,tensors):
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 save_file({k:v.detach().to('cpu').contiguous() for k,v in tensors.items()},str(tmp));os.replace(tmp,path)

def run(job_path):
 job=read_json(job_path);out=Path(job['output_root'])
 complete=out/'FEATURES_COMPLETE.json'
 if complete.is_file() and read_json(complete).get('status')=='PASS':
  print(json.dumps({'status':'already_complete','job_id':job['job_id']}));return 0
 if out.exists() and any(out.iterdir()):raise RuntimeError(f'preserving incomplete feature extraction output: {out}')
 out.mkdir(parents=True,exist_ok=True)
 started=time.time();status_path=out/'RUN_STATUS.json'
 def status(value,**kw):atomic_json(status_path,{'schema_version':1,'job_id':job['job_id'],'status':value,'updated_utc':time.time(),**kw})
 status('running',gpu_watcher_physical=os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','unknown'))
 try:
  if not torch.cuda.is_available():raise RuntimeError('base hidden extraction requires a watcher-authorized CUDA device')
  seed=int(job['seed']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  device=torch.device('cuda:0');base=job['base']
  norm=read_json(Path(base['normalizer_path']))
  model,before,manifest=load_base(base,device)
  grid=None if base['video_mode']=='STATE-CTRL' else Q0ShardedGrid(read_json(Path(job['cache_manifest_path'])))
  artifacts=[]
  for dataset in job['datasets']:
   if dataset=='train': data=load_prefix_archive(Path(job['train_prefix_path']),'train')
   elif dataset=='dev': data=load_prefix_archive(Path(job['dev_prefix_path']),'dev')
   else:raise ValueError(dataset)
   n=len(data['anchor_id']);h_parts=[];z_parts=[]
   with torch.no_grad():
    for start in range(0,n,int(job['batch_size'])):
     ix=np.arange(start,min(start+int(job['batch_size']),n),dtype=np.int64)
     prefix,task=make_prefix(data,ix,norm,grid,str(base['video_mode']),device)
     h,z=capture_forward(model,prefix,task)
     if not torch.isfinite(h).all() or not torch.isfinite(z).all():raise FloatingPointError(f'nonfinite hidden/output in {dataset} batch {start}')
     h_parts.append(h.detach().float().cpu());z_parts.append(z.detach().float().cpu())
   h_all=torch.cat(h_parts);z_all=torch.cat(z_parts)
   if h_all.shape!=(n,30,128) or z_all.shape!=(n,30,28):raise RuntimeError('hidden/output aggregate shape mismatch')
   safe_path=out/f'{dataset.upper()}_FEATURES.safetensors'
   atomic_safe(safe_path,{'h':h_all,'z_base':z_all})
   rows_path=out/f'{dataset.upper()}_ROWS.csv';write_rows(rows_path,data)
   artifacts.append({'dataset':dataset,'rows':n,'hidden_shape':list(h_all.shape),'base_output_shape':list(z_all.shape),
    'features_path':str(safe_path),'features_sha256':sha(safe_path),'rows_path':str(rows_path),'rows_sha256':sha(rows_path),
    'future_target_columns_requested':False,'future_labels_in_forward':False})
   del data,h_parts,z_parts,h_all,z_all
  after=__import__('v3_common').state_hash(model)
  if after!=before:raise RuntimeError('frozen V2 base state changed during hidden extraction')
  torch.cuda.synchronize()
  receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'stage':job['stage'],'model_id':base['model_id'],'seed':seed,
   'base_job_id':base['job_id'],'base_model_path':base['model_path'],'base_model_sha256':base['model_sha256'],
   'base_state_hash_before':before,'base_state_hash_after':after,'base_unchanged':True,
   'normalizer_path':base['normalizer_path'],'normalizer_sha256':base['normalizer_sha256'],
   'feature_cache_manifest_sha256':job['cache_manifest_sha256'],'video_mode':base['video_mode'],'batch_size':int(job['batch_size']),
   'artifacts':artifacts,'encoder_training':0,'encoder_extraction':0,'frozen_V2_forward_only':True,
   'gpu_physical':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
   'logical_device':str(device),'peak_memory_allocated_mib':torch.cuda.max_memory_allocated(device)/(1024**2),
   'peak_memory_reserved_mib':torch.cuda.max_memory_reserved(device)/(1024**2),'elapsed_seconds':time.time()-started,
   'final_heldout_payload_access':0}
  atomic_json(complete,receipt);status('PASS',completion_marker=str(complete),base_state_unchanged=True)
  print(json.dumps(receipt,sort_keys=True));return 0
 except Exception as e:
  status('FAILED',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),final_heldout_payload_access=0);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-json',type=Path,required=True);raise SystemExit(run(p.parse_args().job_json))
