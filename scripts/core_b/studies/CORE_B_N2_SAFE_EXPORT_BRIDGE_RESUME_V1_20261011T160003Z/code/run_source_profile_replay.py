#!/usr/bin/env python3
"""QA-only source-profile replay from accepted safe tensors; never loads raw checkpoints or targets."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
N2_CODE=ROOT/'code'
sys.path.insert(0,str(N2_CODE))
from run_n2_inference import GridStore,model_for,select_inputs,sha_file,sha_query

INPUTS=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz'
CACHE_MANIFEST=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json'
VIDEO_NORMALIZER=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_VIDEO_NORMALIZER.json'
TARGET_NORMALIZER=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_NORMALIZERS.json'
QUERY_ARCHIVE=PROJECT/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z/protocol/HADAMARD_QUERY_TABLES.npz'
REQUEST=ROOT/'MODEL_EXPORT_REQUEST.csv'

def utc_now():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--backend',choices=('q0_cpu','q1_cuda'),required=True)
 args=ap.parse_args()
 kind=args.backend
 with REQUEST.open(newline='',encoding='utf-8') as f:all_rows=list(csv.DictReader(f))
 if kind=='q0_cpu':
  rows=[r for r in all_rows if r['query']=='Q0'];profile='N1_AS_RUN_CPU_Q0'
  torch.set_num_threads(8);torch.set_num_interop_threads(1);device=torch.device('cpu');physical=None;cvd=None;uuid=None
 else:
  rows=[r for r in all_rows if r['query']=='Q1'];profile='N1_AS_RUN_CUDA_Q1'
  physical=int(os.environ.get('GPU_WATCHER_PHYSICAL_GPU','-1'));cvd=os.environ.get('CUDA_VISIBLE_DEVICES','')
  if physical!=0 or cvd!='0':raise RuntimeError('strict Q1 source-profile replay requires original physical GPU0 and CUDA_VISIBLE_DEVICES=0')
  if not torch.cuda.is_available():raise RuntimeError('CUDA0 source-profile replay requires available CUDA')
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
  torch.use_deterministic_algorithms(False)
  torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True)
  torch.cuda.set_device(0);device=torch.device('cuda:0')
  query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  uuid_by_index={}
  for line in query.stdout.splitlines():
   fields=[x.strip() for x in line.split(',')]
   if len(fields)==2:uuid_by_index[int(fields[0])]=fields[1]
  if 0 not in uuid_by_index:raise RuntimeError('original physical GPU0 UUID not visible')
  uuid=uuid_by_index[0]
 if len(rows)!=6:raise RuntimeError('source-profile replay must select six fixed models')
 with np.load(INPUTS,allow_pickle=False) as archive:
  split=archive['split_code'].astype(np.int8,copy=True);idx=np.flatnonzero(split==1).astype(np.int64)
  if len(idx)!=2322:raise RuntimeError('source replay dev row count mismatch')
  data=select_inputs(archive,idx)
 ids=data['anchor_id'].astype('U96')
 if len(set(ids.tolist()))!=2322:raise RuntimeError('source replay dev IDs not unique')
 with np.load(QUERY_ARCHIVE,allow_pickle=False) as z:table=np.asarray(z['candidate_h30'],dtype=np.float32).copy()
 cache=json.loads(CACHE_MANIFEST.read_text());video=json.loads(VIDEO_NORMALIZER.read_text());target=json.loads(TARGET_NORMALIZER.read_text())
 mean=np.asarray(video['mean'],dtype=np.float32);std=np.asarray(video['safe_std'],dtype=np.float32)
 sigma=torch.as_tensor(np.asarray(target['target_sigma'],dtype=np.float32),device=device).view(1,1,28)
 state=torch.from_numpy(data['history_standardized'].astype(np.float32,copy=False))
 mask=torch.from_numpy(data['history_mask'].astype(np.float32,copy=False))
 relative=torch.from_numpy(data['history_relative_delta_standardized'].astype(np.float32,copy=False))
 anchor=torch.from_numpy(data['anchor_command_native'].astype(np.float32,copy=False))
 grid=GridStore(cache)
 records=[]
 if kind=='q1_cuda':torch.cuda.reset_peak_memory_stats(device)
 for row in rows:
  model_id=row['model_id'];safe=ROOT/'safe_models'/model_id
  manifest=json.loads((safe/'EXPORT_MANIFEST.json').read_text());safe_file=safe/'model.safetensors'
  if sha(safe_file)!=manifest['model_file_sha256'] or manifest['source_checkpoint_sha256']!=row['checkpoint_sha256']:
   raise RuntimeError('safe model binding mismatch '+model_id)
  model,nparams=model_for(row,table)
  sd=load_file(str(safe_file),device='cpu');incompatible=model.load_state_dict(sd,strict=True)
  if incompatible.missing_keys or incompatible.unexpected_keys or nparams!=890780:
   raise RuntimeError('strict source-profile model reload failed '+model_id)
  model.to(device);model.eval()
  output=np.empty((2322,30,28),dtype=np.float32)
  t0=time.perf_counter()
  with torch.inference_mode():
   for start in range(0,2322,64):
    end=min(start+64,2322);dynamic,static=grid.get(idx[start:end])
    dynamic=((dynamic-mean[None,None,None,:])/std[None,None,None,:]).astype(np.float32)
    static=((static-mean[None,None,None,:])/std[None,None,None,:]).astype(np.float32)
    state_b=state[start:end].to(device=device,dtype=torch.float32);mask_b=mask[start:end].to(device=device,dtype=torch.float32)
    rel_b=relative[start:end].to(device=device,dtype=torch.float32);anchor_b=anchor[start:end].to(device=device,dtype=torch.float32)
    residual=model(state_b,mask_b,rel_b,torch.from_numpy(dynamic).to(device),torch.from_numpy(static).to(device),anchor_b)
    output[start:end]=(anchor_b[:,None,:]+residual*sigma).detach().cpu().numpy().astype(np.float32,copy=False)
  if output.shape!=(2322,30,28) or not np.isfinite(output).all():raise RuntimeError('source-profile output shape/finite failure '+model_id)
  refpath=Path(row['accepted_n1_dev_prediction_path'])
  if sha(refpath)!=row['accepted_n1_dev_prediction_sha256']:raise RuntimeError('accepted N1 source prediction SHA changed '+model_id)
  with np.load(refpath,allow_pickle=False) as z:
   refids=z['anchor_id'].astype('U96');reference=z['predictions_native'].astype(np.float32,copy=True)
  if not np.array_equal(refids,ids) or reference.shape!=output.shape:raise RuntimeError('source replay reference IDs/shape mismatch '+model_id)
  absdiff=np.abs(output-reference);max_abs=float(absdiff.max());mean_abs=float(absdiff.mean());p99=float(np.quantile(absdiff,.99))
  side=json.loads(Path(row['accepted_profile_sidecar_path']).read_text())
  if sha(Path(row['accepted_profile_sidecar_path']))!=row['accepted_profile_sidecar_sha256']:raise RuntimeError('source-profile sidecar SHA mismatch '+model_id)
  if side.get('status')!='PASS' or int(side.get('rows',-1))!=2322:raise RuntimeError('source-profile sidecar not accepted '+model_id)
  if kind=='q0_cpu' and side.get('device')!='cpu':raise RuntimeError('Q0 accepted source profile is not CPU '+model_id)
  if kind=='q1_cuda':
   rt=side.get('runtime',{})
   expected={'physical_gpu_index':'0','cuda_visible_devices':'0','torch_num_threads':4,'torch_num_interop_threads':1,
    'tf32_matmul_enabled':False,'tf32_cudnn_enabled':True,'deterministic_algorithms':False,
    'sdpa_flash_enabled':True,'sdpa_math_enabled':True,'sdpa_memory_efficient_enabled':True,
    'autocast_used':False,'torch_compile_used':False}
   for k,v in expected.items():
    if rt.get(k)!=v:raise RuntimeError('Q1 original runtime metadata mismatch '+k+' for '+model_id)
  # Strict numeric parity is exact: do not relax the historical reference tolerance.
  status='PASS_EXACT' if max_abs==0.0 else 'FAIL_TOLERANCE_EXCEEDED'
  records.append({'model_id':model_id,'model':row['model'],'query':row['query'],'seed':int(row['seed']),
   'source_profile':profile,'device':str(device),'physical_gpu':physical,'cuda_visible_devices':cvd,'gpu_uuid':uuid,
   'historical_profile_sidecar_sha256':row['accepted_profile_sidecar_sha256'],
   'historical_prediction_path':str(refpath),'historical_prediction_sha256':row['accepted_n1_dev_prediction_sha256'],
   'safe_model_sha256':sha(safe_file),'checkpoint_sha256':row['checkpoint_sha256'],
   'input_sha256':sha(INPUTS),'feature_cache_manifest_sha256':sha(CACHE_MANIFEST),
   'video_normalizer_sha256':sha(VIDEO_NORMALIZER),'target_normalizer_sha256':sha(TARGET_NORMALIZER),
   'query_table_sha256':row['accepted_query_table_sha256'],'rows':2322,'shape':[2322,30,28],
   'max_abs_difference':max_abs,'mean_abs_difference':mean_abs,'p99_abs_difference':p99,'tolerance_abs':0.0,
   'elapsed_seconds':round(time.perf_counter()-t0,6),'targets_loaded':False,'final_test_access':0,'status':status})
  print(json.dumps({'view':model_id,'status':status,'max_abs_difference':max_abs,'mean_abs_difference':mean_abs},sort_keys=True),flush=True)
  del model,sd,output
 if kind=='q1_cuda':records[0]['gpu_peak_allocated_mib']=int(torch.cuda.max_memory_allocated(device)/1024/1024)
 out=ROOT/'qa'/'SOURCE_PROFILE_REPLAY_Q0_CPU.json' if kind=='q0_cpu' else ROOT/'qa'/'SOURCE_PROFILE_REPLAY_Q1_CUDA.json'
 result={'schema_version':1,'status':'PASS' if all(r['status']=='PASS_EXACT' for r in records) else 'FAIL_TOLERANCE_EXCEEDED',
  'source_profile':profile,'models':records,'model_count':len(records),'targets_loaded':False,'final_test_access':0,
  'tolerance_abs':0.0,'created_utc':utc_now(),'runner_sha256':sha(Path(__file__)),
  'source_replay_scope':'QA-only; not included in locked N2_MATRIX24 and not scored'}
 write_json(out,result)
 if result['status']!='PASS':raise RuntimeError('strict source-profile replay did not reproduce exact accepted stored predictions')
 return 0

if __name__=='__main__':raise SystemExit(main())
