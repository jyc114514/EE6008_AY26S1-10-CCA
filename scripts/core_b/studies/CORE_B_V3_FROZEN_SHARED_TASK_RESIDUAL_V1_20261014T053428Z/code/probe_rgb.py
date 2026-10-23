#!/usr/bin/env python3
"""Fixed-checkpoint RGB zero/donor sensitivity prediction; never reads outcomes."""
from __future__ import annotations
import argparse,csv,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from v3_common import ROOT,V2,TASK_IDS,read_json,atomic_json,sha,load_prefix_archive,make_prefix,select_base,load_base,capture_forward,Q0ShardedGrid

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save_safe(path,tensors):
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.part');save_file({k:v.detach().cpu().contiguous() for k,v in tensors.items()},str(tmp));os.replace(tmp,path)
def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(job_path):
 job=read_json(job_path);out=Path(job['output_root']);marker=out/'PREDICTION_COMPLETE.json'
 if marker.exists() and read_json(marker).get('status')=='PASS':print(json.dumps({'status':'already_complete','job_id':job['job_id']}));return 0
 if out.exists() and any(out.iterdir()):raise RuntimeError(f'preserving incomplete RGB probe attempt: {out}')
 out.mkdir(parents=True,exist_ok=True);started=time.time();status_path=out/'RUN_STATUS.json'
 def status(s,**kw):atomic_json(status_path,{'schema_version':1,'job_id':job['job_id'],'status':s,'updated_utc':time.time(),**kw})
 status('running',gpu_watcher_physical=os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','unknown'))
 try:
  if not torch.cuda.is_available():raise RuntimeError('RGB probe requires watcher-authorized CUDA device')
  seed=int(job['seed']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  device=torch.device('cuda:0');model_id=str(job['model_id']);seed=int(job['seed']);condition=str(job['condition'])
  if condition not in {'ZERO_NORMALIZED_BLOCK','SAME_TASK_DONOR'}:raise ValueError(condition)
  stage='formal' if model_id=='G-V' else 'probe_control'
  base=select_base(stage,model_id,seed);norm=read_json(Path(base['normalizer_path']))
  data=load_prefix_archive(Path(job['dev_prefix_path']),'dev')
  keep=np.ones(len(data['anchor_id']),dtype=bool)
  if model_id=='S-V-90':keep &= data['task_id']==90
  indices=np.flatnonzero(keep);indices=indices[np.argsort(data['anchor_id'][indices],kind='stable')]
  if len(indices)!=(185 if model_id=='G-V' else 41):raise RuntimeError(f'probe dev support changed for {model_id}/{seed}')
  grid=Q0ShardedGrid(read_json(Path(job['cache_manifest_path'])))
  matched_prefix,task_rank=make_prefix(data,indices,norm,grid,'TS11-D',device)
  matched_h=None;matched_z=None;matched_replay_forwards=0
  if model_id=='G-V':
   feature_root=ROOT/'features/formal'/f'G_V_s{seed}'/'attempt_01'
   feat=load_file(str(feature_root/'DEV_FEATURES.safetensors'),device='cpu');frows=read_csv(feature_root/'DEV_ROWS.csv')
   fids=[x['anchor_id'] for x in frows];wanted=[str(data['anchor_id'][i]) for i in indices]
   if len(set(fids))!=len(fids) or set(fids)!=set(wanted):raise RuntimeError('G-V matched hidden ID set differs from locked probe rows')
   fidx={x:i for i,x in enumerate(fids)};order=np.asarray([fidx[x] for x in wanted],dtype=np.int64)
   matched_h=feat['h'][order].to(device);matched_z=feat['z_base'][order].to(device)
   matched_source={'kind':'formal V3 frozen base feature extraction','path':str(feature_root/'DEV_FEATURES.safetensors'),
      'sha256':sha(feature_root/'DEV_FEATURES.safetensors')}
  else:
   model,before,_=load_base(base,device)
   matched_h,matched_z=capture_forward(model,matched_prefix,task_rank);matched_replay_forwards=1
   # Validate the same-checkpoint matched output against the accepted V2 prediction for these IDs.
   ref=job['matched_v2_prediction_reference'];old_rows=read_csv(Path(ref['ids_path']))
   old=load_file(ref['prediction_path'],device='cpu')['prediction_native'].numpy()
   old_map={x['anchor_id']:old[i] for i,x in enumerate(old_rows)}
   anchors=data['anchor_native'][indices].astype(np.float64);sigma=np.asarray(norm['target_sigma'],dtype=np.float64)
   matched_native=anchors[:,None,:]+sigma[None,None,:]*matched_z.detach().cpu().numpy().astype(np.float64)
   target_ref=np.stack([old_map[str(data['anchor_id'][i])] for i in indices]).astype(np.float64)
   if not np.allclose(matched_native,target_ref,rtol=1e-6,atol=1e-6):raise RuntimeError('S-V-90 matched forward differs from accepted V2 output')
   if __import__('v3_common').state_hash(model)!=before:raise RuntimeError('S-V-90 base changed during matched QA forward')
   matched_source={'kind':'fresh matched replay validated against V2 accepted prediction','path':ref['prediction_path'],
      'prediction_sha256':ref['prediction_sha256'],'matched_replay_forwards':matched_replay_forwards}
  condition_prefix={k:v.clone() for k,v in matched_prefix.items()}
  donors=[]
  if condition=='ZERO_NORMALIZED_BLOCK':
   condition_prefix['dynamic']=torch.zeros_like(matched_prefix['dynamic']);condition_prefix['static']=torch.zeros_like(matched_prefix['static'])
   for i in indices:donors.append({'donor_status':'ZERO'})
  else:
   donor_rows={x['receiver_anchor_id']:x for x in read_csv(Path(job['donor_map_path']))}
   vmean=np.asarray(norm['video_mean'],dtype=np.float32)[None,None,None,:];vstd=np.asarray(norm['video_safe_std'],dtype=np.float32)[None,None,None,:]
   for pos,ix in enumerate(indices.tolist()):
    rid=str(data['anchor_id'][ix]);dr=donor_rows[rid]
    if dr['donor_status']!='PASS' or int(dr['task_id'])!=int(data['task_id'][ix]) or int(dr['donor_episode_id'])==int(data['episode_id'][ix]):
     raise RuntimeError(f'invalid same-task different-episode donor for receiver {rid}')
    donor_id=dr['donor_anchor_id'];dix=np.flatnonzero(data['anchor_id'].astype(str)==donor_id)
    if len(dix)!=1:raise RuntimeError(f'donor anchor missing from label-free dev prefix: {donor_id}')
    di=int(dix[0]);raw=grid.row(int(data['cache_index'][di]),str(data['anchor_id'][di]),str(data['episode_id'][di])).astype(np.float32,copy=False)
    condition_prefix['dynamic'][pos]=torch.as_tensor((raw[0]-vmean[0])/vstd[0],device=device,dtype=torch.float32)
    condition_prefix['static'][pos]=torch.as_tensor((raw[1]-vmean[0])/vstd[0],device=device,dtype=torch.float32)
    donors.append(dr)
  for key in ('state','history_mask','relative_delta','anchor_command'):
   if not torch.equal(condition_prefix[key],matched_prefix[key]):raise RuntimeError(f'RGB probe altered non-RGB prefix field {key}')
  if condition=='ZERO_NORMALIZED_BLOCK':
   if torch.count_nonzero(condition_prefix['dynamic']).item()!=0 or torch.count_nonzero(condition_prefix['static']).item()!=0:raise RuntimeError('zero condition did not produce exact standardized zero blocks')
  model,before,_=load_base(base,device)
  h_condition,z_condition=capture_forward(model,condition_prefix,task_rank)
  # The matched view is in the same checkpoint coordinate; do not compare hidden states across seeds/models.
  h_delta=(h_condition-matched_h).detach().float().cpu().numpy().astype(np.float64)
  z_delta=(z_condition-matched_z).detach().float().cpu().numpy().astype(np.float64)
  h_match=matched_h.detach().float().cpu().numpy().astype(np.float64)
  pred_res=z_condition.detach().float().cpu().numpy().astype(np.float32)
  anchors=data['anchor_native'][indices].astype(np.float64);sigma=np.asarray(norm['target_sigma'],dtype=np.float64)
  native=anchors[:,None,:]+sigma[None,None,:]*pred_res.astype(np.float64)
  diag=[]
  for pos,ix in enumerate(indices.tolist()):
   dh=h_delta[pos];dm=h_match[pos];dz=z_delta[pos]
   donor=donors[pos]
   diag.append({'row':pos,'anchor_id':str(data['anchor_id'][ix]),'episode_id':int(data['episode_id'][ix]),
    'task_id':int(data['task_id'][ix]),'task_rank':int(data['task_rank'][ix]),'source_cluster':str(data['source_cluster'][ix]),
    'condition':condition,'h_mean_abs_change':float(np.abs(dh).mean()),
    'h_relative_l2_change':float(np.linalg.norm(dh)/(np.linalg.norm(dm)+1e-12)),
    'base_residual_mean_abs_change':float(np.abs(dz).mean()),'base_residual_max_abs_change':float(np.abs(dz).max()),
    'donor_status':donor.get('donor_status','PASS'),'donor_episode_id':donor.get('donor_episode_id',''),
    'donor_anchor_id':donor.get('donor_anchor_id',''),'donor_selection_sha256':donor.get('selection_sha256','')})
  # Checkpoint unchanged through matched and perturbation forwards.
  if __import__('v3_common').state_hash(model)!=before:raise RuntimeError('video base changed during RGB probe')
  pred=out/'PREDICTIONS.safetensors';tmp=pred.with_name(f'.{pred.name}.{os.getpid()}.part')
  save_file({'action_residual':torch.as_tensor(pred_res).contiguous(),'prediction_native':torch.as_tensor(native,dtype=torch.float64).contiguous()},str(tmp));os.replace(tmp,pred)
  idrows=[]
  for out_i,ix in enumerate(indices.tolist()):
   idrows.append({'row':out_i,'anchor_id':str(data['anchor_id'][ix]),'episode_id':int(data['episode_id'][ix]),
    'task_id':int(data['task_id'][ix]),'task_rank':int(data['task_rank'][ix]),'source_cluster':str(data['source_cluster'][ix]),
    'anchor_frame':int(data['anchor_frame'][ix])})
  idpath=out/'PREDICTION_ROWS.csv';
  with idpath.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(idrows[0]));w.writeheader();w.writerows(idrows)
  diagpath=out/'PROBE_DIAGNOSTICS.csv';write_csv(diagpath,diag)
  torch.cuda.synchronize()
  receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'model_id':model_id,'seed':seed,'condition':condition,
   'row_count':len(indices),'prediction_shape':list(native.shape),'prediction_path':str(pred),'prediction_sha256':sha(pred),
   'prediction_rows_path':str(idpath),'prediction_rows_sha256':sha(idpath),'probe_diagnostics_path':str(diagpath),'probe_diagnostics_sha256':sha(diagpath),
   'matched_reference':matched_source,'matched_replay_forwards':matched_replay_forwards,'condition_intervention':job['condition_definition'],
   'non_rgb_prefix_fields_unchanged':['state','history_mask','relative_delta','anchor_command','task_id'],
   'base_state_hash':before,'base_model_sha256':base['model_sha256'],'normalizer_sha256':base['normalizer_sha256'],
   'dev_target_values_loaded':False,'dev_target_labels_used_for_condition_choice':False,
   'gpu_physical':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
   'logical_device':str(device),'peak_memory_allocated_mib':torch.cuda.max_memory_allocated(device)/(1024**2),
   'peak_memory_reserved_mib':torch.cuda.max_memory_reserved(device)/(1024**2),'elapsed_seconds':time.time()-started,
   'final_heldout_payload_access':0}
  atomic_json(marker,receipt);status('PASS',completion_marker=str(marker),prediction_sha256=receipt['prediction_sha256'])
  print(json.dumps(receipt,sort_keys=True));return 0
 except Exception as e:
  status('FAILED',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),dev_target_values_loaded=False,final_heldout_payload_access=0);raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);raise SystemExit(run(ap.parse_args().job_json))
