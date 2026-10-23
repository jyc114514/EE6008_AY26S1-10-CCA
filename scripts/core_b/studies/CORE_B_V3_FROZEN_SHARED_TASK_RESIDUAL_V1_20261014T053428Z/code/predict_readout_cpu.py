#!/usr/bin/env python3
"""CPU prediction for already-trained small readouts and train-only bias constants."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from v3_common import ROOT,V2,TASK_IDS,read_json,sha,load_prefix_archive
from train_readout import Readout

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def atomic_json(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def write_rows(p,rows):
 fields=['row','anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame']
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def run(job_path:Path):
 job=read_json(job_path);out=Path(job['output_root']);marker=out/'PREDICTION_COMPLETE.json'
 if marker.exists() and read_json(marker).get('status')=='PASS':print(json.dumps({'status':'already_complete','job_id':job['job_id']}));return 0
 if out.exists() and any(out.iterdir()):raise RuntimeError(f'preserving incomplete CPU prediction attempt: {out}')
 out.mkdir(parents=True,exist_ok=True);started=time.time();status_path=out/'RUN_STATUS.json'
 def status(s,**kw):atomic_json(status_path,{'schema_version':1,'job_id':job['job_id'],'status':s,'updated_utc':time.time(),**kw})
 status('running',execution_device='cpu',gpu_started=False,final_heldout_payload_access=0)
 try:
  mode=job['method_kind'];prefix=load_prefix_archive(V2/'data/DEV_PREFIXES_NO_LABELS.npz','dev')
  prefix_ids=prefix['anchor_id'].astype(str).tolist()
  if len(set(prefix_ids))!=len(prefix_ids):raise RuntimeError('duplicate V2 dev prefix IDs')
  pidx={x:i for i,x in enumerate(prefix_ids)}
  # Output IDs follow canonical anchor_id order, identical to V2 inference row order.
  ids=sorted(prefix_ids)
  indices=np.asarray([pidx[x] for x in ids],dtype=np.int64)
  task_ids=prefix['task_id'][indices].astype(np.int64)
  if set(task_ids.tolist())!=set(TASK_IDS):raise RuntimeError('V3 readout predictions lack one or more fixed tasks')
  anchors=prefix['anchor_native'][indices].astype(np.float64)
  normalizer=read_json(Path(job['normalizer_path']));sigma=np.asarray(normalizer['target_sigma'],dtype=np.float64)
  if mode=='readout':
   fit_path=Path(job['readout_path']);fit_complete=read_json(Path(job['fit_complete_path']))
   if fit_complete.get('status')!='PASS' or sha(fit_path)!=fit_complete.get('readout_sha256'):
    raise RuntimeError('formal readout file does not match accepted FIT_COMPLETE')
   feature_path=Path(job['dev_feature_path']);feature_complete=read_json(Path(job['dev_feature_complete_path']))
   if feature_complete.get('status')!='PASS':raise RuntimeError('formal base dev feature receipt not PASS')
   feat=load_file(str(feature_path),device='cpu');row_list=read_csv(Path(job['dev_feature_rows_path']))
   if len(row_list)!=len(prefix_ids) or {x['anchor_id'] for x in row_list}!=set(prefix_ids):raise RuntimeError('dev base feature IDs differ from V2 label-free prefix')
   fid={x['anchor_id']:i for i,x in enumerate(row_list)};frows=np.asarray([fid[x] for x in ids],dtype=np.int64)
   if feat['h'].shape!=(len(prefix_ids),30,128) or feat['z_base'].shape!=(len(prefix_ids),30,28):raise RuntimeError('formal dev hidden/base output shape mismatch')
   h=feat['h'][frows].float();z=feat['z_base'][frows].float()
   readout=Readout(str(job['program'])).cpu().eval();weights=load_file(str(fit_path),device='cpu')
   missing,unexpected=readout.load_state_dict(weights,strict=True)
   if missing or unexpected:raise RuntimeError(f'safe readout strict reload mismatch: {missing}/{unexpected}')
   ranks=torch.as_tensor(prefix['task_rank'][indices].astype(np.int64),dtype=torch.long)
   with torch.no_grad():corr=readout(h,ranks);residual=(z+corr).numpy().astype(np.float32)
   native=anchors[:,None,:]+sigma[None,None,:]*residual.astype(np.float64)
   source={'readout_path':str(fit_path),'readout_sha256':sha(fit_path),'fit_complete_path':str(job['fit_complete_path']),
    'fit_complete_sha256':sha(Path(job['fit_complete_path'])),'base_model_id':job['base_model_id'],
    'base_model_sha256':job['base_model_sha256'],'dev_features_sha256':sha(feature_path),'base_mode':job['modality']}
   correction=corr.numpy().astype(np.float32)
  elif mode=='bias':
   base_ref=job['base_prediction_reference'];pred_path=Path(base_ref['prediction_path']);ids_path=Path(base_ref['ids_path'])
   tensors=load_file(str(pred_path),device='cpu');old_rows=read_csv(ids_path)
   oldids=[x['anchor_id'] for x in old_rows]
   if set(oldids)!=set(ids) or len(oldids)!=len(ids):raise RuntimeError('V2 base prediction IDs differ from V2 dev prefix')
   old_ix={x:i for i,x in enumerate(oldids)};order=np.asarray([old_ix[x] for x in ids],dtype=np.int64)
   base_native=tensors['prediction_native'][order].numpy().astype(np.float64)
   biasrows=read_csv(Path(job['bias_constants_path']))
   selected=[x for x in biasrows if int(x['seed'])==int(job['seed']) and x['model_id']==job['base_model_id']]
   bmap={(int(x['task_id']),int(x['action_axis_index'])):float(x['bias_native']) for x in selected}
   if len(bmap)!=5*28:raise RuntimeError('bias table is incomplete for selected formal base')
   correction=np.asarray([[bmap[(int(t),axis)] for axis in range(28)] for t in task_ids],dtype=np.float64)
   native=base_native+correction[:,None,:]
   residual=((native-anchors[:,None,:])/sigma[None,None,:]).astype(np.float32)
   source={'bias_constants_path':str(job['bias_constants_path']),'bias_constants_sha256':sha(Path(job['bias_constants_path'])),
    'base_model_id':job['base_model_id'],'base_model_sha256':job['base_model_sha256'],'v2_base_prediction_path':str(pred_path),
    'v2_base_prediction_sha256':sha(pred_path),'v2_base_ids_path':str(ids_path),'v2_base_ids_sha256':sha(ids_path),'base_mode':job['modality']}
  else:raise ValueError(f'unknown method_kind {mode}')
  if native.shape!=(185,30,28) or not np.isfinite(native).all():raise RuntimeError(f'V3 prediction shape/finite failure: {native.shape}')
  tensors={'prediction_native':torch.as_tensor(native,dtype=torch.float64).contiguous(),
   'action_residual':torch.as_tensor(residual,dtype=torch.float32).contiguous()}
  if mode=='readout':tensors['readout_correction_normalized']=torch.as_tensor(correction,dtype=torch.float32).contiguous()
  else:tensors['bias_correction_native']=torch.as_tensor(correction,dtype=torch.float64).contiguous()
  pred_path=out/'PREDICTIONS.safetensors';tmp=pred_path.with_name(f'.{pred_path.name}.{os.getpid()}.part');save_file(tensors,str(tmp));os.replace(tmp,pred_path)
  rowlist=[]
  for row,i in enumerate(indices):
   rowlist.append({'row':row,'anchor_id':str(prefix['anchor_id'][i]),'episode_id':int(prefix['episode_id'][i]),
    'task_id':int(prefix['task_id'][i]),'task_rank':int(prefix['task_rank'][i]),'source_cluster':str(prefix['source_cluster'][i]),
    'anchor_frame':int(prefix['anchor_frame'][i])})
  ids_out=out/'PREDICTION_ROWS.csv';write_rows(ids_out,rowlist)
  receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'method_kind':mode,'program':job.get('program',job['method_id']),
   'method_id':job['method_id'],'modality':job['modality'],'seed':int(job['seed']),'row_count':len(ids),'shape':[185,30,28],
   'ordered_ids_sha256':hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
   'prediction_path':str(pred_path),'prediction_sha256':sha(pred_path),'prediction_rows_path':str(ids_out),'prediction_rows_sha256':sha(ids_out),
   'source':source,'future_labels_loaded':False,'future_labels_used_in_forward':False,'execution_device':'CPU','gpu_started':False,
   'elapsed_seconds':time.time()-started,'final_heldout_payload_access':0}
  atomic_json(marker,receipt);status('PASS',completion_marker=str(marker),prediction_sha256=receipt['prediction_sha256'])
  print(json.dumps(receipt,sort_keys=True));return 0
 except Exception as e:
  status('FAILED',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),final_heldout_payload_access=0);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-json',type=Path,required=True);raise SystemExit(run(p.parse_args().job_json))
