#!/usr/bin/env python3
"""Summarize sealed V8 train predictions and Y/Z curves without dev targets."""
from __future__ import annotations
import os
import csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT');SEEDS=(202610131,202610132,202610133);TASKS=(119,94,90,141,230)
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def read_csv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writej(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def write_csv(p,rows):
 if not rows:raise RuntimeError(f'empty V8 train summary {p}')
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 tmp=p.with_name(f'.{p.name}.{os.getpid()}.part')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def loss(pred,target,anchor,sigma,fdscale):
 p=np.abs(pred-target)/sigma[None,None,:];pos=.5*(p[...,:14].mean(axis=(1,2))+p[...,14:].mean(axis=(1,2)))
 d=np.abs(np.diff(pred,axis=1)/fdscale[None,None,:]-np.diff(target,axis=1)/fdscale[None,None,:]);fd=.5*(d[...,:14].mean(axis=(1,2))+d[...,14:].mean(axis=(1,2)))
 s=np.abs((pred[:,0]-anchor)/fdscale[None,:]-(target[:,0]-anchor)/fdscale[None,:]);seam=.5*(s[:,:14].mean(axis=1)+s[:,14:].mean(axis=1))
 return {'position_mae':pos,'safe_fd_l1':fd,'seam_safe_fd_l1':seam}
def macro(values,data):
 by={}
 for i,task in enumerate(data['task_id']):by.setdefault(int(task),{}).setdefault(int(data['episode_id'][i]),[]).append(i)
 task_rows={}
 for task,episodes in by.items():
  episode_metrics=[]
  for ep,idx in episodes.items():episode_metrics.append({m:float(np.mean(values[m][idx])) for m in METRICS})
  task_rows[str(task)]={m:float(np.mean([x[m] for x in episode_metrics])) for m in METRICS}
 return {m:float(np.mean([task_rows[str(task)][m] for task in TASKS])) for m in METRICS},task_rows

def load_prediction(program,seed,data):
 out=ROOT/'predictions/train'/program/f'seed{seed}'/'attempt_01';receipt=readj(out/'PREDICTION_COMPLETE.json');model=readj(ROOT/'model_only'/program/f'seed{seed}'/'attempt_01/MODEL_MANIFEST.json')
 if receipt.get('status')!='PASS' or receipt.get('dataset_role')!='train' or receipt.get('program_id')!=program or int(receipt.get('seed',-1))!=seed:raise RuntimeError(f'train prediction receipt mismatch {program}/{seed}')
 if receipt.get('teacher_dependencies_required_at_inference') is not False or receipt.get('future_GT_requested_by_inference_loader') is not False:raise RuntimeError('train inference was not teacher-free')
 p=Path(receipt['prediction_path']);rp=Path(receipt['prediction_rows_path'])
 if sha(p)!=receipt['prediction_sha256'] or sha(rp)!=receipt['prediction_rows_sha256'] or receipt.get('model_sha256')!=model.get('model_file_sha256'):raise RuntimeError(f'train prediction/model checksum mismatch {program}/{seed}')
 arr=load_file(str(p),device='cpu')['prediction_native'].numpy()
 if arr.shape!=(1280,30,28) or arr.dtype!=np.float64 or not np.isfinite(arr).all():raise RuntimeError(f'train prediction schema mismatch {program}/{seed}')
 with rp.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 by={r['anchor_id']:r for r in rows};ids=data['anchor_id'].astype(str).tolist();index={aid:i for i,aid in enumerate(ids)}
 if len(by)!=1280 or set(by)!=set(ids):raise RuntimeError(f'train prediction ID support mismatch {program}/{seed}')
 order=[]
 for aid in ids:
  r=by[aid];ix=index[aid]
  if (int(r['episode_id']),int(r['task_id']),str(r['source_cluster']))!=(int(data['episode_id'][ix]),int(data['task_id'][ix]),str(data['source_cluster'][ix])):raise RuntimeError(f'train prediction metadata mismatch {aid}')
  order.append(int(r['row']))
 if not np.array_equal(np.sort(order),np.arange(1280)):raise RuntimeError('train prediction row mapping not bijective')
 return arr[np.asarray(order,dtype=np.int64)],receipt,model

def load_teacher(role,seed,data):
 out=V7/'teacher_predictions/source'/role/f'seed{seed}';receipt=readj(out/'PREDICTION_COMPLETE.json')
 if receipt.get('status')!='PASS' or receipt.get('teacher_role')!=role or int(receipt.get('seed',-1))!=seed:raise RuntimeError(f'frozen teacher train receipt mismatch {role}/{seed}')
 pp=Path(receipt['prediction_path']);rp=Path(receipt['prediction_rows_path'])
 if sha(pp)!=receipt['prediction_sha256'] or sha(rp)!=receipt['prediction_rows_sha256']:raise RuntimeError(f'frozen teacher prediction checksum mismatch {role}/{seed}')
 arr=load_file(str(pp),device='cpu')['prediction_native'].numpy()
 with rp.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 by={r['anchor_id']:r for r in rows};ids=data['anchor_id'].astype(str).tolist()
 if arr.shape!=(1280,30,28) or len(by)!=1280 or set(by)!=set(ids):raise RuntimeError(f'frozen teacher train support mismatch {role}/{seed}')
 order=[]
 for i,aid in enumerate(ids):
  rr=by[aid]
  if (int(rr['episode_id']),int(rr['task_id']),str(rr['source_cluster']))!=(int(data['episode_id'][i]),int(data['task_id'][i]),str(data['source_cluster'][i])):raise RuntimeError(f'frozen teacher identity mismatch {role}/{seed}/{aid}')
  order.append(int(rr['row']))
 if not np.array_equal(np.sort(order),np.arange(1280)):raise RuntimeError(f'frozen teacher row map not bijective {role}/{seed}')
 return arr[np.asarray(order,dtype=np.int64)],receipt

def main():
 branch=readj(ROOT/'protocol/BRANCH_LOCK.json');recipe=readj(ROOT/'protocol/RECIPE_LOCK.json');seal=readj(ROOT/'qa/PREDICTION_SEAL.json')
 if branch.get('branch')!='B_SYNCHRONOUS_VIDEO_9' or seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS':raise RuntimeError('V8 train summary waits for branch lock and prediction seal')
 train_path=Path(recipe['train_npz_path'])
 with np.load(train_path,allow_pickle=False) as z:data={k:z[k].copy() for k in ('anchor_id','episode_id','task_id','source_cluster','anchor_native','target_native_h30')}
 if len(data['anchor_id'])!=1280:raise RuntimeError('train archive rows differ from 1280')
 y=np.asarray(data['target_native_h30'],dtype=np.float64);anchor=np.asarray(data['anchor_native'],dtype=np.float64)
 normalizer=readj(Path(recipe['normalizer_path']));sigma=np.asarray(normalizer['target_sigma'],dtype=np.float64);fdscale=np.asarray(normalizer['safe_fd_scale'],dtype=np.float64)
 all_curves=[];summary=[];prefit=[];resource=[]
 for seed in SEEDS:
  s,sr=load_teacher('S_OPT',seed,data);v,vr=load_teacher('V_MATCHED',seed,data)
  teachers={'GT_RETRAIN':v,'SELF_SOFT':v,'SV_SOFT':.5*s+.5*v}
  for program in PROGRAMS:
   slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';manifest=readj(slot/'MODEL_MANIFEST.json');complete=readj(slot/'RUN_COMPLETE.json');curve=read_csv(slot/'EPOCH_LOG.csv')
   if manifest.get('status')!='PASS' or manifest.get('training_runner_sha256')!=sha(ROOT/'code/v8_train.py') or int(manifest['epochs'])!=160 or int(manifest['optimizer_steps'])!=3200:raise RuntimeError(f'fit receipt invalid {program}/{seed}')
   if len(curve)!=160 or int(curve[-1]['epoch'])!=160:raise RuntimeError(f'fit curve incomplete {program}/{seed}')
   target_path=ROOT/'targets'/program/f'seed{seed}'/'Z_TRAIN_NATIVE.npy';target_manifest=readj(target_path.parent/'TARGET_MANIFEST.json');z=np.load(target_path,allow_pickle=False)
   if sha(target_path)!=target_manifest['target_sha256'] or not np.isfinite(z).all():raise RuntimeError(f'train target hash/finite issue {program}/{seed}')
   pred,pr,model=load_prediction(program,seed,data)
   vals_y=loss(pred,y,anchor,sigma,fdscale);vals_z=loss(pred,z,anchor,sigma,fdscale);vals_t=loss(pred,teachers[program],anchor,sigma,fdscale)
   macro_y,task_y=macro(vals_y,data);macro_z,task_z=macro(vals_z,data);macro_t,task_t=macro(vals_t,data)
   d={}
   for row in curve:
    d[int(row['epoch'])]=row
    all_curves.append({'program_id':program,'seed':seed,'epoch':int(row['epoch']),'optimizer_steps':int(row['optimizer_steps']),
     'lr_after_epoch':float(row['lr_after_epoch']),'training_target_position':float(row['training_target_position']),
     'training_target_safe_fd':float(row['training_target_safe_fd']),'training_target_objective':float(row['training_target_objective']),
     'true_y_position':float(row['true_y_position']),'true_y_safe_fd':float(row['true_y_safe_fd']),'true_y_seam_safe_fd':float(row['true_y_seam_safe_fd']),
     'clip_updates':int(row['clip_updates']),'clip_frequency':float(row['clip_frequency']),'preclip_grad_norm_mean':float(row['preclip_grad_norm_mean']),'preclip_grad_norm_max':float(row['preclip_grad_norm_max']),
     'target_role':manifest['target_role'],'target_expression':manifest['target_expression']})
   for epoch in (40,80,160):
    row=d[epoch];teacher_cols={}
    if epoch==160:teacher_cols={'eval_train_y_position':macro_y['position_mae'],'eval_train_y_safe_fd':macro_y['safe_fd_l1'],'eval_train_y_seam':macro_y['seam_safe_fd_l1'],
     'eval_train_z_position':macro_z['position_mae'],'eval_train_z_safe_fd':macro_z['safe_fd_l1'],'eval_train_z_seam':macro_z['seam_safe_fd_l1'],
     'teacher_distance_position':macro_t['position_mae'],'teacher_distance_safe_fd':macro_t['safe_fd_l1'],'teacher_distance_seam':macro_t['seam_safe_fd_l1'],
     'teacher_train_predictions_are_oof':False}
    summary.append({'program_id':program,'seed':seed,'epoch':epoch,'optimizer_steps':int(row['optimizer_steps']),'target_mode':manifest['target_mode'],
      'target_expression':manifest['target_expression'],'target_role':manifest['target_role'],'training_target_position':float(row['training_target_position']),
      'training_target_safe_fd':float(row['training_target_safe_fd']),'training_target_objective':float(row['training_target_objective']),
      'training_true_y_position':float(row['true_y_position']),'training_true_y_safe_fd':float(row['true_y_safe_fd']),'training_true_y_seam':float(row['true_y_seam_safe_fd']),
      'clip_updates':int(row['clip_updates']),'clip_frequency':float(row['clip_frequency']),'lr_after_epoch':float(row['lr_after_epoch']),**teacher_cols})
   prefit.append({'program_id':program,'seed':seed,'train_target_sha256':target_manifest['target_sha256'],'student_train_prediction_sha256':pr['prediction_sha256'],
      'teacher_source_roles':['V_MATCHED'] if program!='SV_SOFT' else ['S_OPT','V_MATCHED'],'train_y_macro':macro_y,'train_target_macro':macro_z,'teacher_distance_macro':macro_t,
      'training_target_epoch160':{'position':float(d[160]['training_target_position']),'safe_fd':float(d[160]['training_target_safe_fd']),'objective':float(d[160]['training_target_objective'])},
      'teacher_train_predictions_are_oof':False,'dev_target_archive_opened':False})
   resource.append({'record_type':'fit','job_id':manifest['job_id'],'program_id':program,'seed':seed,'status':manifest['status'],'epochs':manifest['epochs'],'optimizer_steps':manifest['optimizer_steps'],
     'physical_gpu':manifest['physical_gpu'],'cuda_visible_devices':manifest['cuda_visible_devices'],'logical_device':manifest['logical_device'],
     'peak_allocated_mib':manifest['peak_memory_allocated_mib'],'peak_reserved_mib':manifest['peak_memory_reserved_mib'],'elapsed_seconds':complete['elapsed_seconds'],
     'training_sample_presentations_total':complete['training_sample_presentations_total'],'model_sha256':manifest['model_file_sha256'],'runner_sha256':manifest['training_runner_sha256'],
     'job_config_sha256':manifest['job_config_sha256'],'sampler_sequence_sha256':manifest['sampler_sequence_sha256'],'target_sha256':manifest['training_target_sha256']})
 write_csv(ROOT/'figure_data/TRAINING_CURVES.csv',all_curves)
 write_csv(ROOT/'results/TRAIN_Y_Z_AND_IMITATION.csv',summary)
 write_csv(ROOT/'results/TRAIN_FINAL_SUMMARY.csv',prefit)
 write_csv(ROOT/'results/RESOURCE_LEDGER.csv',resource)
 qa={'schema_version':1,'status':'PASS','created_utc':utc(),'fits':9,'train_rows':1280,'train_prediction_bundles':9,'train_target_reads':'outer-train only',
  'train_target_presentations':sum(int(x['training_sample_presentations_total']) for x in resource),'epochs_per_fit':160,'optimizer_steps_per_fit':3200,
  'historical_teacher_source_oof':False,'dev_target_archive_opened':False,'final_heldout_payload_access':0,
  'curves_path':str(ROOT/'figure_data/TRAINING_CURVES.csv'),'train_summary_path':str(ROOT/'results/TRAIN_Y_Z_AND_IMITATION.csv'),
  'final_train_summary_path':str(ROOT/'results/TRAIN_FINAL_SUMMARY.csv')}
 writej(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json',qa)
 print(json.dumps({'status':'PASS','fits':9,'curve_rows':len(all_curves),'summary_rows':len(summary),'train_target_presentations':qa['train_target_presentations'],'dev_target_opened':False},sort_keys=True))
if __name__=='__main__':main()
