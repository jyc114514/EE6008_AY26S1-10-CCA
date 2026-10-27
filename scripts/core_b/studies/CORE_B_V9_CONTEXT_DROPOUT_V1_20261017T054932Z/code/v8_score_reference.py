#!/usr/bin/env python3
"""Seal-gated one-read V8 dev scoring, paired contrasts, LOO, and group bootstrap."""
from __future__ import annotations
import os
import csv,hashlib,io,json,os,zipfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.torch import load_file
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT=Path(__file__).resolve().parents[1]
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V5=PROJECT/'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT')
OLD_METHODS=('S','A','V','SS50','SV50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV')
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
CONTRASTS={
 'SELF_TARGET_GAIN':('GT_RETRAIN','SELF_SOFT'),
 'SV_TARGET_GAIN':('GT_RETRAIN','SV_SOFT'),
 'SV_VS_SELF':('SELF_SOFT','SV_SOFT'),
 'HARD_VS_OLD_V':('V','GT_RETRAIN'),
 'HARD_VS_OLD_A':('A','GT_RETRAIN'),
 'GT_VS_OLD_S':('S','GT_RETRAIN'),
 'GT_VS_OLD_SV50':('SV50','GT_RETRAIN'),
 'GT_VS_OLD_SS50':('SS50','GT_RETRAIN'),
 'GT_VS_PERSISTENCE':('PERSISTENCE','GT_RETRAIN'),
}
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_json(path:Path):return json.loads(path.read_text(encoding='utf-8'))
def write_json(path:Path,obj:Any):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def write_csv(path:Path,rows:list[dict[str,Any]]):
 if not rows:raise RuntimeError(f'refusing empty table {path}')
 path.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for row in rows:
  for k in row:
   if k not in fields:fields.append(k)
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def read_csv(path:Path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def prefix_metadata(path:Path):
 fields=('anchor_id','episode_id','task_id','source_cluster','anchor_frame','anchor_native')
 with np.load(path,allow_pickle=False) as z:
  if 'target_native_h30' in z.files:raise RuntimeError('V8 label-free dev prefix contains target field')
  data={k:z[k].copy() for k in fields}
 ids=data['anchor_id'].astype(str).tolist();out={}
 for i,aid in enumerate(ids):
  out[aid]={'episode_id':int(data['episode_id'][i]),'task_id':int(data['task_id'][i]),'source_group':str(data['source_cluster'][i]),
   'anchor_frame':int(data['anchor_frame'][i]),'anchor_native':np.asarray(data['anchor_native'][i],dtype=np.float64)}
 if len(out)!=185 or len(out)!=len(ids):raise RuntimeError('dev prefix must contain 185 unique IDs')
 return ids,out

def ensure_pre_target_gate():
 branch=read_json(ROOT/'protocol/BRANCH_LOCK.json');recipe=read_json(ROOT/'protocol/RECIPE_LOCK.json');seal_path=ROOT/'qa/PREDICTION_SEAL.json'
 if branch.get('branch')!='B_SYNCHRONOUS_VIDEO_9' or branch.get('branch_selected_from_new_dev_outcomes') is not False:raise RuntimeError('V8 branch lock absent/invalid')
 if recipe.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV':raise RuntimeError('V8 recipe lock absent/invalid')
 if not seal_path.is_file():raise RuntimeError('dev target reader refuses missing prediction seal')
 seal=read_json(seal_path)
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('accepted_train_bundles')!=9 or seal.get('accepted_dev_bundles')!=9 or seal.get('missing_bundles')!=0:
  raise RuntimeError('dev target reader refuses incomplete branch-B prediction matrix')
 if seal.get('dev_target_archive_opened') is not False or seal.get('preseal_dev_target_reads')!=0:raise RuntimeError('dev target was accessed before prediction seal')
 if seal.get('branch_lock_sha256')!=sha(ROOT/'protocol/BRANCH_LOCK.json') or seal.get('recipe_lock_sha256')!=sha(ROOT/'protocol/RECIPE_LOCK.json'):raise RuntimeError('prediction seal not bound to V8 branch/recipe')
 synth=read_json(ROOT/'qa/SYNTHETIC_SCORER_QA.json');train_qa=read_json(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json');input_qa=read_json(ROOT/'qa/SCORE_INPUT_PREFLIGHT.json')
 if synth.get('status')!='PASS' or synth.get('dev_target_archive_opened') is not False:raise RuntimeError('synthetic scorer QA absent/invalid before dev target reader')
 if train_qa.get('status')!='PASS' or train_qa.get('dev_target_archive_opened') is not False:raise RuntimeError('train-only summary preflight absent/invalid before dev target reader')
 if input_qa.get('status')!='PASS' or input_qa.get('dev_target_archive_opened') is not False:raise RuntimeError('sealed prediction/reference input preflight absent/invalid')
 return seal

def npy_header(z:zipfile.ZipFile,name:str):
 with z.open(name,'r') as f:
  version=np.lib.format.read_magic(f)
  if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
  elif version in {(2,0),(3,0)}:shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
  else:raise RuntimeError(f'unsupported NPY header {name}: {version}')
 return {'shape':list(shape),'fortran_order':bool(fortran),'dtype':str(dtype)}

def metadata_only_target_preflight(seal):
 path=V2/'data/DEV_TARGETS_SCORER_ONLY.npz';v5=read_json(V5/'inventory/OUTCOME_EXPOSURE.json')
 expected_sha=v5['target_archive_sha256'];expected_bytes=int(v5['target_archive_bytes'])
 if not path.is_file() or path.stat().st_size!=expected_bytes:raise RuntimeError('scorer-only dev archive bytes differ from accepted V5 exposure')
 required=('anchor_id.npy','episode_id.npy','task_id.npy','source_cluster.npy','target_native_h30.npy')
 with zipfile.ZipFile(path,'r') as z:
  names=set(z.namelist())
  if not set(required).issubset(names):raise RuntimeError(f'dev archive schema lacks {sorted(set(required)-names)}')
  headers={x[:-4]:npy_header(z,x) for x in required}
 if headers['target_native_h30']['shape']!=[185,30,28]:raise RuntimeError('dev target header differs from H30x28 contract')
 for k in ('anchor_id','episode_id','task_id','source_cluster'):
  if headers[k]['shape']!=[185]:raise RuntimeError(f'dev archive metadata header invalid: {k}')
 result={'schema_version':1,'status':'PASS','created_utc':utc(),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
  'archive_path':str(path),'expected_archive_sha256_from_V5':expected_sha,'archive_bytes':expected_bytes,
  'metadata_only_zip_opens':1,'target_numpy_content_reads':0,'headers':headers,'preseal_dev_target_reads':0,
  'final_heldout_payload_access':0,'sealed_train_bundles':9,'sealed_dev_bundles':9}
 write_json(ROOT/'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json',result);return result

def read_prediction(path:Path,rows_path:Path,meta,ids):
 tensor=load_file(str(path),device='cpu')
 if 'prediction_native' not in tensor:raise RuntimeError(f'prediction_native missing: {path}')
 values=tensor['prediction_native'].numpy()
 if values.shape!=(185,30,28) or values.dtype!=np.float64 or not np.isfinite(values).all():raise RuntimeError(f'dev prediction schema invalid: {path}')
 with rows_path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 by={str(x['anchor_id']):x for x in rows}
 if len(rows)!=185 or len(by)!=185 or set(by)!=set(ids):raise RuntimeError(f'dev prediction IDs incomplete: {path}')
 order=[]
 for aid in ids:
  row=by[aid];m=meta[aid]
  if (int(row['episode_id']),int(row['task_id']),str(row['source_cluster']),int(row['anchor_frame']))!=(m['episode_id'],m['task_id'],m['source_group'],m['anchor_frame']):raise RuntimeError(f'dev prediction metadata mismatch {aid}')
  order.append(int(row['row']))
 if not np.array_equal(np.sort(order),np.arange(185)):raise RuntimeError('dev prediction row map is not one-to-one')
 return values[np.asarray(order,dtype=np.int64)]

def score_prediction(pred,target,anchor,sigma,fdscale):
 pos_err=np.abs(pred-target)/sigma[None,None,:]
 pos=.5*(pos_err[...,:14].mean(axis=(1,2))+pos_err[...,14:].mean(axis=(1,2)))
 fd_err=np.abs(np.diff(pred,axis=1)/fdscale[None,None,:]-np.diff(target,axis=1)/fdscale[None,None,:])
 fd=.5*(fd_err[...,:14].mean(axis=(1,2))+fd_err[...,14:].mean(axis=(1,2)))
 seam_err=np.abs((pred[:,0]-anchor)/fdscale[None,:]-(target[:,0]-anchor)/fdscale[None,:])
 seam=.5*(seam_err[:,:14].mean(axis=1)+seam_err[:,14:].mean(axis=1))
 return {'position_mae':pos,'safe_fd_l1':fd,'seam_safe_fd_l1':seam}

def load_new_prediction_bundles(ids,meta):
 rows=[];arrays={};receipts=[]
 for seed in SEEDS:
  for program in PROGRAMS:
   out=ROOT/'predictions/dev'/program/f'seed{seed}'/'attempt_01';rec=read_json(out/'PREDICTION_COMPLETE.json')
   manifest=read_json(ROOT/'model_only'/program/f'seed{seed}'/'attempt_01/MODEL_MANIFEST.json')
   if rec.get('status')!='PASS' or int(rec.get('seed',-1))!=seed or rec.get('program_id')!=program or rec.get('dataset_role')!='dev':raise RuntimeError(f'dev prediction receipt mismatch {program}/{seed}')
   if rec.get('model_sha256')!=manifest.get('model_file_sha256') or rec.get('model_state_hash')!=manifest.get('final_state_hash'):raise RuntimeError(f'dev prediction model provenance mismatch {program}/{seed}')
   pp=Path(rec['prediction_path']);rp=Path(rec['prediction_rows_path'])
   if sha(pp)!=rec['prediction_sha256'] or sha(rp)!=rec['prediction_rows_sha256']:raise RuntimeError(f'dev prediction checksum mismatch {program}/{seed}')
   pred=read_prediction(pp,rp,meta,ids);arrays[(program,seed)]=pred
   receipts.append({'program_id':program,'seed':seed,'prediction_sha256':rec['prediction_sha256'],'model_sha256':rec['model_sha256'],'model_state_hash':rec['model_state_hash']})
 return arrays,receipts

def load_v7_historical_per_window(ids,meta):
 path=V7/'results/PER_WINDOW_ALL_METHODS.csv';source=read_csv(path);keep=[]
 for row in source:
  method=row['method']
  if method not in OLD_METHODS:continue
  aid=str(row['anchor_id'])
  if aid not in meta:raise RuntimeError(f'historical V7 per-window row outside locked dev support: {aid}')
  m=meta[aid]
  if (int(row['episode_id']),int(row['task_id']),str(row['source_group']))!=(m['episode_id'],m['task_id'],m['source_group']):raise RuntimeError(f'historical V7 source row metadata mismatch {aid}')
  seed=int(row['seed'])
  if seed not in SEEDS:raise RuntimeError(f'historical V7 seed invalid: {row["seed"]}')
  vals={metric:float(row[metric]) for metric in METRICS}
  if not np.isfinite(list(vals.values())).all():raise RuntimeError(f'historical V7 metric nonfinite: {method}/{aid}')
  keep.append({'method':method,'seed':seed,'anchor_id':aid,'episode_id':m['episode_id'],'task_id':m['task_id'],'source_group':m['source_group'],**vals,
   'source':'accepted_V7_per_window_reuse','program_id':row.get('program_id','')})
 expected=185*len(SEEDS)*len(OLD_METHODS)
 if len(keep)!=expected:raise RuntimeError(f'historical V7 known-grain rows expected {expected}, got {len(keep)}')
 counts={}
 for x in keep:counts[(x['method'],x['seed'])]=counts.get((x['method'],x['seed']),0)+1
 if any(n!=185 for n in counts.values()):raise RuntimeError('historical V7 method/seed per-window support differs from 185')
 return keep,sha(path)

def aggregate(rows):
 vals={}
 for r in rows:
  key=(r['method'],int(r['seed']),int(r['task_id']),int(r['episode_id']))
  vals.setdefault(key,{m:[] for m in METRICS})
  for metric in METRICS:vals[key][metric].append(float(r[metric]))
 episode={k:{m:float(np.mean(v[m])) for m in METRICS} for k,v in vals.items()}
 result=[];task_seed={};methods=sorted({k[0] for k in episode})
 for method in methods:
  for seed in SEEDS:
   for task in TASKS:
    eps=[k for k in episode if k[:3]==(method,seed,task)]
    if not eps:continue
    row={'method':method,'seed':seed,'task_id':task,**{m:float(np.mean([episode[k][m] for k in eps])) for m in METRICS},
      'episodes':len(eps),'anchors':sum(len(vals[k]['position_mae']) for k in eps)}
    task_seed[(method,seed,task)]=row;result.append(row)
   taskrows=[task_seed[(method,seed,t)] for t in TASKS if (method,seed,t) in task_seed]
   if len(taskrows)==5:result.append({'method':method,'seed':seed,'task_id':'macro',**{m:float(np.mean([x[m] for x in taskrows])) for m in METRICS},'episodes':'task-equal','anchors':185})
 return result,task_seed,episode,vals

def paired_tables(rows,task_seed,episode_values,window_values,meta):
 paired_task_seed=[];effects=[];seed_loo=[];task_loo=[];episode_delta={}
 by_anchor={(r['method'],int(r['seed']),str(r['anchor_id'])):r for r in rows}
 for name,(reference,candidate) in CONTRASTS.items():
  for metric in METRICS:
   for seed in SEEDS:
    for task in TASKS:
     rk=(reference,seed,task);ck=(candidate,seed,task)
     if rk not in task_seed or ck not in task_seed:raise RuntimeError(f'paired support missing: {name}/{seed}/{task}')
     gain=float(task_seed[rk][metric]-task_seed[ck][metric])
     task_ids=[aid for aid,m in meta.items() if m['task_id']==task]
     paired_task_seed.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'seed':seed,'task_id':task,
       'gain_reference_minus_candidate':gain,'episodes':task_seed[rk]['episodes'],'anchors':len(task_ids)})
     ep_group={}
     for aid in task_ids:
      rr=float(by_anchor[(reference,seed,aid)][metric]);cc=float(by_anchor[(candidate,seed,aid)][metric])
      ep=int(meta[aid]['episode_id']);group=str(meta[aid]['source_group']);ep_group.setdefault((group,ep),[]).append(rr-cc)
     for (group,ep),deltas in ep_group.items():episode_delta.setdefault((name,metric,seed,task,group),{})[ep]=float(np.mean(deltas))
   # Seed and task macro summaries are computed only after all 3x5 paired rows exist.
   seed_values=[]
   for seed in SEEDS:
    vals=[x['gain_reference_minus_candidate'] for x in paired_task_seed if x['contrast']==name and x['metric']==metric and x['seed']==seed]
    value=float(np.mean(vals));seed_values.append(value)
    effects.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_seed_macro','seed':seed,'task_id':'macro','gain_reference_minus_candidate':value})
   effects.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_three_seed_macro','seed':'macro','task_id':'macro','gain_reference_minus_candidate':float(np.mean(seed_values))})
   for task in TASKS:
    vals=[x['gain_reference_minus_candidate'] for x in paired_task_seed if x['contrast']==name and x['metric']==metric and x['task_id']==task]
    effects.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'three_seed_task_macro','seed':'macro','task_id':task,'gain_reference_minus_candidate':float(np.mean(vals))})
   for omit in SEEDS:
    remain=[seed for seed in SEEDS if seed!=omit]
    vals=[x['gain_reference_minus_candidate'] for x in paired_task_seed if x['contrast']==name and x['metric']==metric and x['seed'] in remain]
    seed_loo.append({'analysis':'paired_contrast','contrast':name,'metric':metric,'omitted_seed':omit,'remaining_seeds':','.join(map(str,remain)),
      'n_remaining_seeds':2,'loo_five_task_macro':float(np.mean(vals))})
   for omit in TASKS:
    remain=[task for task in TASKS if task!=omit]
    vals=[x['gain_reference_minus_candidate'] for x in paired_task_seed if x['contrast']==name and x['metric']==metric and x['task_id'] in remain]
    task_loo.append({'analysis':'paired_contrast','contrast':name,'metric':metric,'omitted_task':omit,'remaining_tasks':','.join(map(str,remain)),
      'n_remaining_tasks':4,'seeds_included':3,'loo_macro':float(np.mean(vals))})
 # Draw common source-group bootstrap indices once, then apply them to every paired contrast and seed.
 rng=np.random.default_rng(20261016);draws={};digest=hashlib.sha256();support={}
 groups_by_task={task:sorted({str(m['source_group']) for m in meta.values() if m['task_id']==task}) for task in TASKS}
 for task in TASKS:
  groups=groups_by_task[task]
  group_eps={group:sorted({int(m['episode_id']) for m in meta.values() if m['task_id']==task and str(m['source_group'])==group}) for group in groups}
  probs=np.asarray([len(group_eps[group]) for group in groups],dtype=np.float64);probs/=probs.sum()
  arr=np.empty((5000,len(groups)),dtype=np.int32)
  for b in range(5000):arr[b]=rng.choice(len(groups),size=len(groups),replace=True,p=probs)
  draws[task]=(groups,group_eps,arr);digest.update(arr.tobytes(order='C'))
  support[str(task)]={'groups':len(groups),'episodes':int(sum(len(v) for v in group_eps.values()))}
 ci=[]
 for name,(reference,candidate) in CONTRASTS.items():
  for metric in METRICS:
   boot=np.empty(5000,dtype=np.float64)
   for b in range(5000):
    task_effects=[]
    for task in TASKS:
     groups,group_eps,arr=draws[task];selected=[]
     for gi in arr[b]:
      group=groups[int(gi)]
      for episode in group_eps[group]:
       seed_values=[]
       for seed in SEEDS:
        by_episode=episode_delta.get((name,metric,seed,task,group),{})
        if episode in by_episode:seed_values.append(by_episode[episode])
       if seed_values:selected.append(float(np.mean(seed_values)))
     task_effects.append(float(np.mean(selected)) if selected else float('nan'))
    boot[b]=float(np.mean(task_effects))
   point=next(x['gain_reference_minus_candidate'] for x in effects if x['contrast']==name and x['metric']==metric and x['grain']=='five_task_three_seed_macro')
   ci.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_three_seed_macro','estimate':point,
    'ci_2p5':float(np.percentile(boot,2.5)),'ci_97p5':float(np.percentile(boot,97.5)),'bootstrap_draws':5000,'bootstrap_seed':20261016,
    'source_group_resampling':'within task; sample groups proportional to contained episodes; retain all anchors; common draw indices across seeds/programs'})
 manifest={'schema_version':1,'status':'PASS','seed':20261016,'draws':5000,'group_support_by_task':support,'shared_draw_index_sha256':digest.hexdigest(),
  'sampling':'source groups sampled with replacement within task, probability proportional to contained episode count; all episodes/anchors in selected group retained',
  'common_draws_across_programs_and_seeds':True,'final_heldout_payload_access':0}
 return paired_task_seed,effects,seed_loo,task_loo,ci,manifest

def prepare_trajectory(new_predictions,target,ids,meta):
 old=read_csv(V7/'figure_data/TRAJECTORY_DATA.csv');cases={}
 for r in old:
  key=(str(r['task_id']),str(r['episode_id']),str(r['anchor_id']),str(r['source_group']))
  cases.setdefault(key,{'task_id':int(r['task_id']),'episode_id':int(r['episode_id']),'anchor_id':str(r['anchor_id']),'source_group':str(r['source_group']),
   'case_selection':str(r['case_selection']),'reused_v5_case_id':str(r['reused_v5_case_id'])})
 if len(cases)!=5:raise RuntimeError(f'V7 metadata-selected case count differs from expected five: {len(cases)}')
 index={aid:i for i,aid in enumerate(ids)};case_rows=[];traj=[]
 for key,case in sorted(cases.items()):
  aid=case['anchor_id']
  if aid not in index or aid not in meta:raise RuntimeError(f'V7 selected case missing from current dev prefix: {aid}')
  case_rows.append(case);j=index[aid]
  # target rows are loaded once and retained in memory for this figure export.
  for ch in (0,14):
   for step in range(30):traj.append({'task_id':case['task_id'],'episode_id':case['episode_id'],'anchor_id':aid,'source_group':case['source_group'],'seed':'shared','action_index':ch,'future_step':step+1,'method':'Y_TARGET','native_command_value':float(target[j,step,ch]),'case_selection':case['case_selection'],'unit_note':'numeric native command; physical units unresolved'})
   for (program,seed),pred in new_predictions.items():
    for step in range(30):traj.append({'task_id':case['task_id'],'episode_id':case['episode_id'],'anchor_id':aid,'source_group':case['source_group'],'seed':seed,'action_index':ch,'future_step':step+1,'method':program,'native_command_value':float(pred[j,step,ch]),'case_selection':case['case_selection'],'unit_note':'numeric native command; physical units unresolved'})
   old_v=[x for x in old if x['anchor_id']==aid and int(x['action_index'])==ch and x['method']=='V']
   for x in old_v:traj.append({'task_id':case['task_id'],'episode_id':case['episode_id'],'anchor_id':aid,'source_group':case['source_group'],'seed':int(x['seed']),'action_index':ch,'future_step':int(x['future_step']),'method':'V_OLD','native_command_value':float(x['native_command_value']),'case_selection':case['case_selection'],'unit_note':'numeric native command; physical units unresolved'})
 return case_rows,traj

def main():
 seal=ensure_pre_target_gate()
 train_qa=read_json(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json');resource_rows=read_csv(ROOT/'results/RESOURCE_LEDGER.csv')
 if len(resource_rows)!=9 or any(r.get('status')!='PASS' for r in resource_rows):raise RuntimeError('accepted V8 fit/resource ledger incomplete before dev target reader')
 meta_path=Path(read_json(ROOT/'inventory/RESOLVED_PATHS.json')['dev_prefix_npz']);ids,meta=prefix_metadata(meta_path)
 # All prediction payloads and historical references are checked before opening new dev targets.
 predictions,receipts=load_new_prediction_bundles(ids,meta)
 hist,hist_sha=load_v7_historical_per_window(ids,meta)
 schema=read_json(ROOT/'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json')
 if schema.get('status')!='PASS' or schema.get('target_numpy_content_reads')!=0 or schema.get('prediction_seal_sha256')!=sha(ROOT/'qa/PREDICTION_SEAL.json'):raise RuntimeError('dev target schema preflight does not match current seal')
 # Exactly one read_bytes + one np.load from in-memory bytes for all new models/figures.
 target_path=V2/'data/DEV_TARGETS_SCORER_ONLY.npz';v5=read_json(V5/'inventory/OUTCOME_EXPOSURE.json');blob=target_path.read_bytes();target_sha=hashlib.sha256(blob).hexdigest()
 if len(blob)!=int(v5['target_archive_bytes']) or target_sha!=v5['target_archive_sha256']:raise RuntimeError('single dev target archive identity differs from accepted exposure')
 with np.load(io.BytesIO(blob),allow_pickle=False) as z:
  required={'anchor_id','episode_id','task_id','source_cluster','target_native_h30'}
  if not required.issubset(z.files):raise RuntimeError(f'dev target members missing {sorted(required-set(z.files))}')
  t_ids=z['anchor_id'].astype(str).tolist();eps=z['episode_id'].copy();tasks=z['task_id'].copy();groups=z['source_cluster'].astype(str);target=np.asarray(z['target_native_h30'],dtype=np.float64).copy()
 if len(t_ids)!=185 or len(set(t_ids))!=185 or set(t_ids)!=set(ids):raise RuntimeError('dev target IDs differ from sealed prefixes')
 tidx={aid:i for i,aid in enumerate(t_ids)};target=np.stack([target[tidx[aid]] for aid in ids]);
 for aid in ids:
  i=tidx[aid];m=meta[aid]
  if (int(eps[i]),int(tasks[i]),str(groups[i]))!=(m['episode_id'],m['task_id'],m['source_group']):raise RuntimeError(f'dev target identity/task/source-group mismatch {aid}')
 if target.shape!=(185,30,28) or not np.isfinite(target).all():raise RuntimeError('dev target shape/finite check failed')
 anchors=np.stack([meta[a]['anchor_native'] for a in ids]);metric=json.loads((ROOT/'protocol/METRIC_SCALE_LOCK.json').read_text());sigma=np.asarray(metric['target_sigma'],dtype=np.float64);fdscale=np.asarray(metric['safe_fd_scale'],dtype=np.float64)
 if sigma.shape!=(28,) or fdscale.shape!=(28,) or np.any(sigma<=0) or np.any(fdscale<=0):raise RuntimeError('locked metric scale invalid')
 new_rows=[]
 for (program,seed),pred in sorted(predictions.items()):
  comps=score_prediction(pred,target,anchors,sigma,fdscale)
  for i,aid in enumerate(ids):new_rows.append({'method':program,'seed':seed,'anchor_id':aid,'episode_id':meta[aid]['episode_id'],'task_id':meta[aid]['task_id'],'source_group':meta[aid]['source_group'],
   **{m:float(comps[m][i]) for m in METRICS},'source':'V8_new_prediction_single_dev_load','program_id':program})
 if len(new_rows)!=9*185:raise RuntimeError(f'V8 new dev evaluation count mismatch: {len(new_rows)}')
 # Persist the complete new loss table before downstream aggregation/figures for recovery without another target read.
 write_csv(ROOT/'results/PER_WINDOW_NEW_DEV_LOSSES.csv',new_rows)
 all_rows=hist+new_rows
 write_csv(ROOT/'results/PER_WINDOW_ALL_METHODS.csv',all_rows)
 primary,task_seed,episode_values,window_values=aggregate(all_rows)
 write_csv(ROOT/'results/PRIMARY_TASK_SEED.csv',primary)
 method_rows=[]
 for method in sorted({r['method'] for r in primary if r['task_id']=='macro'}):
  macros=[next(r for r in primary if r['method']==method and r['seed']==seed and r['task_id']=='macro') for seed in SEEDS]
  for metric in METRICS:method_rows.append({'method':method,'metric':metric,'seed_202610131':macros[0][metric],'seed_202610132':macros[1][metric],'seed_202610133':macros[2][metric],
    'mean':float(np.mean([x[metric] for x in macros])),'sample_sd':float(np.std([x[metric] for x in macros],ddof=1)),'source':'V8 scored or sealed historical V7 per-window'})
 write_csv(ROOT/'results/METHOD_MACRO.csv',method_rows)
 ptask,effects,seedloo,taskloo,ci,boot=paired_tables(all_rows,task_seed,episode_values,window_values,meta)
 write_csv(ROOT/'results/PAIRED_TASK_SEED.csv',ptask);write_csv(ROOT/'results/PAIRED_EFFECTS.csv',effects);write_csv(ROOT/'results/SEED_LOO.csv',seedloo);write_csv(ROOT/'results/TASK_LOO.csv',taskloo);write_csv(ROOT/'results/PAIRED_GROUP_CI.csv',ci)
 write_json(ROOT/'qa/BOOTSTRAP_DRAW_MANIFEST.json',boot)
 case_rows,traj=prepare_trajectory(predictions,target,ids,meta);write_csv(ROOT/'figure_data/TRAJECTORY_CASES.csv',case_rows);write_csv(ROOT/'figure_data/TRAJECTORY_DATA.csv',traj)
 # Training curves/Y/Z/teacher distances and the 9-fit resource ledger were sealed before this dev-target read.
 if not (ROOT/'figure_data/TRAINING_CURVES.csv').is_file() or not (ROOT/'results/TRAIN_Y_Z_AND_IMITATION.csv').is_file():
  raise RuntimeError('train-only curves/imitation artifacts missing after the single dev target read')
 fit_rows=resource_rows
 scoring={'schema_version':1,'status':'PASS','completed_utc':utc(),'branch':'B_SYNCHRONOUS_VIDEO_9','branch_lock_sha256':sha(ROOT/'protocol/BRANCH_LOCK.json'),
  'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),'train_score_preflight_sha256':sha(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json'),
  'dev_target_archive_path':str(target_path),'dev_target_archive_sha256':target_sha,'dev_target_archive_bytes':len(blob),
  'dev_target_content_reads':1,'numpy_load_calls':1,'dev_rows_materialized':185,'new_student_dev_evaluations':1665,
  'V7_per_window_rows_reused':len(hist),'V7_per_window_source_sha256':hist_sha,'historical_V7_methods_reused':list(OLD_METHODS),
  'new_train_prediction_bundles':9,'new_dev_prediction_bundles':9,'new_train_prediction_rows':9*1280,'new_dev_prediction_rows':9*185,
  'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,'candidate_fits':fit_rows,
  'metric_definition':'Position MAE; 29 future-internal safe-FD L1; seam Safe-FD L1. Window means within episode, episode equal within task, five tasks equal.',
  'bootstrap_seed':20261016,'bootstrap_draws':5000,'p_values':False,'dev_used_for_branch_or_epoch_selection':False}
 write_json(ROOT/'results/V8_SCORING_MANIFEST.json',scoring)
 exposure=read_json(ROOT/'inventory/OUTCOME_EXPOSURE.json');prior=7770;new=1665
 exposure.update({'status':'scored','v8_branch':'B_SYNCHRONOUS_VIDEO_9','v8_formal_fits_started':9,'v8_completed_fits':9,'v8_train_inference_bundles':9,'v8_dev_inference_bundles':9,
  'v8_dev_target_content_reads':1,'v8_dev_rows_loaded':185,'v8_new_dev_student_evaluations':new,'v8_optimizer0_dev_target_reads':0,
  'v8_train_target_presentations':sum(int(x['training_sample_presentations_total']) for x in fit_rows),
  'v5_v6_v7_known_grain_primary_loss_evaluations':prior,'v8_known_grain_primary_loss_evaluations_added':new,'v5_v6_v7_v8_known_grain_primary_loss_evaluations':prior+new,
  'historical_v7_per_window_rows_reused':len(hist),'legacy_35523_count':'unit_unverified_excluded','final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0});write_json(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
 acceptance={'schema_version':1,'status':'PASS','created_utc':utc(),'study':'post_V7_controlled_runner_and_target_replay_development','branch':'B_SYNCHRONOUS_VIDEO_9',
  'branch_selected_from_new_dev_outcomes':False,'branch_lock_sha256':sha(ROOT/'protocol/BRANCH_LOCK.json'),'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),
  'formal_fits':'9/9 epoch160, 3200 optimizer steps each','student_train_bundles':'9/9','student_dev_bundles':'9/9','prediction_seal':'PASS before dev target content read',
  'dev_target_content_reads':1,'dev_rows_loaded':185,'new_dev_student_evaluations':1665,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,
  'metric_tables':'PASS','train_score_preflight':'PASS','paired_effects_and_LOO':'PASS','paired_group_bootstrap':'PASS 5000 draws seed 20261016','figures':'PENDING'}
 write_json(ROOT/'qa/FINAL_ACCEPTANCE.json',acceptance)
 print(json.dumps({'status':'PASS','new_dev_evaluations':1665,'archive_reads':1,'historical_rows_reused':len(hist),'methods_in_primary':len(set(x['method'] for x in all_rows)),'bootstrap_rows':len(ci)},sort_keys=True))
if __name__=='__main__':main()
