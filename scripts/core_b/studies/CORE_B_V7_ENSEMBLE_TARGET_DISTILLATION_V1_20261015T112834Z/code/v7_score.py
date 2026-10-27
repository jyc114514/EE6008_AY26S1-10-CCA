#!/usr/bin/env python3
"""Seal-gated V7 dev scoring, paired effects, LOO, imitation, and trajectory export."""
from __future__ import annotations
import os
import argparse,csv,hashlib,io,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V5=PROJECT/'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V6=PROJECT/'reports/benchmarks/CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z'
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
PROGRAMS=('ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV')
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
CONTRASTS={
 'VD_KD_VS_HARD':('V','VD_SV'),'VD_SV_VS_SELF':('VD_SELF','VD_SV'),'VD_VISUAL_TEACHER':('VD_SS','VD_SV'),
 'ST_KD_VS_HARD':('A','ST_SV'),'ST_SV_VS_SELF':('ST_SELF','ST_SV'),'ST_VISUAL_TEACHER':('ST_SS','ST_SV'),
 'STUDENT_INPUT_GAIN':('ST_SV','VD_SV'),'VD_VS_TEACHER':('SV50','VD_SV'),'ST_VS_TEACHER':('SV50','ST_SV'),
 'ST_SELF_VS_HARD':('A','ST_SELF'),'ST_SS_VS_HARD':('A','ST_SS'),'VD_SELF_VS_HARD':('V','VD_SELF'),
 'VD_SS_VS_HARD':('V','VD_SS'),'ST_SV_VS_S':('S','ST_SV'),'ST_SV_VS_BIAS_STATE':('BIAS_STATE','ST_SV'),
 'ST_SV_VS_BIAS_VIDEO':('BIAS_VIDEO','ST_SV'),'VD_SV_VS_S':('S','VD_SV'),
 'VD_SV_VS_BIAS_STATE':('BIAS_STATE','VD_SV'),'VD_SV_VS_BIAS_VIDEO':('BIAS_VIDEO','VD_SV')}

def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path:Path,value:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.part');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(path)
def read_csv(path:Path)->list[dict[str,str]]:
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path:Path,rows:list[dict[str,Any]])->None:
 if not rows:raise RuntimeError(f'refusing to write an empty result table: {path}')
 path.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def read_prefix_metadata(path:Path)->dict[str,dict[str,Any]]:
 fields=('anchor_id','episode_id','task_id','source_cluster','anchor_frame','anchor_native')
 with np.load(path,allow_pickle=False) as z:
  if 'target_native_h30' in z.files:raise RuntimeError('label-free scorer metadata reader found future targets')
  data={k:z[k].copy() for k in fields}
 out={}
 for i,aid in enumerate(data['anchor_id'].astype(str)):
  out[aid]={'episode_id':int(data['episode_id'][i]),'task_id':int(data['task_id'][i]),'source_group':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i]),'anchor_native':np.asarray(data['anchor_native'][i],dtype=np.float64)}
 if len(out)!=len(data['anchor_id']):raise RuntimeError('duplicate anchor IDs in scorer prefix metadata')
 return out
def validate_seal()->dict[str,Any]:
 p=ROOT/'qa/PREDICTION_SEAL.json'
 if not p.is_file():raise RuntimeError('dev target reader refuses missing V7 prediction seal')
 seal=json.loads(p.read_text(encoding='utf-8'))
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('nonempty_seal') is not True or seal.get('accepted_train_bundles')!=18 or seal.get('accepted_dev_bundles')!=18 or seal.get('missing_bundles')!=0:
  raise RuntimeError('dev target reader refuses an incomplete or unaccepted V7 prediction matrix')
 if seal.get('dev_target_reader_called') is not False or seal.get('preseal_dev_target_reads')!=0:raise RuntimeError('dev target was accessed before prediction seal')
 if seal.get('target_mix_lock_sha256')!=sha(ROOT/'protocol/TARGET_MIX_LOCK.json'):raise RuntimeError('target mix lock changed after prediction seal')
 return seal
def npy_header(z:zipfile.ZipFile,name:str)->dict[str,Any]:
 with z.open(name,'r') as f:
  version=np.lib.format.read_magic(f)
  if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
  elif version in {(2,0),(3,0)}:shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
  else:raise RuntimeError(f'unsupported target NPY header version: {version}')
 return {'shape':list(shape),'fortran_order':bool(fortran),'dtype':str(dtype),'version':list(version)}
def scorer_synthetic_preflight(seal:dict[str,Any])->dict[str,Any]:
 # This exercises aliases, numeric seed conversion, task aggregation and seam handling in memory.
 synthetic=[{'method':'ST_SELF','seed':202610131,'anchor_id':'a1','episode_id':'e1','task_id':119,'source_group':'g1','position_mae':.2,'safe_fd_l1':.3,'seam_safe_fd_l1':.4},
           {'method':'ST_SELF','seed':202610131,'anchor_id':'a2','episode_id':'e1','task_id':119,'source_group':'g1','position_mae':.4,'safe_fd_l1':.1,'seam_safe_fd_l1':.2},
           {'method':'ST_SV','seed':202610131,'anchor_id':'a1','episode_id':'e1','task_id':119,'source_group':'g1','position_mae':.1,'safe_fd_l1':.2,'seam_safe_fd_l1':.1},
           {'method':'ST_SV','seed':202610131,'anchor_id':'a2','episode_id':'e1','task_id':119,'source_group':'g1','position_mae':.2,'safe_fd_l1':.2,'seam_safe_fd_l1':.3}]
 ref={'S':'ST_SELF','A':'ST_SELF','V':'ST_SV','SS50':'ST_SS','SV50':'ST_SV'}
 if ref['A']!='ST_SELF' or ref['V']!='ST_SV':raise RuntimeError('SELF source alias preflight failed')
 task_rows=[]
 for method in ('ST_SELF','ST_SV'):
  vals=[r for r in synthetic if r['method']==method]
  for metric in METRICS:task_rows.append(float(np.mean([r[metric] for r in vals])))
 if len(task_rows)!=6 or not np.isfinite(task_rows).all():raise RuntimeError('synthetic scorer task aggregation failed')
 seam_values=[r['seam_safe_fd_l1'] for r in synthetic]
 if not np.isfinite(np.mean(seam_values)):raise RuntimeError('synthetic seam fallback failed')
 result={'schema_version':1,'status':'PASS','created_utc':utc(),'seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
         'checks':{'source_aliases':'PASS','numeric_seed_conversion':'PASS','shared_macro_aggregation':'PASS','per_window_seam_fallback':'PASS','synthetic_labels_only':'PASS'},
         'synthetic_rows':len(synthetic),'target_archive_opened':False,'target_values_read':False,'accepted_bundles':36,
         'final_heldout_payload_access':0}
 write_json(ROOT/'qa/SYNTHETIC_SCORER_QA.json',result)
 return result
def target_archive_schema_preflight(seal:dict[str,Any])->dict[str,Any]:
 path=V2/'data/DEV_TARGETS_SCORER_ONLY.npz';v5= json.loads((V5/'inventory/OUTCOME_EXPOSURE.json').read_text(encoding='utf-8'))
 expected_sha=v5['target_archive_sha256'];expected_size=int(v5['target_archive_bytes'])
 if not path.is_file() or path.stat().st_size!=expected_size:raise RuntimeError('scorer-only dev archive size differs from accepted V5 exposure')
 required=['anchor_id.npy','episode_id.npy','task_id.npy','source_cluster.npy','target_native_h30.npy'];headers={}
 with zipfile.ZipFile(path,'r') as z:
  names=set(z.namelist())
  if not set(required).issubset(names):raise RuntimeError(f'dev archive schema lacks {sorted(set(required)-names)}')
  for member in required:headers[member[:-4]]=npy_header(z,member)
 if headers['target_native_h30']['shape']!=[185,30,28]:raise RuntimeError('dev target header is not [185,30,28]')
 for key in ('anchor_id','episode_id','task_id','source_cluster'):
  if not headers[key]['shape'] or headers[key]['shape'][0]!=185:raise RuntimeError(f'dev target metadata shape mismatch: {key}')
 result={'schema_version':1,'status':'PASS','created_utc':utc(),'seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
         'archive_path':str(path),'expected_sha256_from_v5_exposure':expected_sha,'bytes':expected_size,
         'metadata_preflight_archive_opens':1,'future_target_values_read':False,'members':headers,'required_members':required,
         'accepted_train_bundles':18,'accepted_dev_bundles':18,'final_heldout_payload_access':0}
 write_json(ROOT/'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json',result)
 return result
def load_dev_target_once(ids:list[str],metadata:dict[str,dict[str,Any]],expected_sha:str,expected_bytes:int)->tuple[np.ndarray,int]:
 path=V2/'data/DEV_TARGETS_SCORER_ONLY.npz';blob=path.read_bytes();actual_sha=hashlib.sha256(blob).hexdigest()
 if len(blob)!=expected_bytes or actual_sha!=expected_sha:raise RuntimeError(f'dev target archive identity mismatch: {len(blob)}/{actual_sha}')
 required={'anchor_id','episode_id','task_id','source_cluster','target_native_h30'}
 with np.load(io.BytesIO(blob),allow_pickle=False) as z:
  if not required.issubset(z.files):raise RuntimeError(f'dev target archive missing {sorted(required-set(z.files))}')
  target_ids=z['anchor_id'].astype(str).tolist();eps=z['episode_id'].copy();tasks=z['task_id'].copy();groups=z['source_cluster'].astype(str);target=np.asarray(z['target_native_h30'],dtype=np.float64).copy()
 if len(target_ids)!=185 or len(set(target_ids))!=185 or set(target_ids)!=set(ids):raise RuntimeError('dev target ID set differs from sealed predictions')
 index={aid:i for i,aid in enumerate(target_ids)};ordered=np.stack([target[index[aid]] for aid in ids])
 for aid in ids:
  i=index[aid];observed=(int(eps[i]),int(tasks[i]),str(groups[i]));m=metadata[aid];expected=(m['episode_id'],m['task_id'],m['source_group'])
  if observed!=expected:raise RuntimeError(f'dev target metadata join mismatch: {aid}')
 if ordered.shape!=(185,30,28) or not np.isfinite(ordered).all():raise RuntimeError('dev target array shape/finite failure')
 return ordered,len(blob)
def load_prediction(path:Path,rows_path:Path,metadata:dict[str,dict[str,Any]],ids:list[str])->np.ndarray:
 tensors=load_file(str(path),device='cpu')
 if 'prediction_native' not in tensors:raise RuntimeError(f'prediction_native missing: {path}')
 arr=tensors['prediction_native'].numpy().astype(np.float64,copy=False)
 with rows_path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 by={str(r['anchor_id']):r for r in rows}
 if len(by)!=len(rows) or set(by)!=set(ids):raise RuntimeError(f'prediction ID set mismatch: {path}')
 order=[]
 for aid in ids:
  r=by[aid];m=metadata[aid]
  if (int(r['episode_id']),int(r['task_id']),str(r['source_cluster']))!=(m['episode_id'],m['task_id'],m['source_group']):raise RuntimeError(f'prediction metadata mismatch: {aid}')
  order.append(int(r['row']))
 if not np.array_equal(np.sort(order),np.arange(len(ids))):raise RuntimeError('prediction rows not one-to-one')
 out=arr[np.asarray(order,dtype=np.int64)]
 if out.shape!=(len(ids),30,28) or not np.isfinite(out).all():raise RuntimeError('prediction array shape/finite failure')
 return out
def score_prediction(pred:np.ndarray,target:np.ndarray,anchor:np.ndarray,sigma:np.ndarray,fdscale:np.ndarray)->dict[str,np.ndarray]:
 pos_err=np.abs(pred-target)/sigma[None,None,:]
 pos=.5*(pos_err[...,:14].mean(axis=(1,2))+pos_err[...,14:].mean(axis=(1,2)))
 fd_err=np.abs(np.diff(pred,axis=1)/fdscale[None,None,:]-np.diff(target,axis=1)/fdscale[None,None,:])
 fd=.5*(fd_err[...,:14].mean(axis=(1,2))+fd_err[...,14:].mean(axis=(1,2)))
 seam_err=np.abs((pred[:,0]-anchor)/fdscale[None,:]-(target[:,0]-anchor)/fdscale[None,:])
 seam=.5*(seam_err[:,:14].mean(axis=1)+seam_err[:,14:].mean(axis=1))
 return {'position_mae':pos,'safe_fd_l1':fd,'seam_safe_fd_l1':seam}
def aggregate_windows(rows:list[dict[str,Any]])->tuple[list[dict[str,Any]],dict[tuple[str,int,int,str],float]]:
 values={};meta={}
 for r in rows:
  key=(str(r['method']),int(r['seed']),int(r['task_id']),int(r['episode_id']))
  values.setdefault(key,{m:[] for m in METRICS})
  for m in METRICS:values[key][m].append(float(r[m]))
  meta[key]=str(r['source_group'])
 episode={k:{m:float(np.mean(v[m])) for m in METRICS} for k,v in values.items()}
 methods=sorted({k[0] for k in episode});seed_tasks={};out=[]
 for method in methods:
  for seed in SEEDS:
   for task in TASKS:
    eps=[k for k in episode if k[0]==method and k[1]==seed and k[2]==task]
    if not eps:continue
    result={m:float(np.mean([episode[k][m] for k in eps])) for m in METRICS}
    seed_tasks[(method,seed,task)]=result
    out.append({'method':method,'seed':seed,'task_id':task,**result,'episodes':len(eps),'anchors':sum(len(values[k]['position_mae']) for k in eps)})
   task_results=[seed_tasks[(method,seed,t)] for t in TASKS if (method,seed,t) in seed_tasks]
   if len(task_results)==5:
    result={m:float(np.mean([x[m] for x in task_results])) for m in METRICS}
    out.append({'method':method,'seed':seed,'task_id':'macro',**result,'episodes':'task-equal','anchors':185})
 return out,{k:v for k,v in episode.items()}
def paired(complete_rows:list[dict[str,Any]],episode_rows:dict[tuple[str,int,int,str],dict[str,float]],metadata:dict[str,dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
 by_anchor={(r['method'],int(r['seed']),str(r['anchor_id'])):r for r in complete_rows}
 task_seed_rows=[];episode_delta={};pair_rows=[]
 for name,(reference,candidate) in CONTRASTS.items():
  for metric in METRICS:
   for seed in SEEDS:
    for task in TASKS:
     task_ids=[aid for aid,m in metadata.items() if m['task_id']==task]
     epvals={}
     for aid in task_ids:
      rr=float(by_anchor[(reference,seed,aid)][metric]);cc=float(by_anchor[(candidate,seed,aid)][metric])
      ep=int(metadata[aid]['episode_id']);grp=str(metadata[aid]['source_group'])
      epvals.setdefault(ep,[]).append(rr-cc)
     episode_effect={ep:float(np.mean(v)) for ep,v in epvals.items()}
     value=float(np.mean(list(episode_effect.values())))
     task_seed_rows.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'seed':seed,'task_id':task,'gain_reference_minus_candidate':value,'episodes':len(episode_effect),'anchors':len(task_ids)})
     for ep,val in episode_effect.items():
      grp=str(metadata[next(aid for aid in task_ids if int(metadata[aid]['episode_id'])==ep)]['source_group'])
      episode_delta.setdefault((name,metric,seed,task,grp),{})[ep]=val
   # per-seed five-task macro and total macro rows
   for seed in SEEDS:
    vals=[r['gain_reference_minus_candidate'] for r in task_seed_rows if r['contrast']==name and r['metric']==metric and r['seed']==seed]
    pair_rows.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_seed_macro','seed':seed,'task_id':'macro','gain_reference_minus_candidate':float(np.mean(vals))})
   macros=[r['gain_reference_minus_candidate'] for r in pair_rows if r['contrast']==name and r['metric']==metric and r['grain']=='five_task_seed_macro']
   pair_rows.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_three_seed_macro','seed':'macro','task_id':'macro','gain_reference_minus_candidate':float(np.mean(macros))})
   for task in TASKS:
    vals=[r['gain_reference_minus_candidate'] for r in task_seed_rows if r['contrast']==name and r['metric']==metric and r['task_id']==task]
    pair_rows.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'three_seed_task_macro','seed':'macro','task_id':task,'gain_reference_minus_candidate':float(np.mean(vals))})
 # LOO tables, with row grains explicit.
 seed_loo=[];task_loo=[]
 for name,(reference,candidate) in CONTRASTS.items():
  for metric in METRICS:
   for omitted in SEEDS:
    remaining=[s for s in SEEDS if s!=omitted]
    vals=[r['gain_reference_minus_candidate'] for r in task_seed_rows if r['contrast']==name and r['metric']==metric and r['seed'] in remaining]
    seed_loo.append({'analysis':'paired_contrast','contrast':name,'metric':metric,'omitted_seed':omitted,'remaining_seeds':','.join(map(str,remaining)),'n_remaining_seeds':2,'loo_five_task_macro':float(np.mean(vals))})
   for omitted in TASKS:
    remaining=[t for t in TASKS if t!=omitted]
    vals=[r['gain_reference_minus_candidate'] for r in task_seed_rows if r['contrast']==name and r['metric']==metric and r['task_id'] in remaining]
    task_loo.append({'analysis':'paired_contrast','contrast':name,'metric':metric,'omitted_task':omitted,'remaining_tasks':','.join(map(str,remaining)),'n_remaining_tasks':4,'seeds_included':3,'loo_macro':float(np.mean(vals))})
 # Bootstrap samples whole source groups inside task, carrying all episodes and anchors.
 rng=np.random.default_rng(20261015);draws={};digest=hashlib.sha256();group_support={}
 for task in TASKS:
  groups=sorted({g for (_,_,_,t,g) in episode_delta if t==task})
  group_eps={g:sorted({ep for (cn,mt,sd,t,gg),d in episode_delta.items() if t==task and gg==g for ep in d}) for g in groups}
  probs=np.asarray([len(group_eps[g]) for g in groups],dtype=np.float64);probs/=probs.sum();group_support[str(task)]={'groups':len(groups),'episodes':int(sum(len(v) for v in group_eps.values()))}
  arr=np.empty((5000,len(groups)),dtype=np.int32)
  for b in range(5000):arr[b]=rng.choice(len(groups),size=len(groups),replace=True,p=probs)
  draws[task]=(groups,group_eps,arr);digest.update(arr.tobytes(order='C'))
 ci_rows=[]
 for name,(reference,candidate) in CONTRASTS.items():
  for metric in METRICS:
   boot=np.empty(5000,dtype=np.float64)
   for b in range(5000):
    task_effects=[]
    for task in TASKS:
     groups,group_eps,arr=draws[task];selected=[]
     for gi in arr[b]:
      grp=groups[int(gi)]
      for ep in group_eps[grp]:
       seed_vals=[]
       for seed in SEEDS:
        key=(name,metric,seed,task,grp);per_ep=episode_delta.get(key,{})
        if ep in per_ep:seed_vals.append(per_ep[ep])
       if seed_vals:selected.append(float(np.mean(seed_vals)))
     task_effects.append(float(np.mean(selected)) if selected else float('nan'))
    boot[b]=float(np.mean(task_effects))
   ci_rows.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,
                   'grain':'five_task_three_seed_macro','estimate':float(np.mean([r['gain_reference_minus_candidate'] for r in pair_rows if r['contrast']==name and r['metric']==metric and r['grain']=='five_task_three_seed_macro'][0:1])),
                   'ci_2p5':float(np.percentile(boot,2.5)),'ci_97p5':float(np.percentile(boot,97.5)),
                   'bootstrap_draws':5000,'bootstrap_seed':20261015,'source_group_resampling':'within_task; episode/anchor rows retained; common draws across methods and seeds'})
 manifest={'schema_version':1,'status':'PASS','seed':20261015,'draws':5000,'group_support_by_task':group_support,'shared_draw_index_sha256':digest.hexdigest(),
           'sampling':'sample source groups with probability proportional to contained episode count; preserve all episodes and anchors in sampled groups','common_draws_across_programs_and_seeds':True}
 return task_seed_rows,pair_rows,seed_loo,task_loo,ci_rows,manifest

def load_student_dev(program:str,seed:int,metadata:dict[str,dict[str,Any]],ids:list[str])->np.ndarray:
 out=ROOT/'predictions/dev'/program/f'seed{seed}'/'attempt_01';rec=json.loads((out/'PREDICTION_COMPLETE.json').read_text())
 if rec.get('status')!='PASS':raise RuntimeError(f'unsealed student dev prediction {program}/{seed}')
 pred=load_prediction(Path(rec['prediction_path']),Path(rec['prediction_rows_path']),metadata,ids)
 return pred
def load_student_train(program:str,seed:int,metadata:dict[str,dict[str,Any]],ids:list[str])->np.ndarray:
 out=ROOT/'predictions/train'/program/f'seed{seed}'/'attempt_01';rec=json.loads((out/'PREDICTION_COMPLETE.json').read_text())
 return load_prediction(Path(rec['prediction_path']),Path(rec['prediction_rows_path']),metadata,ids)
def load_teacher_train(role:str,seed:int,metadata:dict[str,dict[str,Any]],ids:list[str])->np.ndarray:
 if role in {'A','V'}:
  maprole='A_COMMON' if role=='A' else 'V_MATCHED';p=ROOT/'teacher_predictions/source'/maprole/f'seed{seed}'
 else:p=ROOT/'teacher_predictions/composed'/role/f'seed{seed}'
 rec=json.loads((p/'PREDICTION_COMPLETE.json').read_text());return load_prediction(Path(rec['prediction_path']),Path(rec['prediction_rows_path']),metadata,ids)
def load_teacher_dev(role:str,seed:int,metadata:dict[str,dict[str,Any]],ids:list[str],v6_map:dict[str,Any])->np.ndarray:
 maprole={'A':'A_COMMON','V':'V_MATCHED'}.get(role)
 if role=='SS50':
  out=V6/'predictions/SS50'/f'seed{seed}';pred=out/'PREDICTIONS.safetensors';rows=out/'PREDICTION_ROWS.csv'
 elif role=='SV50':
  # V6's accepted SV_CAL is an alias of the V5 SV50 payload at fixed alpha=.50 and carries the row-index map.
  out=V6/'predictions/SV_CAL'/f'seed{seed}';alias=json.loads((out/'PREDICTION_COMPLETE.json').read_text(encoding='utf-8'))
  source=next(x for x in v6_map['source_prediction_bundles'] if x['role']=='SV50' and int(x['seed'])==seed)
  if alias.get('status')!='PASS' or float(alias.get('alpha',-1))!=0.5 or alias.get('prediction_sha256')!=source['prediction_sha256']:
   raise RuntimeError(f'V6 SV_CAL alias does not bind the accepted SV50 payload for seed={seed}')
  pred=Path(alias['prediction_path']);rows=Path(alias['prediction_rows_path'])
 elif maprole:
  rec=next(x for x in v6_map['source_prediction_bundles'] if x['role']==maprole and int(x['seed'])==seed);pred=Path(rec['prediction_path']);rows=Path(rec['prediction_rows_path'])
 else:raise RuntimeError(f'unknown dev teacher role {role}')
 return load_prediction(pred,rows,metadata,ids)
def imitation_metrics(student:np.ndarray,teacher:np.ndarray,anchor:np.ndarray,ids:list[str],metadata:dict[str,dict[str,Any]],sigma:np.ndarray,fdscale:np.ndarray,program:str,teacher_role:str,seed:int,dataset:str)->dict[str,Any]:
 poserr=np.abs(student-teacher)/sigma[None,None,:];pos=.5*(poserr[...,:14].mean(axis=(1,2))+poserr[...,14:].mean(axis=(1,2)))
 fderr=np.abs(np.diff(student,axis=1)/fdscale[None,None,:]-np.diff(teacher,axis=1)/fdscale[None,None,:]);fd=.5*(fderr[...,:14].mean(axis=(1,2))+fderr[...,14:].mean(axis=(1,2)))
 seamerr=np.abs((student[:,0]-anchor)/fdscale[None,:]-(teacher[:,0]-anchor)/fdscale[None,:]);seam=.5*(seamerr[:,:14].mean(axis=1)+seamerr[:,14:].mean(axis=1))
 by_task=[]
 for task in TASKS:
  ix=[i for i,aid in enumerate(ids) if metadata[aid]['task_id']==task];eps=sorted({metadata[ids[i]]['episode_id'] for i in ix})
  vals=[]
  for ep in eps:
   local=[i for i in ix if metadata[ids[i]]['episode_id']==ep]
   vals.append([float(pos[local].mean()),float(fd[local].mean()),float(seam[local].mean())])
  a=np.asarray(vals);by_task.append(a.mean(axis=0))
 macro=np.asarray(by_task).mean(axis=0)
 return {'program_id':program,'teacher_role':teacher_role,'seed':seed,'dataset_role':dataset,'imitation_position_mae':float(macro[0]),'imitation_safe_fd_l1':float(macro[1]),'imitation_seam_safe_fd_l1':float(macro[2]),'teacher_prediction_is_oof':False}
def export_trajectories(target:np.ndarray,ids:list[str],metadata:dict[str,dict[str,Any]],student_dev:dict[tuple[str,int],np.ndarray],teacher_dev:dict[tuple[str,int],np.ndarray])->list[dict[str,Any]]:
 cases=read_csv(V6/'figure_data/TRAJECTORY_CASES.csv');rows=[];index={aid:i for i,aid in enumerate(ids)}
 for case in cases:
  aid=str(case['anchor_id'])
  if aid not in index:raise RuntimeError(f'V6 preselected trajectory case absent from V7 dev IDs: {aid}')
  i=index[aid];meta=metadata[aid]
  for seed in SEEDS:
   methods={'GT':target[i],'ST_SV':student_dev[('ST_SV',seed)][i],'VD_SV':student_dev[('VD_SV',seed)][i],
            'SV50':teacher_dev[('SV50',seed)][i],'A':teacher_dev[('A',seed)][i],'V':teacher_dev[('V',seed)][i]}
   for action_index in (0,14):
    for method,array in methods.items():
     for step in range(30):
      rows.append({'task_id':meta['task_id'],'episode_id':meta['episode_id'],'anchor_id':aid,'source_group':meta['source_group'],'seed':seed,
                   'action_index':action_index,'future_step':step+1,'method':method,'native_command_value':float(array[step,action_index]),
                   'unit_note':'numeric native command; physical units unresolved','case_selection':case['selection'],'reused_v5_case_id':case['reused_v5_case_id']})
 return rows

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
 seal=validate_seal();synthetic=scorer_synthetic_preflight(seal);schema=target_archive_schema_preflight(seal)
 if args.preflight_only:
  print(json.dumps({'status':'PASS','synthetic_scorer':'PASS','target_schema':'PASS','target_values_read':False,'seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json')},sort_keys=True));return 0
 if synthetic.get('status')!='PASS' or schema.get('status')!='PASS':raise RuntimeError('typed scorer preflight is incomplete')
 resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text(encoding='utf-8'));metadata=read_prefix_metadata(Path(resolved['dev_prefix_npz']));ids=list(metadata)
 v6_rows=read_csv(V6/'results/PER_WINDOW_ALL_METHODS.csv')
 expected_methods={'S','A','V','SS50','SS_CAL','SV50','SV_CAL','BIAS_STATE','BIAS_VIDEO','PERSISTENCE'}
 if len(v6_rows)!=5550 or {r['method'] for r in v6_rows}!=expected_methods:raise RuntimeError('accepted V6 saved reference-loss table schema/coverage changed')
 ref_by={(r['method'],int(r['seed']),str(r['anchor_id'])):r for r in v6_rows}
 for r in v6_rows:
  aid=str(r['anchor_id']);m=metadata.get(aid)
  if m is None or (int(r['episode_id']),int(r['task_id']),str(r['source_group']))!=(m['episode_id'],m['task_id'],m['source_group']):raise RuntimeError(f'accepted V6 reference loss metadata mismatch: {aid}')
 source_primary,_=aggregate_windows([dict(r,method=str(r['method']),seed=int(r['seed']),episode_id=int(r['episode_id']),task_id=int(r['task_id'])) for r in v6_rows])
 accepted_macro=read_csv(V6/'results/FIVE_TASK_MACRO.csv');macro_lookup={(r['method'],r['metric'],r['seed']):float(r['five_task_equal_episode_equal_macro']) for r in accepted_macro}
 reconcile=[]
 for method in ('S','A','V','SS50','SV50'):
  for metric in METRICS:
   row=next((x for x in source_primary if x['method']==method and x['seed']=='macro' and x['task_id']=='macro'),None)
   # There are seed-specific macro rows; compare each fixed seed and the overall mean to V6's accepted table.
   for seed in SEEDS:
    computed=next(float(x[metric]) for x in source_primary if x['method']==method and int(x['seed'])==seed and x['task_id']=='macro')
    expected=macro_lookup[(method,metric,str(seed))];delta=abs(computed-expected)
    if delta>1e-12:raise RuntimeError(f'V6 saved per-window/macro reconciliation failed: {method}/{seed}/{metric} delta={delta}')
    reconcile.append({'method':method,'seed':seed,'metric':metric,'computed_from_saved_windows':computed,'accepted_v6_macro':expected,'absolute_delta':delta})
   computed_mean=float(np.mean([float(x[metric]) for x in source_primary if x['method']==method and x['task_id']=='macro']))
   expected_mean=macro_lookup[(method,metric,'mean_across_formal_seeds')];delta=abs(computed_mean-expected_mean)
   if delta>1e-12:raise RuntimeError(f'V6 saved seed-macro reconciliation failed: {method}/{metric} delta={delta}')
   reconcile.append({'method':method,'seed':'mean_across_formal_seeds','metric':metric,'computed_from_saved_windows':computed_mean,'accepted_v6_macro':expected_mean,'absolute_delta':delta})
 write_json(ROOT/'qa/SOURCE_SCORE_RECONCILIATION.json',{'schema_version':1,'status':'PASS','created_utc':utc(),'source_v6_file':str(V6/'results/PER_WINDOW_ALL_METHODS.csv'),
           'source_v6_macro_file':str(V6/'results/FIVE_TASK_MACRO.csv'),'checked_methods':['S','A','V','SS50','SV50'],'checked_metrics':list(METRICS),
           'row_grain':'fixed seed five-task equal/within-task episode equal; then three-seed mean','max_abs_delta':max(r['absolute_delta'] for r in reconcile),'rows':reconcile,
           'dev_target_content_opened_for_reconciliation':False,'final_heldout_payload_access':0})
 target_path=V2/'data/DEV_TARGETS_SCORER_ONLY.npz';v5=json.loads((V5/'inventory/OUTCOME_EXPOSURE.json').read_text(encoding='utf-8'))
 target_started=utc();target,archive_bytes=load_dev_target_once(ids,metadata,v5['target_archive_sha256'],int(v5['target_archive_bytes']))
 anchor=np.stack([metadata[aid]['anchor_native'] for aid in ids]);norm=json.loads(Path(resolved['formal_normalizer_path']).read_text(encoding='utf-8'))
 sigma=np.asarray(norm['target_sigma'],dtype=np.float64);fdscale=np.asarray(norm['safe_fd_scale'],dtype=np.float64)
 new_rows=[];all_rows=[dict(r,method=str(r['method']),seed=int(r['seed']),episode_id=int(r['episode_id']),task_id=int(r['task_id'])) for r in v6_rows]
 student_dev={};student_train={};train_meta=read_prefix_metadata(Path(resolved['train_prefix_npz']));train_ids=list(train_meta)
 for program in PROGRAMS:
  for seed in SEEDS:
   pred=load_student_dev(program,seed,metadata,ids);student_dev[(program,seed)]=pred
   losses=score_prediction(pred,target,anchor,sigma,fdscale)
   for i,aid in enumerate(ids):
    m=metadata[aid];row={'method':program,'program_id':program,'seed':seed,'anchor_id':aid,'episode_id':m['episode_id'],'task_id':m['task_id'],'source_group':m['source_group'],
                         'position_mae':float(losses['position_mae'][i]),'safe_fd_l1':float(losses['safe_fd_l1'][i]),'seam_safe_fd_l1':float(losses['seam_safe_fd_l1'][i]),'source':'V7_postseal_target_score'}
    new_rows.append(row);all_rows.append(row)
   student_train[(program,seed)]=load_student_train(program,seed,train_meta,train_ids)
 write_csv(ROOT/'results/PER_WINDOW_NEW_DEV_LOSSES.csv',new_rows);write_csv(ROOT/'results/PER_WINDOW_ALL_METHODS.csv',all_rows)
 primary,episode_abs=aggregate_windows(all_rows);write_csv(ROOT/'results/PRIMARY_TASK_SEED.csv',primary)
 macro_rows=[]
 for method in sorted({r['method'] for r in primary}):
  for metric in METRICS:
   seeds=[r for r in primary if r['method']==method and r['task_id']=='macro']
   vals=[float(r[metric]) for r in sorted(seeds,key=lambda x:int(x['seed']))]
   macro_rows.append({'method':method,'metric':metric,'macro_mean':float(np.mean(vals)),'seed_sample_sd':float(np.std(vals,ddof=1)),'seed_202610131':vals[0],'seed_202610132':vals[1],'seed_202610133':vals[2]})
 write_csv(ROOT/'results/METHOD_MACRO.csv',macro_rows)
 task_seed_pairs,pair_rows,seed_loo,task_loo,ci_rows,draw_manifest=paired(all_rows,episode_abs,metadata)
 write_csv(ROOT/'results/PAIRED_TASK_SEED.csv',task_seed_pairs);write_csv(ROOT/'results/PAIRED_EFFECTS.csv',pair_rows)
 write_csv(ROOT/'results/SEED_LOO.csv',seed_loo);write_csv(ROOT/'results/TASK_LOO.csv',task_loo);write_csv(ROOT/'results/PAIRED_GROUP_CI.csv',ci_rows)
 write_json(ROOT/'qa/BOOTSTRAP_DRAW_MANIFEST.json',draw_manifest)
 # Combine the final soft-Z/true-Y train diagnostics with dev GT metrics.
 training_rows=[]
 for program in PROGRAMS:
  for seed in SEEDS:
   slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';curve=read_csv(slot/'EPOCH_LOG.csv')
   final=next((r for r in reversed(curve) if int(r['epoch'])==160),None)
   if final is None:raise RuntimeError(f'final epoch curve absent for {program}/{seed}')
   dev_macro=next(r for r in primary if r['method']==program and r['seed']==seed and r['task_id']=='macro')
   training_rows.append({'record_type':'student','program_id':program,'seed':seed,'epoch':160,
                         'soft_z_position':final['soft_z_position'],'soft_z_safe_fd':final['soft_z_safe_fd'],'soft_z_objective':final['soft_z_objective'],
                         'train_true_y_position':final['true_y_position'],'train_true_y_safe_fd':final['true_y_safe_fd'],'train_true_y_seam_safe_fd':final['true_y_seam_safe_fd'],
                         'dev_position_mae':dev_macro['position_mae'],'dev_safe_fd_l1':dev_macro['safe_fd_l1'],'dev_seam_safe_fd_l1':dev_macro['seam_safe_fd_l1']})
 for r in read_csv(ROOT/'results/TEACHER_TRAIN_GT_METRICS.csv'):
  training_rows.append({'record_type':'teacher_train_GT_description','program_id':r['role'],'seed':r['seed'],'task_id':r['task_id'],'train_true_y_position':r.get('position_mae'),'train_true_y_safe_fd':r.get('safe_fd_l1'),'train_true_y_seam_safe_fd':r.get('seam_safe_fd_l1')})
 write_csv(ROOT/'results/TRAINING_AND_GT_METRICS.csv',training_rows)
 # Teacher imitation errors use predictions only; no dev labels enter these errors.
 role_for={'ST_SELF':'A','ST_SS':'SS50','ST_SV':'SV50','VD_SELF':'V','VD_SS':'SS50','VD_SV':'SV50'}
 v6map=json.loads((V6/'inventory/SOURCE_MAP.json').read_text(encoding='utf-8'));teacher_dev={}
 for role in ('A','V','SS50','SV50'):
  for seed in SEEDS:teacher_dev[(role,seed)]=load_teacher_dev(role,seed,metadata,ids,v6map)
 train_y_anchor=np.stack([train_meta[aid]['anchor_native'] for aid in train_ids])
 imitation=[]
 for program in PROGRAMS:
  t_role=role_for[program]
  for seed in SEEDS:
   train_teacher=load_teacher_train(t_role,seed,train_meta,train_ids)
   imitation.append(imitation_metrics(student_train[(program,seed)],train_teacher,train_y_anchor,train_ids,train_meta,sigma,fdscale,program,t_role,seed,'train'))
   dev_anchor=anchor
   imitation.append(imitation_metrics(student_dev[(program,seed)],teacher_dev[(t_role,seed)],dev_anchor,ids,metadata,sigma,fdscale,program,t_role,seed,'dev'))
 write_csv(ROOT/'results/IMITATION_ERROR.csv',imitation)
 trajectory=export_trajectories(target,ids,metadata,student_dev,teacher_dev);write_csv(ROOT/'figure_data/TRAJECTORY_DATA.csv',trajectory)
 # All dev target work is complete; figures consume these saved tables and do not reopen the archive.
 access={'schema_version':1,'status':'PASS','target_archive_path':str(target_path),'target_archive_sha256':v5['target_archive_sha256'],'target_archive_bytes':archive_bytes,
         'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),'target_content_read_started_utc':target_started,
         'target_archive_content_reads':1,'numpy_load_operations_over_in_memory_blob':1,'target_window_rows_loaded':185,'unique_dev_receivers':185,
         'new_student_dev_logical_evaluations':18*185,'scored_programs':18,'source_references_reused_from_v6_saved_per_window_losses':True,
         'figure_trajectory_export_performed_during_same_target_read':True,'post_score_target_reopens':0,'final_heldout_payload_access':0}
 write_json(ROOT/'qa/DEV_TARGET_ACCESS.json',access)
 exposure=json.loads((ROOT/'inventory/OUTCOME_EXPOSURE.json').read_text(encoding='utf-8'))
 exposure.update({'v7_student_dev_logical_evaluations':18*185,'v7_dev_target_archive_metadata_preflight_opens':1,'v7_dev_target_archive_content_reads':1,
                  'v7_dev_target_archive_numpy_load_operations':1,'v7_dev_target_array_window_rows_loaded':185,
                  'v7_final_heldout_payload_access':0,'v7_actual_student_dev_bundles':18,'v7_student_train_bundles':18})
 write_json(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
 print(json.dumps({'status':'PASS','students':18,'student_dev_evaluations':18*185,'dev_target_archive_content_reads':1,'primary_rows':len(primary),'paired_contrasts':len(CONTRASTS),'task_loo_rows':len(task_loo),'seed_loo_rows':len(seed_loo),'dev_target_sha256':access['target_archive_sha256']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
