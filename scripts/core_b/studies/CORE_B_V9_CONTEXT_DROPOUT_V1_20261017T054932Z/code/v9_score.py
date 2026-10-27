#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse,csv,hashlib,io,json,os,zipfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V5=PROJECT/'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
PROGRAMS=('A0','AD','V0','VD');PRIMARY=('A0','AD','V0','VD','MIX0','MIXD')
HIST=('S','A','V','SV50','SS50','PERSISTENCE')
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
CONTRASTS={
 'NUMCTX_DROP_VIDEO':('V0','VD'),
 'NUMCTX_DROP_STATE':('A0','AD'),
 'VIDEO_GAIN_0':('A0','V0'),
 'VIDEO_GAIN_D':('AD','VD'),
 'INTERACTION':('VIDEO_GAIN_D','VIDEO_GAIN_0'),
 'MIX_GAIN':('MIX0','MIXD'),
 'VD_VS_MIX0':('MIX0','VD'),
 'VD_VS_S':('S','VD'),
 'VD_VS_PERSISTENCE':('PERSISTENCE','VD'),
 'MIXD_VS_S':('S','MIXD'),
 'MIXD_VS_V0':('V0','MIXD'),
 'MIXD_VS_OLD_SV50':('SV50','MIXD'),
 'V0_VS_OLD_V':('V','V0'),
 'A0_VS_OLD_A':('A','A0'),
}
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def wj(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def read_csv(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:raise RuntimeError('refusing empty CSV: '+str(path))
 if fields is None:
  fields=[]
  for row in rows:
   for k in row:
    if k not in fields:fields.append(k)
 tmp=path.with_name('.'+path.name+'.part')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def gate_sealed():
 p=ROOT/'qa/PREDICTION_SEAL.json'
 if not p.is_file():raise RuntimeError('scorer refuses absent V9 prediction seal')
 seal=rj(p)
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('dev_target_archive_opened') is not False or seal.get('preseal_dev_target_reads')!=0:
  raise RuntimeError('scorer refuses target read before complete prediction seal')
 if seal.get('accepted_train_bundles')!=12 or seal.get('accepted_dev_bundles')!=30 or seal.get('missing_bundles')!=0:
  raise RuntimeError('scorer refuses incomplete 12/30/0 prediction matrix')
 if seal.get('code_lock_sha256')!=sha(ROOT/'protocol/CODE_LOCK.json'):raise RuntimeError('prediction seal/code lock mismatch')
 lock_path=ROOT/'protocol/SCORING_CODE_LOCK.json'
 if not lock_path.is_file():raise RuntimeError('scoring code lock missing before score execution')
 code_lock=rj(lock_path)
 if code_lock.get('status')!='LOCKED_BEFORE_DEV_TARGET_ACCESS' or code_lock.get('files',{}).get('code/v9_score.py',{}).get('sha256')!=sha(ROOT/'code/v9_score.py'):
  raise RuntimeError('scorer code hash differs from pre-target lock')
 return seal
def load_prefix(path,expected):
 fields=('anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame','anchor_native')
 with np.load(path,allow_pickle=False) as z:
  if 'target_native_h30' in z.files:raise RuntimeError('scorer prefix contains future target')
  data={k:z[k].copy() for k in fields}
 ids=data['anchor_id'].astype(str).tolist()
 if len(ids)!=expected or len(set(ids))!=expected:raise RuntimeError('prefix ordered IDs/support invalid')
 meta={}
 for i,aid in enumerate(ids):
  meta[aid]={'episode_id':int(data['episode_id'][i]),'task_id':int(data['task_id'][i]),
   'source_group':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i]),
   'anchor_native':np.asarray(data['anchor_native'][i],dtype=np.float64)}
 return ids,meta
def score_prediction(pred,target,anchor,sigma,fdscale):
 if pred.shape!=target.shape or pred.shape!=(len(anchor),30,28):raise RuntimeError('prediction/target/anchor shape mismatch')
 pos_err=np.abs(pred-target)/sigma[None,None,:]
 pos=.5*(pos_err[...,:14].mean(axis=(1,2))+pos_err[...,14:].mean(axis=(1,2)))
 fd_err=np.abs(np.diff(pred,axis=1)/fdscale[None,None,:]-np.diff(target,axis=1)/fdscale[None,None,:])
 fd=.5*(fd_err[...,:14].mean(axis=(1,2))+fd_err[...,14:].mean(axis=(1,2)))
 seam_err=np.abs((pred[:,0]-anchor)/fdscale[None,:]-(target[:,0]-anchor)/fdscale[None,:])
 seam=.5*(seam_err[:,:14].mean(axis=1)+seam_err[:,14:].mean(axis=1))
 return {'position_mae':pos,'safe_fd_l1':fd,'seam_safe_fd_l1':seam}
def load_predictions(seal,dev_ids,train_ids):
 arrays={};receipts={}
 for r in seal['prediction_records']:
  key=(r['dataset_role'],r['input_policy'],r['program_id'],int(r['seed']))
  path=Path(r['prediction_path']);rows_path=Path(r['prediction_rows_path'])
  if sha(path)!=r['prediction_sha256'] or sha(rows_path)!=r['prediction_rows_sha256']:raise RuntimeError('sealed neural prediction hash mismatch')
  tensor=load_file(str(path),device='cpu')
  pred=tensor['prediction_native'].numpy()
  ids=train_ids if r['dataset_role']=='train' else dev_ids
  if pred.shape!=(len(ids),30,28) or pred.dtype!=np.float64 or not np.isfinite(pred).all():raise RuntimeError('sealed prediction array invalid')
  rows=read_csv(rows_path)
  if len(rows)!=len(ids) or [x['anchor_id'] for x in rows]!=ids:raise RuntimeError('sealed prediction rows differ from prefix order')
  arrays[key]=pred;receipts[key]=r
 for r in seal['derived_prediction_records']:
  key=('dev','derived_full_context',r['program_id'],int(r['seed']))
  path=Path(r['prediction_path']);rows_path=Path(r['prediction_rows_path'])
  if sha(path)!=r['prediction_sha256'] or sha(rows_path)!=r['prediction_rows_sha256']:raise RuntimeError('sealed mixture hash mismatch')
  tensor=load_file(str(path),device='cpu');pred=tensor['prediction_native'].numpy()
  if pred.shape!=(185,30,28) or pred.dtype!=np.float64 or not np.isfinite(pred).all():raise RuntimeError('sealed mixture array invalid')
  rows=read_csv(rows_path)
  if len(rows)!=185 or [x['anchor_id'] for x in rows]!=dev_ids:raise RuntimeError('sealed mixture ordered IDs differ')
  arrays[key]=pred;receipts[key]=r
 return arrays,receipts
def load_historical(ids,meta):
 path=V7/'results/PER_WINDOW_ALL_METHODS.csv';source=read_csv(path);out=[]
 keep=set(HIST)
 by={}
 for row in source:
  if row['method'] not in keep:continue
  aid=str(row['anchor_id'])
  if aid not in meta:continue
  m=meta[aid];seed=int(row['seed']);method=row['method']
  if seed not in SEEDS:raise RuntimeError('historical reference has an unexpected seed')
  if (int(row['episode_id']),int(row['task_id']),str(row['source_group']))!=(m['episode_id'],m['task_id'],m['source_group']):
   raise RuntimeError('historical reference metadata differs from accepted dev prefix')
  key=(method,seed,aid)
  if key in by:raise RuntimeError('duplicate historical method/seed/anchor row')
  vals={metric:float(row[metric]) for metric in METRICS}
  if not np.isfinite(list(vals.values())).all():raise RuntimeError('nonfinite historical metric')
  record={'method':method,'seed':seed,'anchor_id':aid,'episode_id':m['episode_id'],'task_id':m['task_id'],
   'source_group':m['source_group'],**vals,'source':'accepted_V7_per_window_reuse'}
  by[key]=record;out.append(record)
 for method in HIST:
  for seed in SEEDS:
   n=sum(1 for k in by if k[0]==method and k[1]==seed)
   if n!=185:raise RuntimeError(f'historical support mismatch {method}/{seed}: {n}')
 return out,sha(path)
def load_old_s_and_mix(seal,arrays,dev_ids,meta):
 source=rj(ROOT/'inventory/SOURCE_MAP.json')['source_assets']['accepted_S_OPT_dev_predictions']
 mixes=[];preds={}
 for seed in SEEDS:
  src=source[str(seed)];pred_path=Path(src['prediction']['path']);rows_path=Path(src['rows']['path'])
  if sha(pred_path)!=src['prediction']['sha256'] or sha(rows_path)!=src['rows']['sha256']:raise RuntimeError('accepted S_OPT source hash mismatch')
  ten=load_file(str(pred_path),device='cpu')
  if 'prediction_native' not in ten:raise RuntimeError('S_OPT prediction tensor key missing')
  values=np.asarray(ten['prediction_native'].numpy(),dtype=np.float64)
  if values.shape!=(185,30,28) or not np.isfinite(values).all():raise RuntimeError('S_OPT prediction array invalid')
  rows=read_csv(rows_path);sid=[str(x['anchor_id']) for x in rows]
  if len(sid)!=185 or len(set(sid))!=185 or set(sid)!=set(dev_ids):raise RuntimeError('S_OPT IDs differ from locked dev support')
  order=np.asarray([sid.index(aid) for aid in dev_ids],dtype=np.int64);s=values[order]
  for row in rows:
   aid=str(row['anchor_id']);m=meta[aid]
   if (int(row['episode_id']),int(row['task_id']),str(row['source_cluster']),int(row['anchor_frame']))!=(m['episode_id'],m['task_id'],m['source_group'],m['anchor_frame']):
    raise RuntimeError('S_OPT row identity mismatch')
  for mix,video in (('MIX0','V0'),('MIXD','VD')):
   mix_path=ROOT/'predictions/dev_derived'/mix/f'seed{seed}'/'PREDICTIONS.safetensors'
   done=rj(mix_path.parent/'DERIVED_COMPLETE.json')
   v=arrays[('dev','full_context',video,seed)]
   expected=0.5*s+0.5*v
   got=load_file(str(mix_path),device='cpu')['prediction_native'].numpy()
   if not np.array_equal(got,expected):raise RuntimeError('fixed native mixture differs from 0.5*(S+video)')
   if done.get('status')!='PASS' or done.get('formula')!='0.5*S_OPT + 0.5*'+video:raise RuntimeError('derived mixture receipt mismatch')
   preds[(mix,seed)]=got
   mixes.append({'method':mix,'seed':seed,'prediction_sha256':sha(mix_path),'formula':done['formula'],'support':185})
 return preds,mixes
def aggregate(rows):
 per_episode={}
 for r in rows:
  key=(r['method'],int(r['seed']),int(r['task_id']),int(r['episode_id']))
  cell=per_episode.setdefault(key,{m:[] for m in METRICS})
  for metric in METRICS:cell[metric].append(float(r[metric]))
 episode={k:{m:float(np.mean(v[m])) for m in METRICS} for k,v in per_episode.items()}
 result=[];task_seed={}
 methods=sorted({k[0] for k in episode})
 for method in methods:
  for seed in SEEDS:
   task_rows=[]
   for task in TASKS:
    eps=[k for k in episode if k[:3]==(method,seed,task)]
    if not eps:continue
    r={'method':method,'seed':seed,'task_id':task,**{m:float(np.mean([episode[k][m] for k in eps])) for m in METRICS},
      'episodes':len(eps),'anchors':sum(len(per_episode[k]['position_mae']) for k in eps)}
    task_seed[(method,seed,task)]=r;task_rows.append(r);result.append(r)
   if len(task_rows)==5:
    result.append({'method':method,'seed':seed,'task_id':'macro',**{m:float(np.mean([x[m] for x in task_rows])) for m in METRICS},
      'episodes':'task_equal','anchors':185})
 return result,task_seed,episode,per_episode
def primary_tables(rows):
 primary=[r for r in rows if r['method'] in set(PRIMARY)|set(HIST)]
 result,task_seed,episode,per_episode=aggregate(primary)
 method_macro=[]
 for method in sorted({r['method'] for r in result if r['task_id']=='macro'}):
  macros=[next(r for r in result if r['method']==method and r['seed']==seed and r['task_id']=='macro') for seed in SEEDS]
  for metric in METRICS:
   vals=[float(x[metric]) for x in macros]
   method_macro.append({'method':method,'metric':metric,'seed_202610131':vals[0],'seed_202610132':vals[1],'seed_202610133':vals[2],
    'mean':float(np.mean(vals)),'sample_sd':float(np.std(vals,ddof=1)),'source':'V9 full-context score or accepted V7 per-window reuse'})
 return result,task_seed,episode,per_episode,method_macro
def contrast_delta(name,metric,seed,aid,by):
 if name=='INTERACTION':
  return (by[('AD',seed,aid)][metric]-by[('VD',seed,aid)][metric])-(by[('A0',seed,aid)][metric]-by[('V0',seed,aid)][metric])
 ref,cand=CONTRASTS[name]
 return by[(ref,seed,aid)][metric]-by[(cand,seed,aid)][metric]
def paired_tables(rows,meta):
 by={(r['method'],int(r['seed']),str(r['anchor_id'])):r for r in rows}
 effects=[];seedloo=[];taskloo=[];task_seed={};episode_delta={}
 for name,(reference,candidate) in CONTRASTS.items():
  for metric in METRICS:
   for seed in SEEDS:
    for task in TASKS:
     aids=[aid for aid,m in meta.items() if m['task_id']==task]
     episode_group={}
     for aid in aids:
      delta=contrast_delta(name,metric,seed,aid,by);m=meta[aid]
      episode_group.setdefault((m['source_group'],m['episode_id']),[]).append(delta)
     episode_values={k:float(np.mean(v)) for k,v in episode_group.items()}
     for (group,episode_id),value in episode_values.items():
      episode_delta[(name,metric,seed,task,group,episode_id)]=value
     value=float(np.mean(list(episode_values.values())))
     task_seed[(name,metric,seed,task)]=value
     effects.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'task_seed',
       'seed':seed,'task_id':task,'gain_reference_minus_candidate':value,'episodes':len(episode_values),'anchors':len(aids)})
   seed_values=[]
   for seed in SEEDS:
    val=float(np.mean([task_seed[(name,metric,seed,t)] for t in TASKS]));seed_values.append(val)
    effects.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_seed_macro',
       'seed':seed,'task_id':'macro','gain_reference_minus_candidate':val})
   effects.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_three_seed_macro',
       'seed':'macro','task_id':'macro','gain_reference_minus_candidate':float(np.mean(seed_values))})
   for task in TASKS:
    val=float(np.mean([task_seed[(name,metric,s,task)] for s in SEEDS]))
    effects.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'three_seed_task_macro',
       'seed':'macro','task_id':task,'gain_reference_minus_candidate':val})
   for omit in SEEDS:
    remain=[s for s in SEEDS if s!=omit]
    val=float(np.mean([task_seed[(name,metric,s,t)] for s in remain for t in TASKS]))
    seedloo.append({'contrast':name,'metric':metric,'omitted_seed':omit,'remaining_seeds':','.join(map(str,remain)),
       'n_remaining_seeds':2,'loo_five_task_macro':val})
   for omit in TASKS:
    remain=[t for t in TASKS if t!=omit]
    val=float(np.mean([task_seed[(name,metric,s,t)] for s in SEEDS for t in remain]))
    taskloo.append({'contrast':name,'metric':metric,'omitted_task':omit,'remaining_tasks':','.join(map(str,remain)),
       'n_remaining_tasks':4,'seeds_included':3,'loo_macro':val})
 # Common within-task source-group draws, reused for every contrast and metric.
 rng=np.random.default_rng(20261016);draws={};digest=hashlib.sha256();support={}
 for task in TASKS:
  groups=sorted({m['source_group'] for m in meta.values() if m['task_id']==task})
  group_eps={g:sorted({m['episode_id'] for m in meta.values() if m['task_id']==task and m['source_group']==g}) for g in groups}
  probs=np.asarray([len(group_eps[g]) for g in groups],dtype=np.float64);probs/=probs.sum()
  arr=np.empty((5000,len(groups)),dtype=np.int32)
  for b in range(5000):arr[b]=rng.choice(len(groups),size=len(groups),replace=True,p=probs)
  draws[task]=(groups,group_eps,arr);digest.update(arr.tobytes(order='C'))
  support[str(task)]={'source_groups':len(groups),'episodes':int(sum(len(x) for x in group_eps.values()))}
 cis=[]
 for name,(reference,candidate) in CONTRASTS.items():
  for metric in METRICS:
   boot=np.empty(5000,dtype=np.float64)
   for b in range(5000):
    task_values=[]
    for task in TASKS:
     groups,group_eps,arr=draws[task];sampled=[]
     for gi in arr[b]:
      group=groups[int(gi)]
      for episode_id in group_eps[group]:
       vals=[episode_delta[(name,metric,seed,task,group,episode_id)] for seed in SEEDS]
       sampled.append(float(np.mean(vals)))
     task_values.append(float(np.mean(sampled)))
    boot[b]=float(np.mean(task_values))
   point=next(r['gain_reference_minus_candidate'] for r in effects if r['contrast']==name and r['metric']==metric and r['grain']=='five_task_three_seed_macro')
   cis.append({'contrast':name,'reference':reference,'candidate':candidate,'metric':metric,'grain':'five_task_three_seed_macro',
     'point_estimate':point,'ci_2_5':float(np.quantile(boot,.025)),'ci_97_5':float(np.quantile(boot,.975)),
     'bootstrap_draws':5000,'bootstrap_seed':20261016,'source_group_support':json.dumps(support,sort_keys=True),
     'conditional_on_fixed_models_and_dev_exposure':True,'p_value':None})
 boot_manifest={'schema_version':1,'status':'PASS','seed':20261016,'draws':5000,'draw_sha256':digest.hexdigest(),
   'same_draws_across_all_programs_seeds_policies_mixtures':True,'support':support,
   'interpretation':'within-task paired source-group bootstrap; fixed models and repeated dev exposure; not population training-seed uncertainty'}
 return effects,seedloo,taskloo,cis,boot_manifest
def masked_diagnostic(rows):
 full={(r['method'],int(r['seed']),r['anchor_id']):r for r in rows if r['method'] in PROGRAMS}
 masked={(r['program_id'],int(r['seed']),r['anchor_id']):r for r in rows if r.get('input_policy')=='force_numeric_context_drop'}
 result=[]
 for p in PROGRAMS:
  for seed in SEEDS:
   for task in TASKS:
    ids=[aid for aid in set(k[2] for k in full if k[0]==p and k[1]==seed) if any(r['anchor_id']==aid and int(r['task_id'])==task for r in rows)]
    by_ep={}
    for aid in ids:
     a=full[(p,seed,aid)];b=masked[(p,seed,aid)]
     by_ep.setdefault(int(a['episode_id']),{m:{'full':[],'mask':[]} for m in METRICS})
     for m in METRICS:
      by_ep[int(a['episode_id'])][m]['full'].append(float(a[m]));by_ep[int(a['episode_id'])][m]['mask'].append(float(b[m]))
    local={'method':p,'seed':seed,'task_id':task,'grain':'task_seed','episodes':len(by_ep),'anchors':len(ids)}
    for m in METRICS:
     ef=np.mean([np.mean(v[m]['full']) for v in by_ep.values()]);em=np.mean([np.mean(v[m]['mask']) for v in by_ep.values()])
     local[m+'_full']=float(ef);local[m+'_force_masked']=float(em);local[m+'_masked_minus_full']=float(em-ef)
    result.append(local)
   task_rows=[r for r in result if r['method']==p and r['seed']==seed and r['grain']=='task_seed']
   macro={'method':p,'seed':seed,'task_id':'macro','grain':'five_task_seed_macro','episodes':'task_equal','anchors':185}
   for m in METRICS:
    macro[m+'_full']=float(np.mean([r[m+'_full'] for r in task_rows]))
    macro[m+'_force_masked']=float(np.mean([r[m+'_force_masked'] for r in task_rows]))
    macro[m+'_masked_minus_full']=float(np.mean([r[m+'_masked_minus_full'] for r in task_rows]))
   result.append(macro)
 return result
def score_train(seal,arrays,train_ids,train_meta,source,metric):
 train_path=Path(source['train_npz']['path']);per_path=ROOT/'results/PER_WINDOW_TRAIN_FULL.csv'
 read_receipt=ROOT/'qa/TRAIN_TARGET_SINGLE_READ.json'
 if per_path.is_file() and read_receipt.is_file():
  receipt=rj(read_receipt)
  if receipt.get('status')!='PASS_RAW_TRAIN_SCORES_SAVED' or sha(per_path)!=receipt.get('per_window_train_sha256'):
   raise RuntimeError('saved train score/read receipt mismatch; refusing another Y load')
  out=read_csv(per_path)
  if len(out)!=15360:raise RuntimeError('saved full-context train score row count mismatch')
  train_archive_reads=int(receipt['train_target_archive_content_reads'])
 else:
  with np.load(train_path,allow_pickle=False) as z:
   required={'anchor_id','episode_id','task_id','source_cluster','anchor_native','target_native_h30'}
   if not required.issubset(z.files):raise RuntimeError('train target archive schema invalid')
   data={k:z[k].copy() for k in required}
  ids=data['anchor_id'].astype(str).tolist()
  if ids!=train_ids:raise RuntimeError('train target IDs/order differ from sealed train prefixes')
  target=np.asarray(data['target_native_h30'],dtype=np.float64);anchor=np.asarray(data['anchor_native'],dtype=np.float64)
  sigma=np.asarray(metric['target_sigma'],dtype=np.float64);fdscale=np.asarray(metric['safe_fd_scale'],dtype=np.float64)
  out=[]
  for p in PROGRAMS:
   for seed in SEEDS:
    pred=arrays[('train','full_context',p,seed)]
    comps=score_prediction(pred,target,anchor,sigma,fdscale)
    for i,aid in enumerate(ids):
     out.append({'method':p,'seed':seed,'anchor_id':aid,'episode_id':int(data['episode_id'][i]),'task_id':int(data['task_id'][i]),
       'source_group':str(data['source_cluster'][i]),'input_policy':'full_context','source':'V9_full_context_train_prediction',
       **{m:float(comps[m][i]) for m in METRICS}})
  if len(out)!=15360:raise RuntimeError('full-context train evaluation count mismatch')
  write_csv(per_path,out)
  read_receipt={'schema_version':1,'status':'PASS_RAW_TRAIN_SCORES_SAVED','created_utc':utc(),
   'train_target_archive_path':str(train_path),'accepted_train_archive_sha256':source['train_npz']['sha256'],
   'train_target_archive_content_reads':1,'numpy_load_calls':1,'train_target_rows_materialized':1280,
   'full_context_train_prediction_bundles_scored':12,'per_window_train_rows_saved':len(out),'per_window_train_sha256':sha(per_path),
   'dev_target_archive_opened':False,'dev_target_content_reads':0,'final_heldout_payload_access':0}
  wj(ROOT/'qa/TRAIN_TARGET_SINGLE_READ.json',read_receipt);train_archive_reads=1
 _,task_seed,_,_,_=primary_tables(out)
 fit_rows=[];curve=[]
 drop_probability={'A0':0.0,'AD':0.5,'V0':0.0,'VD':0.5}
 aggregated=aggregate(out)[0]
 for p in PROGRAMS:
  for seed in SEEDS:
   epoch=read_csv(ROOT/'model_only'/p/f'seed{seed}'/'attempt_01/EPOCH_LOG.csv')
   mask=read_csv(ROOT/'model_only'/p/f'seed{seed}'/'attempt_01/MASK_EPOCH_LOG.csv')
   if len(epoch)!=160:raise RuntimeError('training curve length not 160')
   curve.extend([{'program_id':p,'seed':seed,**r} for r in epoch])
   last=epoch[-1]
   masklast=[r for r in mask if int(r['epoch'])==160]
   drop_total=sum(int(r['receiver_n']) for r in masklast);drop_n=sum(int(r['dropped_n']) for r in masklast)
   for task in TASKS:
    row=task_seed[(p,seed,task)]
    task_mask=next(r for r in masklast if int(r['task_id'])==task)
    fit_rows.append({'method':p,'seed':seed,'task_id':task,'grain':'task_seed_full_context_train_eval',
      'episodes':row['episodes'],'anchors':row['anchors'],
      'train_full_position_mae':row['position_mae'],'train_full_safe_fd_l1':row['safe_fd_l1'],'train_full_seam_safe_fd_l1':row['seam_safe_fd_l1'],
      'configured_mask_probability':drop_probability[p],'observed_mask_rate_epoch160_task':float(task_mask['drop_rate_observed']),
      'masked_training_last_epoch_metrics_available':False})
   macro=next(r for r in aggregated if r['method']==p and int(r['seed'])==seed and r['task_id']=='macro')
   fit_rows.append({'method':p,'seed':seed,'task_id':'macro','grain':'five_task_seed_macro',
      'episodes':'task_equal','anchors':1280,'train_full_position_mae':macro['position_mae'],
      'train_full_safe_fd_l1':macro['safe_fd_l1'],'train_full_seam_safe_fd_l1':macro['seam_safe_fd_l1'],
      'configured_mask_probability':drop_probability[p],
      'observed_mask_rate_epoch160_task':drop_n/drop_total if drop_total else 0.0,
      'masked_training_last_epoch_position_mae':float(last['masked_input_true_y_position_mae']),
      'masked_training_last_epoch_safe_fd_l1':float(last['masked_input_true_y_safe_fd_l1']),
      'masked_training_last_epoch_seam_safe_fd_l1':float(last['masked_input_true_y_seam_safe_fd_l1']),
      'masked_training_last_epoch_true_y_objective':float(last['masked_input_true_y_objective']),
      'mask_receiver_rows_dropped_epoch160':drop_n,'mask_receiver_rows_seen_epoch160':drop_total})
 write_csv(ROOT/'results/TRAIN_AND_MASK_STATS.csv',fit_rows)
 write_csv(ROOT/'figure_data/TRAINING_CURVES.csv',curve)
 train_qa={'schema_version':1,'status':'PASS','created_utc':utc(),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
  'train_target_archive_path':str(train_path),'train_target_archive_reads':train_archive_reads,'train_target_rows':1280,'new_train_full_context_evaluations':12*1280,
  'new_train_predictions':12,'dev_target_archive_opened':False,'dev_target_content_reads':0,'final_heldout_payload_access':0,
  'train_per_window_sha256':sha(per_path),'mask_curve_rows':len(curve),'final_mask_epoch_rows':len(fit_rows)}
 wj(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json',train_qa)
 return fit_rows,train_qa
def prepare_dev_targets_once(dev_ids,meta,source):
 target_path=Path(source['dev_target_archive']['path']);expected_sha=source['dev_target_archive']['sha256_expected_from_accepted_V5_exposure']
 expected_bytes=int(source['dev_target_archive']['bytes_expected_from_accepted_V5_exposure'])
 if (ROOT/'qa/DEV_TARGET_SINGLE_READ.json').exists():
  raise RuntimeError('dev target already read; resume from saved per-window losses, never reopen')
 latch=ROOT/'qa/DEV_TARGET_READ_STARTED.json'
 if latch.exists():raise RuntimeError('a dev target read was already started; do not reopen')
 if target_path.stat().st_size!=expected_bytes:raise RuntimeError('dev target bytes differ from accepted V5 exposure')
 wj(latch,{'schema_version':1,'status':'READ_STARTED','started_utc':utc(),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
  'archive_path':str(target_path),'expected_sha256':expected_sha,'expected_bytes':expected_bytes,'planned_archive_content_reads':1})
 blob=target_path.read_bytes();actual_sha=hashlib.sha256(blob).hexdigest()
 if len(blob)!=expected_bytes or actual_sha!=expected_sha:raise RuntimeError('dev target archive identity differs from accepted V5 exposure')
 with np.load(io.BytesIO(blob),allow_pickle=False) as z:
  required={'anchor_id','episode_id','task_id','source_cluster','target_native_h30'}
  if not required.issubset(z.files):raise RuntimeError('dev target schema invalid')
  ids=z['anchor_id'].astype(str).tolist();eps=z['episode_id'].copy();tasks=z['task_id'].copy();groups=z['source_cluster'].astype(str)
  target=np.asarray(z['target_native_h30'],dtype=np.float64).copy()
 if len(ids)!=185 or len(set(ids))!=185 or set(ids)!=set(dev_ids):raise RuntimeError('dev target ordered ID set differs from sealed prefix')
 order=np.asarray([ids.index(aid) for aid in dev_ids],dtype=np.int64);target=target[order]
 for i,aid in enumerate(dev_ids):
  j=int(order[i]);m=meta[aid]
  if (int(eps[j]),int(tasks[j]),str(groups[j]))!=(m['episode_id'],m['task_id'],m['source_group']):raise RuntimeError('dev target identity mismatch '+aid)
 if target.shape!=(185,30,28) or not np.isfinite(target).all():raise RuntimeError('dev target shape/finite check failed')
 single={'schema_version':1,'status':'PASS','completed_utc':utc(),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
  'archive_path':str(target_path),'archive_sha256':actual_sha,'archive_bytes':len(blob),'archive_content_reads':1,'numpy_load_calls':1,
  'dev_rows_materialized':185,'target_payload_saved_to_v9':False,'preseal_target_reads':0,'final_heldout_payload_access':0}
 wj(ROOT/'qa/DEV_TARGET_SINGLE_READ.json',single)
 return target,single
def score_dev():
 seal=gate_sealed()
 train_gate=ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json'
 if not train_gate.is_file() or rj(train_gate).get('status')!='PASS':raise RuntimeError('train-only score preflight must pass before dev target read')
 source=rj(ROOT/'inventory/SOURCE_MAP.json')['source_assets']
 train_ids,train_meta=load_prefix(Path(source['train_prefix_npz']['path']),1280)
 dev_ids,meta=load_prefix(Path(source['dev_prefix_npz']['path']),185)
 arrays,receipts=load_predictions(seal,dev_ids,train_ids)
 train_rows,train_qa=score_train(seal,arrays,train_ids,train_meta,source,rj(source['metric_scale_lock']['path']))
 historical,hist_sha=load_historical(dev_ids,meta)
 mix_arrays,mixes=load_old_s_and_mix(seal,arrays,dev_ids,meta)
 per_path=ROOT/'results/PER_WINDOW_NEW_DEV_LOSSES.csv'
 if per_path.exists() and (ROOT/'qa/DEV_TARGET_SINGLE_READ.json').exists():
  new_rows=read_csv(per_path);single=rj(ROOT/'qa/DEV_TARGET_SINGLE_READ.json')
  if len(new_rows)!=30*185 or single.get('status')!='PASS':raise RuntimeError('saved dev loss table/target receipt invalid')
 else:
  target,single=prepare_dev_targets_once(dev_ids,meta,source)
  metric=rj(source['metric_scale_lock']['path']);sigma=np.asarray(metric['target_sigma'],dtype=np.float64);fd=np.asarray(metric['safe_fd_scale'],dtype=np.float64)
  anchors=np.stack([meta[aid]['anchor_native'] for aid in dev_ids])
  pred_specs=[]
  for p in PROGRAMS:
   for seed in SEEDS:
    pred_specs.append((p,seed,'full_context',arrays[('dev','full_context',p,seed)]))
    pred_specs.append((p+'_MASKED',seed,'force_numeric_context_drop',arrays[('dev','force_numeric_context_drop',p,seed)]))
  for mix in ('MIX0','MIXD'):
   for seed in SEEDS:pred_specs.append((mix,seed,'derived_full_context',mix_arrays[(mix,seed)]))
  new_rows=[]
  for method,seed,policy,pred in pred_specs:
   comps=score_prediction(pred,target,anchors,sigma,fd)
   for i,aid in enumerate(dev_ids):
    new_rows.append({'method':method,'program_id':(method.removesuffix('_MASKED') if method.endswith('_MASKED') else method),'seed':seed,'anchor_id':aid,'episode_id':meta[aid]['episode_id'],'task_id':meta[aid]['task_id'],
     'source_group':meta[aid]['source_group'],'input_policy':policy,'position_mae':float(comps['position_mae'][i]),
     'safe_fd_l1':float(comps['safe_fd_l1'][i]),'seam_safe_fd_l1':float(comps['seam_safe_fd_l1'][i]),
     'source':'V9_sealed_prediction_single_dev_target_read'})
  if len(new_rows)!=5550:raise RuntimeError('V9 dev logical evaluation count mismatch')
  write_csv(per_path,new_rows)
  exposure=rj(ROOT/'inventory/OUTCOME_EXPOSURE.json')
  prior=int(exposure['prior_v5_v6_v7_v8_known_grain_primary_loss_evaluations'])
  exposure.update({'status':'scored','v9_dev_target_archive_content_reads':1,'v9_dev_rows_loaded':185,
   'v9_new_dev_logic_bundles_scored':30,'v9_new_dev_loss_evaluations':5550,'v9_new_dev_task_views':150,
   'v9_primary_dev_loss_evaluations':3330,'v9_forced_mask_dev_loss_evaluations':2220,
   'known_grain_primary_loss_evaluations_after_v9':prior+3330,'known_grain_all_policy_loss_evaluations_after_v9':prior+5550,
   'legacy_35523_count':'unit_unverified_excluded','final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0})
  wj(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
 # Resume all post-read aggregation/report work from the seal-bound loss table only.
 new_rows=read_csv(per_path)
 if not (ROOT/'results/V9_SCORING_MANIFEST.json').is_file():
  wj(ROOT/'results/V9_SCORING_MANIFEST.json',{'schema_version':1,'status':'PASS','completed_utc':utc(),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
   'dev_target_archive_path':single['archive_path'],'dev_target_archive_sha256':single['archive_sha256'],'dev_target_archive_content_reads':1,
   'numpy_load_calls':1,'dev_rows_materialized':185,'new_dev_logic_bundles':30,'new_dev_loss_evaluations':5550,
   'primary_full_context_bundles':18,'primary_full_context_loss_evaluations':3330,'forced_mask_diagnostic_bundles':12,
   'forced_mask_loss_evaluations':2220,'derived_mixtures':6,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,
   'historical_V7_per_window_rows_reused':len(historical),'historical_V7_source_sha256':hist_sha,
   'metrics':'30x28 position MAE; 29 future-internal safe-FD L1; anchor seam Safe-FD L1',
   'aggregation':'anchors equal within episode; episodes equal within task; five tasks equal; seed reported separately'})
 all_rows=historical+new_rows
 primary,task_seed,episode,per_episode,macro=primary_tables(all_rows)
 write_csv(ROOT/'results/PER_WINDOW_ALL_METHODS.csv',all_rows)
 write_csv(ROOT/'results/PRIMARY_TASK_SEED.csv',primary)
 write_csv(ROOT/'results/METHOD_MACRO.csv',macro)
 masked=masked_diagnostic(new_rows);write_csv(ROOT/'results/MASKED_DIAGNOSTIC.csv',masked)
 effects,seedloo,taskloo,ci,boot=paired_tables(all_rows,meta)
 write_csv(ROOT/'results/PAIRED_EFFECTS.csv',effects);write_csv(ROOT/'results/SEED_LOO.csv',seedloo)
 write_csv(ROOT/'results/TASK_LOO.csv',taskloo);write_csv(ROOT/'results/PAIRED_GROUP_CI.csv',ci)
 wj(ROOT/'figure_data/BOOTSTRAP_DRAW_MANIFEST.json',boot)
 write_csv(ROOT/'figure_data/PRIMARY_PLOT_DATA.csv',[r for r in macro if r['method'] in PRIMARY])
 write_csv(ROOT/'figure_data/PAIRED_EFFECTS_PLOT_DATA.csv',[r for r in effects if r['grain']=='five_task_three_seed_macro'])
 write_csv(ROOT/'figure_data/MASK_DIAGNOSTIC_PLOT_DATA.csv',[r for r in masked if r['grain']=='five_task_seed_macro'])
 write_csv(ROOT/'figure_data/MIXTURE_PLOT_DATA.csv',[r for r in macro if r['method'] in ('S','V0','VD','SV50','MIX0','MIXD')])
 result={'schema_version':1,'status':'PASS','completed_utc':utc(),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
  'train_target_reads':1,'train_full_context_evaluations':15360,'dev_target_archive_content_reads':1,
  'new_dev_logic_bundles':30,'new_dev_loss_evaluations':5550,'primary_bundles':18,'masked_diagnostic_bundles':12,
  'new_dev_task_views':150,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,
  'bootstrap_draws':5000,'bootstrap_seed':20261016,'p_values':False}
 wj(ROOT/'qa/FINAL_ACCEPTANCE.json',result)
 print(json.dumps({'status':'PASS','new_dev_loss_evaluations':5550,'primary':3330,'forced_mask':2220,
  'archive_reads':1,'bootstrap_rows':len(ci),'dev_rows':185,'train_evaluations':15360},sort_keys=True))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train-only','dev']);args=ap.parse_args()
 seal=gate_sealed()
 if args.mode=='train-only':
  source=rj(ROOT/'inventory/SOURCE_MAP.json')['source_assets'];train_ids,train_meta=load_prefix(Path(source['train_prefix_npz']['path']),1280)
  dev_ids,_=load_prefix(Path(source['dev_prefix_npz']['path']),185);arrays,_=load_predictions(seal,dev_ids,train_ids)
  _,qa=score_train(seal,arrays,train_ids,train_meta,source,rj(source['metric_scale_lock']['path']))
  print(json.dumps({'status':'PASS','train_target_reads':1,'dev_target_reads':0,'train_evaluations':15360},sort_keys=True))
 else:score_dev()
if __name__=='__main__':main()

