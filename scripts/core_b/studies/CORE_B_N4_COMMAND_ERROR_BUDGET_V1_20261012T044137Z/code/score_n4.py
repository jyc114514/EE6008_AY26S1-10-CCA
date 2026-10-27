#!/usr/bin/env python3
"""Locked N4 score and error budget; target gate requires accepted 24/24 bundles."""
from __future__ import annotations
import os
import argparse,csv,hashlib,importlib.util,json,sys,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
N1=P/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
V3=P/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
EQ=P/'reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3'
METRIC_SOURCE=N2/'source/analyze_n1_metrics.py'
TRAIN_CACHE=EQ/'train_targets_trainonly.npz'
TRAIN_CACHE_MANIFEST=EQ/'TRAIN_TARGET_CACHE_MANIFEST.json'
THRESHOLD=N2/'n1_refs/accepted_n1_refs/MOTION_ACTIVITY_THRESHOLD.json'
SEEDS=(20261001,20261002,20261003)
SCOPES={'raw28':np.arange(28),'block0':np.arange(0,14),'block1':np.arange(14,28)}
ATOL=1e-12;RTOL=1e-10
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def C(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writej(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.part')
 t.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');t.replace(p)
def writec(p,rows):
 if not rows:raise ValueError('refusing to write empty table: '+str(p))
 p.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)
def appendj(p,o):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(o,sort_keys=True,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def metric_module():
 sys.dont_write_bytecode=True
 spec=importlib.util.spec_from_file_location('n4_accepted_n1_metrics',METRIC_SOURCE)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def group_reduce(metric_source,values,episodes,groups,group_order):
 # The locked split map proves one canonical episode per source group. This reducer
 # computes the accepted episode-mean then group-mean objective in O(N), while
 # retaining NaN support for metrics defined only on active-direction anchors.
 v=np.asarray(values,dtype=np.float64);ep=np.asarray(episodes,dtype=np.int64);grp=np.asarray(groups,dtype='U128')
 if v.ndim!=1 or len(v)!=len(ep) or len(v)!=len(grp):raise ValueError('group reducer vector shape mismatch')
 ep_to_group={}
 group_to_ep={}
 for e,g in zip(ep,grp):
  e=int(e);g=str(g)
  if e in ep_to_group and ep_to_group[e]!=g:raise ValueError('episode maps to multiple source groups')
  ep_to_group[e]=g
  if g in group_to_ep and group_to_ep[g]!=e:raise ValueError('source group maps to multiple episodes')
  group_to_ep[g]=e
 ep_order,inv=np.unique(ep,return_inverse=True)
 finite=np.isfinite(v)
 sums=np.bincount(inv[finite],weights=v[finite],minlength=len(ep_order))
 counts=np.bincount(inv[finite],minlength=len(ep_order))
 ep_mean=np.divide(sums,counts,out=np.full(len(ep_order),np.nan),where=counts>0)
 ep_pos={int(e):i for i,e in enumerate(ep_order)}
 out=np.full(len(group_order),np.nan,dtype=np.float64)
 for i,g in enumerate(group_order):
  e=group_to_ep.get(str(g))
  if e is not None:out[i]=ep_mean[ep_pos[e]]
 if set(group_to_ep)!=set(str(g) for g in group_order):raise ValueError('group reducer support differs from locked order')
 return out
def mean_finite(x):
 x=np.asarray(x,dtype=np.float64);m=np.isfinite(x)
 return float(x[m].mean()) if m.any() else float('nan')
def window_row_at(values,index):
 row={}
 for key,value in values.items():
  if isinstance(value,np.ndarray):
   item=value.item() if value.ndim==0 else value[index]
  elif isinstance(value,(list,tuple)):
   item=value[index]
  else:
   item=value
  if isinstance(item,np.generic):item=item.item()
  row[key]=item
 return row
def serialization_fixture():
 sample={'model_id':'fixture','seed':20261001,'anchor_id':np.asarray(['a','b']),
  'episode_index':np.asarray([1,2],dtype=np.int64),'score':np.asarray([0.25,0.5])}
 row=window_row_at(sample,1)
 assert row=={'model_id':'fixture','seed':20261001,'anchor_id':'b','episode_index':2,'score':0.5}
 return True
def derived(T,Pv,Cv):
 T=float(T);Pv=float(Pv);Cv=float(Cv);mse=T+Pv-2*Cv
 if not np.isfinite([T,Pv,Cv]).all():raise ValueError('nonfinite energy moment')
 if Cv*Cv>T*Pv+ATOL+RTOL*T*Pv:raise ArithmeticError('aggregate C^2 <= T*P failed')
 if T<=0:return {'fd_mse':mse,'fd_mse_over_T':float('nan'),'r_norm_ratio':float('nan'),
  'c_zero_lag_cosine':float('nan'),'fd_mse_skill':float('nan'),
  'norm_error_term':float('nan'),'alignment_error_term':float('nan')}
 r=float(np.sqrt(max(Pv,0)/T))
 if Pv<=0:c=float('nan');skill=0.0;norm=1.0;align=0.0
 else:
  c=Cv/np.sqrt(T*Pv);skill=(2*Cv-Pv)/T;norm=(1-r)**2;align=2*r*(1-c)
 rel=mse/T
 if not np.isclose(rel,norm+align,atol=ATOL,rtol=RTOL):raise ArithmeticError('aggregate FD geometry identity failed')
 return {'fd_mse':mse,'fd_mse_over_T':rel,'r_norm_ratio':r,'c_zero_lag_cosine':float(c),
  'fd_mse_skill':float(skill),'norm_error_term':float(norm),'alignment_error_term':float(align)}
def scoped_e0(pred,target,anchor,sigma,safe,active_threshold,axes):
 pred=np.asarray(pred,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
 anchor=np.asarray(anchor,dtype=np.float64);sigma=np.asarray(sigma,dtype=np.float64)[axes]
 safe=np.asarray(safe,dtype=np.float64)[axes];thr=np.asarray(active_threshold,dtype=np.float64)[axes]
 err=pred[:,:,axes]-target[:,:,axes];sp=err/sigma[None,None,:]
 dp=np.diff(pred[:,:,axes],axis=1);dy=np.diff(target[:,:,axes],axis=1);de=dp-dy
 absdy=np.abs(dy);active=absdy>thr[None,None,:]
 ac=active.sum(axis=(1,2));cc=((np.sign(dp)==np.sign(dy))&active).sum(axis=(1,2))
 direction=np.divide(cc,ac,out=np.full(len(pred),np.nan),where=ac>0)
 return {'position_standardized_mae':np.abs(sp).mean(axis=(1,2)),
  'safe_fd_normalized':(np.abs(de)/safe[None,None,:]).mean(axis=(1,2)),
  'native_future_internal_fd':np.abs(de).mean(axis=(1,2)),
  'seam_error_native':np.abs(err[:,0]).mean(axis=1),
  'seam_pred_step_abs':np.abs(pred[:,0,axes]-anchor[:,axes]).mean(axis=1),
  'seam_gt_step_abs':np.abs(target[:,0,axes]-anchor[:,axes]).mean(axis=1),
  'tv_pred':np.abs(dp).sum(axis=1).mean(axis=1),'tv_gt':np.abs(dy).sum(axis=1).mean(axis=1),
  'direction_accuracy_active':direction,'active_step_count':ac.astype(np.int64),
  'direction_correct_count':cc.astype(np.int64)}
def outcome_preflight(dry_run=False):
 accept=J(R/'qa/PREDICTION_ACCEPTANCE.json')
 if accept.get('status')!='PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES' or accept.get('bundles_accepted')!=24:
  raise SystemExit('target gate closed: prediction acceptance is not 24/24')
 exposure=J(R/'protocol/OUTCOME_EXPOSURE.json')
 if (R/'qa/ANALYSIS_ACCEPTANCE.json').exists():
  raise SystemExit('N4 analysis is already accepted; scoring is closed.')
 if exposure.get('n4_outcomes_opened') and not (R/'qa/ANALYSIS_ACCEPTANCE.json').exists():
  # A failed post-gate analysis may resume in a new numbered attempt using the same locked target scope.
  pass
 elif exposure.get('n4_outcomes_opened') is not False:
  raise SystemExit('N4 outcome state is ambiguous; inspect access ledger before proceeding.')
 amendment=J(R/'protocol/METRIC_SOURCE_AMENDMENT_001.json')
 source=J(R/'inventory/SOURCE_ACCEPTANCE.json')
 for key,field in [('train_target_cache','train_target_cache_sha256'),
                   ('target_normalizer','target_normalizer_sha256'),
                   ('dev_target_source','dev_target_source_sha256'),
                   ('motion_activity_threshold','motion_activity_threshold_sha256')]:
  if key=='train_target_cache':path=Path(amendment['train_target_cache_path'])
  elif key=='target_normalizer':path=V3/'data/V3_NORMALIZERS.json'
  elif key=='dev_target_source':path=Path(amendment['dev_target_source_path'])
  else:path=Path(amendment['motion_activity_threshold_path'])
  expected=(amendment[field] if field in amendment else amendment.get(key,{}).get('sha256'))
  if expected is None:raise SystemExit('source amendment missing hash '+key)
  if sha(path)!=expected:raise SystemExit('target source hash changed '+str(path))
 for rec in source.get('additional_locked_sources',{}).values():
  if isinstance(rec,dict) and 'path' in rec and sha(Path(rec['path']))!=rec['sha256']:
   raise SystemExit('additional locked source changed '+rec['path'])
 if sha(TRAIN_CACHE_MANIFEST)!=amendment['train_target_cache_manifest_sha256']:
  raise SystemExit('train target cache manifest changed')
 if sha(EQ/'TRAIN_PARQUET_SHA256.csv')!=amendment['train_parquet_hash_manifest_sha256']:
  raise SystemExit('train parquet hash manifest changed')
 if sha(METRIC_SOURCE)!=J(N2/'qa/N1_STORED_SCORE_RECONCILIATION.json')['metric_source_sha256']:
  raise SystemExit('accepted N1 scorer source hash mismatch')
 cache_meta=J(TRAIN_CACHE_MANIFEST)
 if cache_meta.get('status')!='PASS' or cache_meta.get('scope')!='primary_train_only' or cache_meta.get('final_test_access')!=0:
  raise SystemExit('train target cache acceptance metadata failed')
 if sha(TRAIN_CACHE)!=cache_meta.get('target_cache_sha256'):raise SystemExit('train target cache byte hash mismatch')
 if cache_meta.get('normalizer_sha256')!=sha(V3/'data/V3_NORMALIZERS.json'):
  raise SystemExit('train target cache normalizer mismatch')
 if not dry_run:
  prior=(R/'qa/SCORE_TARGET_ACCESS.jsonl').read_text(encoding='utf-8').splitlines() if (R/'qa/SCORE_TARGET_ACCESS.jsonl').exists() else []
  prior_start_count=sum(1 for line in prior if json.loads(line).get('event')=='N4_TARGET_READ_STARTED_AFTER_24_OF_24_PREDICTION_ACCEPTANCE')
  event={'attempt':prior_start_count+1,'utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
   'event':'N4_TARGET_READ_STARTED_AFTER_24_OF_24_PREDICTION_ACCEPTANCE',
   'prediction_acceptance_sha256':sha(R/'qa/PREDICTION_ACCEPTANCE.json'),
   'target_sources':[str(TRAIN_CACHE),str(V3/'data/V3_MAIN_INPUTS.npz')],
   'train_target_rows_authorized':17757,'dev_target_rows_authorized':2322,
   'final_test_access':0,'heldout_access':0,'purpose':'locked exploratory N4 error budget'}
  appendj(R/'qa/SCORE_TARGET_ACCESS.jsonl',event)
  exposure.update({'status':'N4_TARGETS_OPENED_AFTER_PREDICTION_GATE',
   'n4_outcomes_opened':True,'n4_target_read_started_utc':event['utc'],
   'n4_target_access_attempts':event['attempt'],
   'n4_prediction_acceptance_sha256':event['prediction_acceptance_sha256'],
   'n4_access':{'train_target_rows':17757,'dev_target_rows':2322,'final_test_access':0,'heldout_access':0}})
  writej(R/'protocol/OUTCOME_EXPOSURE.json',exposure)
 return accept
def load_targets_and_metadata():
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  split=np.asarray(z['split_code'],dtype=np.int8)
  ids=np.asarray(z['anchor_id']).astype('U96')
  episodes=np.asarray(z['episode_index'],dtype=np.int64)
  train_idx=np.flatnonzero(split==0);dev_idx=np.flatnonzero(split==1)
  # Only after the accepted prediction gate: open the authorized dev target fields.
  dev_targets=np.asarray(z['targets_native'][dev_idx,:30],dtype=np.float32).copy()
  dev_anchors=np.asarray(z['anchor_command_native'][dev_idx],dtype=np.float32).copy()
 train_ids=ids[train_idx];dev_ids=ids[dev_idx]
 train_eps=episodes[train_idx];dev_eps=episodes[dev_idx]
 with np.load(TRAIN_CACHE,allow_pickle=False) as cache:
  cache_ids=cache['anchor_id'].astype('U96')
  cache_eps=np.asarray(cache['episode_index'],dtype=np.int64)
  # This archive is separately accepted as primary_train_only.
  train_targets=np.asarray(cache['targets_native_60'][:,:30],dtype=np.float32).copy()
  train_anchors=np.asarray(cache['anchor_command_native'],dtype=np.float32).copy()
 if not np.array_equal(cache_ids,train_ids) or not np.array_equal(cache_eps,train_eps):
  raise AssertionError('train target cache canonical ID/episode order mismatch')
 if train_targets.shape!=(17757,30,28) or dev_targets.shape!=(2322,30,28):
  raise AssertionError('native target shape mismatch')
 return {'train':{'ids':train_ids,'episodes':train_eps,'targets':train_targets,'anchors':train_anchors},
  'dev':{'ids':dev_ids,'episodes':dev_eps,'targets':dev_targets,'anchors':dev_anchors}}
def main():
 ap=argparse.ArgumentParser(description='N4 locked exploratory position/FD error budget')
 ap.add_argument('--dry-run',action='store_true',help='validate target gate and source hashes without opening target arrays')
 args=ap.parse_args()
 accept=outcome_preflight(dry_run=args.dry_run)
 if args.dry_run:
  serial=serialization_fixture()
  print(json.dumps({'status':'PASS_TARGET_GATE_DRY_RUN','prediction_acceptance_sha256':sha(R/'qa/PREDICTION_ACCEPTANCE.json'),
   'target_files_opened':False,'train_target_cache_hash_verified':True,
   'window_serialization_fixture':'PASS' if serial else 'FAIL','final_test_access':0},ensure_ascii=False,indent=2));return
 src=metric_module()
 sys.path.insert(0,str(R/'code'))
 import n4_budget_math as mathmod
 target_data=load_targets_and_metadata()
 target_load_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 access_lines=(R/'qa/SCORE_TARGET_ACCESS.jsonl').read_text(encoding='utf-8').splitlines()
 current_attempt=max(int(json.loads(line).get('attempt',0)) for line in access_lines
  if json.loads(line).get('event')=='N4_TARGET_READ_STARTED_AFTER_24_OF_24_PREDICTION_ACCEPTANCE')
 appendj(R/'qa/SCORE_TARGET_ACCESS.jsonl',{'attempt':current_attempt,'utc':target_load_utc,'event':'N4_TRAIN_AND_DEV_TARGET_ARRAYS_LOADED',
  'train_rows':17757,'dev_rows':2322,'source_group_unit':'603 dev source groups; train-only cache',
  'final_test_access':0,'heldout_access':0})
 norm=J(V3/'data/V3_NORMALIZERS.json')
 sigma=np.asarray(norm['target_sigma'],dtype=np.float64);safe=np.asarray(norm['safe_fd_scale'],dtype=np.float64)
 threshold=J(THRESHOLD);active=np.asarray(threshold['direction_threshold_vector'],dtype=np.float64)
 if sigma.shape!=(28,) or safe.shape!=(28,) or active.shape!=(28,) or np.any(sigma<=0) or np.any(safe<=0):
  raise AssertionError('accepted scale/threshold shape or positivity mismatch')
 group_rows=C(N2/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv')
 ep_group={int(x['episode_index']):x['source_group_id'] for x in group_rows}
 split_key={'train':'train','dev':'dev'}
 groups={}
 for sp,d in target_data.items():
  groups[sp]=np.asarray([ep_group[int(e)] for e in d['episodes']],dtype='U128')
  if len(np.unique(groups[sp]))!=len(np.unique(d['episodes'])):
   raise AssertionError('source-group/episode mapping no longer one-to-one')
 train_order=sorted(set(groups['train'].tolist()));dev_order=sorted(set(groups['dev'].tolist()))
 with np.load(N2/'protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as z:
  draw_groups=z['source_group_ids'].astype('U128');draw_idx=np.asarray(z['group_draw_indices'],dtype=np.int64)
  seed_draw=np.asarray(z['seed_draw_indices'],dtype=np.int64)
  draw_seed=int(np.asarray(z['prng_seed']).item());draw_reps=int(np.asarray(z['replicates']).item())
 if draw_seed!=20261011 or draw_reps!=10000:
  raise AssertionError('N2 bootstrap draw seed/count differs from the lock')
 if not np.array_equal(draw_groups,np.asarray(dev_order,dtype='U128')) or draw_idx.shape!=(10000,603) or seed_draw.shape!=(10000,3):
  raise AssertionError('locked dev draw manifest no longer matches groups/seeds')
 metric_source_sha=sha(METRIC_SOURCE)
 if metric_source_sha!=J(N2/'qa/N1_STORED_SCORE_RECONCILIATION.json')['metric_source_sha256']:
  raise AssertionError('metric source changed after dry-run')
 attempt_parent=R/'tmp';attempt_parent.mkdir(parents=True,exist_ok=True)
 previous=[]
 for path in attempt_parent.glob('analysis_attempt_*'):
  try:previous.append(int(path.name.rsplit('_',1)[1]))
  except ValueError:pass
 attempt_no=max(previous,default=0)+1
 attempt_root=attempt_parent/f'analysis_attempt_{attempt_no:03d}'/'payload'
 attempt_root.mkdir(parents=True,exist_ok=False)
 (attempt_root/'results/PER_WINDOW_MOMENTS').mkdir(parents=True,exist_ok=True)
 seeds=[str(s) for s in SEEDS]
 matrix=C(R/'protocol/MATRIX24.csv')
 acc={(x['model'],x['query'],str(x['seed']),x['split']):x for x in accept['bundle_records']}
 group_order_by={'train':np.asarray(train_order,dtype='U128'),'dev':np.asarray(dev_order,dtype='U128')}
 # endpoint group-value store for paired gains and the locked dev bootstrap.
 cell_groups={};cell_point={};window_rows=[];episode_rows=[];budget_rows=[];oracle_rows=[];fd_rows=[];axis_rows=[];outconst_rows=[]
 analysis_checks=[]
 for mr in matrix:
  system,q,seed=mr['model'],mr['query'],str(mr['seed'])
  sp='train' if mr['split']=='primary_train' else 'dev'
  ck=(system,q,seed,sp);d=target_data[sp];gid=groups[sp];eps=d['episodes']
  receipt=Path(mr['prediction_receipt_path']);pfile=Path(mr['prediction_path'])
  if sha(pfile)!=J(receipt).get('prediction_sha256'):raise AssertionError('prediction changed after acceptance '+mr['view_id'])
  with np.load(pfile,allow_pickle=False) as z:
   pred=np.asarray(z['predictions_native'],dtype=np.float32).copy()
   got_ids=z['anchor_id'].astype('U96');got_eps=np.asarray(z['episode_index'],dtype=np.int64)
  if not np.array_equal(got_ids,d['ids']) or not np.array_equal(got_eps,eps):
   raise AssertionError('prediction IDs changed after acceptance '+mr['view_id'])
  y=d['targets'];anchor=d['anchors']
  if pred.shape!=y.shape:raise AssertionError('prediction/target shape mismatch '+mr['view_id'])
  e0=src.metric_arrays(pred,y,anchor,sigma.astype(np.float32),safe.astype(np.float32),active.astype(np.float32))
  pos=mathmod.position_budget(pred,y,sigma)
  if np.any(pos['oracle_removable_standardized_mae'] < -ATOL) or not np.allclose(
      pos['oracle_original_standardized_mae'],
      pos['oracle_removable_standardized_mae']+pos['oracle_remaining_standardized_mae'],
      atol=ATOL,rtol=RTOL):
   raise ArithmeticError('oracle normalized MAE budget identity failed '+mr['view_id'])
  # Native errors and exact future-internal derivative moments.
  err=pred.astype(np.float64)-y.astype(np.float64)
  dp=np.diff(pred.astype(np.float64),axis=1);dy=np.diff(y.astype(np.float64),axis=1)
  dnorm=dy/safe[None,None,:];p_norm=dp/safe[None,None,:]
  axis_T=np.mean(dnorm**2,axis=1);axis_P=np.mean(p_norm**2,axis=1);axis_C=np.mean(dnorm*p_norm,axis=1)
  for name,arr in [('T',axis_T),('P',axis_P),('C',axis_C)]:
   if arr.shape!=(len(pred),28) or not np.isfinite(arr).all():raise AssertionError('FD moment axis support mismatch '+name)
  if np.any(axis_C**2>axis_T*axis_P+ATOL+RTOL*axis_T*axis_P):
   raise ArithmeticError('per-window per-axis C^2 <= T*P failed')
  # Full 28-axis identities through the pure-arithmetic helper.
  fde=mathmod.fd_energy(pred,y,safe)
  outconst=pos['output_constant_native'].astype(np.float32)
  const_err=outconst-y
  const_dp=np.diff(outconst.astype(np.float64),axis=1)
  const_e0=scoped_e0(outconst,y,anchor,sigma,safe,active,np.arange(28))
  model_id=f'{system}__{q}__S{seed}__H30'
  scope_data={}
  for scope,axes in {**SCOPES,**{f'axis_{i:02d}':np.asarray([i]) for i in range(28)}}.items():
   if scope=='raw28':
    e0scope={k:np.asarray(v) for k,v in e0.items() if np.asarray(v).shape==(len(pred),)}
   else:e0scope=scoped_e0(pred,y,anchor,sigma,safe,active,axes)
   lvl=pos['level_mse_axis'][:,axes].mean(axis=1)
   ctr=pos['time_centered_error_mse_axis'][:,axes].mean(axis=1)
   orign=pos['oracle_original_standardized_mae_axis'][:,axes].mean(axis=1)
   remn=pos['oracle_remaining_standardized_mae_axis'][:,axes].mean(axis=1)
   rmn=orign-remn
   orig_native=np.mean(np.abs(err[:,:,axes]),axis=(1,2))
   rem_native=np.mean(np.abs(pos['oracle_remaining_error'][:,:,axes]*sigma[axes][None,None,:]),axis=(1,2))
   remv_native=orig_native-rem_native
   t=axis_T[:,axes].mean(axis=1);pv=axis_P[:,axes].mean(axis=1);c=axis_C[:,axes].mean(axis=1)
   mse=t+pv-2*c
   pos_direct=np.mean(pos['error_standardized'][:,:,axes]**2,axis=(1,2))
   if not np.allclose(pos_direct,lvl+ctr,atol=ATOL,rtol=RTOL):
    raise ArithmeticError('per-window position L2 level/time-centered identity failed '+scope)
   if np.any(c*c>t*pv+ATOL+RTOL*t*pv):raise ArithmeticError('per-window C^2<=TP failed '+scope)
   if not np.allclose(mse,np.mean(((p_norm-dnorm)[:,:,axes])**2,axis=(1,2)),atol=ATOL,rtol=RTOL):
    raise ArithmeticError('per-window FD MSE identity failed '+scope)
   derv=derived(float(np.mean(t)),float(np.mean(pv)),float(np.mean(c)))
   # episode/group reductions preserve the fixed episode-equal objective.
   endpoints={}
   for nm,val in e0scope.items():
    if np.asarray(val).shape==(len(pred),):
     endpoints[nm]=group_reduce(src,val,eps,gid,group_order_by[sp])
   per_values={'position_mse':lvl+ctr,'level_mse':lvl,'time_centered_error_mse':ctr,
    'oracle_original_standardized_mae':orign,'oracle_remaining_standardized_mae':remn,
    'oracle_removable_standardized_mae':rmn,'oracle_original_native_mae':orig_native,
    'oracle_remaining_native_mae':rem_native,'oracle_removable_native_mae':remv_native,
    'target_energy_T':t,'prediction_energy_P':pv,'cross_alignment_C':c,'fd_mse':mse}
   endpoints.update({nm:group_reduce(src,val,eps,gid,group_order_by[sp]) for nm,val in per_values.items()})
   summary={nm:mean_finite(val) for nm,val in endpoints.items()}
   aggT=summary['target_energy_T'];aggP=summary['prediction_energy_P'];aggC=summary['cross_alignment_C']
   aggD=derived(aggT,aggP,aggC)
   summary.update(aggD)
   # Per-episode distribution of per-window ratios is descriptive, not the main ratio estimator.
   ratios=fde
   if scope=='raw28':
    rwin=ratios['r_norm_ratio'];cwin=ratios['c_zero_lag_cosine'];swin=ratios['fd_mse_skill']
   else:
    rwin=np.full(len(pred),np.nan);cwin=np.full(len(pred),np.nan);swin=np.full(len(pred),np.nan)
    good=t>0;gp=good&(pv>0)
    rwin[good]=np.sqrt(pv[good]/t[good]);swin[good]=(2*c[good]-pv[good])/t[good]
    cwin[gp]=c[gp]/np.sqrt(t[gp]*pv[gp])
    rwin[good&(pv==0)]=0.;swin[good&(pv==0)]=0.
   for nm,val in [('path_r',rwin),('path_c',cwin),('path_skill',swin)]:
    gv=group_reduce(src,val,eps,gid,group_order_by[sp]);finite=gv[np.isfinite(gv)]
    summary[nm+'_q05']=float(np.percentile(finite,5)) if len(finite) else float('nan')
    summary[nm+'_median']=float(np.percentile(finite,50)) if len(finite) else float('nan')
    summary[nm+'_q95']=float(np.percentile(finite,95)) if len(finite) else float('nan')
    summary[nm+'_valid_groups']=int(len(finite))
   summary.update({'system':system,'query':q,'seed':int(seed),'model_id':model_id,'split':sp,'axis_scope':scope,
    'anchors':len(pred),'episodes':len(np.unique(eps)),'source_groups':len(np.unique(gid)),
    'valid_direction_groups':int(np.isfinite(endpoints.get('direction_accuracy_active',np.array([]))).sum()),
    'target_dynamic_zero_anchors':int(np.sum(fde['target_energy_T']==0)),
    'prediction_dynamic_zero_anchors':int(np.sum(fde['prediction_energy_P']==0))})
   scope_data[scope]={'group':endpoints,'summary':summary}
   axis_rows.append(summary)
   for endpoint,val in endpoints.items():
    cell_groups[(sp,scope,endpoint,system,q,int(seed))]=val
    cell_point[(sp,scope,endpoint,system,q,int(seed))]=mean_finite(val)
   for endpoint in ('fd_mse_over_T','r_norm_ratio','c_zero_lag_cosine','fd_mse_skill',
                    'norm_error_term','alignment_error_term'):
    cell_point[(sp,scope,endpoint,system,q,int(seed))]=summary[endpoint]
   # Core budget tables (full vector blocks and raw28 only).
   if scope in SCOPES:
    budget_rows.append({k:summary[k] for k in ('model_id','system','query','seed','split','axis_scope','position_mse',
      'level_mse','time_centered_error_mse','anchors','episodes','source_groups')})
    oracle_rows.append({k:summary[k] for k in ('model_id','system','query','seed','split','axis_scope',
      'oracle_original_standardized_mae','oracle_removable_standardized_mae','oracle_remaining_standardized_mae',
      'oracle_original_native_mae','oracle_removable_native_mae','oracle_remaining_native_mae',
      'anchors','episodes','source_groups')})
    fd_rows.append({k:summary[k] for k in ('model_id','system','query','seed','split','axis_scope',
      'target_energy_T','prediction_energy_P','cross_alignment_C','fd_mse','fd_mse_over_T',
      'r_norm_ratio','c_zero_lag_cosine','fd_mse_skill','norm_error_term','alignment_error_term',
      'path_r_q05','path_r_median','path_r_q95','path_r_valid_groups',
      'path_c_q05','path_c_median','path_c_q95','path_c_valid_groups',
      'path_skill_q05','path_skill_median','path_skill_q95','path_skill_valid_groups',
      'target_dynamic_zero_anchors','prediction_dynamic_zero_anchors','anchors','episodes','source_groups')})
  # Output-only constant trajectory diagnostic, independent of target in its construction.
  cm=const_e0
  outconst_rows.append({'model_id':model_id,'system':system,'query':q,'seed':int(seed),'split':sp,
   'output_construction':'repeat mean_t(yhat) per axis','future_gt_used_for_construction':False,
   'future_gt_used_for_evaluation':True,'position_standardized_mae':mean_finite(group_reduce(src,cm['position_standardized_mae'],eps,gid,group_order_by[sp])),
   'safe_fd_normalized':mean_finite(group_reduce(src,cm['safe_fd_normalized'],eps,gid,group_order_by[sp])),
   'native_future_internal_fd':mean_finite(group_reduce(src,cm['native_future_internal_fd'],eps,gid,group_order_by[sp])),
   'seam_error_native':mean_finite(group_reduce(src,cm['seam_error_native'],eps,gid,group_order_by[sp])),
   'tv_pred':mean_finite(group_reduce(src,cm['tv_pred'],eps,gid,group_order_by[sp])),
   'final_test_access':0})
  # Anchor-level sufficient metrics plus the original N1 scorer output.
  global_levels=pos['level_mse_axis'].mean(axis=1);global_centers=pos['time_centered_error_mse_axis'].mean(axis=1)
  win={'model_id':model_id,'system':system,'query':q,'seed':int(seed),'split':sp,
       'anchor_id':d['ids'],'episode_index':eps,'source_group_id':gid}
  for name,val in e0.items():win[name]=np.asarray(val)
  win.update({'position_mse':global_levels+global_centers,'level_mse':global_levels,
   'time_centered_error_mse':global_centers,
   'oracle_original_standardized_mae':pos['oracle_original_standardized_mae_axis'].mean(axis=1),
   'oracle_remaining_standardized_mae':pos['oracle_remaining_standardized_mae_axis'].mean(axis=1),
   'oracle_removable_standardized_mae':pos['oracle_removable_standardized_mae_axis'].mean(axis=1),
   'target_energy_T':fde['target_energy_T'],'prediction_energy_P':fde['prediction_energy_P'],
   'cross_alignment_C':fde['cross_alignment_C'],'fd_mse':fde['fd_mse'],
   'r_norm_ratio':fde['r_norm_ratio'],'c_zero_lag_cosine':fde['c_zero_lag_cosine'],
   'fd_mse_skill':fde['fd_mse_skill'],'fd_mse_over_T':fde['fd_mse_over_T'],
   'norm_error_term':fde['norm_error_term'],'alignment_error_term':fde['alignment_error_term'],
   'position_mse_block0':pos['level_mse_axis'][:,:14].mean(axis=1)+pos['time_centered_error_mse_axis'][:,:14].mean(axis=1),
   'position_mse_block1':pos['level_mse_axis'][:,14:].mean(axis=1)+pos['time_centered_error_mse_axis'][:,14:].mean(axis=1)})
  # Write raw per-window rows for audit and episode-level rows after grouping anchors.
  for i in range(len(pred)):
   window_rows.append(window_row_at(win,i))
  for ep in np.unique(eps):
   ix=np.flatnonzero(eps==ep);i0=int(ix[0]);erow={'model_id':model_id,'system':system,'query':q,'seed':int(seed),
    'split':sp,'episode_index':int(ep),'source_group_id':str(gid[i0]),'anchors_in_episode':len(ix)}
   for name,val in win.items():
    if name in ('model_id','system','query','seed','split','anchor_id','episode_index','source_group_id'):continue
    vv=np.asarray(val)[ix].astype(np.float64)
    erow[name]=float(np.nanmean(vv)) if np.isfinite(vv).any() else float('nan')
   episode_rows.append(erow)
  # Recomputable per-window/per-axis moments for all 24 bundles.
  moment_path=attempt_root/'results/PER_WINDOW_MOMENTS'/f'{model_id}__{sp}.npz'
  with moment_path.open('wb') as f:
   np.savez_compressed(f,anchor_id=d['ids'],episode_index=eps,source_group_id=gid,
    level_offset=pos['level_offset'],level_mse_axis=pos['level_mse_axis'],
    time_centered_error_mse_axis=pos['time_centered_error_mse_axis'],
    oracle_level_delta=pos['oracle_level_delta'],
    oracle_original_standardized_mae_axis=pos['oracle_original_standardized_mae_axis'],
    oracle_remaining_standardized_mae_axis=pos['oracle_remaining_standardized_mae_axis'],
    target_energy_T_axis=axis_T,prediction_energy_P_axis=axis_P,cross_alignment_C_axis=axis_C)
  appendj(R/'logs/ANALYSIS_PROGRESS.jsonl',{'utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
   'event':'bundle_scored','model_id':model_id,'split':sp,'anchor_rows':len(pred),
   'target_arrays_opened_after_prediction_gate':True})
  print(json.dumps({'status':'bundle_scored','model_id':model_id,'split':sp,'anchors':len(pred)},sort_keys=True),flush=True)
 # N2 original-score reconciliation for the two co-primary endpoints.
 old_rows=C(N2/'results/N2_GROUP_LEVEL_METRICS.csv')
 old={}
 for row in old_rows:
  if row.get('profile')=='CUDA_COMMON' and row.get('metric') in ('position_standardized_mae','safe_fd_normalized'):
   key=(row['model'],row['query'],str(row['seed']),row['metric'])
   old.setdefault(key,[]).append((row['source_group_id'],float(row['group_value'])))
 rec=[];maxdiff=0.0
 for model in ('TS11-D','STATE-CTRL'):
  for q in ('Q0','Q1'):
   for seed in seeds:
    for endpoint in ('position_standardized_mae','safe_fd_normalized'):
     selected=[row for row in window_rows if row['system']==model and row['query']==q and
      str(row['seed'])==seed and row['split']=='dev']
     names=dev_order
     vals=group_reduce(src,np.asarray([float(row[endpoint]) for row in selected],dtype=np.float64),
      np.asarray([int(row['episode_index']) for row in selected],dtype=np.int64),
      np.asarray([str(row['source_group_id']) for row in selected],dtype='U128'),names)
     actual=float(np.mean(vals));oldkey=(model,q,seed,endpoint)
     oldvals=old.get(oldkey)
     if not oldvals:raise AssertionError('N2 old metric row missing '+str(oldkey))
     oldmap=dict(oldvals)
     if names!=sorted(oldmap):raise AssertionError('N2 group order mismatch '+str(oldkey))
     expected=float(np.mean([oldmap[n] for n in names]));delta=actual-expected;maxdiff=max(maxdiff,abs(delta))
     rec.append({'model':model,'query':q,'seed':int(seed),'split':'dev','metric':endpoint,
      'n_source_groups':len(names),'new_n4_score':actual,'accepted_n2_cuda_common_score':expected,
      'difference':delta,'tolerance_abs':1e-12,'status':'PASS' if abs(delta)<=1e-12 else 'FAIL'})
 status='PASS' if all(r['status']=='PASS' for r in rec) else 'FAIL'
 # Build paired G/V/I for every separate loss/component; never add unlike endpoints.
 paired=[];byseed={}
 target_energy_invariance_checks=0
 endpoints=list(dict.fromkeys([
  'position_standardized_mae','safe_fd_normalized','native_future_internal_fd',
  'position_mse','level_mse','time_centered_error_mse',
  'oracle_original_standardized_mae','oracle_removable_standardized_mae','oracle_remaining_standardized_mae',
  'oracle_original_native_mae','oracle_removable_native_mae','oracle_remaining_native_mae',
  'target_energy_T','prediction_energy_P','cross_alignment_C','fd_mse',
  'fd_mse_over_T','r_norm_ratio','c_zero_lag_cosine','fd_mse_skill','norm_error_term','alignment_error_term']))
 for sp in ('train','dev'):
  for scope in SCOPES:
   for seed in SEEDS:
    for endpoint in endpoints:
     cells={}
     for system in ('TS11-D','STATE-CTRL'):
      for q in ('Q0','Q1'):
       cells[(system,q)]=cell_point[(sp,scope,endpoint,system,q,seed)]
     if endpoint=='target_energy_T':
      tv=np.asarray(list(cells.values()),dtype=np.float64)
      if not np.allclose(tv,tv[0],atol=ATOL,rtol=RTOL):
       raise ArithmeticError('target energy T differs across models on identical support')
      target_energy_invariance_checks+=1
     if endpoint in ('r_norm_ratio','c_zero_lag_cosine','fd_mse_skill','norm_error_term','alignment_error_term','fd_mse_over_T'):
      # Values already use aggregate T/P/C for the fixed model/seed.
      pass
     GTS=cells[('TS11-D','Q0')]-cells[('TS11-D','Q1')]
     GST=cells[('STATE-CTRL','Q0')]-cells[('STATE-CTRL','Q1')]
     V0=cells[('STATE-CTRL','Q0')]-cells[('TS11-D','Q0')]
     V1=cells[('STATE-CTRL','Q1')]-cells[('TS11-D','Q1')]
     I=GTS-GST
     if not np.isclose(I,V1-V0,atol=ATOL,rtol=RTOL):raise ArithmeticError('G/V/I identity failed')
     for contrast,value in [('G_TS11-D',GTS),('G_STATE-CTRL',GST),('V_Q0',V0),('V_Q1',V1),('I',I)]:
      row={'split':sp,'axis_scope':scope,'seed':seed,'endpoint':endpoint,'contrast':contrast,
       'estimate':value,'interpretation':'positive favors Q1 for G; positive favors TS11-D for V; I=G_TS11-D-G_STATE-CTRL'}
      paired.append(row);byseed[(sp,scope,endpoint,contrast,seed)]=value
 # Check the linear E1 and E2 signed-component identities at each fixed seed.
 fd_component_identity_checks=0
 for sp in ('train','dev'):
  for scope in SCOPES:
   for seed in SEEDS:
    for contrast in ('G_TS11-D','G_STATE-CTRL','V_Q0','V_Q1','I'):
     posv=byseed[(sp,scope,'position_mse',contrast,seed)]
     lv=byseed[(sp,scope,'level_mse',contrast,seed)]
     cv=byseed[(sp,scope,'time_centered_error_mse',contrast,seed)]
     if not np.isclose(posv,lv+cv,atol=ATOL,rtol=RTOL):
      raise ArithmeticError('paired position gain is not sum of level/centered gains')
     tv=byseed[(sp,scope,'target_energy_T',contrast,seed)]
     pv=byseed[(sp,scope,'prediction_energy_P',contrast,seed)]
     xc=byseed[(sp,scope,'cross_alignment_C',contrast,seed)]
     fmv=byseed[(sp,scope,'fd_mse',contrast,seed)]
     if not np.isclose(tv,0.0,atol=ATOL,rtol=RTOL):
      raise ArithmeticError('T contrast must be zero under identical support')
     if not np.isclose(fmv,pv-2*xc,atol=ATOL,rtol=RTOL):
      raise ArithmeticError('FD_MSE contrast must be P contrast minus 2C contrast')
     fd_component_identity_checks+=1
 # Seed summaries and all three LOO means.
 loo=[]
 for sp in ('train','dev'):
  for scope in SCOPES:
   for endpoint in endpoints:
    for contrast in ('G_TS11-D','G_STATE-CTRL','V_Q0','V_Q1','I'):
     v=np.asarray([byseed[(sp,scope,endpoint,contrast,s)] for s in SEEDS],dtype=np.float64)
     for i,seed in enumerate(SEEDS):
      loo.append({'split':sp,'axis_scope':scope,'endpoint':endpoint,'contrast':contrast,
       'seed':seed,'seed_estimate':v[i],'three_seed_mean':float(v.mean()),
       'three_seed_sample_sd':float(v.std(ddof=1)),'three_seed_median':float(np.median(v)),
       'loo_drop_seed':seed,'loo_mean':float(np.delete(v,i).mean()),
       'n_fixed_seeds':3,'p_values_computed':False})
 # Source-group paired percentile bootstrap; common group and seed draws for all raw/block endpoints.
 draw_path=N2/'protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz'
 with np.load(draw_path,allow_pickle=False) as z:
  draws=np.asarray(z['group_draw_indices'],dtype=np.int64);seed_draw=np.asarray(z['seed_draw_indices'],dtype=np.int64)
  draw_ids=z['source_group_ids'].astype('U128')
 if not np.array_equal(draw_ids,np.asarray(dev_order,dtype='U128')):raise AssertionError('bootstrap group order mismatch')
 weight_counts=np.zeros((draws.shape[0],len(dev_order)),dtype=np.int16)
 for i,row in enumerate(draws):weight_counts[i]=np.bincount(row,minlength=len(dev_order)).astype(np.int16)
 W=weight_counts.astype(np.float64)/len(dev_order)
 boot_rows=[]
 boot_linear=['position_standardized_mae','safe_fd_normalized','native_future_internal_fd',
  'position_mse','level_mse','time_centered_error_mse','oracle_original_standardized_mae',
  'oracle_removable_standardized_mae','oracle_remaining_standardized_mae']
 for scope in SCOPES:
  # One matrix multiply per endpoint obtains the same 10,000 group draws for 12 fixed model/seed cells.
  for endpoint in boot_linear:
   cols=[]
   for seed in SEEDS:
    for system,q in [('TS11-D','Q0'),('TS11-D','Q1'),('STATE-CTRL','Q0'),('STATE-CTRL','Q1')]:
     gv=cell_groups[('dev',scope,endpoint,system,q,seed)]
     if not np.isfinite(gv).all():raise AssertionError('nonfinite core bootstrap endpoint '+endpoint)
     cols.append(gv)
   estimate=W@np.column_stack(cols)
   estimate=estimate.reshape(len(draws),3,4)
   emit_boot(scope,endpoint,estimate,seed_draw,boot_rows)
  # Resample groups for T/P/C, then derive each fixed-seed ratio within each draw.
  moment_est={}
  for endpoint in ('target_energy_T','prediction_energy_P','cross_alignment_C'):
   cols=[]
   for seed in SEEDS:
    for system,q in [('TS11-D','Q0'),('TS11-D','Q1'),('STATE-CTRL','Q0'),('STATE-CTRL','Q1')]:
     gv=cell_groups[('dev',scope,endpoint,system,q,seed)]
     if not np.isfinite(gv).all():raise AssertionError('nonfinite FD moments in bootstrap')
     cols.append(gv)
   moment_est[endpoint]=(W@np.column_stack(cols)).reshape(len(draws),3,4)
  t=moment_est['target_energy_T'];pv=moment_est['prediction_energy_P'];c=moment_est['cross_alignment_C']
  ratio_draws={}
  names=('fd_mse','fd_mse_over_T','r_norm_ratio','c_zero_lag_cosine','fd_mse_skill','norm_error_term','alignment_error_term')
  for endpoint in names:
   val=np.full_like(t,np.nan,dtype=np.float64)
   for si in range(3):
    for ci in range(4):
     for b in range(len(draws)):
      d=derived(t[b,si,ci],pv[b,si,ci],c[b,si,ci])
      val[b,si,ci]=d[endpoint]
   ratio_draws[endpoint]=val
   emit_boot(scope,endpoint,val,seed_draw,boot_rows)
 # Add N2 reconciliation/analysis receipts and staged output tables.
 stage=attempt_root
 writec(stage/'results/PER_WINDOW_METRICS.csv',window_rows)
 writec(stage/'results/PER_EPISODE_METRICS.csv',episode_rows)
 writec(stage/'results/BY_MODEL_SPLIT_AXIS.csv',axis_rows)
 writec(stage/'results/POSITION_BUDGET.csv',budget_rows)
 writec(stage/'results/ORACLE_MAE_BUDGET.csv',oracle_rows)
 writec(stage/'results/FD_ENERGY_ALIGNMENT.csv',fd_rows)
 writec(stage/'results/PAIRED_COMPONENT_GAINS.csv',paired)
 writec(stage/'results/ALL_SEED_LOO.csv',loo)
 writec(stage/'results/GROUP_BOOTSTRAP_CI.csv',boot_rows)
 writec(stage/'results/OUTPUT_CONSTANT_DIAGNOSTIC.csv',outconst_rows)
 rec_summary={'schema_version':1,'status':status,'metric_source_path':str(METRIC_SOURCE),
  'metric_source_sha256':metric_source_sha,'co_primary_count':len(rec),'max_abs_difference':maxdiff,
  'tolerance_abs':1e-12,'development_source':'reused N2 CUDA_COMMON 12 views; previously exposed',
  'rows':rec,'targets_loaded_after_prediction_acceptance':True,'final_test_access':0}
 writej(stage/'qa/ORIGINAL_METRIC_RECONCILIATION.json',rec_summary)
 # Store the bootstrap draw manifest hash and arithmetic identities.
 math_result=mathmod.run_fixtures()
 writej(stage/'qa/ARITHMETIC_IDENTITIES.json',math_result)
 # Make checks for all 12 seeds and all scope budgets explicit.
 e1check={'status':'PASS','position_identity_checks':0,'fd_energy_identity_checks':0,
  'cauchy_schwarz_checks':0,'target_energy_model_invariance_checks':target_energy_invariance_checks,
  'fd_component_contrast_identity_checks':fd_component_identity_checks,
  'n2_draw_manifest_sha256':sha(draw_path),
  'bootstrap_replicates':10000,'bootstrap_seed':draw_seed,
  'bootstrap_group_unit':'603 source groups; each currently maps to one canonical episode',
  'bootstrap_seed_resampling':'reuse N2 seed_draw_indices; compute per-seed nonlinear ratios then resample seed weights',
  'p_values':'not computed','final_test_access':0}
 for r in axis_rows:
  if not np.isclose(r['position_mse'],r['level_mse']+r['time_centered_error_mse'],atol=ATOL,rtol=RTOL):
   raise ArithmeticError('saved position budget identity mismatch')
  e1check['position_identity_checks']+=1
  if not np.isclose(r['fd_mse'],r['target_energy_T']+r['prediction_energy_P']-2*r['cross_alignment_C'],atol=ATOL,rtol=RTOL):
   raise ArithmeticError('saved FD moment identity mismatch')
  e1check['fd_energy_identity_checks']+=1
  e1check['cauchy_schwarz_checks']+=1
 writej(stage/'qa/ANALYSIS_IDENTITIES.json',e1check)
 # Generate acceptance metadata in staging; only after every computation and original-score reconciliation pass.
 if status!='PASS':raise AssertionError('N1 co-primary reconciliation failed; do not commit N4 analysis tables.')
 final={'schema_version':1,'status':'PASS_N4_ERROR_BUDGET_ANALYSIS',
  'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'prediction_acceptance_sha256':sha(R/'qa/PREDICTION_ACCEPTANCE.json'),
  'metric_source_sha256':metric_source_sha,'analysis_lock_sha256':sha(R/'protocol/ANALYSIS_LOCK.json'),
  'metric_budget_lock_sha256':sha(R/'protocol/METRIC_BUDGET_LOCK.json'),
  'source_amendment_sha256':sha(R/'protocol/METRIC_SOURCE_AMENDMENT_001.json'),
  'target_arrays_opened_after_acceptance':True,'train_targets_read':17757,'dev_targets_read':2322,
  'final_test_access':0,'heldout_access':0,'new_training':0,'optimizer_steps':0,'encoder_extraction':0,
  'new_train_forwards':12,'new_dev_formal_forwards':0,'N2_rerun':False,'N3_rescored':False,
  'n3_direct_reference':str(R/'n3_refs/N3_DIRECT_REUSE.json'),
  'original_metric_reconciliation':status,'max_abs_difference':maxdiff,
  'bootstrap_reused_n2_draws':True,'bootstrap_replicates':10000,'p_values_computed':False,
  'train_ci':'not computed; in-sample','development':'previously exposed; exploratory only',
  'output_counts':{'per_window_rows':len(window_rows),'per_episode_rows':len(episode_rows),
   'axis_rows':len(axis_rows),'paired_component_rows':len(paired),'seed_loo_rows':len(loo),
   'bootstrap_ci_rows':len(boot_rows),'moment_bundles':24}}
 writej(stage/'qa/ANALYSIS_ACCEPTANCE.json',final)
 # Check no staged output collides, then atomically commit each result.
 files=sorted(x for x in stage.rglob('*') if x.is_file())
 for f in files:
  dst=R/f.relative_to(stage)
  if dst.exists():raise FileExistsError('preserving existing N4 output; staged output collision: '+str(dst))
  dst.parent.mkdir(parents=True,exist_ok=True)
 for f in files:os.replace(f,R/f.relative_to(stage))
 # Exposure update follows actual target load and completed scores.
 exposure=J(R/'protocol/OUTCOME_EXPOSURE.json')
 exposure.update({'status':'N4_TARGETS_SCORED_AFTER_24_OF_24_ACCEPTANCE',
  'n4_target_access_completed_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'train_target_rows_read':17757,'dev_target_rows_read':2322,'final_test_access':0,
  'heldout_access':0,'analysis_acceptance_sha256':sha(R/'qa/ANALYSIS_ACCEPTANCE.json')})
 writej(R/'protocol/OUTCOME_EXPOSURE.json',exposure)
 print(json.dumps({'status':'PASS_N4_ERROR_BUDGET_ANALYSIS','root':str(R),
  'co_primary_reconciliation_max_abs':maxdiff,'analysis_acceptance':str(R/'qa/ANALYSIS_ACCEPTANCE.json'),
  'target_arrays_opened_after_24_of_24':True,'final_test_access':0,'p_values_computed':False,
  'ci_rows':len(boot_rows),'seed_loo_rows':len(loo)},ensure_ascii=False,indent=2))
def emit_boot(scope,endpoint,est,seed_draw,rows):
 # est is [replicate, fixed_seed, four system/query cells].
 B=est.shape[0];names={'G_TS11-D':(0,1,2,3),'G_STATE-CTRL':(0,1,2,3),
  'V_Q0':(0,1,2,3),'V_Q1':(0,1,2,3),'I':(0,1,2,3)}
 # condition indexes are TS-Q0, TS-Q1, STATE-Q0, STATE-Q1.
 contrast=np.empty((B,3,5),dtype=np.float64)
 contrast[:,:,0]=est[:,:,0]-est[:,:,1]
 contrast[:,:,1]=est[:,:,2]-est[:,:,3]
 contrast[:,:,2]=est[:,:,2]-est[:,:,0]
 contrast[:,:,3]=est[:,:,3]-est[:,:,1]
 contrast[:,:,4]=contrast[:,:,0]-contrast[:,:,1]
 if not np.allclose(contrast[:,:,4],contrast[:,:,3]-contrast[:,:,2],atol=ATOL,rtol=RTOL,equal_nan=True):
  raise ArithmeticError('bootstrap G/V/I identity failed')
 contrasts=['G_TS11-D','G_STATE-CTRL','V_Q0','V_Q1','I']
 # Cell estimates and paired contrast estimates use the identical resampled groups and seed weights.
 for ci,cell in enumerate([('TS11-D','Q0'),('TS11-D','Q1'),('STATE-CTRL','Q0'),('STATE-CTRL','Q1')]):
  sval=est[:,:,ci]
  selected=np.take_along_axis(sval,seed_draw,axis=1)
  valid=np.isfinite(selected).all(axis=1);boot=np.full(B,np.nan,dtype=np.float64)
  boot[valid]=selected[valid].mean(axis=1)
  _cirow(scope,endpoint,'CELL:'+cell[0]+':'+cell[1],boot,rows)
 for i,name in enumerate(contrasts):
  selected=np.take_along_axis(contrast[:,:,i],seed_draw,axis=1)
  valid=np.isfinite(selected).all(axis=1);boot=np.full(B,np.nan,dtype=np.float64)
  boot[valid]=selected[valid].mean(axis=1)
  _cirow(scope,endpoint,name,boot,rows)
def _cirow(scope,endpoint,estimand,boot,rows):
 v=np.asarray(boot,dtype=np.float64);v=v[np.isfinite(v)]
 if not len(v):lo=hi=mean=float('nan')
 else:lo,hi=np.percentile(v,[2.5,97.5]);mean=float(v.mean())
 rows.append({'split':'dev','axis_scope':scope,'endpoint':endpoint,'estimand':estimand,
  'bootstrap_mean':mean,'ci_percentile_lower_2_5':float(lo),'ci_percentile_upper_97_5':float(hi),
  'valid_draws':int(len(v)),'requested_draws':10000,'draw_manifest':'N2_GROUP_BOOTSTRAP_DRAWS.npz',
  'ci_scope':'conditional on fixed 12 models and previously exposed dev; exploratory',
  'p_values_computed':False,'independent_confirmation':False})
if __name__=='__main__':main()
