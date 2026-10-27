#!/usr/bin/env python3
"""Locked N2 scoring using the hash-pinned accepted N1 metric functions."""
from __future__ import annotations
import os
import csv,hashlib,importlib.util,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BLOCKED=PROJECT/'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
INPUTS=V3/'data/V3_MAIN_INPUTS.npz'
TARGET_NORMALIZER=V3/'data/V3_NORMALIZERS.json'
THRESHOLD=ROOT/'n1_refs/accepted_n1_refs/MOTION_ACTIVITY_THRESHOLD.json'
GROUP_MAP=ROOT/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv'
REQUEST=ROOT/'MODEL_EXPORT_REQUEST.csv'
MATRIX=ROOT/'protocol/N2_MATRIX24.csv'
N1_METRIC_SOURCE=ROOT/'source/analyze_n1_metrics.py'
N1_METRIC_SHA='bd18d29fe35eda689e2fd0fa6030685e81391dcbfa09926ce45d690f0e06c138'
N2_MATRIX_SHA='70e81b236727c617aeadc21e67dbb0412d6a8c56a2ab3ec03151e077897a6ffd'
BOOTSTRAP_SEED=20261011
BOOTSTRAP_REPLICATES=10000
SEEDS=(20261001,20261002,20261003)
MAIN_METRICS=('position_standardized_mae','safe_fd_normalized')

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sha_array(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(path:Path,value)->None:path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def write_csv(path:Path,rows:list[dict])->None:
 if not rows:raise RuntimeError('refuse empty results table '+str(path))
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=[]
 for row in rows:
  for k in row:
   if k not in fields:fields.append(k)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)

def load_n1_metrics():
 if sha(N1_METRIC_SOURCE)!=N1_METRIC_SHA:raise RuntimeError('locked N1 metric source SHA mismatch')
 spec=importlib.util.spec_from_file_location('locked_n1_metrics',N1_METRIC_SOURCE)
 module=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=module
 spec.loader.exec_module(module)
 return module

def make_estimands(cells:np.ndarray,stored:np.ndarray)->dict[str,np.ndarray]:
 cell_order=[('TS11-D','Q0','CPU_COMMON'),('TS11-D','Q1','CPU_COMMON'),
  ('STATE-CTRL','Q0','CPU_COMMON'),('STATE-CTRL','Q1','CPU_COMMON'),
  ('TS11-D','Q0','CUDA_COMMON'),('TS11-D','Q1','CUDA_COMMON'),
  ('STATE-CTRL','Q0','CUDA_COMMON'),('STATE-CTRL','Q1','CUDA_COMMON')]
 stored_order=[('TS11-D','Q0'),('TS11-D','Q1'),('STATE-CTRL','Q0'),('STATE-CTRL','Q1')]
 ci={k:i for i,k in enumerate(cell_order)};si={k:i for i,k in enumerate(stored_order)}
 out={}
 for key,i in ci.items():out['L/'+ '/'.join(key)]=cells[...,i]
 for model in ('TS11-D','STATE-CTRL'):
  q0cpu=cells[...,ci[(model,'Q0','CPU_COMMON')]]
  q1cpu=cells[...,ci[(model,'Q1','CPU_COMMON')]]
  q0cuda=cells[...,ci[(model,'Q0','CUDA_COMMON')]]
  q1cuda=cells[...,ci[(model,'Q1','CUDA_COMMON')]]
  bq0=q0cuda-q0cpu;bq1=q1cuda-q1cpu
  gcpu=q0cpu-q1cpu;gcuda=q0cuda-q1cuda
  mixed=q0cpu-q1cuda
  oldstored=stored[...,si[(model,'Q0')]]
  hadstored=stored[...,si[(model,'Q1')]]
  driftold=q0cpu-oldstored;drifthad=q1cuda-hadstored
  out['B/'+model+'/Q0']=bq0;out['B/'+model+'/Q1']=bq1
  out['G/'+model+'/CPU_COMMON']=gcpu;out['G/'+model+'/CUDA_COMMON']=gcuda
  out['G_CUDA_MINUS_CPU/'+model]=gcuda-gcpu
  out['G_MIXED_COMMON/'+model]=mixed
  out['DRIFT_OLD/'+model]=driftold;out['DRIFT_HAD/'+model]=drifthad
  out['G_MIXED_STORED/'+model]=mixed-driftold+drifthad
 for query in ('Q0','Q1'):
  for backend in ('CPU_COMMON','CUDA_COMMON'):
   state=cells[...,ci[('STATE-CTRL',query,backend)]]
   ts=cells[...,ci[('TS11-D',query,backend)]]
   out['V/'+query+'/'+backend]=state-ts
 for backend in ('CPU_COMMON','CUDA_COMMON'):
  gts=cells[...,ci[('TS11-D','Q0',backend)]]-cells[...,ci[('TS11-D','Q1',backend)]]
  gsc=cells[...,ci[('STATE-CTRL','Q0',backend)]]-cells[...,ci[('STATE-CTRL','Q1',backend)]]
  out['I/'+backend]=gts-gsc
 return out

def main()->int:
 gate=json.loads((ROOT/'qa/N2_PREDICTION_ACCEPTANCE.json').read_text())
 if gate.get('status')!='PASS_24_OF_24_TARGET_BLIND_PREDICTIONS' or int(gate.get('accepted_views',-1))!=24:
  raise RuntimeError('refuse target scoring until all 24 target-blind predictions are accepted')
 if int(gate.get('final_test_access',-1))!=0 or gate.get('target_arrays_loaded_by_inference') is not False:
  raise RuntimeError('prediction acceptance violates final-test/target-blind lock')
 if sha(MATRIX)!=N2_MATRIX_SHA:raise RuntimeError('N2 matrix hash differs from locked matrix')
 if sha(N1_METRIC_SOURCE)!=N1_METRIC_SHA:raise RuntimeError('N1 evaluator hash differs from locked evaluator')
 scorer=load_n1_metrics()
 metric_names=list(scorer.METRICS)
 with REQUEST.open(newline='',encoding='utf-8') as f:request_rows=list(csv.DictReader(f))
 with MATRIX.open(newline='',encoding='utf-8') as f:matrix=list(csv.DictReader(f))
 with (ROOT/'results/N2_VIEW_ACCEPTANCE.csv').open(newline='',encoding='utf-8') as f:accept_rows=list(csv.DictReader(f))
 if len(request_rows)!=12 or len(matrix)!=24 or len(accept_rows)!=24:raise RuntimeError('locked N2 inputs must contain 12 requests and 24 accepted views')
 request_by={(r['model'],r['query'],r['seed']):r for r in request_rows}
 acceptance_by={r['view_id']:r for r in accept_rows}
 # First load and validate every accepted prediction. Targets remain unopened through this gate.
 with np.load(INPUTS,allow_pickle=False) as archive:
  split=archive['split_code'].astype(np.int8,copy=True)
  ids_all=archive['anchor_id'].astype('U96',copy=True)
  episodes_all=archive['episode_index'].astype(np.int64,copy=True)
 dev_idx=np.flatnonzero(split==1);dev_ids=ids_all[dev_idx]
 if len(dev_idx)!=2322:raise RuntimeError('canonical dev split row count mismatch')
 group_map={}
 with GROUP_MAP.open(newline='',encoding='utf-8') as f:
  for x in csv.DictReader(f):group_map[int(x['episode_index'])]=(x['split'],x['source_group_id'])
 dev_eps=episodes_all[dev_idx]
 if any(group_map[int(e)][0]!='dev' for e in dev_eps):raise RuntimeError('dev episode split identity mismatch')
 dev_groups=np.asarray([group_map[int(e)][1] for e in dev_eps],dtype='U128')
 group_names=sorted(np.unique(dev_groups).tolist())
 if len(group_names)!=603:raise RuntimeError('accepted dev source-group count mismatch')
 prediction_by_cell={}
 metadata_by_cell={}
 for row in matrix:
  vid=row['view_id'];acc=acceptance_by.get(vid)
  if acc is None or acc.get('status')!='PASS':raise RuntimeError('missing accepted prediction row '+vid)
  key=(row['model'],row['query'],row['profile'],int(row['seed']))
  req=request_by[(row['model'],row['query'],row['seed'])]
  prediction_path=Path(acc['prediction_path']);sidecar=Path(acc['sidecar_path'])
  expected_query=req['accepted_query_table_sha256'] if row['query']=='Q1' else None
  pred,meta=scorer.load_prediction(prediction_path,sidecar,vid,'dev',dev_ids,req['checkpoint_sha256'],expected_query)
  prediction_by_cell[key]=pred;metadata_by_cell[key]=meta
 if len(prediction_by_cell)!=24:raise RuntimeError('not all 24 accepted prediction arrays loaded')
 # Authorization point: record the bounded, previously exposed development target access.
 input_sha=sha(INPUTS);target_norm_sha=sha(TARGET_NORMALIZER);group_map_sha=sha(GROUP_MAP)
 access={'schema_version':1,'status':'AUTHORIZED_EXPLORATORY_DEV_SCORING',
  'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'prediction_acceptance_sha256':sha(ROOT/'qa/N2_PREDICTION_ACCEPTANCE.json'),
  'prediction_matrix_acceptance_sha256':sha(ROOT/'results/N2_VIEW_ACCEPTANCE.csv'),
  'target_input_path':str(INPUTS),'input_sha256':input_sha,'target_normalizer_sha256':target_norm_sha,
  'dev_target_rows_read':2322,'dev_episodes':603,'dev_source_groups':603,
  'dev_previously_exposed':True,'purpose':'post-prediction paired N2 development scoring and profile/backend decomposition',
  'training_targets_read':0,'final_test_access':0,'heldout_access':0}
 access_path=ROOT/'qa/SCORE_TARGET_ACCESS.jsonl'
 with access_path.open('a',encoding='utf-8') as f:f.write(json.dumps(access,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 # Only after every prediction/sidecar passed above, open development targets.
 target_norm=json.loads(TARGET_NORMALIZER.read_text(encoding='utf-8'))
 sigma=np.asarray(target_norm['target_sigma'],dtype=np.float32)
 safe=np.asarray(target_norm['safe_fd_scale'],dtype=np.float32)
 threshold=json.loads((ROOT/'n1_refs/accepted_n1_refs/MOTION_ACTIVITY_THRESHOLD.json').read_text(encoding='utf-8'))
 active_threshold=np.asarray(threshold['direction_threshold_vector'],dtype=np.float32)
 if sigma.shape!=(28,) or safe.shape!=(28,) or active_threshold.shape!=(28,):raise RuntimeError('N1 scorer normalizer/activity vectors must be 28D')
 with np.load(INPUTS,allow_pickle=False) as archive:
  dev_targets=archive['targets_native'][dev_idx,:30].astype(np.float32,copy=True)
  dev_anchors=archive['anchor_command_native'][dev_idx].astype(np.float32,copy=True)
 if dev_targets.shape!=(2322,30,28) or dev_anchors.shape!=(2322,28):raise RuntimeError('dev scoring target/anchor shape mismatch')
 # Score all 24 common-profile predictions with the exact locked N1 metric and group reducers.
 scores={};n2_vectors={};n2_rows=[];group_rows=[]
 for row in matrix:
  key=(row['model'],row['query'],row['profile'],int(row['seed']))
  pred=prediction_by_cell[key]
  metrics=scorer.metric_arrays(pred,dev_targets,dev_anchors,sigma,safe,active_threshold)
  summary=scorer.group_summary(metrics,dev_eps,dev_groups)
  for metric in metric_names:
   scores[(row['model'],row['query'],row['profile'],int(row['seed']),metric)]=float(summary.get(metric,float('nan')))
   n2_rows.append({'view_id':row['view_id'],'model':row['model'],'query':row['query'],'seed':int(row['seed']),
    'profile':row['profile'],'metric':metric,'group_mean':summary.get(metric,float('nan')),
    'valid_groups':summary.get(metric+'_valid_groups',603),'n_anchors':2322,'n_episodes':603,'n_source_groups':603,
    'dev_evaluation':'exploratory_previously_exposed','final_test_access':0})
  for metric in MAIN_METRICS:
   names,values=scorer.nested_group_values(metrics[metric],dev_eps,dev_groups)
   if names!=group_names or len(values)!=603:raise RuntimeError('N2 group vector incomplete/order mismatch '+row['view_id']+'/'+metric)
   n2_vectors[(row['model'],row['query'],row['profile'],int(row['seed']),metric)]=values
   scores[(row['model'],row['query'],row['profile'],int(row['seed']),metric)]=float(np.mean(values))
   for g,v in zip(names,values):
    group_rows.append({'view_id':row['view_id'],'model':row['model'],'query':row['query'],'seed':int(row['seed']),
     'profile':row['profile'],'metric':metric,'source_group_id':g,'group_value':float(v)})
 # Score the already accepted N1 stored predictions for profile-drift decomposition.
 stored_scores={};stored_vectors={};stored_meta={};stored_rows=[];stored_group_rows=[];profile_rows=[]
 for req in request_rows:
  model,query,seed=req['model'],req['query'],int(req['seed'])
  pred_path=Path(req['accepted_n1_dev_prediction_path'])
  side_path=Path(req['accepted_profile_sidecar_path'])
  if sha(pred_path)!=req['accepted_n1_dev_prediction_sha256'] or sha(side_path)!=req['accepted_profile_sidecar_sha256']:
   raise RuntimeError('accepted N1 stored prediction/profile reference SHA mismatch '+req['model_id'])
  expected_query=req['accepted_query_table_sha256'] if query=='Q1' else None
  pred,meta=scorer.load_prediction(pred_path,side_path,req['checkpoint_source_run_id'],'dev',dev_ids,req['checkpoint_sha256'],expected_query)
  runtime=meta.get('runtime',{})
  profile_info={'device':meta.get('device'),'logical_device':meta.get('logical_device'),
   'physical_gpu':meta.get('physical_gpu'),'cuda_visible_devices':meta.get('cuda_visible_devices'),
   'dtype':meta.get('dtype'),'batch_size':meta.get('batch_size'),'runtime':runtime}
  profile_rows.append({'model':model,'query':query,'seed':seed,'stored_profile':'as_run',
   'stored_device':meta.get('device'),'stored_physical_gpu':meta.get('physical_gpu'),
   'profile_sidecar_path':str(side_path),'profile_sidecar_sha256':req['accepted_profile_sidecar_sha256'],
   'prediction_path':str(pred_path),'prediction_sha256':req['accepted_n1_dev_prediction_sha256'],
   'runtime_json':json.dumps(profile_info,sort_keys=True,ensure_ascii=False),'status':'PASS'})
  metrics=scorer.metric_arrays(pred,dev_targets,dev_anchors,sigma,safe,active_threshold)
  summary=scorer.group_summary(metrics,dev_eps,dev_groups)
  for metric in metric_names:
   stored_rows.append({'model':model,'query':query,'seed':seed,'stored_profile':'as_run','metric':metric,
    'group_mean':summary.get(metric,float('nan')),'valid_groups':summary.get(metric+'_valid_groups',603),
    'n_source_groups':603,'final_test_access':0})
  for metric in MAIN_METRICS:
   names,values=scorer.nested_group_values(metrics[metric],dev_eps,dev_groups)
   if names!=group_names or len(values)!=603:raise RuntimeError('stored N1 group vector mismatch '+req['model_id']+'/'+metric)
   stored_vectors[(model,query,seed,metric)]=values
   stored_scores[(model,query,seed,metric)]=float(np.mean(values))
   for g,v in zip(names,values):
    stored_group_rows.append({'model':model,'query':query,'seed':seed,'stored_profile':'as_run',
     'metric':metric,'source_group_id':g,'group_value':float(v)})
 # Reconcile stored N1 predictions to the already accepted N1 table using the exact locked evaluator.
 n1_table_path=ROOT/'n1_refs/accepted_n1_refs/TRAIN_DEV_BY_SEED.csv'
 with n1_table_path.open(newline='',encoding='utf-8') as f:n1_table=list(csv.DictReader(f))
 n1_by={(x['config'],x['query_variant'],int(x['seed'])):x for x in n1_table if x.get('split')=='dev' and x.get('config')!='PERSISTENCE'}
 recon=[]
 for req in request_rows:
  query_variant='old' if req['query']=='Q0' else 'had'
  ref=n1_by[(req['model'],query_variant,int(req['seed']))]
  for metric in MAIN_METRICS:
   actual=stored_scores[(req['model'],req['query'],int(req['seed']),metric)]
   expected=float(ref[metric]);delta=actual-expected
   recon.append({'model':req['model'],'query':req['query'],'seed':int(req['seed']),'metric':metric,
    'n1_accepted_score':expected,'recomputed_stored_prediction_score':actual,'absolute_difference':abs(delta),
    'tolerance':1e-12,'status':'PASS' if abs(delta)<=1e-12 else 'MISMATCH'})
 max_recon=max(x['absolute_difference'] for x in recon)
 if max_recon>1e-12:raise RuntimeError('N1 stored prediction scorer reconciliation exceeded 1e-12')
 # Prepare score and contrast tables.
 cell_order=[('TS11-D','Q0','CPU_COMMON'),('TS11-D','Q1','CPU_COMMON'),
  ('STATE-CTRL','Q0','CPU_COMMON'),('STATE-CTRL','Q1','CPU_COMMON'),
  ('TS11-D','Q0','CUDA_COMMON'),('TS11-D','Q1','CUDA_COMMON'),
  ('STATE-CTRL','Q0','CUDA_COMMON'),('STATE-CTRL','Q1','CUDA_COMMON')]
 stored_order=[('TS11-D','Q0'),('TS11-D','Q1'),('STATE-CTRL','Q0'),('STATE-CTRL','Q1')]
 contrast_seed={};backend_rows=[];effect_rows=[];decomp_rows=[]
 def cell_score(model,query,backend,seed,metric):
  return scores[(model,query,backend,seed,metric)]
 for metric in metric_names:
  seed_cells=np.empty((3,8),dtype=np.float64);seed_stored=np.empty((3,4),dtype=np.float64)
  for si,seed in enumerate(SEEDS):
   for ci,(model,query,backend) in enumerate(cell_order):
    seed_cells[si,ci]=cell_score(model,query,backend,seed,metric)
   for sj,(model,query) in enumerate(stored_order):
    seed_stored[si,sj]=stored_scores[(model,query,seed,metric)] if metric in MAIN_METRICS else float(
      next(x['group_mean'] for x in stored_rows if x['model']==model and x['query']==query and x['seed']==seed and x['metric']==metric))
  est=make_estimands(seed_cells,seed_stored)
  for name,values in est.items():contrast_seed[(metric,name)]=np.asarray(values,dtype=np.float64)
  for seed in SEEDS:
   # all backend and profile drift terms per fixed seed
   for model in ('TS11-D','STATE-CTRL'):
    for query in ('Q0','Q1'):
     cpu=cell_score(model,query,'CPU_COMMON',seed,metric);cuda=cell_score(model,query,'CUDA_COMMON',seed,metric)
     B=cuda-cpu;stored=next(x['group_mean'] for x in stored_rows if x['model']==model and x['query']==query and x['seed']==seed and x['metric']==metric)
     drift=(cpu-stored) if query=='Q0' else (cuda-stored)
     asrun=next(x for x in profile_rows if x['model']==model and x['query']==query and x['seed']==seed)
     backend_rows.append({'model':model,'query':query,'seed':seed,'metric':metric,
      'L_CPU_COMMON':cpu,'L_CUDA_COMMON':cuda,'B_CUDA_minus_CPU':B,'N1_stored_as_run_score':stored,
      'N1_stored_device':asrun['stored_device'],'N1_stored_physical_gpu':asrun['stored_physical_gpu'],
      'profile_drift_common_minus_stored':drift,'drift_scope':'CPU profile drift' if query=='Q0' else 'CUDA profile and physical-device drift',
      'n_source_groups':603})
   for backend in ('CPU_COMMON','CUDA_COMMON'):
    gts=cell_score('TS11-D','Q0',backend,seed,metric)-cell_score('TS11-D','Q1',backend,seed,metric)
    gsc=cell_score('STATE-CTRL','Q0',backend,seed,metric)-cell_score('STATE-CTRL','Q1',backend,seed,metric)
    interaction=gts-gsc
    effect_rows.extend([
     {'estimand':'G','model':'TS11-D','query':'Q0-minus-Q1','backend':backend,'seed':seed,'metric':metric,'value':gts,'n_source_groups':603},
     {'estimand':'G','model':'STATE-CTRL','query':'Q0-minus-Q1','backend':backend,'seed':seed,'metric':metric,'value':gsc,'n_source_groups':603},
     {'estimand':'I','model':'TS11-D-minus-STATE-CTRL','query':'query-gain-interaction','backend':backend,'seed':seed,'metric':metric,'value':interaction,'n_source_groups':603}])
    for query in ('Q0','Q1'):
     v=cell_score('STATE-CTRL',query,backend,seed,metric)-cell_score('TS11-D',query,backend,seed,metric)
     effect_rows.append({'estimand':'V','model':'STATE-CTRL-minus-TS11-D','query':query,'backend':backend,'seed':seed,'metric':metric,'value':v,'n_source_groups':603})
   for model in ('TS11-D','STATE-CTRL'):
    q0cpu=cell_score(model,'Q0','CPU_COMMON',seed,metric);q1cpu=cell_score(model,'Q1','CPU_COMMON',seed,metric)
    q0cuda=cell_score(model,'Q0','CUDA_COMMON',seed,metric);q1cuda=cell_score(model,'Q1','CUDA_COMMON',seed,metric)
    bq0=q0cuda-q0cpu;bq1=q1cuda-q1cpu
    g_cpu=q0cpu-q1cpu;g_cuda=q0cuda-q1cuda
    mixed_common=q0cpu-q1cuda
    drift_old=q0cpu-next(x['group_mean'] for x in stored_rows if x['model']==model and x['query']=='Q0' and x['seed']==seed and x['metric']==metric)
    drift_had=q1cuda-next(x['group_mean'] for x in stored_rows if x['model']==model and x['query']=='Q1' and x['seed']==seed and x['metric']==metric)
    mixed_stored=mixed_common-drift_old+drift_had
    terms=np.asarray([g_cuda,g_cpu,bq0,bq1,mixed_common,drift_old,drift_had,mixed_stored],dtype=np.float64)
    identity_status='INSUFFICIENT_SUPPORT'
    if np.isfinite(terms).all():
     if not np.isclose(g_cuda-g_cpu,bq0-bq1,atol=1e-12,rtol=1e-10):raise RuntimeError('G_CUDA-G_CPU identity failed')
     if not np.isclose(mixed_common,g_cpu-bq1,atol=1e-12,rtol=1e-10) or not np.isclose(mixed_common,g_cuda-bq0,atol=1e-12,rtol=1e-10):raise RuntimeError('mixed-common decomposition identity failed')
     identity_status='PASS'
    decomp_rows.append({'model':model,'seed':seed,'metric':metric,'B_Q0':bq0,'B_Q1':bq1,
     'G_CPU':g_cpu,'G_CUDA':g_cuda,'G_CUDA_minus_G_CPU':g_cuda-g_cpu,'B_Q0_minus_B_Q1':bq0-bq1,
     'G_mixed_common':mixed_common,'drift_old':drift_old,'drift_had':drift_had,'G_mixed_stored':mixed_stored,
     'identity_check_status':identity_status,'n_source_groups':603})
 # Group-level tables for the two co-primary metrics, preserving the paired unit.
 n2_group_rows=[];stored_group_rows=[];contrast_group_rows=[]
 for metric in MAIN_METRICS:
  for model,query,backend in cell_order:
   for seed in SEEDS:
    vals=n2_vectors[(model,query,backend,seed,metric)]
    for g,v in zip(group_names,vals):
     n2_group_rows.append({'model':model,'query':query,'profile':backend,'seed':seed,'metric':metric,'source_group_id':g,'group_value':float(v)})
  for model,query in stored_order:
   for seed in SEEDS:
    vals=stored_vectors[(model,query,seed,metric)]
    for g,v in zip(group_names,vals):
     stored_group_rows.append({'model':model,'query':query,'seed':seed,'metric':metric,'source_group_id':g,'group_value':float(v)})
  for seed in SEEDS:
   cell_arr=np.stack([n2_vectors[(m,q,b,seed,metric)] for m,q,b in cell_order],axis=0)
   stored_arr=np.stack([stored_vectors[(m,q,seed,metric)] for m,q in stored_order],axis=0)
   group_ests=make_estimands(cell_arr.T,stored_arr.T)
   for name,values in group_ests.items():
    if name.startswith(('B/','G/','V/','I/','G_CUDA_MINUS_CPU/','G_MIXED_COMMON/','DRIFT_OLD/','DRIFT_HAD/','G_MIXED_STORED/')):
     for g,v in zip(group_names,values):
      contrast_group_rows.append({'estimand':name,'seed':seed,'metric':metric,'source_group_id':g,'group_value':float(v)})
 # Reproducible shared paired source-group bootstrap, conditional on the fixed three seeds.
 rng=np.random.default_rng(BOOTSTRAP_SEED)
 G=len(group_names)
 draws=rng.integers(0,G,size=(BOOTSTRAP_REPLICATES,G),endpoint=False)
 seed_draws=rng.integers(0,len(SEEDS),size=(BOOTSTRAP_REPLICATES,len(SEEDS)),endpoint=False)
 draws_sha=sha_array(draws);seed_draws_sha=sha_array(seed_draws)
 draw_path=ROOT/'protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz'
 with draw_path.open('wb') as f:np.savez_compressed(f,source_group_ids=np.asarray(group_names,dtype='U128'),
  group_draw_indices=draws,seed_draw_indices=seed_draws,prng_seed=np.asarray(BOOTSTRAP_SEED),
  replicates=np.asarray(BOOTSTRAP_REPLICATES))
 boot_rows=[]
 for metric in MAIN_METRICS:
  arr=np.stack([[n2_vectors[(m,q,b,seed,metric)] for m,q,b in cell_order] for seed in SEEDS],axis=0)
  st=np.stack([[stored_vectors[(m,q,seed,metric)] for m,q in stored_order] for seed in SEEDS],axis=0)
  boot_cells=np.empty((len(SEEDS),len(cell_order),BOOTSTRAP_REPLICATES),dtype=np.float64)
  boot_stored=np.empty((len(SEEDS),len(stored_order),BOOTSTRAP_REPLICATES),dtype=np.float64)
  for si in range(len(SEEDS)):
   for ci in range(len(cell_order)):boot_cells[si,ci,:]=arr[si,ci,draws].mean(axis=1)
   for sj in range(len(stored_order)):boot_stored[si,sj,:]=st[si,sj,draws].mean(axis=1)
  fixed_cells=boot_cells.mean(axis=0).T;fixed_stored=boot_stored.mean(axis=0).T
  cross_cells=np.empty((BOOTSTRAP_REPLICATES,len(cell_order)),dtype=np.float64)
  cross_stored=np.empty((BOOTSTRAP_REPLICATES,len(stored_order)),dtype=np.float64)
  for b in range(BOOTSTRAP_REPLICATES):
   cross_cells[b,:]=boot_cells[seed_draws[b],:,b].mean(axis=0)
   cross_stored[b,:]=boot_stored[seed_draws[b],:,b].mean(axis=0)
  boot_est=make_estimands(fixed_cells,fixed_stored)
  cross_est=make_estimands(cross_cells,cross_stored)
  seed_est=make_estimands(np.stack([[scores[(m,q,b,seed,metric)] for m,q,b in cell_order] for seed in SEEDS]),
   np.stack([[stored_scores[(m,q,seed,metric)] for m,q in stored_order] for seed in SEEDS]))
  for name,dist in boot_est.items():
   vals=np.asarray(seed_est[name],dtype=np.float64)
   boot_rows.append({'metric':metric,'estimand':name,'point_estimate_mean_fixed_seeds':float(np.mean(vals)),
    'seed_sample_sd':float(np.std(vals,ddof=1)),'seed_median':float(np.median(vals)),
    'group_bootstrap_replicates':BOOTSTRAP_REPLICATES,'prng_seed':BOOTSTRAP_SEED,'group_unit':'paired source group',
    'group_draw_matrix_sha256':draws_sha,'group_ci95_low':float(np.quantile(dist,.025)),
    'group_ci95_high':float(np.quantile(dist,.975)),
    'crossed_seed_group_ci95_low_low_precision':float(np.quantile(cross_est[name],.025)),
    'crossed_seed_group_ci95_high_low_precision':float(np.quantile(cross_est[name],.975)),
    'interval_scope':'conditional on these 12 fixed checkpoints and previously exposed development set; crossed seed interval is low precision',
    'p_value':'NOT_COMPUTED','final_test_access':0})
 # All seed means, sample SD, medians and all three LOO means for every metric/estimand.
 allseed_rows=[]
 for metric in metric_names:
  seed_cells=np.stack([[scores[(m,q,b,seed,metric)] for m,q,b in cell_order] for seed in SEEDS])
  seed_stored=np.stack([[next(x['group_mean'] for x in stored_rows if x['model']==m and x['query']==q and x['seed']==seed and x['metric']==metric)
   for m,q in stored_order] for seed in SEEDS])
  est=make_estimands(seed_cells,seed_stored)
  for name,vals in est.items():
   row={'metric':metric,'estimand':name,'mean_three_fixed_seeds':float(np.mean(vals)),
    'sample_sd_ddof1':float(np.std(vals,ddof=1)),'median_three_fixed_seeds':float(np.median(vals)),
    'final_test_access':0}
   for i,seed in enumerate(SEEDS):row['seed_'+str(seed)]=float(vals[i])
   for i,seed in enumerate(SEEDS):row['loo_excluding_'+str(seed)]=float(np.mean(np.delete(vals,i)))
   allseed_rows.append(row)
 # Save output tables to isolated staging and commit only after all metric and decomposition gates pass.
 stage=ROOT/'tmp'/('n2_analysis_stage_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 stage.mkdir()
 tables={
  'N2_BY_SEED.csv':n2_rows,'N1_STORED_REFERENCE_BY_SEED.csv':stored_rows,
  'N1_STORED_PROFILE_REFERENCE.csv':profile_rows,'BACKEND_PROFILE_DRIFT.csv':backend_rows,
  'QUERY_SYSTEM_EFFECTS.csv':effect_rows,'PROFILE_DRIFT_DECOMPOSITION.csv':decomp_rows,
  'N2_GROUP_LEVEL_METRICS.csv':n2_group_rows,'N1_STORED_GROUP_LEVEL_METRICS.csv':stored_group_rows,
  'N2_GROUP_CONTRASTS.csv':contrast_group_rows,'BOOTSTRAP_CI.csv':boot_rows,'ALL_SEED_LOO.csv':allseed_rows,
  'N1_STORED_SCORE_RECONCILIATION.csv':recon,'PREDICTION_BACKEND_DIFFERENCES.csv':[]}
 # Prediction absolute differences, one aligned CPU/CUDA pair per fixed model/query/seed.
 diff_rows=[]
 for model in ('TS11-D','STATE-CTRL'):
  for query in ('Q0','Q1'):
   for seed in SEEDS:
    cpu=prediction_by_cell[(model,query,'CPU_COMMON',seed)]
    cuda=prediction_by_cell[(model,query,'CUDA_COMMON',seed)]
    diff=np.abs(cuda-cpu);den=np.maximum(np.maximum(np.abs(cuda),np.abs(cpu)),1e-8);reldiff=diff/den
    diff_rows.append({'model':model,'query':query,'seed':seed,'rows':2322,'elements':int(diff.size),
     'mean_abs_difference':float(np.mean(diff)),'max_abs_difference':float(np.max(diff)),
     'p99_abs_difference':float(np.quantile(diff,.99)),'mean_relative_difference_absdenom_1e-8':float(np.mean(reldiff)),
     'p99_relative_difference_absdenom_1e-8':float(np.quantile(reldiff,.99)),'n_source_groups':603})
 tables['PREDICTION_BACKEND_DIFFERENCES.csv']=diff_rows
 for name,rows in tables.items():write_csv(stage/name,rows)
 n1_recon={'status':'PASS' if max_recon<=1e-12 else 'MISMATCH','rows':len(recon),
  'max_abs_difference':max_recon,'tolerance':1e-12,'metric_source_sha256':sha(N1_METRIC_SOURCE),
  'accepted_n1_table_sha256':sha(n1_table_path)}
 if n1_recon['status']!='PASS':raise RuntimeError('N1 stored-score reconciliation gate failed')
 for f in sorted(stage.iterdir()):
  dest=ROOT/'results'/f.name
  if dest.exists():raise FileExistsError('refuse overwrite results file '+str(dest))
  dest.parent.mkdir(parents=True,exist_ok=True)
 for f in sorted(stage.iterdir()):os.replace(f,ROOT/'results'/f.name)
 analysis={'schema_version':1,'status':'PASS_N2_SCORING',
  'n2_prediction_acceptance_sha256':sha(ROOT/'qa/N2_PREDICTION_ACCEPTANCE.json'),
  'n1_metric_source_path':str(N1_METRIC_SOURCE),'n1_metric_source_sha256':sha(N1_METRIC_SOURCE),
  'source_group_map_sha256':group_map_sha,'target_normalizer_sha256':target_norm_sha,
  'direction_threshold_sha256':sha(THRESHOLD),'input_sha256':input_sha,
  'target_access_log_sha256':sha(access_path),'dev_target_rows_read':2322,'source_groups':603,
  'metric_count':len(metric_names),'co_primary_metrics':list(MAIN_METRICS),
  'N1_stored_score_reconciliation':n1_recon,
  'bootstrap':{'replicates':BOOTSTRAP_REPLICATES,'prng_seed':BOOTSTRAP_SEED,
   'group_draw_matrix_sha256':draws_sha,'seed_draw_matrix_sha256':seed_draws_sha,
   'group_draws_path':str(draw_path),'unit':'paired source group',
   'same_group_draws_for_all_24_views_and_all_co_primary_metrics':True,
   'fixed_seed_scope':list(SEEDS),'p_values':'NOT_COMPUTED'},
  'all_seed_summaries':'mean, sample SD ddof=1, median, and all 3 leave-one-seed-out means recorded',
  'final_test_access':0,'independent_confirmation':False,
  'training':False,'optimizer_steps':0,'encoder_extraction':0,'N3_rescoring':0}
 write_json(ROOT/'qa/N1_STORED_SCORE_RECONCILIATION.json',n1_recon)
 write_json(ROOT/'qa/ANALYSIS_ACCEPTANCE.json',analysis)
 exposure={**access,'status':'SCORED_EXPLORATORY_DEV_AFTER_24_VIEW_ACCEPTANCE',
  'analysis_acceptance_sha256':sha(ROOT/'qa/ANALYSIS_ACCEPTANCE.json'),
  'output_table_count':len(tables),'n1_stored_score_reconciliation':'PASS',
  'final_test_access':0,'training':False,'N3_rescoring':0}
 write_json(ROOT/'qa/OUTCOME_EXPOSURE.json',exposure)
 print(json.dumps({'status':analysis['status'],'N1_stored_score_reconciliation':n1_recon,
  'metrics':metric_names,'co_primary_metrics':MAIN_METRICS,'bootstrap_replicates':BOOTSTRAP_REPLICATES,
  'bootstrap_seed':BOOTSTRAP_SEED,'source_groups':603,'tables':len(tables),'final_test_access':0},sort_keys=True),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
