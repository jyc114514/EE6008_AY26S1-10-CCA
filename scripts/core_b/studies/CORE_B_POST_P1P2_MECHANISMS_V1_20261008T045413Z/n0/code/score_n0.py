#!/usr/bin/env python3
"""Branchwise separate N0 dev scorer; targets open only after bundle acceptance."""
from __future__ import annotations
import os
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
N0_ROOT=HERE.parent
NEXT_ROOT=N0_ROOT.parent
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
P1_ROOT=PROJECT/'reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z'
V3_ROOT=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
MODELS=('F-POOL-D','ALT-POOL-R3D18')
SEEDS=(20260950,20260951,20260952)
CONDITIONS=('REAL','HIST','ZERO','PERM')

def sha_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_csv(path:Path,rows:list[dict[str,object]])->None:
 if path.exists():raise FileExistsError(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:raise RuntimeError(f'no rows for {path.name}')
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:row.get(k,'') for k in fields})

def load_prediction(path:Path,expected_sha:str,ids:np.ndarray)->np.ndarray:
 if not path.is_file() or sha_file(path)!=expected_sha:raise RuntimeError(f'prediction hash mismatch: {path}')
 with np.load(path,allow_pickle=False) as z:
  if {'targets_native','future_targets'} & set(z.files):raise RuntimeError('prediction bundle contains target/future keys')
  if not {'anchor_id','predictions_native'}.issubset(z.files):raise RuntimeError(f'prediction bundle schema missing keys: {path}')
  got=z['anchor_id'].astype(str);pred=z['predictions_native'].astype(np.float32,copy=True)
 lookup={str(value):i for i,value in enumerate(got)}
 wanted=ids.astype(str).tolist()
 if len(lookup)!=len(got) or set(lookup)!=set(wanted):raise RuntimeError(f'prediction ID mismatch: {path}')
 pred=pred[np.asarray([lookup[x] for x in wanted],dtype=np.int64)]
 if pred.shape!=(2322,30,28) or not np.isfinite(pred).all():raise RuntimeError(f'prediction shape/nonfinite: {path}')
 return pred

def mean_by_episode(values:np.ndarray,episodes:np.ndarray)->dict[int,float]:
 return {int(e):float(np.mean(values[episodes==e])) for e in np.unique(episodes)}

def group_by_episode(values:dict[int,float],episodes:np.ndarray,groups:np.ndarray)->dict[str,float]:
 gby={}
 for e,g in zip(episodes,groups):gby.setdefault(str(g),[]).append(values[int(e)])
 return {g:float(np.mean(v)) for g,v in gby.items()}

def equal_mean(values:dict[int,float])->float:return float(np.mean(list(values.values())))

def main()->int:
 acceptance_path=NEXT_ROOT/'qa/PREDICTION_ACCEPTANCE.json'
 if not acceptance_path.is_file():raise RuntimeError('N0 scorer is sealed until prediction acceptance exists')
 acceptance=json.loads(acceptance_path.read_text(encoding='utf-8'))
 if acceptance.get('status') not in ('PASS_ALL_6_REAL_REPLAYS_AND_12_INTERVENTIONS','PARTIAL_N0_ACCEPTANCE'):
  raise RuntimeError('N0 acceptance has an invalid status')
 accepted_real={(r['model'],int(r['seed'])):r for r in acceptance.get('real_replays',[])}
 accepted_probe={(r['model'],int(r['seed']),r['condition']):r for r in acceptance.get('interventions',[])}
 branches=[(m,s) for m in MODELS for s in SEEDS if (m,s) in accepted_real and any((m,s,c) in accepted_probe for c in ('ZERO','PERM'))]
 if not branches:raise RuntimeError('no complete model-seed branch is available for development scoring')
 # Target opening occurs only below the 6+12 (or explicit partial branchwise) prediction gate.
 with np.load(V3_ROOT/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as source:
  split=source['split_code'].copy();rows=np.flatnonzero(split==1)
  if len(rows)!=2322 or set(np.unique(split).tolist())!={0,1}:raise RuntimeError('primary dev/train-only split mismatch')
  ids=source['anchor_id'][rows].astype(str).copy();episodes=source['episode_index'][rows].astype(np.int64,copy=True)
  tasks=source['original_task_index'][rows].astype(np.int64,copy=True)
  target=source['targets_native'][rows,:30].astype(np.float32,copy=True)
 if target.shape!=(2322,30,28) or not np.isfinite(target).all():raise RuntimeError(f'dev target shape/nonfinite: {target.shape}')
 norm=json.loads((V3_ROOT/'data/V3_NORMALIZERS.json').read_text(encoding='utf-8'))
 sigma=np.asarray(norm['target_sigma'],dtype=np.float32)
 if sigma.shape!=(28,) or not np.isfinite(sigma).all() or np.any(sigma<=0):raise RuntimeError('target_sigma schema invalid')
 group_map={}
 with (P1_ROOT/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r['split']=='dev':group_map[int(r['episode_index'])]=r['source_group_id']
 groups=np.asarray([group_map[int(e)] for e in episodes],dtype='U64')
 if len(np.unique(groups))!=603:raise RuntimeError('P1 dev source-group support mismatch')
 primary=list(csv.DictReader((P1_ROOT/'predictions/PRIMARY_MANIFEST.csv').open(newline='',encoding='utf-8')))
 rows_output=[];rows_error=[];rows_level=[];rows_groups=[]
 for model,seed in branches:
  real_record=accepted_real[(model,seed)]
  real_path=NEXT_ROOT/real_record['bundle_path']
  real=load_prediction(real_path,real_record['bundle_sha256'],ids)
  real_error=np.abs(real.astype(np.float64)-target.astype(np.float64))/sigma[None,None,:]
  real_anchor=real_error.mean(axis=(1,2));real_episode=mean_by_episode(real_anchor,episodes)
  real_group=group_by_episode(real_episode,episodes,groups)
  hist_row=next(r for r in primary if r['condition']=='S-HIST' and int(r['seed'])==seed)
  hist_path=Path(hist_row['bundle_path']);hist_path=hist_path if hist_path.is_absolute() else P1_ROOT/hist_path
  if sha_file(hist_path)!=hist_row['bundle_sha256']:raise RuntimeError('accepted S-HIST bundle hash changed')
  hist=load_prediction(hist_path,hist_row['bundle_sha256'],ids)
  hist_error=np.abs(hist.astype(np.float64)-target.astype(np.float64))/sigma[None,None,:]
  hist_anchor=hist_error.mean(axis=(1,2));hist_episode=mean_by_episode(hist_anchor,episodes)
  hist_group=group_by_episode(hist_episode,episodes,groups)
  for method,anchor_error,episode_error,group_error in [('REAL',real_anchor,real_episode,real_group),('HIST',hist_anchor,hist_episode,hist_group)]:
   rows_error.append({'model':model,'seed':seed,'condition':method,'anchor_equal_mae_std':float(anchor_error.mean()),'episode_equal_mae_std':equal_mean(episode_error),
    'source_group_equal_mae_std':float(np.mean(list(group_error.values()))),'deltaL_episode_equal_vs_REAL':0.0 if method=='REAL' else equal_mean(episode_error)-equal_mean(real_episode),
    'source_groups':len(group_error),'episodes':len(episode_error),'anchors':len(anchor_error),'tasks':int(len(np.unique(tasks))),'final_test_access':0})
   for g,value in group_error.items():
    rows_groups.append({'model':model,'seed':seed,'condition':method,'source_group_id':g,'group_equal_mae_std':value,'D_out':None,'D_level':None,'D_shape':None,'final_test_access':0})
  for condition in ('ZERO','PERM'):
   record=accepted_probe.get((model,seed,condition))
   if not record:continue
   path=NEXT_ROOT/record['bundle_path'];probe=load_prediction(path,record['bundle_sha256'],ids)
   diff=(probe.astype(np.float64)-real.astype(np.float64))/sigma[None,None,:]
   dout=np.abs(diff).mean(axis=(1,2));dlevel=np.abs(diff.mean(axis=1)).mean(axis=1);dshape=np.abs(diff-diff.mean(axis=1,keepdims=True)).mean(axis=(1,2))
   err=np.abs(probe.astype(np.float64)-target.astype(np.float64))/sigma[None,None,:]
   anchor_error=err.mean(axis=(1,2));delta_anchor=anchor_error-real_anchor
   ep_dout=mean_by_episode(dout,episodes);ep_dlevel=mean_by_episode(dlevel,episodes);ep_dshape=mean_by_episode(dshape,episodes)
   ep_error=mean_by_episode(anchor_error,episodes);ep_delta=mean_by_episode(delta_anchor,episodes)
   g_dout=group_by_episode(ep_dout,episodes,groups);g_dlevel=group_by_episode(ep_dlevel,episodes,groups);g_dshape=group_by_episode(ep_dshape,episodes,groups)
   g_err=group_by_episode(ep_error,episodes,groups);g_real=group_by_episode(real_episode,episodes,groups);g_delta=group_by_episode(ep_delta,episodes,groups)
   for i in range(len(ids)):
    rows_output.append({'model':model,'seed':seed,'condition':condition,'anchor_id':ids[i],'episode_index':int(episodes[i]),'task':int(tasks[i]),'source_group_id':str(groups[i]),
     'D_out':float(dout[i]),'D_level':float(dlevel[i]),'D_shape':float(dshape[i]),'real_mae_anchor':float(real_anchor[i]),'probe_mae_anchor':float(anchor_error[i]),'deltaL_anchor':float(delta_anchor[i]),'final_test_access':0})
   for metric_name,vals,epis,gv in [('D_out',dout,ep_dout,g_dout),('D_level',dlevel,ep_dlevel,g_dlevel),('D_shape',dshape,ep_dshape,g_dshape)]:
    rows_level.append({'model':model,'seed':seed,'condition':condition,'metric':metric_name,'anchor_mean':float(vals.mean()),'episode_equal_mean':equal_mean(epis),
     'anchor_median':float(np.median(vals)),'anchor_q05':float(np.quantile(vals,.05)),'anchor_q95':float(np.quantile(vals,.95)),
     'source_group_equal_mean':float(np.mean(list(gv.values()))),'final_test_access':0})
   rows_error.append({'model':model,'seed':seed,'condition':condition,'anchor_equal_mae_std':float(anchor_error.mean()),'episode_equal_mae_std':equal_mean(ep_error),
    'source_group_equal_mae_std':float(np.mean(list(g_err.values()))),'deltaL_episode_equal_vs_REAL':equal_mean(ep_delta),
    'deltaL_source_group_equal_vs_REAL':float(np.mean(list(g_delta.values()))),'D_out_episode_equal':equal_mean(ep_dout),'D_level_episode_equal':equal_mean(ep_dlevel),
    'D_shape_episode_equal':equal_mean(ep_dshape),'source_groups':len(g_err),'episodes':len(ep_error),'anchors':len(anchor_error),'tasks':int(len(np.unique(tasks))),'final_test_access':0})
   for g in sorted(g_err):
    rows_groups.append({'model':model,'seed':seed,'condition':condition,'source_group_id':g,'group_equal_mae_std':g_err[g],'real_group_equal_mae_std':g_real[g],
     'deltaL_group':g_delta[g],'D_out':g_dout[g],'D_level':g_dlevel[g],'D_shape':g_dshape[g],'final_test_access':0})
 summaries=[]
 for model in MODELS:
  for condition in CONDITIONS:
   accepted_seeds=sorted(s for m,s in branches if m==model and (condition in ('REAL','HIST') or (m,s,condition) in accepted_probe))
   values=[next((r['episode_equal_mae_std'] for r in rows_error if r['model']==model and int(r['seed'])==seed and r['condition']==condition),None) for seed in accepted_seeds]
   values=[v for v in values if v is not None]
   delta_values=[next(r['deltaL_episode_equal_vs_REAL'] for r in rows_error if r['model']==model and int(r['seed'])==seed and r['condition']==condition) for seed in accepted_seeds]
   summaries.append({'model':model,'condition':condition,'accepted_seed_numbers':json.dumps(accepted_seeds),'accepted_seed_count':len(values),'expected_seed_count':3,
    'seed_values_episode_equal_mae':json.dumps(values),'mean_over_accepted_fixed_seeds':float(np.mean(values)) if values else None,
    'sample_sd_across_accepted_fixed_seeds':float(statistics.stdev(values)) if len(values)>1 else None,
    'deltaL_vs_REAL_seed_values':json.dumps(delta_values),'mean_deltaL_vs_REAL':float(np.mean(delta_values)) if delta_values else None,
    'display_delta_not_margin':0.0022313516970703788,'CI_computed':False,'p_value_computed':False,'final_test_access':0})
 if not rows_error:raise RuntimeError('no N0 development score rows produced')
 outdir=N0_ROOT/'results';outdir.mkdir(parents=True,exist_ok=True)
 outputs=[('N0_OUTPUT_DEPENDENCE.csv',rows_output),('N0_ERROR_SHIFTS.csv',rows_error),('N0_LEVEL_SHAPE.csv',rows_level),
          ('N0_GROUP_DISTRIBUTIONS.csv',rows_groups),('N0_FIXED_SEED_SUMMARY.csv',summaries)]
 for name,rows_out in outputs:write_csv(outdir/name,rows_out)
 status='PASS_EXPLORATORY_N0_SCORING' if acceptance.get('status')=='PASS_ALL_6_REAL_REPLAYS_AND_12_INTERVENTIONS' else 'PARTIAL_EXPLORATORY_N0_SCORING'
 record={'status':status,'created_utc':datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'),
  'target_access_stage':'separate scorer after branchwise prediction acceptance','target_source':str(V3_ROOT/'data/V3_MAIN_INPUTS.npz'),
  'target_array':'targets_native rows split_code==1, first 30 steps, action[0:28]','rows':2322,'episodes':603,'source_groups':603,'tasks':int(len(np.unique(tasks))),
  'target_arrays_opened_by_inference':False,'target_arrays_opened_by_this_scorer':True,'final_heldout_access':0,'CI_or_p_value_computed':False,
  'accepted_probe_bundles_scored':len(rows_output),'accepted_model_seed_branches_scored':len(branches),'acceptance_status':acceptance['status'],
  'effect_direction':'positive DeltaL means probe MAE exceeds REAL; D_out/level/shape are descriptive and not additive.',
  'outputs':[f'results/{x[0]}' for x in outputs]}
 (N0_ROOT/'results/N0_SCORE_QA.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':status,'anchor_rows':len(rows_output),'source_group_rows':len(rows_groups),'summary_rows':len(summaries),'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
