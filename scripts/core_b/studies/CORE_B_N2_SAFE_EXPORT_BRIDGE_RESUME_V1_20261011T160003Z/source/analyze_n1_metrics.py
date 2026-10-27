#!/usr/bin/env python3
"""Separate paired train/dev scoring after all prediction bundles are accepted."""
from __future__ import annotations
import os
import csv,hashlib,json,math,os
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
EQ=PROJECT/'reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z'
SCHED=PROJECT/'reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z'
MECH=PROJECT/'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'
EPS=1e-8
OUTPUT=ROOT
SEEDS=(20261001,20261002,20261003)
CONDITIONS=(('TS11-D','old'),('TS11-D','had'),('STATE-CTRL','old'),('STATE-CTRL','had'))
METRICS=('position_standardized_mae','safe_fd_normalized','native_future_internal_fd','endpoint_native_mae','endpoint_standardized_mae',
         'seam_error_native','seam_pred_step_abs','seam_gt_step_abs','early_position_standardized_mae','mid_position_standardized_mae',
         'late_position_standardized_mae','early_safe_fd_normalized','mid_safe_fd_normalized','late_safe_fd_normalized')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_csv(path:Path,rows:list[dict[str,Any]]):
 if not rows:raise ValueError(f'refusing to write empty CSV {path}')
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=[]
 for row in rows:
  for k in row:
   if k not in fields:fields.append(k)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)
def write_json(path:Path,value:dict[str,Any]):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def begin_staging()->Path:
 parent=ROOT/'tmp';parent.mkdir(parents=True,exist_ok=True);indices=[]
 for p in parent.glob('analysis_attempt_*'):
  try:indices.append(int(p.name.rsplit('_',1)[1]))
  except ValueError:pass
 staging=parent/f'analysis_attempt_{max(indices,default=0)+1:03d}'/'payload'
 staging.mkdir(parents=True,exist_ok=False);return staging

def commit_staging(staging:Path)->int:
 files=sorted(p for p in staging.rglob('*') if p.is_file())
 if not files:raise RuntimeError('analysis produced no staged files')
 for src in files:
  dst=ROOT/src.relative_to(staging)
  if dst.exists():raise FileExistsError(f'preserving preexisting analysis output: {dst}')
  dst.parent.mkdir(parents=True,exist_ok=True)
 for src in files:os.replace(src,ROOT/src.relative_to(staging))
 return len(files)

def nested_group_values(values:np.ndarray,episodes:np.ndarray,groups:np.ndarray)->tuple[list[str],np.ndarray]:
 values=np.asarray(values,dtype=np.float64);episodes=np.asarray(episodes,dtype=np.int64);groups=np.asarray(groups,dtype='U128')
 out=[];names=[]
 for group in np.unique(groups):
  epmeans=[]
  for ep in np.unique(episodes[groups==group]):
   rows=(groups==group)&(episodes==ep);v=values[rows];v=v[np.isfinite(v)]
   if len(v):epmeans.append(float(np.mean(v)))
  if epmeans:names.append(str(group));out.append(float(np.mean(epmeans)))
 return names,np.asarray(out,dtype=np.float64)

def metric_arrays(pred:np.ndarray,target:np.ndarray,anchor:np.ndarray,sigma:np.ndarray,safe:np.ndarray,active_threshold:np.ndarray)->dict[str,np.ndarray]:
 pred=np.asarray(pred,dtype=np.float32);target=np.asarray(target,dtype=np.float32);anchor=np.asarray(anchor,dtype=np.float32)
 if pred.shape!=target.shape or pred.ndim!=3 or pred.shape[1:]!=(30,28):raise ValueError(f'metric array shape mismatch {pred.shape}/{target.shape}')
 error=pred-target;std_error=error/sigma[None,None,:]
 dp=np.diff(pred,axis=1);dy=np.diff(target,axis=1);de=dp-dy
 std_fd=np.abs(de)/safe[None,None,:]
 act=np.abs(dy)>active_threshold[None,None,:]
 near=np.abs(dy)<=EPS;small=(~act)&(~near)
 correct=(np.sign(dp)==np.sign(dy))&act
 active_count=act.sum(axis=(1,2));correct_count=correct.sum(axis=(1,2))
 direction=np.divide(correct_count,active_count,out=np.full(len(pred),np.nan,dtype=np.float64),where=active_count>0)
 tvp=np.abs(dp).sum(axis=1).mean(axis=1);tvy=np.abs(dy).sum(axis=1).mean(axis=1)
 tvratio=np.divide(tvp,tvy,out=np.full(len(pred),np.nan,dtype=np.float64),where=tvy>0)
 chunks=np.array_split(np.arange(30),3);out={
  'position_standardized_mae':np.abs(std_error).mean(axis=(1,2)),
  'safe_fd_normalized':std_fd.mean(axis=(1,2)),
  'native_future_internal_fd':np.abs(de).mean(axis=(1,2)),
  'endpoint_native_mae':np.abs(error[:,-1]).mean(axis=1),
  'endpoint_standardized_mae':np.abs(std_error[:,-1]).mean(axis=1),
  'seam_error_native':np.abs((pred[:,0]-anchor)-(target[:,0]-anchor)).mean(axis=1),
  'seam_pred_step_abs':np.abs(pred[:,0]-anchor).mean(axis=1),
  'seam_gt_step_abs':np.abs(target[:,0]-anchor).mean(axis=1),
  'tv_pred':tvp,'tv_gt':tvy,'tv_ratio_window':tvratio,
  'direction_accuracy_active':direction,'active_step_count':active_count.astype(np.int64),
  'direction_correct_count':correct_count.astype(np.int64),
  'near_zero_gt_step_count':near.sum(axis=(1,2)).astype(np.int64),
  'small_nonzero_gt_step_count':small.sum(axis=(1,2)).astype(np.int64),
  'zero_gt_tv_window':(tvy==0).astype(np.int64)}
 for label,idx in zip(('early','mid','late'),chunks):
  out[f'{label}_position_standardized_mae']=np.abs(std_error[:,idx]).mean(axis=(1,2))
  # For a time block, include the adjacent transition ending in that block; seam remains separate.
  fdidx=np.arange(max(int(idx[0])-1,0),int(idx[-1]))
  out[f'{label}_safe_fd_normalized']=std_fd[:,fdidx].mean(axis=(1,2))
 return out

def load_prediction(path:Path,sidecar:Path,run_id:str,split:str,expected_ids:np.ndarray,expected_checkpoint:str,expected_query:str|None):
 meta=read_json(sidecar)
 if meta.get('status')!='PASS' or meta.get('run_id')!=run_id or meta.get('split')!=split:raise AssertionError(f'unaccepted prediction sidecar {run_id}/{split}')
 if int(meta.get('rows',-1))!=len(expected_ids) or sha(path)!=meta.get('prediction_sha256'):raise AssertionError(f'prediction rows/hash mismatch {run_id}/{split}')
 if meta.get('checkpoint_sha256')!=expected_checkpoint:raise AssertionError(f'prediction checkpoint mismatch {run_id}/{split}')
 if expected_query is not None and meta.get('query_table_sha256')!=expected_query:raise AssertionError(f'prediction query mismatch {run_id}/{split}')
 with np.load(path,allow_pickle=False) as z:
  ids=z['anchor_id'].astype('U96');pred=z['predictions_native'].astype(np.float32,copy=True)
 if len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(expected_ids.tolist()):raise AssertionError(f'prediction ID set mismatch {run_id}/{split}')
 order={v:i for i,v in enumerate(ids.tolist())};pred=pred[np.asarray([order[v] for v in expected_ids.tolist()],dtype=np.int64)]
 if pred.shape!=(len(expected_ids),30,28) or not np.isfinite(pred).all():raise AssertionError(f'prediction shape/finite mismatch {run_id}/{split}')
 return pred,meta

def group_summary(metric:dict[str,np.ndarray],episodes:np.ndarray,groups:np.ndarray)->dict[str,Any]:
 result={}
 for name,v in metric.items():
  if name in ('active_step_count','direction_correct_count','near_zero_gt_step_count','small_nonzero_gt_step_count','zero_gt_tv_window'):
   result[name+'_total']=int(np.sum(v));continue
  group_ids,gm=nested_group_values(v,episodes,groups)
  result[name]=float(np.mean(gm)) if len(gm) and np.isfinite(gm).any() else float('nan')
  result[name+'_valid_groups']=int(np.isfinite(gm).sum())
 result['tv_ratio_aggregate']=result['tv_pred']/result['tv_gt'] if result.get('tv_gt',0)>0 else float('nan')
 result['direction_accuracy_active_pooled']=float(np.sum(metric['direction_correct_count'])/np.sum(metric['active_step_count'])) if np.sum(metric['active_step_count'])>0 else float('nan')
 result['direction_active_support_steps']=int(np.sum(metric['active_step_count']))
 result['direction_zero_support_anchors']=int(np.sum(metric['active_step_count']==0))
 result['n_anchors']=int(len(episodes));result['n_episodes']=int(len(np.unique(episodes)));result['n_source_groups']=int(len(np.unique(groups)))
 return result

def main():
 global OUTPUT
 OUTPUT=begin_staging()
 pred_accept=read_json(ROOT/'qa/PREDICTION_ACCEPTANCE.json')
 if pred_accept.get('status')!='PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES' or pred_accept.get('train_bundles_accepted')!='12/12' or pred_accept.get('dev_bundles_accepted')!='12/12':
  raise RuntimeError('scoring is gated until all 12 train and 12 dev prediction bundles are accepted')
 case_lock=read_json(ROOT/'qa/CASE_SELECTION_LOCKED.json')
 if case_lock.get('status')!='PASS_CASE_IDS_LOCKED_BEFORE_SCORING' or sha(ROOT/'qualitative/CASE_MANIFEST.csv')!=case_lock.get('case_manifest_sha256'):
  raise RuntimeError('dev case IDs are not fixed/hashed before scoring')
 input_path=V3/'data/V3_MAIN_INPUTS.npz'
 archive=np.load(input_path,allow_pickle=False)
 splitcode=archive['split_code'].astype(np.int8,copy=True);ids_all=archive['anchor_id'].astype('U96',copy=True);episodes_all=archive['episode_index'].astype(np.int64,copy=True)
 train_idx=np.flatnonzero(splitcode==0);dev_idx=np.flatnonzero(splitcode==1)
 if len(train_idx)!=17757 or len(dev_idx)!=2322:raise AssertionError('locked train/dev population mismatch')
 train_ids=ids_all[train_idx];dev_ids=ids_all[dev_idx]
 group_map={}
 with (ROOT/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:
  for x in csv.DictReader(f):group_map[int(x['episode_index'])]=(x['split'],x['source_group_id'])
 train_eps=episodes_all[train_idx];dev_eps=episodes_all[dev_idx]
 train_groups=np.asarray([group_map[int(e)][1] for e in train_eps],dtype='U128');dev_groups=np.asarray([group_map[int(e)][1] for e in dev_eps],dtype='U128')
 if set(train_groups.tolist())&set(dev_groups.tolist()):raise AssertionError('train/dev source group overlap')
 # Only now, after the prediction and case-selection gates, open the authorized scoring targets.
 target_norm=read_json(V3/'data/V3_NORMALIZERS.json');sigma=np.asarray(target_norm['target_sigma'],dtype=np.float32);safe=np.asarray(target_norm['safe_fd_scale'],dtype=np.float32)
 train_cache=np.load(EQ/'d3/train_targets_trainonly.npz',allow_pickle=False)
 cache_ids=train_cache['anchor_id'].astype('U96');cache_eps=train_cache['episode_index'].astype(np.int64);train_targets=train_cache['targets_native_60'][:,:30].astype(np.float32,copy=True);train_anchors=train_cache['anchor_command_native'].astype(np.float32,copy=True)
 if not np.array_equal(cache_ids,train_ids) or not np.array_equal(cache_eps,train_eps):raise AssertionError('train target cache ordering mismatch')
 dev_targets=archive['targets_native'][dev_idx,:30].astype(np.float32,copy=True);dev_anchors=archive['anchor_command_native'][dev_idx].astype(np.float32,copy=True)
 if dev_targets.shape!=(2322,30,28):raise AssertionError('dev target shape mismatch')
 access={'schema_version':1,'status':'AUTHORIZED_EXPLORATORY_DEV_SCORING',
  'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'prediction_acceptance_sha256':sha(ROOT/'qa/PREDICTION_ACCEPTANCE.json'),
  'case_selection_sha256':sha(ROOT/'qa/CASE_SELECTION_LOCKED.json'),'train_target_rows_read':17757,'dev_target_rows_read':2322,
  'dev_previously_exposed':True,'purpose':'post-prediction paired exploratory score; no model selection','final_test_access':0,'heldout_access':0}
 with (ROOT/'qa/SCORE_TARGET_ACCESS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(access,sort_keys=True)+'\n');f.flush()
 thresholds=read_json(ROOT/'protocol/MOTION_ACTIVITY_THRESHOLD.json');active_threshold=np.asarray(thresholds['direction_threshold_vector'],dtype=np.float32)
 if sigma.shape!=(28,) or safe.shape!=(28,) or active_threshold.shape!=(28,):raise AssertionError('locked normalizer/activity vector schema mismatch')
 matrix=list(csv.DictReader((ROOT/'protocol/MATRIX12.csv').open(newline='',encoding='utf-8')))
 reuse={x['q0_run_id']:x for x in csv.DictReader((ROOT/'protocol/REUSE_ELIGIBILITY.csv').open(newline='',encoding='utf-8'))}
 newpred={(x['run_id'],x['split']):x for x in pred_accept['bundle_records'] if x['source']=='new'}
 predictions={};records=[]
 for row in matrix:
  config=row['config'];seed=int(row['seed']);run=row['run_id'];query='old' if row['action']=='reuse' else 'had'
  if row['action']=='reuse':
   ref=reuse[run];src=ref['source_run_id'];checkpoint=ref['checkpoint_sha256']
  else:
   q1=read_json(ROOT/'runs/formal'/run/'RUN_COMPLETE.json');src=run;checkpoint=q1['checkpoint_last_sha256']
  for split,expected_ids in [('train',train_ids),('dev',dev_ids)]:
   if query=='old':
    path=SCHED/'predictions'/split/f'{src}.npz';side=SCHED/'predictions'/split/f'{src}.json';expected=ref[f'{split}_prediction_sha256'];
    if sha(path)!=expected:raise AssertionError(f'Q0 prediction source hash changed {src}/{split}')
   else:
    b=newpred[(run,split)];path=Path(b['prediction_path']);side=Path(b['sidecar_path']);expected=b['prediction_sha256']
    if sha(path)!=expected:raise AssertionError(f'Q1 prediction source hash changed {run}/{split}')
   pred,meta=load_prediction(path,side,src,split,expected_ids,checkpoint,(ROOT/'protocol/QUERY_TABLES_AND_HASHES.json' and read_json(ROOT/'protocol/QUERY_TABLES_AND_HASHES.json')['candidate_table_sha256_h30']) if query=='had' else None)
   predictions[(split,config,query,seed)]=pred
   records.append({'split':split,'config':config,'query_variant':query,'seed':seed,'run_id':run,'source_run_id':src,'prediction_path':str(path),'sidecar_path':str(side),'prediction_sha256':expected,'checkpoint_sha256':checkpoint,'rows':len(expected_ids),'status':'PASS'})
 # Persistence is deterministic and is scored against the exact same manifest.
 anchors={'train':train_anchors,'dev':dev_anchors};targets={'train':train_targets,'dev':dev_targets};episodes={'train':train_eps,'dev':dev_eps};groups={'train':train_groups,'dev':dev_groups}
 persistence={sp:np.repeat(anchors[sp][:,None,:],30,axis=1).astype(np.float32) for sp in ('train','dev')}
 metrics={};metric_rows=[];anchor_rows=[];episode_rows=[];action_rows=[];support_rows=[]
 all_groups={}
 for split in ('train','dev'):
  y=targets[split];anc=anchors[split];ep=episodes[split];grp=groups[split]
  for config,query,seed in [(x['config'],'old' if x['action']=='reuse' else 'had',int(x['seed'])) for x in matrix]:
   pred=predictions[(split,config,query,seed)]
   m=metric_arrays(pred,y,anc,sigma,safe,active_threshold);summary=group_summary(m,ep,grp)
   key=(split,config,query,seed);metrics[key]=m
   _,gvals=nested_group_values(m['position_standardized_mae'],ep,grp); # force nested reducer validation
   metric_rows.append({'split':split,'config':config,'query_variant':query,'seed':seed,'run_id':next(r['run_id'] for r in matrix if r['config']==config and r['seed']==str(seed) and (r['action']=='reuse' if query=='old' else r['action']=='train_had')),
                       **summary,'dev_evaluation':'exploratory_previously_exposed' if split=='dev' else 'in_sample_fit','final_test_access':0})
   # Numeric raw action slices, never anatomical aliases.
   for label,a,b in [('action[0:14]',0,14),('action[14:28]',14,28),('action[0:28]',0,28)]:
    pos=np.abs((pred[:,:,a:b]-y[:,:,a:b])/sigma[None,None,a:b]).mean(axis=(1,2))
    fd=np.abs(np.diff(pred[:,:,a:b],axis=1)-np.diff(y[:,:,a:b],axis=1)).mean(axis=(1,2))
    safe_slice=(np.abs(np.diff(pred[:,:,a:b],axis=1)-np.diff(y[:,:,a:b],axis=1))/safe[None,None,a:b]).mean(axis=(1,2))
    _,pvalues=nested_group_values(pos,ep,grp);_,fdvalues=nested_group_values(fd,ep,grp);_,safevalues=nested_group_values(safe_slice,ep,grp)
    action_rows.append({'split':split,'config':config,'query_variant':query,'seed':seed,'raw_action_slice':label,
                        'standardized_position_mae':float(np.mean(pvalues)),'native_future_internal_fd':float(np.mean(fdvalues)),
                        'safe_fd_normalized':float(np.mean(safevalues)),'anchors':len(ep),'source_groups':len(np.unique(grp)),
                        'final_test_access':0})
   # Motion/direction support by frozen train-defined group strata.
   motion=(np.abs(np.diff(y,axis=1))/sigma[None,None,:]).mean(axis=(1,2))
   group_motion={g:float(np.mean(motion[grp==g])) for g in np.unique(grp)}
   q25=float(thresholds['strata_thresholds']['30']['q25']);q75=float(thresholds['strata_thresholds']['30']['q75'])
   for stratum,predicate in [('low',lambda v:v<=q25),('mid',lambda v:q25<v<q75),('high',lambda v:v>=q75)]:
    chosen={g for g,v in group_motion.items() if predicate(v)};mask=np.asarray([g in chosen for g in grp],dtype=bool)
    if not np.any(mask):continue
    sm=group_summary({k:v[mask] for k,v in m.items()},ep[mask],grp[mask])
    support_rows.append({'split':split,'config':config,'query_variant':query,'seed':seed,'stratum':stratum,
                         'train_q25':q25,'train_q75':q75,'motion_threshold_fit_split':'train',**sm,'final_test_access':0})
 # Persistence summaries.
 for split in ('train','dev'):
  m=metric_arrays(persistence[split],targets[split],anchors[split],sigma,safe,active_threshold)
  sm=group_summary(m,episodes[split],groups[split])
  metric_rows.append({'split':split,'config':'PERSISTENCE','query_variant':'anchor_hold','seed':'ALL','run_id':'PERSISTENCE','status':'PASS',**sm,
                      'dev_evaluation':'exploratory_previously_exposed' if split=='dev' else 'in_sample_reference','final_test_access':0})

 # Three-seed descriptive summaries; seeds are not treated as additional source groups.
 for split in ('train','dev'):
  for metric in ('position_standardized_mae','safe_fd_normalized'):
   for model in ('TS11-D','STATE-CTRL'):
    vals=[byseed[(split,model,'old',s)][metric]-byseed[(split,model,'had',s)][metric] for s in SEEDS]
    gain_rows.append({'split':split,'model':model,'seed':'THREE_SEED_MEAN','metric':metric,'G_Q_old_minus_had':float(np.mean(vals)),
                      'seed_sample_sd':float(np.std(vals,ddof=1)),'n_fixed_seeds':3,'final_test_access':0})
   vold=[];vhad=[];inter=[]
   for s in SEEDS:
    tsold=byseed[(split,'TS11-D','old',s)][metric];tshad=byseed[(split,'TS11-D','had',s)][metric]
    scold=byseed[(split,'STATE-CTRL','old',s)][metric];schad=byseed[(split,'STATE-CTRL','had',s)][metric]
    vold.append(scold-tsold);vhad.append(schad-tshad);inter.append((tsold-tshad)-(scold-schad))
   video_rows.append({'split':split,'seed':'THREE_SEED_MEAN','metric':metric,'V_old_state_minus_TS':float(np.mean(vold)),
                      'V_old_seed_sample_sd':float(np.std(vold,ddof=1)),'V_Had_state_minus_TS':float(np.mean(vhad)),
                      'V_Had_seed_sample_sd':float(np.std(vhad,ddof=1)),
                      'positive_value_means':'TS11-D lower error than STATE-CTRL under the named query','final_test_access':0})
   interaction_rows.append({'split':split,'seed':'THREE_SEED_MEAN','metric':metric,'G_Q_TS11D':float(np.mean([byseed[(split,'TS11-D','old',s)][metric]-byseed[(split,'TS11-D','had',s)][metric] for s in SEEDS])),
                            'G_Q_STATE_CTRL':float(np.mean([byseed[(split,'STATE-CTRL','old',s)][metric]-byseed[(split,'STATE-CTRL','had',s)][metric] for s in SEEDS])),
                            'I_GTS_minus_GSTATE':float(np.mean(inter)),'I_seed_sample_sd':float(np.std(inter,ddof=1)),
                            'V_Had_minus_V_old':float(np.mean(vhad)-np.mean(vold)),'identity_check_pass':True,
                            'interpretation':'system interaction only; no video causal claim','final_test_access':0})

 # Paired anchor and episode differences; positive old-minus-Had favors Hadamard.
 for split in ('train','dev'):
  for config in ('TS11-D','STATE-CTRL'):
   for seed in SEEDS:
    old=metrics[(split,config,'old',seed)];had=metrics[(split,config,'had',seed)]
    for i,(aid,ep,grp) in enumerate(zip(ids_all[train_idx] if split=='train' else ids_all[dev_idx],episodes[split],groups[split])):
     anchor_rows.append({'split':split,'config':config,'seed':seed,'episode_index':int(ep),'source_group_id':str(grp),'anchor_id':str(aid),
       'old_position':float(old['position_standardized_mae'][i]),'had_position':float(had['position_standardized_mae'][i]),
       'G_Q_position_old_minus_had':float(old['position_standardized_mae'][i]-had['position_standardized_mae'][i]),
       'old_safe_fd':float(old['safe_fd_normalized'][i]),'had_safe_fd':float(had['safe_fd_normalized'][i]),
       'G_Q_safe_fd_old_minus_had':float(old['safe_fd_normalized'][i]-had['safe_fd_normalized'][i]),
       'old_native_fd':float(old['native_future_internal_fd'][i]),'had_native_fd':float(had['native_future_internal_fd'][i]),
       'G_Q_native_fd_old_minus_had':float(old['native_future_internal_fd'][i]-had['native_future_internal_fd'][i]),
       'old_tv_pred':float(old['tv_pred'][i]),'had_tv_pred':float(had['tv_pred'][i]),'gt_tv':float(old['tv_gt'][i]),
       'old_tv_ratio_window':float(old['tv_ratio_window'][i]),'had_tv_ratio_window':float(had['tv_ratio_window'][i]),
       'old_direction_accuracy_active':float(old['direction_accuracy_active'][i]),'had_direction_accuracy_active':float(had['direction_accuracy_active'][i]),
       'active_step_count_gt':int(old['active_step_count'][i]),'final_test_access':0})
    # Per-episode means nested before pairing; source groups stay explicit.
    for ep in np.unique(episodes[split]):
     mask=episodes[split]==ep;g=str(groups[split][np.flatnonzero(mask)[0]])
     row={'split':split,'config':config,'seed':seed,'episode_index':int(ep),'source_group_id':g}
     for name in ('position_standardized_mae','safe_fd_normalized','native_future_internal_fd','tv_pred','tv_gt','tv_ratio_window','direction_accuracy_active'):
      ov=old[name][mask];hv=had[name][mask];row[f'old_{name}']=float(np.nanmean(ov)) if np.isfinite(ov).any() else float('nan');row[f'had_{name}']=float(np.nanmean(hv)) if np.isfinite(hv).any() else float('nan')
      if name in ('position_standardized_mae','safe_fd_normalized','native_future_internal_fd'):
       row[f'G_Q_{name}_old_minus_had']=row[f'old_{name}']-row[f'had_{name}']
     row['anchors_in_episode']=int(mask.sum());row['final_test_access']=0;episode_rows.append(row)

 # Aggregate Q0/Q1 by seed, then across the three fixed seeds (mean and sample SD).
 byseed={(r['split'],r['config'],r['query_variant'],r['seed']):r for r in metric_rows if r.get('config')!='PERSISTENCE'}
 aggregate=[]
 for split in ('train','dev'):
  for config in ('TS11-D','STATE-CTRL'):
   for query in ('old','had'):
    items=[byseed[(split,config,query,s)] for s in SEEDS]
    out={'split':split,'config':config,'query_variant':query,'seed_count':3,'aggregation':'equal mean of fixed seed group-equal summaries',
         'dev_evaluation':'exploratory_previously_exposed' if split=='dev' else 'in_sample_fit','final_test_access':0}
    for name in METRICS+('tv_pred','tv_gt','tv_ratio_aggregate','direction_accuracy_active_pooled','direction_active_support_steps','active_step_count_total','near_zero_gt_step_count_total','small_nonzero_gt_step_count_total','zero_gt_tv_window_total'):
     vals=np.asarray([x.get(name,float('nan')) for x in items],dtype=np.float64);finite=vals[np.isfinite(vals)]
     out[f'{name}_mean']=float(np.mean(finite)) if len(finite) else float('nan');out[f'{name}_seed_sample_sd']=float(np.std(finite,ddof=1)) if len(finite)>1 else float('nan')
    aggregate.append(out)
 # Persistence is a deterministic reference, not a trained cell.
 for split in ('train','dev'):
  p_row=next(x for x in metric_rows if x.get('split')==split and x.get('config')=='PERSISTENCE')
  aggregate.append({'split':split,'config':'PERSISTENCE','query_variant':'anchor_hold','seed_count':0,'aggregation':'deterministic anchor persistence; no training',
                    **{k:v for k,v in p_row.items() if k not in ('run_id','seed','status','split','config','query_variant','dev_evaluation')}})
 # Gain, absolute video-system differences, and interaction.
 gain_rows=[];video_rows=[];interaction_rows=[]
 for split in ('train','dev'):
  for metric in ('position_standardized_mae','safe_fd_normalized'):
   for seed in SEEDS:
    tsold=byseed[(split,'TS11-D','old',seed)][metric];tshad=byseed[(split,'TS11-D','had',seed)][metric]
    scold=byseed[(split,'STATE-CTRL','old',seed)][metric];schad=byseed[(split,'STATE-CTRL','had',seed)][metric]
    gts=tsold-tshad;gsc=scold-schad;vold=scold-tsold;vhad=schad-tshad;interaction=gts-gsc
    if not np.isclose(interaction,vhad-vold,atol=1e-12,rtol=1e-8):raise AssertionError('I != V_Had - V_old')
    gain_rows.extend([{'split':split,'model':'TS11-D','seed':seed,'metric':metric,'G_Q_old_minus_had':gts,'n_source_groups':byseed[(split,'TS11-D','old',seed)]['n_source_groups'],'final_test_access':0},
                      {'split':split,'model':'STATE-CTRL','seed':seed,'metric':metric,'G_Q_old_minus_had':gsc,'n_source_groups':byseed[(split,'STATE-CTRL','old',seed)]['n_source_groups'],'final_test_access':0}])
    video_rows.append({'split':split,'seed':seed,'metric':metric,'V_old_state_minus_TS':vold,'V_Had_state_minus_TS':vhad,
                       'positive_means_TS11D_lower_error':True,'final_test_access':0})
    interaction_rows.append({'split':split,'seed':seed,'metric':metric,'G_Q_TS11D':gts,'G_Q_STATE_CTRL':gsc,'I_GTS_minus_GSTATE':interaction,
                             'V_Had_minus_V_old':vhad-vold,'identity_check_pass':True,'interpretation':'system interaction only; no video causal claim','final_test_access':0})
 # Paired source-group bootstrap on dev. One group-index matrix is shared across every cell, seed and co-primary endpoint.
 rng=np.random.default_rng(20261008);dev_group_names=sorted(np.unique(dev_groups).tolist());G=len(dev_group_names)
 if G!=603:raise AssertionError(f'dev bootstrap group count={G}')
 condition_keys=[('TS11-D','old'),('TS11-D','had'),('STATE-CTRL','old'),('STATE-CTRL','had')]
 draws=rng.integers(0,G,size=(10000,G),endpoint=False)
 seed_draws=rng.integers(0,3,size=(10000,3),endpoint=False)
 bootstrap_samples={};crossed_samples={}
 for metric in ('position_standardized_mae','safe_fd_normalized'):
  arr=np.empty((3,4,G),dtype=np.float64)
  for si,seed in enumerate(SEEDS):
   for ci,(config,query) in enumerate(condition_keys):
    names,vals=nested_group_values(metrics[('dev',config,query,seed)][metric],dev_eps,dev_groups)
    if names!=dev_group_names or len(vals)!=G:raise AssertionError('bootstrap group vector incomplete or group order mismatch')
    arr[si,ci,:]=vals
  boot={key:np.empty(10000,dtype=np.float64) for key in ('G_TS','G_STATE','V_old','V_Had','I')}
  cross={key:np.empty(10000,dtype=np.float64) for key in boot}
  for b,idx in enumerate(draws):
   per_seed=np.empty((3,4),dtype=np.float64)
   for si in range(3):
    for ci in range(4):per_seed[si,ci]=float(np.mean(arr[si,ci,idx]))
   means=per_seed.mean(axis=0)
   gts=means[0]-means[1];gstate=means[2]-means[3];vold=means[2]-means[0];vhad=means[3]-means[1]
   for key,value in {'G_TS':gts,'G_STATE':gstate,'V_old':vold,'V_Had':vhad,'I':gts-gstate}.items():boot[key][b]=value
   seed_means=np.empty((3,4),dtype=np.float64)
   for j,si in enumerate(seed_draws[b]):
    for ci in range(4):seed_means[j,ci]=float(np.mean(arr[int(si),ci,idx]))
   cm=seed_means.mean(axis=0);cgts=cm[0]-cm[1];cgstate=cm[2]-cm[3]
   for key,value in {'G_TS':cgts,'G_STATE':cgstate,'V_old':cm[2]-cm[0],'V_Had':cm[3]-cm[1],'I':cgts-cgstate}.items():cross[key][b]=value
  for key in boot:
   bootstrap_samples[(metric,key)]=boot[key];crossed_samples[(metric,key)]=cross[key]
 boot_rows=[]
 for metric in ('position_standardized_mae','safe_fd_normalized'):
  for key in ('G_TS','G_STATE','V_old','V_Had','I'):
   point_values=[]
   for seed in SEEDS:
    tsold=byseed[('dev','TS11-D','old',seed)][metric];tshad=byseed[('dev','TS11-D','had',seed)][metric]
    scold=byseed[('dev','STATE-CTRL','old',seed)][metric];schad=byseed[('dev','STATE-CTRL','had',seed)][metric]
    point_values.append({'G_TS':tsold-tshad,'G_STATE':scold-schad,'V_old':scold-tsold,'V_Had':schad-tshad,'I':(tsold-tshad)-(scold-schad)}[key])
   dist=bootstrap_samples[(metric,key)];crossdist=crossed_samples[(metric,key)]
   boot_rows.append({'split':'dev','metric':metric,'estimand':key,'point_estimate_mean_fixed_seeds':float(np.mean(point_values)),
    'seed_sample_sd':float(np.std(point_values,ddof=1)),'group_bootstrap_replicates':10000,'prng_seed':20261008,
    'group_bootstrap_ci95_low':float(np.quantile(dist,.025)),'group_bootstrap_ci95_high':float(np.quantile(dist,.975)),
    'crossed_seed_group_ci95_low_low_precision':float(np.quantile(crossdist,.025)),
    'crossed_seed_group_ci95_high_low_precision':float(np.quantile(crossdist,.975)),
    'bootstrap_unit':'source group; one shared draw matrix across all four systems, three fixed seeds and both primary metrics',
    'interval_scope':'conditional on fixed trained models and current previously exposed dev; not independent confirmation',
    'p_value':'NOT_COMPUTED','final_test_access':0})

 # Top ten improvements and deteriorations are automatic tail descriptions, never selection.
 top_rows=[]
 for config in ('TS11-D','STATE-CTRL'):
  for metric in ('position_standardized_mae','safe_fd_normalized'):
   groups_by_seed={}
   for seed in SEEDS:
    old=metrics[('dev',config,'old',seed)][metric];had=metrics[('dev',config,'had',seed)][metric]
    names,values=nested_group_values(old-had,dev_eps,dev_groups)
    groups_by_seed[seed]=dict(zip(names,values.astype(float).tolist()))
   summary={g:float(np.mean([groups_by_seed[s][g] for s in SEEDS])) for g in dev_group_names}
   for direction,vals in [('improvement',sorted(summary.items(),key=lambda x:x[1],reverse=True)[:10]),('deterioration',sorted(summary.items(),key=lambda x:x[1])[:10])]:
    for rank,(g,v) in enumerate(vals,1):top_rows.append({'config':config,'metric':metric,'direction':direction,'rank':rank,'source_group_id':g,
      'G_Q_old_minus_had_mean_across_three_fixed_seeds':v,'interpretation':'automatic tail description; not a case selection or model-selection rule','final_test_access':0})

 # Write analysis tables only after all predictions have passed acceptance.
 write_csv(OUTPUT/'results/TRAIN_DEV_BY_SEED.csv',metric_rows)
 write_csv(OUTPUT/'results/AGGREGATE_RESULTS.csv',aggregate)
 write_csv(OUTPUT/'results/QUERY_GAINS.csv',gain_rows)
 write_csv(OUTPUT/'results/ABSOLUTE_VIDEO_GAINS.csv',video_rows)
 write_csv(OUTPUT/'results/QUERY_MODEL_INTERACTION.csv',interaction_rows)
 write_csv(OUTPUT/'results/BOOTSTRAP_CI.csv',boot_rows)
 write_csv(OUTPUT/'results/ANCHOR_PAIRED.csv',anchor_rows)
 write_csv(OUTPUT/'results/EPISODE_PAIRED.csv',episode_rows)
 write_csv(OUTPUT/'results/ACTION_SLICE_METRICS.csv',action_rows)
 write_csv(OUTPUT/'results/MOTION_DIRECTION_SUPPORT.csv',support_rows)
 write_csv(OUTPUT/'results/TOP_DELTAS.csv',top_rows)
 # Training curves for Q1 are measured in the runner; Q0 objective/LR curves are reused from accepted logs.
 curve_rows=[]
 for row in matrix:
  run=row['run_id'];config=row['config'];seed=int(row['seed']);query='old' if row['action']=='reuse' else 'had'
  if query=='old':path=SCHED/'runs/formal'/row['source_run_id']/'EPOCH_LOG.csv'
  else:path=ROOT/'runs/formal'/run/'EPOCH_LOG.csv'
  for epoch in csv.DictReader(path.open(newline='',encoding='utf-8')):
   curve_rows.append({'run_id':run,'config':config,'query_variant':query,'seed':seed,'epoch':int(epoch['epoch']),
     'optimizer_steps_total':int(epoch['optimizer_steps_total']),'rows_seen':int(epoch['rows_seen']),
     'weighted_loss_sum_over_seen_rows':float(epoch['weighted_loss_sum_over_seen_rows']),
     'unweighted_mean_loss':float(epoch['unweighted_mean_loss']),'lr':float(epoch['lr']),
     'monitor_position_standardized_mae':float(epoch['monitor_position_standardized_mae']) if epoch.get('monitor_position_standardized_mae') else float('nan'),
     'monitor_safe_fd_normalized':float(epoch['monitor_safe_fd_normalized']) if epoch.get('monitor_safe_fd_normalized') else float('nan'),
     'monitor_tv_ratio':float(epoch['monitor_tv_ratio']) if epoch.get('monitor_tv_ratio') else float('nan'),
     'monitor_available':query=='had','monitor_scope':'fixed 128 train anchors; targets used in fitting' if query=='had' else 'not recorded in prior accepted Q0 run','final_test_access':0})
 write_csv(OUTPUT/'results/TRAIN_CURVES.csv',curve_rows)
 # Case trajectories, four systems, every fixed seed, raw numeric action indices only.
 case_rows=list(csv.DictReader((ROOT/'qualitative/CASE_MANIFEST.csv').open(newline='',encoding='utf-8')))
 case_data={}
 for case in case_rows:
  split='dev';aid=case['anchor_id'];i=int(np.flatnonzero(dev_ids==aid)[0])
  for config in ('TS11-D','STATE-CTRL'):
   for query in ('old','had'):
    for seed in SEEDS:
     cond_pred=predictions[(split,config,query,seed)][i]
     for action in (0,14):
      for step in range(30):
       case_data.setdefault((case['case_id'],action),[]).append({'case_id':case['case_id'],'stratum':case['stratum'],'source_group_id':case['source_group_id'],
        'episode_index':int(case['episode_index']),'anchor_id':aid,'config':config,'query_variant':query,'seed':seed,'future_step':step+1,
        'target_timestamp_seconds':float(case['anchor_timestamp_seconds'])+(step+1)/30.0,'raw_action_index':action,
        'target_native':float(dev_targets[i,step,action]),'prediction_native':float(cond_pred[step,action]),
        'final_test_access':0})
 for (case_id,action),rows in case_data.items():
  write_csv(OUTPUT/f'figure_data/case_{case_id}_action_{action}.csv',rows)

 # Count and aggregate acceptance summary.
 acc={'schema_version':1,'status':'PASS_PAIRED_TRAIN_DEV_ANALYSIS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'matrix_sha256':sha(ROOT/'protocol/MATRIX12.csv'),'prediction_acceptance_sha256':sha(ROOT/'qa/PREDICTION_ACCEPTANCE.json'),
  'train_target_rows_scored':17757,'dev_target_rows_scored':2322,'source_groups':{'train':2712,'dev':603},
  'episode_paired_rows':len(episode_rows),'anchor_paired_rows':len(anchor_rows),'bootstrap_replicates':10000,'bootstrap_prng_seed':20261008,
  'p_values_computed':False,'dev_exploratory_previously_exposed':True,'independent_confirmation':False,
  'new_dev_model_selection':False,'final_test_access':0,'heldout_access':0,
  'prediction_bundles_used':len(records),'persistence_scored':True,
  'interpretation':'All estimates use native Full28 targets and train-fitted normalizers. Query interaction is a system contrast, not a video causal effect.'}
 write_json(OUTPUT/'qa/ANALYSIS_ACCEPTANCE.json',acc)
 committed=commit_staging(OUTPUT)
 print(json.dumps({**acc,'committed_output_files':committed},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
