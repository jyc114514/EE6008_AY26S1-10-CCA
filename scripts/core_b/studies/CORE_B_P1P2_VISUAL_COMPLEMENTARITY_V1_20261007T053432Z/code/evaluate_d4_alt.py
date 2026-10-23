from __future__ import annotations
import os
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
LAB=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z'
RESULT=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z'
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
RAW=PROJECT/'data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d'
SIGMA=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
SEEDS=(20260950,20260951,20260952)
BODIES={'Hand':slice(0,14),'Arm':slice(14,28),'Full28':slice(0,28)}
N_BOOT=10000;BOOT_SEED=20261007

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 if p.exists():raise FileExistsError(p)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def pair_metrics(old,new,truth,sigma=SIGMA):
 if old.shape!=new.shape or old.shape!=truth.shape or old.ndim!=2 or old.shape[1]!=28 or not len(old):raise ValueError('D4 support arrays must share nonempty (T,28) shape')
 out={}
 for body,sl in BODIES.items():
  o=old[:,sl];n=new[:,sl];y=truth[:,sl];scale=sigma[sl]
  eo=np.abs(o-y)/scale[None,:];en=np.abs(n-y)/scale[None,:]
  out[f'{body}|L_old']=float(eo.mean());out[f'{body}|L_new']=float(en.mean());out[f'{body}|U']=float(eo.mean()-en.mean());out[f'{body}|R']=float(np.mean(np.abs(o-n)/scale[None,:]))
  out[f'{body}|native_MAE_old']=float(np.mean(np.abs(o-y)));out[f'{body}|native_MAE_new']=float(np.mean(np.abs(n-y)))
  out[f'{body}|FD_old']=float(np.mean(np.abs(np.diff(o,axis=0)-np.diff(y,axis=0)))) if len(o)>1 else float('nan')
  out[f'{body}|FD_new']=float(np.mean(np.abs(np.diff(n,axis=0)-np.diff(y,axis=0)))) if len(n)>1 else float('nan')
  out[f'{body}|endpoint_old']=float(np.mean(np.abs(o[-1]-y[-1])/scale));out[f'{body}|endpoint_new']=float(np.mean(np.abs(n[-1]-y[-1])/scale))
 return out

def main()->int:
 input_pairs=read_csv(RESULT/'d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv');time_rows=read_csv(RESULT/'d4/D4_PAIR_TIME_SUPPORT.csv')
 pairs={}
 for r in input_pairs:
  if r['horizon_steps']!='30':continue
  key=r['pair_id'];
  if key in pairs:
   old=pairs[key]
   for field in ('episode_index','old_dense_id','new_dense_id','common_start_frame','common_end_frame','common_target_steps','source_group_id'):
    if old[field]!=r[field]:raise RuntimeError(f'inconsistent duplicate D4 input pair {key}:{field}')
  else:pairs[key]=r
 support={}
 for r in time_rows:
  if r['pair_id'] in pairs:
   if r['pair_id'] in support:raise RuntimeError(f'duplicate D4 timestamp support {r["pair_id"]}')
   support[r['pair_id']]=r
 if set(pairs)!=set(support) or not pairs:raise RuntimeError('D4 H30 pair/time-support IDs differ')
 dense_manifest=EXP/'predictions/DENSE_MANIFEST.csv'
 if not dense_manifest.is_file():raise RuntimeError('D4 scoring blocked until dense ALT manifest is accepted')
 with dense_manifest.open(newline='',encoding='utf-8') as f:accepted_dense=list(csv.DictReader(f))
 accepted_alt=[r for r in accepted_dense if r.get('condition')=='ALT-POOL-R3D18' and r.get('horizon_steps')=='30' and r.get('status')=='accepted']
 if {int(r['seed']) for r in accepted_alt}!=set(SEEDS) or len(accepted_alt)!=3:raise RuntimeError('D4 scoring requires all three accepted ALT dense bundles')
 pair_list=[pairs[k] for k in sorted(pairs)]
 panel=[]
 with np.load(LAB/'d2/D2_DENSE_INPUTS.npz',allow_pickle=False) as z:
  panel={k:z[k].copy() for k in ('dense_id','episode_index','anchor_frame','anchor_timestamp')}
 panel_ids=panel['dense_id'].astype('U96').tolist();panel_index={x:i for i,x in enumerate(panel_ids)}
 if len(panel_ids)!=len(set(panel_ids)) or len(panel_ids)!=3718:raise RuntimeError('D4 dense panel ID contract mismatch')
 dense_meta={}
 for i,did in enumerate(panel_ids):dense_meta[did]=(int(panel['episode_index'][i]),int(panel['anchor_frame'][i]),float(panel['anchor_timestamp'][i]))
 episode_ids=sorted({int(r['episode_index']) for r in pair_list})
 source_by_episode={}
 parquet_by_episode={}
 for pair in pair_list:
  e=int(pair['episode_index']);sup=support[pair['pair_id']]
  g=pair['source_group_id']
  if e in source_by_episode and source_by_episode[e]!=g:raise RuntimeError('D4 source group varies within episode')
  source_by_episode[e]=g;parquet_by_episode[e]=sup['raw_target_parquet']
 if len(set(source_by_episode.values()))!=len(episode_ids):raise RuntimeError('D4 panel source groups are not one-to-one with episodes; nested aggregation required')
 # Load only the fixed development episodes listed in the D4 pair/time manifests.
 episodes={}
 for e in episode_ids:
  relative=Path(parquet_by_episode[e]);path=(RAW/relative).resolve()
  if not path.is_relative_to(RAW.resolve()):raise RuntimeError('D4 target path escapes G1 allowlist')
  table=pq.read_table(path,columns=['timestamp','action'])
  ts=table['timestamp'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
  action=np.asarray(table['action'].to_pylist(),dtype=np.float64)
  if action.ndim!=2 or action.shape[1]!=28 or len(ts)!=len(action) or not np.isfinite(action).all() or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):raise RuntimeError(f'raw dev target contract failed for episode {e}')
  episodes[e]=(ts,action)
 # Exact anchor-time checks for the fixed dense query panel used by H30 pairs.
 for pair in pair_list:
  e=int(pair['episode_index'])
  for did,frame_field in ((pair['old_dense_id'],'old_anchor_frame'),(pair['new_dense_id'],'new_anchor_frame')):
   if did not in dense_meta:raise RuntimeError(f'D4 pair references missing dense query {did}')
   ep,frame,timestamp=dense_meta[did]
   if ep!=e or frame!=int(pair[frame_field]):raise RuntimeError(f'D4 dense identity mismatch for {did}')
   if abs(timestamp-float(episodes[e][0][frame]))>1e-6:raise RuntimeError(f'D4 dense anchor time differs from raw Parquet: {did}')
 # Accepted historical dense metrics are the locked controls; do not regenerate them.
 accepted=read_csv(RESULT/'d4/D4_PAIR_METRICS.csv')
 control={}
 for r in accepted:
  if r['horizon_steps']=='30' and r['model'] in ('S-HIST','F-POOL-D') and r['body']=='Full28':
   control[(r['pair_id'],r['model'],int(r['seed']))]=r
 controls={m for _,m,_ in control}
 if controls!={'S-HIST','F-POOL-D'}:raise RuntimeError('accepted D4 H30 history/pooled controls incomplete')
 alt_by_seed={}
 for seed in SEEDS:
  path=EXP/'predictions/dense'/f'ALT-POOL-R3D18-H30-S{seed}.npz'
  if not path.is_file():raise RuntimeError(f'missing accepted ALT dense bundle {path}')
  with np.load(path,allow_pickle=False) as z:
   ids=z['source_id'].astype('U96');pred=z['predictions_native'].copy();eps=z['episode_index'].astype(np.int64);frames=z['anchor_frame'].astype(np.int64);times=z['anchor_timestamp'].astype(np.float64)
  if len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(panel_ids) or pred.shape!=(len(panel_ids),30,28) or not np.isfinite(pred).all():raise RuntimeError(f'ALT dense IDs/shape/finiteness mismatch seed={seed}')
  lookup={x:i for i,x in enumerate(ids)};aligned=np.empty((len(panel_ids),30,28),dtype=np.float32)
  for i,did in enumerate(panel_ids):
   j=lookup[did];ep,frame,timestamp=dense_meta[did]
   if int(eps[j])!=ep or int(frames[j])!=frame or abs(float(times[j])-timestamp)>1e-6:raise RuntimeError(f'ALT dense metadata mismatch seed={seed} ID={did}')
   aligned[i]=pred[j]
  alt_by_seed[seed]=(aligned,{x:i for i,x in enumerate(panel_ids)})
 alt_pair=[]
 for seed in SEEDS:
  pred,idx=alt_by_seed[seed]
  for pair in pair_list:
   pid=pair['pair_id'];e=int(pair['episode_index']);a=int(pair['old_anchor_frame']);b=int(pair['new_anchor_frame']);sup=support[pid]
   frames=np.asarray(json.loads(sup['common_target_frames']),dtype=np.int64);timestamps=np.asarray(json.loads(sup['common_target_timestamps']),dtype=np.float64)
   expected=np.arange(int(pair['common_start_frame']),int(pair['common_end_frame'])+1,dtype=np.int64)
   if not np.array_equal(frames,expected) or len(frames)!=int(pair['common_target_steps']) or not len(frames):raise RuntimeError(f'D4 fixed support differs for {pid}')
   if b-a not in (7,8):raise RuntimeError(f'D4 input interval is not the observed 7/8-frame gap: {pid}')
   ts,actions=episodes[e]
   if np.max(np.abs(ts[frames]-timestamps))>1e-6:raise RuntimeError(f'D4 target timestamp join mismatch for {pid}')
   old_i=idx[pair['old_dense_id']];new_i=idx[pair['new_dense_id']]
   old_offsets=frames-a-1;new_offsets=frames-b-1
   if np.any(old_offsets<0) or np.any(new_offsets<0) or np.any(old_offsets>=30) or np.any(new_offsets>=30):raise RuntimeError(f'D4 support outside native H30 forecast: {pid}')
   old_pred=pred[old_i,old_offsets];new_pred=pred[new_i,new_offsets];truth=actions[frames]
   m=pair_metrics(old_pred,new_pred,truth)
   gap=float(dense_meta[pair['new_dense_id']][2]-dense_meta[pair['old_dense_id']][2])
   row={'pair_id':pid,'task':int(pair['task']),'model':'ALT-POOL-R3D18','seed':seed,'horizon_steps':30,'episode_index':e,'source_group_id':pair['source_group_id'],'old_dense_id':pair['old_dense_id'],'new_dense_id':pair['new_dense_id'],'old_anchor_frame':a,'new_anchor_frame':b,'actual_anchor_interval_seconds':gap,'common_target_steps':len(frames),'old_remaining_lead_mean_seconds':float(np.mean(timestamps-dense_meta[pair['old_dense_id']][2])),'new_remaining_lead_mean_seconds':float(np.mean(timestamps-dense_meta[pair['new_dense_id']][2])),'common_start_frame':int(frames[0]),'common_end_frame':int(frames[-1]),'common_target_timestamps':json.dumps(timestamps.tolist()),**m,'final_test_access':0}
   alt_pair.append(row)
 # Join matched method U/L_new values on exact pair/time support, then aggregate pair->episode->video group.
 pair_index={(r['pair_id'],r['model'],int(r['seed'])):r for r in alt_pair}
 full_rows=[]
 for seed in SEEDS:
  for pair in pair_list:
   pid=pair['pair_id'];alt=pair_index[(pid,'ALT-POOL-R3D18',seed)]
   for model in ('S-HIST','F-POOL-D'):
    old=control.get((pid,model,seed))
    if old is None:raise RuntimeError(f'accepted D4 control row missing {pid} {model} {seed}')
    full_rows.append({'pair_id':pid,'seed':seed,'model':model,'task':int(pair['task']),'episode_index':int(pair['episode_index']),'source_group_id':pair['source_group_id'],'body':'Full28','U':float(old['U']),'L_old':float(old['L_old']),'L_new':float(old['L_new']),'R':float(old['R']),'FD_new':float(old['FD_new']),'common_target_steps':int(pair['common_target_steps']),'final_test_access':0})
   full_rows.append({'pair_id':pid,'seed':seed,'model':'ALT-POOL-R3D18','episode_index':int(pair['episode_index']),'source_group_id':pair['source_group_id'],'body':'Full28','U':float(alt['Full28|U']),'L_old':float(alt['Full28|L_old']),'L_new':float(alt['Full28|L_new']),'R':float(alt['Full28|R']),'FD_new':float(alt['Full28|FD_new']),'common_target_steps':int(alt['common_target_steps']),'final_test_access':0})
 # Source-group paired bootstrap; one group per episode is validated above.
 groups=sorted(set(r['source_group_id'] for r in pair_list));group_episode={r['source_group_id']:int(r['episode_index']) for r in pair_list}
 rng=np.random.default_rng(BOOT_SEED);draws=rng.integers(0,len(groups),size=(N_BOOT,len(groups)),dtype=np.int32)
 summary=[];contrasts=[];delta_display=float(json.loads((EXP/'protocol/ANALYSIS_LOCKED.json').read_text())['display_delta']['value'])
 grouped={}
 for r in full_rows:grouped[(r['model'],int(r['seed']),r['source_group_id'])]=grouped.get((r['model'],int(r['seed']),r['source_group_id']),[])+[r]
 group_stats={}
 for model in ('S-HIST','F-POOL-D','ALT-POOL-R3D18'):
  for seed in SEEDS:
   for metric in ('U','L_old','L_new','R','FD_new'):
    vals=[]
    for g in groups:
     rr=grouped[(model,seed,g)]
     vals.append(float(np.mean([float(x[metric]) for x in rr if np.isfinite(float(x[metric]))])))
    vals=np.asarray(vals,dtype=np.float64);group_stats[(model,seed,metric)]=vals
    boot=vals[draws].mean(axis=1)
    summary.append({'model':model,'seed':seed,'horizon_steps':30,'body':'Full28','metric':metric,'estimate_group_equal':float(vals.mean()),'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),'N_pair':len(pair_list),'N_episode':len(episode_ids),'N_source_group':len(groups),'N_task':len(set(int(r['task']) for r in pair_list)) if 'task' in pair_list[0] else None,'bootstrap_replicates':N_BOOT,'bootstrap_seed':BOOT_SEED,'final_test_access':0})
 for model in ('F-POOL-D','ALT-POOL-R3D18'):
  for seed in SEEDS:
   du=group_stats[(model,seed,'U')]-group_stats[('S-HIST',seed,'U')]
   dl=group_stats[(model,seed,'L_new')]-group_stats[('S-HIST',seed,'L_new')]
   boot_u=du[draws].mean(axis=1);boot_l=dl[draws].mean(axis=1)
   contrasts.append({'contrast':f'DeltaU_{model}_minus_S-HIST','seed':seed,'horizon_steps':30,'body':'Full28','delta_u':float(du.mean()),'delta_u_ci95_low':float(np.quantile(boot_u,.025)),'delta_u_ci95_high':float(np.quantile(boot_u,.975)),'delta_l_new':float(dl.mean()),'delta_l_new_ci95_low':float(np.quantile(boot_l,.025)),'delta_l_new_ci95_high':float(np.quantile(boot_l,.975)),'N_pair':len(pair_list),'N_episode':len(episode_ids),'N_source_group':len(groups),'N_task':len(set(int(r['task']) for r in pair_list)) if 'task' in pair_list[0] else None,'final_test_access':0})
 write_csv(EXP/'analysis/P2_UPDATE_ALT_PAIRS.csv',alt_pair)
 write_csv(EXP/'analysis/P2_UPDATE.csv',summary)
 write_csv(EXP/'analysis/P2_UPDATE_CONTRASTS.csv',contrasts)
 alt_contrasts=[r for r in contrasts if r['contrast']=='DeltaU_ALT-POOL-R3D18_minus_S-HIST']
 gate_directions={str(r['seed']):float(r['delta_u']) for r in alt_contrasts}
 gate_ci_low={str(r['seed']):float(r['delta_u_ci95_low']) for r in alt_contrasts}
 gate_lnew={str(r['seed']):float(r['delta_l_new']) for r in alt_contrasts}
 support_tasks=len(set(int(r['task']) for r in pair_list))
 gate_pass=(all(gate_directions[str(s)]>0 for s in SEEDS) and all(gate_ci_low[str(s)]>0 for s in SEEDS) and all(gate_lnew[str(s)]<=delta_display for s in SEEDS) and len(groups)>=50 and support_tasks>=5)
 gate={'status':'PASS_EXPLORATORY' if gate_pass else 'NOT_PASS','rule':'D4 DeltaU all 3 ALT seeds positive and group bootstrap 95% CI lower bound >0; ALT L_new no worse than history by more than display delta; >=50 groups and >=5 tasks','delta_display':delta_display,'seed_directions':gate_directions,'seed_ci95_lowers':gate_ci_low,'seed_delta_l_new':gate_lnew,'all_seed_directions_positive':all(gate_directions[str(s)]>0 for s in SEEDS),'all_ci_lower_positive':all(gate_ci_low[str(s)]>0 for s in SEEDS),'all_seed_l_new_within_delta':all(gate_lnew[str(s)]<=delta_display for s in SEEDS),'support_source_groups':len(groups),'support_tasks':support_tasks,'final_test_access':0}
 q=EXP/'analysis/P2_UPDATE_GATE.json'
 if q.exists():raise FileExistsError(q)
 q.write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','pair_rows':len(pair_list),'source_groups':len(groups),'contrasts':contrasts,'gate':gate,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
