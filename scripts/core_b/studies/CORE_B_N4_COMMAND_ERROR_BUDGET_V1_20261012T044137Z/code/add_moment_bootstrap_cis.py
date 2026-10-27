#!/usr/bin/env python3
"""Add locked paired-bootstrap intervals for T/P/C from saved N4 sufficient statistics only."""
import os
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
SEEDS=(20261001,20261002,20261003)
SYSTEMS=('TS11-D','STATE-CTRL')
QUERIES=('Q0','Q1')
SCOPES={'raw28':np.arange(28),'block0':np.arange(0,14),'block1':np.arange(14,28)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writej(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.part')
 t.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');t.replace(p)
def writecsv(p,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 t=p.with_name('.'+p.name+'.part')
 with t.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 t.replace(p)
def main():
 acceptance=readj(R/'qa/ANALYSIS_ACCEPTANCE.json')
 if acceptance.get('status')!='PASS_N4_ERROR_BUDGET_ANALYSIS':raise SystemExit('N4 analysis not accepted.')
 access=readj(R/'protocol/OUTCOME_EXPOSURE.json')
 if not access.get('n4_outcomes_opened'):raise SystemExit('N4 analysis targets were not previously accepted/opened.')
 draws_path=N2/'protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz'
 with np.load(draws_path,allow_pickle=False) as z:
  group_ids=z['source_group_ids'].astype('U128')
  draw_idx=np.asarray(z['group_draw_indices'],dtype=np.int64)
  seed_draw=np.asarray(z['seed_draw_indices'],dtype=np.int64)
  seed=int(z['prng_seed']);reps=int(z['replicates'])
 source_rows=readcsv(N2/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv')
 ep_group={int(x['episode_index']):x['source_group_id'] for x in source_rows if x['split']=='dev'}
 if reps!=10000 or seed!=20261011 or draw_idx.shape!=(10000,603) or seed_draw.shape!=(10000,3):
  raise SystemExit('N2 bootstrap shape/seed mismatch.')
 if not np.array_equal(group_ids,np.asarray(sorted(set(ep_group.values())),dtype='U128')):
  raise SystemExit('N2 draw source-group order mismatch.')
 counts=np.zeros(draw_idx.shape,dtype=np.int16)
 for i,row in enumerate(draw_idx):counts[i]=np.bincount(row,minlength=len(group_ids)).astype(np.int16)
 W=counts.astype(np.float64)/len(group_ids)
 base=R/'results/PER_WINDOW_MOMENTS'
 components=('target_energy_T','prediction_energy_P','cross_alignment_C')
 component_array={'target_energy_T':'target_energy_T_axis',
  'prediction_energy_P':'prediction_energy_P_axis','cross_alignment_C':'cross_alignment_C_axis'}
 additions=[]
 for scope,axes in SCOPES.items():
  for component in components:
   values=[]
   for s in SEEDS:
    for system in SYSTEMS:
     for q in QUERIES:
      model_id=f'{system}__{q}__S{s}__H30'
      path=base/f'{model_id}__dev.npz'
      with np.load(path,allow_pickle=False) as z:
       ep=np.asarray(z['episode_index'],dtype=np.int64)
       gids=np.asarray(z['source_group_id']).astype('U128')
       v=np.asarray(z[component_array[component]],dtype=np.float64)[:,axes].mean(axis=1)
      if len(v)!=2322 or not np.isfinite(v).all():raise SystemExit('Nonfinite/shape mismatch in saved moments '+str(path))
      group_values=[]
      for gid in group_ids:
       mask=gids==gid
       episodes=np.unique(ep[mask])
       if len(episodes)!=1:raise SystemExit('Expected one episode per dev source group.')
       group_values.append(float(np.mean(v[mask])))
      values.append(np.asarray(group_values,dtype=np.float64))
   matrix=np.column_stack(values) # source groups x (seed,system,query)
   boot=(W@matrix).reshape(10000,3,4)
   # cell intervals and paired G/V/I intervals share identical group/seed draws.
   contrast=np.empty((10000,3,5),dtype=np.float64)
   contrast[:,:,0]=boot[:,:,0]-boot[:,:,1]
   contrast[:,:,1]=boot[:,:,2]-boot[:,:,3]
   contrast[:,:,2]=boot[:,:,2]-boot[:,:,0]
   contrast[:,:,3]=boot[:,:,3]-boot[:,:,1]
   contrast[:,:,4]=contrast[:,:,0]-contrast[:,:,1]
   if not np.allclose(contrast[:,:,4],contrast[:,:,3]-contrast[:,:,2],atol=1e-12,rtol=1e-10):
    raise SystemExit('bootstrap G/V/I identity failed for '+component)
   estimands=[('CELL:TS11-D:Q0',boot[:,:,0]),('CELL:TS11-D:Q1',boot[:,:,1]),
    ('CELL:STATE-CTRL:Q0',boot[:,:,2]),('CELL:STATE-CTRL:Q1',boot[:,:,3]),
    ('G_TS11-D',contrast[:,:,0]),('G_STATE-CTRL',contrast[:,:,1]),
    ('V_Q0',contrast[:,:,2]),('V_Q1',contrast[:,:,3]),('I',contrast[:,:,4])]
   for estimand,seed_values in estimands:
    selected=np.take_along_axis(seed_values,seed_draw,axis=1)
    valid=np.isfinite(selected).all(axis=1)
    boot_mean=np.full(10000,np.nan,dtype=np.float64)
    boot_mean[valid]=selected[valid].mean(axis=1)
    vv=boot_mean[np.isfinite(boot_mean)]
    lo,hi=np.percentile(vv,[2.5,97.5]) if len(vv) else (np.nan,np.nan)
    additions.append({'split':'dev','axis_scope':scope,'endpoint':component,'estimand':estimand,
     'bootstrap_mean':float(np.mean(vv)) if len(vv) else float('nan'),
     'ci_percentile_lower_2_5':float(lo),'ci_percentile_upper_97_5':float(hi),
     'valid_draws':int(len(vv)),'requested_draws':10000,'draw_manifest':'N2_GROUP_BOOTSTRAP_DRAWS.npz',
     'draw_manifest_sha256':sha(draws_path),
     'ci_scope':'conditional on fixed 12 models and previously exposed dev; exploratory',
     'p_values_computed':False,'independent_confirmation':False,
     'source_statistics':'N4 saved PER_WINDOW_MOMENTS only; no target arrays reopened'})
 old_path=R/'results/GROUP_BOOTSTRAP_CI.csv';old=readcsv(old_path)
 if any(r.get('endpoint') in components for r in old):raise SystemExit('T/P/C intervals already present; inspect instead of duplicating.')
 add_path=R/'results/FD_MOMENT_BOOTSTRAP_ADDENDUM.csv'
 writecsv(add_path,additions)
 # Preserve the original table rows and append the locked-moment component intervals.
 writecsv(old_path,old+additions)
 addendum={'schema_version':1,'status':'PASS_TPC_BOOTSTRAP_CI_ADDED_FROM_SAVED_MOMENTS',
  'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'source_moments':'results/PER_WINDOW_MOMENTS/*.npz','target_arrays_reopened':False,
  'draw_manifest_path':str(draws_path),'draw_manifest_sha256':sha(draws_path),
  'group_draws':10000,'seed_draws':True,'components':['target_energy_T','prediction_energy_P','cross_alignment_C'],
  'axis_scopes':list(SCOPES),'rows_added':len(additions),'valid_draws_per_interval':10000,
  'p_values_computed':False,'independent_confirmation':False}
 writej(R/'qa/BOOTSTRAP_MOMENT_ADDENDUM.json',addendum)
 acceptance['output_counts']['bootstrap_ci_rows']=len(old)+len(additions)
 acceptance['bootstrap_moment_addendum_path']=str(R/'qa/BOOTSTRAP_MOMENT_ADDENDUM.json')
 acceptance['bootstrap_moment_addendum_sha256']=sha(R/'qa/BOOTSTRAP_MOMENT_ADDENDUM.json')
 acceptance['bootstrap_ci_components']=['position_standardized_mae','safe_fd_normalized','native_future_internal_fd',
  'position_mse','level_mse','time_centered_error_mse','oracle_original_standardized_mae',
  'oracle_removable_standardized_mae','oracle_remaining_standardized_mae','target_energy_T',
  'prediction_energy_P','cross_alignment_C','fd_mse','fd_mse_over_T','r_norm_ratio',
  'c_zero_lag_cosine','fd_mse_skill','norm_error_term','alignment_error_term']
 writej(R/'qa/ANALYSIS_ACCEPTANCE.json',acceptance)
 print(json.dumps({'status':addendum['status'],'rows_added':len(additions),
  'total_bootstrap_rows':len(old)+len(additions),'targets_reopened':False,
  'draw_manifest_sha256':addendum['draw_manifest_sha256']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
