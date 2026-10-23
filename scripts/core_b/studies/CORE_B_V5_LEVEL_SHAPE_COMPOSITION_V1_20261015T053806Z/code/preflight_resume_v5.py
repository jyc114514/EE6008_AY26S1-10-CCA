#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
V4=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'))
TASKS=[119,94,90,141,230];SEEDS=[202610131,202610132,202610133]
MAP={'BASE_STATE':'S','B_M':'V','A_M':'A','C_M':'C','BIAS_STATE':'BIAS_STATE','BIAS_VIDEO':'BIAS_VIDEO','PERSISTENCE':'PERSISTENCE'}
METRICS=['position_mae','safe_fd_l1','seam_safe_fd_l1']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writec(p,rows,fields=None):
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields:fields.append(k)
 with Path(p).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def writej(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def main():
 sys.path.insert(0,str(ROOT/'code'))
 from score_v5 import aggregate_rows,bootstrap_group_stats,CONTRASTS
 seal=json.loads((ROOT/'qa/PREDICTION_SEAL.json').read_text());fh=sha(ROOT/'protocol/FORMULA_LOCK.json')
 if seal.get('status')!='PASS' or seal.get('formula_lock_sha256')!=fh:raise RuntimeError('sealed matrix/formula lock mismatch')
 formula=json.loads((ROOT/'protocol/FORMULA_LOCK.json').read_text())
 ids=[r['anchor_id'] for r in sorted(read(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv'),key=lambda x:int(x['canonical_row_index']))]
 if len(ids)!=185 or hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()!=formula['ordered_receiver_ids_sha256']:raise RuntimeError('ordered receiver IDs invalid')
 sm=json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text());prefix=Path(next(x['path'] for x in sm['source_artifacts'] if Path(x['path']).name=='DEV_PREFIXES_NO_LABELS.npz'))
 with np.load(prefix,allow_pickle=False) as z:meta={str(z['anchor_id'][i]):(str(z['episode_id'][i]),str(z['task_id'][i]),str(z['source_cluster'][i])) for i in range(len(z['anchor_id']))}
 # Validate saved target-scored rows from attempt 2 without reopening any target.
 loss_path=ROOT/'results/PER_WINDOW_DERIVED_LOSSES.csv';failure=json.loads((ROOT/'qa/SCORER_ATTEMPT_02_FAILURE.json').read_text())
 if sha(loss_path)!=failure['derived_window_loss_table_sha256'] or failure['prediction_seal_sha256']!=sha(ROOT/'qa/PREDICTION_SEAL.json') or failure['formula_lock_sha256']!=fh:raise RuntimeError('saved loss table hash/provenance mismatch')
 derived=read(loss_path)
 if len(derived)!=3330:raise RuntimeError('saved loss table row count mismatch')
 dgroup={}
 for r in derived:
  key=(r['method'],int(r['seed']))
  if r['anchor_id'] not in set(ids) or (r['episode_id'],r['task_id'],r['source_group'])!=meta[r['anchor_id']]:raise RuntimeError('saved score row metadata mismatch')
  for m in METRICS:
   r[m]=float(r[m])
   if not np.isfinite(r[m]):raise RuntimeError('saved score table has nonfinite values')
  dgroup.setdefault(key,[]).append(r)
 if set(dgroup)!={(m,s) for m in ['H_SV','H_VS','H_AV','H_SC','H_PV','MIX50'] for s in SEEDS} or any(len(v)!=185 for v in dgroup.values()):raise RuntimeError('saved derived score grid incomplete')
 # Read V4's accepted per-window table once, filter the seven named source references, and freeze that selected source for recovery.
 srcpath=V4/'results/PER_WINDOW_LOSSES.csv';srcsha=sha(srcpath);src_all=read(srcpath)
 ref=[];excluded=0
 for r in src_all:
  if r['method'] not in MAP:excluded+=1;continue
  method=MAP[r['method']]
  row={'method':method,'seed':str(r['seed']),'anchor_id':r['anchor_id'],'episode_id':str(r['episode_id']),'task_id':str(r['task_id']),
       'source_group':str(r['source_cluster']),'position_mae':float(r['position_mae']),'safe_fd_l1':float(r['safe_fd_l1']),
       'seam_safe_fd_l1':float(r['seam_safe_fd_l1']),'source':'accepted_V4_per_window_loss_reuse','source_method':r['method']}
  if row['anchor_id'] not in meta or (row['episode_id'],row['task_id'],row['source_group'])!=meta[row['anchor_id']]:raise RuntimeError('accepted V4 reference metadata mismatch')
  ref.append(row)
 if len(src_all)!=8325 or len(ref)!=3885 or excluded!=4440:raise RuntimeError('V4 source score filter counts changed')
 for m in MAP.values():
  for seed in SEEDS:
   rr=[r for r in ref if r['method']==m and r['seed']==str(seed)]
   if len(rr)!=185 or {r['anchor_id'] for r in rr}!=set(ids):raise RuntimeError(f'incomplete V4 reference {m}/{seed}')
 ref_path=ROOT/'results/PER_WINDOW_ACCEPTED_REFERENCES.csv';writec(ref_path,ref)
 refsha=sha(ref_path)
 # Reconcile the old source scores against the V4 primary table's actual two metrics; seam stays sourced at window level.
 prim_path=V4/'results/PRIMARY_TASK_SEED.csv';primsha=sha(prim_path);prim=read(prim_path)
 pmap={(int(r['task_id']),str(r['seed'])):r for r in prim}
 primary_names={'S':'BASE_STATE','V':'B_M','A':'A_M','C':'C_M','BIAS_STATE':'BIAS_STATE','BIAS_VIDEO':'BIAS_VIDEO','PERSISTENCE':'PERSISTENCE'}
 reconcile=[]
 for m,v4m in primary_names.items():
  for seed in SEEDS:
   rr=[r for r in ref if r['method']==m and r['seed']==str(seed)]
   for task in TASKS:
    old=pmap[(task,str(seed))]
    for metric in ('position_mae','safe_fd_l1'):
     agg=aggregate_rows(rr,metric)[task]['episode_macro'];expected=float(old[f'{v4m}_{metric}']);delta=abs(agg-expected)
     reconcile.append({'method':m,'v4_method':v4m,'task_id':task,'seed':seed,'metric':metric,'recomputed_from_v4_per_window':agg,
                       'v4_primary_task_seed':expected,'abs_delta':delta,'tolerance':1e-9,'pass':delta<=1e-9})
 if not all(r['pass'] for r in reconcile):raise RuntimeError('V4 source-score reconciliation failed')
 # Validate complete method grid aggregation and paired contrast support before the target retry.
 all_rows={m:{str(s):[] for s in SEEDS} for m in ['S','V','A','C','H_SV','H_VS','H_AV','H_SC','H_PV','MIX50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE']}
 for r in ref:all_rows[r['method']][r['seed']].append(r)
 for (m,s),rr in dgroup.items():all_rows[m][str(s)]=rr
 agg_check={}
 for m in all_rows:
  for seed in SEEDS:
   rr=all_rows[m][str(seed)]
   if len(rr)!=185:raise RuntimeError(f'aggregation input support mismatch {m}/{seed}')
   for metric in METRICS:
    ag=aggregate_rows(rr,metric)
    if set(ag)!=set(TASKS) or any(not np.isfinite(v['episode_macro']) for v in ag.values()):raise RuntimeError(f'aggregation failed {m}/{seed}/{metric}')
    agg_check[f'{m}:{seed}:{metric}']=True
 for name,(refm,cand) in CONTRASTS.items():
  for seed in SEEDS:
   rm={r['anchor_id']:r for r in all_rows[refm][str(seed)]};cm={r['anchor_id']:r for r in all_rows[cand][str(seed)]}
   if set(rm)!=set(cm):raise RuntimeError(f'paired IDs mismatch {name}/{seed}')
 # Exercise the locked group-bootstrap helper with common fixed draws, no target data.
 rng=np.random.default_rng(20261015);draws={}
 for task in TASKS:
  groups=sorted(set(meta[a][2] for a in ids if int(meta[a][1])==task));draws[task]=rng.integers(0,len(groups),size=(5000,len(groups)),dtype=np.int32)
 sample=[]
 for name,(refm,cand) in CONTRASTS.items():
  seed=SEEDS[0];rm={r['anchor_id']:r for r in all_rows[refm][str(seed)]};cm={r['anchor_id']:r for r in all_rows[cand][str(seed)]}
  rows=[]
  for a in ids:
   x=rm[a];y=cm[a];rows.append({'anchor_id':a,'episode_id':x['episode_id'],'task_id':x['task_id'],'source_group':x['source_group'],
    **{m:float(x[m])-float(y[m]) for m in METRICS}})
  b=bootstrap_group_stats(rows,'position_mae',draws)
  if any(b[t] is None or b[t]['draws'].shape!=(5000,) for t in TASKS):raise RuntimeError('bootstrap helper preflight failed')
  sample.append(name)
 primcols=set(prim[0]);
 if 'BASE_STATE_position_mae' not in primcols or 'BASE_STATE_safe_fd_l1' not in primcols or 'BASE_STATE_seam_safe_fd_l1' in primcols:raise RuntimeError('V4 primary schema assumption changed')
 # Write auditable preflight outputs with source file hashes and no target load.
 writec(ROOT/'results/SOURCE_SCORE_RECONCILIATION.csv',reconcile)
 recon={'schema_version':1,'status':'PASS','v4_per_window_path':str(srcpath),'v4_per_window_sha256':srcsha,
   'v4_primary_task_seed_path':str(prim_path),'v4_primary_task_seed_sha256':primsha,'selected_reference_copy_path':str(ref_path),'selected_reference_copy_sha256':refsha,
   'accepted_reference_methods':list(MAP.values()),'v4_rows_parsed':len(src_all),'selected_reference_rows':len(ref),'unselected_rows_excluded':excluded,
   'donor_map_B_D_C_D_outputs_used':False,'reconciled_metrics':['position_mae','safe_fd_l1'],'seam_metric_source':'accepted V4 per-window table only; primary table has no seam column',
   'reconciled_comparisons':len(reconcile),'max_abs_delta':max(r['abs_delta'] for r in reconcile),'tolerance':1e-9,
   'final_heldout_payload_access':0,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
 writej(ROOT/'qa/SOURCE_SCORE_RECONCILIATION.json',recon)
 pre={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),'formula_lock_sha256':fh,'saved_derived_loss_table_sha256':sha(loss_path),
  'saved_derived_loss_rows':len(derived),'saved_derived_bundles':len(dgroup),'saved_score_keys_checked':len(agg_check),
  'accepted_reference_rows':len(ref),'accepted_reference_copy_path':str(ref_path),'accepted_reference_copy_sha256':refsha,
  'v4_per_window_path':str(srcpath),'v4_per_window_sha256':srcsha,'v4_primary_sha256':primsha,'source_reconciliation_comparisons':len(reconcile),
  'source_reconciliation_max_abs_delta':max(r['abs_delta'] for r in reconcile),'paired_contrasts_preflighted':sample,
  'paired_group_bootstrap_helper':'PASS; fixed seed 20261015; shared within-task group draws','target_archive_payload_opened':False,
  'target_archive_loads_added':0,'final_heldout_payload_access':0}
 writej(ROOT/'qa/SCORER_RESUME_PREFLIGHT.json',pre)
 ex_path=ROOT/'inventory/OUTCOME_EXPOSURE.json';ex=json.loads(ex_path.read_text());ex.update({'v4_resume_preflight_per_window_rows_parsed':len(src_all),
  'v4_resume_preflight_reference_rows_selected':len(ref),'v4_resume_preflight_unselected_rows_excluded':excluded,'v4_resume_preflight_primary_rows_parsed':len(prim),
  'v4_per_window_file_read_operations_before_retry':int(ex.get('v4_per_window_file_read_operations_before_retry',0))+1,
  'v4_primary_table_read_operations_before_retry':int(ex.get('v4_primary_table_read_operations_before_retry',0))+1,
  'v4_reference_rows_reused_unique':len(ref),'last_updated_utc':pre['created_utc']})
 ex['history'].append({'phase':'V5_resume_preflight_saved_scores_and_v4_references','target_archive_loads':0,'saved_derived_rows_validated':len(derived),
  'v4_rows_parsed':len(src_all),'accepted_reference_rows_copied':len(ref),'target_archive_payload_opened':False,'status':'PASS'})
 writej(ex_path,ex)
 print(json.dumps(pre,indent=2))
if __name__=='__main__':main()
