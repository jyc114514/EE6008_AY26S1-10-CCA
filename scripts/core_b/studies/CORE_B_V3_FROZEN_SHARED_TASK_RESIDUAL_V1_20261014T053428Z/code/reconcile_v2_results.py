#!/usr/bin/env python3
"""Independently reconstruct V2 contrasts and both valid leave-one-out analyses."""
from __future__ import annotations
import os
import csv, hashlib, json, shutil, io
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
ROOT=Path(__file__).resolve().parents[1]
TASKS=(119,94,90,141,230)
SEEDS=(202610131,202610132,202610133)
CORE=('D_S','D_V','V_G','V_S','I')
WAYPOINT=('B_GAIN','AUX_VS_SV','GOAL_VS_SV')

def read_csv(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(p, rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else []
    b=io.StringIO(newline='')
    w=csv.DictWriter(b,fieldnames=fields);w.writeheader();w.writerows(rows)
    content=b.getvalue()
    if p.exists():
        if p.read_bytes().decode('utf-8')!=content: raise RuntimeError(f'preserving differing existing V3 CSV: {p}')
        return
    p.write_text(content,encoding='utf-8')
def write_json(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if p.exists():
        old=json.loads(p.read_text(encoding='utf-8'));new=json.loads(content)
        old.pop('created_utc',None);new.pop('created_utc',None)
        if old!=new: raise RuntimeError(f'preserving differing existing V3 JSON: {p}')
        return
    p.write_text(content,encoding='utf-8')
def read_map():
    rows=read_csv(V2/'results/ABSOLUTE_LOSSES.csv')
    out={}
    for r in rows:
        if r['status']!='PASS' or r['seed'] not in {str(x) for x in SEEDS}:continue
        key=(str(r['model_id']),int(r['seed']),int(r['task_id']))
        out[key]={m:float(r[m]) for m in ('position_mae','safe_fd_l1','seam_safe_fd_l1','native_mae') if r.get(m) not in ('','NA')}
    return out

def contrasts(task,seed,metric,vals):
    gs=vals.get((f'G-S',seed,task),{}).get(metric)
    gv=vals.get((f'G-V',seed,task),{}).get(metric)
    ss=vals.get((f'S-S-{task}',seed,task),{}).get(metric)
    sv=vals.get((f'S-V-{task}',seed,task),{}).get(metric)
    if any(x is None for x in (gs,gv,ss,sv)):
        core={c:None for c in CORE}
    else:
        ds=gs-ss; dv=gv-sv; vg=gs-gv; vs=ss-sv; inter=dv-ds
        if abs(inter-(vs-vg))>1e-12:raise AssertionError('D/V/I identity failed')
        core={'D_S':ds,'D_V':dv,'V_G':vg,'V_S':vs,'I':inter}
    if task==119:
        aux=vals.get(('B-AUX_ONLY',seed,task),{}).get(metric)
        goal=vals.get(('B-GOAL_COND',seed,task),{}).get(metric)
        if sv is None or aux is None or goal is None: wp={x:None for x in WAYPOINT}
        else:wp={'B_GAIN':aux-goal,'AUX_VS_SV':sv-aux,'GOAL_VS_SV':sv-goal}
    else:wp={}
    return {**core,**wp}

def summary(x):
    a=np.asarray(x,dtype=np.float64)
    return (float(np.mean(a)),float(np.std(a,ddof=1)) if len(a)>1 else float('nan'),float(np.median(a)))

def main():
    vals=read_map()
    metrics=('position_mae','safe_fd_l1','seam_safe_fd_l1','native_mae')
    point={}
    point_rows=[]
    for task in TASKS:
      for seed in SEEDS:
       for metric in metrics:
        c=contrasts(task,seed,metric,vals)
        for name,value in c.items():
         if value is None:continue
         point[(task,seed,metric,name)]=float(value)
         point_rows.append({'task_id':task,'seed':seed,'metric':metric,'contrast':name,'value':f'{value:.15g}','positive_means_benefit':True})
    # Compare reconstructed point estimates to the accepted V2 all-seed table; round-four-decimal agreement is the display contract.
    stored=read_csv(V2/'results/ALL_SEED_D_V_I.csv')
    stored_map={(int(r['task_id']),int(r['seed']),r['metric'],r['contrast']):float(r['value']) for r in stored if r['value']!='NA'}
    common=set(point)&set(stored_map)
    diffs=[abs(point[k]-stored_map[k]) for k in common]
    round4_fail=[(k,point[k],stored_map[k]) for k in common if round(point[k],4)!=round(stored_map[k],4)]
    if round4_fail:raise RuntimeError(f'V2 independently reconstructed point differs at 4 decimals: {round4_fail[:5]}')
    # Recompute per-seed task and five-task macros, then mean/sample SD/median across the three trained seeds.
    summaries=[]
    for metric in metrics:
      for contrast in (*CORE,*WAYPOINT):
       task_scope=(119,) if contrast in WAYPOINT else TASKS
       seed_vals={}
       for seed in SEEDS:
        taskvals=[point.get((task,seed,metric,contrast)) for task in task_scope]
        taskvals=[v for v in taskvals if v is not None]
        seed_vals[seed]=float(np.mean(taskvals)) if taskvals else None
       scopes=[]
       for task in task_scope:
        per=[point.get((task,seed,metric,contrast)) for seed in SEEDS]
        per=[v for v in per if v is not None]
        if per:
         mean,sd,med=summary(per);scopes.append({'scope':'task' if contrast in CORE else 'task_119','task_id':task,'contrast':contrast,'metric':metric,'mean_across_seeds':mean,'sample_sd_across_seeds':sd,'median_across_seeds':med,'n_seeds':len(per)})
       per_seed=[seed_vals[s] for s in SEEDS if seed_vals[s] is not None]
       if per_seed:
        mean,sd,med=summary(per_seed)
        scopes.append({'scope':'five_task_macro' if contrast in CORE else 'task_119','task_id':'ALL5_OR_AVAILABLE' if contrast in CORE else 119,'contrast':contrast,'metric':metric,'mean_across_seeds':mean,'sample_sd_across_seeds':sd,'median_across_seeds':med,'n_seeds':len(per_seed)})
       for row in scopes:
        summaries.append({k:(f'{v:.15g}' if isinstance(v,float) and np.isfinite(v) else v) for k,v in row.items()})
    v2_summary=read_csv(V2/'results/SUMMARY_STATISTICS.csv')
    sum_map={(r['scope'],str(r['task_id']),r['contrast'],r['metric']):r for r in v2_summary}
    summary_common=[]; summary_round4_fail=[]
    for r in summaries:
      key=(r['scope'],str(r['task_id']),r['contrast'],r['metric'])
      if key not in sum_map:continue
      old=sum_map[key]
      for col in ('mean_across_seeds','sample_sd_across_seeds','median_across_seeds'):
       if old[col] in ('','NA') or r[col] in ('','NA'):continue
       delta=abs(float(r[col])-float(old[col]));summary_common.append(delta)
       if round(float(r[col]),4)!=round(float(old[col]),4):summary_round4_fail.append((key,col,r[col],old[col]))
    if summary_round4_fail:raise RuntimeError(f'V2 summary reconstructed values differ at 4 decimals: {summary_round4_fail[:5]}')
    # True leave-one-seed-out: remove one trained seed, keep all five tasks, average the other two seed macros.
    seed_loo=[]
    for metric in metrics:
      for contrast in (*CORE,*WAYPOINT):
       tasks=(119,) if contrast in WAYPOINT else TASKS
       for scope_task in (*tasks, 'ALL5_OR_AVAILABLE'):
        for omitted in SEEDS:
         included=[s for s in SEEDS if s!=omitted]
         vals2=[]
         for s in included:
          tset=(119,) if scope_task=='ALL5_OR_AVAILABLE' and contrast in WAYPOINT else (TASKS if scope_task=='ALL5_OR_AVAILABLE' else (int(scope_task),))
          tv=[point.get((t,s,metric,contrast)) for t in tset]
          tv=[x for x in tv if x is not None]
          if tv:vals2.append(float(np.mean(tv)))
         estimate=float(np.mean(vals2)) if vals2 else None
         seed_loo.append({'scope':'five_task_macro' if scope_task=='ALL5_OR_AVAILABLE' else ('task_119' if contrast in WAYPOINT else 'task'),
          'task_id':scope_task,'omitted_seed':omitted,'included_seeds':';'.join(map(str,included)),
          'included_seed_count':len(vals2),'contrast':contrast,'metric':metric,'estimate':'NA' if estimate is None else f'{estimate:.15g}',
          'planned_tasks':1 if contrast in WAYPOINT else (5 if scope_task=='ALL5_OR_AVAILABLE' else 1)})
    # True leave-one-task-out, preserving all three seeds; waypoint-only contrasts are explicitly not defined across tasks.
    task_loo=[]
    for metric in metrics:
      for contrast in CORE:
       for seed in SEEDS:
        for omitted in TASKS:
         include=[point.get((t,seed,metric,contrast)) for t in TASKS if t!=omitted]
         finite=[x for x in include if x is not None]
         estimate=float(np.mean(finite)) if finite else None
         task_loo.append({'seed':seed,'metric':metric,'contrast':contrast,'omitted_task_id':omitted,
           'included_tasks':len(finite),'planned_tasks':4,'loo_mean':'NA' if estimate is None else f'{estimate:.15g}'})
    task_loo_summary=[]
    for metric in metrics:
      for contrast in CORE:
       for omitted in TASKS:
        v=[float(x['loo_mean']) for x in task_loo if x['metric']==metric and x['contrast']==contrast and int(x['omitted_task_id'])==omitted and x['loo_mean']!='NA']
        if v:
         m,sd,med=summary(v)
         task_loo_summary.append({'metric':metric,'contrast':contrast,'omitted_task_id':omitted,'included_tasks_per_seed':4,
           'n_seeds':len(v),'mean_across_seeds':f'{m:.15g}','sample_sd_across_seeds':f'{sd:.15g}','median_across_seeds':f'{med:.15g}'})
    # Verify the existing V2 task LOO rows really omit one task and retain 4 per seed.
    old_task_loo=read_csv(V2/'results/ALL_SEED_LOO.csv')
    bad=[r for r in old_task_loo if r['contrast'] in CORE and (int(r['included_tasks'])!=4 or int(r['planned_tasks'])!=4)]
    if bad:raise RuntimeError(f'V2 ALL_SEED_LOO does not match leave-one-task-out shape: {bad[:3]}')
    ci=read_csv(V2/'results/CONDITIONAL_CI.csv')
    ci_status=sorted(set(r['status'] for r in ci));ci_seed=sorted(set(r['bootstrap_seed'] for r in ci));ci_reps=sorted(set(r['bootstrap_replicates'] for r in ci))
    # Preserve point tables as source copies and write independently reconstructed summaries/LOO.
    for source,name in [('results/ABSOLUTE_LOSSES.csv','V2_ABSOLUTE_LOSSES.csv'),('results/ALL_SEED_D_V_I.csv','V2_ALL_SEED_D_V_I.csv'),('results/SUMMARY_STATISTICS.csv','V2_SUMMARY_STATISTICS.csv'),('results/CONDITIONAL_CI.csv','V2_CONDITIONAL_CI.csv')]:
       dest=ROOT/'diagnostics'/name
       if dest.exists():
          if dest.read_bytes()!=(V2/source).read_bytes(): raise RuntimeError(f'preserving differing copied V2 receipt: {dest}')
       else: shutil.copyfile(V2/source,dest)
    write_csv(ROOT/'diagnostics/V2_RECOMPUTED_PER_SEED_CONTRASTS.csv',point_rows)
    write_csv(ROOT/'diagnostics/V2_RECOMPUTED_SUMMARY_STATISTICS.csv',summaries)
    write_csv(ROOT/'diagnostics/V2_LEAVE_ONE_SEED_OUT.csv',seed_loo)
    write_csv(ROOT/'diagnostics/V2_LEAVE_ONE_TASK_OUT.csv',task_loo)
    write_csv(ROOT/'diagnostics/V2_LEAVE_ONE_TASK_OUT_SUMMARY.csv',task_loo_summary)
    report={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
      'source':'V2 accepted saved losses only; V2 model fitting and scorer were not rerun for this reconciliation',
      'v2_absolute_losses_sha256':hashlib.sha256((V2/'results/ABSOLUTE_LOSSES.csv').read_bytes()).hexdigest(),
      'v2_all_seed_contrasts_sha256':hashlib.sha256((V2/'results/ALL_SEED_D_V_I.csv').read_bytes()).hexdigest(),
      'independent_point_rows':len(point_rows),'point_rows_compared_to_v2':len(common),'max_abs_point_difference':max(diffs) if diffs else None,
      'point_rows_matching_at_4_decimals':len(common)-len(round4_fail),'v2_summary_rows_compared':len(summary_common),
      'max_abs_summary_field_difference':max(summary_common) if summary_common else None,
      'summary_fields_matching_at_4_decimals':len(summary_common)-len(summary_round4_fail),
      'existing_v2_loo_interpreted_as':'leave-one-task-out; 4 tasks retained per seed; verified from rows',
      'new_leave_one_seed_out_definition':'exclude one of three trained seeds; average remaining two seed macros; task set unchanged at five',
      'new_leave_one_task_out_definition':'exclude one of five tasks separately per seed; four tasks retained; all three seeds retained',
      'v2_ci_method_from_source':'paired source-group bootstrap; clusters sampled with replacement; all episodes from each sampled cluster retained; per-seed task means then five-task macro; 5000 replicates; PCG64 NumPy seed 202610134',
      'v2_ci_rows':len(ci),'v2_ci_status':ci_status,'v2_ci_replicates':ci_reps,'v2_ci_seed':ci_seed,
      'v2_ci_draw_manifest':'not stored; V3 will generate and lock a new reusable source-group draw manifest with seed 202610134',
      'session_trial_independence':'unverified; inherited V2 limitation','final_heldout_payload_access':0,
      'outputs':{'per_seed':'diagnostics/V2_RECOMPUTED_PER_SEED_CONTRASTS.csv','summary':'diagnostics/V2_RECOMPUTED_SUMMARY_STATISTICS.csv',
       'leave_one_seed_out':'diagnostics/V2_LEAVE_ONE_SEED_OUT.csv','leave_one_task_out':'diagnostics/V2_LEAVE_ONE_TASK_OUT.csv',
       'leave_one_task_out_summary':'diagnostics/V2_LEAVE_ONE_TASK_OUT_SUMMARY.csv'}}
    write_json(ROOT/'diagnostics/V2_RECONCILIATION.json',report)
    print(json.dumps({k:report[k] for k in ('status','independent_point_rows','point_rows_compared_to_v2','max_abs_point_difference','point_rows_matching_at_4_decimals','summary_rows_compared','summary_fields_matching_at_4_decimals','existing_v2_loo_interpreted_as','v2_ci_method_from_source','final_heldout_payload_access') if k in report},indent=2))
if __name__=='__main__': main()
