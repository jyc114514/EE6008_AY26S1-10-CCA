#!/usr/bin/env python3
"""Reaggregate already-scored V3 RGB probes before any V4 target read."""
from __future__ import annotations
import os
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
V3=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'))
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def episode_equal(rows,metric):
    by=defaultdict(list)
    for r in rows:by[int(r['episode_id'])].append(float(r[metric]))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float('nan')
def pooled(rows,metric):return float(np.mean([float(r[metric]) for r in rows])) if rows else float('nan')
def main():
    rows=read(V3/'results/scoring/attempt_02/PER_WINDOW_LOSSES.csv')
    by={(r['method'],int(r['seed']),int(r['task_id'])):[] for r in rows}
    for r in rows:
        k=(r['method'],int(r['seed']),int(r['task_id']))
        if k in by:by[k].append(r)
    probe_names={'SAME_TASK_DONOR':'RGB_PROBE_G-V_SAME_TASK_DONOR','ZERO_NORMALIZED_BLOCK':'RGB_PROBE_G-V_ZERO_NORMALIZED_BLOCK'}
    out=[];verify=[]
    for seed in (202610131,202610132,202610133):
        task_values={cond:{metric:[] for metric in METRICS} for cond in probe_names}
        pooled_values={cond:{metric:[] for metric in METRICS} for cond in probe_names}
        all_windows={cond:{metric:[] for metric in METRICS} for cond in probe_names}
        matched_windows=[]
        for task in (119,94,90,141,230):
            ref=by.get(('BASE_VIDEO',seed,task),[])
            if len(ref)==0:raise RuntimeError(f'missing V3 matched BASE_VIDEO rows for seed={seed} task={task}')
            for cond,meth in probe_names.items():
                cur=by.get((meth,seed,task),[])
                if len(cur)!=len(ref):raise RuntimeError(f'V3 probe support mismatch seed={seed} task={task} condition={cond}')
                rmap={r['anchor_id']:r for r in ref};cmap={r['anchor_id']:r for r in cur}
                if set(rmap)!=set(cmap):raise RuntimeError('V3 probe IDs do not join exactly to matched BASE_VIDEO')
                for metric in METRICS:
                    m_ep=episode_equal(ref,metric);c_ep=episode_equal(cur,metric);m_pool=pooled(ref,metric);c_pool=pooled(cur,metric)
                    d_ep=c_ep-m_ep;d_pool=c_pool-m_pool
                    task_values[cond][metric].append(d_ep);pooled_values[cond][metric].append((c_pool,m_pool))
                    all_windows[cond][metric].extend([float(cmap[a][metric])-float(rmap[a][metric]) for a in rmap])
                    out.append({'model_id':'G-V','seed':seed,'task_id':task,'condition':cond,'metric':metric,'aggregation':'task_episode_equal','matched':m_ep,'condition_value':c_ep,'condition_minus_matched':d_ep,'dev_episodes':len({int(x['episode_id']) for x in ref}),'dev_anchors':len(ref)})
                    out.append({'model_id':'G-V','seed':seed,'task_id':task,'condition':cond,'metric':metric,'aggregation':'task_anchor_pooled','matched':m_pool,'condition_value':c_pool,'condition_minus_matched':d_pool,'dev_episodes':len({int(x['episode_id']) for x in ref}),'dev_anchors':len(ref)})
                    # Compare against the accepted V3 per-task aggregate receipt.
                    receipt=[x for x in read(V3/'results/scoring/attempt_02/RGB_PROBE_RESULTS.csv') if x['model_id']=='G-V' and int(x['seed'])==seed and x['condition']==cond and int(x['task_id'])==task]
                    rkey={'position_mae':'position_mae_condition_minus_matched','safe_fd_l1':'safe_fd_l1_condition_minus_matched','seam_safe_fd_l1':'seam_safe_fd_l1_condition_minus_matched'}[metric]
                    if len(receipt)!=1:raise RuntimeError('accepted V3 RGB probe summary row missing/duplicated')
                    verify.append(abs(float(receipt[0][rkey])-d_ep))
        for cond in probe_names:
            for metric in METRICS:
                deltas=task_values[cond][metric]
                out.append({'model_id':'G-V','seed':seed,'task_id':'macro5','condition':cond,'metric':metric,'aggregation':'five_task_macro_episode_equal','matched':'computed_from_task_rows','condition_value':'computed_from_task_rows','condition_minus_matched':float(np.mean(deltas)),'dev_episodes':47,'dev_anchors':185})
                # Equal raw receiver-window weighting across all five tasks.
                out.append({'model_id':'G-V','seed':seed,'task_id':'all','condition':cond,'metric':metric,'aggregation':'all_anchor_pooled','matched':'computed_from_task_rows','condition_value':'computed_from_task_rows','condition_minus_matched':float(np.mean(all_windows[cond][metric])),'dev_episodes':47,'dev_anchors':185})
    out_path=ROOT/'audit/V3_RGB_PER_TASK_AGGREGATION.csv'
    with out_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    obj={'schema_version':1,'status':'PASS','source_path':str(V3/'results/scoring/attempt_02/PER_WINDOW_LOSSES.csv'),'source_sha256':__import__('hashlib').sha256((V3/'results/scoring/attempt_02/PER_WINDOW_LOSSES.csv').read_bytes()).hexdigest(),'prior_RGB_PROBE_RESULTS_sha256':__import__('hashlib').sha256((V3/'results/scoring/attempt_02/RGB_PROBE_RESULTS.csv').read_bytes()).hexdigest(),'V4_dev_target_reads_for_this_reaggregation':0,'task_seed_rows':len(out),'metrics':list(METRICS),'conditions':list(probe_names),'max_abs_difference_vs_accepted_V3_per_task_summary':max(verify),'output_csv':str(out_path),'final_heldout_payload_access':0}
    (ROOT/'audit/V3_RGB_PER_TASK_AGGREGATION.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(obj,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
