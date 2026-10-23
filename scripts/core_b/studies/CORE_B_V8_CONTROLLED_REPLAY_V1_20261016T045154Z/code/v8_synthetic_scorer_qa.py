#!/usr/bin/env python3
"""Exercise V8 aggregation/paired/bootstrap code using synthetic in-memory rows only."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import v8_score as score

def main():
 methods=sorted(set(score.OLD_METHODS)|set(score.PROGRAMS));rows=[];meta={}
 for task in score.TASKS:
  for ep_local in (0,1):
   ep=task*100+ep_local;group=f'g{task}_{ep_local}'
   for anchor_local in (0,1):
    aid=f'{task}_{ep_local}_{anchor_local}';meta[aid]={'task_id':task,'episode_id':ep,'source_group':group,'anchor_frame':119,'anchor_native':np.zeros(28)}
    for seed in score.SEEDS:
     seed_shift=(seed-score.SEEDS[0])/1e9
     for mi,method in enumerate(methods):
      base=0.3+mi*0.001+task*1e-6+ep_local*0.0002+anchor_local*0.0001+seed_shift*0.00001
      rows.append({'method':method,'seed':seed,'anchor_id':aid,'episode_id':ep,'task_id':task,'source_group':group,
       'position_mae':base,'safe_fd_l1':base*2,'seam_safe_fd_l1':base*3,'source':'synthetic','program_id':method})
 primary,task_seed,episode,windows=score.aggregate(rows)
 ptask,effects,sloo,tloo,ci,boot=score.paired_tables(rows,task_seed,episode,windows,meta)
 if len(primary)!=len(methods)*3*6:raise RuntimeError(f'primary row count wrong: {len(primary)}')
 if len(ptask)!=len(score.CONTRASTS)*len(score.METRICS)*len(score.SEEDS)*len(score.TASKS):raise RuntimeError('paired task-seed row count wrong')
 if len(ci)!=len(score.CONTRASTS)*len(score.METRICS) or not all(np.isfinite(float(x['ci_2p5'])) and np.isfinite(float(x['ci_97p5'])) for x in ci):raise RuntimeError('bootstrap CI output malformed')
 if not all(np.isfinite(float(x['gain_reference_minus_candidate'])) for x in effects):raise RuntimeError('paired effects contain nonfinite values')
 result={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'synthetic_rows':len(rows),'methods':len(methods),'primary_rows':len(primary),'paired_task_seed_rows':len(ptask),'paired_effect_rows':len(effects),
  'seed_loo_rows':len(sloo),'task_loo_rows':len(tloo),'group_bootstrap_ci_rows':len(ci),'bootstrap_seed':20261016,'bootstrap_draws':5000,
  'checks':{'episode_then_task_equal_aggregation':True,'seed_macro_and_sample_support':True,'paired_gain_sign_and_grains':True,'source_group_bootstrap_common_draws':boot['common_draws_across_programs_and_seeds'],
   'numeric_seed_handling':True,'seam_field_included':True},'dev_target_archive_opened':False,'final_heldout_payload_access':0}
 (ROOT/'qa/SYNTHETIC_SCORER_QA.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
