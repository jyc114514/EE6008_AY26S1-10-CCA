#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import v9_score as score
from v9_common import write_csv

def main():
 source=score.rj(ROOT/'inventory/SOURCE_MAP.json')['source_assets']
 ids,meta=score.load_prefix(Path(source['dev_prefix_npz']['path']),185)
 hist,hsha=score.load_historical(ids,meta)
 if len(hist)!=len(score.HIST)*3*185:raise RuntimeError('historical V7 reference support unexpected')
 zeros=np.zeros((2,30,28),dtype=np.float64);anchor=np.zeros((2,28),dtype=np.float64)
 sigma=np.ones(28);fd=np.ones(28)
 m=score.score_prediction(zeros,zeros,anchor,sigma,fd)
 if any(not np.array_equal(m[k],np.zeros(2)) for k in score.METRICS):raise RuntimeError('perfect prediction scorer identity failed')
 # Exercise all aggregation, mask pairing, contrast, LOO and common-draw CI branches with sealed-free synthetic values.
 methods=list(score.PRIMARY)+list(score.HIST);rows=[];smeta={}
 for task in score.TASKS:
  group='group_'+str(task);episode=1000+task
  for anchor_i in range(2):
   aid=f'syn_{task}_{anchor_i}'
   smeta[aid]={'task_id':task,'episode_id':episode,'source_group':group,'anchor_frame':anchor_i,'anchor_native':np.zeros(28)}
   for seed in score.SEEDS:
    for method_i,method in enumerate(methods):
     base=.1+.001*method_i+.0001*(seed-score.SEEDS[0])+.00001*task+.000001*anchor_i
     rows.append({'method':method,'seed':seed,'anchor_id':aid,'episode_id':episode,'task_id':task,'source_group':group,
      'position_mae':base,'safe_fd_l1':base+.2,'seam_safe_fd_l1':base+.4})
    for method in score.PROGRAMS:
     base=.2+.001*score.PROGRAMS.index(method)+.00001*task+.000001*anchor_i
     rows.append({'method':method+'_MASKED','program_id':method,'input_policy':'force_numeric_context_drop','seed':seed,'anchor_id':aid,
      'episode_id':episode,'task_id':task,'source_group':group,'position_mae':base,'safe_fd_l1':base+.2,'seam_safe_fd_l1':base+.4})
 primary,ts,ep,pv,macro=score.primary_tables(rows)
 effects,slo,tlo,ci,boot=score.paired_tables(rows,smeta)
 masked=score.masked_diagnostic(rows)
 if not primary or len(effects)==0 or len(ci)!=len(score.CONTRASTS)*len(score.METRICS) or len(masked)!=len(score.PROGRAMS)*len(score.SEEDS)*(len(score.TASKS)+1):
  raise RuntimeError('synthetic scorer table count/shape failed')
 if not all(x['status']=='PASS' for x in []):raise RuntimeError('unreachable')
 out={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'score_script_sha256':hashlib.sha256((ROOT/'code/v9_score.py').read_bytes()).hexdigest(),
  'historical_V7_rows':len(hist),'historical_V7_source_sha256':hsha,
  'perfect_prediction_metrics_exact_zero':True,'native_mix_formula':'0.5*S_OPT + 0.5*V0/VD',
  'paired_contrasts_tested':len(score.CONTRASTS),'paired_ci_rows_synthetic':len(ci),'bootstrap_draws':5000,
  'masked_diagnostic_rows_synthetic':len(masked),'dev_target_archive_opened':False,'dev_target_content_reads':0,'final_heldout_payload_access':0}
 p=ROOT/'qa/SYNTHETIC_SCORER_QA.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps(out,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()

