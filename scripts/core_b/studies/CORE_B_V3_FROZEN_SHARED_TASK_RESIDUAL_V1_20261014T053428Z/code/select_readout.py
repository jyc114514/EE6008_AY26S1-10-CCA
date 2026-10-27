#!/usr/bin/env python3
"""Select one lambda/epoch per program from inner-tune only."""
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROGRAMS=('SHARED_RES_STATE','TASK_RES_STATE','SHARED_RES_VIDEO','TASK_RES_VIDEO')
def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_json(p,obj):
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def main():
 q=json.loads((ROOT/'coordination/queues/SCREENING_READOUT.json').read_text())
 observed=[];by_program={p:[] for p in PROGRAMS};accepted=failed=0
 for raw in q['jobs']:
  watcher_path=Path(raw);watcher=json.loads(watcher_path.read_text());job_root=Path(watcher['output_root'])
  marker=job_root/'FIT_COMPLETE.json'
  if not marker.is_file() or json.loads(marker.read_text()).get('status')!='PASS':
   failed+=1;continue
  accepted+=1
  rows=read_csv(job_root/'CANDIDATES.csv')
  if len(rows)!=3:raise RuntimeError(f"expected checkpoints 20/40/80 in {job_root}")
  for r in rows:
   record={'program':watcher_path.stem.split('_lambda')[0].replace('v3_screen_','').upper(),
    'lambda':float(r['lambda']),'epoch':int(r['epoch']),'tune_j':float(r['tune_j']),
    'tune_position_mae':float(r['tune_position_mae']),'tune_safe_fd_l1':float(r['tune_safe_fd_l1']),
    'tune_episodes':int(r['tune_episodes']),'candidate_path':str(job_root/'CANDIDATES.csv'),
    'candidate_sha256':__import__('hashlib').sha256((job_root/'CANDIDATES.csv').read_bytes()).hexdigest(),
    'fit_complete_path':str(marker),'fit_complete_sha256':__import__('hashlib').sha256(marker.read_bytes()).hexdigest(),
    'status':'PASS'}
   # Derive program from the job config, not from its display label.
   fit=__import__('json').loads(Path(watcher['command'][-1]).read_text());record['program']=fit['program']
   by_program[record['program']].append(record);observed.append(record)
 choices=[]
 for program in PROGRAMS:
  rows=by_program[program]
  if not rows:
   choices.append({'program':program,'status':'NA_NO_VALID_TUNE_CANDIDATE','selected_lambda':'NA','selected_epoch':'NA','selected_tune_j':'NA','valid_candidate_count':0})
   continue
  best=min(r['tune_j'] for r in rows)
  tie=[r for r in rows if r['tune_j']<=best+1e-12]
  chosen=sorted(tie,key=lambda r:(r['epoch'],-r['lambda']))[0]
  choices.append({'program':program,'status':'LOCKED' if len(rows)==6 else 'LOCKED_WITH_PARTIAL_SEARCH',
   'selected_lambda':chosen['lambda'],'selected_epoch':chosen['epoch'],'selected_tune_j':chosen['tune_j'],
   'selected_tune_position_mae':chosen['tune_position_mae'],'selected_tune_safe_fd_l1':chosen['tune_safe_fd_l1'],
   'selected_tune_episodes':chosen['tune_episodes'],'valid_candidate_count':len(rows),'planned_candidate_count':6,
   'tie_rule':'J within 1e-12: shorter epoch, then larger lambda','selection_source':'inner_tune only'})
 status='LOCKED' if all(x['valid_candidate_count']==6 for x in choices) else 'LOCKED_WITH_PARTIAL_SEARCH'
 if any(x['valid_candidate_count']==0 for x in choices):status='PARTIAL'
 receipt={'schema_version':1,'status':status,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'planned_fits':8,'accepted_fits':accepted,'failed_or_incomplete_fits':failed,'program_count':4,
  'candidate_grid_per_program':'lambda 0/1 at epoch 20/40/80','valid_candidate_rows':len(observed),
  'selection_metric':'five-task equal-task macro canonical position_mae + 0.1 * safe_fd_l1',
  'dev_used_for_selection':False,'inner_tune_only':True,'minimum_support_gate':False,'positive_gain_gate':False,
  'choices':choices,'candidates':observed,'final_heldout_payload_access':0}
 write_json(ROOT/'protocol/READOUT_SELECTION.json',receipt)
 print(json.dumps({'status':status,'accepted_fits':accepted,'failed_fits':failed,'choices':choices,'dev_selection':False,'final_heldout_payload_access':0},indent=2))
if __name__=='__main__':main()
