#!/usr/bin/env python3
"""Nonempty, ID-ordered preseal acceptance of V3 main new predictions."""
from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from v3_common import ROOT,V2,sha

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 qpath=ROOT/'coordination/queues/MAIN_NEW_PREDICTIONS_ATTEMPT_02.json';q=json.loads(qpath.read_text())
 accepted=[];missing=[]
 for raw in q['jobs']:
  jp=Path(raw);job=json.loads(jp.read_text());out=Path(job['output_root']);marker=out/'PREDICTION_COMPLETE.json'
  if not marker.is_file():missing.append(job['job_id']);continue
  comp=json.loads(marker.read_text())
  if comp.get('status')!='PASS':missing.append(job['job_id']);continue
  pred_path=out/'PREDICTIONS.safetensors';ids_path=out/'PREDICTION_ROWS.csv'
  if sha(pred_path)!=comp.get('prediction_sha256') or sha(ids_path)!=comp.get('prediction_rows_sha256'):raise RuntimeError(f'prediction artifact hash mismatch: {job["job_id"]}')
  tensors=load_file(str(pred_path),device='cpu');rows=read_csv(ids_path)
  if len(rows)!=185 or tensors['prediction_native'].shape!=(185,30,28) or tensors['action_residual'].shape!=(185,30,28):raise RuntimeError(f'prediction schema/support invalid: {job["job_id"]}')
  if not np.isfinite(tensors['prediction_native'].numpy()).all() or not np.isfinite(tensors['action_residual'].numpy()).all():raise RuntimeError(f'nonfinite predictions {job["job_id"]}')
  ids=[x['anchor_id'] for x in rows]
  if len(set(ids))!=len(ids) or ids!=sorted(ids):raise RuntimeError(f'prediction IDs duplicate or unordered {job["job_id"]}')
  task_counts={str(t):sum(int(x['task_id'])==t for x in rows) for t in (119,94,90,141,230)}
  if task_counts!={'119':48,'94':48,'90':41,'141':20,'230':28}:raise RuntimeError(f'task support mismatch {job["job_id"]}: {task_counts}')
  accepted.append({'job_id':job['job_id'],'method_id':job['method_id'],'seed':int(job['seed']),'rows':len(rows),
   'task_counts':task_counts,'prediction_path':str(pred_path),'prediction_sha256':sha(pred_path),
   'prediction_rows_sha256':sha(ids_path),'completion_sha256':sha(marker),'execution_device':'CPU','gpu_started':False})
 if len(accepted)!=18:missing.extend([x for x in (job.get('job_id') for raw in q['jobs'] for job in [json.loads(Path(raw).read_text())]) if x not in {r['job_id'] for r in accepted}])
 # Verify old V2 base references are the six accepted G-S/G-V jobs and preserve their complete IDs.
 stage=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text());base_refs=[]
 for mid in ('G-S','G-V'):
  for seed in (202610131,202610132,202610133):
   ref=stage['reused_v2_accepted_prediction_references'][f'{mid}:{seed}']
   rows=read_csv(Path(ref['ids_path']));t=load_file(ref['prediction_path'],device='cpu')['prediction_native']
   if len(rows)!=185 or t.shape!=(185,30,28):raise RuntimeError(f'V2 base reference invalid {mid}/{seed}')
   base_refs.append({'model_id':mid,'seed':seed,'rows':len(rows),'prediction_sha256':ref['prediction_sha256'],'ids_sha256':ref['ids_sha256'],'source':'accepted V2 seal'})
 status='PASS' if len(accepted)==18 and not missing else 'PARTIAL'
 receipt={'schema_version':1,'status':status,'phase':'main_cpu_predictions_attempt02','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),
  'planned_new_jobs':18,'accepted_new_jobs':len(accepted),'missing_job_ids':sorted(set(missing)),
  'new_task_method_seed_views':len(accepted)*5,'old_v2_base_jobs_reused':6,'old_v2_base_task_views':30,
  'accepted_new_predictions':accepted,'reused_base_predictions':base_refs,
  'predictions_nonempty':bool(accepted),'id_order_and_task_support_valid':True,'finite_arrays_valid':True,
  'dev_targets_opened':False,'dev_target_rows_read':0,'empty_or_missing_prediction_target_read_allowed':False,
  'final_heldout_payload_access':0,'preseal_only':True}
 p=ROOT/'qa/MAIN_PREDICTION_ACCEPTANCE_PRESEAL.json'
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':status,'accepted_new_jobs':len(accepted),'expected_new_jobs':18,'views':len(accepted)*5,
  'missing':len(set(missing)),'old_base_views':30,'dev_target_rows_read':0,'final_heldout_payload_access':0},indent=2))
 if status!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
