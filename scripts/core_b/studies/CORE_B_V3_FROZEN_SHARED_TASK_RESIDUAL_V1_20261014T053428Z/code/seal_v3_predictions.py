#!/usr/bin/env python3
"""Validate exact nonempty V3 prediction batches, then seal before any dev target access."""
from __future__ import annotations
import csv,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from v3_common import ROOT,V2,read_json,sha

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def verify_job(config_path,expected_rows,expected_task_counts):
 cfg=read_json(config_path);out=Path(cfg['output_root']);marker=out/str(cfg.get('completion_marker','PREDICTION_COMPLETE.json'))
 if not marker.is_file():return None
 comp=read_json(marker)
 if comp.get('status')!='PASS':return None
 pred=out/'PREDICTIONS.safetensors';ids=out/'PREDICTION_ROWS.csv'
 if not pred.is_file() or not ids.is_file():raise RuntimeError(f'accepted marker lacks prediction/ID file: {cfg["job_id"]}')
 if sha(pred)!=comp.get('prediction_sha256') or sha(ids)!=comp.get('prediction_rows_sha256'):
  raise RuntimeError(f'prediction hash differs from completion marker: {cfg["job_id"]}')
 rows=read_csv(ids);tensors=load_file(str(pred),device='cpu')
 if len(rows)!=expected_rows or tensors['prediction_native'].shape!=(expected_rows,30,28) or tensors['action_residual'].shape!=(expected_rows,30,28):
  raise RuntimeError(f'prediction support/shape invalid: {cfg["job_id"]}')
 if not np.isfinite(tensors['prediction_native'].numpy()).all() or not np.isfinite(tensors['action_residual'].numpy()).all():
  raise RuntimeError(f'nonfinite predictions: {cfg["job_id"]}')
 ordered_ids=[x['anchor_id'] for x in rows]
 if len(set(ordered_ids))!=len(ordered_ids) or ordered_ids!=sorted(ordered_ids):raise RuntimeError(f'duplicate or unordered IDs: {cfg["job_id"]}')
 counts={str(t):sum(int(x['task_id'])==t for x in rows) for t in (119,94,90,141,230) if any(int(x['task_id'])==t for x in rows)}
 if counts!=expected_task_counts:raise RuntimeError(f'task support mismatch {cfg["job_id"]}: {counts} != {expected_task_counts}')
 return {'job_id':cfg['job_id'],'method_id':cfg.get('method_id',cfg.get('model_id')),'method_kind':cfg.get('method_kind','rgb_probe'),
  'model_id':cfg.get('model_id',cfg.get('base_model_id')),'seed':int(cfg['seed']),'condition':cfg.get('condition','MATCHED'),
  'row_count':len(rows),'task_counts':counts,'output_root':str(out),'completion_marker':str(marker),
  'completion_sha256':sha(marker),'prediction_path':str(pred),'prediction_sha256':sha(pred),
  'prediction_rows_path':str(ids),'prediction_rows_sha256':sha(ids),'job_config_path':str(config_path),'job_config_sha256':sha(config_path)}
def main():
 main_pre=read_json(ROOT/'qa/MAIN_PREDICTION_ACCEPTANCE_PRESEAL.json')
 if main_pre.get('status')!='PASS' or main_pre.get('dev_target_rows_read')!=0:raise RuntimeError('main CPU predictions not accepted preseal')
 state_queue=read_json(ROOT/'coordination/queues/STATE_RGB_QA.json');state_job=json.loads(Path(state_queue['jobs'][0]).read_text())
 state_complete=Path(state_job['output_root'])/'STATE_RGB_QA_COMPLETE.json'
 if not state_complete.is_file() or read_json(state_complete).get('status')!='PASS':raise RuntimeError('state RGB invariance QA incomplete')
 new=[];missing=[];main_queue=read_json(ROOT/'coordination/queues/MAIN_NEW_PREDICTIONS_ATTEMPT_02.json')
 for raw in main_queue['jobs']:
  cfg=json.loads(Path(raw).read_text());item=verify_job(Path(raw),185,{'119':48,'94':48,'90':41,'141':20,'230':28})
  if item is None:missing.append(cfg['job_id'])
  else:new.append(item)
 probe_queue=read_json(ROOT/'coordination/queues/RGB_PROBE.json')
 for raw in probe_queue['jobs']:
  wrapper=json.loads(Path(raw).read_text())
  command=wrapper.get('command',[])
  if '--job-json' not in command:raise RuntimeError(f'RGB watcher wrapper lacks --job-json: {raw}')
  config_path=Path(command[command.index('--job-json')+1])
  cfg=read_json(config_path)
  if cfg['model_id']=='G-V':expected_rows=185;expected_counts={'119':48,'94':48,'90':41,'141':20,'230':28}
  else:expected_rows=41;expected_counts={'90':41}
  item=verify_job(config_path,expected_rows,expected_counts)
  if item is None:missing.append(cfg['job_id'])
  else:new.append(item)
 # Reuse only accepted current-V2 sealed references; these are not new V3 predictions.
 stage=read_json(ROOT/'inventory/BASE_STAGE_MAP.json');reused=[]
 for key,ref in sorted(stage['reused_v2_accepted_prediction_references'].items()):
  p=Path(ref['prediction_path']);ids=Path(ref['ids_path']);complete=Path(ref['completion_path']) if 'completion_path' in ref else Path(ref['output_root'])/'PREDICTION_COMPLETE.json'
  if not p.is_file() or not ids.is_file():raise RuntimeError(f'V2 matched prediction reference missing: {key}')
  if sha(p)!=ref['prediction_sha256'] or sha(ids)!=ref['ids_sha256']:
   raise RuntimeError(f'V2 matched prediction reference hash drift: {key}')
  rows=read_csv(ids);expected=185 if ref['model_id'] in {'G-S','G-V'} else 41
  if len(rows)!=expected:raise RuntimeError(f'V2 matched prediction support invalid: {key}')
  reused.append({'model_id':ref['model_id'],'seed':int(ref['seed']),'rows':len(rows),'prediction_path':str(p),
   'prediction_sha256':sha(p),'prediction_rows_path':str(ids),'prediction_rows_sha256':sha(ids),
   'source':'V2 current corrected split, accepted sealed fresh-process prediction'})
 if len({x['job_id'] for x in new})!=len(new):raise RuntimeError('V3 prediction job IDs are duplicated')
 if not new:raise RuntimeError('refusing to seal/read targets for an empty V3 prediction batch')
 planned=30;status='SEALED_BEFORE_TARGET_ACCESS' if len(new)==planned and not missing and len(reused)==9 else 'SEALED_PARTIAL_BEFORE_TARGET_ACCESS'
 if status=='SEALED_PARTIAL_BEFORE_TARGET_ACCESS' and len(new)==0:raise RuntimeError('partial seal still requires nonempty accepted predictions')
 new_path=ROOT/'protocol/PREDICTION_BATCH_SEAL.json'
 if new_path.exists():raise FileExistsError(new_path)
 seal={'schema_version':1,'status':status,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'planned_new_prediction_jobs':planned,'sealed_new_prediction_jobs':len(new),'missing_new_prediction_jobs':sorted(set(missing)),
  'new_prediction_entries':new,'reused_v2_prediction_reference_jobs':len(reused),'reused_v2_prediction_references':reused,
  'main_readout_bias_new_jobs_expected':18,'main_new_task_method_seed_views_max':90,
  'rgb_probe_new_jobs_expected':12,'rgb_probe_new_task_views_max':36,'rgb_matched_v2_views_reused':18,
  'empty_prediction_batch_refused':True,'dev_target_reader_called':False,'dev_target_rows_read':0,
  'future_labels_used_in_forward':False,'final_heldout_payload_access':0}
 new_path.write_text(json.dumps(seal,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 qa={'schema_version':1,'status':'PASS' if status=='SEALED_BEFORE_TARGET_ACCESS' else 'PARTIAL',
  'sealed_prediction_jobs':len(new),'planned_prediction_jobs':planned,'missing_prediction_jobs':sorted(set(missing)),
  'nonempty_prediction_manifest':bool(new),'main_preseal_status':main_pre['status'],'state_rgb_qa_status':read_json(state_complete).get('status'),
  'v2_reference_jobs_reused':len(reused),'rgb_probe_jobs_accepted':sum(x['method_kind']=='rgb_probe' for x in new),
  'gpu_device_queries_by_sealer':0,'target_read_before_seal':False,'final_heldout_payload_access':0}
 p=ROOT/'qa/PREDICTION_ACCEPTANCE_PRESEAL.json'
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(qa,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':status,'new_prediction_jobs':len(new),'planned_new_prediction_jobs':planned,
  'missing_jobs':len(set(missing)),'reused_V2_prediction_jobs':len(reused),'target_rows_read':0,'final_heldout_payload_access':0},indent=2))
if __name__=='__main__':main()
