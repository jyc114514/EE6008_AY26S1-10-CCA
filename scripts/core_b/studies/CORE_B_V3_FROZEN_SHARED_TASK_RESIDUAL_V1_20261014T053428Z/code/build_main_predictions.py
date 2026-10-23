#!/usr/bin/env python3
"""Register 12 readout and six train-bias CPU-only dev prediction jobs."""
from __future__ import annotations
import os
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V2=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'))
SEEDS=(202610131,202610132,202610133)
PROGRAMS=('SHARED_RES_STATE','TASK_RES_STATE','SHARED_RES_VIDEO','TASK_RES_VIDEO')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if path.exists():
  try:
   old=json.loads(path.read_text(encoding='utf-8'));new=json.loads(content);old.pop('created_utc',None);new.pop('created_utc',None);equal=old==new
  except Exception:equal=path.read_bytes().decode('utf-8')==content
  if not equal:raise RuntimeError(f'preserving differing V3 prediction config: {path}')
 else:path.write_text(content,encoding='utf-8')
def accepted_fit(program,seed):
 base_root=ROOT/'model_only/formal'/program/f'seed{seed}'
 found=[]
 if base_root.exists():
  for marker in base_root.glob('attempt_*/FIT_COMPLETE.json'):
   d=json.loads(marker.read_text())
   if d.get('status')=='PASS':found.append((marker,d))
 if len(found)!=1:raise RuntimeError(f'expected one accepted formal fit for {program}/{seed}, found {len(found)}')
 return found[0]
def main():
 selection=json.loads((ROOT/'protocol/READOUT_SELECTION.json').read_text())
 if selection.get('status')!='LOCKED' or len(selection.get('choices',[]))!=4:raise RuntimeError('V3 screening selection is not fully locked')
 bias_receipt=json.loads((ROOT/'calibration/BIAS_CALIBRATION_COMPLETE.json').read_text())
 if bias_receipt.get('status')!='PASS' or bias_receipt.get('calibration_count')!=6:raise RuntimeError('six train-only bias calibrations are not accepted')
 stage=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text());base_by={(x['model_id'],int(x['seed'])):x for x in stage['formal_bases']}
 preds=[]
 attempt=2
 for program in PROGRAMS:
  base_id='G-S' if program.endswith('_STATE') else 'G-V';modality='STATE' if base_id=='G-S' else 'VIDEO'
  for seed in SEEDS:
   base=base_by[(base_id,seed)];fit_marker,fit=accepted_fit(program,seed)
   readout_path=Path(fit['readout_path'])
   if not readout_path.is_file() or sha(readout_path)!=fit.get('readout_sha256'):raise RuntimeError(f'readout file/fit receipt mismatch: {program}/{seed}')
   feature_root=ROOT/'features/formal'/f"{base_id.replace('-','_')}_s{seed}"/'attempt_01'
   feature_marker=feature_root/'FEATURES_COMPLETE.json'
   if json.loads(feature_marker.read_text()).get('status')!='PASS':raise RuntimeError(f'formal dev feature missing: {feature_marker}')
   jid=f'v3_main_{program.lower()}_s{seed}_a02'
   out=ROOT/'predictions/main'/program/f'seed{seed}'/'attempt_02'
   job={'schema_version':1,'job_id':jid,'method_kind':'readout','method_id':program,'program':program,'modality':modality,
    'base_model_id':base_id,'seed':seed,'base_model_sha256':base['model_sha256'],'normalizer_path':base['normalizer_path'],
    'readout_path':str(readout_path),'readout_sha256':sha(readout_path),'fit_complete_path':str(fit_marker),
    'dev_feature_path':str(feature_root/'DEV_FEATURES.safetensors'),'dev_feature_rows_path':str(feature_root/'DEV_ROWS.csv'),
    'dev_feature_complete_path':str(feature_marker),'dev_prefix_path':str(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),
    'metric_scale_lock_path':str(V2/'protocol/METRIC_SCALE_LOCK.json'),'output_root':str(out),
    'completion_marker':'PREDICTION_COMPLETE.json','final_test_access':0}
   jp=ROOT/'configs/jobs/MAIN_PREDICTIONS'/f'{jid}.json';write(jp,job);preds.append(str(jp))
 for base_id,modality in [('G-S','STATE'),('G-V','VIDEO')]:
  for seed in SEEDS:
   base=base_by[(base_id,seed)];key=f'{base_id}:{seed}';ref=stage['reused_v2_accepted_prediction_references'][key]
   jid=f'v3_main_bias_{modality.lower()}_s{seed}_a02';out=ROOT/'predictions/main'/f'BIAS_{modality}'/f'seed{seed}'/'attempt_02'
   job={'schema_version':1,'job_id':jid,'method_kind':'bias','method_id':f'BIAS_{modality}','program':f'BIAS_{modality}',
    'modality':modality,'base_model_id':base_id,'seed':seed,'base_model_sha256':base['model_sha256'],
    'normalizer_path':base['normalizer_path'],'bias_constants_path':str(ROOT/'calibration/BIAS_CONSTANTS.csv'),
    'base_prediction_reference':ref,'dev_prefix_path':str(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),
    'metric_scale_lock_path':str(V2/'protocol/METRIC_SCALE_LOCK.json'),'output_root':str(out),
    'completion_marker':'PREDICTION_COMPLETE.json','final_test_access':0}
   jp=ROOT/'configs/jobs/MAIN_PREDICTIONS'/f'{jid}.json';write(jp,job);preds.append(str(jp))
 if len(preds)!=18:raise RuntimeError(f'main new prediction plan must be 12 residual + 6 bias, got {len(preds)}')
 q=ROOT/'coordination/queues/MAIN_NEW_PREDICTIONS_ATTEMPT_02.json'
 write(q,{'schema_version':1,'phase':'main_cpu_predictions_attempt02','attempt':2,'jobs':sorted(preds),'job_count':len(preds),
  'readout_predictions':12,'bias_predictions':6,'new_task_method_seed_views_max':90,
  'labels_loaded':False,'score_after_predictions_sealed':True,'final_test_access':0})
 plan={'schema_version':1,'status':'LOCKED','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'new_prediction_jobs':18,'readout_prediction_jobs':12,'bias_prediction_jobs':6,'new_task_method_seed_views_max':90,
  'old_v2_base_jobs_reused':6,'old_v2_base_task_views_reused':30,'v2_match_sha_source':str(ROOT/'qa/V2_MATCHED_BASE_REPLAY.json'),
  'prediction_device':'CPU; cached frozen hidden and readout only, no CUDA required','dev_targets_read_before_seal':False,
  'final_heldout_payload_access':0,'queue':str(q),'selection_sha256':sha(ROOT/'protocol/READOUT_SELECTION.json'),
  'bias_constants_sha256':sha(ROOT/'calibration/BIAS_CONSTANTS.csv')}
 plan.update({'attempt':2,'retry_reason':'attempt_01 preserved after pre-seal CPU prediction serialization errors'})
 write(ROOT/'protocol/MAIN_PREDICTION_PLAN_ATTEMPT_02.json',plan)
 print(json.dumps({'status':'PASS','new_jobs':len(preds),'readout_jobs':12,'bias_jobs':6,'task_views_max':90,
  'device':'CPU','dev_targets_read':False,'final_heldout_payload_access':0},indent=2))
if __name__=='__main__':main()
