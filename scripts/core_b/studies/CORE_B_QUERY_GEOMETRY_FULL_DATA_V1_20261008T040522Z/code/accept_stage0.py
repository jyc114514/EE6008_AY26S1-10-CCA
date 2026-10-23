#!/usr/bin/env python3
"""Accept training Stage0 first, then fresh-process inference smoke and measured peaks."""
from __future__ import annotations
import argparse,csv,json,math,hashlib
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def readj(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def receipt(job_id):
 p=ROOT/'coordination/JOB_LEDGER.jsonl';matches=[]
 for line in p.read_text(encoding='utf-8').splitlines():
  if line.strip():
   x=json.loads(line)
   if x.get('event')=='watcher_finished' and x.get('job_id')==job_id:matches.append(x)
 if not matches:return None
 return matches[-1]
def stage0_cells(qsha):
 rows=list(csv.DictReader((ROOT/'protocol/MATRIX12.csv').open(newline='',encoding='utf-8')));out=[]
 for config in ('TS11-D','STATE-CTRL'):
  run=next(x['run_id'] for x in rows if x['config']==config and int(x['seed'])==20261001 and x['action']=='train_had')
  rr=ROOT/'runs/stage0'/run;report=readj(rr/'STAGE0_REPORT.json');job_id=f'stage0-{config}-H30-S20261001';rec=receipt(job_id)
  if report.get('status')!='PASS' or int(report.get('optimizer_steps',-1))!=4 or int(report.get('scheduler_step_calls',-1))!=4 or int(report.get('legacy_epoch_step_calls',-1))!=1:raise RuntimeError(f'Stage0 training update/scheduler gate failed {config}')
  if report.get('query_table_sha256')!=qsha or report.get('parameter_count')!=890780 or report.get('candidate_trainable_parameters_added')!=0:raise RuntimeError(f'Stage0 query/parameter gate failed {config}')
  if sha(rr/'checkpoint_stage0.pt')!=report.get('checkpoint_sha256'):raise RuntimeError(f'Stage0 checkpoint hash failed {config}')
  if not rec or rec.get('watcher_exit_code')!=0 or rec.get('selected_physical_gpu') is None or not rec.get('gpu_uuid'):raise RuntimeError(f'Stage0 watcher receipt failed {config}')
  out.append({'config':config,'run_id':run,'seed':20261001,'stage0_updates':4,'stage0_checkpoint_sha256':report['checkpoint_sha256'],
   'query_table_sha256':qsha,'parameter_count':report['parameter_count'],'candidate_trainable_parameters_added':0,
   'training_peak_allocated_bytes':report['peak_allocated_bytes'],'training_peak_reserved_bytes':report['peak_reserved_bytes'],
   'training_peak_mib':int(math.ceil(report['peak_reserved_bytes']/2**20)),
   'training_physical_gpu':rec['selected_physical_gpu'],'training_gpu_uuid':rec['gpu_uuid'],'training_gpu_name':rec['gpu_name'],'training_watcher_receipt':rec['watcher_receipt']})
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('training','all'),default='training');args=ap.parse_args()
 lock=readj(ROOT/'protocol/PROTOCOL_LOCKED.json');qsha=lock['query_table_sha256_h30'];cells=stage0_cells(qsha)
 now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 train_accept={'schema_version':1,'created_utc':now,'status':'PASS_2_OF_2_STAGE0_TRAINING','cells':cells,
  'stage0_weights_used_for_formal':False,'dev_targets_opened':False,'final_test_access':0,'heldout_access':0}
 if not (ROOT/'qa/STAGE0_TRAINING_ACCEPTANCE.json').exists():
  (ROOT/'qa/STAGE0_TRAINING_ACCEPTANCE.json').write_text(json.dumps(train_accept,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 else:
  old=readj(ROOT/'qa/STAGE0_TRAINING_ACCEPTANCE.json')
  if old.get('status')!='PASS_2_OF_2_STAGE0_TRAINING':raise RuntimeError('existing Stage0 training acceptance is not PASS')
  train_accept=old
 train_peak=max(x['training_peak_mib'] for x in train_accept['cells'])
 margin=max(512,math.ceil(.1*train_peak))
 gate_train={'schema_version':1,'created_utc':now,'status':'PASS_STAGE0_TRAINING_RESOURCE_GATE','measured_training_peak_mib':train_peak,
  'safety_margin_mib':margin,'required_free_memory_mib':train_peak+margin,'source':'2 accepted 4-update candidate Stage0 cells',
  'pre_registered_peak_mib':1024,'final_test_access':0,'heldout_access':0}
 if not (ROOT/'protocol/RESOURCE_GATE_STAGE0_TRAINING.json').exists():
  (ROOT/'protocol/RESOURCE_GATE_STAGE0_TRAINING.json').write_text(json.dumps(gate_train,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if args.phase=='training':
  print(json.dumps({'status':train_accept['status'],'measured_training_peak_mib':train_peak,'required_free_mib':gate_train['required_free_memory_mib']},indent=2));return
 # Full canary acceptance is written only after both separate inference-smoke jobs pass.
 if not (ROOT/'qa/STAGE0_TRAINING_ACCEPTANCE.json').exists():raise RuntimeError('Stage0 training gate missing')
 combined=[]
 for cell in train_accept['cells']:
  rr=ROOT/'runs/stage0'/cell['run_id'];smoke=readj(rr/'INFER_CANARY.json');job_id=f"infer-smoke-{cell['config']}-H30-S20261001";rec=receipt(job_id)
  if smoke.get('status')!='PASS' or smoke.get('run_id')!=cell['run_id'] or smoke.get('query_table_sha256')!=qsha:raise RuntimeError(f'fresh-process inference smoke failed {cell["config"]}')
  if smoke.get('prediction_shape')!=[128,30,28] or smoke.get('target_arrays_loaded') or smoke.get('optimizer_created') or smoke.get('backward_called'):raise RuntimeError('inference smoke target/shape boundary failed')
  if not rec or rec.get('watcher_exit_code')!=0 or not rec.get('gpu_uuid'):raise RuntimeError(f'inference smoke watcher receipt failed {cell["config"]}')
  combined.append({**cell,'inference_smoke_rows':128,'inference_smoke_shape':smoke['prediction_shape'],
   'inference_smoke_sha256':smoke['prediction_sha256'],'inference_peak_allocated_bytes':smoke['peak_allocated_bytes'],
   'inference_peak_reserved_bytes':smoke['peak_reserved_bytes'],'inference_peak_mib':smoke['measured_peak_mib'],
   'fresh_process_reload':smoke['fresh_process_reload'],'inference_physical_gpu':rec['selected_physical_gpu'],
   'inference_gpu_uuid':rec['gpu_uuid'],'inference_gpu_name':rec['gpu_name'],'inference_watcher_receipt':rec['watcher_receipt']})
 acceptance={'schema_version':1,'created_utc':now,'status':'PASS_2_OF_2_TRAIN_AND_INFER_SMOKE','cells':combined,
  'candidate_query_sha256':qsha,'candidate_trainable_parameters_added':0,'dev_targets_opened':False,'final_test_access':0,'heldout_access':0,
  'stage0_updates_each':4,'stage0_weights_used_for_formal_training':False}
 canary=ROOT/'qa/CANARY_ACCEPTANCE.json'
 if canary.exists():raise FileExistsError(canary)
 canary.write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 measured=max(max(x['training_peak_mib'],x['inference_peak_mib']) for x in combined);margin=max(512,math.ceil(.1*measured))
 resource={'schema_version':1,'created_utc':now,'status':'PASS_STAGE0_RESOURCE_GATE','gpu_pool':'dynamic_all_visible',
  'measured_peak_mib':measured,'safety_margin_mib':margin,'required_free_memory_mib':measured+margin,
  'training_peak_mib_by_config':{x['config']:x['training_peak_mib'] for x in combined},
  'inference_peak_mib_by_config':{x['config']:x['inference_peak_mib'] for x in combined},
  'canary_cells':2,'updates_each':4,'formal_new_cells_authorized':6,
  'project_watcher_child_slots':'dynamic_visible_gpu_count_at_admission','final_test_access':0,'heldout_access':0}
 resource_path=ROOT/'protocol/RESOURCE_GATE_POST_STAGE0.json'
 if resource_path.exists():raise FileExistsError(resource_path)
 resource_path.write_text(json.dumps(resource,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':acceptance['status'],'cells':combined,'measured_peak_mib':measured,'required_free_memory_mib':measured+margin},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
