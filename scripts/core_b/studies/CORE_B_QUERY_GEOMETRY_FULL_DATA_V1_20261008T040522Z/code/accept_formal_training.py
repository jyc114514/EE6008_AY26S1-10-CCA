#!/usr/bin/env python3
"""CPU-side acceptance of the six locked Q1 H30 formal training cells."""
from __future__ import annotations
import os
import csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ETA_MIN=1e-5;LR0=1e-3;T_MAX=11120;UPDATES_PER_EPOCH=278

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def cosine_lr(step):return ETA_MIN+(LR0-ETA_MIN)*(1+math.cos(math.pi*step/T_MAX))/2
def watcher_finishes():
 records=[]
 for line in (ROOT/'coordination/JOB_LEDGER.jsonl').read_text(encoding='utf-8').splitlines():
  if not line.strip():continue
  x=json.loads(line)
  if x.get('event')=='watcher_finished' and x.get('phase')=='formal':records.append(x)
 best={}
 for x in records:
  if x['job_id'] not in best or int(x['attempt'])>int(best[x['job_id']]['attempt']):best[x['job_id']]=x
 return best
def validate_lr_trace(path):
 with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=T_MAX:raise AssertionError(f'LR trace rows {len(rows)} != {T_MAX}')
 max_error=0.0
 for i,row in enumerate(rows):
  before=int(row['update_count_before']);after=int(row['update_count_after']);epoch=int(row['epoch'])
  if before!=i or after!=i+1 or epoch!=i//UPDATES_PER_EPOCH+1:raise AssertionError(f'LR trace counter/epoch mismatch at update {i}')
  used=float(row['lr_used_for_update']);after_lr=float(row['lr_after_update'])
  err=max(abs(used-cosine_lr(i)),abs(after_lr-cosine_lr(i+1)));max_error=max(max_error,err)
  if not math.isclose(used,cosine_lr(i),rel_tol=1e-8,abs_tol=2e-12) or not math.isclose(after_lr,cosine_lr(i+1),rel_tol=1e-8,abs_tol=2e-12):raise AssertionError(f'cosine LR mismatch at update {i}')
 return {'rows':len(rows),'first_update_lr_used':float(rows[0]['lr_used_for_update']),'last_update_lr_used':float(rows[-1]['lr_used_for_update']),'lr_after_update_11120':float(rows[-1]['lr_after_update']),'max_closed_form_abs_error':max_error,'sha256':sha(path)}
def main():
 lock=readj(ROOT/'protocol/PROTOCOL_LOCKED.json');lock_sha=sha(ROOT/'protocol/PROTOCOL_LOCKED.json')
 amendment=readj(ROOT/'protocol/AMENDMENT_001_QUERY_ARCHIVE.json');amend_sha=sha(ROOT/'protocol/AMENDMENT_001_QUERY_ARCHIVE.json')
 amendment2=readj(ROOT/'protocol/AMENDMENT_002_PREDICTION_ACCEPTANCE.json');amend2_sha=sha(ROOT/'protocol/AMENDMENT_002_PREDICTION_ACCEPTANCE.json')
 amendment3=readj(ROOT/'protocol/AMENDMENT_003_FORMAL_ACCEPTANCE.json');amend3_sha=sha(ROOT/'protocol/AMENDMENT_003_FORMAL_ACCEPTANCE.json')
 receipt=readj(ROOT/'coordination/AMENDMENT_003_RECEIPT.json');own_sha=sha(Path(__file__).resolve())
 if (lock.get('status')!='LOCKED' or amendment.get('parent_protocol_sha256')!=lock_sha
     or amendment2.get('parent_protocol_sha256')!=lock_sha or amendment2.get('parent_amendment_001_sha256')!=amend_sha
     or amendment3.get('parent_protocol_sha256')!=lock_sha or amendment3.get('parent_amendment_002_sha256')!=amend2_sha
     or amendment3.get('task_code_sha256_overrides',{}).get('code/accept_formal_training.py')!=own_sha
     or receipt.get('amendment_sha256')!=amend3_sha):raise AssertionError('protocol/amendment chain or receipt mismatch')
 with (ROOT/'protocol/MATRIX12.csv').open(newline='',encoding='utf-8') as f:matrix=list(csv.DictReader(f))
 cells=[x for x in matrix if x['action']=='train_had']
 if len(cells)!=6:raise AssertionError('locked matrix must have six new Hadamard cells')
 with (ROOT/'protocol/SEED_INIT_MANIFEST.csv').open(newline='',encoding='utf-8') as f:inits={x['seed']:x for x in csv.DictReader(f)}
 watcher=watcher_finishes();results=[]
 for row in cells:
  run_id=row['run_id'];config=row['config'];seed=row['seed'];run=ROOT/'runs/formal'/run_id
  complete=readj(run/'RUN_COMPLETE.json');status=readj(run/'RUN_STATUS.json');env=readj(run/'RUN_ENVIRONMENT.json')
  cp=run/'checkpoint_last.pt';cph=sha(cp);lr=validate_lr_trace(run/'LR_TRACE_ACTUAL.csv')
  job_id=f'train-{config}-H30-S{seed}';receipt=watcher.get(job_id)
  if (complete.get('status')!='PASS' or status.get('status') not in ('PASS','complete') or int(complete.get('epochs',-1))!=40
      or int(complete.get('optimizer_steps',-1))!=11120 or int(complete.get('scheduler_step_calls',-1))!=11120
      or int(complete.get('legacy_epoch_step_calls',-1))!=40 or not math.isclose(float(complete.get('scheduler_lr_after_last_update',-1)),ETA_MIN,rel_tol=0,abs_tol=2e-11)
      or complete.get('checkpoint_last_sha256')!=cph or complete.get('study_protocol_hash')!=lock_sha
      or complete.get('query_table_sha256')!=lock['query_table_sha256_h30'] or complete.get('init_hash')!=inits[seed]['init_state_hash']
      or int(complete.get('parameter_count',-1))!=890780 or int(complete.get('candidate_trainable_parameters_added',-1))!=0
      or complete.get('protocol_amendment_id')!=amendment['amendment_id'] or complete.get('protocol_amendment_sha256')!=amend_sha):
   raise AssertionError(f'formal run completion/protocol/checkpoint contract failed: {run_id}')
  epochs=list(csv.DictReader((run/'EPOCH_LOG.csv').open(newline='',encoding='utf-8')))
  monitors=list(csv.DictReader((run/'TRAIN_MONITOR.csv').open(newline='',encoding='utf-8')))
  if len(epochs)!=40 or len(monitors)!=40:raise AssertionError(f'epoch/train-monitor row count failed: {run_id}')
  for i,entry in enumerate(epochs,1):
   if int(entry['epoch'])!=i or int(entry['optimizer_steps_total'])!=i*UPDATES_PER_EPOCH or int(entry['rows_seen'])!=17757:raise AssertionError(f'epoch/update/population mismatch: {run_id} epoch{i}')
  for i,entry in enumerate(monitors,1):
   if int(entry['epoch'])!=i:raise AssertionError(f'train monitor epoch mismatch: {run_id}')
  if not receipt or receipt.get('watcher_exit_code')!=0 or receipt.get('watcher_status')!='complete' or not receipt.get('lease_released') or receipt.get('non_child_process_signaled'):
   raise AssertionError(f'watcher acceptance/lease-release failed: {run_id}')
  phys=int(receipt['selected_physical_gpu']);uuid=receipt['gpu_uuid']
  if str(env.get('physical_gpu_index'))!=str(phys) or env.get('cuda_visible_devices')!=str(phys) or complete.get('physical_gpu_index')!=str(phys) or complete.get('cuda_visible_devices')!=str(phys) or receipt.get('logical_device')!='cuda:0':
   raise AssertionError(f'physical/logical GPU mapping mismatch: {run_id}')
  expected_sources=complete.get('execution_code_sha256s',{})
  for code_path,code_sha in expected_sources.items():
   if sha(ROOT/code_path)!=code_sha:raise AssertionError(f'run source code changed after execution: {run_id}/{code_path}')
   if code_path in amendment['task_code_sha256_overrides'] and amendment['task_code_sha256_overrides'][code_path]!=code_sha:raise AssertionError(f'run amendment code hash mismatch: {run_id}/{code_path}')
  required_sources={'code/train_full_query.py','code/d3_model.py','code/d3_eq_contract.py','code/corrected_scheduler.py','code/fixed_future_query_model.py','code/run_orchestrator.py'}
  if not required_sources.issubset(expected_sources):raise AssertionError(f'run source hash manifest incomplete: {run_id}')
  results.append({'run_id':run_id,'config':config,'seed':int(seed),'horizon_steps':30,'query_table_sha256':lock['query_table_sha256_h30'],'init_state_hash':complete['init_hash'],'checkpoint_sha256':cph,'run_complete_sha256':sha(run/'RUN_COMPLETE.json'),'epochs':40,'optimizer_updates':11120,'scheduler_step_calls':11120,'legacy_epoch_step_calls':40,'lr_trace':lr,'train_monitor_rows':40,'parameter_count':890780,'candidate_trainable_parameters_added':0,'protocol_amendment_id':amendment['amendment_id'],'protocol_amendment_sha256':amend_sha,'formal_acceptance_amendment_id':amendment3['amendment_id'],'formal_acceptance_amendment_sha256':amend3_sha,'run_environment_sha256':sha(run/'RUN_ENVIRONMENT.json'),'execution_code_sha256s':complete['execution_code_sha256s'],'physical_gpu_index':phys,'gpu_uuid':uuid,'cuda_visible_devices':str(phys),'logical_device':'cuda:0','elapsed_seconds':complete.get('elapsed_seconds'),'watcher_receipt':receipt.get('watcher_receipt'),'lease_released':True,'final_test_access':0,'heldout_access':0})
 if len(results)!=6:raise AssertionError('six formal training receipts were not accepted')
 result={'schema_version':1,'status':'PASS_6_OF_6','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'protocol_sha256':lock_sha,'amendment_id':amendment['amendment_id'],'amendment_sha256':amend_sha,'formal_acceptance_amendment_id':amendment3['amendment_id'],'formal_acceptance_amendment_sha256':amend3_sha,'matrix_sha256':lock['matrix12_sha256'],'q0_reused_cells':6,'old_control_repairs':0,'q1_hadamard_cells_accepted':6,'optimizer_updates_each':11120,'formal_cells':results,'final_test_access':0,'heldout_access':0}
 out=ROOT/'qa/FORMAL_TRAINING_ACCEPTANCE.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':result['status'],'accepted_cells':len(results),'assigned_gpu_uuids':sorted(set(x['gpu_uuid'] for x in results))},indent=2))
if __name__=='__main__':main()
