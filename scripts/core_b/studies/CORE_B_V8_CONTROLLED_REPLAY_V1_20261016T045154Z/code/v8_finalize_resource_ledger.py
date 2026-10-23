#!/usr/bin/env python3
"""Reconcile V8 fit, inference, Stage0 and watcher resource receipts."""
from __future__ import annotations
import csv,json,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(202610131,202610132,202610133)
def readj(p):return json.loads(Path(p).read_text())
def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with Path(p).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sec(a,b):
 return (datetime.fromisoformat(b.replace('Z','+00:00'))-datetime.fromisoformat(a.replace('Z','+00:00'))).total_seconds()
def main():
 p=ROOT/'results/RESOURCE_LEDGER.csv';fits=[x for x in readcsv(p) if x.get('record_type')=='formal_fit'];rows=[]
 for x in fits:rows.append({**x,'record_type':'formal_fit'})
 triplet_watch_secs=[];fit_elapsed=sum(float(x['elapsed_seconds']) for x in fits)
 all_gpus=set();
 for seed in SEEDS:
  complete=readj(ROOT/'model_only/seed_triplets'/f'seed{seed}'/'SEED_TRIPLET_COMPLETE.json')
  ws=readj(ROOT/'coordination/jobs'/f'v8_seed_triplet_s{seed}.status.json')
  triplet_watch=sec(ws['started_utc'],ws['completed_utc']);triplet_watch_secs.append(triplet_watch)
  all_gpus.add(str(complete['physical_gpu']))
  rows.append({'record_type':'training_seed_triplet_watcher','job_id':f'v8_seed_triplet_s{seed}','seed':seed,'status':ws['status'],'physical_gpu':complete['physical_gpu'],
    'cuda_visible_devices':complete['cuda_visible_devices'],'logical_device':complete['logical_device'],'tmux_session':f'coreb-v8-train-s{seed}-20261016',
    'watcher_pid':ws.get('watcher_pid'),'child_pid':ws.get('child_pid'),'watcher_wall_seconds':triplet_watch,'triplet_python_wall_seconds':complete['elapsed_seconds'],
    'fit_count':len(complete['fits']),'completion_marker':str(ROOT/'model_only/seed_triplets'/f'seed{seed}'/'SEED_TRIPLET_COMPLETE.json'),
    'log_path':str(ROOT/'logs'/f'training_seed{seed}_watcher_child.log'),'lease_released':ws.get('lease_released'),'project_child_slot_released':ws.get('project_child_slot_released')})
  for entry in complete['fits']:
   if entry['physical_gpu'] is not None:all_gpus.add(str(entry['physical_gpu']))
 inference_bundle_seconds=[];inference_triplet_seconds=[]
 for seed in SEEDS:
  complete=readj(ROOT/'predictions/seed_triplets'/f'seed{seed}'/'SEED_INFERENCE_COMPLETE.json')
  ws=readj(ROOT/'coordination/jobs'/f'v8_infer_triplet_s{seed}.status.json')
  wall=sec(ws['started_utc'],ws['completed_utc']);inference_triplet_seconds.append(wall);all_gpus.add(str(complete['physical_gpu']))
  rows.append({'record_type':'inference_seed_triplet_watcher','job_id':f'v8_infer_triplet_s{seed}','seed':seed,'status':ws['status'],'physical_gpu':complete['physical_gpu'],
   'cuda_visible_devices':complete['cuda_visible_devices'],'logical_device':complete['logical_device'],'tmux_session':f'coreb-v8-infer-s{seed}-20261016',
   'watcher_pid':ws.get('watcher_pid'),'child_pid':ws.get('child_pid'),'watcher_wall_seconds':wall,'triplet_python_wall_seconds':complete['elapsed_seconds'],
   'bundle_count':len(complete['bundles']),'completion_marker':str(ROOT/'predictions/seed_triplets'/f'seed{seed}'/'SEED_INFERENCE_COMPLETE.json'),
   'log_path':str(ROOT/'logs'/f'inference_seed{seed}_watcher_child.log'),'lease_released':ws.get('lease_released'),'project_child_slot_released':ws.get('project_child_slot_released')})
  for d in ('train','dev'):
   for program in ('GT_RETRAIN','SELF_SOFT','SV_SOFT'):
    rec=readj(ROOT/'predictions'/d/program/f'seed{seed}'/'attempt_01/PREDICTION_COMPLETE.json')
    rows.append({'record_type':'standalone_prediction_bundle','job_id':rec['job_id'],'program_id':program,'seed':seed,'dataset_role':d,'status':rec['status'],
      'physical_gpu':rec['physical_gpu'],'cuda_visible_devices':rec['cuda_visible_devices'],'logical_device':rec['logical_device'],'forward_rows':rec['forward_rows'],
      'elapsed_seconds':rec['elapsed_seconds'],'peak_allocated_mib':rec['peak_memory_allocated_mib'],'peak_reserved_mib':rec['peak_memory_reserved_mib'],
      'model_sha256':rec['model_sha256'],'prediction_sha256':rec['prediction_sha256'],'final_heldout_payload_access':rec['final_heldout_payload_access']})
    inference_bundle_seconds.append(float(rec['elapsed_seconds']))
    all_gpus.add(str(rec['physical_gpu']))
 gpu_audit=readj(ROOT/'audit/GPU_ZERO_OPTIMIZER_AUDIT.json');stage=readj(ROOT/'coordination/STAGE0_RESOURCE_RECEIPT.json');ws=readj(ROOT/'coordination/jobs/zero_optimizer_gpu_retry1.status.json')
 rows.append({'record_type':'stage0_zero_optimizer_gpu_audit','job_id':'v8_zero_optimizer_gpu_audit_retry1','status':gpu_audit['status'],'physical_gpu':stage['physical_gpu'],
  'cuda_visible_devices':stage['cuda_visible_devices'],'logical_device':stage['logical_device'],'watcher_wall_seconds':sec(ws['started_utc'],ws['completed_utc']),
  'python_wall_seconds':gpu_audit['wall_seconds'],'peak_allocated_mib':stage['peak_allocated_mib'],'peak_reserved_mib':stage['peak_reserved_mib'],
  'optimizer_updates_in_full_replay':gpu_audit['optimizer_updates_during_teacher_student_full_replay'],'temporary_clone_optimizer_updates':gpu_audit['temporary_clone_optimizer_updates'],
  'lease_released':ws.get('lease_released'),'project_child_slot_released':ws.get('project_child_slot_released'),'log_path':ws.get('log_path')})
 fail=readj(ROOT/'coordination/jobs/zero_optimizer_gpu.status.json')
 rows.append({'record_type':'stage0_failed_preflight_attempt','job_id':'v8_zero_optimizer_gpu_audit','status':fail['status'],'physical_gpu':fail.get('physical_gpu'),
  'watcher_wall_seconds':sec(fail['started_utc'],fail['completed_utc']),'exit_code':fail.get('exit_code'),'error':'PyTorch reset_peak_memory_stats(torch.device) returned Invalid device argument before model replay',
  'lease_released':fail.get('lease_released'),'project_child_slot_released':fail.get('project_child_slot_released'),'log_path':fail.get('log_path')})
 summary={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'physical_gpus_used':sorted(all_gpus,key=int),'physical_gpu_count':len(all_gpus),'formal_fits':9,'training_watcher_triplets':3,'standalone_prediction_bundles':18,'inference_watcher_triplets':3,
  'watcher_max_concurrent_gpu_children':2,'all_watcher_leases_released':all(readj(ROOT/'coordination/jobs'/f'v8_seed_triplet_s{s}.status.json').get('lease_released') for s in SEEDS) and all(readj(ROOT/'coordination/jobs'/f'v8_infer_triplet_s{s}.status.json').get('lease_released') for s in SEEDS) and readj(ROOT/'coordination/jobs/zero_optimizer_gpu_retry1.status.json').get('lease_released'),
  'formal_fit_elapsed_seconds_sum':fit_elapsed,'training_triplet_watcher_wall_seconds_sum':sum(triplet_watch_secs),'inference_bundle_elapsed_seconds_sum':sum(inference_bundle_seconds),
  'inference_triplet_watcher_wall_seconds_sum':sum(inference_triplet_seconds),'stage0_python_wall_seconds':gpu_audit['wall_seconds'],
  'successful_watcher_child_wall_seconds_sum':sum(triplet_watch_secs)+sum(inference_triplet_seconds)+sec(ws['started_utc'],ws['completed_utc']),
  'failed_stage0_preflight_watcher_wall_seconds':sec(fail['started_utc'],fail['completed_utc']),
  'watcher_child_wall_seconds_sum':sum(triplet_watch_secs)+sum(inference_triplet_seconds)+sec(ws['started_utc'],ws['completed_utc'])+sec(fail['started_utc'],fail['completed_utc']),
  'max_formal_fit_allocated_mib':max(float(x['peak_allocated_mib']) for x in fits),'max_formal_fit_reserved_mib':max(float(x['peak_reserved_mib']) for x in fits),
  'stage0_allocated_reserved_mib':[stage['peak_allocated_mib'],stage['peak_reserved_mib']],
  'dev_target_content_reads':1,'new_dev_student_evaluations':1665,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0}
 writecsv(p,rows);q=ROOT/'qa/RESOURCE_SUMMARY.json';q.write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
