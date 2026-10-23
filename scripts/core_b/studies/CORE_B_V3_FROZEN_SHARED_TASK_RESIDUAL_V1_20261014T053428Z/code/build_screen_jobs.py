#!/usr/bin/env python3
"""Register eight V3 inner-fit readout-screening jobs."""
from __future__ import annotations
import os
import hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
WATCHER=Path(json.loads((ROOT/'inventory/V3_PATHS.json').read_text())['shared_watcher'])
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
PROGRAMS=(('SHARED_RES_STATE','G-S'),('TASK_RES_STATE','G-S'),('SHARED_RES_VIDEO','G-V'),('TASK_RES_VIDEO','G-V'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if path.exists():
  try:
   old_obj=json.loads(path.read_text(encoding='utf-8'));new_obj=json.loads(content)
   old_obj.pop('created_utc',None);new_obj.pop('created_utc',None)
   equal=old_obj==new_obj
  except Exception:equal=path.read_bytes().decode('utf-8')==content
  if not equal:raise RuntimeError(f'preserving differing config: {path}')
 else:path.write_text(content,encoding='utf-8')
def main():
 source=json.loads((ROOT/'inventory/V2_SOURCE_ACCEPTANCE.json').read_text())
 actual=sha(WATCHER)
 if actual!=source['shared_watcher_sha256']:raise RuntimeError('shared watcher source hash differs from V2 accepted policy')
 resource=ROOT/'qa/V3_RESOURCE_ESTIMATE_LOCK.json';resource_receipt=json.loads(resource.read_text())
 if resource_receipt.get('status')!='LOCKED_AFTER_STAGE0':raise RuntimeError('Stage0 measured resource lock missing')
 if resource_receipt['registered_v3_peak_mib']!=128:raise RuntimeError('registered screening estimate does not match Stage0 measurement')
 configs=[]
 for program,base_id in PROGRAMS:
  base=next(x for x in json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text())['screening_bases'] if x['model_id']==base_id)
  feature_root=ROOT/'features/screening'/f"{base_id.replace('-','_')}_s{base['seed']}"/'attempt_01'
  complete=feature_root/'FEATURES_COMPLETE.json'
  if not complete.is_file() or json.loads(complete.read_text()).get('status')!='PASS':raise RuntimeError(f'screening base features not accepted: {complete}')
  for lam in (0.0,1.0):
   jid=f"v3_screen_{program.lower()}_lambda{int(lam)}_s{base['seed']}"
   out=ROOT/'model_only/screening'/program/f"lambda{int(lam)}_s{base['seed']}"/'attempt_01'
   fit={'schema_version':1,'job_id':jid,'stage':'screening','program':program,'base_model_id':base_id,'base':base,
    'seed':int(base['seed']),'lambda':lam,'lr':0.001,'epochs':80,'checkpoint_epochs':[20,40,80],
    'feature_path':str(feature_root/'TRAIN_FEATURES.safetensors'),'feature_rows_path':str(feature_root/'TRAIN_ROWS.csv'),
    'feature_complete_path':str(complete),'train_prefix_path':str(V2/'data/TRAIN_SELECTED_H30.npz'),
    'fold_membership_path':str(ROOT/'inventory/ACTUAL_FOLD_MEMBERSHIP.csv'),
    'metric_scale_lock_path':str(V2/'protocol/METRIC_SCALE_LOCK.json'),
    'output_root':str(out),'final_test_access':0}
   fitp=ROOT/'configs/jobs/SCREENING'/f'{jid}.json';write(fitp,fit)
   watcher={'job_id':jid,'protocol_id':'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014','output_root':str(out),
    'completion_marker':'FIT_COMPLETE.json','command':[PYTHON,str(ROOT/'code/train_readout.py'),'--job-json',str(fitp)],
    'resource_estimate':{'estimate_type':'V3_STAGE0_MEASURED_UPPER_ENVELOPE','measured_peak_mib':128,
     'source_receipt':str(resource),'source_receipt_sha256':sha(resource),'stage0_max_reserved_mib':resource_receipt['stage0_max_measured_peak_reserved_mib'],
     'batch_size':64,'readout_parameters_max':18060},'log_path':str(ROOT/'logs/watcher_children'/f'{jid}.log'),
    'max_oom_requeues':0,'environment':{'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'},
    'final_test_access':0}
   wp=ROOT/'configs/watcher/SCREENING'/f'{jid}.json';write(wp,watcher);configs.append(str(wp))
 # Dry-run all configs before any optimizer job.
 dry=[]
 for raw in sorted(configs):
  wp=Path(raw);w=json.loads(wp.read_text());sp=ROOT/'coordination/watcher_status'/f"dryrun_{w['job_id']}.json"
  pr=subprocess.run([PYTHON,str(WATCHER),'--job-json',str(wp),'--lease-dir',str(ROOT/'coordination/locks/gpu_lease'),
   '--status-path',str(sp),'--dry-run'],cwd=str(PROJECT),text=True,capture_output=True,check=False,
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
  if pr.returncode:raise RuntimeError(f"watcher dry-run failed for {w['job_id']}: {pr.stderr}")
  d=json.loads(pr.stdout)
  if d.get('status')!='valid':raise RuntimeError(f"invalid V3 fit watcher config {w['job_id']}: {d}")
  dry.append({'job_id':w['job_id'],'status':d['status'],'required_free_memory_mib':d['required_free_memory_mib'],
   'measured_peak_mib':d['measured_peak_mib'],'watcher_child_cap':d['max_concurrent_project_gpu_children'],'gpu_device_queries':0})
 q=ROOT/'coordination/queues/SCREENING_READOUT.json'
 write(q,{'schema_version':1,'phase':'screening_readout','fit_count':len(configs),'expected_fit_count':8,'jobs':sorted(configs),
  'candidate_grid':'4 programs x lambda{0,1}; checkpoints 20/40/80','final_test_access':0})
 receipt={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'phase':'screening_readout','jobs_checked':len(dry),'results':dry,'gpu_device_queries':0,'labels_loaded':False,'final_heldout_payload_access':0}
 write(ROOT/'qa/SCREENING_READOUT_WATCHER_DRYRUN.json',receipt)
 print(json.dumps({'status':'PASS','fit_jobs':len(configs),'programs':4,'lambda_values':[0,1],'checkpoints':[20,40,80],
  'registered_peak_mib':128,'queue':str(q)},indent=2))
if __name__=='__main__':main()
