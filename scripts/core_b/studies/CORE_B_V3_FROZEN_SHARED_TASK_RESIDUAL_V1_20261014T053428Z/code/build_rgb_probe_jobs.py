#!/usr/bin/env python3
"""Build the locked 12 video-base zero/donor GPU diagnostic prediction jobs."""
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
CONDITIONS=('ZERO_NORMALIZED_BLOCK','SAME_TASK_DONOR')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if path.exists():
  try:
   old=json.loads(path.read_text());new=json.loads(text);old.pop('created_utc',None);new.pop('created_utc',None);equal=old==new
  except Exception:equal=path.read_bytes().decode('utf-8')==text
  if not equal:raise RuntimeError(f'preserving differing RGB probe config: {path}')
 else:path.write_text(text,encoding='utf-8')
def main():
 plan=json.loads((ROOT/'diagnostics/RGB_PROBE_PLAN.json').read_text())
 if plan.get('status')!='LOCKED_BEFORE_PROBE_PREDICTIONS' or plan.get('new_prediction_jobs_planned')!=12:raise RuntimeError('RGB probe plan is not locked')
 stage=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text());bases=stage['formal_bases']+stage['probe_control_bases']
 configs=[]
 for base in bases:
  if base['model_id'] not in {'G-V','S-V-90'}:continue
  for condition in CONDITIONS:
   model=base['model_id'];seed=int(base['seed']);modality='VIDEO'
   jid=f"v3_probe_{model.replace('-','_')}_s{seed}_{condition.lower()}"
   out=ROOT/'predictions/rgb_probe'/model/f'seed{seed}'/condition/'attempt_01'
   feature_root=ROOT/'features/formal'/f"G_V_s{seed}"/'attempt_01' if model=='G-V' else None
   ref=stage['reused_v2_accepted_prediction_references'][f'{model}:{seed}']
   job={'schema_version':1,'job_id':jid,'model_id':model,'seed':seed,'condition':condition,'base':base,
    'normalizer_path':base['normalizer_path'],'dev_prefix_path':str(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),
    'cache_manifest_path':str(json.loads((ROOT/'inventory/V3_PATHS.json').read_text())['cache_manifest']),
    'cache_manifest_sha256':json.loads((ROOT/'inventory/CACHE_AND_NORMALIZER_MAP.json').read_text())['v2_q0_cache_manifest_sha256'],
    'donor_map_path':str(ROOT/'diagnostics/RGB_DONOR_MAP.csv'),'matched_v2_prediction_reference':ref,
    'matched_feature_root':str(feature_root) if feature_root else None,
    'condition_definition':plan['zero_condition'] if condition=='ZERO_NORMALIZED_BLOCK' else plan['same_task_donor'],
    'output_root':str(out),'completion_marker':'PREDICTION_COMPLETE.json','final_test_access':0}
   jp=ROOT/'configs/jobs/RGB_PROBE'/f'{jid}.json';write(jp,job)
   resource=ROOT/'qa/V3_RESOURCE_ESTIMATE_LOCK.json'
   wc={'job_id':jid,'protocol_id':'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014','output_root':str(out),
    'completion_marker':'PREDICTION_COMPLETE.json','command':[PYTHON,str(ROOT/'code/probe_rgb.py'),'--job-json',str(jp)],
    'resource_estimate':{'estimate_type':'V3_STAGE0_MEASURED_UPPER_ENVELOPE','measured_peak_mib':128,
     'source_receipt':str(resource),'source_receipt_sha256':sha(resource),'batch_size':64,
     'two_base_forwards_per_sample_if_sv90':model=='S-V-90'},
    'log_path':str(ROOT/'logs/watcher_children'/f'{jid}.log'),'max_oom_requeues':0,
    'environment':{'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'},
    'final_test_access':0}
   wp=ROOT/'configs/watcher/RGB_PROBE'/f'{jid}.json';write(wp,wc);configs.append(str(wp))
 if len(configs)!=12:raise RuntimeError(f'expected 12 diagnostic prediction jobs; got {len(configs)}')
 dry=[]
 for raw in sorted(configs):
  p=Path(raw);w=json.loads(p.read_text());sp=ROOT/'coordination/watcher_status'/f"dryrun_{w['job_id']}.json"
  pr=subprocess.run([PYTHON,str(WATCHER),'--job-json',str(p),'--lease-dir',str(ROOT/'coordination/locks/gpu_lease'),
   '--status-path',str(sp),'--dry-run'],cwd=str(PROJECT),text=True,capture_output=True,check=False,
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
  if pr.returncode:raise RuntimeError(f"RGB watcher dry-run failed: {pr.stderr}")
  d=json.loads(pr.stdout)
  if d.get('status')!='valid':raise RuntimeError(f"RGB watcher invalid: {w['job_id']} {d}")
  dry.append({'job_id':w['job_id'],'status':d['status'],'required_free_memory_mib':d['required_free_memory_mib'],
   'measured_peak_mib':d['measured_peak_mib'],'watcher_child_cap':d['max_concurrent_project_gpu_children']})
 q=ROOT/'coordination/queues/RGB_PROBE.json'
 write(q,{'schema_version':1,'phase':'rgb_probe','jobs':sorted(configs),'job_count':len(configs),
  'new_condition_prediction_jobs':12,'condition_task_views_max':36,'matched_old_v2_task_views_reused':18,
  'matched_S_V_90_replay_forwards_itemized_in_jobs':6,'dev_targets_loaded':False,'final_test_access':0})
 receipt={'schema_version':1,'status':'PASS','phase':'rgb_probe','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'jobs_checked':len(dry),'gpu_device_queries':0,'results':dry,'labels_loaded':False,'final_heldout_payload_access':0}
 write(ROOT/'qa/RGB_PROBE_WATCHER_DRYRUN.json',receipt)
 print(json.dumps({'status':'PASS','jobs':len(configs),'views_max':36,'matched_old_views':18,'registered_peak_mib':128},indent=2))
if __name__=='__main__':main()
