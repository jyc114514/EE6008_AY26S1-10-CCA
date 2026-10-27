#!/usr/bin/env python3
"""Build CPU-dry-run watcher configs for two bounded Stage0 smoke jobs."""
from __future__ import annotations
import os
import hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
WATCHER=Path(json.loads((ROOT/'inventory/V3_PATHS.json').read_text())['shared_watcher'])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if path.exists():
  if path.read_text(encoding='utf-8')!=text:raise RuntimeError(f'preserving differing config {path}')
 else:path.write_text(text,encoding='utf-8')

def main():
 accepted=json.loads((ROOT/'inventory/V2_SOURCE_ACCEPTANCE.json').read_text())
 actual=sha(WATCHER)
 expected=accepted['shared_watcher_sha256']
 if actual!=expected:raise RuntimeError(f'shared watcher hash mismatch: {actual} != {expected}')
 stage=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text())
 queue=[];dry=[]
 for base in sorted(stage['screening_bases'],key=lambda x:x['model_id']):
  model_slug=base['model_id'].replace('-','_')
  attempt=1
  while True:
   suffix='' if attempt==1 else f'_a{attempt:02d}'
   job=f"v3_stage0_{model_slug}_s{base['seed']}{suffix}"
   out=ROOT/'qa/STAGE0'/base['model_id']/f'attempt_{attempt:02d}'
   if not out.exists():break
   marker=out/'SMOKE_COMPLETE.json'
   if marker.is_file() and json.loads(marker.read_text()).get('status')=='PASS':break
   attempt+=1
   if attempt>20:raise RuntimeError('Stage0 attempt limit exceeded')
  fit={'schema_version':1,'job_id':job,'model_id':base['model_id'],'seed':int(base['seed']),
   'stage':'screening','video_mode':base['video_mode'],'base':base,'train_prefix_path':str(V2/'data/TRAIN_SELECTED_H30.npz'),
   'cache_manifest_path':str(json.loads((ROOT/'inventory/V3_PATHS.json').read_text())['cache_manifest']),
   'output_root':str(out),'final_test_access':0}
  fitpath=ROOT/'configs/jobs/STAGE0'/f'{job}.json';write(fitpath,fit)
  wc={'job_id':job,'protocol_id':'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014',
   'output_root':str(out),'completion_marker':'SMOKE_COMPLETE.json',
   'command':[PYTHON,str(ROOT/'code/stage0_smoke.py'),'--job-json',str(fitpath)],
   'resource_estimate':{'estimate_type':'V3_STAGE0_CONSERVATIVE_PRE_REGISTERED','measured_peak_mib':1024,
      'basis':'V2 accepted screening model peak allocated 124-138 MiB/reserved 170-178 MiB; Stage0 adds one frozen-base 64-row forward plus 3,612-parameter readout backward; 1 GiB is a pre-registered conservative estimate before same-run measurement',
      'v2_source_models':[x['job_id'] for x in stage['screening_bases']]},
   'log_path':str(ROOT/'logs/watcher_children'/f'{job}.log'),'max_oom_requeues':0,
   'environment':{'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'},
   'final_test_access':0}
  wcpath=ROOT/'configs/watcher/STAGE0'/f'{job}.json';write(wcpath,wc);queue.append(str(wcpath))
  status=ROOT/'coordination/watcher_status'/f'dryrun_{job}.json'
  proc=subprocess.run([PYTHON,str(WATCHER),'--job-json',str(wcpath),'--lease-dir',str(ROOT/'coordination/locks/gpu_lease'),
    '--status-path',str(status),'--dry-run'],cwd=str(PROJECT),text=True,capture_output=True,check=False,
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
  if proc.returncode:raise RuntimeError(f'Stage0 watcher dry-run failed for {job}: {proc.stderr}')
  result=json.loads(proc.stdout)
  if result.get('status')!='valid':raise RuntimeError(f'Stage0 watcher contract invalid for {job}: {result}')
  dry.append({'job_id':job,'status':result['status'],'measured_peak_mib':result.get('measured_peak_mib'),
    'required_free_memory_mib':result.get('required_free_memory_mib'),'visible_gpu_pool':result.get('visible_gpu_pool'),
    'watcher_child_cap':result.get('max_concurrent_project_gpu_children'),'gpu_device_queries':0})
 attempt_ids=[Path(x).stem.rsplit('_a',1)[-1] if '_a' in Path(x).stem else '01' for x in queue]
 attempt=max(int(x) for x in attempt_ids)
 phase=f'stage0_smoke_attempt{attempt:02d}'
 queue_path=ROOT/'coordination/queues'/f'STAGE0_ATTEMPT_{attempt:02d}.json'
 write(queue_path,{'schema_version':1,'phase':phase,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'jobs':sorted(queue),'job_count':len(queue),'neural_fit_count':0,'smoke_optimizer_updates':2,'attempt':attempt,'final_test_access':0})
 dry_receipt={'schema_version':1,'status':'PASS','phase':phase,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'jobs_checked':len(dry),'gpu_device_queries':0,'labels_loaded':False,'results':dry,'final_heldout_payload_access':0}
 write(ROOT/'qa'/f'STAGE0_WATCHER_DRYRUN_ATTEMPT_{attempt:02d}.json',dry_receipt)
 print(json.dumps({'status':'PASS','queue':str(queue_path),'jobs':len(queue),'max_estimate_mib':1024,
   'shared_watcher_sha256':actual,'dryrun':dry},indent=2))
if __name__=='__main__':main()
