#!/usr/bin/env python3
"""Build 2 screening and 6 matching formal frozen-base feature jobs."""
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
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if path.exists():
  if path.read_bytes().decode('utf-8')!=content:raise RuntimeError(f'preserving differing V3 config: {path}')
 else:path.write_text(content,encoding='utf-8')
def build(base,stage):
 seed=int(base['seed']);mid=base['model_id']
 jid=f"v3_feature_{stage}_{mid.replace('-','_')}_s{seed}"
 datasets=['train'] if stage=='screening' else ['train','dev']
 out=ROOT/'features'/stage/f"{mid.replace('-','_')}_s{seed}"/'attempt_01'
 fit={'schema_version':1,'job_id':jid,'stage':stage,'model_id':mid,'seed':seed,'base':base,'datasets':datasets,
  'train_prefix_path':str(V2/'data/TRAIN_SELECTED_H30.npz'),'dev_prefix_path':str(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),
  'cache_manifest_path':str(json.loads((ROOT/'inventory/V3_PATHS.json').read_text())['cache_manifest']),
  'cache_manifest_sha256':json.loads((ROOT/'inventory/CACHE_AND_NORMALIZER_MAP.json').read_text())['v2_q0_cache_manifest_sha256'],
  'batch_size':64,'output_root':str(out),'future_labels_in_forward':False,'final_test_access':0}
 fp=ROOT/'configs/jobs/FEATURES'/f'{jid}.json';write(fp,fit)
 watcher={'job_id':jid,'protocol_id':'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014','output_root':str(out),
  'completion_marker':'FEATURES_COMPLETE.json','command':[PYTHON,str(ROOT/'code/extract_base_features.py'),'--job-json',str(fp)],
  'resource_estimate':{'estimate_type':'V3_STAGE0_MEASURED_UPPER_ENVELOPE','measured_peak_mib':128,
   'source_receipt':str(ROOT/'qa/V3_RESOURCE_ESTIMATE_LOCK.json'),'source_receipt_sha256':sha(ROOT/'qa/V3_RESOURCE_ESTIMATE_LOCK.json'),
   'stage0_max_peak_reserved_mib':74.0,'batch_size':64,'basis':'same V2 base forward and larger Stage0 readout-backward pass; rounded upper envelope'},
  'log_path':str(ROOT/'logs/watcher_children'/f'{jid}.log'),'max_oom_requeues':1,
  'environment':{'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'},'final_test_access':0}
 wp=ROOT/'configs/watcher/FEATURES'/f'{jid}.json';write(wp,watcher)
 return jid,wp

def dry_run(queue_path):
 q=json.loads(queue_path.read_text());rows=[]
 for raw in q['jobs']:
  p=Path(raw);c=json.loads(p.read_text());status=ROOT/'coordination/watcher_status'/f"dryrun_{c['job_id']}.json"
  pr=subprocess.run([PYTHON,str(WATCHER),'--job-json',str(p),'--lease-dir',str(ROOT/'coordination/locks/gpu_lease'),
    '--status-path',str(status),'--dry-run'],cwd=str(PROJECT),text=True,capture_output=True,check=False,
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
  if pr.returncode:raise RuntimeError(f"watcher dry-run failed {c['job_id']}: {pr.stderr}")
  d=json.loads(pr.stdout)
  if d.get('status')!='valid':raise RuntimeError(f"invalid watcher contract {c['job_id']}: {d}")
  rows.append({'job_id':c['job_id'],'status':d['status'],'measured_peak_mib':d['measured_peak_mib'],
   'required_free_memory_mib':d['required_free_memory_mib'],'visible_gpu_pool':d['visible_gpu_pool'],
   'watcher_child_cap':d['max_concurrent_project_gpu_children'],'gpu_device_queries':0})
 receipt={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'phase':q['phase'],'jobs_checked':len(rows),'results':rows,'gpu_device_queries':0,'labels_loaded':False,'final_heldout_payload_access':0}
 dest=ROOT/'qa'/f"{q['phase'].upper()}_WATCHER_DRYRUN.json"
 write(dest,receipt)
 return receipt

def main():
 v2map=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text())
 jobs={'SCREEN_FEATURES':[],'FORMAL_FEATURES':[]}
 for base in v2map['screening_bases']:
  _,p=build(base,'screening');jobs['SCREEN_FEATURES'].append(str(p))
 for base in v2map['formal_bases']:
  _,p=build(base,'formal');jobs['FORMAL_FEATURES'].append(str(p))
 result={}
 for name,paths in jobs.items():
  phase=name.lower()
  qpath=ROOT/'coordination/queues'/f'{name}.json'
  write(qpath,{'schema_version':1,'phase':phase,'jobs':sorted(paths),'job_count':len(paths),'final_test_access':0})
  result[name]=dry_run(qpath)
 print(json.dumps({'status':'PASS','screen_jobs':result['SCREEN_FEATURES']['jobs_checked'],
  'formal_jobs':result['FORMAL_FEATURES']['jobs_checked'],'screen_queue':str(ROOT/'coordination/queues/SCREEN_FEATURES.json'),
  'formal_queue':str(ROOT/'coordination/queues/FORMAL_FEATURES.json'),'dryrun_peak_mib':128,
  'final_heldout_payload_access':0},indent=2))
if __name__=='__main__':main()
