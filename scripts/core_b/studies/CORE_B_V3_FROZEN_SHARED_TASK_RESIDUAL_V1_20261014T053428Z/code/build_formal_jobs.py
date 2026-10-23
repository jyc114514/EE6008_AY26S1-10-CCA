#!/usr/bin/env python3
"""Build 12 fresh formal readout fits, inheriting only locked hyperparameters."""
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
SEEDS=(202610131,202610132,202610133)
PROGRAMS=('SHARED_RES_STATE','TASK_RES_STATE','SHARED_RES_VIDEO','TASK_RES_VIDEO')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if path.exists():
  try:
   old=json.loads(path.read_text(encoding='utf-8'));new=json.loads(content);old.pop('created_utc',None);new.pop('created_utc',None);equal=old==new
  except Exception:equal=path.read_bytes().decode('utf-8')==content
  if not equal:raise RuntimeError(f'preserving differing V3 config: {path}')
 else:path.write_text(content,encoding='utf-8')
def main():
 selection_path=ROOT/'protocol/READOUT_SELECTION.json';selection=json.loads(selection_path.read_text());sel_sha=sha(selection_path)
 if selection.get('status') not in {'LOCKED','LOCKED_WITH_PARTIAL_SEARCH'}:raise RuntimeError(f'readout selection is not usable: {selection.get("status")}')
 if len(selection.get('choices',[]))!=4:raise RuntimeError('selection receipt must include the four predeclared programs')
 readout_lock=json.loads((ROOT/'protocol/MODEL_READOUT_LOCK.json').read_text())
 bases=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text())['formal_bases']
 configs=[]
 for choice in selection['choices']:
  program=choice['program']
  if choice.get('selected_lambda')=='NA' or choice.get('selected_epoch')=='NA':continue
  if program not in PROGRAMS:raise RuntimeError(f'unplanned program: {program}')
  base_id='G-S' if program.endswith('_STATE') else 'G-V'
  for seed in SEEDS:
   base=next(x for x in bases if x['model_id']==base_id and int(x['seed'])==seed)
   feature_root=ROOT/'features/formal'/f"{base_id.replace('-','_')}_s{seed}"/'attempt_01'
   feature_complete=feature_root/'FEATURES_COMPLETE.json'
   if not feature_complete.is_file() or json.loads(feature_complete.read_text()).get('status')!='PASS':raise RuntimeError(f'formal base features not accepted: {feature_complete}')
   attempt=1
   model_root=ROOT/'model_only/formal'/program/f'seed{seed}'
   while True:
    suffix='' if attempt==1 else f'_a{attempt:02d}'
    jid=f"v3_formal_{program.lower()}_s{seed}{suffix}"
    out=model_root/f'attempt_{attempt:02d}'
    if not out.exists():break
    marker=out/'FIT_COMPLETE.json'
    if marker.is_file() and json.loads(marker.read_text()).get('status')=='PASS':break
    attempt+=1
    if attempt>20:raise RuntimeError(f'formal attempt limit exceeded for {program}/{seed}')
   fit={'schema_version':1,'job_id':jid,'stage':'formal','program':program,'base_model_id':base_id,'base':base,
    'seed':seed,'lambda':float(choice['selected_lambda']),'lr':0.001,'epochs':int(choice['selected_epoch']),
    'checkpoint_epochs':[int(choice['selected_epoch'])],'selection_sha256':sel_sha,'selection_status':selection['status'],
    'selection_tune_J':float(choice['selected_tune_j']),'feature_path':str(feature_root/'TRAIN_FEATURES.safetensors'),
    'feature_rows_path':str(feature_root/'TRAIN_ROWS.csv'),'feature_complete_path':str(feature_complete),
    'train_prefix_path':str(V2/'data/TRAIN_SELECTED_H30.npz'),'fold_membership_path':str(ROOT/'inventory/ACTUAL_FOLD_MEMBERSHIP.csv'),
    'metric_scale_lock_path':str(V2/'protocol/METRIC_SCALE_LOCK.json'),'output_root':str(out),'final_test_access':0}
   fitp=ROOT/'configs/jobs/FORMAL'/f'{jid}.json';write(fitp,fit)
   resource=ROOT/'qa/V3_RESOURCE_ESTIMATE_LOCK.json'
   watcher={'job_id':jid,'protocol_id':'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014','output_root':str(out),
    'completion_marker':'FIT_COMPLETE.json','command':[PYTHON,str(ROOT/'code/train_readout.py'),'--job-json',str(fitp)],
    'resource_estimate':{'estimate_type':'V3_STAGE0_MEASURED_UPPER_ENVELOPE','measured_peak_mib':128,
      'source_receipt':str(resource),'source_receipt_sha256':sha(resource),'selection_sha256':sel_sha,'batch_size':64},
    'log_path':str(ROOT/'logs/watcher_children'/f'{jid}.log'),'max_oom_requeues':0,
    'environment':{'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'},
    'final_test_access':0}
   wp=ROOT/'configs/watcher/FORMAL'/f'{jid}.json';write(wp,watcher);configs.append(str(wp))
 dry=[]
 for raw in sorted(configs):
  wp=Path(raw);w=json.loads(wp.read_text());sp=ROOT/'coordination/watcher_status'/f"dryrun_{w['job_id']}.json"
  pr=subprocess.run([PYTHON,str(WATCHER),'--job-json',str(wp),'--lease-dir',str(ROOT/'coordination/locks/gpu_lease'),
   '--status-path',str(sp),'--dry-run'],cwd=str(PROJECT),text=True,capture_output=True,check=False,
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
  if pr.returncode:raise RuntimeError(f"formal watcher dry-run failed {w['job_id']}: {pr.stderr}")
  d=json.loads(pr.stdout)
  if d.get('status')!='valid':raise RuntimeError(f"formal watcher invalid {w['job_id']}: {d}")
  dry.append({'job_id':w['job_id'],'status':d['status'],'required_free_memory_mib':d['required_free_memory_mib'],
   'measured_peak_mib':d['measured_peak_mib'],'selection_sha256':sel_sha,'watcher_child_cap':d['max_concurrent_project_gpu_children']})
 attempt_nums=[]
 for raw in configs:
  fit=json.loads(Path(json.loads(Path(raw).read_text())['command'][-1]).read_text())
  attempt_nums.append(int(Path(fit['output_root']).name.split('_')[-1]))
 attempt=max(attempt_nums) if attempt_nums else 1
 phase='formal_readout' if attempt==1 else f'formal_readout_attempt{attempt:02d}'
 qname='FORMAL_READOUT.json' if attempt==1 else f'FORMAL_READOUT_ATTEMPT_{attempt:02d}.json'
 q=ROOT/'coordination/queues'/qname
 write(q,{'schema_version':1,'phase':phase,'attempt':attempt,'jobs':sorted(configs),'job_count':len(configs),'planned_fit_count':12,
  'selection_sha256':sel_sha,'formal_seeds':list(SEEDS),'final_test_access':0})
 receipt={'schema_version':1,'status':'PASS','phase':'formal_readout','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'jobs_checked':len(dry),'results':dry,'gpu_device_queries':0,'labels_loaded':False,'selection_sha256':sel_sha,'final_heldout_payload_access':0}
 dry_name='FORMAL_READOUT_WATCHER_DRYRUN.json' if attempt==1 else f'FORMAL_READOUT_ATTEMPT_{attempt:02d}_WATCHER_DRYRUN.json'
 write(ROOT/'qa'/dry_name,receipt)
 print(json.dumps({'status':'PASS' if len(configs)==12 else 'PARTIAL','formal_jobs':len(configs),'attempt':attempt,'queue':str(q),'selection_sha256':sel_sha,'dryrun_rows':len(dry)},indent=2))
if __name__=='__main__':main()
