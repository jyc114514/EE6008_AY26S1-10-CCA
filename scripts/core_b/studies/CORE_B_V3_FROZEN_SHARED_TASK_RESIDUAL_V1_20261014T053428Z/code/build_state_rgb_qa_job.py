#!/usr/bin/env python3
"""Register one fixed-fixture, nonzero R_STATE RGB-invariance GPU QA."""
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
def write(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if p.exists():
  old=json.loads(p.read_text());new=json.loads(t);old.pop('created_utc',None);new.pop('created_utc',None)
  if old!=new:raise RuntimeError(f'preserving differing state RGB QA config {p}')
 else:p.write_text(t,encoding='utf-8')
def accepted(program,seed):
 root=ROOT/'model_only/formal'/program/f'seed{seed}';found=[]
 for p in root.glob('attempt_*/FIT_COMPLETE.json'):
  d=json.loads(p.read_text())
  if d.get('status')=='PASS':found.append((p,d))
 if len(found)!=1:raise RuntimeError(f'expected one accepted {program}/{seed} readout, found {len(found)}')
 return found[0]
def main():
 seed=202610131;base=next(x for x in json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text())['formal_bases'] if x['model_id']=='G-S' and int(x['seed'])==seed)
 shared_m,shared_fit=accepted('SHARED_RES_STATE',seed);task_m,task_fit=accepted('TASK_RES_STATE',seed)
 jid='v3_state_rgb_invariance_seed202610131';out=ROOT/'qa/STATE_RGB_INVARIANCE/attempt_01'
 job={'schema_version':1,'job_id':jid,'seed':seed,'base':base,
  'shared_readout_path':shared_fit['readout_path'],'shared_fit_complete_path':str(shared_m),
  'task_readout_path':task_fit['readout_path'],'task_fit_complete_path':str(task_m),
  'dev_prefix_path':str(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),
  'cache_manifest_path':str(json.loads((ROOT/'inventory/V3_PATHS.json').read_text())['cache_manifest']),
  'donor_map_path':str(ROOT/'diagnostics/RGB_DONOR_MAP.csv'),'output_root':str(out),'final_test_access':0}
 jp=ROOT/'configs/jobs/STATE_RGB_QA'/f'{jid}.json';write(jp,job)
 receipt=ROOT/'qa/V3_RESOURCE_ESTIMATE_LOCK.json'
 wc={'job_id':jid,'protocol_id':'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014','output_root':str(out),
  'completion_marker':'STATE_RGB_QA_COMPLETE.json','command':[PYTHON,str(ROOT/'code/state_rgb_invariance_qa.py'),'--job-json',str(jp)],
  'resource_estimate':{'estimate_type':'V3_STAGE0_MEASURED_UPPER_ENVELOPE','measured_peak_mib':128,
   'source_receipt':str(receipt),'source_receipt_sha256':sha(receipt),'batch_size':1},
  'log_path':str(ROOT/'logs/watcher_children'/f'{jid}.log'),'max_oom_requeues':0,
  'environment':{'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'},
  'final_test_access':0}
 wp=ROOT/'configs/watcher/STATE_RGB_QA'/f'{jid}.json';write(wp,wc)
 status=ROOT/'coordination/watcher_status'/f'dryrun_{jid}.json'
 pr=subprocess.run([PYTHON,str(WATCHER),'--job-json',str(wp),'--lease-dir',str(ROOT/'coordination/locks/gpu_lease'),
  '--status-path',str(status),'--dry-run'],cwd=str(PROJECT),text=True,capture_output=True,check=False,
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
 if pr.returncode:raise RuntimeError(pr.stderr)
 d=json.loads(pr.stdout)
 if d.get('status')!='valid':raise RuntimeError(f'invalid state RGB QA watcher config: {d}')
 q=ROOT/'coordination/queues/STATE_RGB_QA.json';write(q,{'schema_version':1,'phase':'state_rgb_qa','jobs':[str(wp)],'job_count':1,
  'fixture_tasks':[119,94,90,141,230],'conditions':['MATCHED','ZERO_NORMALIZED_BLOCK','SAME_TASK_DONOR'],'final_test_access':0})
 print(json.dumps({'status':'PASS','queue':str(q),'required_free_memory_mib':d['required_free_memory_mib'],'registered_peak_mib':128},indent=2))
if __name__=='__main__':main()
