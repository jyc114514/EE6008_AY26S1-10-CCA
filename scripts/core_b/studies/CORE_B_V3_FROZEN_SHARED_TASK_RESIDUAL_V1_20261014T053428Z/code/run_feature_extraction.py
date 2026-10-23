#!/usr/bin/env python3
"""Own the V3 lock while extracting screening/formal frozen-base h and z only."""
from __future__ import annotations
import os
import fcntl,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
SESSION='ee6008-v3-residual-20261014'
LOCK=ROOT/'coordination/OWNER.lock'
STATE=ROOT/'coordination/OWNER_STATE.json'
LOG=ROOT/'logs/V3_PIPELINE.log'
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def log(msg):
 line=f'{utc()} {msg}';print(line,flush=True)
 with LOG.open('a',encoding='utf-8') as f:f.write(line+'\n')
def set_owner(phase,status,active,next_gate):
 prior=json.loads(STATE.read_text()) if STATE.exists() else {}
 atomic(STATE,{'schema_version':1,'created_utc':prior.get('created_utc',utc()),'updated_utc':utc(),
  'out_root':str(ROOT),'owner_lock_acquired':status=='running','owner_lock_path':str(LOCK),
  'owner_pid':os.getpid(),'tmux_session':SESSION,'phase':phase,'status':status,'active_jobs':active,
  'parent_root_read_only':True,'final_heldout_payload_access':0,'next_gate':next_gate})
def run_queue(name):
 qp=ROOT/'coordination/queues'/name
 with LOG.open('a',encoding='utf-8') as out:
  log(f'START phase={name} queue={qp}')
  res=subprocess.run([PYTHON,str(ROOT/'code/run_v3_queue.py'),'--queue',str(qp)],cwd=str(PROJECT),stdout=out,stderr=subprocess.STDOUT,check=False,
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
  log(f'END phase={name} returncode={res.returncode}')
 return res.returncode

def main():
 LOCK.parent.mkdir(parents=True,exist_ok=True)
 with LOCK.open('a+') as handle:
  try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('another V3 owner holds the root lock')
  set_owner('screening_and_formal_base_feature_extraction','running',2,'two screening and six formal frozen-base feature queues via shared watcher')
  try:
   for name in ('SCREEN_FEATURES.json','FORMAL_FEATURES.json'):
    set_owner(name.replace('.json','').lower(),'running',2,f'waiting for {name}')
    if run_queue(name):raise RuntimeError(f'feature queue failed: {name}')
   set_owner('feature_extraction_complete','preparing',0,'run 8 screening readout fits after inspecting feature manifests')
   log('FEATURE_EXTRACTION_PHASE_PASS screening=2/2 formal=6/6 target_reads=0 heldout=0')
  except Exception as exc:
   set_owner('feature_extraction_failure','partial',0,f'inspect preserved status/logs and resume: {type(exc).__name__}: {exc}')
   log(f'FEATURE_EXTRACTION_FAILURE {type(exc).__name__}: {exc}')
   raise
if __name__=='__main__':main()
