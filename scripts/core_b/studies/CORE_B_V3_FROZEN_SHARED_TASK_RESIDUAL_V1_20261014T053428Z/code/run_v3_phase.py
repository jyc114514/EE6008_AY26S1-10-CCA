#!/usr/bin/env python3
"""Own the V3 run lock during one resumable lease-watcher queue phase."""
from __future__ import annotations
import os
import argparse,fcntl,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
SESSION='ee6008-v3-residual-20261014'
LOCK=ROOT/'coordination/OWNER.lock';STATE=ROOT/'coordination/OWNER_STATE.json';LOG=ROOT/'logs/V3_PIPELINE.log'
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def log(message):
 line=f'{utc()} {message}';print(line,flush=True)
 with LOG.open('a',encoding='utf-8') as f:f.write(line+'\n')
def write_owner(phase,status,active,next_gate):
 old=json.loads(STATE.read_text()) if STATE.exists() else {}
 atomic(STATE,{'schema_version':1,'created_utc':old.get('created_utc',utc()),'updated_utc':utc(),'out_root':str(ROOT),
  'owner_lock_acquired':status=='running','owner_lock_path':str(LOCK),'owner_pid':os.getpid(),'tmux_session':SESSION,
  'phase':phase,'status':status,'active_jobs':active,'parent_root_read_only':True,'final_heldout_payload_access':0,'next_gate':next_gate})
def run(args):
 q=ROOT/'coordination/queues'/args.queue
 with LOCK.open('a+') as h:
  try:fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as exc:raise RuntimeError('another V3 owner holds the root lock') from exc
  write_owner(args.phase,'running',2,f'waiting on shared watcher queue {args.queue}')
  log(f'PHASE_START name={args.phase} queue={q} pid={os.getpid()} cwd={PROJECT}')
  with LOG.open('a',encoding='utf-8') as out:
   result=subprocess.run([PYTHON,str(ROOT/'code/run_v3_queue.py'),'--queue',str(q)],cwd=str(PROJECT),stdout=out,stderr=subprocess.STDOUT,check=False,
     env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
  if result.returncode:
   write_owner(f'{args.phase}_partial','partial',0,f'inspect queue receipt/logs and preserve failures: {args.queue}')
   log(f'PHASE_PARTIAL name={args.phase} returncode={result.returncode}')
   return result.returncode
  if args.select_readout:
   log('START inner_tune_selection')
   with LOG.open('a',encoding='utf-8') as out:
    selected=subprocess.run([PYTHON,str(ROOT/'code/select_readout.py')],cwd=str(PROJECT),stdout=out,stderr=subprocess.STDOUT,check=False,
      env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
   log(f'END inner_tune_selection returncode={selected.returncode}')
   if selected.returncode:
    write_owner('selection_partial','partial',0,'inspect preserved screening candidates; selection did not lock')
    return selected.returncode
   selection=json.loads((ROOT/'protocol/READOUT_SELECTION.json').read_text())
   next_gate='build formal readout jobs from READOUT_SELECTION.json' if selection['status'] in {'LOCKED','LOCKED_WITH_PARTIAL_SEARCH'} else 'inspect programs with no valid tune candidate; no dev/final access'
  else:next_gate='continue V3 task from latest PASS phase and protocol locks'
  write_owner(f'{args.phase}_complete','preparing',0,next_gate)
  log(f'PHASE_PASS name={args.phase} next_gate={next_gate}')
  return 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--queue',required=True);p.add_argument('--select-readout',action='store_true');raise SystemExit(run(p.parse_args()))
if __name__=='__main__':main()
