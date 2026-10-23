#!/usr/bin/env python3
"""Tmux CPU-job wrapper with task-owned PID/status/log records."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
N1_ROOT=PROJECT/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
NEXT_ROOT=PROJECT/'reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z'

def now():return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def atomic_json(path:Path,value:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 tmp.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);ap.add_argument('--status-path',type=Path,required=True);args=ap.parse_args()
 job=json.loads(args.job_json.read_text(encoding='utf-8'))
 jobid=str(job['job_id']);output=Path(job['output_root']).resolve();status=args.status_path.resolve()
 if not any(output.is_relative_to(root.resolve()) for root in (N1_ROOT,NEXT_ROOT)):raise RuntimeError('CPU output root outside authorized N1/NEXT_ROOT')
 if not any(status.is_relative_to(root.resolve()) for root in (N1_ROOT,NEXT_ROOT)):raise RuntimeError('CPU status path outside authorized roots')
 command=job.get('command')
 if not isinstance(command,list) or not command or not all(isinstance(x,str) and x for x in command):raise ValueError('invalid CPU command')
 if status.exists():raise FileExistsError(status)
 env=os.environ.copy();env.update({str(k):str(v) for k,v in job.get('environment',{}).items()});env['PYTHONDONTWRITEBYTECODE']='1'
 log=Path(job.get('log_path',output/'cpu_job.log')).resolve()
 if not any(log.is_relative_to(root.resolve()) for root in (N1_ROOT,NEXT_ROOT)):raise RuntimeError('CPU log path outside authorized roots')
 cwd=Path(job.get('cwd',N1_ROOT)).resolve()
 if not any(cwd.is_relative_to(root.resolve()) for root in (N1_ROOT,NEXT_ROOT)):raise RuntimeError('CPU cwd outside authorized roots')
 base={'schema_version':1,'job_id':jobid,'phase':job.get('phase'),'status':'launching','command':command,
       'command_sha256':hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest(),
       'cwd':str(cwd),'output_root':str(output),'log_path':str(log),'wrapper_pid':os.getpid(),'started_utc':now(),
       'gpu_access':False,'target_arrays_opened_by_wrapper':False,'non_child_process_signaled':False,'final_test_access':0,'heldout_access':0}
 atomic_json(status,base)
 child=None;rc=127;error=None
 original=signal.getsignal(signal.SIGTERM)
 def forward(signum,_frame):
  if child is not None and child.poll() is None:child.send_signal(signum)
 signal.signal(signal.SIGTERM,forward)
 try:
  log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w',encoding='utf-8') as fh:
   child=subprocess.Popen(command,cwd=str(cwd),env=env,stdout=fh,stderr=subprocess.STDOUT,text=True)
   base.update({'status':'running','child_pid':child.pid,'launched_utc':now()});atomic_json(status,base)
   rc=child.wait()
 except Exception as exc:
  error=f'{type(exc).__name__}: {exc}'
 finally:
  signal.signal(signal.SIGTERM,original)
 base.update({'status':'complete' if rc==0 and error is None else 'failed','exit_code':rc,'error':error,'completed_utc':now()})
 atomic_json(status,base)
 print(json.dumps({'job_id':jobid,'status':base['status'],'exit_code':rc,'child_pid':base.get('child_pid')},sort_keys=True),flush=True)
 return rc
if __name__=='__main__':raise SystemExit(main())
