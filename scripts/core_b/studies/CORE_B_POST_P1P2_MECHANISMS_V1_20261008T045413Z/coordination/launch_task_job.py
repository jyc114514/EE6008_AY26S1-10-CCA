#!/usr/bin/env python3
"""Start one registered GPU/CPU job in tmux and verify its real owner/status."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
NEXT_ROOT=PROJECT/'reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z'
N0_ROOT=NEXT_ROOT/'n0'
N1_ROOT=PROJECT/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
PYTHON=PROJECT/'envs/jyc/bin/python'
WATCHER=NEXT_ROOT/'coordination/gpu_lease_watcher_task.py'
CPU_WRAPPER=NEXT_ROOT/'coordination/run_cpu_job.py'
SHARED_LEASES=PROJECT/'repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease'
RECORDS=NEXT_ROOT/'coordination/launch_records'

def now():return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def readj(p):return json.loads(p.read_text(encoding='utf-8'))
def in_owned_root(path:Path)->bool:
 p=path.resolve()
 return any(p.is_relative_to(r.resolve()) for r in (NEXT_ROOT,N1_ROOT))
def has_session(name:str)->bool:
 return subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
def main()->int:
 ap=argparse.ArgumentParser()
 ap.add_argument('--kind',choices=('gpu','cpu'),required=True)
 ap.add_argument('--job-json',type=Path,required=True)
 ap.add_argument('--status-path',type=Path,required=True)
 ap.add_argument('--session',required=True)
 ap.add_argument('--cwd',type=Path,required=True)
 ap.add_argument('--launch',action='store_true')
 args=ap.parse_args()
 if not args.session.replace('_','').replace('-','').isalnum():raise ValueError('unsafe tmux session name')
 job_path=args.job_json.resolve();status_path=args.status_path.resolve();cwd=args.cwd.resolve()
 if not in_owned_root(job_path) or not in_owned_root(status_path) or not in_owned_root(cwd):raise RuntimeError('job/status/cwd must stay in NEXT_ROOT or resumed N1_ROOT')
 if status_path.exists():raise FileExistsError(status_path)
 job=readj(job_path);job_id=str(job['job_id'])
 out=Path(job['output_root']).resolve()
 if not in_owned_root(out):raise RuntimeError('job output_root must stay in NEXT_ROOT or resumed N1_ROOT')
 if not job.get('command') or not all(isinstance(x,str) for x in job['command']):raise ValueError('job command is not a valid argv list')
 env={'PYTHONDONTWRITEBYTECODE':'1',**{str(k):str(v) for k,v in job.get('environment',{}).items()}}
 if args.kind=='gpu':
  if not WATCHER.is_file():raise FileNotFoundError(WATCHER)
  dry=[str(PYTHON),str(WATCHER),'--job-json',str(job_path),'--lease-dir',str(SHARED_LEASES),'--status-path',str(status_path),'--dry-run']
  check=subprocess.run(dry,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  if check.returncode!=0:raise RuntimeError(f'watcher dry-run failed: {check.stderr}')
  dry_record=json.loads(check.stdout)
  if dry_record.get('status')!='valid' or dry_record.get('visible_gpu_pool')!='dynamic_all_visible':raise RuntimeError('task watcher did not validate dynamic all-visible job')
  command=['env',*(f'{key}={value}' for key,value in env.items()),str(PYTHON),str(WATCHER),'--job-json',str(job_path),'--lease-dir',str(SHARED_LEASES),'--status-path',str(status_path),'--poll-seconds','15','--max-wait-seconds','0','--max-oom-requeues','0']
  wrapper_log=NEXT_ROOT/'logs'/f'{job_id}.{job_path.stem}.watcher.tmux.log'
  shell=shlex.join(command)+' > '+shlex.quote(str(wrapper_log))+' 2>&1'
 else:
  command=['env',*(f'{k}={v}' for k,v in env.items()),str(PYTHON),str(CPU_WRAPPER),'--job-json',str(job_path),'--status-path',str(status_path)]
  wrapper_log=NEXT_ROOT/'logs'/f'{job_id}.{job_path.stem}.cpu.tmux.log'
  shell=shlex.join(command)+' > '+shlex.quote(str(wrapper_log))+' 2>&1'
 record_path=RECORDS/f'{job_id}.{job_path.stem}.json'
 if record_path.exists():raise FileExistsError(record_path)
 plan={'job_id':job_id,'kind':args.kind,'session':args.session,'job_json':str(job_path),'status_path':str(status_path),
       'output_root':str(out),'log_path':str(job.get('log_path',wrapper_log)),'wrapper_log':str(wrapper_log),
       'working_directory':str(cwd),'argv':command,'tmux_shell_command':shell,'shared_lease_dir':str(SHARED_LEASES) if args.kind=='gpu' else None,
       'dynamic_pool':'dynamic_all_visible' if args.kind=='gpu' else None,'dry_run':dry_record if args.kind=='gpu' else None,'created_utc':now()}
 if not args.launch:
  print(json.dumps(plan,indent=2,sort_keys=True,ensure_ascii=False));return 0
 if has_session(args.session):raise RuntimeError(f'tmux session already exists: {args.session}')
 wrapper_log.parent.mkdir(parents=True,exist_ok=True);RECORDS.mkdir(parents=True,exist_ok=True)
 result=subprocess.run(['tmux','new-session','-d','-s',args.session,'-c',str(cwd),shell],capture_output=True,text=True)
 if result.returncode!=0:raise RuntimeError(f'tmux new-session failed: {result.stderr}')
 pane=subprocess.run(['tmux','display-message','-p','-t',args.session,'#{pane_pid}'],capture_output=True,text=True,check=True).stdout.strip()
 deadline=time.monotonic()+20
 while time.monotonic()<deadline and not status_path.is_file() and has_session(args.session):time.sleep(.1)
 status=readj(status_path) if status_path.is_file() else None
 session_alive=has_session(args.session)
 plan.update({'launch_status':'started' if status else 'session_without_status','pane_pid':int(pane) if pane.isdigit() else pane,
              'watcher_pid':status.get('watcher_pid') if status else None,'child_pid':status.get('child_pid') if status else None,
              'watcher_status':status.get('status') if status else None,'physical_gpu':status.get('selected_gpu') if status else None,
              'lease_path':status.get('lease_path') if status else None,'status_seen':status is not None,'tmux_session_active':session_alive,
              'launch_verified':bool(status is not None and session_alive),'launch_utc':now()})
 record_path.write_text(json.dumps(plan,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(plan,sort_keys=True,ensure_ascii=False))
 return 0 if plan['launch_verified'] or (status and status.get('status') in ('complete','failed')) else 1
if __name__=='__main__':raise SystemExit(main())
