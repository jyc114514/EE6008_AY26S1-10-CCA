#!/usr/bin/env python3
"""Persistent bounded dispatcher; every GPU child is launched via gpu_lease_watcher.py."""
from __future__ import annotations
import os
import argparse,csv,json,os,subprocess,sys,time,fcntl
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
LEASE_DIR=PROJECT/'repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease'
MAX_ACTIVE=2

def now():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def readj(p):return json.loads(p.read_text(encoding='utf-8'))
def writej(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def append_jsonl(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def sha_file(path):
 h=__import__('hashlib').sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def acquire_controller_lock(tmux_session,job_file):
 p=ROOT/'coordination/CONTROLLER.lock';p.parent.mkdir(parents=True,exist_ok=True);fh=p.open('a+',encoding='utf-8')
 try:fcntl.flock(fh.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  fh.close();raise RuntimeError('another controller holds this RUN_ROOT lock')
 record={'owner_key':'CORE_B_QUERY_GEOMETRY_FULL_DATA_H30_V1','pid':os.getpid(),'tmux_session':tmux_session,'job_file':job_file,'started_utc':now(),'lease_dir':str(LEASE_DIR)}
 fh.seek(0);fh.truncate();fh.write(json.dumps(record,ensure_ascii=False)+'\n');fh.flush();os.fsync(fh.fileno())
 return fh,record

def job_done(job):
 r=job['run_id'];phase=job['phase']
 try:
  if phase=='stage0':
   p=ROOT/'runs/stage0'/r/'STAGE0_REPORT.json';x=readj(p)
   return p.is_file() and x.get('status')=='PASS' and int(x.get('optimizer_steps',-1))==4
  if phase=='infer_smoke':
   p=ROOT/'runs/stage0'/r/'INFER_CANARY.json';x=readj(p)
   return p.is_file() and x.get('status')=='PASS' and x.get('prediction_shape')==[128,30,28]
  if phase=='formal':
   run=ROOT/'runs/formal'/r; p=run/'RUN_COMPLETE.json';x=readj(p)
   return p.is_file() and x.get('status')=='PASS' and int(x.get('epochs',-1))==40 and int(x.get('optimizer_steps',-1))==11120 and sha_file(run/'checkpoint_last.pt')==x.get('checkpoint_last_sha256')
  if phase=='inference':
   for split,expected in [('train',17757),('dev',2322)]:
    d=ROOT/'predictions'/split/r;accepted=False
    for side in d.glob('attempt_*.json') if d.exists() else []:
     x=readj(side);prediction=d/x.get('prediction_file','')
     if (x.get('status')=='PASS' and x.get('run_id')==r and x.get('split')==split and int(x.get('rows',-1))==expected
         and prediction.is_file() and sha_file(prediction)==x.get('prediction_sha256')):accepted=True;break
    if not accepted:return False
   return True
 except (OSError,ValueError,KeyError,json.JSONDecodeError):
  return False
 return False

def attempt_counts_from_ledger(path):
 counts={};ledger_path=Path(path)
 for line in ledger_path.read_text(encoding='utf-8').splitlines() if ledger_path.exists() else []:
  if not line.strip():continue
  rec=json.loads(line);jid=rec.get('job_id')
  if jid and rec.get('event')=='watcher_started':counts[jid]=counts.get(jid,0)+1
 return counts

def load_jobs(path):
 jobs=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
 if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('duplicate job IDs in queue')
 return jobs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--poll-seconds',type=float,default=2.0);ap.add_argument('--tmux-session',default=None);args=ap.parse_args()
 controller_lock,owner=acquire_controller_lock(args.tmux_session,args.jobs)
 queue=load_jobs(ROOT/args.jobs);ledger=ROOT/'coordination/JOB_LEDGER.jsonl';lease_ledger=ROOT/'coordination/GPU_LEASES.jsonl';state_path=ROOT/'coordination/RUN_STATE.json'
 append_jsonl(ledger,{'utc':now(),'event':'controller_started','owner_key':owner['owner_key'],'controller_pid':owner['pid'],'tmux_session':owner['tmux_session'],'job_file':args.jobs,'shared_lease_dir':str(LEASE_DIR),'project_child_limit':MAX_ACTIVE})
 pending=[]
 for j in queue:
  if job_done(j):
   append_jsonl(ledger,{'utc':now(),'event':'verified_skip','phase':j['phase'],'job_id':j['job_id'],'run_id':j['run_id'],'accepted_artifact_exists':True})
  else:pending.append(j)
 active={};completed=0;failed=[]
 writej(state_path,{'schema_version':1,'status':'running','phase':queue[0]['phase'] if queue else 'empty','job_file':args.jobs,'queued_jobs':len(queue),'pending_jobs':len(pending),'active_jobs':[],'started_utc':now(),'max_project_children':MAX_ACTIVE,'controller_owner':owner['owner_key'],'controller_pid':owner['pid'],'controller_started_utc':owner['started_utc'],'controller_heartbeat_utc':now(),'tmux_session':args.tmux_session,'shared_lease_dir':str(LEASE_DIR),'gpu_pool':'dynamic_all_visible'})
 attempt_counts=attempt_counts_from_ledger(ledger)
 while pending or active:
  while pending and len(active)<MAX_ACTIVE:
   job=pending.pop(0);attempt=attempt_counts.get(job['job_id'],0)+1;attempt_counts[job['job_id']]=attempt
   actual=dict(job);actual['log_path']=str(ROOT/'logs'/f"{job['job_id']}.attempt_{attempt:02d}.log")
   job_json=ROOT/'coordination/jobs/attempts'/f"{job['job_id']}.attempt_{attempt:02d}.json"
   status=ROOT/'receipts/watcher'/f"{job['job_id']}.attempt_{attempt:02d}.json"
   writej(job_json,actual)
   cmd=[sys.executable,str(WATCHER),'--job-json',str(job_json),'--lease-dir',str(LEASE_DIR),'--status-path',str(status),'--poll-seconds','15','--max-wait-seconds','0','--max-oom-requeues','0']
   log=(ROOT/'logs'/f"{job['job_id']}.watcher.attempt_{attempt:02d}.log");log.parent.mkdir(parents=True,exist_ok=True)
   fh=log.open('w',encoding='utf-8');proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=fh,stderr=subprocess.STDOUT,text=True)
   active[proc]={'job':job,'actual':actual,'attempt':attempt,'job_json':job_json,'status':status,'watcher_log':log,'handle':fh,'cmd':cmd,'started':now()}
   append_jsonl(ledger,{'utc':now(),'event':'watcher_started','phase':job['phase'],'job_id':job['job_id'],'run_id':job['run_id'],'attempt':attempt,'pid':proc.pid,'command':cmd,'cwd':str(ROOT),'job_json':str(job_json),'watcher_log':str(log),'status_path':str(status),'lease_dir':str(LEASE_DIR),'controller_pid':owner['pid'],'controller_owner':owner['owner_key']})
  finished=[]
  for proc,meta in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   meta['handle'].close();status=readj(meta['status']) if meta['status'].is_file() else {'status':'missing_receipt'}
   snapshot=status.get('final_recheck') or {}
   selected=int(status.get('selected_gpu',snapshot.get('index',-1))) if status.get('selected_gpu',snapshot.get('index')) is not None else -1
   uuid=snapshot.get('uuid')
   record={'utc':now(),'event':'watcher_finished','phase':meta['job']['phase'],'job_id':meta['job']['job_id'],'run_id':meta['job']['run_id'],
           'attempt':meta['attempt'],'watcher_exit_code':rc,'watcher_status':status.get('status'),'selected_physical_gpu':selected,
           'gpu_uuid':uuid,'gpu_name':snapshot.get('name'),'cuda_visible_devices':str(selected) if selected>=0 else None,
           'logical_device':'cuda:0','lease_released':status.get('lease_released'),'started_utc':meta['started'],
           'completed_utc':now(),'watcher_receipt':str(meta['status']),'watcher_log':str(meta['watcher_log']),
           'visible_snapshots':status.get('visible_snapshots'),'final_recheck':status.get('final_recheck'),'output_root':meta['job']['output_root'],
           'non_child_process_signaled':status.get('non_child_process_signaled',False)}
   append_jsonl(ledger,record)
   append_jsonl(lease_ledger,{'utc':now(),'job_id':meta['job']['job_id'],'run_id':meta['job']['run_id'],'attempt':meta['attempt'],
      'status':status.get('status'),'selected_physical_gpu':selected,'gpu_uuid':uuid,'gpu_name':snapshot.get('name'),
      'final_recheck':status.get('final_recheck'),'lease_released':status.get('lease_released')})
   accepted=job_done(meta['job'])
   if rc!=0 or not accepted:failed.append({'job_id':meta['job']['job_id'],'exit_code':rc,'artifact_gate_pass':accepted,'receipt':str(meta['status'])})
   else:completed+=1
   finished.append(proc)
  for proc in finished:active.pop(proc,None)
  writej(state_path,{'schema_version':1,'status':'running' if pending or active else ('partial' if failed else 'phase_complete'),
      'phase':queue[0]['phase'] if queue else 'empty','job_file':args.jobs,'queued_jobs':len(queue),'completed_jobs':completed,
      'failed_jobs':failed,'pending_jobs':[j['job_id'] for j in pending],'active_jobs':[v['job']['job_id'] for v in active.values()],
      'updated_utc':now(),'controller_heartbeat_utc':now(),'controller_pid':owner['pid'],'controller_owner':owner['owner_key'],'tmux_session':args.tmux_session,'shared_lease_dir':str(LEASE_DIR),'max_project_children':MAX_ACTIVE,'gpu_pool':'dynamic_all_visible','final_test_access':0})
  if pending or active:time.sleep(args.poll_seconds)
 final={'schema_version':1,'status':'partial' if failed else 'phase_complete','phase':queue[0]['phase'] if queue else 'empty',
        'completed_jobs':completed,'failed_jobs':failed,'completed_utc':now(),'controller_pid':owner['pid'],'controller_owner':owner['owner_key'],'tmux_session':args.tmux_session,'shared_lease_dir':str(LEASE_DIR),'gpu_pool':'dynamic_all_visible','max_project_children':MAX_ACTIVE}
 writej(state_path,final);print(json.dumps(final,ensure_ascii=False,indent=2));return 1 if failed else 0
if __name__=='__main__':raise SystemExit(main())
