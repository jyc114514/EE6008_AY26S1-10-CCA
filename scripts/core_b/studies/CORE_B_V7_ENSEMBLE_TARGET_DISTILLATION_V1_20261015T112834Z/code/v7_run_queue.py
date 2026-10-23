#!/usr/bin/env python3
"""Run one bounded V7 queue through the immutable shared GPU watcher."""
from __future__ import annotations
import os
import argparse,concurrent.futures,hashlib,json,os,re,subprocess,threading
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python');WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
LEASE=ROOT/'coordination/locks/gpu_lease';STATUS=ROOT/'coordination/watcher_status';LOGS=ROOT/'logs/console'
OWNER=ROOT/'coordination/OWNER_STATE.json';LOCK=threading.Lock();ACTIVE=0

def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p:Path,obj)->None:
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def set_owner(active_delta=0,**updates):
    global ACTIVE
    with LOCK:
        ACTIVE=max(0,ACTIVE+active_delta);state=json.loads(OWNER.read_text(encoding='utf-8'))
        if state.get('status')=='complete':raise RuntimeError('V7 owner was already released')
        state.update(updates);state['active_jobs']=ACTIVE;state['updated_utc']=utc();state['owner_pid']=os.getpid()
        atomic_json(OWNER,state)
def run_one(job:dict)->dict:
    jobid=job['job_id'];out=Path(job['output_root']).resolve()
    if (ROOT/'coordination/STOP_REQUESTED').exists():return {'job_id':jobid,'status':'stopped_before_launch','returncode':125}
    if ROOT.resolve() not in out.parents:raise RuntimeError(f'job output escapes V7 root: {out}')
    marker=out/str(job.get('completion_marker','RUN_COMPLETE.json'))
    if marker.is_file():
        try:
            done=json.loads(marker.read_text(encoding='utf-8'))
            if done.get('status')=='PASS':return {'job_id':jobid,'status':'already_complete','returncode':0,'marker':str(marker)}
        except Exception:pass
    job_path=ROOT/'coordination/jobs'/f'{jobid}.watcher.json';atomic_json(job_path,job)
    status=STATUS/f'{jobid}.json';log=LOGS/f'{jobid}.watcher.log';STATUS.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    cmd=[PYTHON,str(WATCHER),'--job-json',str(job_path),'--lease-dir',str(LEASE),'--status-path',str(status),'--poll-seconds','10']
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    set_owner(active_delta=1,phase_job=jobid)
    try:
        with log.open('ab',buffering=0) as handle:res=subprocess.run(cmd,cwd=str(PROJECT),stdout=handle,stderr=subprocess.STDOUT,env=env,check=False)
        watch=json.loads(status.read_text()) if status.exists() else {}
        done=json.loads(marker.read_text()) if marker.exists() else {}
        passed=res.returncode==0 and done.get('status')=='PASS'
        row={'job_id':jobid,'status':'PASS' if passed else 'FAILED','returncode':res.returncode,'watcher_status':watch.get('status'),
             'selected_gpu':watch.get('selected_gpu'),'child_pid':watch.get('child_pid'),'watcher_pid':watch.get('watcher_pid'),
             'lease_released':watch.get('lease_released'),'completion_marker':str(marker) if marker.exists() else None,
             'watcher_status_path':str(status),'console_log':str(log),'actual_resource':done.get('peak_memory_reserved_mib',done.get('peak_memory_allocated_mib'))}
        return row
    finally:set_owner(active_delta=-1,phase_job=None)
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);args=ap.parse_args()
    qpath=args.queue.resolve();q=json.loads(qpath.read_text(encoding='utf-8'));jobs=q.get('jobs')
    if q.get('schema_version')!=1 or not isinstance(jobs,list) or not jobs:raise RuntimeError('invalid or empty V7 watcher queue')
    plan=json.loads((ROOT/'coordination/JOB_PLAN.json').read_text(encoding='utf-8'))
    if sha(WATCHER)!=plan['watcher_sha256']:raise RuntimeError('shared watcher changed after V7 source audit')
    m=re.search(r'^MAX_CONCURRENT_PROJECT_GPU_CHILDREN\s*=\s*(\d+)\s*$',WATCHER.read_text(encoding='utf-8'),re.M)
    if not m:raise RuntimeError('shared watcher concurrency cap is not readable')
    cap=int(m.group(1))
    ids=[j['job_id'] for j in jobs]
    if len(set(ids))!=len(ids):raise RuntimeError('queue contains duplicate job IDs')
    for j in jobs:
        missing={'job_id','protocol_id','output_root','command','resource_estimate'}-set(j)
        if missing:raise RuntimeError(f'{j.get("job_id")}: missing watcher fields {sorted(missing)}')
        if not isinstance(j['output_root'],str) or not Path(j['output_root']).is_absolute():raise RuntimeError(f'{j["job_id"]}: output_root is not absolute')
        for token in j['command']:
            if isinstance(token,str) and token.startswith('/') and not token.startswith(str(ROOT)) and not token.startswith(str(PROJECT)):
                raise RuntimeError(f'{j["job_id"]}: command path is outside project/V7 roots: {token}')
    started=utc();results=[];phase=str(q.get('phase',qpath.stem));set_owner(status='running',phase=phase,queue_path=str(qpath),queue_started_utc=started)
    state={'schema_version':1,'queue':qpath.name,'phase':phase,'status':'running','started_utc':started,'max_concurrency':cap,'results':[]}
    status_path=qpath.with_suffix('.status.json');atomic_json(status_path,state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cap) as pool:
        futures=[pool.submit(run_one,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();results.append(row);state['results']=sorted(results,key=lambda r:r['job_id']);state['updated_utc']=utc();atomic_json(status_path,state)
            print(json.dumps(row,sort_keys=True),flush=True)
    failed=[r for r in results if r['status']=='FAILED']
    stopped=any(r['status']=='stopped_before_launch' for r in results)
    queue_status='partial' if failed else ('stopped' if stopped else 'complete')
    state.update({'status':queue_status,'finished_utc':utc(),'failed_job_ids':[r['job_id'] for r in failed],
                  'stop_requested':stopped,'skipped_job_ids':[r['job_id'] for r in results if r['status']=='stopped_before_launch']});atomic_json(status_path,state)
    set_owner(status='running' if queue_status=='complete' else 'partial',phase=phase+'_complete' if queue_status=='complete' else phase+'_'+queue_status,queue_status_path=str(status_path),last_failed_job_ids=[r['job_id'] for r in failed])
    print(json.dumps({'queue':qpath.name,'status':queue_status,'jobs':len(results),'failed':len(failed),'stopped':stopped,'max_concurrency':cap},sort_keys=True),flush=True)
    return 0 if queue_status=='complete' else (3 if stopped and not failed else 2)
if __name__=='__main__':raise SystemExit(main())
