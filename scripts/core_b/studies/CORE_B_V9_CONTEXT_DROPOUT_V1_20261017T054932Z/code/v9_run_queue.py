#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse,concurrent.futures,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=PROJECT/'envs/jyc/bin/python'
WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
LEASE_DIR=ROOT/'coordination/leases'
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def wj(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part')
    tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def one(job_path:Path,stage0=False):
    job=rj(job_path)
    job_id=str(job['job_id'])
    status=ROOT/'coordination/watcher_status'/f'{job_id}.json'
    log=ROOT/'logs/watcher_control'/f'{job_id}.stdout.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    if status.exists():
        prev=rj(status)
        if prev.get('status')=='complete' and prev.get('exit_code')==0:
            return {'job_id':job_id,'status':'SKIPPED_ALREADY_COMPLETE','watcher_status_path':str(status),'exit_code':0}
    argv=[str(PYTHON),str(WATCHER),'--job-json',str(job_path),'--lease-dir',str(LEASE_DIR),
        '--status-path',str(status),'--poll-seconds','30','--max-wait-seconds','0','--max-oom-requeues',str(job.get('max_oom_requeues',1))]
    started=utc();start=time.monotonic()
    with log.open('a',encoding='utf-8') as fh:
        fh.write(json.dumps({'event':'watcher_start','utc':started,'pid_pending':True,'command':argv,'cwd':ROOT.as_posix()},sort_keys=True)+'\n');fh.flush()
        proc=subprocess.Popen(argv,cwd=str(ROOT),stdout=fh,stderr=subprocess.STDOUT,text=True)
        active={'event':'watcher_pid','utc':utc(),'job_id':job_id,'watcher_pid':proc.pid,'job_path':str(job_path),'status_path':str(status),'log_path':str(log)}
        with (ROOT/'coordination/ACTIVE_WATCHERS.jsonl').open('a',encoding='utf-8') as active_f:
            active_f.write(json.dumps(active,sort_keys=True)+'\n');active_f.flush();os.fsync(active_f.fileno())
        code=proc.wait()
        fh.write(json.dumps({'event':'watcher_exit','utc':utc(),'pid':proc.pid,'exit_code':code,'elapsed_seconds':time.monotonic()-start},sort_keys=True)+'\n');fh.flush()
    result={'job_id':job_id,'watcher_pid':proc.pid,'exit_code':code,'elapsed_seconds':time.monotonic()-start,
        'watcher_status_path':str(status),'watcher_stdout_path':str(log),'completed_utc':utc()}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue-json',type=Path,required=True);args=ap.parse_args()
    queue=rj(args.queue_json)
    entries=queue.get('jobs',[])
    if not entries: raise SystemExit('queue contains no jobs')
    is_stage0='queue_stage0' in args.queue_json.name
    paths=[]
    for item in entries:
        if isinstance(item,str): paths.append(Path(item))
        elif isinstance(item,dict) and item.get('job_json'): paths.append(Path(item['job_json']))
        else: raise RuntimeError('queue job entry has invalid schema')
    max_workers=int(queue.get('max_concurrent_watchers',1 if is_stage0 else 2))
    if max_workers not in (1,2): raise RuntimeError('watcher concurrency must respect shared cap of 2')
    started=utc()
    if is_stage0:
        suffix=args.queue_json.stem.removeprefix('queue_stage0').upper()
        result_path=ROOT/'coordination'/('EXECUTION_STAGE0'+suffix+'.json')
    else:
        result_path=ROOT/'coordination'/('EXECUTION_INFERENCE.json' if 'inference' in args.queue_json.name else 'EXECUTION_TRAIN.json')
    state={'schema_version':1,'status':'running','started_utc':started,'queue_path':str(args.queue_json),'queue_sha256':__import__('hashlib').sha256(args.queue_json.read_bytes()).hexdigest(),
        'tmux_required':True,'max_concurrent_watchers':max_workers,'lease_dir':str(LEASE_DIR),'job_count':len(paths),'results':[]}
    wj(result_path,state)
    results=[];pending=[];stop_path=ROOT/'coordination/STOP_REQUESTED'
    for offset in range(0,len(paths),max_workers):
        if not is_stage0 and stop_path.exists():
            pending=[str(p) for p in paths[offset:]]
            state.update({'status':'paused_by_stop_request','completed_utc':utc(),'pending_job_paths':pending,
                'stop_request_path':str(stop_path),'safe_stop':'current watcher pair completed before this queue boundary; no signals sent'})
            wj(result_path,state)
            print(json.dumps({'status':state['status'],'completed_jobs':len(results),'pending_jobs':len(pending),'execution_path':str(result_path)},sort_keys=True))
            return 75
        wave=paths[offset:offset+max_workers]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map={pool.submit(one,path,is_stage0):path for path in wave}
            for future in concurrent.futures.as_completed(future_map):
                try: item=future.result()
                except Exception as exc: item={'job_id':future_map[future].stem,'status':'launcher_error','error':f'{type(exc).__name__}: {exc}'}
                results.append(item);state['results']=results.copy();wj(result_path,state)
    failed=[r for r in results if int(r.get('exit_code',1))!=0 or r.get('status')=='launcher_error']
    state.update({'status':'failed' if failed else 'complete','completed_utc':utc(),'failed_jobs':failed,
        'successful_watcher_count':sum(int(r.get('exit_code',1))==0 for r in results),
        'pending_job_paths':[],'safe_stop':'touch coordination/STOP_REQUESTED; active watcher wave completes, later jobs are not launched'})
    wj(result_path,state)
    print(json.dumps({'status':state['status'],'job_count':len(paths),'failed_jobs':len(failed),'execution_path':str(result_path)},sort_keys=True))
    return 1 if failed else 0
if __name__=='__main__':raise SystemExit(main())

