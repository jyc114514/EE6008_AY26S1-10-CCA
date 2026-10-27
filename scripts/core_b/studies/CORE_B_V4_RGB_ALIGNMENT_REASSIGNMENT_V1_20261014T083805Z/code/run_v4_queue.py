#!/usr/bin/env python3
"""Launch a bounded V4 phase only through the shared project GPU lease watcher."""
from __future__ import annotations
import os
import argparse,concurrent.futures,hashlib,json,os,re,subprocess,threading
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
LEASE_DIR=ROOT/'coordination/locks/gpu_lease'
STATUS_DIR=ROOT/'coordination/watcher_status'
LOG_DIR=ROOT/'logs/console'
STATE=ROOT/'coordination/OWNER_STATE.json'
LOCK=threading.Lock();ACTIVE=0
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_update(active_delta=0,**fields):
    global ACTIVE
    with LOCK:
        ACTIVE=max(0,ACTIVE+active_delta)
        d=json.loads(STATE.read_text())
        d.update(fields);d['active_jobs']=ACTIVE;d['updated_utc']=utc();d['owner_pid']=os.getpid()
        tmp=STATE.with_name('.OWNER_STATE.part');tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');tmp.replace(STATE)
def run_one(config_path:Path)->dict:
    c=json.loads(config_path.read_text());jobid=c['job_id'];out=Path(c['output_root']);marker=out/str(c.get('completion_marker','RUN_COMPLETE.json'))
    if marker.is_file():
        try:
            d=json.loads(marker.read_text())
            if d.get('status')=='PASS':return {'job_id':jobid,'status':'already_complete','returncode':0,'completion_marker':str(marker)}
        except Exception:pass
    STATUS_DIR.mkdir(parents=True,exist_ok=True);LOG_DIR.mkdir(parents=True,exist_ok=True)
    status=STATUS_DIR/f'{jobid}.json';log=LOG_DIR/f'{jobid}.log'
    cmd=[PYTHON,str(WATCHER),'--job-json',str(config_path),'--lease-dir',str(LEASE_DIR),'--status-path',str(status),'--poll-seconds','10']
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    state_update(active_delta=1,phase_job=jobid)
    try:
        with log.open('ab',buffering=0) as h:res=subprocess.run(cmd,cwd=str(PROJECT),stdout=h,stderr=subprocess.STDOUT,check=False,env=env)
        wr=json.loads(status.read_text()) if status.exists() else {}
        done=json.loads(marker.read_text()) if marker.exists() else {}
        passed=res.returncode==0 and done.get('status')=='PASS'
        return {'job_id':jobid,'status':'PASS' if passed else 'FAILED','returncode':res.returncode,'watcher_status':wr.get('status'),'selected_gpu':wr.get('selected_gpu'),'child_pid':wr.get('child_pid'),'lease_released':wr.get('lease_released'),'marker':str(marker) if marker.exists() else None,'watcher_status_path':str(status),'console_log':str(log)}
    finally:state_update(active_delta=-1,phase_job=None)
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);args=ap.parse_args()
    qpath=args.queue.resolve();q=json.loads(qpath.read_text())
    if q.get('schema_version')!=1 or not isinstance(q.get('jobs'),list) or not q['jobs']:raise RuntimeError('invalid or empty V4 queue')
    source=json.loads((ROOT/'inventory/WATCHER_POLICY.json').read_text())
    actual=sha(WATCHER)
    if actual!=source['source_sha256']:raise RuntimeError('shared watcher changed after V4 pre-registration')
    m=re.search(r'^MAX_CONCURRENT_PROJECT_GPU_CHILDREN\s*=\s*(\d+)\s*$',WATCHER.read_text(),re.M)
    if not m:raise RuntimeError('shared watcher has no parseable project child limit')
    cap=int(m.group(1));configs=[Path(x).resolve() for x in q['jobs']]
    if len(set(configs))!=len(configs):raise RuntimeError('queue repeats watcher configuration')
    if any(ROOT not in p.parents for p in configs):raise RuntimeError('watcher config escapes V4 root')
    state_update(phase=q.get('phase'),status='running',tmux_session='ee6008-v4-20261014t083805z')
    started=utc();rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=cap) as pool:
        fs=[pool.submit(run_one,p) for p in configs]
        for f in concurrent.futures.as_completed(fs):
            row=f.result();rows.append(row);print(json.dumps({'utc':utc(),**row},sort_keys=True),flush=True)
    rows.sort(key=lambda x:x['job_id']);accepted=sum(r['status'] in {'PASS','already_complete'} for r in rows);failed=len(rows)-accepted
    payload={'schema_version':1,'phase':q.get('phase'),'queue_path':str(qpath),'started_utc':started,'completed_utc':utc(),'scheduled_jobs':len(configs),'accepted_jobs':accepted,'failed_jobs':failed,'shared_watcher_sha256':actual,'shared_watcher_max_children':cap,'resource_estimate_basis':'prior V2/V3 measured runner peaks and V4 prelaunch pilot estimate','results':rows,'final_heldout_payload_access':0}
    p=ROOT/'coordination'/f"PHASE_STATUS_{str(q.get('phase','unknown')).upper()}.json";tmp=p.with_name(f'.{p.name}.part');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(p)
    sp=qpath.with_suffix('.status.json');sp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    state_update(phase=q.get('phase'),active_jobs=0)
    print(json.dumps({'phase':q.get('phase'),'scheduled':len(configs),'accepted':accepted,'failed':failed,'watcher_cap':cap}),flush=True)
    return 0 if failed==0 else 2
if __name__=='__main__':raise SystemExit(main())
