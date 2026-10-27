#!/usr/bin/env python3
"""Run a bounded V3 queue exclusively through the shared GPU lease watcher."""
from __future__ import annotations
import os
import concurrent.futures,hashlib,json,re,subprocess,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
V3PATH=json.loads((ROOT/'inventory/V3_PATHS.json').read_text())
WATCHER=Path(V3PATH['shared_watcher'])
LEASE_DIR=ROOT/'coordination/locks/gpu_lease'
STATUS_DIR=ROOT/'coordination/watcher_status'
LOG_DIR=ROOT/'logs/watcher_children'
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run_one(config_path:Path):
 c=json.loads(config_path.read_text());out=Path(c['output_root'])
 marker=out/str(c.get('completion_marker','RUN_COMPLETE.json'))
 if marker.exists() and json.loads(marker.read_text()).get('status')=='PASS':
  return {'job_id':c['job_id'],'status':'already_complete','returncode':0,'marker':str(marker)}
 STATUS_DIR.mkdir(parents=True,exist_ok=True);LOG_DIR.mkdir(parents=True,exist_ok=True)
 status=STATUS_DIR/f"{c['job_id']}.json";console=LOG_DIR/f"{c['job_id']}.log"
 cmd=[PYTHON,str(WATCHER),'--job-json',str(config_path),'--lease-dir',str(LEASE_DIR),'--status-path',str(status),'--poll-seconds','10']
 env={**os.environ,**{str(k):str(v) for k,v in c.get('environment',{}).items()}}
 with console.open('ab',buffering=0) as h:res=subprocess.run(cmd,cwd=str(PROJECT),stdout=h,stderr=subprocess.STDOUT,check=False,env=env)
 complete=json.loads(marker.read_text()) if marker.is_file() else {}
 watcher=json.loads(status.read_text()) if status.is_file() else {}
 passed=res.returncode==0 and complete.get('status')=='PASS'
 return {'job_id':c['job_id'],'status':'PASS' if passed else 'FAILED','returncode':res.returncode,
   'completion_marker':str(marker) if marker.exists() else None,'watcher_status':watcher.get('status'),
   'watcher_status_path':str(status),'console_log':str(console),
   'selected_gpu':watcher.get('selected_gpu'),'child_pid':watcher.get('child_pid'),
   'lease_released':watcher.get('lease_released')}
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);args=ap.parse_args()
 qpath=args.queue.resolve();q=json.loads(qpath.read_text())
 if q.get('schema_version')!=1 or not isinstance(q.get('jobs'),list):raise RuntimeError('invalid V3 queue schema')
 source=json.loads((ROOT/'inventory/V2_SOURCE_ACCEPTANCE.json').read_text())
 actual=sha(WATCHER)
 if actual!=source.get('shared_watcher_sha256'):raise RuntimeError('shared watcher differs from accepted V2 policy hash')
 source_text=WATCHER.read_text();m=re.search(r'^MAX_CONCURRENT_PROJECT_GPU_CHILDREN\s*=\s*(\d+)\s*$',source_text,re.M)
 if not m or int(m.group(1))<=0:raise RuntimeError('shared watcher concurrent child cap not found')
 cap=int(m.group(1));configs=[Path(x).resolve() for x in q['jobs']]
 if len(configs)!=len(set(configs)):raise RuntimeError('V3 queue repeats watcher configs')
 start=utc();results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=cap) as pool:
  futures=[pool.submit(run_one,p) for p in configs]
  for f in concurrent.futures.as_completed(futures):
   row=f.result();results.append(row);print(json.dumps({'utc':utc(),**row},sort_keys=True),flush=True)
 results.sort(key=lambda x:x['job_id']);accepted=sum(x['status'] in {'PASS','already_complete'} for x in results);failed=len(results)-accepted
 summary={'schema_version':1,'phase':q.get('phase'),'queue_path':str(qpath),'started_utc':start,'completed_utc':utc(),
  'scheduled_jobs':len(configs),'accepted_jobs':accepted,'failed_jobs':failed,'shared_watcher_sha256':actual,
  'shared_watcher_max_children':cap,'results':results,'final_heldout_payload_access':0}
 sp=ROOT/'coordination'/f"PHASE_STATUS_{str(q.get('phase','unknown')).upper()}.json"
 tmp=sp.with_name(f'.{sp.name}.part');tmp.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');tmp.replace(sp)
 print(json.dumps({'status':'PASS' if failed==0 else 'PARTIAL','phase':q.get('phase'),'scheduled_jobs':len(configs),'accepted_jobs':accepted,'failed_jobs':failed,'shared_watcher_max_children':cap}))
 if failed:raise SystemExit(2)
if __name__=='__main__':main()
