#!/usr/bin/env python3
"""Run a CPU-only V3 prediction queue without accessing any CUDA device."""
from __future__ import annotations
import os
import concurrent.futures,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def run_one(raw):
 p=Path(raw);c=json.loads(p.read_text());out=Path(c['output_root']);marker=out/c.get('completion_marker','PREDICTION_COMPLETE.json')
 if marker.is_file() and json.loads(marker.read_text()).get('status')=='PASS':return {'job_id':c['job_id'],'status':'already_complete','returncode':0}
 out.mkdir(parents=True,exist_ok=True)
 status=ROOT/'coordination/cpu_status'/f"{c['job_id']}.json";log=ROOT/'logs/cpu_predictions'/f"{c['job_id']}.log";status.parent.mkdir(parents=True,exist_ok=True);log.parent.mkdir(parents=True,exist_ok=True)
 cmd=[PYTHON,str(ROOT/'code/predict_readout_cpu.py'),'--job-json',str(p)]
 with log.open('ab',buffering=0) as f:r=subprocess.run(cmd,cwd=str(PROJECT),stdout=f,stderr=subprocess.STDOUT,check=False,
  env={**os.environ,**{str(k):str(v) for k,v in c.get('environment',{}).items()},'CUDA_VISIBLE_DEVICES':''})
 complete=json.loads(marker.read_text()) if marker.is_file() else {}
 row={'job_id':c['job_id'],'status':'PASS' if r.returncode==0 and complete.get('status')=='PASS' else 'FAILED',
  'returncode':r.returncode,'execution_device':'CPU','gpu_started':False,'completion_marker':str(marker) if marker.exists() else None,'log':str(log)}
 status.write_text(json.dumps({'schema_version':1,'completed_utc':utc(),**row},indent=2,sort_keys=True)+'\n')
 return row
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);args=p.parse_args();q=json.loads(args.queue.read_text())
 if q.get('schema_version')!=1 or not isinstance(q.get('jobs'),list):raise RuntimeError('invalid CPU queue')
 started=utc();results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for x in concurrent.futures.as_completed([pool.submit(run_one,r) for r in q['jobs']]):
   row=x.result();results.append(row);print(json.dumps({'utc':utc(),**row},sort_keys=True),flush=True)
 results.sort(key=lambda x:x['job_id']);accepted=sum(x['status'] in {'PASS','already_complete'} for x in results)
 phase=q.get('phase','cpu');summary={'schema_version':1,'phase':phase,'queue_path':str(args.queue.resolve()),'started_utc':started,
  'completed_utc':utc(),'scheduled_jobs':len(results),'accepted_jobs':accepted,'failed_jobs':len(results)-accepted,
  'gpu_jobs_started':0,'results':results,'final_heldout_payload_access':0}
 dst=ROOT/'coordination'/f"PHASE_STATUS_{str(phase).upper()}.json";dst.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS' if accepted==len(results) else 'PARTIAL','phase':phase,'accepted_jobs':accepted,'scheduled_jobs':len(results),'gpu_jobs_started':0}))
 if accepted!=len(results):raise SystemExit(2)
if __name__=='__main__':main()
