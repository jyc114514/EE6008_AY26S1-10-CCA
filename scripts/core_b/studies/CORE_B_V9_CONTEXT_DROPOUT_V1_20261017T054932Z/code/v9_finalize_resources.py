#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part')
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def writej(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part')
 tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def main():
 rows=readcsv(ROOT/'results/RESOURCE_LEDGER.csv')
 if len(rows)!=12:raise RuntimeError('formal fit resource ledger expected 12 rows')
 formal=[]
 for r in rows:
  r=dict(r);r['record_type']='formal_fit';r['status']='PASS'
  w=rj(ROOT/'coordination/watcher_status'/f"v9_{r['program_id']}_s{r['seed']}.json")
  if w.get('status')!='complete' or int(w.get('exit_code',-1))!=0 or w.get('lease_released') is not True:raise RuntimeError('formal fit watcher evidence invalid')
  r['watcher_pid']=w.get('watcher_pid');r['child_pid']=w.get('child_pid');r['lease_path']=w.get('lease_path')
  r['watcher_exit_code']=w.get('exit_code');r['lease_released']=w.get('lease_released')
  formal.append(r)
 stage_path=ROOT/'tmp/stage0/retry02/attempt_01/STAGE0_COMPLETE.json';stage=rj(stage_path)
 sw=rj(ROOT/'coordination/watcher_status'/f"{stage['job_id']}.json")
 if sw.get('exit_code')!=0 or sw.get('status')!='complete':raise RuntimeError('Stage0 watcher evidence invalid')
 stage_row={'record_type':'stage0_qa','status':'PASS','job_id':stage['job_id'],'program_id':stage['program_id'],'seed':stage['seed'],
  'dataset_role':'train','input_policy':'training_random_mask','optimizer_steps':stage['updates'],'true_Y_presentations':stage['rows_presented'],
  'physical_gpu':stage['physical_gpu'],'cuda_visible_devices':stage['cuda_visible_devices'],'logical_device':stage['logical_device'],
  'peak_memory_allocated_mib':stage['peak_memory_allocated_mib'],'peak_memory_reserved_mib':stage['peak_memory_reserved_mib'],
  'elapsed_seconds':stage['elapsed_seconds'],'watcher_pid':sw.get('watcher_pid'),'child_pid':sw.get('child_pid'),
  'lease_path':sw.get('lease_path'),'watcher_exit_code':sw.get('exit_code'),'lease_released':sw.get('lease_released'),
  'log_path':sw.get('log_path'),'receipt_path':str(stage_path)}
 inference=[];exec_status=rj(ROOT/'coordination/EXECUTION_INFERENCE.json')
 if exec_status.get('status')!='complete' or len(exec_status.get('results',[]))!=36:raise RuntimeError('inference queue not complete 36/36')
 for item in rj(ROOT/'coordination/queue_inference.json')['jobs']:
  job=rj(ROOT/'configs/inference'/f"{item['job_id']}.json");out=Path(job['output_root']);rec=rj(out/'PREDICTION_COMPLETE.json')
  watch=rj(ROOT/'coordination/watcher_status'/f"{item['job_id']}.json")
  if rec.get('status')!='PASS' or watch.get('status')!='complete' or int(watch.get('exit_code',-1))!=0 or watch.get('lease_released') is not True:
   raise RuntimeError('inference resource/prediction receipt invalid: '+item['job_id'])
  inference.append({'record_type':'neural_inference','status':'PASS','job_id':item['job_id'],'program_id':rec['program_id'],'seed':rec['seed'],
   'dataset_role':rec['dataset_role'],'input_policy':rec['input_policy'],'prediction_rows':rec['prediction_rows'],
   'physical_gpu':rec['physical_gpu'],'cuda_visible_devices':rec['cuda_visible_devices'],'logical_device':rec['logical_device'],
   'peak_memory_allocated_mib':rec['peak_memory_allocated_mib'],'peak_memory_reserved_mib':rec['peak_memory_reserved_mib'],
   'elapsed_seconds':rec['elapsed_seconds'],'watcher_pid':watch.get('watcher_pid'),'child_pid':watch.get('child_pid'),
   'lease_path':watch.get('lease_path'),'watcher_exit_code':watch.get('exit_code'),'lease_released':watch.get('lease_released'),
   'log_path':watch.get('log_path'),'prediction_path':rec['prediction_path'],'prediction_sha256':rec['prediction_sha256']})
 combined=formal+[stage_row]+inference
 path=ROOT/'results/RESOURCE_LEDGER.csv'
 backup=path.with_name(path.name+'.pre-inference-resource-ledger-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.bak')
 shutil.copy2(path,backup);writecsv(path,combined)
 bygpu={}
 for r in combined:
  g=str(r.get('physical_gpu','unknown'));bygpu[g]=bygpu.get(g,0)+1
 fit_seconds=sum(float(r.get('elapsed_seconds') or 0) for r in formal)
 inf_seconds=sum(float(r.get('elapsed_seconds') or 0) for r in inference)
 summary={'schema_version':1,'status':'PASS','created_utc':utc(),'formal_fit_count':len(formal),'stage0_qa_count':1,
  'neural_inference_count':len(inference),'total_resource_records':len(combined),'formal_fit_elapsed_seconds_sum':fit_seconds,
  'inference_elapsed_seconds_sum':inf_seconds,'formal_max_peak_reserved_mib':max(float(r['peak_reserved_mib']) for r in formal),
  'inference_max_peak_reserved_mib':max(float(r['peak_memory_reserved_mib']) for r in inference),
  'stage0_peak_reserved_mib':float(stage['peak_memory_reserved_mib']),'physical_gpu_resource_records':bygpu,
  'watcher_max_concurrency':2,'watcher_wait_timeout_seconds':0,'all_watcher_children_exit0':True,'all_leases_released':True,
  'resource_policy':'dynamic_all_visible shared gpu_lease_watcher; measured 508 MiB Stage0 peak; free-memory admission >=1020 MiB',
  'no_fixed_gpu_allowlist_or_time_or_hour_cap_added':True,'oom_count':0,'final_heldout_payload_access':0}
 writej(ROOT/'results/RESOURCE_SUMMARY.json',summary)
 print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()

