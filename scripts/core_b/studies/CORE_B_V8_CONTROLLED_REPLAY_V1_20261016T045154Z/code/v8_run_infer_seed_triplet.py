#!/usr/bin/env python3
"""Run six teacher-free train/dev inference jobs for one seed under one watcher lease."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import v8_infer

def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def valid_existing(entry):
 out=Path(entry['output_root']);complete=out/'PREDICTION_COMPLETE.json'
 if not complete.exists():return False
 rec=json.loads(complete.read_text());job=json.loads(Path(entry['config_path']).read_text())
 if rec.get('status')!='PASS' or rec.get('program_id')!=job['program_id'] or rec.get('dataset_role')!=job['dataset_role'] or int(rec.get('seed',-1))!=int(job['seed']):return False
 return sha(Path(rec['prediction_path']))==rec['prediction_sha256'] and sha(Path(rec['prediction_rows_path']))==rec['prediction_rows_sha256'] and rec.get('model_sha256')==job['model_sha256']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--triplet-json',required=True,type=Path);args=ap.parse_args();cfg=json.loads(args.triplet_json.read_text())
 out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 status={'schema_version':1,'status':'running','created_utc':utc(),'seed':cfg['seed'],'program_order':cfg['program_order'],'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':'cuda:0','bundles':[]}
 writej(out/'SEED_INFERENCE_STATUS.json',status)
 for entry in cfg['jobs']:
  if Path(cfg['stop_sentinel']).exists():status.update({'status':'stop_requested','updated_utc':utc(),'elapsed_seconds':time.perf_counter()-start});writej(out/'SEED_INFERENCE_STATUS.json',status);return 75
  if valid_existing(entry):status['bundles'].append({'job_id':entry['job_id'],'status':'already_accepted','output_root':entry['output_root']});writej(out/'SEED_INFERENCE_STATUS.json',status);continue
  rc=v8_infer.run(Path(entry['config_path']))
  receipt=Path(entry['output_root'])/'PREDICTION_COMPLETE.json'
  record={'job_id':entry['job_id'],'dataset_role':entry['dataset_role'],'program_id':entry['program_id'],'exit_code':rc,'completed_utc':utc()}
  if rc!=0 or not receipt.is_file():
   record['status']='failed';status['bundles'].append(record);status.update({'status':'failed','updated_utc':utc()});writej(out/'SEED_INFERENCE_STATUS.json',status);return rc or 1
  rec=json.loads(receipt.read_text());record.update({'status':'PASS','prediction_sha256':rec['prediction_sha256'],'model_sha256':rec['model_sha256']});status['bundles'].append(record);status.update({'updated_utc':utc(),'completed_bundles':len(status['bundles'])});writej(out/'SEED_INFERENCE_STATUS.json',status)
 if len(status['bundles'])!=6:raise RuntimeError('inference seed triplet did not seal six bundles')
 done={'schema_version':1,'status':'PASS','completed_utc':utc(),'seed':cfg['seed'],'bundles':status['bundles'],'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),
  'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':'cuda:0','elapsed_seconds':time.perf_counter()-start,
  'branch_lock_sha256':cfg['branch_lock_sha256'],'recipe_lock_sha256':cfg['recipe_lock_sha256'],'dev_target_archive_opened':False,'final_heldout_payload_access':0}
 writej(out/'SEED_INFERENCE_COMPLETE.json',done);status.update({'status':'complete','completed_utc':done['completed_utc'],'elapsed_seconds':done['elapsed_seconds']});writej(out/'SEED_INFERENCE_STATUS.json',status)
 print(json.dumps({'status':'PASS','seed':cfg['seed'],'bundles':len(done['bundles']),'physical_gpu':done['physical_gpu']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
