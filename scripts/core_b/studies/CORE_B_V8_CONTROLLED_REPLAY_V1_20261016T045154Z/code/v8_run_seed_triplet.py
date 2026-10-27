#!/usr/bin/env python3
"""Run one seed's fixed-order three-arm training triplet as one watcher child."""
from __future__ import annotations
import argparse,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import v8_train

def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def writej(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--triplet-json',required=True,type=Path);args=ap.parse_args()
 cfg=json.loads(args.triplet_json.read_text());out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
 status={'schema_version':1,'status':'running','created_utc':utc(),'seed':cfg['seed'],'program_order':cfg['program_order'],
  'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':'cuda:0','fits':[]}
 writej(out/'SEED_TRIPLET_STATUS.json',status);start=time.perf_counter()
 for entry in cfg['fits']:
  if Path(cfg['stop_sentinel']).exists():
   status.update({'status':'stop_requested_after_fit_boundary','updated_utc':utc(),'elapsed_seconds':time.perf_counter()-start});writej(out/'SEED_TRIPLET_STATUS.json',status);return 75
  job_path=Path(entry['config_path']);job=json.loads(job_path.read_text())
  if int(job['seed'])!=int(cfg['seed']) or job['program_id']!=entry['program_id']:raise RuntimeError('triplet fit order/config seed mismatch')
  if job.get('branch_lock_sha256')!=cfg['branch_lock_sha256'] or job.get('recipe_lock_sha256')!=cfg['recipe_lock_sha256']:raise RuntimeError('fit config lock hash mismatch')
  began=utc();rc=v8_train.run(job_path)
  slot=Path(job['attempt_dir']);complete=slot/'RUN_COMPLETE.json';manifest=slot/'MODEL_MANIFEST.json'
  record={'job_id':job['job_id'],'program_id':job['program_id'],'seed':job['seed'],'started_utc':began,'exit_code':rc,
    'run_complete_path':str(complete),'model_manifest_path':str(manifest),'completed_utc':utc() if rc==0 and complete.is_file() else None}
  if rc!=0 or not complete.is_file() or not manifest.is_file():
   record['status']='failed';status['fits'].append(record);status.update({'status':'failed','updated_utc':utc(),'elapsed_seconds':time.perf_counter()-start});writej(out/'SEED_TRIPLET_STATUS.json',status);return rc or 1
  done=json.loads(complete.read_text());mm=json.loads(manifest.read_text())
  if done.get('status')!='PASS' or int(done.get('optimizer_steps',-1))!=3200 or int(done.get('epochs_completed',-1))!=160 or mm.get('status')!='PASS':
   record['status']='rejected';status['fits'].append(record);status.update({'status':'failed_acceptance','updated_utc':utc()});writej(out/'SEED_TRIPLET_STATUS.json',status);return 1
  record.update({'status':'PASS','epochs':160,'optimizer_steps':3200,'model_sha256':mm['model_file_sha256'],'training_runner_sha256':mm['training_runner_sha256'],'sampler_sequence_sha256':mm['sampler_sequence_sha256'],'physical_gpu':mm['physical_gpu']})
  status['fits'].append(record);status.update({'updated_utc':utc(),'completed_fits':len(status['fits'])});writej(out/'SEED_TRIPLET_STATUS.json',status)
 receipt={'schema_version':1,'status':'PASS','completed_utc':utc(),'seed':cfg['seed'],'program_order':cfg['program_order'],'fits':status['fits'],
  'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':'cuda:0',
  'elapsed_seconds':time.perf_counter()-start,'branch_lock_sha256':cfg['branch_lock_sha256'],'recipe_lock_sha256':cfg['recipe_lock_sha256'],'final_heldout_payload_access':0}
 writej(out/'SEED_TRIPLET_COMPLETE.json',receipt);status.update({'status':'complete','completed_utc':receipt['completed_utc'],'elapsed_seconds':receipt['elapsed_seconds']});writej(out/'SEED_TRIPLET_STATUS.json',status)
 print(json.dumps({'status':'PASS','seed':cfg['seed'],'fits':len(receipt['fits']),'physical_gpu':receipt['physical_gpu'],'elapsed_seconds':receipt['elapsed_seconds']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
