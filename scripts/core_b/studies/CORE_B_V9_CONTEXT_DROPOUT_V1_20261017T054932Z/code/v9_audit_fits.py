#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V8=PROJECT/'reports/benchmarks/CORE_B_V8_CONTROLLED_REPLAY_V1_20261016T045154Z'
PROGRAMS={'A0':('STATE-CTRL',0.0),'AD':('STATE-CTRL',0.5),'V0':('TS11-D',0.0),'VD':('TS11-D',0.5)}
SEEDS=(202610131,202610132,202610133)
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def wj(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 tmp=p.with_name('.'+p.name+'.part')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 queue=rj(ROOT/'coordination/queue_train.json');execution=rj(ROOT/'coordination/EXECUTION_TRAIN.json')
 if execution.get('status')!='complete' or len(execution.get('results',[]))!=12 or execution.get('failed_jobs'):raise RuntimeError('formal queue not accepted 12/12')
 rows=[];v0=[];by={}
 for seed in SEEDS:
  for program,(mode,p) in PROGRAMS.items():
   d=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'
   m=rj(d/'MODEL_MANIFEST.json');complete=rj(d/'RUN_COMPLETE.json')
   if m.get('status')!='PASS' or complete.get('status')!='PASS':raise RuntimeError('formal model/complete missing')
   if int(complete.get('epochs_completed',-1))!=160 or int(complete.get('optimizer_steps',-1))!=3200:raise RuntimeError('formal fit incomplete')
   if m.get('program_id')!=program or m.get('video_mode')!=mode or float(m.get('numeric_context_drop_probability',-1))!=p:raise RuntimeError('program recipe mismatch')
   if complete.get('target_role')!='TRUE_Y' or complete.get('teacher_target_used') is not False or complete.get('dev_or_final_target_rows')!=0:raise RuntimeError('fit target provenance invalid')
   if int(m.get('target_presentations',-1))!=204800 or int(complete.get('training_sample_presentations',-1))!=204800:raise RuntimeError('true-Y presentation count mismatch')
   if sha(m['model_file'])!=m['model_file_sha256']:raise RuntimeError('checkpoint hash mismatch')
   job=rj(ROOT/'configs/fits'/f'{program}_seed{seed}.json')
   if job['code_lock_sha256']!=sha(ROOT/'protocol/CODE_LOCK.json') or job['training_runner_sha256']!=sha(ROOT/'code/v9_train.py'):raise RuntimeError('fit code lock mismatch')
   watcher=rj(ROOT/'coordination/watcher_status'/f'v9_{program}_s{seed}.json')
   if watcher.get('status')!='complete' or int(watcher.get('exit_code',-1))!=0 or watcher.get('lease_released') is not True:raise RuntimeError('watcher exit/lease receipt mismatch')
   me=readcsv(d/'MASK_EPOCH_LOG.csv')
   if len(me)!=800:raise RuntimeError('per-task mask schedule log incomplete')
   dropped=sum(int(x['dropped_n']) for x in me);presented=sum(int(x['receiver_n']) for x in me)
   observed=dropped/presented
   if p==0 and dropped!=0:raise RuntimeError('p0 arm applied numeric mask')
   if p==.5 and not (0.45<observed<0.55):raise RuntimeError('observed p=.5 mask rate outside broad receipt sanity band')
   epoch=readcsv(d/'EPOCH_LOG.csv')
   if len(epoch)!=160:raise RuntimeError('training curve incomplete')
   start_rng=m['rng_at_seed'];step_rng=m['rng_before_first_optimizer_step']
   by[(program,seed)]=m
   rows.append({'program_id':program,'seed':seed,'video_mode':mode,'p':p,'epochs':160,'optimizer_steps':3200,
      'true_Y_presentations':204800,'training_target_role':'TRUE_Y','teacher_target_used':False,
      'observed_mask_rate_all_epochs':observed,'epoch160_mask_rate':float(epoch[-1]['numeric_context_drop_rate_observed']),
      'masked_train_objective_epoch160':float(epoch[-1]['masked_input_true_y_objective']),
      'model_file_sha256':m['model_file_sha256'],'model_state_hash':m['final_state_hash'],
      'init_state_hash':m['init_state_hash'],'rng_at_seed_sha':hashlib.sha256(json.dumps(start_rng,sort_keys=True).encode()).hexdigest(),
      'rng_before_first_step_sha':hashlib.sha256(json.dumps(step_rng,sort_keys=True).encode()).hexdigest(),
      'sampler_sequence_sha256':m['sampler_sequence_sha256'],'mask_bank_bits_sha256':m['mask_bank_bits_sha256'],
      'physical_gpu':m['physical_gpu'],'cuda_visible_devices':m['cuda_visible_devices'],'logical_device':m['logical_device'],
      'elapsed_seconds':complete['elapsed_seconds'],'peak_allocated_mib':m['peak_memory_allocated_mib'],'peak_reserved_mib':m['peak_memory_reserved_mib'],
      'watcher_status_path':str(ROOT/'coordination/watcher_status'/f'v9_{program}_s{seed}.json'),
      'watcher_log_path':str(ROOT/'logs'/f'fit_{program}_seed{seed}.watcher_child.log')})
 for seed in SEEDS:
  group=[by[(p,seed)] for p in PROGRAMS]
  for key in ('init_state_hash','sampler_sequence_sha256','rng_at_seed','rng_before_first_optimizer_step'):
   if len({json.dumps(x[key],sort_keys=True) for x in group})!=1:raise RuntimeError(f'same-seed {key} differs across conditions')
  if by[('AD',seed)]['mask_bank_bits_sha256']!=by[('VD',seed)]['mask_bank_bits_sha256']:raise RuntimeError('AD/VD bank mismatch')
  new=by[('V0',seed)];old=rj(V8/'model_only/GT_RETRAIN'/f'seed{seed}'/'attempt_01/MODEL_MANIFEST.json')
  v0.append({'seed':seed,'v0_model_sha256':new['model_file_sha256'],'v8_hard_video_model_sha256':old['model_file_sha256'],
   'model_file_exact_match':new['model_file_sha256']==old['model_file_sha256'],
   'final_state_hash_exact_match':new['final_state_hash']==old['final_state_hash'],
   'interpretation':'V8 accepted GT_RETRAIN is the hard-video V baseline reference'})
 writecsv(ROOT/'results/RESOURCE_LEDGER.csv',rows)
 writecsv(ROOT/'qa/V0_MODEL_REPLAY.csv',v0)
 summary={'schema_version':1,'status':'PASS','created_utc':utc(),'formal_fits':12,'successful':12,'failed':0,'oom':0,
  'epochs_per_fit':160,'updates_per_fit':3200,'true_Y_presentations_per_fit':204800,
  'total_true_Y_presentations':sum(r['true_Y_presentations'] for r in rows),'teacher_targets_used':False,
  'same_init_rng_and_sampler_by_seed':True,'AD_VD_mask_bank_match':True,
  'final_heldout_payload_access':0,'dev_target_payload_access':0,
  'programs':{p:{'mean_observed_mask_rate':float(np.mean([r['observed_mask_rate_all_epochs'] for r in rows if r['program_id']==p])),
    'mean_peak_reserved_mib':float(np.mean([r['peak_reserved_mib'] for r in rows if r['program_id']==p]))} for p in PROGRAMS},
  'V0_exact_weight_replay_all_seeds':all(x['model_file_exact_match'] and x['final_state_hash_exact_match'] for x in v0),
  'V0_replay':v0,'queue_execution_path':str(ROOT/'coordination/EXECUTION_TRAIN.json')}
 wj(ROOT/'qa/FORMAL_FIT_ACCEPTANCE.json',summary)
 print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()

