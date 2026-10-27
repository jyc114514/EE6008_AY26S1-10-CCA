#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-013; source SHA-256: f0bb1c40e693fedee13687687e4651815d22d311721c953b3c1e1440b72a3255
"""CPU-only frozen-source, matrix, schema, INIT, and prediction preflight."""
from __future__ import annotations
import csv,hashlib,json
import os
from pathlib import Path
import numpy as np
import torch
from schema_adapter import canonical_run,canonical_prediction,read_json
from d3_model import D3TimeSpacePredictor,parameter_count,state_hash
from fixed_future_query_model import FixedFutureQueryOverride
ROOT=Path(os.environ.get("EE6008_QUERY_GEOMETRY_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT=Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
SCHED=PROJECT/'reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z'
MECH=PROJECT/'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def sha_float32_array(value):
 a=np.ascontiguousarray(value,dtype=np.float32);h=hashlib.sha256();h.update(a.dtype.str.encode('ascii'));h.update(b'\0');h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes());return h.hexdigest()

def main():
 lock=read_json(ROOT/'protocol/PROTOCOL_LOCKED.json')
 if lock.get('status')!='LOCKED':raise AssertionError('formal protocol is not locked')
 gpu_policy=lock.get('gpu_policy',{})
 shared_watcher=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
 shared_lease=PROJECT/'repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease'
 if gpu_policy.get('watcher_path')!=str(shared_watcher) or gpu_policy.get('watcher_source_sha256')!=sha(shared_watcher):raise AssertionError('protocol does not point to the live shared watcher')
 if gpu_policy.get('lease_dir')!=str(shared_lease) or gpu_policy.get('account_child_slots_current_watcher')!=2:raise AssertionError('shared lease path or two-child policy mismatch')
 if gpu_policy.get('shared_watcher_modified') is not False or gpu_policy.get('local_relaxed_watcher_used') is not False:raise AssertionError('a relaxed local watcher must not be in the execution path')
 resolution=read_json(ROOT/'protocol/RESOURCE_POLICY_RESOLUTION.json')
 if resolution.get('status')!='RESOLVED_KEEP_SHARED_WATCHER_TWO_PROJECT_CHILDREN' or resolution.get('execution_route',{}).get('watcher')!=str(shared_watcher):raise AssertionError('resource policy resolution mismatch')
 if sha(ROOT/'protocol/RESOURCE_POLICY_RESOLUTION.json')!=lock.get('resource_policy_resolution_sha256'):raise AssertionError('resource policy resolution hash mismatch')
 controller=(ROOT/'code/run_orchestrator.py').read_text(encoding='utf-8')
 project_binding = f"PROJECT=Path({str(PROJECT)!r})"
 if ('MAX_ACTIVE=2' not in controller or (project_binding not in controller and "EE6008_PROJECT_ROOT" not in controller) or "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py" not in controller or "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease" not in controller):raise AssertionError('controller shared watcher/two-child route mismatch')
 if 'REJECTED_WATCHER_COPY' in controller:raise AssertionError('historical watcher snapshot was referenced by the controller')
 if sha(ROOT/'protocol/MATRIX12.csv')!=lock['matrix12_sha256']:raise AssertionError('MATRIX12 hash mismatch')
 if sha(ROOT/'protocol/QUERY_TABLES_AND_HASHES.json')!=lock['query_tables_json_sha256']:raise AssertionError('query JSON hash mismatch')
 if sha(ROOT/'protocol/HADAMARD_QUERY_TABLES.npz')!=lock['query_archive_sha256']:raise AssertionError('query archive hash mismatch')
 amendment_path=ROOT/'protocol/AMENDMENT_001_QUERY_ARCHIVE.json'
 amendment=None;amendment_sha256=None;overrides={}
 if amendment_path.is_file():
  amendment=read_json(amendment_path);amendment_sha256=sha(amendment_path)
  receipt_path=ROOT/'coordination/AMENDMENT_001_RECEIPT.json'
  if (amendment.get('status')!='LOCKED_IMPLEMENTATION_AMENDMENT'
      or amendment.get('parent_protocol_sha256')!=sha(ROOT/'protocol/PROTOCOL_LOCKED.json')
      or not receipt_path.is_file()
      or read_json(receipt_path).get('amendment_sha256')!=amendment_sha256):raise AssertionError('protocol amendment/receipt does not match locked parent')
  overrides=amendment.get('task_code_sha256_overrides',{})
 effective_code_hashes=dict(lock['task_code_hashes']);effective_code_hashes.update(overrides)
 for rel,want in effective_code_hashes.items():
  if not (ROOT/rel).is_file() or sha(ROOT/rel)!=want:raise AssertionError(f'locked protocol/amendment code hash mismatch: {rel}')
 external_count=0
 for source_path,want in lock['external_source_observed_hashes'].items():
  q=Path(source_path)
  if not q.is_file() or sha(q)!=want:raise AssertionError(f'locked external source changed: {source_path}')
  external_count+=1
 if external_count!=lock['external_source_hashes_rechecked_at_lock']:raise AssertionError('external source count differs from protocol lock')
 with (ROOT/'protocol/MATRIX12.csv').open(newline='',encoding='utf-8') as f:matrix=list(csv.DictReader(f))
 if len(matrix)!=12 or len({r['run_id'] for r in matrix})!=12:raise AssertionError('MATRIX12 unique-row contract failed')
 if sum(r['action']=='reuse' for r in matrix)!=6 or sum(r['action']=='train_had' for r in matrix)!=6:raise AssertionError('MATRIX12 action counts differ from 6+6')
 if any(int(r['horizon_steps'])!=30 for r in matrix):raise AssertionError('non-H30 cell in matrix')
 # Verify all three accepted INIT states in both model configurations, without CUDA initialization.
 init_rows=list(csv.DictReader((ROOT/'protocol/SEED_INIT_MANIFEST.csv').open(newline='',encoding='utf-8')))
 if len(init_rows)!=3:raise AssertionError('seed INIT manifest must have 3 rows')
 with np.load(ROOT/'protocol/HADAMARD_QUERY_TABLES.npz',allow_pickle=False) as q:
  table=np.asarray(q['candidate_h30'],dtype=np.float32).copy();old_table=np.asarray(q['old_future_h30'],dtype=np.float32).copy()
 parameter_rows=[]
 for row in init_rows:
  init_path=Path(row['init_state_path'])
  if sha(init_path)!=row['init_state_file_sha256']:raise AssertionError(f'INIT file hash mismatch seed={row["seed"]}')
  state=torch.load(init_path,map_location='cpu',weights_only=True)
  for config in ('TS11-D','STATE-CTRL'):
   base_old=D3TimeSpacePredictor(horizon=30,seed=int(row['seed']),video_mode=config);base_old.load_state_dict(state,strict=True)
   base_had=D3TimeSpacePredictor(horizon=30,seed=int(row['seed']),video_mode=config);base_had.load_state_dict(state,strict=True)
   if state_hash(base_old)!=row['init_state_hash'] or state_hash(base_had)!=row['init_state_hash'] or parameter_count(base_old)!=890780:raise AssertionError(f'INIT strict load/count mismatch {config}/{row["seed"]}')
   old_wrap=FixedFutureQueryOverride(base_old,torch.from_numpy(old_table));had_wrap=FixedFutureQueryOverride(base_had,torch.from_numpy(table))
   if parameter_count(old_wrap)!=890780 or parameter_count(had_wrap)!=890780:raise AssertionError('query wrapper changed trainable parameter count')
   old_state=old_wrap.state_dict();had_state=had_wrap.state_dict()
   if old_state.keys()!=had_state.keys():raise AssertionError('old/Hadamard wrapper state key sets differ')
   changed=[name for name in old_state if not torch.equal(old_state[name],had_state[name])]
   if changed!=['future_query_table']:raise AssertionError(f'only the fixed future-query buffer may differ; found {changed}')
   if sha_float32_array(had_wrap.future_query_table.detach().numpy())!=lock['query_table_sha256_h30']:raise AssertionError('candidate buffer hash mismatch')
   parameter_rows.append({'config':config,'seed':int(row['seed']),'base_init_hash':row['init_state_hash'],
    'base_parameter_count':parameter_count(base_old),'old_wrapper_parameter_count':parameter_count(old_wrap),
    'had_wrapper_parameter_count':parameter_count(had_wrap),'candidate_trainable_parameters_added':0,
    'changed_state_keys':changed,'candidate_buffer_sha256':lock['query_table_sha256_h30'],'status':'PASS'})
 param_diff={'schema_version':1,'status':'PASS_ONLY_FUTURE_QUERY_BUFFER_DIFF','checked_pairs':len(parameter_rows),
  'base_parameter_count':890780,'candidate_trainable_parameters_added':0,'state_key_prefix_map':'base.* wrapper prefix; base tensors strict-loaded and equal',
  'only_changed_state_key':'future_query_table','old_query_hash':sha_float32_array(old_table),'candidate_query_hash':sha_float32_array(table),
  'pairs':parameter_rows,'optimizer_created':False,'cuda_initialized':False}
 param_path=ROOT/'qa/PARAM_BUFFER_DIFF.json';param_path.write_text(json.dumps(param_diff,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 # Verify each reusable old control through the typed adapter and both prediction sidecars.
 reuse=list(csv.DictReader((ROOT/'protocol/REUSE_ELIGIBILITY.csv').open(newline='',encoding='utf-8')))
 if len(reuse)!=6:raise AssertionError('expected six reusable Q0 controls')
 oldrunmap={r['source_run_id']:r for r in reuse}
 pa=read_json(SCHED/'PREDICTION_ACCEPTANCE.json')
 pidx={(x['run_id'],x['split']):x for x in pa['bundles']}
 for source_run,row in oldrunmap.items():
  rc=canonical_run(source_run,read_json(SCHED/'runs/formal'/source_run/'RUN_COMPLETE.json'))
  if (rc.status,rc.epochs,rc.optimizer_updates,rc.scheduler_step_calls,rc.parameter_count,rc.horizon_steps)!=('PASS',40,11120,11120,890780,30):raise AssertionError(f'old Q0 run not eligible {source_run}')
  for split,n in [('train',17757),('dev',2322)]:
   b=pidx[(source_run,split)];p=SCHED/'predictions'/split/f'{source_run}.npz';side=canonical_prediction(read_json(SCHED/'predictions'/split/f'{source_run}.json'))
   if b['status']!='PASS' or int(b['rows'])!=n or sha(p)!=b['prediction_sha256'] or side.prediction_sha256!=b['prediction_sha256'] or side.checkpoint_sha256!=rc.checkpoint_sha256:
    raise AssertionError(f'old Q0 prediction sidecar/hash mismatch {source_run}/{split}')
 if list((ROOT/'runs/formal').glob('*/RUN_COMPLETE.json')):raise AssertionError('formal output exists before Stage0')
 result={'schema_version':1,'status':'PASS_CPU_PREFLIGHT','created_utc':lock['locked_utc'],
  'matrix12_rows':12,'q0_reuse_cells':6,'q1_hadamard_cells':6,'old_checkpoint_acceptance':'PASS_6_OF_6',
  'old_train_dev_prediction_hashes':'PASS_12_OF_12','seed_init_strict_loads':'PASS_6_OF_6_MODEL_SEED_PAIRS',
  'candidate_query_hash':lock['query_table_sha256_h30'],'candidate_trainable_parameters_added':0,
  'target_arrays_loaded':False,'dev_targets_opened':False,'final_test_access':0,'heldout_access':0,
  'formal_run_dirs':0,'cuda_initialized':False,'external_sources_rechecked':external_count,'protocol_amendment_id':amendment.get('amendment_id') if amendment else None,'protocol_amendment_sha256':amendment_sha256,'amended_code_hash_count':len(overrides),'parameter_buffer_diff_sha256':sha(ROOT/'qa/PARAM_BUFFER_DIFF.json')}
 p=ROOT/'qa/PREFLIGHT_ACCEPTANCE.json';p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
