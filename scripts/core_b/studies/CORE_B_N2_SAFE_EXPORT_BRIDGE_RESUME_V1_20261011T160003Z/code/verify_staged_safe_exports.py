#!/usr/bin/env python3
"""Fresh-process verification and promotion of task-owned safetensors staging.

This program reads only safetensors and JSON/CSV receipts. It never calls torch.load.
"""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
SAFE_ROOT=ROOT/'safe_models'
REQUEST=ROOT/'MODEL_EXPORT_REQUEST.csv'
SCHEMA=ROOT/'export_review/EXPECTED_STATE_SCHEMA.json'
PLAN=ROOT/'export_review/PLAN.json'
QARCH=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z/protocol/HADAMARD_QUERY_TABLES.npz'))

def sha_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()

def table_sha(values:np.ndarray)->str:
 a=np.ascontiguousarray(values,dtype=np.float32)
 h=hashlib.sha256();h.update(a.dtype.str.encode('ascii'));h.update(b'\0')
 h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
 return h.hexdigest()

def tensor_sha(t:torch.Tensor)->str:
 return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes(order='C'))

def write_json(p:Path,o)->None:
 p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def main()->int:
 ap=argparse.ArgumentParser()
 ap.add_argument('--staging-dir',required=True)
 ap.add_argument('--expected-plan-sha256',required=True)
 args=ap.parse_args()
 root=ROOT.resolve();stage=Path(args.staging_dir).resolve()
 tmp=(ROOT/'tmp').resolve();safe=SAFE_ROOT.resolve()
 if not stage.is_dir() or not stage.is_relative_to(tmp):
  raise RuntimeError('staging path must be an existing directory under this task root tmp')
 if SAFE_ROOT.is_symlink() or any((SAFE_ROOT/r['model_id']).exists() for r in csv.DictReader(REQUEST.open(newline='',encoding='utf-8'))):
  raise RuntimeError('safe_models destination exists or is symlinked; no overwrite')
 with REQUEST.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
 plan=json.loads(PLAN.read_text(encoding='utf-8'))
 if sha_file(PLAN)!=args.expected_plan_sha256:raise RuntimeError('plan SHA does not match command pin')
 if sha_file(REQUEST)!=plan['request_csv_sha256']:raise RuntimeError('model export request SHA changed')
 schema=json.loads(SCHEMA.read_text(encoding='utf-8'))
 if len(rows)!=12 or len({r['model_id'] for r in rows})!=12:raise RuntimeError('request schema is not 12 unique models')
 if sha_file(PLAN)!=plan.get('plan_sha256',sha_file(PLAN)):
  raise RuntimeError('plan file checksum error')
 with np.load(QARCH,allow_pickle=False) as archive:
  query=np.asarray(archive['candidate_h30'],dtype=np.float32).copy()
 if table_sha(query)!=plan['query_table_sha256'] or sha_file(QARCH)!=plan['query_archive_sha256']:
  raise RuntimeError('accepted query table/archive identity changed')
 if sha_file(ROOT/'source/d3_model.py')!=plan['source_model_code_sha256']:raise RuntimeError('copied D3 source hash mismatch')
 if sha_file(ROOT/'source/fixed_future_query_model.py')!=plan['source_q1_wrapper_sha256']:raise RuntimeError('copied Q1 wrapper hash mismatch')
 import sys
 sys.path.insert(0,str(ROOT/'source'))
 from d3_model import D3TimeSpacePredictor,parameter_count
 from fixed_future_query_model import FixedFutureQueryOverride
 checked=[]
 for row in rows:
  mid=row['model_id'];src=stage/mid
  if not src.is_dir() or src.is_symlink():raise RuntimeError('missing or symlinked staging model '+mid)
  manifest=json.loads((src/'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
  cat=json.loads((src/'TENSOR_CATALOGUE.json').read_text(encoding='utf-8'))
  qspec=json.loads((src/'QUERY_SPEC.json').read_text(encoding='utf-8'))
  buffers=json.loads((src/'BUFFER_INVENTORY.json').read_text(encoding='utf-8'))
  model_file=src/'model.safetensors'
  if sha_file(model_file)!=manifest['model_file_sha256']:raise RuntimeError('safetensors file hash mismatch '+mid)
  if sha_file(src/'TENSOR_CATALOGUE.json')!=manifest['tensor_catalogue_sha256']:raise RuntimeError('tensor catalogue file hash mismatch '+mid)
  if sha_file(src/'QUERY_SPEC.json')!=manifest['query_spec_sha256']:raise RuntimeError('query spec file hash mismatch '+mid)
  if sha_file(src/'BUFFER_INVENTORY.json')!=manifest['buffer_inventory_sha256']:raise RuntimeError('buffer inventory file hash mismatch '+mid)
  if manifest['source_checkpoint_path']!=row['checkpoint_path'] or manifest['source_checkpoint_sha256']!=row['checkpoint_sha256']:
   raise RuntimeError('staged export source binding mismatch '+mid)
  if manifest['source_checkpoint_epoch40_sha256']!=row['checkpoint_epoch40_sha256'] or manifest['source_run_complete_sha256']!=row['run_complete_sha256']:
   raise RuntimeError('staged export completion receipt mismatch '+mid)
  if manifest['model_code_sha256']!=row['model_code_sha256'] or manifest['training_protocol_lock_sha256']!=row['training_protocol_lock_sha256']:
   raise RuntimeError('staged export code/protocol binding mismatch '+mid)
  if manifest['query_table_sha256']!=row['accepted_query_table_sha256'] or qspec['required_auxiliary_names']!=[] or buffers['required_auxiliary_names']!=[]:
   raise RuntimeError('staged export query/auxiliary contract mismatch '+mid)
  state=load_file(str(model_file),device='cpu')
  expected=schema[row['query']]['state_dict']
  if set(state)!=set(expected):raise RuntimeError('tensor key mismatch '+mid)
  by_name={x['name']:x for x in cat['tensors']}
  if set(by_name)!=set(state):raise RuntimeError('tensor catalogue keys mismatch '+mid)
  for name,tensor in state.items():
   spec=expected[name];item=by_name[name]
   if type(tensor) is not torch.Tensor or tensor.device.type!='cpu' or tensor.dtype!=torch.float32:
    raise RuntimeError('tensor dtype/device mismatch '+mid+'::'+name)
   if list(tensor.shape)!=spec['shape'] or item['shape']!=spec['shape'] or item['dtype']!='float32':
    raise RuntimeError('tensor shape/catalogue mismatch '+mid+'::'+name)
   if tensor_sha(tensor)!=item['source_tensor_sha256']:raise RuntimeError('tensor fingerprint mismatch '+mid+'::'+name)
  if row['query']=='Q0':
   model=D3TimeSpacePredictor(horizon=30,seed=int(row['seed']),video_mode=row['model'])
  else:
   base=D3TimeSpacePredictor(horizon=30,seed=int(row['seed']),video_mode=row['model'])
   model=FixedFutureQueryOverride(base,torch.from_numpy(query))
   if table_sha(state['future_query_table'].numpy())!=row['accepted_query_table_sha256']:
    raise RuntimeError('Q1 table tensor mismatch '+mid)
  incompatible=model.load_state_dict(state,strict=True)
  if incompatible.missing_keys or incompatible.unexpected_keys:raise RuntimeError('strict load reported key mismatch '+mid)
  if parameter_count(model)!=int(schema['expected_parameter_count']):raise RuntimeError('parameter count mismatch '+mid)
  model.eval()
  if set(model.state_dict())!=set(state):raise RuntimeError('fresh model state keys differ '+mid)
  for name,tensor in model.state_dict().items():
   if tensor_sha(tensor)!=by_name[name]['source_tensor_sha256']:raise RuntimeError('strict model state changed '+mid+'::'+name)
  checked.append({'model_id':mid,'status':'PASS','safetensors_sha256':sha_file(model_file),
   'tensor_count':len(state),'parameter_count':parameter_count(model),'strict_missing_keys':[],
   'strict_unexpected_keys':[],'query_table_sha256':row['accepted_query_table_sha256']})
  del model,state
 if len(checked)!=12:raise RuntimeError('not all 12 staged model reloads passed')
 if any((SAFE_ROOT/r['model_id']).exists() for r in rows):raise FileExistsError('destination exists before promotion')
 for row in rows:os.replace(stage/row['model_id'],SAFE_ROOT/row['model_id'])
 receipt={'status':'PASS_SAFE_TENSORS_FRESH_PROCESS_RELOAD_AND_PROMOTION','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'fresh_process':True,'loader':'safetensors.torch.load_file(device=cpu)','torch_load_called':False,
  'models':checked,'promoted_count':12,'safe_models_root':str(SAFE_ROOT),'final_test_access':0}
 write_json(ROOT/'qa/SAFE_ONLY_RELOAD.json',receipt)
 write_json(ROOT/'qa/SAFE_EXPORT_ACCEPTANCE.json',receipt)
 write_json(stage/'PROMOTION_RECEIPT.json',{'status':'PASS','promoted_model_ids':[r['model_id'] for r in rows],
  'receipt_sha256':sha_file(ROOT/'qa/SAFE_ONLY_RELOAD.json')})
 print(json.dumps({'status':receipt['status'],'promoted_count':12,'safe_models_root':str(SAFE_ROOT)},sort_keys=True),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
