#!/usr/bin/env python3
"""Seal all V7 student train/dev prediction bundles before any dev target reader."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V6=PROJECT/'reports/benchmarks/CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z'
PROGRAMS=('ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV');SEEDS=(202610131,202610132,202610133)
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read_prefix(path:Path)->dict[str,np.ndarray]:
 fields=('anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame')
 with np.load(path,allow_pickle=False) as z:
  if 'target_native_h30' in z.files:raise RuntimeError(f'label field in prefix archive: {path}')
  return {k:z[k].copy() for k in fields}
def check_rows(path:Path,prefix:dict[str,np.ndarray],dataset:str)->None:
 with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 ids=prefix['anchor_id'].astype(str);n=len(ids)
 if len(rows)!=n or len({str(r['anchor_id']) for r in rows})!=n or {str(r['anchor_id']) for r in rows}!=set(ids):raise RuntimeError(f'{dataset} prediction IDs incomplete or duplicated: {path}')
 by={str(r['anchor_id']):r for r in rows}
 for i,aid in enumerate(ids.tolist()):
  r=by[aid]
  expected=(int(prefix['episode_id'][i]),int(prefix['task_id'][i]),str(prefix['source_cluster'][i]),int(prefix['anchor_frame'][i]))
  observed=(int(r['episode_id']),int(r['task_id']),str(r['source_cluster']),int(r['anchor_frame']))
  if expected!=observed:raise RuntimeError(f'{dataset} metadata join mismatch for {aid}')
  if dataset=='dev' and r.get('dataset_role')!='dev':raise RuntimeError(f'dev prediction row has wrong role {aid}')

def main()->int:
 resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text(encoding='utf-8'))
 train=read_prefix(Path(resolved['train_prefix_npz']));dev=read_prefix(Path(resolved['dev_prefix_npz']))
 if len(train['anchor_id'])!=1280 or len(dev['anchor_id'])!=185:raise RuntimeError('locked train/dev prefix row counts differ')
 accepted=[];missing=[]
 for program in PROGRAMS:
  for seed in SEEDS:
   slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';manifest_path=slot/'MODEL_MANIFEST.json';model_path=slot/'model.safetensors'
   if not manifest_path.is_file() or not model_path.is_file():raise RuntimeError(f'student model missing: {program}/{seed}')
   manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
   if manifest.get('status')!='PASS' or manifest.get('student_inference_requires_teacher') is not False:raise RuntimeError(f'model manifest not teacher-independent: {program}/{seed}')
   if sha(model_path)!=manifest.get('model_file_sha256'):raise RuntimeError(f'model weight checksum mismatch: {program}/{seed}')
   for dataset,prefix,expected_rows in (('train',train,1280),('dev',dev,185)):
    out=ROOT/'predictions'/dataset/program/f'seed{seed}'/'attempt_01';complete=out/'PREDICTION_COMPLETE.json'
    if not complete.is_file():missing.append(f'{dataset}:{program}:{seed}');continue
    receipt=json.loads(complete.read_text(encoding='utf-8'))
    if receipt.get('status')!='PASS' or receipt.get('program_id')!=program or int(receipt.get('seed',-1))!=seed or receipt.get('dataset_role')!=dataset:raise RuntimeError(f'prediction completion receipt mismatch: {dataset}/{program}/{seed}')
    if receipt.get('teacher_or_target_paths_supplied') is not False or receipt.get('future_GT_requested_by_inference_loader') is not False:raise RuntimeError(f'inference receipt reports teacher/target dependency: {dataset}/{program}/{seed}')
    pred_path=Path(receipt['prediction_path']);rows_path=Path(receipt['prediction_rows_path'])
    if sha(pred_path)!=receipt.get('prediction_sha256') or sha(rows_path)!=receipt.get('prediction_rows_sha256'):raise RuntimeError(f'prediction or row checksum mismatch: {dataset}/{program}/{seed}')
    tensors=load_file(str(pred_path),device='cpu')
    if 'prediction_native' not in tensors:raise RuntimeError(f'prediction_native key absent: {dataset}/{program}/{seed}')
    pred=tensors['prediction_native']
    if tuple(pred.shape)!=(expected_rows,30,28) or not np.isfinite(pred.numpy()).all():raise RuntimeError(f'prediction shape/finite failure: {dataset}/{program}/{seed}')
    if pred.dtype!=__import__('torch').float64:raise RuntimeError(f'prediction dtype differs from native float64 contract: {dataset}/{program}/{seed}')
    check_rows(rows_path,prefix,dataset)
    if receipt.get('model_sha256')!=manifest.get('model_file_sha256') or receipt.get('model_state_hash')!=manifest.get('final_state_hash'):raise RuntimeError(f'prediction model provenance mismatch: {dataset}/{program}/{seed}')
    accepted.append({'dataset_role':dataset,'program_id':program,'seed':seed,'rows':expected_rows,'prediction_path':str(pred_path),'prediction_sha256':receipt['prediction_sha256'],'prediction_rows_path':str(rows_path),'prediction_rows_sha256':receipt['prediction_rows_sha256'],'model_path':str(model_path),'model_sha256':manifest['model_file_sha256'],'completion_path':str(complete),'completion_sha256':sha(complete)})
 if missing:raise RuntimeError(f'cannot seal incomplete student matrix: {missing}')
 if len([r for r in accepted if r['dataset_role']=='train'])!=18 or len([r for r in accepted if r['dataset_role']=='dev'])!=18:raise RuntimeError('student bundle matrix is not 18 train and 18 dev')
 lock={'schema_version':1,'status':'SEALED_BEFORE_TARGET_ACCESS','nonempty_seal':True,'created_utc':utc(),
       'study_lock_sha256':sha(ROOT/'protocol/STUDY_LOCK.json'),'target_mix_lock_sha256':sha(ROOT/'protocol/TARGET_MIX_LOCK.json'),
       'accepted_train_bundles':18,'accepted_dev_bundles':18,'accepted_logical_bundles':36,'missing_bundles':0,'empty_bundles':0,
       'expected_student_programs':list(PROGRAMS),'formal_seeds':list(SEEDS),'train_rows_per_bundle':1280,'dev_rows_per_bundle':185,
       'dev_unique_receivers':185,'dev_target_reader_called':False,'preseal_dev_target_reads':0,'dev_target_archive_opened':False,
       'all_predictions_finite':True,'all_ids_and_source_groups_joined':True,'model_only_standalone_inference':True,
       'teacher_or_GT_paths_supplied_to_inference':False,'final_heldout_payload_access':0,'bundles':accepted}
 p=ROOT/'qa/PREDICTION_SEAL.json';p.write_text(json.dumps(lock,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':lock['status'],'accepted_train':18,'accepted_dev':18,'seal_sha256':sha(p),'dev_target_opened':False},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
