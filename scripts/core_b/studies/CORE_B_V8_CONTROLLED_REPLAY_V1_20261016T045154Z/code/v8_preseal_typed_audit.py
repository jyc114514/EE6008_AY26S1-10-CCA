#!/usr/bin/env python3
"""Typed, label-free audit of all V8 train/dev predictions before dev target scoring."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT');SEEDS=(202610131,202610132,202610133)
FORBIDDEN={'teacher_target_path','teacher_target_rows_path','teacher_target_manifest_path','teacher_model_path','dev_target_path','target_archive_path','train_npz_path'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p):return json.loads(Path(p).read_text())
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 resolved=j(ROOT/'inventory/RESOLVED_PATHS.json');results=[]
 for dataset,prefix_path,expected in [('train',Path(resolved['train_prefix_npz']),1280),('dev',Path(resolved['dev_prefix_npz']),185)]:
  with np.load(prefix_path,allow_pickle=False) as z:
   if 'target_native_h30' in z.files:raise RuntimeError(f'{dataset} prefix has target field')
   ids=z['anchor_id'].astype(str).tolist();eps=z['episode_id'];tasks=z['task_id'];groups=z['source_cluster'].astype(str);frames=z['anchor_frame']
  if len(ids)!=expected or len(set(ids))!=expected:raise RuntimeError(f'{dataset} prefix support failure')
  for program in PROGRAMS:
   for seed in SEEDS:
    cp=ROOT/'configs/inference'/f'v8_pred_{dataset}_{program}_s{seed}.json';cfg=j(cp)
    if type(cfg.get('seed')) is not int or cfg['seed']!=seed or type(cfg.get('expected_rows')) is not int or cfg['expected_rows']!=expected:raise RuntimeError(f'numeric config seed/rows invalid {dataset}/{program}/{seed}')
    if cfg.get('dataset_role')!=dataset or cfg.get('program_id')!=program or cfg.get('video_mode')!='TS11-D':raise RuntimeError(f'config role/profile invalid {dataset}/{program}/{seed}')
    if FORBIDDEN.intersection(cfg):raise RuntimeError(f'forbidden target/teacher paths in inference config: {dataset}/{program}/{seed}')
    if cfg.get('branch_lock_sha256')!=sha(ROOT/'protocol/BRANCH_LOCK.json') or cfg.get('recipe_lock_sha256')!=sha(ROOT/'protocol/RECIPE_LOCK.json'):raise RuntimeError('inference config lock hash mismatch')
    out=ROOT/'predictions'/dataset/program/f'seed{seed}'/'attempt_01';rec=j(out/'PREDICTION_COMPLETE.json');mp=Path(cfg['model_manifest_path']);manifest=j(mp)
    if rec.get('status')!='PASS' or type(rec.get('seed')) is not int or rec['seed']!=seed or rec.get('dataset_role')!=dataset or rec.get('program_id')!=program:raise RuntimeError(f'prediction receipt types/roles mismatch {dataset}/{program}/{seed}')
    if rec.get('teacher_or_target_paths_supplied') is not False or rec.get('future_GT_requested_by_inference_loader') is not False:raise RuntimeError('prediction receipt indicates target/teacher access')
    pp=Path(rec['prediction_path']);rp=Path(rec['prediction_rows_path'])
    if sha(pp)!=rec['prediction_sha256'] or sha(rp)!=rec['prediction_rows_sha256'] or sha(mp)!=cfg['model_manifest_sha256'] or sha(Path(cfg['model_path']))!=cfg['model_sha256']:raise RuntimeError('prediction/model checksum mismatch')
    t=load_file(str(pp),device='cpu')
    if set(t.keys())!={'prediction_native'}:raise RuntimeError(f'illegal prediction tensor keys: {set(t.keys())}')
    a=t['prediction_native'].numpy()
    if a.shape!=(expected,30,28) or a.dtype!=np.float64 or not np.isfinite(a).all():raise RuntimeError(f'prediction shape/dtype/finite failure {dataset}/{program}/{seed}')
    with rp.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    by={x['anchor_id']:x for x in rows}
    if len(rows)!=expected or len(by)!=expected or set(by)!=set(ids):raise RuntimeError(f'prediction rows support mismatch {dataset}/{program}/{seed}')
    order=[]
    for i,aid in enumerate(ids):
     row=by[aid]
     if (int(row['episode_id']),int(row['task_id']),str(row['source_cluster']),int(row['anchor_frame']),row['dataset_role'])!=(int(eps[i]),int(tasks[i]),str(groups[i]),int(frames[i]),dataset):raise RuntimeError(f'prediction typed identity mismatch {aid}')
     order.append(int(row['row']))
    if not np.array_equal(np.sort(order),np.arange(expected)):raise RuntimeError(f'prediction row map not one-to-one {dataset}/{program}/{seed}')
    if rec.get('model_sha256')!=manifest.get('model_file_sha256') or rec.get('model_state_hash')!=manifest.get('final_state_hash'):raise RuntimeError(f'prediction/model manifest bind mismatch {dataset}/{program}/{seed}')
    results.append({'dataset_role':dataset,'program_id':program,'seed':seed,'rows':expected,'prediction_sha256':rec['prediction_sha256'],'rows_sha256':rec['prediction_rows_sha256'],'model_sha256':rec['model_sha256'],'numeric_seed':True,'finite_float64':True,'IDs_joined':True})
 # Validate old source per-window CSV numeric columns/seam before reusing it; no dev target archive is read.
 old=V7/'results/PER_WINDOW_ALL_METHODS.csv';old_rows=[]
 with old.open(newline='',encoding='utf-8') as f:old_rows=list(csv.DictReader(f))
 old_methods={'S','A','V','SS50','SV50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV'}
 selected=[x for x in old_rows if x['method'] in old_methods]
 for x in selected:
  seed=int(x['seed'])
  if seed not in SEEDS:raise RuntimeError('historical V7 seed fails numeric conversion')
  for m in ('position_mae','safe_fd_l1','seam_safe_fd_l1'):float(x[m])
 if len(selected)!=7770:raise RuntimeError(f'historical V7 known-grain rows expected7770 got{len(selected)}')
 seal={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'prediction_bundles_checked':18,
  'train_rows_per_bundle':1280,'dev_rows_per_bundle':185,'prediction_rows':results,'historical_v7_rows_numeric_and_seam_present':len(selected),
  'historical_v7_source_sha256':sha(old),'dev_target_archive_opened':False,'dev_target_values_read':False,'final_heldout_payload_access':0,
  'all_program_seed_roles_and_numeric_seeds':'PASS','only_prediction_native_key':'PASS','all_predictions_nonempty_finite_float64':'PASS'}
 write(ROOT/'qa/PRESEAL_TYPED_AUDIT.json',seal)
 print(json.dumps({'status':'PASS','bundles':len(results),'historical_rows':len(selected),'dev_target_opened':False},sort_keys=True))
if __name__=='__main__':main()
