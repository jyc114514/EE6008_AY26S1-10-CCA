#!/usr/bin/env python3
"""Reconcile V8 GT_RETRAIN with same-seed accepted old V hard model/predictions; no targets read."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
SEEDS=(202610131,202610132,202610133)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p):return json.loads(Path(p).read_text())
def load_bundle(pred_path,rows_path,expected_sha,expected_rows_sha,ids):
 if sha(pred_path)!=expected_sha or sha(rows_path)!=expected_rows_sha:raise RuntimeError(f'old prediction checksum mismatch: {pred_path}')
 a=load_file(str(pred_path),device='cpu')['prediction_native'].numpy()
 if a.dtype not in (np.float32,np.float64) or not np.isfinite(a).all():raise RuntimeError(f'prediction dtype/finite mismatch: {pred_path}')
 a=np.asarray(a,dtype=np.float64)
 with Path(rows_path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 by={str(x['anchor_id']):x for x in rows}
 if len(by)!=len(ids) or set(by)!=set(ids):raise RuntimeError(f'prediction ID set differs: {pred_path}')
 order=[]
 for aid in ids:order.append(int(by[aid].get('row',by[aid].get('target_row'))))
 if not np.array_equal(np.sort(order),np.arange(len(ids))):raise RuntimeError(f'prediction row mapping invalid: {pred_path}')
 return a[np.asarray(order,dtype=np.int64)]
def compare(new_root,old_path,old_rows,old_sha,old_rows_sha,ids,label,seed):
 newrec=j(new_root/'PREDICTION_COMPLETE.json');newp=Path(newrec['prediction_path']);newr=Path(newrec['prediction_rows_path'])
 if sha(newp)!=newrec['prediction_sha256'] or sha(newr)!=newrec['prediction_rows_sha256']:raise RuntimeError(f'new prediction checksum mismatch {label}/{seed}')
 new=load_bundle(newp,newr,newrec['prediction_sha256'],newrec['prediction_rows_sha256'],ids)
 old=load_bundle(old_path,old_rows,old_sha,old_rows_sha,ids)
 d=np.abs(new-old)
 tol_ratio=d/(1e-6+1e-6*np.abs(old));within=bool(np.allclose(new,old,atol=1e-6,rtol=1e-6))
 if not within:raise RuntimeError(f'{label} prediction replay outside atol=rtol=1e-6: ratio={tol_ratio.max()}')
 return {'dataset_role':label,'seed':seed,'rows':len(ids),'old_prediction_sha256':old_sha,'new_prediction_sha256':newrec['prediction_sha256'],
  'max_abs_diff':float(d.max()),'max_rel_diff':float((d/(np.abs(old)+1e-12)).max()),'max_atol_rtol_tolerance_ratio':float(tol_ratio.max()),
  'passed_atol_rtol_1e-6':within,'array_exact':bool(np.array_equal(new,old)),'old_rows_sha256':old_rows_sha,'new_rows_sha256':newrec['prediction_rows_sha256']}
def main():
 role=j(V7/'protocol/TEACHER_ROLE_MAP.json');source_map=j(V7/'inventory/SOURCE_MAP.json');report=[]
 for seed in SEEDS:
  v=role['sources'][str(seed)]['V_MATCHED'];slot=ROOT/'model_only/GT_RETRAIN'/f'seed{seed}'/'attempt_01';m=j(slot/'MODEL_MANIFEST.json')
  model_exact=sha(v['model_path'])==sha(slot/'model.safetensors') and v['model_state_hash']==m['final_state_hash']
  if not model_exact:raise RuntimeError(f'new hard weights differ from old V weights seed={seed}')
  old_train_rec=j(V7/'teacher_predictions/source/V_MATCHED'/f'seed{seed}'/'PREDICTION_COMPLETE.json')
  oldtrain=compare(ROOT/'predictions/train/GT_RETRAIN'/f'seed{seed}'/'attempt_01',Path(old_train_rec['prediction_path']),Path(old_train_rec['prediction_rows_path']),old_train_rec['prediction_sha256'],old_train_rec['prediction_rows_sha256'],
   np.load(V7/'data/TRAIN_PREFIXES_NO_LABELS.npz',allow_pickle=False)['anchor_id'].astype(str).tolist(),'train',seed)
  dev_ref=next(x for x in source_map['dev_reference_prediction_bundles'] if x.get('role')=='V_MATCHED' and int(x['seed'])==seed)
  dev_ids=np.load(V2/'data/DEV_PREFIXES_NO_LABELS.npz',allow_pickle=False)['anchor_id'].astype(str).tolist()
  olddev=compare(ROOT/'predictions/dev/GT_RETRAIN'/f'seed{seed}'/'attempt_01',Path(dev_ref['prediction_path']),Path(dev_ref['prediction_rows_path']),dev_ref['prediction_sha256'],dev_ref['prediction_rows_sha256'],dev_ids,'dev',seed)
  report.append({'seed':seed,'model_file_sha_equal':True,'model_state_hash_equal':True,'old_v_model_sha256':v['model_sha256'],'new_GT_model_sha256':m['model_file_sha256'],
    'old_v_state_hash':v['model_state_hash'],'new_GT_state_hash':m['final_state_hash'],'train_prediction_replay':oldtrain,'dev_prediction_replay':olddev})
 result={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'comparison':'branch-B GT_RETRAIN fresh hard fits vs accepted same-seed old V hard-video model and its sealed train/dev predictions',
  'all_three_hard_model_files_bit_identical':True,'all_three_state_hashes_identical':True,'prediction_replay_tolerance':{'atol':1e-6,'rtol':1e-6},'all_train_dev_replays_within_tolerance':True,'train_target_archive_read':False,'dev_target_archive_opened':False,
  'final_heldout_payload_access':0,'by_seed':report}
 p=ROOT/'qa/HARD_BASELINE_REPRODUCTION.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','model_weight_sha_equal_seeds':3,'train_prediction_exact':all(x['train_prediction_replay']['array_exact'] for x in report),
  'dev_prediction_within_1e-6':all(x['dev_prediction_replay']['passed_atol_rtol_1e-6'] for x in report),'max_dev_abs_diff':max(x['dev_prediction_replay']['max_abs_diff'] for x in report)},sort_keys=True))
if __name__=='__main__':main()
