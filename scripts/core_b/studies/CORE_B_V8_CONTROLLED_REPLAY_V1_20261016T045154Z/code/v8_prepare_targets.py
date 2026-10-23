#!/usr/bin/env python3
"""Prepare V8 provenance and train-only fixed Y/Z banks; never opens dev targets."""
from __future__ import annotations
import os
import csv, hashlib, json, os, socket
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from safetensors.torch import load_file

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT=Path(__file__).resolve().parents[1]
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V6=PROJECT/'reports/benchmarks/CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
SEEDS=(202610131,202610132,202610133)
TASKS=(119,94,90,141,230)
PYTHON=PROJECT/'envs/jyc/bin/python'
WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'

def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def write_json(path:Path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def atomic_npy(path:Path,array:np.ndarray):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 with tmp.open('wb') as f: np.save(f,np.asarray(array),allow_pickle=False); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def read_json(path:Path): return json.loads(path.read_text(encoding='utf-8'))
def read_npz(path:Path):
 with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def read_prediction(role:str,seed:int,data:dict[str,np.ndarray]):
 bundle=V7/'teacher_predictions/source'/role/f'seed{seed}'; receipt=read_json(bundle/'PREDICTION_COMPLETE.json')
 if receipt.get('status')!='PASS' or receipt.get('teacher_role')!=role or int(receipt.get('seed',-1))!=seed: raise RuntimeError(f'unaccepted source replay {role}/{seed}')
 pp=Path(receipt['prediction_path']); rp=Path(receipt['prediction_rows_path'])
 if sha(pp)!=receipt['prediction_sha256'] or sha(rp)!=receipt['prediction_rows_sha256']: raise RuntimeError(f'prediction checksum mismatch {role}/{seed}')
 tensor=load_file(str(pp),device='cpu')['prediction_native'].numpy()
 if tensor.shape!=(1280,30,28) or tensor.dtype!=np.float64 or not np.isfinite(tensor).all(): raise RuntimeError(f'prediction schema failure {role}/{seed}')
 with rp.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
 by_id={r['anchor_id']:r for r in rows}; ids=data['anchor_id'].astype(str)
 if len(rows)!=1280 or len(by_id)!=1280 or set(by_id)!=set(ids): raise RuntimeError(f'prediction row ID set mismatch {role}/{seed}')
 order=np.empty(1280,np.int64)
 for i,aid in enumerate(ids):
  row=by_id[str(aid)]
  if (int(row['episode_id'])!=int(data['episode_id'][i]) or int(row['task_id'])!=int(data['task_id'][i]) or str(row['source_cluster'])!=str(data['source_cluster'][i])): raise RuntimeError(f'prediction metadata mismatch {role}/{seed}/{aid}')
  order[i]=int(row['row'])
 if not np.array_equal(np.sort(order),np.arange(1280)): raise RuntimeError(f'prediction row map not bijective {role}/{seed}')
 return tensor[order].copy(),receipt

def write_target(program:str,seed:int,values:np.ndarray,data:dict[str,np.ndarray],*,mode:str,beta:float,expression:str,roles:list[str],source_hashes:dict):
 out=ROOT/'targets'/program/f'seed{seed}'; out.mkdir(parents=True,exist_ok=True)
 target=out/'Z_TRAIN_NATIVE.npy'; atomic_npy(target,values)
 rows=out/'TARGET_ROWS.csv'; fields=['target_row','anchor_id','episode_id','task_id','source_cluster','anchor_frame','accepted_train_row']
 with rows.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for i in range(1280): w.writerow({'target_row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),'task_id':int(data['task_id'][i]),'source_cluster':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i]),'accepted_train_row':i})
 manifest={'schema_version':1,'status':'PASS','created_utc':utc(),'program_id':program,'seed':seed,'target_mode':mode,'target_expression':expression,'beta':beta,
           'target_role':'HARD_Y' if mode=='hard_y' else ('V_MATCHED' if program=='SELF_SOFT' else 'SV50'),
           'source_roles':roles,'source_prediction_sha256':source_hashes,'teacher_train_predictions_are_oof':False,
           'train_npz_path':str(V2/'data/TRAIN_SELECTED_H30.npz'),'train_npz_sha256':sha(V2/'data/TRAIN_SELECTED_H30.npz'),
           'normalizer_path':str(V7/'normalizers/NORMALIZER_FORMAL.json'),'normalizer_sha256':sha(V7/'normalizers/NORMALIZER_FORMAL.json'),'normalizer_refit_on_Z':False,
           'target_path':str(target),'target_sha256':sha(target),'target_array_sha256':hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
           'target_shape':list(values.shape),'target_dtype':str(values.dtype),'rows_path':str(rows),'rows_sha256':sha(rows),'row_count':1280,'unique_anchor_ids':1280,
           'dev_target_rows':0,'final_heldout_payload_access':0}
 write_json(out/'TARGET_MANIFEST.json',manifest)
 return {'program_id':program,'seed':seed,'target_path':str(target),'target_sha256':manifest['target_sha256'],'rows_path':str(rows),'rows_sha256':manifest['rows_sha256'],'manifest_path':str(out/'TARGET_MANIFEST.json'),'manifest_sha256':sha(out/'TARGET_MANIFEST.json'),'beta':beta,'target_mode':mode,'target_expression':expression,'source_roles':roles,'source_prediction_sha256':source_hashes}

def main():
 train_path=V2/'data/TRAIN_SELECTED_H30.npz'; prefix_path=V7/'data/TRAIN_PREFIXES_NO_LABELS.npz'; dev_prefix=V4/'data/DEV_PREFIXES_NO_LABELS.npz'
 data=read_npz(train_path); prefix=read_npz(prefix_path); norm_path=V7/'normalizers/NORMALIZER_FORMAL.json'; norm=read_json(norm_path)
 if len(data['anchor_id'])!=1280 or len(set(data['anchor_id'].astype(str)))!=1280: raise RuntimeError('outer-train population identity failure')
 prefix_by={str(x):i for i,x in enumerate(prefix['anchor_id'].astype(str))}
 if set(prefix_by)!=set(data['anchor_id'].astype(str)): raise RuntimeError('train prefix IDs differ from accepted train')
 for key in ('episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native'):
  ix=np.asarray([prefix_by[str(a)] for a in data['anchor_id'].astype(str)])
  if not np.array_equal(data[key],prefix[key][ix]): raise RuntimeError(f'prefix input mismatch {key}')
 y=np.asarray(data['target_native_h30'],dtype=np.float64)
 if y.shape!=(1280,30,28) or not np.isfinite(y).all(): raise RuntimeError('Y target shape/finite failure')
 target_rows=[]; role_rows=[]
 for seed in SEEDS:
  s,sr=read_prediction('S_OPT',seed,data); v,vr=read_prediction('V_MATCHED',seed,data)
  zself=y+0.5*(v-y); zsv=y+0.25*(s-y)+0.25*(v-y)
  for program,mode,beta,expr,values,roles,hashes in [
   ('GT_RETRAIN','hard_y',0.0,'Z=Y',y,['Y'],{}),
   ('SELF_SOFT','fixed_soft_z',0.5,'Z=0.5*Y+0.5*V',zself,['Y','V_MATCHED'],{'V_MATCHED':vr['prediction_sha256']}),
   ('SV_SOFT','fixed_soft_z',0.5,'Z=0.5*Y+0.25*S+0.25*V',zsv,['Y','S_OPT','V_MATCHED'],{'S_OPT':sr['prediction_sha256'],'V_MATCHED':vr['prediction_sha256']})]:
    row=write_target(program,seed,values,data,mode=mode,beta=beta,expression=expr,roles=roles,source_hashes=hashes); target_rows.append(row)
    role_rows.append({'seed':seed,'program_id':program,'target_role':row['source_roles'],'target_sha256':row['target_sha256'],'source_prediction_sha256':hashes})
  # Verify V8's direct-native target build against the accepted V7 VIDEO target banks.
  for old_name,new_values in [('VD_SELF',zself),('VD_SV',zsv)]:
   old_path=V7/'teacher_targets'/old_name/f'seed{seed}'/'Z_TRAIN_NATIVE.npy'
   old=np.load(old_path,mmap_mode='r',allow_pickle=False)
   if not np.array_equal(old,new_values): raise RuntimeError(f'new V8 direct formula differs from accepted V7 bank {old_name}/{seed}')
  if not np.isfinite(s).all() or not np.isfinite(v).all(): raise RuntimeError(f'nonfinite source replay seed {seed}')
  role_rows.append({'seed':seed,'program_id':'SOURCE_REPLAYS','target_role':['S_OPT','V_MATCHED'],'target_sha256':None,'source_prediction_sha256':{'S_OPT':sr['prediction_sha256'],'V_MATCHED':vr['prediction_sha256']},'model_path_S':sr['model_path'],'model_sha256_S':sr['model_sha256'],'model_path_V':vr['model_path'],'model_sha256_V':vr['model_sha256']})

 split=V2/'protocol/INNER_SPLIT.csv'; cache=Path(read_json(V7/'inventory/RESOLVED_PATHS.json')['feature_cache'])
 watcher_hash=sha(WATCHER)
 resolved={'schema_version':1,'status':'PASS','project_root':str(PROJECT),'repo_root':str(PROJECT/'repo/ee6008-humanoid-vjepa'),'git_state':'UNKNOWN: git executable/.git metadata absent in accepted V4 locator',
  'v2_root':str(V2),'v4_root':str(V4),'v6_root':str(V6),'v7_root':str(V7),'out_root':str(ROOT),'train_npz':str(train_path),'train_npz_sha256':sha(train_path),
  'train_prefix_npz':str(prefix_path),'train_prefix_npz_sha256':sha(prefix_path),'dev_prefix_npz':str(dev_prefix),'dev_prefix_sha256':sha(dev_prefix),
  'dev_target_archive_path':str(V2/'data/DEV_TARGETS_SCORER_ONLY.npz'),'dev_target_archive_sha256':'NOT_READ_BY_V8',
  'split_path':str(split),'split_sha256':sha(split),'normalizer_path':str(norm_path),'normalizer_sha256':sha(norm_path),
  'metric_scale_lock_path':str(V4/'protocol/METRIC_SCALE_LOCK.json'),'metric_scale_lock_sha256':sha(V4/'protocol/METRIC_SCALE_LOCK.json'),
  'feature_cache':str(cache),'feature_cache_manifest_sha256':sha(cache),'watcher':str(WATCHER),'watcher_sha256':watcher_hash,'python_bin':str(PYTHON),
  'resolved_utc':utc(),'outer_train_rows':1280,'outer_train_episodes':int(len(np.unique(data['episode_id']))),'dev_rows':185,'dev_target_content_read_by_v8':False,'final_payload_access':0}
 write_json(ROOT/'inventory/RESOLVED_PATHS.json',resolved)
 v7_fold=read_json(V7/'inventory/FOLD_REUSE.json')
 fold={'schema_version':1,'status':'PASS','reuse_source':'V2 corrected literal train/dev membership and ordered IDs, accepted through V7','source_path':str(V7/'inventory/FOLD_REUSE.json'),
       'source_sha256':sha(V7/'inventory/FOLD_REUSE.json'),'v2_split_path':str(split),'v2_split_sha256':sha(split),'same_split_salt_or_repartition':False,
       'outer_train':{'episodes':220,'anchors':1280},'dev':{'episodes':47,'anchors':185},'tasks':list(TASKS),'formal_seeds':list(SEEDS),
       'session_or_trial_independence_verified':v7_fold.get('session_or_trial_independence_verified',False),'final_heldout_payload_access':0}
 write_json(ROOT/'inventory/FOLD_REUSE.json',fold)
 source_map={'schema_version':1,'created_utc':utc(),'hostname':socket.gethostname(),'unix_uid':os.getuid(),'git_state':resolved['git_state'],'node_classification':'configured shared GPU host exposes physical GPU devices; no scheduler environment variables were present; login-vs-interactive classification not independently established',
  'source_roots':{'V2':str(V2),'V4':str(V4),'V6':str(V6),'V7':str(V7)},'source_assets':{},'reused_teacher_sources':role_rows,
  'v7_runner_source_sha256':sha(V7/'code/v7_train.py'),'v7_runner_has_fit_level_source_hash_receipts':False,
  'v4_loss_source_sha256':sha(V4/'code/locked_source/d3_eq_contract.py'),'v7_copied_loss_source_sha256':sha(V7/'code/locked_source/d3_eq_contract.py'),
  'v8_copied_loss_source_sha256':sha(ROOT/'code/locked_source/d3_eq_contract.py'),'shared_loss_source_exact_match_V4_V7_V8':len({sha(V4/'code/locked_source/d3_eq_contract.py'),sha(V7/'code/locked_source/d3_eq_contract.py'),sha(ROOT/'code/locked_source/d3_eq_contract.py')})==1,
  'v8_local_runner_sha256':sha(ROOT/'code/v8_train.py'),'old_roots_written':False,'final_heldout_payload_access':0}
 for name,path in [('train_npz',train_path),('train_prefix_npz',prefix_path),('dev_prefix_npz',dev_prefix),('split',split),('normalizer',norm_path),('metric_scale_lock',V4/'protocol/METRIC_SCALE_LOCK.json'),('feature_cache_manifest',cache),('watcher',WATCHER),('v7_target_mix_lock',V7/'protocol/TARGET_MIX_LOCK.json'),('v7_teacher_role_map',V7/'protocol/TEACHER_ROLE_MAP.json')]:
  source_map['source_assets'][name]={'path':str(path),'sha256':sha(path),'read_only':True}
 write_json(ROOT/'inventory/SOURCE_MAP.json',source_map)
 v7_exp=read_json(V7/'inventory/OUTCOME_EXPOSURE.json')
 exposure={'schema_version':1,'status':'running','created_utc':utc(),'prior_v5_v6_v7_known_grain_primary_loss_evaluations':7770,'legacy_35523_count':'unit_unverified_excluded',
  'v8_unique_dev_windows':185,'v8_dev_target_archive_content_reads':0,'v8_dev_rows_loaded':0,'v8_new_dev_student_evaluations':0,'v8_optimizer0_dev_target_reads':0,
  'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,'v7_prior_dev_target_content_reads':v7_exp.get('dev_target_content_reads',1),
  'v7_prior_unique_dev_windows':185,'v8_training_rows':0,'v8_train_target_presentations':0,'v8_full_formal_fit_count':0,'status_detail':'V8 has not opened any dev target content; new target banks use outer-train Y and frozen train-only teacher predictions.'}
 write_json(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)

 curves=V7/'figure_data/TRAINING_CURVES.csv'; teachers=V7/'results/TEACHER_TRAIN_GT_METRICS.csv'; imitation=V7/'results/IMITATION_ERROR.csv'
 with curves.open(newline='',encoding='utf-8') as f: curve_rows=list(csv.DictReader(f))
 with teachers.open(newline='',encoding='utf-8') as f: teacher_rows=list(csv.DictReader(f))
 with imitation.open(newline='',encoding='utf-8') as f: imitation_rows=list(csv.DictReader(f))
 tmap={'A':'A_COMMON','V':'V_MATCHED','SS50':'SS50','SV50':'SV50','S':'S_OPT'}
 # The train-only V7 source has actual curves and teacher/imitation descriptions; no dev table is read here.
 teacher_index={(r['role'],int(r['seed']),r['task_id']):r for r in teacher_rows}
 im_index={(r['program_id'],int(r['seed']),r['dataset_role']):r for r in imitation_rows if r['dataset_role']=='train'}
 role_by_program={'ST_SELF':'A','ST_SS':'SS50','ST_SV':'SV50','VD_SELF':'V','VD_SS':'SS50','VD_SV':'SV50'}
 last={(r['program_id'],int(r['seed'])):r for r in curve_rows if int(r['epoch'])==160}
 records=[]
 for (program,seed),r in sorted(last.items()):
  role=role_by_program[program]; teacher=teacher_index.get((tmap[role],seed,'macro'),{})
  imi=im_index.get((program,seed,'train'),{})
  records.append({'program_id':program,'seed':seed,'epoch':160,'v7_training_target_position':r['soft_z_position'],'v7_training_target_safe_fd':r['soft_z_safe_fd'],'v7_training_target_objective':r['soft_z_objective'],
   'v7_true_y_position':r['true_y_position'],'v7_true_y_safe_fd':r['true_y_safe_fd'],'v7_true_y_seam_safe_fd':r['true_y_seam_safe_fd'],'teacher_role':role,
   'teacher_train_gt_position':teacher.get('position_mae'),'teacher_train_gt_safe_fd':teacher.get('safe_fd_l1'),'teacher_train_gt_seam':teacher.get('seam_safe_fd_l1'),
   'student_teacher_distance_position':imi.get('imitation_position_mae'),'student_teacher_distance_safe_fd':imi.get('imitation_safe_fd_l1'),'student_teacher_distance_seam':imi.get('imitation_seam_safe_fd_l1'),
   'teacher_train_prediction_oof':False,'source_curve_path':str(curves),'source_curve_sha256':sha(curves),'source_imitation_path':str(imitation),'source_imitation_sha256':sha(imitation)})
 out=ROOT/'results/V7_TRAIN_Y_Z_AND_IMITATION.csv'; out.parent.mkdir(parents=True,exist_ok=True)
 if records:
  with out.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 # Preserve the historical per-epoch Y/Z curve as a separate train-only source artifact.
 with (ROOT/'results/V7_TRAINING_CURVES.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(curve_rows[0]));w.writeheader();w.writerows(curve_rows)
 target_roles={'schema_version':1,'status':'PASS','created_utc':utc(),'branch':'pending_zero_optimizer_audit','tasks':list(TASKS),'seeds':list(SEEDS),'input_mode':'TS11-D','targets':target_rows,
  'fixed_teacher_sources':'corresponding seed V_MATCHED and S_OPT outer-train in-sample predictions from sealed V7 source replays','target_native_space_mixture_before_shared_Y_fitted_normalizer':True,
  'normalizer_refit_on_Z':False,'teacher_parameter_updates':0,'teacher_train_predictions_are_oof':False,'dev_target_rows':0,'final_heldout_payload_access':0}
 write_json(ROOT/'protocol/TARGET_ROLE_MAP.json',target_roles)
 print(json.dumps({'status':'PASS','root':str(ROOT),'target_banks':len(target_rows),'train_metric_rows':len(records),'curve_rows':len(curve_rows),'old_roots_written':False,'dev_target_content_opened':False,'loss_helper_hashes_match':source_map['shared_loss_source_exact_match_V4_V7_V8']},indent=2))
if __name__=='__main__': main()
