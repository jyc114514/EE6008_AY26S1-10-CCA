#!/usr/bin/env python3
"""Seal V8 branch-B target/sampler/recipe locks before formal training."""
from __future__ import annotations
import os
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
SEEDS=(202610131,202610132,202610133); TASKS=(119,94,90,141,230); PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT')
ARM_ORDER={202610131:('GT_RETRAIN','SELF_SOFT','SV_SOFT'),202610132:('SELF_SOFT','SV_SOFT','GT_RETRAIN'),202610133:('SV_SOFT','GT_RETRAIN','SELF_SOFT')}
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def writej(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def main():
 branch=readj(ROOT/'protocol/BRANCH_LOCK.json')
 if branch.get('branch')!='B_SYNCHRONOUS_VIDEO_9' or branch.get('branch_selected_from_new_dev_outcomes') is not False:raise RuntimeError('branch lock is missing/invalid')
 data_path=V2/'data/TRAIN_SELECTED_H30.npz'
 with np.load(data_path,allow_pickle=False) as z:n=len(z['anchor_id'])
 if n!=1280:raise RuntimeError(f'accepted train rows differ from 1280: {n}')
 rows_by_seed={}
 for seed in SEEDS:
  epoch_orders=[]
  for epoch in range(1,161):
   order=np.arange(n,dtype=np.int64);np.random.default_rng(seed+epoch).shuffle(order);epoch_orders.append(order)
  rows_by_seed[f'seed_{seed}']=np.stack(epoch_orders,axis=0)
 sampler_path=ROOT/'protocol/SAMPLER_ORDERS.npz'
 tmp=sampler_path.with_name(f'.{sampler_path.name}.{os.getpid()}.part')
 with tmp.open('wb') as f:np.savez(f,**rows_by_seed);f.flush();os.fsync(f.fileno())
 os.replace(tmp,sampler_path)
 per_seed={k:hashlib.sha256(np.ascontiguousarray(v).tobytes(order='C')).hexdigest() for k,v in rows_by_seed.items()}
 sampler={'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS','created_utc':utc(),'path':str(sampler_path),'sha256':sha(sampler_path),
  'shape_per_seed':[160,n],'dtype':'int64','row_domain':'accepted TRAIN_SELECTED_H30.npz row indices','epochs_one_based':True,
  'generation':'order=np.arange(1280,dtype=int64); np.random.default_rng(seed+epoch).shuffle(order)','per_seed_sequence_sha256':per_seed,
  'optimizer_steps_per_epoch':20,'effective_batch':64,'last_group_rows':64,'target_independent':True}
 writej(ROOT/'protocol/SAMPLER_SEQUENCE_MANIFEST.json',sampler)
 role=readj(ROOT/'protocol/TARGET_ROLE_MAP.json')
 mix={'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS','created_utc':utc(),'branch':'B_SYNCHRONOUS_VIDEO_9','task_ids':list(TASKS),'seeds':list(SEEDS),
  'programs':{'GT_RETRAIN':{'target_mode':'hard_y','target_expression':'Z=Y','beta':0.0,'source_roles':['Y']},
   'SELF_SOFT':{'target_mode':'fixed_soft_z','target_expression':'Z=0.5*Y+0.5*V','beta':0.5,'source_roles':['Y','V_MATCHED']},
   'SV_SOFT':{'target_mode':'fixed_soft_z','target_expression':'Z=0.5*Y+0.25*S+0.25*V','beta':0.5,'source_roles':['Y','S_OPT','V_MATCHED']}},
  'mix_space':'native numeric command values before the single fixed target transform','normalizer_source':'accepted outer-train real Y','normalizer_path':str(V7/'normalizers/NORMALIZER_FORMAL.json'),
  'normalizer_sha256':sha(V7/'normalizers/NORMALIZER_FORMAL.json'),'normalizer_refit_on_Z':False,'teacher_parameter_updates':0,
  'teacher_train_predictions_are_oof':False,'dev_target_rows':0,'final_heldout_payload_access':0,'target_role_map_sha256':sha(ROOT/'protocol/TARGET_ROLE_MAP.json'),
  'target_banks':[{'program_id':x['program_id'],'seed':x['seed'],'target_sha256':x['target_sha256'],'manifest_sha256':x['manifest_sha256'],'rows_sha256':x['rows_sha256']} for x in role['targets']]}
 writej(ROOT/'protocol/TARGET_MIX_LOCK.json',mix)
 source_role=readj(V7/'protocol/TEACHER_ROLE_MAP.json');inits={}
 for seed in SEEDS:
  init=source_role['sources'][str(seed)]['V_MATCHED']
  if init['init_sha256']!=sha(init['init_path']):raise RuntimeError(f'fresh INIT file hash mismatch {seed}')
  inits[str(seed)]={'path':init['init_path'],'file_sha256':init['init_sha256'],'state_hash':init['init_state_hash'],'source_role':'accepted fresh five-task INIT; not a trained teacher'}
 stage0=readj(ROOT/'coordination/STAGE0_RESOURCE_RECEIPT.json');gpu=readj(ROOT/'audit/GPU_ZERO_OPTIMIZER_AUDIT.json')
 train_code=ROOT/'code/v8_train.py'
 recipe={'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV','created_utc':utc(),'study':'post_V7_controlled_runner_and_target_replay_development',
  'branch':'B_SYNCHRONOUS_VIDEO_9','branch_lock_sha256':sha(ROOT/'protocol/BRANCH_LOCK.json'),'target_mix_lock_sha256':sha(ROOT/'protocol/TARGET_MIX_LOCK.json'),
  'target_role_map_sha256':sha(ROOT/'protocol/TARGET_ROLE_MAP.json'),'train_npz_path':str(data_path),'train_npz_sha256':sha(data_path),
  'train_prefix_path':str(V7/'data/TRAIN_PREFIXES_NO_LABELS.npz'),'train_prefix_sha256':sha(V7/'data/TRAIN_PREFIXES_NO_LABELS.npz'),
  'q0_cache_manifest_path':readj(ROOT/'inventory/RESOLVED_PATHS.json')['feature_cache'],'q0_cache_manifest_sha256':readj(ROOT/'inventory/RESOLVED_PATHS.json')['feature_cache_manifest_sha256'],
  'normalizer_path':str(V7/'normalizers/NORMALIZER_FORMAL.json'),'normalizer_sha256':sha(V7/'normalizers/NORMALIZER_FORMAL.json'),
  'metric_scale_lock_path':str(ROOT/'protocol/METRIC_SCALE_LOCK.json'),'metric_scale_lock_sha256':sha(ROOT/'protocol/METRIC_SCALE_LOCK.json'),
  'split_path':str(V2/'protocol/INNER_SPLIT.csv'),'split_sha256':sha(V2/'protocol/INNER_SPLIT.csv'),'tasks':list(TASKS),'seeds':list(SEEDS),
  'programs':list(PROGRAMS),'seed_arm_order':{str(k):list(v) for k,v in ARM_ORDER.items()},'input_mode':'TS11-D','waypoint_mode':None,'horizon':30,'target_dim':28,
  'model':'V2TaskConditionedPredictor Q0; same 891548 parameter architecture; no new layer','fresh_init_by_seed':inits,
  'optimizer':{'name':'AdamW','lr':0.0005,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':1e-4},
  'schedule':{'name':'update-driven cosine','T_max':3200,'eta_min':1e-5,'step_location':'optimizer post-step hook','steps_per_epoch':20},
  'training':{'epochs':160,'checkpoint':'fixed last epoch 160','effective_batch':64,'microbatch_size':64,'accumulation_steps':1,'dtype':'float32','amp':False,
    'gradient_clip_norm':1.0,'sample_weight':'W/(5*E_task*n_episode_windows)','batch_reduction':'sum(w_i*ell_i)/B_actual','sampler':'locked seed+epoch permutations in protocol/SAMPLER_ORDERS.npz',
    'sampler_manifest_sha256':sha(ROOT/'protocol/SAMPLER_SEQUENCE_MANIFEST.json'),'sampler_archive_sha256':sampler['sha256'],
    'torch_seed_per_fit':'same formal seed for all three arms; torch/cuda/numpy/python seed set before target/model loads','dropout_rng_target_independent':True},
  'runtime':{'torch_version':gpu['device']['torch_version'],'torch_cuda_version':gpu['device']['torch_cuda_version'],'cudnn_version':gpu['device']['cudnn_version'],
    'deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,'matmul_tf32':False,'cudnn_tf32':False,'autocast':False,
    'physical_gpu_pool':'dynamic_all_visible','watcher_concurrency_cap':2},
  'resource_estimate':{'measured_peak_mib':stage0['registered_measured_peak_mib'],'basis':'measured V8 Stage0 full replay + gradient + temporary optimizer clone peak reserved',
    'watcher_required_free_memory_mib':stage0['registered_measured_peak_mib']+512},
  'runner_path':str(train_code),'runner_sha256':sha(train_code),'model_source_sha256':sha(ROOT/'code/v2_model.py'),'loss_source_sha256':sha(ROOT/'code/locked_source/d3_eq_contract.py'),
  'scheduler_source_sha256':sha(ROOT/'code/v2_scheduler.py'),'target_builder_path':str(ROOT/'code/v8_prepare_targets.py'),
  'old_v7_soft_fits_reused_as_paired':0,'formal_fits_expected':9,'new_hyperparameter_search':0,'historical_mae_match_required':False,
  'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,'dev_target_archive_opened':False,'final_heldout_payload_access':0}
 writej(ROOT/'protocol/RECIPE_LOCK.json',recipe)
 qa={'schema_version':1,'status':'PASS','created_utc':utc(),'sampler_sequence_rows':len(SEEDS)*160*n,'unique_seed_permutations':len(SEEDS)*160,
  'all_permutations_bijective':True,'per_seed_sequence_sha256':per_seed,'sampler_archive_sha256':sampler['sha256'],
  'target_mix_lock_sha256':sha(ROOT/'protocol/TARGET_MIX_LOCK.json'),'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),
  'stage0_peak_reserved_mib':gpu['physical_gpu_memory_peak_reserved_mib'],'registered_peak_mib':stage0['registered_measured_peak_mib'],
  'dev_target_archive_opened':False,'formal_fits_started':0,'final_heldout_payload_access':0}
 writej(ROOT/'qa/RECIPE_LOCK_QA.json',qa)
 print(json.dumps({'status':'PASS','sampler_archive_sha256':sampler['sha256'],'sampler_manifest_sha256':sha(ROOT/'protocol/SAMPLER_SEQUENCE_MANIFEST.json'),
  'target_mix_lock_sha256':recipe['target_mix_lock_sha256'],'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),'formal_fits_expected':9},indent=2))
if __name__=='__main__':main()
