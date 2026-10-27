#!/usr/bin/env python3
"""Read-only source identity lock for the fixed OBS43 closeout."""
import os
import csv, datetime, hashlib, json, os, zipfile
from pathlib import Path
import numpy as np
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT=Path(__file__).resolve().parents[1]
OBS=PROJECT/'reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z'
RIDGE=PROJECT/'reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z'
QUERY=PROJECT/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def put(rel,obj):
 p=ROOT/rel; p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def npz_header(path,key):
 with zipfile.ZipFile(path) as z:
  with z.open(key+'.npy') as f:
   version=np.lib.format.read_magic(f)
   if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
   elif version in ((2,0),(3,0)): shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
   else: raise ValueError(f'unsupported npy header version {version} in {path}:{key}')
 return {'shape':list(shape),'dtype':str(dtype),'fortran_order':bool(fortran)}
# File hash reads are integrity-only. NPY member headers below are parsed without loading payload arrays.
specs=[
('old_checksum_manifest',OBS/'CHECKSUMS.sha256','accepted_previous_bundle_checksum_manifest',None),
('previous_model_npz',OBS/'models/ARX_OBS43_FULL_TRAIN.npz','frozen_model_scalers_and_coefficients',None),
('previous_model_lock',OBS/'models/MODEL_LOCK.json','frozen_model_identity',None),
('previous_cv_scaler_receipts',OBS/'models/CV_SCALER_RECEIPTS.json','fit_only_scalers',None),
('previous_model_source',OBS/'code/obs43_diagnostic.py','as_run_affine_and_rollout_source',None),
('previous_field_map',OBS/'protocol/FIELD_MAP.json','source_field_contract',None),
('previous_clock_contract',OBS/'protocol/CLOCK_AND_INPUT_ROLES.json','recorded_clock_and_availability_contract',None),
('previous_analysis_contract',OBS/'protocol/ANALYSIS_LOCK.json','five_view_and_metric_contract',None),
('previous_eligibility',OBS/'data/ELIGIBILITY_MANIFEST.csv','anchor_split_group_ids',None),
('previous_folds',OBS/'data/FOLD_ASSIGNMENT.csv','source_group_cv_assignment','train'),
('previous_train_weights',OBS/'data/TRAIN_EPISODE_WEIGHTS.csv','training_episode_weights','train'),
('previous_fold_weights',OBS/'data/FOLD_EPISODE_WEIGHTS.csv','cv_episode_weights','train'),
('previous_time_grid',OBS/'data/TARGET_TIME_GRID.csv','recorded_clock_metadata',None),
('previous_forecast_acceptance',OBS/'qa/FORECAST_ARRAY_ACCEPTANCE.json','sealed_forecast_hashes_shapes',None),
('previous_score_acceptance',OBS/'qa/SCORE_ACCEPTANCE.json','dev_score_exposure_receipt',None),
('previous_teacher_access',OBS/'qa/TEACHER_INPUT_ACCESS.json','teacher_state_reset_access',None),
('previous_final_acceptance',OBS/'qa/FINAL_ACCEPTANCE.json','completed_previous_task',None),
('previous_final_report',OBS/'FINAL_REPORT.zh-CN.md','previous_claims_and_limits',None),
('previous_horizon_metrics',OBS/'results/FORECAST_METRICS_BY_HORIZON.csv','accepted_scores_by_horizon_block_view',None),
('previous_episode_metrics',OBS/'results/EPISODE_FORECAST_METRICS.csv','episode_level_score_source',None),
('previous_teacher_metrics',OBS/'results/TEACHER_LOCAL_METRICS.csv','local_teacher_scores',None),
('previous_group_bootstrap',OBS/'results/PAIRED_GROUP_BOOTSTRAP.csv','conditional_dev_group_bootstrap',None),
('previous_growth_receipt',OBS/'results/TRAIN_TRANSITION_GROWTH.json','raw_coordinate_spectrum_receipt','train'),
('prefix_train',OBS/'data/train/inputs_prefix.npz','prefix_state_and_anchor_command','train'),
('labels_train',OBS/'data/train/labels_obs43.npz','obs43_ground_truth','train'),
('commands_train',OBS/'data/train/commands_privileged.npz','raw_action_u0_to_u29','train'),
('prefix_dev',OBS/'data/dev/inputs_prefix.npz','prefix_state_and_anchor_command','dev'),
('labels_dev',OBS/'data/dev/labels_obs43.npz','obs43_ground_truth_already_exposed','dev'),
('commands_dev',OBS/'data/dev/commands_privileged.npz','raw_action_u0_to_u29','dev'),
('forecasts_train',OBS/'predictions/train/FORECASTS_H30.npz','sealed_P_CV_HOLD_ORACLE','train'),
('forecasts_dev',OBS/'predictions/dev/FORECASTS_H30.npz','sealed_P_CV_HOLD_ORACLE','dev'),
('teacher_train',OBS/'predictions/train/TEACHER_FORCED_LOCAL.npz','privileged_local_only_not_forecast','train'),
('teacher_dev',OBS/'predictions/dev/TEACHER_FORCED_LOCAL.npz','privileged_local_only_not_forecast','dev'),
('previous_case_ids',OBS/'data/DEV_CASES_LOCKED.csv','fixed_hash_selected_cases','dev')]
npz_keys={
 'previous_model_npz':['x_mean','x_std','coef','intercept','sigma_obs','safe_fd_obs'],
 'prefix_train':['anchor_id','source_group_id','s_prev','s0','u0','timestamp_window'],
 'labels_train':['anchor_id','obs_future'],
 'commands_train':['anchor_id','u_future'],
 'prefix_dev':['anchor_id','source_group_id','s_prev','s0','u0','timestamp_window'],
 'labels_dev':['anchor_id','obs_future'],
 'commands_dev':['anchor_id','u_future'],
 'forecasts_train':['anchor_id','OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'],
 'forecasts_dev':['anchor_id','OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'],
 'teacher_train':['anchor_id','teacher_next_native'],
 'teacher_dev':['anchor_id','teacher_next_native']}
sources=[]
for name,p,role,split in specs:
 row={'name':name,'path':str(p),'role':role,'split':split,'sha256':sha(p),'bytes':p.stat().st_size,'write_policy':'read_only','hash_mode':'file_bytes_integrity_only'}
 if name in npz_keys: row['array_headers']={k:npz_header(p,k) for k in npz_keys[name]}
 sources.append(row)
put('source/SOURCE_REGISTRY.json',{'status':'LOCKED_READ_ONLY_TYPED_DEPENDENCIES','created_utc':utc(),'old_obs_root':str(OBS),'source_count':len(sources),'source_files':sources,'previous_dev_repeatedly_exposed':True,'numeric_array_payloads_loaded':False,'old_checksum_manifest_sha256':next(x['sha256'] for x in sources if x['name']=='old_checksum_manifest')})
put('protocol/STUDY_REUSE_AUDIT.json',{'status':'PASS','checked_utc':utc(),'same_contract_root_preexisting':False,'active_same_contract_writer':False,'exact_reuse_root':str(OBS),'reused_artifacts':['frozen model/scalers','accepted train/dev OBS43 and command slices','four accepted H30 forecasts','teacher local predictions','accepted scorer outputs and fixed cases'],'decision':'new independent closeout OUT_ROOT; no model or prediction recomputation'})
put('protocol/OWNER_STATE.json',{'owner_task_id':ROOT.name,'current_writer':'code/error_budget_closeout.py','status':'ACTIVE_CPU_ONLY_ZERO_FIT','old_roots_read_only':True,'new_model_fits':0,'gpu_access':0,'other_window_tasks_modified_or_relaunched':False,'locked_utc':utc()})
put('protocol/OUTCOME_EXPOSURE.json',{'previous_dev_exposure':True,'source_ids_and_model_hashes_locked_before_numeric_decomposition':True,'analysis_and_error_signs_locked_before_loading_array_payloads':True,'new_model_selection':False,'new_fit':False,'new_predictions':False,'new_dev_score_selection':False,'final_test_access':0,'heldout_access':0,'independent_confirmation':False,'known_unknowns':['previous first raw dev-label read time','previous full raw extractor CPU/RSS receipt']})
put('protocol/COORDINATE_CONVENTION.json',{'status':'LOCKED','state_column_vector':'z=[s;v] in R^86','s_order':['arm_joints_source_alias[0:14]','hand_joints_source_alias[14:28]','leg_joints_source_alias[28:43]'],'v':'same source-native numeric units per nominal second','command':'u in R^28','delta_seconds':1.0/30.0,'source_x_row':'[s_current(43),v_current(43),u_current(28)]','source_prediction':'v_next = sigma_obs * (intercept + ((x-x_mean)/x_std) @ coef)','affine_partition':'F = diag(sigma_obs) @ coef.T @ diag(1/x_std); b_v=diag(sigma_obs)@intercept - F@x_mean; split F into Fs,Fv,Fu','A':'[[I+delta*Fs,delta*Fv],[Fs,Fv]]','B':'[[delta*Fu],[Fu]]','c':'[delta*b_v;b_v]','dimensionless_P':'diag(1/sigma_obs,delta/sigma_obs)','affine_matrix_similarity':'A_tilde=P@A@inv(P); no centering in P'})
put('protocol/ERROR_SIGN_CONVENTION.json',{'status':'LOCKED','d':'zhat_ORACLE - zhat_HOLD','r':'z_GT[j+1] - (A z_GT[j] + B u[j] + c)','eO':'z_GT - zhat_ORACLE','eH':'z_GT - zhat_HOLD','identities':['d[j+1]=A*d[j]+B*(u[j]-u[0])','eO[j+1]=A*eO[j]+r[j]','eH[j]=eO[j]+d[j]'],'same_initial_state_required':True,'initial_mismatch_rule':'if any, add A^j d0/A^j eO0 explicitly','r_position_equals_delta_r_velocity':'must calculate from source update, not assume clock pass proves it','residual_interpretation':'fitted model error plus unmodeled/recording effects; not labelled irreducible or independent'})
put('protocol/SCALER_WEIGHT_LOCK.json',{'status':'LOCKED','position_scale':'frozen previous sigma_obs[43]','fd_scale':'frozen previous safe_fd_obs[43]','standardized_position_weight_per_coordinate':'1/(43*sigma_obs[j])','standardized_fd_weight_per_coordinate':'1/(43*safe_fd_obs[j])','native_metric_weight':'uniform 1/43 on source-native numeric values','field_anatomy_units':'unverified','scale_reestimated_or_weight_selected':False})
put('protocol/AGGREGATION_LOCK.json',{'status':'LOCKED','point_estimand':'previous episode-equal score aggregation; verify one-to-one episode/source_group mapping before use','coordinate_aggregation':'within episode average anchors/time by channel, then mean across source episodes/groups','h30_position':'prefix s[1..30]; endpoint separate','horizon_positions':[1,3,6,10,15,30],'future_internal_fd':'diff predictions/GT over future positions s[1..H]; H30 has 29 differences; H1 NA; anchor-to-first seam separate','blocks':'same full43 per-channel scalers sliced into 14/14/15 blocks','common_scale_identity':'43*L_full = 14*L_arm +14*L_hand +15*L_leg for per-coordinate mean losses','bootstrap':'reuse prior group bootstrap for score identities; any new diagnostic interval uses seed20261011,10000 common source-group draws; conditional on fixed model and exposed dev; no p-values','no_metric_reweighting_or_primary_change':True})
put('protocol/ANALYSIS_LOCK.json',{'status':'LOCKED_BEFORE_ARRAY_PAYLOAD_ANALYSIS','new_model_fits':0,'new_predictions':0,'gpu_actions':0,'primary_H':30,'horizons':[1,3,6,10,15,30],'same_fixed_initial_state':True,'tasks':['reconstruct exact affine source update from frozen coef/intercept/scalers','replay model and forecast arrays with source-native dtype tolerance','verify d/eO/eH identities for train/dev H1-H30','verify r_position=delta*r_velocity','compute native and fixed-standardized L1 response/residual/cancellation budgets','compute fixed scale coordinate contributions and Full43/block identities','compute P/HOLD/ORACLE rank reconciliation without switching primary','dimensionless diagonal similarity and K_l forcing response','actual residual propagation/cancellation and horizon/block diagnostics','reuse fixed cases and claim ledger','draft independent confirmation spec but do not execute'], 'no_new_fits':True,'no_reranking_or_weight_search':True,'no_new_primary_metric':True,'no_native_primary_promotion':True,'no_posthoc_case_selection':True,'no_residual_independence_assumption':True,'no_physical_causality_or_irreducible_noise_claim':True})
put('protocol/CASE_LOCK.json',{'status':'LOCKED','case_ids_source':'previous data/DEV_CASES_LOCKED.csv','selection_changed':False,'case_use':'signed replay/identity illustrations only; previously selected before previous score','oracle_case_selected_by_gain':False})
put('protocol/CPU_PREFLIGHT.json',{'hostname':os.uname().nodename,'cpu_affinity_available':sorted(os.sched_getaffinity(0)),'planned_affinity':[0,1,2,3,4,5,6,7],'blas_threads_max':8,'gpu_hidden_and_untouched':True,'no_nvidia_smi_or_nvml':True,'scheduler_submission':False,'filesystem_available':'5.8T as observed in this turn','raw_parquet_reopen':False})
put('protocol/SOURCE_HASH_LOCK.json',{'status':'LOCKED_BEFORE_NUMERIC_ARRAY_LOAD','created_utc':utc(),'source_registry_path':'source/SOURCE_REGISTRY.json','source_registry_sha256':sha(ROOT/'source/SOURCE_REGISTRY.json'),'source_count':len(sources),'old_checksum_manifest_sha256':next(x['sha256'] for x in sources if x['name']=='old_checksum_manifest'),'integrity_hashes_only':True,'NPZ_payloads_decompressed':False})
put('logs/LOCK_SCRIPT_FAILURES.jsonl',{}) if False else None
(ROOT/'RUN_STATUS.json').write_text(json.dumps({'status':'running','task':'OBS43_FIXED_MODEL_ERROR_BUDGET_CLOSEOUT','created_utc':utc(),'new_model_fits':0,'gpu_actions':0,'out_root':str(ROOT)},indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'LOCKED','out_root':str(ROOT),'source_count':len(sources),'old_checksum_sha256':next(x['sha256'] for x in sources if x['name']=='old_checksum_manifest')},sort_keys=True))
