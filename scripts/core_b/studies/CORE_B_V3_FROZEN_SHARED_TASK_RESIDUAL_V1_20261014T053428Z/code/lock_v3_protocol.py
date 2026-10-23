#!/usr/bin/env python3
"""Lock V3 estimands and deterministic RGB donor plan before dev scoring."""
from __future__ import annotations
import os
import csv,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
ROOT=Path(__file__).resolve().parents[1]
TASKS=(119,94,90,141,230)
SEEDS=(202610131,202610132,202610133)

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def write_json(rel,obj):
 p=ROOT/rel
 p.parent.mkdir(parents=True,exist_ok=True)
 content=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if p.exists():
  old=json.loads(p.read_text(encoding='utf-8'));new=json.loads(content)
  old.pop('created_utc',None);new.pop('created_utc',None)
  if old!=new:raise RuntimeError(f'preserving differing existing protocol lock: {p}')
  return
 p.write_text(content,encoding='utf-8')
def write_csv(rel,rows):
 p=ROOT/rel
 p.parent.mkdir(parents=True,exist_ok=True)
 import io
 b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=list(rows[0]) if rows else [])
 w.writeheader();w.writerows(rows);content=b.getvalue()
 if p.exists():
  if p.read_bytes().decode('utf-8')!=content:raise RuntimeError(f'preserving differing existing donor map: {p}')
  return
 p.write_text(content,encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

v2_map=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text())
dev_path=V2/'data/DEV_PREFIXES_NO_LABELS.npz'
with np.load(dev_path,allow_pickle=False) as a:
 if 'target_native_h30' in a.files: raise RuntimeError('dev prefix archive is not label-free')
 prefix={k:a[k].copy() for k in ('anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame')}
if len(prefix['anchor_id'])!=185:raise RuntimeError('V2 dev prefix support changed')
# Build metadata-only deterministic donor mapping; no cache values or targets are read here.
by_task_episode={}
for i,(task,episode) in enumerate(zip(prefix['task_id'],prefix['episode_id'])):
 by_task_episode.setdefault(int(task),{}).setdefault(int(episode),[]).append(i)
rows=[]
for i,receiver in enumerate(prefix['anchor_id'].tolist()):
 task=int(prefix['task_id'][i]);recv_ep=int(prefix['episode_id'][i])
 candidates=[]
 for donor_ep,indices in by_task_episode[task].items():
  if donor_ep==recv_ep:continue
  donor_i=min(indices,key=lambda j:(int(prefix['anchor_frame'][j]),str(prefix['anchor_id'][j])))
  donor_id=str(prefix['episode_id'][donor_i])
  order=hashlib.sha256(('v3donor|'+str(receiver)+'|'+donor_id).encode('utf-8')).hexdigest()
  candidates.append((order,donor_i,donor_ep,donor_id))
 if not candidates:
  rows.append({'receiver_anchor_id':str(receiver),'receiver_episode_id':recv_ep,'task_id':task,'donor_status':'NA_NO_OTHER_EPISODE',
   'donor_episode_id':'','donor_anchor_id':'','donor_anchor_frame':'','selection_sha256':''})
 else:
  order,donor_i,donor_ep,donor_id=sorted(candidates,key=lambda x:(x[0],x[3]))[0]
  rows.append({'receiver_anchor_id':str(receiver),'receiver_episode_id':recv_ep,'task_id':task,'donor_status':'PASS',
   'donor_episode_id':donor_ep,'donor_anchor_id':str(prefix['anchor_id'][donor_i]),
   'donor_anchor_frame':int(prefix['anchor_frame'][donor_i]),'selection_sha256':order})
if any(r['donor_status']!='PASS' for r in rows):
 missing=[r for r in rows if r['donor_status']!='PASS']
 # Donor NA is allowed by protocol; ensure every diagnostic task has a valid donor.
 if {int(r['task_id']) for r in missing} & {119,94,90,141,230}:
  raise RuntimeError('RGB probe tasks unexpectedly lack a different-episode donor')
write_csv('diagnostics/RGB_DONOR_MAP.csv',rows)

# Keep a metadata copy for later scorer/figure support; no dev future values.
dev_support=[]
with (V2/'inventory/DEV_SUPPORT_V2.csv').open(newline='',encoding='utf-8') as f:dev_support=list(csv.DictReader(f))
# Protocol: fixed stage and model mechanics.
readout={
 'schema_version':1,'status':'LOCKED_BEFORE_V3_SCREENING','created_utc':now(),'study_mode':'post_V2_exploratory_frozen_readout',
 'base_training_updates':0,'encoder_training':0,'encoder_extraction':0,'V2_78_fits_rerun':0,
 'base_sources':{'screening':v2_map['screening_bases'],'formal':v2_map['formal_bases'],'rgb_control':v2_map['probe_control_bases']},
 'feature_encoder':{'frozen':True,'new_raw_video_extraction':0,'accepted_q0_shape':[2,16,4,768],
   'hidden_extraction_method':'forward pre-hook on V2TaskConditionedPredictor.output; no target args; model.eval; requires_grad=False'},
 'hidden_interface':{'source_module':'V2TaskConditionedPredictor.output','tensor':'input activation immediately before the accepted final output Linear',
   'shape_per_anchor':[30,128],'width':128,'horizon':30,'output_dim':28,'task_id_context_already_in_h':True},
 'base_output_coordinates':{'z_base':'V2 action_residual in stage target_sigma-normalized residual coordinates',
   'native_prediction':'anchor_native + stage_target_sigma * (z_base + R(h))',
   'base_bias_native':'base_native + train_only_task_axis_bias'},
 'programs':['SHARED_RES_STATE','TASK_RES_STATE','SHARED_RES_VIDEO','TASK_RES_VIDEO'],
 'readout':{'shared':'one affine Linear(128,28) applied at each future step',
   'task':'five affine Linear(128,28) heads selected by existing task rank; no new task encoder',
   'zero_initialization':True,'formal_reinitialized_from_zero_each_seed':True,'base_state_hash_before_after_equal_required':True,
   'base_eval_mode_required':True,'base_parameters_requires_grad_false_required':True,'base_buffers_unchanged_required':True,
   'shared_parameter_count_total':128*28+28,'task_parameter_count_total':5*(128*28+28),
   'task_parameter_count_active_per_sample':128*28+28},
 'objective':{'action_loss_source_path':str(V2/'code/locked_source/d3_eq_contract.py'),
   'action_loss_source_sha256':sha(V2/'code/locked_source/d3_eq_contract.py'),
   'position':'0.5 * mean_abs_error(first 14 axes) + 0.5 * mean_abs_error(last 14 axes), each divided by stage target_sigma',
   'safe_fd':'0.5 * future-internal 29-difference L1 over first 14 axes + 0.5 over last 14, divided by stage safe_fd_scale',
   'action_objective':'position + 0.1 * safe_fd_l1','seam_in_training_loss':False,
   'readout_penalty':'lambda * per-sample mean over (future_time,28 axes) of R(h)^2',
   'penalty_lambda_candidates':[0.0,1.0],'lambda_units':'stage-normalized correction squared',
   'batch_weight':'w_i = W/(5 * fit_episodes_for_task * fit_windows_in_episode); sum_i(w_i * loss_i)/B_actual',
   'sampling':'V2 locked shuffled_epoch_order; same epoch orders across programs/lambda for a seed'},
 'optimizer':{'name':'AdamW','lr':0.001,'weight_decay':0.0,'beta1':0.9,'beta2':0.999,'eps':1e-8,
   'scheduler':'constant LR; none','effective_batch':64},
 'screening':{'bases':'V2 corrected SCREENING G-S/G-V at selected V2 LR/epoch, seed 202610130',
   'train_role':'inner_fit only','tune_role':'inner_tune for selection only','epochs':80,'checkpoints':[20,40,80],
   'programs':4,'lambda_values':[0.0,1.0],'planned_neural_fits':8,
   'selection_metric':'five-task macro inner-tune J = canonical position_mae + 0.1 * canonical safe_fd_l1',
   'tie_rule':'minimum J; differences <=1e-12: shorter epoch, then larger lambda',
   'tune_support_gate':False,'positive_gain_gate':False},
 'formal':{'bases':'V2 corrected outer-train G-S/G-V matching each formal seed',
   'train_role':'full outer train, all 220 episodes/1280 anchors','screening_readout_weights_transferred':False,
   'selection':'one lambda/epoch per program, shared across all five tasks and formal seeds',
   'fresh_zero_initialization_per_fit':True,'formal_seeds':list(SEEDS),'programs':4,'planned_neural_fits':12},
 'future_labels_in_forward':False,'dev_used_for_lambda_epoch_selection':False,'final_heldout_payload_access':0}
write_json('protocol/MODEL_READOUT_LOCK.json',readout)
write_json('protocol/MATRIX.json',{
 'schema_version':1,'status':'LOCKED_BEFORE_V3_SCREENING','created_utc':now(),'tasks':list(TASKS),'formal_seeds':list(SEEDS),
 'screen_neural_fits_planned':8,'formal_neural_fits_planned':12,'neural_fits_planned':20,
 'cpu_train_only_bias_calibrations_planned':6,'V2_78_fits_rerun':0,
 'screening_programs':readout['programs'],'lambda_candidates':[0.0,1.0],'screen_epochs':80,'screen_checkpoints':[20,40,80],
 'formal_program_seed_cells':[{'program':p,'seed':s} for p in readout['programs'] for s in SEEDS],
 'formal_readout_jobs_planned':12,'bias_jobs_planned':6,
 'main_new_prediction_jobs_planned':18,'main_task_method_seed_views_max':90,
 'old_v2_base_jobs_reused':6,'old_v2_base_task_views':30,
 'rgb_diagnostic_jobs_planned':12,'rgb_diagnostic_task_views_max':36,
 'rgb_matched_old_v2_task_views':18,'extra_matched_replay_forwards':'formal G-V h reused from base extraction; S-V-90 matched replay in its diagnostic condition jobs and itemized separately',
 'new_prediction_jobs_total_planned':30,'new_task_method_seed_views_max':126,
 'bias_predictions_are_neural_fits':False,'sample_or_positive_gain_gate':False,
 'no_dev_selection_of_lambda_epoch_or_task':True,'final_heldout_payload_access':0})
write_json('protocol/ANALYSIS_LOCK.json',{
 'schema_version':1,'status':'LOCKED_BEFORE_V3_SCREENING','created_utc':now(),'metrics':['position_mae','safe_fd_l1','seam_safe_fd_l1'],
 'metrics_source':'V2 accepted canonical METRIC_SCALE_LOCK and scorer definitions',
 'aggregation':'window errors within episode averaged; episode-equal task means; five-task macro is equal task mean',
 'contrasts':{'G_R_shared':'L(BASE_m)-L(SHARED_RES_m)','G_R_task':'L(BASE_m)-L(TASK_RES_m)',
  'T_R':'L(SHARED_RES_m)-L(TASK_RES_m)','G_BIAS':'L(BASE_m)-L(BIAS_m)',
  'R_vs_BIAS':'L(BIAS_m)-L(residual_m)','V_program':'L(program_STATE)-L(program_VIDEO)'},
 'positive_means_latter_named_method_has_lower_error':True,'MAE_and_safe_FD_reported_separately':True,
 'per_seed_points_and_mean_sampleSD_median':True,'seed_LOO':'exclude one trained seed; average remaining two; task set remains five',
 'task_LOO':'exclude one task per seed; four tasks retained; all three trained seeds retained',
 'bootstrap':{'method':'paired source-group bootstrap, resample source clusters with replacement within each task; all episodes in selected cluster retained',
   'replicates':5000,'seed':202610134,'draws_shared_across':'all methods, formal seeds and metrics',
   'interpretation':'conditional on trained seeds and prior tuning; session/trial independence unverified; no p-values or confirmation claim'},
 'bootstrap_draw_manifest':'protocol/SOURCE_GROUP_DRAWS.npz plus protocol/SOURCE_GROUP_DRAWS.json',
 'prediction_seal_gate':'nonempty accepted manifest IDs, finite arrays, shape and ordered IDs verified; only then may V3 scorer open scorer-only dev target',
 'final_heldout_payload_access':0})
write_json('diagnostics/RGB_PROBE_PLAN.json',{
 'schema_version':1,'status':'LOCKED_BEFORE_PROBE_PREDICTIONS','created_utc':now(),'bases':v2_map['formal_bases'] + v2_map['probe_control_bases'],
 'conditions':['MATCHED','ZERO_NORMALIZED_BLOCK','SAME_TASK_DONOR'],
 'new_prediction_conditions':['ZERO_NORMALIZED_BLOCK','SAME_TASK_DONOR'],
 'zero_condition':'set both normalized dynamic/static Q0 cache blocks to exactly zero at the accepted model input interface before V2 video projection; not black video and not raw cache-feature zero',
 'same_task_donor':'replace both full normalized dynamic/static prefix blocks only; preserve state, anchor, task id, history masks and timing',
 'donor_selection':'for each receiver, choose the different-episode same-task candidate with minimum sha256(v3donor|receiver_anchor_id|donor_episode_id); donor row is earliest anchor_frame then anchor_id',
 'donor_map_path':'diagnostics/RGB_DONOR_MAP.csv','donor_map_sha256':sha(ROOT/'diagnostics/RGB_DONOR_MAP.csv'),
 'matched_reference':'V2 accepted G-V/S-V-90 prediction IDs and prediction files; new matched replay costs itemized as hidden/QA forward, not new prediction rows',
 'models':['G-V','S-V-90'],'formal_seeds':list(SEEDS),'new_prediction_jobs_planned':12,
 'task_views_by_condition':{'G-V':15,'S-V-90':3},'task_views_per_new_condition':18,
 'total_new_condition_task_views_max':36,'old_matched_task_views_reused':18,
 'dev_targets_loaded_for_probe':False,'probe_used_for_selection':False,
 'interpretation_limit':'input sensitivity is not visual utility or causal attribution; zero/donor inputs may be out of distribution; hidden coordinates are compared only within the same checkpoint',
 'final_heldout_payload_access':0})
# Persist the exact per-task dev supports used by the locked outputs.
support_src=V2/'inventory/DEV_SUPPORT_V2.csv';support_dst=ROOT/'inventory/DEV_SUPPORT_V2.csv'
if support_dst.exists():
 if support_dst.read_bytes()!=support_src.read_bytes():raise RuntimeError('preserving differing V2 dev support copy')
else:shutil.copyfile(support_src,support_dst)
print(json.dumps({'status':'PASS','v3_root':str(ROOT),'donor_rows':len(rows),'donor_status_counts':dict(__import__('collections').Counter(x['donor_status'] for x in rows)),
 'readout_params_shared':3612,'readout_params_task_total':18060,'planned_new_prediction_jobs':30,'v3_dev_targets_read':0,'final_heldout_payload_access':0},indent=2))
