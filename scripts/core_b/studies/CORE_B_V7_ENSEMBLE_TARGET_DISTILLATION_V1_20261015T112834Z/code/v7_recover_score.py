#!/usr/bin/env python3
"""Resume post-seal results from saved V7 window losses without reopening dev targets."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V6=PROJECT/'reports/benchmarks/CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z'
PROGRAMS=('ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV');SEEDS=(202610131,202610132,202610133);TASKS=(119,94,90,141,230)
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read_json(p:Path)->Any:return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p:Path)->list[dict[str,str]]:
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_json(p:Path,obj:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)
def write_csv(p:Path,rows:list[dict[str,Any]])->None:
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main()->int:
 seal=read_json(ROOT/'qa/PREDICTION_SEAL.json')
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('accepted_dev_bundles')!=18 or seal.get('preseal_dev_target_reads')!=0:raise RuntimeError('score recovery requires the accepted nonempty pre-target prediction seal')
 if read_json(ROOT/'qa/SOURCE_SCORE_RECONCILIATION.json').get('status')!='PASS':raise RuntimeError('V6 source score reconciliation is not PASS')
 new_path=ROOT/'results/PER_WINDOW_NEW_DEV_LOSSES.csv';all_path=ROOT/'results/PER_WINDOW_ALL_METHODS.csv'
 new_saved=read_csv(new_path);all_saved=read_csv(all_path)
 if len(new_saved)!=3330 or len(all_saved)!=8880:raise RuntimeError(f'saved student loss coverage unexpected: {len(new_saved)}/{len(all_saved)}')
 # Verify that saved student rows are the 18x185 completed target scores from the one failed scorer process.
 methods={r['method'] for r in new_saved};seeds={int(r['seed']) for r in new_saved}
 if methods!=set(PROGRAMS) or seeds!=set(SEEDS):raise RuntimeError('saved V7 student loss methods/seeds are incomplete')
 per_slot={(p,s):sum(1 for r in new_saved if r['method']==p and int(r['seed'])==s) for p in PROGRAMS for s in SEEDS}
 if any(n!=185 for n in per_slot.values()):raise RuntimeError(f'saved student losses are not exactly 185 rows per slot: {per_slot}')
 spec=importlib.util.spec_from_file_location('v7_score_recovery_import',ROOT/'code/v7_score.py');score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
 resolved=read_json(ROOT/'inventory/RESOLVED_PATHS.json');metadata=score.read_prefix_metadata(Path(resolved['dev_prefix_npz']))
 all_rows=[dict(r,method=str(r['method']),seed=int(r['seed']),episode_id=int(r['episode_id']),task_id=int(r['task_id'])) for r in all_saved]
 primary,episode_abs=score.aggregate_windows(all_rows)
 # The absolute tables already written by the first score pass must reproduce exactly from saved window losses.
 old_primary=read_csv(ROOT/'results/PRIMARY_TASK_SEED.csv');oldmap={(r['method'],int(r['seed']),str(r['task_id'])):r for r in old_primary}
 for r in primary:
  key=(r['method'],int(r['seed']),str(r['task_id']));old=oldmap.get(key)
  if old is None:raise RuntimeError(f'saved primary row missing during score recovery: {key}')
  for metric in score.METRICS:
   if abs(float(old[metric])-float(r[metric]))>1e-12:raise RuntimeError(f'saved primary aggregation differs from per-window losses: {key}/{metric}')
 task_seed,pair_rows,seed_loo,task_loo,ci_rows,draw_manifest=score.paired(all_rows,episode_abs,metadata)
 write_csv(ROOT/'results/PAIRED_TASK_SEED.csv',task_seed);write_csv(ROOT/'results/PAIRED_EFFECTS.csv',pair_rows)
 write_csv(ROOT/'results/SEED_LOO.csv',seed_loo);write_csv(ROOT/'results/TASK_LOO.csv',task_loo);write_csv(ROOT/'results/PAIRED_GROUP_CI.csv',ci_rows)
 write_json(ROOT/'qa/BOOTSTRAP_DRAW_MANIFEST.json',draw_manifest)
 # Assemble training-side true-Y diagnostics and student dev rows from saved losses.
 training_rows=[]
 for program in PROGRAMS:
  for seed in SEEDS:
   slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';curve=read_csv(slot/'EPOCH_LOG.csv')
   final=next((r for r in reversed(curve) if int(r['epoch'])==160),None)
   if final is None:raise RuntimeError(f'final epoch curve absent for {program}/{seed}')
   dev_macro=next(r for r in primary if r['method']==program and int(r['seed'])==seed and r['task_id']=='macro')
   training_rows.append({'record_type':'student','program_id':program,'seed':seed,'epoch':160,
      'soft_z_position':final['soft_z_position'],'soft_z_safe_fd':final['soft_z_safe_fd'],'soft_z_objective':final['soft_z_objective'],
      'train_true_y_position':final['true_y_position'],'train_true_y_safe_fd':final['true_y_safe_fd'],'train_true_y_seam_safe_fd':final['true_y_seam_safe_fd'],
      'dev_position_mae':dev_macro['position_mae'],'dev_safe_fd_l1':dev_macro['safe_fd_l1'],'dev_seam_safe_fd_l1':dev_macro['seam_safe_fd_l1']})
 for r in read_csv(ROOT/'results/TEACHER_TRAIN_GT_METRICS.csv'):
  training_rows.append({'record_type':'teacher_train_GT_description','program_id':r['role'],'seed':r['seed'],'task_id':r['task_id'],
      'train_true_y_position':r.get('position_mae'),'train_true_y_safe_fd':r.get('safe_fd_l1'),'train_true_y_seam_safe_fd':r.get('seam_safe_fd_l1')})
 write_csv(ROOT/'results/TRAINING_AND_GT_METRICS.csv',training_rows)
 # Imitation error uses predictions only and reuses the accepted V6 SV_CAL alias for SV50.
 train_meta=score.read_prefix_metadata(Path(resolved['train_prefix_npz']));train_ids=list(train_meta);dev_ids=list(metadata)
 norm=json.loads(Path(resolved['formal_normalizer_path']).read_text());sigma=np.asarray(norm['target_sigma'],dtype=np.float64);fdscale=np.asarray(norm['safe_fd_scale'],dtype=np.float64)
 train_anchor=np.stack([train_meta[aid]['anchor_native'] for aid in train_ids]);dev_anchor=np.stack([metadata[aid]['anchor_native'] for aid in dev_ids])
 v6map=read_json(V6/'inventory/SOURCE_MAP.json');teacher_dev={}
 for role in ('A','V','SS50','SV50'):
  for seed in SEEDS:teacher_dev[(role,seed)]=score.load_teacher_dev(role,seed,metadata,dev_ids,v6map)
 student_dev={};student_train={};imit=[];role_for={'ST_SELF':'A','ST_SS':'SS50','ST_SV':'SV50','VD_SELF':'V','VD_SS':'SS50','VD_SV':'SV50'}
 for program in PROGRAMS:
  role=role_for[program]
  for seed in SEEDS:
   student_train[(program,seed)]=score.load_student_train(program,seed,train_meta,train_ids)
   student_dev[(program,seed)]=score.load_student_dev(program,seed,metadata,dev_ids)
   ttrain=score.load_teacher_train(role,seed,train_meta,train_ids)
   imit.append(score.imitation_metrics(student_train[(program,seed)],ttrain,train_anchor,train_ids,train_meta,sigma,fdscale,program,role,seed,'train'))
   imit.append(score.imitation_metrics(student_dev[(program,seed)],teacher_dev[(role,seed)],dev_anchor,dev_ids,metadata,sigma,fdscale,program,role,seed,'dev'))
 write_csv(ROOT/'results/IMITATION_ERROR.csv',imit)
 # Reuse GT trajectories already saved by V6 for its five metadata-preselected cases; do not reopen V7 dev target.
 cases=read_csv(V6/'figure_data/TRAJECTORY_CASES.csv');old_traj=read_csv(V6/'figure_data/TRAJECTORY_DATA.csv')
 gt_lookup={(str(r['anchor_id']),int(r['seed']),int(r['action_index']),int(r['future_step'])):r for r in old_traj if r['method']=='GT'}
 traj=[];id_index={aid:i for i,aid in enumerate(dev_ids)}
 for case in cases:
  aid=str(case['anchor_id'])
  if aid not in id_index:raise RuntimeError(f'V6 trajectory case missing from sealed V7 IDs: {aid}')
  i=id_index[aid];meta=metadata[aid]
  for seed in SEEDS:
   arrays={'ST_SV':student_dev[('ST_SV',seed)][i],'VD_SV':student_dev[('VD_SV',seed)][i],
           'SV50':teacher_dev[('SV50',seed)][i],'A':teacher_dev[('A',seed)][i],'V':teacher_dev[('V',seed)][i]}
   for act in (0,14):
    for step in range(30):
     key=(aid,seed,act,step+1);gt=gt_lookup.get(key)
     if gt is None:raise RuntimeError(f'accepted V6 saved GT trajectory value is missing: {key}')
     observed=(int(gt['task_id']),int(gt['episode_id']),str(gt['source_group']))
     if observed!=(meta['task_id'],meta['episode_id'],meta['source_group']):raise RuntimeError(f'V6 GT trajectory metadata mismatch: {key}')
     traj.append({'task_id':meta['task_id'],'episode_id':meta['episode_id'],'anchor_id':aid,'source_group':meta['source_group'],'seed':seed,
                  'action_index':act,'future_step':step+1,'method':'GT','native_command_value':float(gt['native_command_value']),
                  'unit_note':'numeric native command; physical units unresolved','case_selection':case['selection'],'reused_v5_case_id':case['reused_v5_case_id']})
     for method,array in arrays.items():
      traj.append({'task_id':meta['task_id'],'episode_id':meta['episode_id'],'anchor_id':aid,'source_group':meta['source_group'],'seed':seed,
                   'action_index':act,'future_step':step+1,'method':method,'native_command_value':float(array[step,act]),
                   'unit_note':'numeric native command; physical units unresolved','case_selection':case['selection'],'reused_v5_case_id':case['reused_v5_case_id']})
 write_csv(ROOT/'figure_data/TRAJECTORY_DATA.csv',traj)
 # The failed scorer had already read the archive once and saved every new window loss; recovery used those losses only.
 v5=json.loads((PROJECT/'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/inventory/OUTCOME_EXPOSURE.json').read_text())
 attempt_dirs=sorted((ROOT/'qa/failed_attempts').glob('dev_score_pairing_*'))
 if len(attempt_dirs)!=1:raise RuntimeError(f'expected one archived paired-score failure, found {len(attempt_dirs)}')
 access={'schema_version':1,'status':'PASS_RECOVERED_FROM_SAVED_LOSSES','target_archive_path':v5['target_archive_path'],
         'target_archive_sha256':v5['target_archive_sha256'],'target_archive_bytes':int(v5['target_archive_bytes']),
         'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),'first_score_pass_target_content_reads':1,
         'first_score_pass_target_rows_loaded':185,'first_score_pass_student_loss_evaluations':3330,
         'saved_student_window_loss_rows_reused':3330,'paired_recovery_reopened_dev_target_archive':False,
         'additional_target_archive_reads_during_recovery':0,'trajectory_GT_reused_from_v6_saved_figure_data':True,
         'trajectory_data_rows_written':len(traj),'post_score_target_reopens':0,'final_heldout_payload_access':0,
         'first_score_failure_archive':str(attempt_dirs[0])}
 write_json(ROOT/'qa/DEV_TARGET_ACCESS.json',access)
 exposure=read_json(ROOT/'inventory/OUTCOME_EXPOSURE.json')
 exposure.update({'v7_student_dev_logical_evaluations':3330,'v7_dev_target_archive_metadata_preflight_opens':1,
     'v7_dev_target_archive_content_reads':1,'v7_dev_target_archive_numpy_load_operations':1,'v7_dev_target_array_window_rows_loaded':185,
     'v7_score_recovery_target_reads':0,'v7_student_dev_loss_rows_reused_after_score_error':3330,'v7_student_trajectory_GT_source':'V6 saved figure_data/TRAJECTORY_DATA.csv',
     'v7_final_heldout_payload_access':0})
 write_json(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
 recovery={'schema_version':1,'status':'PASS','recovered_utc':utc(),'failure_type':'TypeError: saved V6 CSV metric strings were subtracted from V7 float losses',
           'paired_effects_and_CI_recomputed_from_saved_per_window_losses':True,'student_dev_scoring_repeated':False,
           'dev_target_archive_reopened':False,'teacher_dev_alias':'V6 SV_CAL alpha=.50 payload verified against SV50 source map',
           'task_loo_rows':len(task_loo),'seed_loo_rows':len(seed_loo),'paired_group_ci_rows':len(ci_rows)}
 write_json(ROOT/'qa/SCORE_RECOVERY.json',recovery)
 print(json.dumps({'status':'PASS','paired_contrasts':len(score.CONTRASTS),'saved_student_loss_rows_reused':3330,
                   'additional_dev_target_reads':0,'trajectory_rows':len(traj),'paired_group_ci_rows':len(ci_rows)},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
