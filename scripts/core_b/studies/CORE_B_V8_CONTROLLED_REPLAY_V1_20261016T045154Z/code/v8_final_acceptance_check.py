#!/usr/bin/env python3
"""CPU-only V8 final acceptance check; does not open any target archive."""
from __future__ import annotations
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(202610131,202610132,202610133);PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 checks={};branch=j(ROOT/'protocol/BRANCH_LOCK.json');recipe=j(ROOT/'protocol/RECIPE_LOCK.json');mix=j(ROOT/'protocol/TARGET_MIX_LOCK.json')
 checks['branch_locked_before_dev']=branch['status']=='LOCKED_BEFORE_FULL_FITS_AND_NEW_DEV' and branch['branch']=='B_SYNCHRONOUS_VIDEO_9' and branch['branch_selected_from_new_dev_outcomes'] is False
 checks['branch_recipe_target_hashes']=recipe['branch_lock_sha256']==sha(ROOT/'protocol/BRANCH_LOCK.json') and recipe['target_mix_lock_sha256']==sha(ROOT/'protocol/TARGET_MIX_LOCK.json') and mix['status']=='LOCKED_BEFORE_FORMAL_FITS'
 checks['training_runner_hash']=recipe['runner_sha256']==sha(ROOT/'code/v8_train.py')
 fits=[];samplers={}
 for seed in SEEDS:
  init=set();sampler=set()
  for program in PROGRAMS:
   slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';m=j(slot/'MODEL_MANIFEST.json');c=j(slot/'RUN_COMPLETE.json')
   cfgp=ROOT/'configs/fits'/f'v8_fit_{program}_s{seed}.json';cfg=j(cfgp)
   assert m['status']=='PASS' and c['status']=='PASS' and m['epochs']==160 and c['optimizer_steps']==3200
   assert m['training_runner_sha256']==sha(ROOT/'code/v8_train.py') and m['job_config_sha256']==sha(cfgp)
   assert m['branch_lock_sha256']==sha(ROOT/'protocol/BRANCH_LOCK.json') and m['recipe_lock_sha256']==sha(ROOT/'protocol/RECIPE_LOCK.json')
   assert m['model_file_sha256']==sha(slot/'model.safetensors') and m['training_target_sha256']==cfg['teacher_target_sha256']
   init.add(m['init_state_hash']);sampler.add(m['sampler_sequence_sha256'])
   fits.append({'program':program,'seed':seed,'epochs':m['epochs'],'optimizer_steps':m['optimizer_steps'],'gpu':m['physical_gpu'],'target_mode':m['target_mode']})
  checks[f'same_init_and_sampler_seed_{seed}']=len(init)==1 and len(sampler)==1
 checks['formal_fits_9_of_9']=len(fits)==9
 seal=j(ROOT/'qa/PREDICTION_SEAL.json');checks['prediction_seal']=seal['status']=='SEALED_BEFORE_TARGET_ACCESS' and seal['accepted_train_bundles']==9 and seal['accepted_dev_bundles']==9 and seal['dev_target_archive_opened'] is False
 checks['preseal_typed_qa']=j(ROOT/'qa/PRESEAL_TYPED_AUDIT.json')['status']=='PASS'
 checks['dev_schema_metadata_only']=j(ROOT/'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json')['target_numpy_content_reads']==0
 checks['train_only_summary']=j(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json')['status']=='PASS' and j(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json')['dev_target_archive_opened'] is False
 checks['synthetic_scorer_qa']=j(ROOT/'qa/SYNTHETIC_SCORER_QA.json')['status']=='PASS'
 checks['hard_baseline_reproduction']=j(ROOT/'qa/HARD_BASELINE_REPRODUCTION.json')['all_three_hard_model_files_bit_identical'] and j(ROOT/'qa/HARD_BASELINE_REPRODUCTION.json')['all_three_state_hashes_identical'] and j(ROOT/'qa/HARD_BASELINE_REPRODUCTION.json')['all_train_dev_replays_within_tolerance']
 checks['figure_build_and_visual_review']=j(ROOT/'qa/FIGURE_BUILD_QA.json')['status']=='PASS' and j(ROOT/'qa/FIGURE_VISUAL_REVIEW.json')['status']=='PASS'
 score=j(ROOT/'results/V8_SCORING_MANIFEST.json');checks['single_dev_target_content_read']=score['dev_target_content_reads']==1 and score['numpy_load_calls']==1 and score['dev_rows_materialized']==185 and score['new_student_dev_evaluations']==1665
 checks['final_test_denial']=score['final_heldout_payload_access']==0 and score['final_task_level_heldout_aggregate_access']==0
 checks['known_grain_counts']=score['V7_per_window_rows_reused']==7770 and score['new_student_dev_evaluations']==1665
 expected={'results/PER_WINDOW_NEW_DEV_LOSSES.csv':1665,'results/PRIMARY_TASK_SEED.csv':306,'results/PAIRED_TASK_SEED.csv':405,'results/PAIRED_EFFECTS.csv':243,
  'results/SEED_LOO.csv':81,'results/TASK_LOO.csv':135,'results/PAIRED_GROUP_CI.csv':27,'results/TRAIN_Y_Z_AND_IMITATION.csv':27,'figure_data/TRAINING_CURVES.csv':1440}
 row_counts={}
 for rel,count in expected.items():
  n=len(rows(ROOT/rel));row_counts[rel]=n;checks['rows_'+rel]=n==count
 res=j(ROOT/'qa/RESOURCE_SUMMARY.json');checks['leases_released']=res['all_watcher_leases_released'] is True and res['watcher_max_concurrent_gpu_children']==2
 checks['exposure_final']=j(ROOT/'inventory/OUTCOME_EXPOSURE.json')['final_heldout_payload_access']==0 and j(ROOT/'inventory/OUTCOME_EXPOSURE.json')['v8_dev_target_content_reads']==1
 checks['old_roots_untouched']=j(ROOT/'inventory/SOURCE_MAP.json')['old_roots_written'] is False
 checks['source_integrity']=j(ROOT/'qa/SOURCE_INTEGRITY.json')['status']=='PASS' and all(x['match'] for x in j(ROOT/'qa/SOURCE_INTEGRITY.json')['source_hash_checks'])
 checks['handoff_report_exists']=(ROOT/'RETURN_TO_CHATGPT.zh-CN.md').is_file() and j(ROOT/'RETURN_TO_CHATGPT.json')['status']=='complete'
 checks['final_acceptance_complete']=j(ROOT/'qa/FINAL_ACCEPTANCE.json')['status']=='complete'
 for name in ['TRAINING_CURVES.png','PAIRED_DUAL_METRICS_AND_SEAM.png','TRAIN_AND_REPLAY_AUDIT.png','TRAJECTORY_CASES.png']:
  checks['figure_'+name]=(ROOT/'figures'/name).is_file() and (ROOT/'figures'/name).stat().st_size>1000
 result={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'fit_count':len(fits),'fits':fits,'result_row_counts':row_counts,
  'dev_target_content_reads':1,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,'old_roots_written':0}
 p=ROOT/'qa/FINAL_CHECK.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':result['status'],'failed_checks':result['failed_checks'],'fit_count':len(fits),'dev_reads':1,'final_access':0},sort_keys=True))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
