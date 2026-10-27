#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROGRAMS=('A0','AD','V0','VD');SEEDS=(202610131,202610132,202610133)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rc(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 study=rj(ROOT/'protocol/STUDY_LOCK.json');recipe=rj(ROOT/'protocol/RECIPE_LOCK.json')
 res=rj(ROOT/'protocol/RESOURCE_LOCK.json');code=rj(ROOT/'protocol/CODE_LOCK.json')
 scoring_code=rj(ROOT/'protocol/SCORING_CODE_LOCK.json');seal=rj(ROOT/'qa/PREDICTION_SEAL.json')
 single=rj(ROOT/'qa/DEV_TARGET_SINGLE_READ.json');scoring=rj(ROOT/'results/V9_SCORING_MANIFEST.json')
 exposure=rj(ROOT/'inventory/OUTCOME_EXPOSURE.json');final=rj(ROOT/'qa/FINAL_ACCEPTANCE.json')
 visual=rj(ROOT/'figures/FIGURE_VISUAL_REVIEW.json');source_map=rj(ROOT/'inventory/SOURCE_MAP.json');old_root=rj(ROOT/'qa/OLD_ROOT_INTEGRITY.json');synth=rj(ROOT/'qa/SYNTHETIC_SCORER_QA.json')
 pre=rj(ROOT/'qa/FORMAL_JOB_PREFLIGHT.json');inferpre=rj(ROOT/'qa/INFERENCE_JOB_PREFLIGHT.json')
 recovery=rj(ROOT/'results/POST_TARGET_AGGREGATION_RECOVERY.json');v0=rj(ROOT/'qa/V0_PREDICTION_REPLAY.json')
 if study.get('formal_fit_count')!=12 or len(PROGRAMS)*len(SEEDS)!=12:raise RuntimeError('study matrix lock mismatch')
 fit_count=0
 for seed in SEEDS:
  for p in PROGRAMS:
   out=ROOT/'model_only'/p/f'seed{seed}'/'attempt_01';m=rj(out/'MODEL_MANIFEST.json');c=rj(out/'RUN_COMPLETE.json')
   if m.get('status')!='PASS' or c.get('status')!='PASS' or int(c.get('epochs_completed',-1))!=160 or int(c.get('optimizer_steps',-1))!=3200:
    raise RuntimeError('formal fit acceptance invalid '+p+'/'+str(seed))
   if c.get('target_role')!='TRUE_Y' or c.get('teacher_target_used') is not False or int(c.get('dev_or_final_target_rows',-1))!=0:
    raise RuntimeError('true-Y/target-isolation fit contract invalid')
   if sha(m['model_file'])!=m['model_file_sha256']:raise RuntimeError('checkpoint digest mismatch')
   fit_count+=1
 if fit_count!=12:raise RuntimeError('formal fit count differs')
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('accepted_neural_prediction_bundles')!=36 or seal.get('accepted_derived_prediction_bundles')!=6 or seal.get('accepted_dev_bundles')!=30 or seal.get('missing_bundles')!=0:
  raise RuntimeError('prediction seal/count invalid')
 if not seal.get('dev_target_archive_opened') is False or seal.get('preseal_dev_target_reads')!=0:raise RuntimeError('prediction seal exposure boundary invalid')
 if single.get('status')!='PASS' or single.get('archive_content_reads')!=1 or single.get('numpy_load_calls')!=1:raise RuntimeError('dev target single-read receipt invalid')
 if scoring.get('dev_target_archive_content_reads')!=1 or scoring.get('new_dev_logic_bundles')!=30 or scoring.get('new_dev_loss_evaluations')!=5550:raise RuntimeError('scoring manifest exposure/count invalid')
 if int(exposure.get('v9_dev_target_archive_content_reads',-1))!=1 or exposure.get('final_heldout_payload_access')!=0 or exposure.get('final_task_level_heldout_aggregate_access')!=0:raise RuntimeError('exposure ledger invalid')
 files={
  'per_window_new_dev':(ROOT/'results/PER_WINDOW_NEW_DEV_LOSSES.csv',5550),
  'per_window_all_methods':(ROOT/'results/PER_WINDOW_ALL_METHODS.csv',8880),
  'primary_task_seed':(ROOT/'results/PRIMARY_TASK_SEED.csv',216),
  'masked_diagnostic':(ROOT/'results/MASKED_DIAGNOSTIC.csv',72),
  'paired_effects':(ROOT/'results/PAIRED_EFFECTS.csv',1008),
  'paired_group_ci':(ROOT/'results/PAIRED_GROUP_CI.csv',42),
  'seed_loo':(ROOT/'results/SEED_LOO.csv',126),
  'task_loo':(ROOT/'results/TASK_LOO.csv',210),
  'train_full':(ROOT/'results/PER_WINDOW_TRAIN_FULL.csv',15360),
}
 counts={}
 for name,(path,expected) in files.items():
  rows=rc(path);counts[name]=len(rows)
  if len(rows)!=expected:raise RuntimeError(f'{name} row count mismatch: {len(rows)} vs {expected}')
  if name in ('per_window_new_dev','per_window_all_methods','primary_task_seed','masked_diagnostic','paired_effects','paired_group_ci'):
   numeric=[v for row in rows for k,v in row.items() if k in ('position_mae','safe_fd_l1','seam_safe_fd_l1','gain_reference_minus_candidate','ci_2_5','ci_97_5') and v not in ('',None)]
   if not np.isfinite(np.asarray(numeric,dtype=np.float64)).all():raise RuntimeError(name+' has nonfinite numeric values')
 if final.get('status')!='PASS' or final.get('figures')!='PASS' or final.get('visual_review')!='PASS':raise RuntimeError('final acceptance/figure review not PASS')
 if visual.get('status')!='PASS' or len(visual.get('figures',{}))!=6:raise RuntimeError('visual review missing')
 if synth.get('status')!='PASS' or pre.get('status')!='PASS' or inferpre.get('status')!='PASS':raise RuntimeError('preflight QA missing')
 if recovery.get('status')!='PASS' or recovery.get('dev_target_archive_reopened') is not False:raise RuntimeError('post-read recovery receipt invalid')
 if source_map.get('status')!='FINAL_SOURCES_AND_OUTCOMES_VERIFIED' or source_map.get('old_roots_written') is not False:raise RuntimeError('source map/old-root write boundary invalid')
 if old_root.get('status')!='PASS' or int(old_root.get('manifest_entries',0))!=1193:raise RuntimeError('V8 old-root checksum verification invalid')
 if not (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').is_file() or not (ROOT/'RETURN_TO_CHATGPT.json').is_file():raise RuntimeError('final report files missing')
 if not v0.get('V9_V0_exact_prediction_replay_all_roles_and_seeds'):raise RuntimeError('V0 exact prediction replay not accepted')
 resource=rj(ROOT/'results/RESOURCE_SUMMARY.json')
 if resource.get('formal_fit_count')!=12 or resource.get('neural_inference_count')!=36 or resource.get('all_leases_released') is not True or resource.get('oom_count')!=0:
  raise RuntimeError('resource summary invalid')
 audit={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'formal_fits':fit_count,'neural_predictions':36,'derived_mixtures':6,'dev_target_reads':1,'dev_target_reopened_after_score':False,
  'final_heldout_payload_access':0,'verified_result_row_counts':counts,'v0_exact_replay':True,
  'resource_records':resource['total_resource_records'],'all_watcher_leases_released':True,'visual_review':'PASS',
 'old_v8_root_checksum_entries_verified':old_root['manifest_entries'],'old_roots_written':False,
 'source_map_status':source_map['status'],'final_report_files_present':True,
  'study_lock_sha256':sha(ROOT/'protocol/STUDY_LOCK.json'),'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),
  'training_code_lock_sha256':sha(ROOT/'protocol/CODE_LOCK.json'),'scoring_code_lock_sha256':sha(ROOT/'protocol/SCORING_CODE_LOCK.json'),
  'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),'dev_target_archive_sha256':single['archive_sha256']}
 p=ROOT/'qa/FINAL_AUDIT.json';p.write_text(json.dumps(audit,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(audit,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()

