#!/usr/bin/env python3
"""Build the V2 partial-acceptance record from current receipts; Core B is explicitly pending."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=P/"reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
V2RUN=P/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1=P/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V1RUN=P/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
OUT=V2/"qa/FINAL_ACCEPTANCE_V2.json"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def j(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csv_rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 freeze=j(V2/'core_a/FREEZE_RECEIPT.json');sem=j(V2/'qa/CORE_A_SEMANTICS_ACCEPTANCE.json');score=j(V2/'core_a/CORE_A_SCORE_RECEIPT.json');exposure=j(V2/'core_a/SCORE_EXPOSURE_SUMMARY.json');ledger=j(V2/'qa/EXPOSURE_AND_RETRY_LEDGER.json');independent=j(V2/'qa/CORE_A_SCORE_INDEPENDENT_VALIDATION.json');package=j(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/qa/CONTENT_AND_NUMERIC_CHECKS.json')
 class_sets=j(V2/'core_a/CLASS_SETS_AND_COVERAGE.json')
 per=csv_rows(V2/'core_a/FULL602_PER_SEED.csv');summ=csv_rows(V2/'core_a/FULL602_SUMMARY.csv');contr=csv_rows(V2/'core_a/PAIRED_CONTRASTS_AND_CI.csv')
 pmanifest=csv_rows(V1/'predictions/CORE_A_PREDICTION_MANIFEST.csv')
 coreb_p1=j(V2/'core_b/CORE_B_GPU_PROBE_STAGE0_P1.json');cross=j(V2/'qa/CROSS_CORE_INPUT_EXPOSURE_SUMMARY.json');watcher_fail=j(V2/'core_b/CORE_B_GPU_WATCHER_ATTEMPT_01.json')
 v1_accept=V1/'FINAL_ACCEPTANCE.json'
 result={
  'schema_version':2,
  'acceptance_status':'PARTIAL_ACCEPTANCE_CORE_A_COMPLETED_CORE_B_IN_PROGRESS',
  'created_utc':datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z'),
  'protocol_id':V2.name,
  'source_v1':{'out_root':str(V1),'run_root':str(V1RUN),'acceptance_sha256':sha(v1_accept),'prediction_seal_sha256':sha(V1/'predictions/CORE_A_PREDICTION_SEAL.json'),'prediction_manifest_sha256':sha(V1/'predictions/CORE_A_PREDICTION_MANIFEST.csv')},
  'protocol_state':{'v2_freeze_path':str(V2/'core_a/FREEZE_RECEIPT.json'),'v2_freeze_revision':freeze['revision'],'v2_freeze_sha256':sha(V2/'core_a/FREEZE_RECEIPT.json'),'core_a_analysis_plan_sha256':freeze['analysis_plan_sha256'],'core_a_protocol_amendment_sha256':sha(V2/'core_a/PROTOCOL_AMENDMENT_V2.json'),'new_training_runs':0},
  'core_a':{
   'original_v1_sidecar_estimand':'not reportable under stale V1 mapping; 93 of 120 column identities disagreed with accepted training target order.',
   'amended_estimand_status':'COMPLETED_AS_AMENDED_ANALYSIS',
   'population_rows':602,'model_output_columns':120,'C_model':class_sets['class_sets']['C_model'],'C_final':class_sets['class_sets']['C_final'],'C_common':class_sets['class_sets']['C_common'],'C_union':class_sets['class_sets']['C_union'],
   'sample_coverage':1.0,'class_coverage':1.0,'unmapped_final_rows':0,'common_class_estimand':'NOT_APPLICABLE_CLASS_SETS_IDENTICAL_TO_FULL',
   'mapping_evidence':{'semantics_acceptance_path':str(V2/'qa/CORE_A_SEMANTICS_ACCEPTANCE.json'),'semantics_acceptance_sha256':sha(V2/'qa/CORE_A_SEMANTICS_ACCEPTANCE.json'),'column_audit_path':str(V2/'core_a/COLUMN_SEMANTICS_AUDIT.csv'),'column_audit_sha256':sha(V2/'core_a/COLUMN_SEMANTICS_AUDIT.csv'),'train_target_encoding_sha256':sha(V2/'core_a/TRAIN_ID_TARGET_ENCODING.csv'),'corrected_class_map_sha256':sha(V2/'core_a/CORRECTED_CLASS_MAP_V2.json')},
   'prediction_cells':[{'program':r['program'],'seed':int(r['seed']),'checkpoint_sha256':r['checkpoint_sha256'],'prediction_sha256':r['prediction_sha256'],'sidecar_sha256':r['sidecar_sha256'],'shape':r['shape']} for r in pmanifest],
   'score_receipt_path':str(V2/'core_a/CORE_A_SCORE_RECEIPT.json'),'score_receipt_sha256':sha(V2/'core_a/CORE_A_SCORE_RECEIPT.json'),'per_seed_path':str(V2/'core_a/FULL602_PER_SEED.csv'),'summary_path':str(V2/'core_a/FULL602_SUMMARY.csv'),'paired_contrasts_path':str(V2/'core_a/PAIRED_CONTRASTS_AND_CI.csv'),'bootstrap_manifest_sha256':sha(V2/'core_a/BOOTSTRAP_DRAW_MANIFEST.json'),
   'per_seed_results':per,'summary_results':summ,'paired_contrasts':contr,
   'macro_f1_contract':score['macro_f1_definition'],'independent_arithmetic_validation':independent['status'],'independent_validation_sha256':sha(V2/'qa/CORE_A_SCORE_INDEPENDENT_VALIDATION.json'),
   'first_v2_label_read_utc':exposure['first_v2_label_read_utc'],'first_v2_metric_computed_utc':exposure['first_v2_metric_computed_utc'],'score_tables_committed_utc':exposure['score_table_commit_utc'],
   'exposure':{'v1_task_id_values_decoded':1204,'v2_score_task_id_values_decoded':1204,'cumulative_explicit_task_id_values_decoded':2408,'v2_pre_score_label_bearing_metadata_rows':4298,'v2_post_score_cross_core_metadata_rows':764,'unique_final_primary_rows':602,'unique_final_source_rows_touched':614,'attempt01_status':'FAILED_AFTER_602_LABELS_BEFORE_ANY_METRIC','attempt01_metrics':0,'attempt02_status':'COMPLETED','label_access_exposure_path':str(V2/'core_a/SCORE_EXPOSURE_SUMMARY.json')},
   'score_is_untouched_blind_confirmation':False,'session_independence_established':False,'supervisor_approved':False},
  'core_b':{
   'status':'DIAGNOSIS_IN_PROGRESS_FINAL_NOT_SCORED','v9_development_status':'ACCEPTED_DEVELOPMENT_ONLY','v9_matrix':'A0/AD/V0/VD × 3 seeds plus S_OPT × 3 seeds; MIX0/MIXD and Persistence derived/reference conditions',
   'primary_single_model_reference':'V0','secondary_dropout_condition':'VD','q0_feature_bridge_status':'FAIL_UNRESOLVED',
   'current_profile_route':'R2 candidate not yet locked; P2-P6 numerical profile checks and full dev replay pending.',
   'p1_gpu3_status':coreb_p1['status'],'p1_gpu3_physical_device':coreb_p1['physical_gpu'],'p1_gpu3_rows':2,'p1_allclose_q0':False,
   'source_hypothesis_status':'Row/frame identity and Q0 read/storage align on two train IDs; historical generation profile lacks runtime/source binding; no GPU-number cause established.',
   'cross_core_exposure':{'core_a_video_index_episodes':602,'core_b_final_anchor_rows':162,'core_b_final_unique_episodes':41,'overlap_episodes':cross['joined']['same_episode_path_overlap_count'],'table_path':str(V2/'core_b/CROSS_CORE_INPUT_EXPOSURE.csv'),'table_sha256':cross['table_sha256']},
   'core_b_v2_final_rgb_extracted':0,'core_b_v2_final_predictor_forwards':0,'core_b_future_targets_materialized':0,
   'model_cells_expected_for_final':'A0/AD/V0/VD × 3 seeds; S_OPT × 3 seeds; MIX0/MIXD per seed; Persistence',
   'watcher_attempt01':{'status':watcher_fail['status'],'selected_gpu':watcher_fail['selected_gpu'],'child_started':False,'cuda_child_started':False,'reason':watcher_fail['failure']},
   'gpu3_p1_watcher_status':j(V2RUN/'status/coreb_p1_gpu3_stage0_watcher.json')['status']},
  'deliverables':{'report_md':str(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/report/EE6008_Project_Report_DRAFT.en.md'),'report_docx':str(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/report/EE6008_Project_Report_DRAFT.en.docx'),'report_pdf':str(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/report/EE6008_Project_Report_DRAFT.en.pdf'),'team_slides':str(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/team_video/TEAM_5MIN_SLIDES.pptx'),'team_script':str(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/team_video/TEAM_5MIN_SCRIPT.en.md'),'personal_slides':str(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/personal_video/JI_YANCHEN_3MIN_SLIDES.pptx'),'personal_script':str(V2/'deliverables/EE6008_PROJECT_CLOSEOUT_V2_20261020T053953Z/personal_video/JI_YANCHEN_3MIN_SCRIPT.en.md')},
  'package_validation_status':package['status'],'human_review':'pending','submission_ready':False
 }
 # Refresh the Core B section from current receipts, preserving the accepted Core A record above.
 lock_path=V2/'core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json'
 dev_accept_path=V2/'core_b/CURRENT_PROFILE_DEV_ACCEPTANCE.json'
 final_score_path=V2/'core_b/CORE_B_V2_FINAL_SCORE_RECEIPT.json'
 final_exposure_path=V2/'core_b/FINAL_TARGET_EXPOSURE.json'
 final_freeze_path=V2/'core_b/CORE_B_V2_FINAL_FREEZE.json'
 final_access_path=V2/'protocol/FINAL_ACCESS_OPENED.json'
 final_feature_path=V2/'core_b/CURRENT_PROFILE_V2_FINAL_FEATURE_MANIFEST.json'
 final_prefix_receipt=V2/'core_b/FINAL_PREFIX_EXTRACTION_RECEIPT.json'
 final_seal_path=V2/'core_b/PREDICTION_MANIFEST_AND_SEAL.json'
 score_adapter_receipt=V2/'coordination/CORE_B_FINAL_SCORE_ADAPTER_EXECUTION.json'
 if all(p.is_file() for p in (dev_accept_path,final_score_path,final_exposure_path,final_freeze_path,final_access_path,final_feature_path,final_prefix_receipt,final_seal_path,score_adapter_receipt)):
  lock_now=j(lock_path);dev_now=j(dev_accept_path);final_now=j(final_score_path);exposure_now=j(final_exposure_path)
  dev_results=csv_rows(V2/'core_b/CURRENT_PROFILE_DEV_RESULTS.csv')
  final_results=csv_rows(V2/'core_b/final_scores/SUMMARY.csv')
  dev_contrasts=csv_rows(V2/'core_b/current_profile_dev_scores/CONTRASTS_AND_SEED_LOO.csv')
  final_contrasts=csv_rows(V2/'core_b/final_scores/CONTRASTS_AND_SEED_LOO.csv')
  dev_bootstrap=csv_rows(V2/'core_b/current_profile_dev_scores/PAIRED_BOOTSTRAP_INTERVALS.csv')
  final_bootstrap=csv_rows(V2/'core_b/final_scores/PAIRED_BOOTSTRAP_INTERVALS.csv')
  final_pred_receipt=j(V2RUN/'replay/core_b/current_profile_final/PREDICTION_RUN_RECEIPT.json')
  result['acceptance_status']='COMPLETED_AS_AMENDED_ANALYSIS_CORE_A_AND_CORE_B_FINAL_SCORED'
  cb=result['core_b']
  cb.update({
   'status':'COMPLETED_CURRENT_PROFILE_V2_DEV_AND_FINAL_SCORED',
   'v9_development_status':'ACCEPTED_DEVELOPMENT_ONLY',
   'current_profile_route':'R2 CURRENT_PROFILE_V2; dev scored under revision4 and final-only entrypoint correction revision5; frozen model, numeric profile, inputs, metrics, and populations are unchanged.',
   'current_profile_lock':{'path':str(lock_path),'revision':lock_now['lock_revision'],'sha256':sha(lock_path),'status':lock_now['status'],'dev_score_lock_revision':4},
   'final_only_entrypoint_amendment':{'path':str(V2/'coordination/CORE_B_FINAL_ENTRYPOINT_REVISION5_COMPARISON.json'),'sha256':sha(V2/'coordination/CORE_B_FINAL_ENTRYPOINT_REVISION5_COMPARISON.json'),'status':'PASS_FINAL_ONLY_CODE_AMENDMENT_PROFILE_UNCHANGED'},
   'current_profile_dev_acceptance':{'path':str(dev_accept_path),'sha256':sha(dev_accept_path),'status':dev_now['status'],'rows':dev_now['rows'],'episodes':dev_now['episode_support'],'prediction_bundles':dev_now['prediction_bundles'],'target_values_materialized_after_seal':dev_now['target_values_materialized_after_seal'],'final_payload_read':dev_now['final_payload_read']},
   'current_profile_dev_results':dev_results,
   'current_profile_dev_contrasts_three_seed_macro':[r for r in dev_contrasts if r.get('grain')=='three_seed_task_macro'],
   'current_profile_dev_paired_bootstrap_intervals':dev_bootstrap,
   'current_profile_final_score_receipt':{'path':str(final_score_path),'sha256':sha(final_score_path),'status':final_now['status'],'rows':final_now['rows'],'episodes':final_now['episode_support'],'prediction_bundles':final_now['prediction_bundles'],'target_rows_materialized_after_seal':final_now['target_rows_materialized_after_seal'],'target_values_materialized_after_seal':final_now['target_values_materialized_after_seal'],'raw_parquet_files_read':final_now['raw_parquet_files_read'],'unique_raw_parquet_action_rows_read':final_now['unique_raw_parquet_action_rows_read']},
   'current_profile_final_results':final_results,
   'current_profile_final_contrasts_three_seed_macro':[r for r in final_contrasts if r.get('grain')=='three_seed_task_macro'],
   'current_profile_final_paired_bootstrap_intervals':final_bootstrap,
   'primary_single_model_reference':'V0','secondary_dropout_condition':'VD',
   'model_cells_expected_for_final':'A0/AD/V0/VD × 3 seeds; S_OPT × 3 seeds; MIX0/MIXD per seed; Persistence',
   'current_profile_model_bundles':{'neural':15,'derived_equal_weight_mixtures':6,'persistence':1,'total':22},
   'q0_feature_bridge_status':'FAIL_UNRESOLVED','q0_generation_root_cause':'UNKNOWN; no GPU-number attribution',
   'dev_target_values_materialized_after_seal':dev_now['target_values_materialized_after_seal'],
   'final_prefix_receipt':{'path':str(final_prefix_receipt),'sha256':sha(final_prefix_receipt),'status':j(final_prefix_receipt)['status'],'rows':j(final_prefix_receipt)['rows'],'target_h30_rows_materialized':j(final_prefix_receipt)['target_h30_rows_materialized']},
   'final_feature_manifest':{'path':str(final_feature_path),'sha256':sha(final_feature_path),'status':j(final_feature_path)['status'],'rows':j(final_feature_path)['rows'],'feature_shape':j(final_feature_path)['feature_shape'],'future_targets_materialized':False},
   'final_prediction_seal':{'path':str(final_seal_path),'sha256':sha(final_seal_path),'status':j(final_seal_path)['status'],'prediction_bundles':j(final_seal_path)['prediction_bundles'],'target_values_materialized':j(final_seal_path)['target_values_materialized']},
   'final_access':{'path':str(final_access_path),'sha256':sha(final_access_path),'status':j(final_access_path)['status'],'authorization_source':j(final_access_path)['authorization_source'],'opened_utc':j(final_access_path)['opened_utc']},
   'final_score_adapter_execution':{'path':str(score_adapter_receipt),'sha256':sha(score_adapter_receipt),'status':j(score_adapter_receipt)['status'],'adapter_path':j(score_adapter_receipt)['score_adapter_path'],'adapter_sha256':j(score_adapter_receipt)['score_adapter_sha256'],'metadata_join':j(score_adapter_receipt)['metadata_join']},
   'core_b_v2_final_rgb_extracted':162,'core_b_v2_final_predictor_forwards':sum(int(r.get('forward_rows',0)) for r in final_pred_receipt['prediction_records']),'core_b_future_target_rows_materialized_after_seal':final_now['target_rows_materialized_after_seal'],'core_b_future_target_values_materialized_after_seal':final_now['target_values_materialized_after_seal'],
   'score_interpretation':'fixed-model descriptive dev and final results under the same CURRENT_PROFILE_V2 profile; dev target read followed the dev prediction seal; final target reads followed the final 22-bundle seal; no new training or effect-based profile selection.',
   'p_values_and_equivalence_claims':'not planned',
   'supervisor_approved':False,
  })
  result['acceptance_status']='COMPLETED_AS_AMENDED_ANALYSIS_CORE_A_AND_CORE_B_FINAL_SCORED'
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'acceptance_status':result['acceptance_status'],'core_a':result['core_a']['amended_estimand_status'],'core_b':result['core_b']['status'],'output':str(OUT)},ensure_ascii=False))
if __name__=='__main__':main()
