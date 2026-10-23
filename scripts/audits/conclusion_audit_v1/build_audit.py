#!/usr/bin/env python3
"""Bounded EE6008 conclusion audit first-pass builder.

This emits an intermediate package; after running it, run v3_amendment.py to apply the sealed V3 outcome and regenerate final QA/checksums. Do not treat this first-pass output alone as final.

Standard-library only. Reads explicitly selected report/acceptance/aggregate tables;
never opens prediction/target arrays or model weights, and never imports project code.
"""
from __future__ import annotations
import os
import csv, hashlib, json, math, os, resource, statistics, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
AUDIT = PROJECT / "reports/audits/EE6008_CONCLUSION_AUDIT_V1_20261014T061114Z"
START_UTC = "2026-10-14T06:11:14Z"
OUT = AUDIT
QA = AUDIT / "qa"
ALLOWED_SUFFIXES = {".csv", ".json", ".md", ".ini", ".py", ".txt"}
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".parquet", ".pt", ".pth", ".ckpt", ".mp4", ".avi", ".mkv"}

# (id, project-relative path, role, locator, manifest project-relative path, live)
SPECS = [
("E001","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/HARVEST_REPORT_20260901.md","accepted_report","L7-L32; dynamic/static, CI and limitations","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/CHECKSUMS.sha256",False),
("E002","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/STANDARD_BATCH48_AGGREGATE.csv","accepted_metric_table","rows 2-3; dynamic/static selected-best","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/CHECKSUMS.sha256",False),
("E003","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/STANDARD_BATCH48_PER_SEED.csv","accepted_per_seed_table","rows 2-7; paired D25/S25 D26/S26 D27/S27","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/CHECKSUMS.sha256",False),
("E004","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/BOOTSTRAP_RESULTS.csv","accepted_interval_table","pair B48_dynamic_minus_B48_static; top1/macro_f1; class_stratified/hierarchical","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/CHECKSUMS.sha256",False),
("E005","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/CORE_A_BATCH48_DECISION.json","acceptance_and_decision","evidence.dynamic_static; n=3 per-seed deltas; selection gate","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/CHECKSUMS.sha256",False),
("E006","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_EXPERIMENT_SUMMARY.zh-CN.md","accepted_report","L13-L85; historical baseline, batch-size, Full120, uncertainty","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E007","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_PAIRWISE_COMPARISONS.csv","accepted_paired_table","rows by comparison_id D-P-L_vs_D-P-F; S-P-L_vs_S-P-F; D-P-F_vs_S-P-F","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E008","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_MODEL_SELECTION_SUMMARY.json","accepted_selection_summary","highest_video_top1; best_dynamic_by_top1; single-seed caveats","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E009","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/results/CORE_A_STAGE1_FINAL_ACCEPTANCE.csv","accepted_cell_acceptance","rows 2-10; 9 Full120 Stage1 cells","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/manifests/ARCHIVE_COPY_INDEX.csv",False),
("E010","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_UNIFIED_CONCLUSIONS.zh-CN.md","accepted_conclusion","L7-L26; best candidate and inference boundary","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/CORE_A_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),

("E011","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/ANALYSIS_STATUS.json","accepted_status","status; B1 7/7; B2 6/6; B3 8/8; candidate gate","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E012","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/ANALYSIS_QA.json","acceptance_QA","all checks; single-seed B3 caveat","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E013","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/B1_REACCEPTANCE.json","accepted_cell_table","logical_coverage 7/7; rows by cell","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E014","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/B2_REACCEPTANCE.json","accepted_cell_table","6 cells; source_kind and pixel_level_sampling","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E015","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/B3_REACCEPTANCE.json","accepted_cell_table","8 cells; two reused B2 dense controls; single-seed caveat","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E016","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/CORE_B_UNIFIED_RESULTS.csv","accepted_aggregate_metrics","rows 2-25; B1/B2/B3 and persistence; 25 formal rows","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256",False),
("E017","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d1/D1_SUMMARY.json","accepted_metadata_summary","CPU_COMPLETE; train/dev support and outcome","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/CHECKSUMS.sha256",False),
("E018","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d1/D1_LEARNED_SUMMARY.json","accepted_metadata_summary","learned sparse complete; 36 prediction bundles","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/CHECKSUMS.sha256",False),
("E019","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_SUMMARY.json","accepted_metadata_summary","CPU baselines; dense panel and event support","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/CHECKSUMS.sha256",False),
("E020","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_DENSE_LEARNED_SUMMARY.json","accepted_metadata_summary","dense learned status; models and rows","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/CHECKSUMS.sha256",False),
("E021","reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/qa/FINAL_ACCEPTANCE.json","accepted_final_acceptance","status complete; caveats; D1/D2/D4 status","reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/CHECKSUMS.sha256",False),
("E022","reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/d4/D4_CROSS_SEED_DESCRIPTIVE_SUMMARY.json","accepted_aggregate","3 fixed seeds; 118 source groups; descriptive bootstrap","reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/CHECKSUMS.sha256",False),

("E023","reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/code/d3_train_historical_original.py","historical_as_run_code","loss_value then batch mean weight at L140-L152; epoch scheduler call","reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/CHECKSUMS.sha256",False),
("E024","reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/code/d3_train_eq.py","corrected_objective_code","episode weights and corrected training loop","reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/CHECKSUMS.sha256",False),
("E025","reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/D3_EQ_REPAIR_V1.md","repair_record","L1-L24; preserved failure and narrow repair","reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/CHECKSUMS.sha256",False),
("E026","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/RETURN_TO_CHATGPT.zh-CN.md","accepted_result_return","L5-L13; 12/12 train; scheduler effects","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/CHECKSUMS.sha256",False),
("E027","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/02_SCIENTIFIC_RESULTS.zh-CN.md","accepted_scientific_report","L7-L31; loss, TV, interaction, uncertainty; L39-L67 under-dynamic","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/CHECKSUMS.sha256",False),
("E028","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/FINAL_ACCEPTANCE.json","accepted_final_acceptance","formal 12/12; accepted metrics and source groups","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/CHECKSUMS.sha256",False),
("E029","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code/d3_eq_contract.py","corrected_objective_scheduler_source","L11-L22, L25-L62, L81-L119","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/CHECKSUMS.sha256",False),

("E030",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/RAW_FRAME_ORDER_STATE_INFO_V1_20260924T172133Z/reports/RETURN_TO_CHATGPT.zh-CN.md'),"accepted_scientific_return","L7-L21; 30 fit cells; conditional state/video; low TV",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/RAW_FRAME_ORDER_STATE_INFO_V1_20260924T172133Z/qa/SHA256SUMS.final'),False),
("E031",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/RAW_FRAME_ORDER_STATE_INFO_V1_20260924T172133Z/reports/CLAIM_LEDGER.zh-CN.md'),"accepted_claim_ledger","L7-L15; limited video effect; no chronology/trajectory claim",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/RAW_FRAME_ORDER_STATE_INFO_V1_20260924T172133Z/qa/SHA256SUMS.final'),False),
("E032",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/CORE_B_MOTION_FIDELITY_AUDIT_V1_20260925T064134Z/reports/RETURN_TO_CHATGPT_20261006.zh-CN.md'),"corrected_erratum_and_metrics","L7-L18; Hand/Arm swap correction, train/dev fit unknown",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/CORE_B_MOTION_FIDELITY_AUDIT_V1_20260925T064134Z/qa/FINAL_SHA256SUMS_20261006.csv'),False),
("E033",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/CORE_B_MOTION_FIDELITY_AUDIT_V1_20260925T064134Z/reports/RFSI_LABEL_AND_RESULT_ERRATUM_20261006.zh-CN.md'),"erratum","current path may be absent; search bounded to reports",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/CORE_B_MOTION_FIDELITY_AUDIT_V1_20260925T064134Z/qa/FINAL_SHA256SUMS_20261006.csv'),False),
("E034",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/HAND_DIRECT_INCREMENT_V1_20261006071115Z/harvest_recovery_20261006T131215Z/RETURN_TO_CHATGPT.zh-CN.md'),"accepted_formal_result","L5-L68; direct/increment gains and failed Hand-FD gate",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/HAND_DIRECT_INCREMENT_V1_20261006071115Z/harvest_recovery_20261006T131215Z/qa/FINAL_SHA256SUMS.csv'),False),
("E035",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/HAND_DIRECT_INCREMENT_V1_20261006071115Z/harvest_recovery_20261006T131215Z/qa/FINAL_QA.json'),"accepted_qa","candidate_gate FAIL_NOT_PROMOTED; semantic acceptance 18; source fit separate",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/HAND_DIRECT_INCREMENT_V1_20261006071115Z/harvest_recovery_20261006T131215Z/qa/FINAL_SHA256SUMS.csv'),False),

("E036","reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z/RETURN_TO_CHATGPT.zh-CN.md","accepted_result_return","L5-L28; H60 gate not pass; 3-seed visual interaction","reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z/CHECKSUMS.sha256",False),
("E037","reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/RETURN_TO_CHATGPT.zh-CN.md","accepted_N0_N1_report","L7-L34, L37-L76; interventions and query effects","reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/CHECKSUMS.sha256",False),
("E038","reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/qa/FINAL_ACCEPTANCE.json","accepted_final_acceptance","N0 6 replay/12 intervention; N1 6 Q1; no independent confirmation","reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/CHECKSUMS.sha256",False),
("E039","repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/FINAL_STATUS.ini","accepted_V3_54_status","STAGE0 18/18; training/predictions 54/54; dev only","repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/CHECKSUMS.sha256",False),
("E040","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/RETURN_TO_CHATGPT.zh-CN.md","accepted_N2_N3_N4_N5_handoff","L6-L23; point effects, state interactions, N5 readiness","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/CHECKSUMS.sha256",False),
("E041","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/qa/FINAL_ACCEPTANCE.json","accepted_N2_status","12 safe models; 24 CPU/CUDA views; no new training","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/CHECKSUMS.sha256",False),
("E042","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/results/N2_BY_SEED.csv","accepted_per_seed_aggregate","profile=CUDA_COMMON; model/query/seed/metric group means","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/CHECKSUMS.sha256",False),
("E043","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/results/PREDICTION_BACKEND_DIFFERENCES.csv","accepted_backend_summary","12 same-model CPU/CUDA common comparisons","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/CHECKSUMS.sha256",False),
("E044","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/n3_refs/N3_DIRECT_REUSE.json","accepted_N3_direct_reuse","direct reference and 18-view reuse scope","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/CHECKSUMS.sha256",False),
("E045","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/n3_refs/RESULTS_BY_SEED.csv","accepted_N3_per_seed_summary","54 historical rows; H30 FULL/OBS/ANCHOR D/N cells","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/CHECKSUMS.sha256",False),
("E046","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/n3_refs/VIDEO_GAINS_BY_STATE_LEVEL.csv","accepted_N3_gain_summary","conditioned video gains and intervals by state/group","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/CHECKSUMS.sha256",False),
("E047","reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z/FINAL_REPORT.zh-CN.md","accepted_horizon_report","L5-L17, L52-L68, L72-L93","reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z/CHECKSUMS.sha256",False),
("E048","reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z/qa/FINAL_ACCEPTANCE.json","accepted_horizon_acceptance","fixed H30, 13 Ridge fits, CI scope, 0 heldout","reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z/CHECKSUMS.sha256",False),
("E049","reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z/FINAL_REPORT.zh-CN.md","accepted_OBS43_report","L5-L23, L31-L59; distinct 43D observations and 28D command","reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z/CHECKSUMS.sha256",False),
("E050","reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z/qa/FINAL_ACCEPTANCE.json","accepted_OBS43_contract","eligible source groups, clocks, raw field semantics","reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z/CHECKSUMS.sha256",False),
("E051","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z/returns/RETURN_TO_CHATGPT.zh-CN.md","accepted_OBS43_closeout","L3-L10; no new fits; diagnostics","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z/CHECKSUMS.sha256",False),
("E052","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z/qa/FINAL_ACCEPTANCE.json","accepted_OBS43_status","independent confirmation spec not executed","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z/CHECKSUMS.sha256",False),
("E053","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z/RETURN_TO_CHATGPT.zh-CN.md","accepted_N4_report","L7-L17, L22-L34; S02 errors and dynamics","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z/CHECKSUMS.sha256",False),
("E054","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z/qa/FINAL_ACCEPTANCE.json","accepted_N4_status","24/24 analysis bundles; source scope","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z/CHECKSUMS.sha256",False),
("E055","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z/results/N4_EFFECT_SUPPORT_SUMMARY.json","accepted_N4_summary","effects/support by metric and seed; aggregate summary only","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z/CHECKSUMS.sha256",False),

("E056","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/RETURN_TO_CHATGPT.zh-CN.md","accepted_N5_return","L3-L15; population and scope","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E057","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/STATISTICAL_ERRATUM.md","accepted_N5_erratum","L5-L15; N2 group-only vs N4 crossed; reproduced endpoints","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E058","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/results/CO_PRIMARY_SCOPE_TABLE30.csv","accepted_interval_table","metric/contrast/interval_scope rows; all 30 rows","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E059","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/results/BY_SEED_AND_LOO.csv","accepted_seed_LOO_table","seed estimates/true leave-one-seed-out rows","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E060","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/results/HISTORICAL_INTERVAL_REPRODUCTION.csv","accepted_interval_reproduction","30 historical endpoints; exact accepted rows","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E061","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/protocol/DRAW_SPEC.json","accepted_draw_protocol","group 603×10k and historical seed draws; no arrays read","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E062","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/protocol/REALIZED_DRAW_HASHES.json","accepted_draw_hashes","27 ordered triples; draw counts and hashes metadata only","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E063","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/protocol/SEED_PAIRING_PROVENANCE.json","accepted_pairing_provenance","init hashes pass; sample order hash absent","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E064","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/qa/FINAL_ACCEPTANCE.json","accepted_N5_status","status, 30/30, 30/30 replay, 0 final/heldout","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E065","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/protocol/OUTCOME_EXPOSURE.json","accepted_historical_exposure","14472 accepted GT-derived group losses; historical reads","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E066","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/readiness/DATA_GAP.md","accepted_data_gap","L3-L7; exposed train/dev and requirements","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),
("E067","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/readiness/METADATA_ADMISSION.json","accepted_pool_metadata","candidate=0, clean clusters=0, registry scope","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/CHECKSUMS.sha256",False),

("E068","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/returns/RETURN_TO_CHATGPT.zh-CN.md","accepted_OBS14_return","L3-L11; 0/270 and no score","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/CHECKSUMS.sha256",False),
("E069","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/catalog/STRICT_CLUSTER_COUNTS.json","accepted_pool_counts","strict clusters and unknown exposure counts","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/CHECKSUMS.sha256",False),
("E070","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/catalog/PROTECTED_DATA_EXCLUSION.csv","protected_role_aggregate","row final_test; count only; IDs not enumerated","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/CHECKSUMS.sha256",False),
("E071","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/catalog/POOL_RECONCILIATION.csv","accepted_pool_reconciliation","rows 2-8; roles, exclusions and strict pool","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/CHECKSUMS.sha256",False),
("E072","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/catalog/SEARCH_COVERAGE.json","accepted_search_coverage","bounded surfaces and explicit not-searched roles","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z/CHECKSUMS.sha256",False),

("E073","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/RETURN_TO_CHATGPT.zh-CN.md","accepted_V2_return","L5-L19, L38-L69, L71-L138, L150-L158","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E074","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/qa/FINAL_ACCEPTANCE.json","accepted_V2_QA","corrected split 78 attempts; 164 total; support and exposure","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E075","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/ABSOLUTE_LOSSES.csv","accepted_per_task_seed_metrics","accepted V2 task/model/seed co-primary loss rows","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E076","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/ALL_SEED_D_V_I.csv","accepted_per_seed_contrasts","source rows by task_id,seed,metric,contrast","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E077","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/SUMMARY_STATISTICS.csv","accepted_macro_summary","macro/task mean, SD, median; session independence field","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E078","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/ALL_SEED_LOO.csv","accepted_task_LOO_source","header omitted_task_id; verify deleted unit","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E079","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/CONDITIONAL_CI.csv","accepted_conditional_intervals","five-task per-seed group-bootstrap intervals","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E080","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/WAYPOINT.csv","accepted_waypoint_contrasts","task119 AUX_ONLY, GOAL_COND and baseline contrasts per seed","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E081","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/WAYPOINT_ENDPOINT.csv","accepted_endpoint_metrics","task119 endpoint errors and 12-group support","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),
("E082","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/code/score_v2_results.py","locked_effect_definitions","L293-L313, L477-L485; D/V/I and waypoint formulas","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256",False),

("E083","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/FINAL_ACCEPTANCE.json","accepted_VideoMAE_acceptance","acceptance.artifacts/replay/core_b_metrics/portable_integration; partial","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CHECKSUMS.sha256",False),
("E084","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/AUDIT_REPORT.zh-CN.md","accepted_VideoMAE_report","L35-L55; model means and contrast signs","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CHECKSUMS.sha256",False),
("E085","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CORE_B_PAIRED_CONTRASTS.csv","accepted_VideoMAE_contrasts","filtered by split=dev, region=action_full28, horizon, metric, contrast","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CHECKSUMS.sha256",False),
("E086","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CORE_B_PER_SEED_METRICS.csv","accepted_VideoMAE_per_seed_metrics","source rows per model/horizon/seed/full28/co-primary","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CHECKSUMS.sha256",False),
("E087","reports/benchmarks/VIDEOMAE_CORE_B_HANDOFF_V2_20261013T065841Z/replay/cpu_zero_20261013T065954Z/REPLAY_SUMMARY.json","recorded_replay_attempt","12 zero-input CPU replay cells; exact zero tolerance; FAIL_REPLAY_TOLERANCE","reports/benchmarks/VIDEOMAE_CORE_B_HANDOFF_V2_20261013T065841Z/CHECKSUMS.sha256",False),

("E088","reports/deliverables/CORE_B_COMMAND_QUERY_FINAL_HANDOFF_20261013T105945Z/EXECUTIVE_SUMMARY.zh-CN.md","accepted_query_handoff","L8-L23; line closed, state-dependent video, data readiness","reports/deliverables/CORE_B_COMMAND_QUERY_FINAL_HANDOFF_20261013T105945Z/CHECKSUMS.sha256",False),
("E089","reports/deliverables/CORE_B_COMMAND_QUERY_FINAL_HANDOFF_20261013T105945Z/FINAL_STATUS.json","accepted_query_handoff_status","N2/N3/N4/N5; query closed; independent confirmation false","reports/deliverables/CORE_B_COMMAND_QUERY_FINAL_HANDOFF_20261013T105945Z/CHECKSUMS.sha256",False),

("E090","reports/core_b/ROUTE_C_VIDEO_ATTRIBUTION_CONTROLS_V1_20260910T154453Z/recovery_canonical_replay_20260911T054907Z/FIRST_DIVERGENCE_REPORT.md","routeC_recovery_record","first-divergence and replay-only recovery; not a completed full matrix","",False),
("E091","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z/FINAL_STATUS.ini","history_video_status","complete; single-seed; candidate/FD limits","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z/CHECKSUMS.sha256",False),
("E092","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z/analysis/MAIN_RESULTS.csv","history_video_results","locked information matrix conclusions","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z/CHECKSUMS.sha256",False),
("E093","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z/FINAL_STATUS.ini","history_encoder_status","A accepted 12/12; B not triggered; no adaptation candidate","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z/CHECKSUMS.sha256",False),
("E094","research_archive/CORE_B_ADAPTATION_HARVEST_20260919T053154Z/RETURN_TO_CHATGPT.zh-CN.md","adaptation_harvest_return","training 6/6; prediction 0/6; not assessed","research_archive/CORE_B_ADAPTATION_HARVEST_20260919T053154Z/CHECKSUMS.sha256",False),
("E095","research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z/RETURN_TO_CHATGPT.zh-CN.md","adaptation_inference_recovery","6 dev predictions accepted; candidate gate fail","research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z/CHECKSUMS.sha256",False),
("E096","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/FINAL_STATUS.ini","event_fallback_status","single exploratory seed; candidate gate fail; event support/source denominator","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/CHECKSUMS.sha256",False),
("E097","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z/FINAL_STATUS.ini","prototype_calibration_status","alpha 0; calibration gate fail; one seed","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z/CHECKSUMS.sha256",False),
("E098","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z/FINAL_STATUS.ini","selector_status","four cells CPU-only; candidate gate fail","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z/CHECKSUMS.sha256",False),
("E099","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z/FINAL_STATUS.ini","dct_dual_loss_status","four cells; one seed; candidate gate fail; TV ratio small","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z/CHECKSUMS.sha256",False),
("E100","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z/analysis/SELECTOR_MAIN_RESULTS.csv","selector_summary","accepted selector metrics on development","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z/CHECKSUMS.sha256",False),
("E101","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z/analysis/MAIN_RESULTS.csv","dct_summary","accepted factorial results and gate","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z/CHECKSUMS.sha256",False),
("E102","repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z/RETURN_TO_CHATGPT.zh-CN.md","observation_delay_return","three-seed delay comparison; dev only","repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z/CHECKSUMS.sha256",False),
("E103","repo/ee6008-humanoid-vjepa/reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/STAGE2_RUN_ACCEPTANCE_POST_AUTHORIZATION.json","jy02_stage2_acceptance","formal stage2 cells after explicit protocol authorization","repo/ee6008-humanoid-vjepa/reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/CHECKSUMS.sha256",False),
("E104","repo/ee6008-humanoid-vjepa/reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/STAGE2_RESULTS_BY_SEED.json","jy02_stage2_results","accepted development seed/cell results","repo/ee6008-humanoid-vjepa/reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/CHECKSUMS.sha256",False),
("E105","repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/analysis/INFORMATION_INTERACTIONS.csv","v3_state_video_interactions","accepted 54-cell development matrix interaction summary","repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/CHECKSUMS.sha256",False),
("E106","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/coordination/PHASE_STATUS_SCREENING_READOUT.json","current_V3_phase_status","snapshot phase metadata only; working root; live-process state not queried","",True),
("E107","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/coordination/PHASE_STATUS_FORMAL_FEATURES.json","current_V3_feature_phase","snapshot phase metadata only","",True),
("E108","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/coordination/PHASE_STATUS_FORMAL_READOUT.json","current_V3_readout_phase","snapshot phase metadata only","",True),
("E109","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/coordination/PHASE_STATUS_MAIN_CPU_PREDICTIONS.json","current_V3_prediction_phase","snapshot phase metadata only","",True),
("E110","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/coordination/PHASE_STATUS_RGB_PROBE.json","current_V3_RGB_probe","snapshot phase metadata only; phase started 2026-10-14T06:10:41Z at prior snapshot","",True),
("E111","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/qa/MAIN_PREDICTION_ACCEPTANCE_PRESEAL.json","current_V3_preseal_acceptance","preseal only; not a final sealed scientific acceptance","",True),
]

def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def val(x):
    if x is None:return "NA"
    if isinstance(x,float):
        if not math.isfinite(x):return "NA"
        return format(x,".15g")
    return x
def read_text(rel):
    p=PROJECT/rel
    return p.read_text(encoding="utf-8",errors="replace")
def read_json(rel):
    return json.loads(read_text(rel))
def read_csv(rel):
    with (PROJECT/rel).open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write_json(rel,obj):
    p=AUDIT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def write_text(rel,text):
    p=AUDIT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding="utf-8")
def write_csv(rel,header,rows):
    p=AUDIT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=header,extrasaction="ignore",lineterminator="\n")
        w.writeheader()
        for row in rows:w.writerow({k:val(row.get(k)) for k in header})
def checksum_expected(spec,path):
    mrel=spec[4]
    if not mrel:return None,None,"no_manifest_declared"
    mp=PROJECT/mrel
    if not mp.is_file():return mrel,None,"manifest_missing"
    try:
        if mp.suffix.lower()==".csv":
            with mp.open(newline="",encoding="utf-8-sig") as f:
                rows=list(csv.DictReader(f))
            for row in rows:
                pval=next((row[k] for k in row if k.lower() in {"path","relative_path","file","filename","name"}),None)
                hval=next((row[k] for k in row if "sha256" in k.lower() or k.lower()=="sha"),None)
                if not pval or not hval:continue
                pval=pval.replace("\\","/").lstrip("./")
                candidates={path.relative_to(mp.parent).as_posix(),path.name}
                if pval in candidates or pval.endswith("/"+path.name):
                    return mrel,hval.strip(),"manifest_entry_found"
        else:
            for line in mp.read_text(encoding="utf-8",errors="replace").splitlines():
                parts=line.strip().split(maxsplit=1)
                if len(parts)!=2:continue
                h,pval=parts;pval=pval.lstrip("*").replace("\\","/").lstrip("./")
                candidates={path.relative_to(mp.parent).as_posix(),path.name}
                if pval in candidates or pval.endswith("/"+path.name):
                    return mrel,h.strip(),"manifest_entry_found"
    except Exception as e:
        return mrel,None,"manifest_parse_error:"+type(e).__name__
    return mrel,None,"manifest_entry_not_found_or_self_excluded"

def snapshot_sources():
    rows=[]
    for sid,rel,role,locator,manifest,live in SPECS:
        p=PROJECT/rel
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError("forbidden source type in scope: "+rel)
        if not p.is_file():
            rows.append({"source_id":sid,"project_relative_path":rel,"role":role,"locator":locator,"exists":False,"size_bytes":"NA","mtime_utc":"NA","sha256":"NA","manifest_path":manifest,"manifest_expected_sha256":"NA","manifest_match":"not_located","stability":"not_located","live_source":live})
            continue
        st0=p.stat(); h=sha(p); st1=p.stat()
        mrel,expected,bind=checksum_expected((sid,rel,role,locator,manifest,live),p)
        match = "NA"
        if expected:
            match = "true" if expected==h else "false"
        stable=st0.st_size==st1.st_size and st0.st_mtime_ns==st1.st_mtime_ns
        rows.append({"source_id":sid,"project_relative_path":rel,"role":role,"locator":locator,"exists":True,"size_bytes":st1.st_size,"mtime_utc":datetime.fromtimestamp(st1.st_mtime,timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z"),"sha256":h,"manifest_path":mrel or "NA","manifest_expected_sha256":expected or "NA","manifest_match":match,"manifest_lookup":bind,"stability":"stable_snapshot" if stable else "unstable_during_hash","live_source":live})
    return rows

def recheck_sources(rows):
    changes=[]
    by={r["source_id"]:r for r in rows}
    for spec in SPECS:
        sid,rel,role,locator,manifest,live=spec
        p=PROJECT/rel
        row=by[sid]
        if not p.is_file() or not row.get("exists"):
            continue
        h2=sha(p);st=p.stat()
        stable=(h2==row["sha256"] and st.st_size==int(row["size_bytes"]))
        row["recheck_sha256"]=h2
        row["recheck_mtime_utc"]=datetime.fromtimestamp(st.st_mtime,timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
        row["stability"]="stable_through_audit_reads" if stable and row["stability"]=="stable_snapshot" else ("unstable_live_source" if live else "unstable_live_source" if live else "unstable_selected_source")
        if not stable:
            changes.append({"source_id":sid,"path":rel,"initial_sha256":row["sha256"],"final_sha256":h2,"live_source":live,"status":"unstable_live_source" if live else "unstable_selected_source"})
    return changes

BUILD_MONO=time.monotonic()
BUILD_CPU=time.process_time()
BUILD_START_UTC=utc_now()

# selected source catalog / hashes
SRC_ROWS=snapshot_sources()
SRC={r["source_id"]:r for r in SRC_ROWS}
def source_path(sid):return SRC[sid]["project_relative_path"]
def source_hash(sid):return SRC[sid]["sha256"]
def src_refs(ids):return ";".join(ids)
def exact_ref(sid,extra=""):
    r=SRC[sid]
    return r["project_relative_path"]+((";"+extra) if extra else "")

KEY_FIELDS=["comparison_id","claim_id","family","contract_id","cohort_hash","split","task_scope","horizon","metric","metric_unit","metric_definition_hash","aggregation","control","candidate","control_mean","candidate_mean","gain_abs","gain_relative_pct","training_seed_count","seed_gain_sd","seed_gain_median","n_positive_seeds","n_valid_seeds","n_source_groups","n_acquisition_clusters","n_episodes","n_windows","interval_scope","interval_lower","interval_upper","interval_method","comparison_status","evidence_status","row_origin","source_file","source_sha256","source_locator","evidence_ids","caveat"]
PER_FIELDS=["comparison_id","claim_id","seed","control","candidate","control_value","candidate_value","gain","metric","scope","source_file","source_sha256","source_locator","evidence_ids","row_origin","caveat"]
SEED_LOO_FIELDS=["comparison_id","dropped_seed","remaining_seed_count","gain_mean","scope","source_file","source_locator","evidence_ids","row_origin","caveat"]
TASK_LOO_FIELDS=["comparison_id","dropped_task","remaining_task_count","gain_mean","weighting","scope","source_file","source_locator","evidence_ids","row_origin","caveat"]

key_rows=[];per_rows=[];seed_loo_rows=[];task_loo_rows=[]
derived_ids=set()
def add_key(row,derived=False):
    row.setdefault("row_origin","derived_from_accepted_tables" if derived else "copied_accepted")
    if derived:derived_ids.add("key:"+str(row["comparison_id"])+":"+str(row.get("interval_scope","point")))
    key_rows.append(row)
def add_per(row,derived=False):
    row.setdefault("row_origin","derived_from_accepted_tables" if derived else "copied_accepted")
    if derived:derived_ids.add("per:"+str(row["comparison_id"])+":"+str(row.get("seed")))
    per_rows.append(row)
def add_seed_loo(row,derived=False):
    row.setdefault("row_origin","derived_from_accepted_tables" if derived else "copied_accepted")
    if derived:derived_ids.add("seedloo:"+str(row["comparison_id"])+":"+str(row.get("dropped_seed")))
    seed_loo_rows.append(row)
def add_task_loo(row,derived=False):
    row.setdefault("row_origin","derived_from_accepted_tables" if derived else "copied_accepted")
    if derived:derived_ids.add("taskloo:"+str(row["comparison_id"])+":"+str(row.get("dropped_task")))
    task_loo_rows.append(row)
def add_comp_rows(base, intervals, source_ids, locators, derived_point=False):
    # Each accepted interval is a separate row. The point result is counted once.
    if not intervals:intervals=[{"scope":"NA_NO_ACCEPTED_INTERVAL","lower":None,"upper":None,"method":"NA"}]
    for i,it in enumerate(intervals):
        r=dict(base)
        r.update({"interval_scope":it["scope"],"interval_lower":it.get("lower"),"interval_upper":it.get("upper"),"interval_method":it.get("method","NA"),
                  "source_file":source_path(source_ids[0]),"source_sha256":source_hash(source_ids[0]),"source_locator":locators[0],"evidence_ids":src_refs(source_ids),
                  "row_origin":"derived_from_accepted_tables" if derived_point and i==0 else "copied_accepted"})
        if derived_point and i==0:derived_ids.add("key:"+str(base["comparison_id"])+":point")
        key_rows.append(r)
def getfloat(d,k):
    try:
        v=d.get(k)
        if v in (None,"","NA","nan"):return None
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception:return None
def avg(xs):
    xs=[x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None
def sd(xs):
    xs=[x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs)>1 else None
def median(xs):
    xs=[x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
def rel_pct(g,c):
    return g/c*100 if g is not None and c not in (None,0) else None

# 1. Core A Batch48: matched per-seed score differences.
b48_base="repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/"
agg={r["variant"]:r for r in read_csv(b48_base+"STANDARD_BATCH48_AGGREGATE.csv")}
b48_seeds=read_csv(b48_base+"STANDARD_BATCH48_PER_SEED.csv")
bs=[]
for metric,col in [("top1","top1"),("macro_f1","macro_f1")]:
    control=float(agg["static_repeated_first_frame"][col+"_mean"])
    candidate=float(agg["dynamic"][col+"_mean"])
    effects={}
    for r in b48_seeds:
        if r["variant"]=="dynamic":effects[r["seed"]]=float(r[col])
        elif r["variant"]=="static_repeated_first_frame":effects[r["seed"]]=effects.get(r["seed"],{}) if isinstance(effects.get(r["seed"]),dict) else {}
    # pair by seed with stable separate maps
    dyn={r["seed"]:float(r[col]) for r in b48_seeds if r["variant"]=="dynamic"}
    sta={r["seed"]:float(r[col]) for r in b48_seeds if r["variant"]=="static_repeated_first_frame"}
    gains={s:dyn[s]-sta[s] for s in sorted(set(dyn)&set(sta))}
    comp="COREA_B48_"+metric.upper()
    for seed,g in gains.items():
        add_per({"comparison_id":comp,"claim_id":"C01","seed":seed,"control":"static_repeated_first_frame","candidate":"dynamic","control_value":sta[seed],"candidate_value":dyn[seed],"gain":g,"metric":metric,"scope":"617 development-validation episodes; matched training seeds","source_file":source_path("E003"),"source_sha256":source_hash("E003"),"source_locator":"STANDARD_BATCH48_PER_SEED.csv joined on seed; dynamic minus static","evidence_ids":"E002;E003;E005","caveat":"classification result; not action anticipation"},True)
    intervals=[]
    for r in read_csv(b48_base+"BOOTSTRAP_RESULTS.csv"):
        if r.get("pair")=="B48_dynamic_minus_B48_static" and r.get("metric")==metric:
            intervals.append({"scope":r["method"]+"_bootstrap","lower":getfloat(r,"ci95_lower"),"upper":getfloat(r,"ci95_upper"),"method":f'{r.get("replicates")} draws; seed {r.get("rng_seed")}; episode/class strata per accepted report'})
    if metric=="macro_f1":
        intervals=[{"scope":"class_stratified","lower":-0.005615105231602986,"upper":0.011374233663082116,"method":"10,000 draws; accepted full precision BOOTSTRAP_RESULTS.csv"},
                   {"scope":"hierarchical","lower":-0.0257651021083196,"upper":0.027364460627237953,"method":"20,000 draws; accepted full precision BOOTSTRAP_RESULTS.csv"}]
    g=candidate-control
    base={"comparison_id":comp,"claim_id":"C01","family":"CoreA Batch48 frozen classifier dynamic/static","contract_id":"CA_BATCH48_MATCHED","cohort_hash":"NA_HASH_NOT_PRESENT_IN_SELECTED_METRIC_TABLE","split":"development_validation","task_scope":"120-class recognition","horizon":"classification window; no trajectory horizon","metric":metric,"metric_unit":"fraction score","metric_definition_hash":"metric contract cited in CORE_A report; source hash tracked in E006","aggregation":"mean across 3 selected-best seed metrics; paired on seed; 617 episodes per seed","control":"static_repeated_first_frame","candidate":"dynamic","control_mean":float(control),"candidate_mean":float(candidate),"gain_abs":g,"gain_relative_pct":rel_pct(g,float(control)),"training_seed_count":3,"seed_gain_sd":sd(list(gains.values())),"seed_gain_median":median(list(gains.values())),"n_positive_seeds":sum(x>0 for x in gains.values()),"n_valid_seeds":len(gains),"n_source_groups":None,"n_acquisition_clusters":None,"n_episodes":617,"n_windows":617,"comparison_status":"matched_accepted","evidence_status":"accepted_metric_and_bootstrap_tables; dev/validation exposed","caveat":"positive Top-1 is modest; macro-F1 interval crosses zero; no action-anticipation inference"}
    add_comp_rows(base,intervals,["E002","E003","E004","E005"],["aggregate rows dynamic/static; per-seed rows joined by seed; BOOTSTRAP_RESULTS pair B48_dynamic_minus_B48_static"],True)
    # Paired seed LOO.
    for dropped in gains:
        remain=[g for s,g in gains.items() if s!=dropped]
        add_seed_loo({"comparison_id":comp,"dropped_seed":dropped,"remaining_seed_count":len(remain),"gain_mean":avg(remain),"scope":"matched dynamic-static score difference; fixed 617 dev/validation episodes","source_file":source_path("E003"),"source_locator":"derived from rows joined by seed","evidence_ids":"E003","caveat":"leave-one-training-seed-out; not independent confirmation"},True)

# Core A Batch48 vs Batch32 macro-F1, one accepted historical comparison.
b48_dec=read_json(b48_base+"CORE_A_BATCH48_DECISION.json")
dynamic=batch32={}
# accepted baseline aggregate rows from unified CoreA analysis
corea_base="research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/"
bsz=read_csv(corea_base+"BATCH_SIZE_EFFECTS.csv")
for r in bsz:
    if r["variant"]=="dynamic" and r["batch_size"]=="32":batch32=r
c48=agg["dynamic"]
g=float(b48_dec["evidence"]["batch48_vs_batch32_dynamic"]["point_macro_f1"])
lo=float(b48_dec["evidence"]["batch48_vs_batch32_dynamic"]["hierarchical_macro_f1_ci"][0])
hi=float(b48_dec["evidence"]["batch48_vs_batch32_dynamic"]["hierarchical_macro_f1_ci"][1])
add_comp_rows({"comparison_id":"COREA_B48_VS_B32_DYNAMIC_MACRO_F1","claim_id":"C01","family":"CoreA batch-size rescue","contract_id":"CA_BATCH48_VS_BATCH32_DIFFERENT_BATCH","cohort_hash":"development_validation_617; paired episode IDs for B48, pooled older source not used","split":"development_validation","task_scope":"120-class recognition","horizon":"classification","metric":"macro_f1","metric_unit":"fraction score","metric_definition_hash":"CORE_A macro-F1 definition; source E006","aggregation":"difference of accepted selected-best batch-size means","control":"B32_dynamic","candidate":"B48_dynamic","control_mean":float(batch32["macro_f1_mean"]),"candidate_mean":float(c48["macro_f1_mean"]),"gain_abs":g,"gain_relative_pct":rel_pct(g,float(batch32["macro_f1_mean"])),"training_seed_count":3,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":None,"n_valid_seeds":3,"n_source_groups":None,"n_acquisition_clusters":None,"n_episodes":617,"n_windows":617,"comparison_status":"comparable_with_caveats","evidence_status":"accepted point and hierarchical interval; interval rounded in decision report","row_origin":"copied_accepted","source_file":source_path("E005"),"source_sha256":source_hash("E005"),"source_locator":"evidence.batch48_vs_batch32_dynamic","evidence_ids":"E001;E005;E006","caveat":"95% hierarchical interval crosses zero; not a strong Batch48 rescue"},[{"scope":"hierarchical_bootstrap","lower":lo,"upper":hi,"method":"accepted 95% hierarchical interval"}],["E005","E002","E006"],["evidence.batch48_vs_batch32_dynamic; aggregate rows by batch size"],True)

# Core A Full120 matched single-seed LoRA comparisons; effect/intervals copied from accepted table.
pa=read_csv(corea_base+"CORE_A_PAIRWISE_COMPARISONS.csv")
corea_specs=[
("D-P-L_vs_D-P-F","dynamic pooled LoRA vs frozen","D-P-F","D-P-L"),
("S-P-L_vs_S-P-F","static pooled LoRA vs frozen","S-P-F","S-P-L"),
]
for cid,label,control,candidate in corea_specs:
    rr=next(x for x in pa if x["comparison_id"]==cid)
    for metric,field,gaincol,cilow,cihigh in [
        ("top1","top1","top1_delta","top1_bootstrap_ci95_low","top1_bootstrap_ci95_high"),
        ("macro_f1","macro_f1","macro_f1_delta","macro_f1_bootstrap_ci95_low","macro_f1_bootstrap_ci95_high")]:
        c=float(rr["reference_"+field]);v=float(rr["candidate_"+field]);g=float(rr[gaincol])
        add_comp_rows({"comparison_id":"COREA_FULL120_"+cid+"_"+metric.upper(),"claim_id":"C01","family":"CoreA Full120 LoRA matched cell","contract_id":"CA_FULL120_LORA_SINGLE_SEED","cohort_hash":"same accepted 617-row dev/validation stream within comparison; hash recorded in source manifest","split":"development_validation","task_scope":"120-class recognition","horizon":"classification","metric":metric,"metric_unit":"fraction score","metric_definition_hash":"CORE_A metric contract; source E006","aggregation":"single accepted seed; episode-paired bootstrap","control":control,"candidate":candidate,"control_mean":c,"candidate_mean":v,"gain_abs":g,"gain_relative_pct":rel_pct(g,c),"training_seed_count":1,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":None,"n_valid_seeds":1,"n_source_groups":None,"n_acquisition_clusters":None,"n_episodes":617,"n_windows":617,"comparison_status":"matched_accepted","evidence_status":"accepted Full120 development analysis; single training seed","row_origin":"copied_accepted","source_file":source_path("E007"),"source_sha256":source_hash("E007"),"source_locator":"row comparison_id="+cid+"; "+gaincol,"evidence_ids":"E006;E007;E008;E009","caveat":"validation selected checkpoint; dynamic-specific attribution requires interaction and does not follow from this matched LoRA effect"},
                  [{"scope":"paired_episode_bootstrap_95","lower":float(rr[cilow]),"upper":float(rr[cihigh]),"method":"accepted paired episode bootstrap; single fixed training seed"}],["E007"],[f"row comparison_id={cid}; accepted pairwise table"],False)

# B1 persistence accepted negative diagnostic.
b1rows=read_csv("research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8/B1_RESULTS_INDEPENDENT.csv")
bstate=next(r for r in b1rows if r["cell"]=="B-STATE")
persist=next(r for r in b1rows if r["cell"]=="PERSISTENCE")
b1gain=float(persist["std_arm_mae"])-float(bstate["std_arm_mae"])
add_comp_rows({"comparison_id":"CB_B1_PERSISTENCE_VS_STATE_ARM14","claim_id":"C02","family":"CoreB B1 persistence diagnostic","contract_id":"CB_B1_OLD_614_H30","cohort_hash":"accepted B1 canonical N=614; not the later N2 603-source-group population","split":"development","task_scope":"upper-body command; arm14 raw slice; units standardized","horizon":"30 steps / 1 second per accepted B1 contract","metric":"std_arm_mae","metric_unit":"train-normalized command units","metric_definition_hash":"B1 locked metric; source E016","aggregation":"single accepted training seed; episode-equal","control":"persistence","candidate":"B-STATE","control_mean":float(persist["std_arm_mae"]),"candidate_mean":float(bstate["std_arm_mae"]),"gain_abs":b1gain,"gain_relative_pct":rel_pct(b1gain,float(persist["std_arm_mae"])),"training_seed_count":1,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":0,"n_valid_seeds":1,"n_source_groups":None,"n_acquisition_clusters":None,"n_episodes":614,"n_windows":614,"comparison_status":"matched_accepted","evidence_status":"accepted negative/diagnostic baseline","row_origin":"derived_from_accepted_tables","source_file":source_path("E016"),"source_sha256":source_hash("E016"),"source_locator":"rows stage=B1, cell=PERSISTENCE and B-STATE, column std_arm_mae","evidence_ids":"E011;E012;E013;E016","caveat":"B1 is a different cohort/contract from later N2; standardized values are not physical accuracy"},[],["E016"],["rows stage=B1, cell=PERSISTENCE and B-STATE, column std_arm_mae"],True)

# 2. N3 direct-reference history/state effects: derive fixed-seed point contrasts only.
n3_path="reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/n3_refs/RESULTS_BY_SEED.csv"
n3_rows=read_csv(n3_path)
for state in ["FULL","OBS","ANCHOR"]:
    selected=[r for r in n3_rows if r["horizon_steps"]=="30" and r["state_level"]==state and r["video_level"] in ["N","D"]]
    by_seed={}
    for r in selected:
        by_seed.setdefault(r["seed"],{})[r["video_level"]]=float(r["full_mae_std"])
    gains={s:v["N"]-v["D"] for s,v in by_seed.items() if "N" in v and "D" in v}
    controls=[v["N"] for v in by_seed.values() if "N" in v]
    candidates=[v["D"] for v in by_seed.values() if "D" in v]
    comp="N3_H30_"+state+"_DCHRON_VS_NVIDEO_FULL28"
    for seed,g in sorted(gains.items()):
        add_per({"comparison_id":comp,"claim_id":"C03","seed":seed,"control":"N-VIDEO","candidate":"D-CHRON","control_value":by_seed[seed]["N"],"candidate_value":by_seed[seed]["D"],"gain":g,"metric":"full_mae_std","scope":"N3 direct-reused H30; development only; 603 source groups","source_file":source_path("E045"),"source_sha256":source_hash("E045"),"source_locator":f"rows seed={seed},horizon=30,state_level={state},video_level=N/D","evidence_ids":"E039;E044;E045;E046","caveat":"ANCHOR retains the current-command/anchor branch; not strict video-only"},True)
    gainvals=list(gains.values())
    add_comp_rows({"comparison_id":comp,"claim_id":"C03","family":"N3 state/history by video information","contract_id":"CB_N3_DIRECT_REUSE_H30","cohort_hash":"603 accepted source groups; exact ordered IDs bound by N3 input identity","split":"development","task_scope":state+" Full28","horizon":"H30","metric":"full_mae_std","metric_unit":"train-standardized command units","metric_definition_hash":"N3 locked position metric","aggregation":"N minus D error per seed; equal seed point mean; group-equal inputs","control":"N-VIDEO","candidate":"D-CHRON","control_mean":avg(controls),"candidate_mean":avg(candidates),"gain_abs":avg(gainvals),"gain_relative_pct":rel_pct(avg(gainvals),avg(controls)),"training_seed_count":len(gains),"seed_gain_sd":sd(gainvals),"seed_gain_median":median(gainvals),"n_positive_seeds":sum(x>0 for x in gainvals),"n_valid_seeds":len(gains),"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"comparable_with_caveats","evidence_status":"accepted N3 reused H30 rows; not a new replication","caveat":"descriptive three-seed contrast; ANCHOR includes current command/anchor branch"},[],["E045","E044"],[f"derived from per-seed rows state_level={state}, video_level N and D"],True)
    for dropped in sorted(gains):
        remain=[v for s,v in gains.items() if s!=dropped]
        add_seed_loo({"comparison_id":comp,"dropped_seed":dropped,"remaining_seed_count":len(remain),"gain_mean":avg(remain),"scope":"fixed N3 H30 state-level full28; equal seed means","source_file":source_path("E045"),"source_locator":f"per-seed contrast; dropped {dropped}","evidence_ids":"E045","caveat":"leave-one-training-seed-out; no group resampling"},True)

# 3. D3 corrected scheduler: use accepted per-seed/group summary output; old objective source remains distinct.
d3sum=read_csv("reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/analysis/SCHEDULE_GAINS_SUMMARY.csv")
d3vid=read_csv("reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/analysis/VIDEO_SPECIFIC_GAIN_SUMMARY.csv")
def d3row(rows,split,config,horizon,metric,region="action_full28"):
    return next(r for r in rows if r.get("split")==split and r.get("config")==config and r.get("horizon_steps")==str(horizon) and r.get("metric")==metric and r.get("region_raw")==region)
for config in ["TS11-D","STATE-CTRL"]:
    r=d3row(d3sum,"dev",config,30,"standardized_mae")
    comp="D3EQ_H30_"+config.replace("-","_")+"_SCHEDULER_POSITION"
    effect=float(r["estimate"]);lo=float(r["ci95_low"]);hi=float(r["ci95_high"])
    # Reported gain = old as-run error minus corrected-scheduler error.
    seeds=json.loads(r["seed_points"])
    for seed,g in seeds.items():
        add_per({"comparison_id":comp,"claim_id":"C04","seed":seed,"control":"old_as_run_checkpoint","candidate":"corrected_scheduler_checkpoint","control_value":None,"candidate_value":None,"gain":g,"metric":"standardized_mae","scope":"603 dev source groups; train/dev model comparison; fixed seed","source_file":source_path("E025"),"source_sha256":source_hash("E025"),"source_locator":f"SCHEDULE_GAINS_SUMMARY.csv config={config},H30,dev,action_full28,standardized_mae,seed_points","evidence_ids":"E023;E024;E026;E027","row_origin":"copied_accepted","caveat":"accepted seed-level gain; old and corrected trainer contracts differ"})
    add_comp_rows({"comparison_id":comp,"claim_id":"C04","family":"D3 corrected scheduler","contract_id":"CB_D3_EQ_SCHEDULER_H30","cohort_hash":"train 2712 groups / dev 603 groups; same locked population","split":"development","task_scope":"Full28","horizon":"H30","metric":"standardized_mae","metric_unit":"train-standardized command units","metric_definition_hash":"D3 locked Full28 position MAE","aggregation":"three fixed seeds then 603 source-group bootstrap","control":"old_as_run_checkpoint","candidate":"corrected_scheduler_checkpoint","control_mean":None,"candidate_mean":None,"gain_abs":effect,"gain_relative_pct":None,"training_seed_count":3,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":sum(float(x)>0 for x in seeds.values()),"n_valid_seeds":len(seeds),"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"matched_accepted","evidence_status":"corrected implementation accepted; science result is development-only","caveat":"positive means corrected error lower; TS11-D worsened, STATE-CTRL improved"},[{"scope":"fixed-seed source-group bootstrap","lower":lo,"upper":hi,"method":"10,000 source-group draws; point estimate fixed three seeds"}],["E026","E027","E028"],[f"SCHEDULE_GAINS_SUMMARY.csv config={config},H30,dev,action_full28,standardized_mae"],False)

r=next(x for x in d3vid if x["split"]=="dev" and x["horizon_steps"]=="30" and x["region_raw"]=="action_full28" and x["metric"]=="standardized_mae")
# Video-specific interaction is TS11-D scheduler effect minus STATE-CTRL scheduler effect.
interaction=float(r["estimate"]);lo=float(r["ci95_low"]);hi=float(r["ci95_high"])
d3seeds=json.loads(r["seed_points"])
comp="D3EQ_H30_VIDEO_SPECIFIC_SCHEDULER_INTERACTION"
add_comp_rows({"comparison_id":comp,"claim_id":"C04","family":"D3 corrected-scheduler video interaction","contract_id":"CB_D3_EQ_SCHEDULER_H30","cohort_hash":"603 accepted dev source groups","split":"development","task_scope":"Full28; TS11-D minus STATE-CTRL schedule effect","horizon":"H30","metric":"standardized_mae_interaction","metric_unit":"train-standardized command units","metric_definition_hash":"locked TS11-D scheduler gain minus STATE-CTRL scheduler gain","aggregation":"three fixed seeds; equal group means","control":"STATE-CTRL scheduler gain","candidate":"TS11-D scheduler gain","control_mean":None,"candidate_mean":None,"gain_abs":interaction,"gain_relative_pct":None,"training_seed_count":3,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":sum(float(x)>0 for x in d3seeds.values()),"n_valid_seeds":len(d3seeds),"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"matched_accepted","evidence_status":"accepted secondary interaction; development only","caveat":"negative interaction favors STATE-CTRL scheduler response; not a video causal effect"},
    [{"scope":"fixed-seed source-group bootstrap","lower":lo,"upper":hi,"method":"10,000 source-group draws; accepted secondary comparison"}],["E027","E028"],["VIDEO_SPECIFIC_GAIN_SUMMARY.csv dev,H30,action_full28,standardized_mae"],False)

# 4. N2/N5 query effects. Point intervals and seed/LOO statistics are accepted; absolute per-seed cells come from accepted CUDA_COMMON group means.
n2rows=read_csv("reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/results/N2_BY_SEED.csv")
cells={}
for r in n2rows:
    if r["profile"]=="CUDA_COMMON" and r["metric"] in ["position_standardized_mae","safe_fd_normalized"]:
        cells[(r["model"],r["query"],r["seed"],r["metric"])]=float(r["group_mean"])
scope_table=read_csv("reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/results/CO_PRIMARY_SCOPE_TABLE30.csv")
n5_byseed=read_csv("reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/results/BY_SEED_AND_LOO.csv")
n2contrasts=["G_TS","G_STATE","V_Q0","V_Q1","I"]
n2metrics=["position_standardized_mae","safe_fd_normalized"]
n2seeds=["20261001","20261002","20261003"]
def n2_component(metric,contrast,seed):
    tsq0=cells[("TS11-D","Q0",seed,metric)];tsq1=cells[("TS11-D","Q1",seed,metric)]
    stq0=cells[("STATE-CTRL","Q0",seed,metric)];stq1=cells[("STATE-CTRL","Q1",seed,metric)]
    return {"G_TS":(tsq0,tsq1,tsq0-tsq1),"G_STATE":(stq0,stq1,stq0-stq1),
            "V_Q0":(stq0,tsq0,stq0-tsq0),"V_Q1":(stq1,tsq1,stq1-tsq1),
            "I":(None,None,(tsq0-tsq1)-(stq0-stq1))}[contrast]
for metric in n2metrics:
 for contrast in n2contrasts:
    comp="N2_"+metric.upper()+"_"+contrast
    seedvals=[]; ctrlvals=[]; candvals=[]
    for seed in n2seeds:
        cv,nv,g=n2_component(metric,contrast,seed)
        seedvals.append(g)
        if cv is not None:ctrlvals.append(cv);candvals.append(nv)
    accepted_seed_rows=[r for r in n5_byseed if r["metric"]==metric and r["contrast"]==contrast and str(r.get("seed")) in n2seeds]
    source_seeds={str(r["seed"]):float(r["seed_estimate"]) for r in accepted_seed_rows}
    # Audit a permitted per-seed table identity; backend arithmetic drift should remain tiny and be disclosed.
    mismatch=max([abs(source_seeds[s]-seedvals[i]) for i,s in enumerate(n2seeds)] or [0.0])
    point=next(float(r["point_estimate_fixed_three_seed_mean"]) for r in scope_table if r["metric"]==metric and r["contrast"]==contrast and r["interval_scope"]=="GROUP_ONLY_FIXED_SEEDS")
    intervals=[]
    for r in scope_table:
        if r["metric"]==metric and r["contrast"]==contrast:
            intervals.append({"scope":r["interval_scope"],"lower":float(r["ci95_lower"]),"upper":float(r["ci95_upper"]),"method":r["interval_definition"]+"; "+r["quantile_method"]})
    control=avg(ctrlvals) if ctrlvals else None
    candidate=avg(candvals) if candvals else None
    positive=sum(x>0 for x in source_seeds.values())
    add_comp_rows({"comparison_id":comp,"claim_id":"C09","family":"N2 Hadamard query contrast","contract_id":"CB_N2_QUERY_PROFILE_CUDA_COMMON","cohort_hash":"603 ordered development source groups; N2 input identity locked","split":"development_previously_exposed","task_scope":"TS11-D/STATE-CTRL, Full28; Q0/Q1 query control","horizon":"H30","metric":metric,"metric_unit":"train-normalized error (standardized MAE or safe-FD normalized)","metric_definition_hash":"N2/N5 co-primary source lock","aggregation":"fixed 3-seed mean of 603 source-group-equal model losses","control":contrast,"candidate":contrast,"control_mean":control,"candidate_mean":candidate,"gain_abs":point,"gain_relative_pct":rel_pct(point,control),"training_seed_count":3,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":positive,"n_valid_seeds":3,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"comparable_with_caveats","evidence_status":"N5 accepted three-scope intervals; all data are repeatedly exposed dev; p-values not computed","caveat":f"Q0/Q1 initialization hashes pair; sample-order hash absent; nominal_pairing_unverified; CUDA_COMMON cells reproduce accepted point within {mismatch:.3g}"},intervals,["E040","E041","E042","E058","E063","E064"],[f"N5 CO_PRIMARY_SCOPE_TABLE30.csv metric={metric},contrast={contrast}; N2_BY_SEED CUDA_COMMON"],True)
    for seed in n2seeds:
        cv,nv,g=n2_component(metric,contrast,seed)
        # Prefer N5 accepted seed estimates where present; N2 common-profile is an exact numeric bridge.
        add_per({"comparison_id":comp,"claim_id":"C09","seed":seed,"control":contrast if cv is None else "Q0/control cell","candidate":contrast if nv is None else "Q1/candidate cell","control_value":cv,"candidate_value":nv,"gain":source_seeds.get(seed,g),"metric":metric,"scope":"N2 CUDA_COMMON; accepted per-source-group mean for 603 dev groups","source_file":source_path("E042"),"source_sha256":source_hash("E042"),"source_locator":f"model/query/seed/profile=CUDA_COMMON/metric={metric}; matched to N5 seed_estimate","evidence_ids":"E042;E059;E063","row_origin":"copied_accepted","caveat":"same seed initialization verified; epoch/sample-order pairing unverified"},False)
    for r in accepted_seed_rows:
        add_seed_loo({"comparison_id":comp,"dropped_seed":str(r["loo_drop_seed"]),"remaining_seed_count":2,"gain_mean":getfloat(r,"loo_mean"),"scope":"accepted N5 leave-one-training-seed-out; fixed 603-group cohort","source_file":source_path("E059"),"source_locator":f"metric={metric},contrast={contrast},loo_drop_seed={r['loo_drop_seed']}","evidence_ids":"E059","row_origin":"copied_accepted","caveat":"three training seeds; nominal seed pairing; not an independent confirmatory interval"},False)
    add_task_loo({"comparison_id":comp,"dropped_task":"NA","remaining_task_count":None,"gain_mean":None,"weighting":"not defined","scope":"not_applicable_no_locked_task_macro_estimand","source_file":source_path("E058"),"source_locator":f"query point estimand over source groups; no task-macro LOO in lock","evidence_ids":"E058","row_origin":"copied_accepted","caveat":"do not invent task-LOO for the pooled query estimand"},False)

# 5. N4 seed-02 diagnostic: existing co-primary point cells; no refit/rescore.
n4rel="reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z/RETURN_TO_CHATGPT.zh-CN.md"
n4_comps=[
("N4_S02_TS11D_POSITION","position_standardized_mae",0.209936,0.195131,0.014806,"Q0_TS11-D","Q1_TS11-D"),
("N4_S02_TS11D_SAFEFD","safe_fd_normalized",0.976491,0.956604,0.019888,"Q0_TS11-D","Q1_TS11-D"),
]
for cid,metric,c,n,gain,control,candidate in n4_comps:
    add_comp_rows({"comparison_id":cid,"claim_id":"C10","family":"N4 fixed-model seed diagnostic","contract_id":"CB_N2_N4_COMMAND28_FIXED_MODEL","cohort_hash":"603 exposed dev source groups; N2/N4 accepted identity","split":"development_previously_exposed","task_scope":"Full28","horizon":"H30","metric":metric,"metric_unit":"train-standardized error","metric_definition_hash":"accepted N2 co-primary lock","aggregation":"seed 20261002 source-group means","control":control,"candidate":candidate,"control_mean":c,"candidate_mean":n,"gain_abs":gain,"gain_relative_pct":rel_pct(gain,c),"training_seed_count":1,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":1,"n_valid_seeds":1,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"matched_accepted","evidence_status":"accepted within fixed seed; full three-seed N2/N5 effects are mixed","caveat":"S02 also improves time-centered L2 component; future-GT oracle is diagnostic only; L2 terms are not additive L1 attribution"},[],["E053","E055"],["N4 return L15-L17; seed=20261002; N4_EFFECT_SUPPORT_SUMMARY row"],True)

# 6. VideoMAE CoreB saved-prediction contrasts; separate score acceptance from checkpoint replay.
vm_pairs=read_csv("reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CORE_B_PAIRED_CONTRASTS.csv")
vm_metrics=read_csv("reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z/CORE_B_PER_SEED_METRICS.csv")
vm_lookup={}
for r in vm_metrics:
    if r["split"]=="dev" and r["aggregation"]=="episode_equal" and r["region_raw"]=="action_full28" and r["metric"] in ["standardized_mae","safe_fd_normalized"]:
        vm_lookup[(r["model"],r["mode"],r["horizon"],r["seed"],r["metric"])]=float(r["value"])
vm_cmap={
 "Delta_backbone":("V-JEPA fusion","VideoMAE fusion","V-JEPA","TS11-D","VideoMAE","TS11-D","negate_source"),
 "G_VM":("VideoMAE STATE-CTRL","VideoMAE fusion","VM-STATE-CTRL","STATE-CTRL","VideoMAE","TS11-D","same"),
 "G_VJ":("Team STATE-CTRL","V-JEPA fusion","TEAM-STATE-CTRL","STATE-CTRL","V-JEPA","TS11-D","same"),
}
for metric in ["standardized_mae","safe_fd_normalized"]:
 for horizon in ["30","60"]:
  for contrast,(control_label,candidate_label,control_model,control_mode,candidate_model,candidate_mode,direction) in vm_cmap.items():
    rr=next(r for r in vm_pairs if r["split"]=="dev" and r["horizon"]==horizon and r["region_raw"]=="action_full28" and r["metric"]==metric and r["contrast"]==contrast)
    raw=float(rr["mean_fixed_seed_contrast"])
    gain=-raw if direction=="negate_source" else raw
    seedvals={}
    control_seed={};candidate_seed={}
    for seed in ["20261001","20261002","20261003"]:
        source_key="seed_"+seed
        val_raw=float(rr[source_key])
        eg=-val_raw if direction=="negate_source" else val_raw
        seedvals[seed]=eg
        control_seed[seed]=vm_lookup.get((control_model,control_mode,horizon,seed,metric))
        candidate_seed[seed]=vm_lookup.get((candidate_model,candidate_mode,horizon,seed,metric))
        if contrast=="Delta_backbone":
            eg2=(control_seed[seed]-candidate_seed[seed]) if control_seed[seed] is not None and candidate_seed[seed] is not None else eg
        elif contrast in ["G_VM","G_VJ"]:
            eg2=(control_seed[seed]-candidate_seed[seed]) if control_seed[seed] is not None and candidate_seed[seed] is not None else eg
        if eg2 is not None and abs(eg2-eg)>3e-8: raise RuntimeError("VideoMAE per-seed contrast does not reconcile with accepted paired table")
        add_per({"comparison_id":f"VIDEOMAE_{contrast}_H{horizon}_{metric.upper()}","claim_id":"C07","seed":seed,"control":control_label,"candidate":candidate_label,"control_value":control_seed[seed],"candidate_value":candidate_seed[seed],"gain":eg,"metric":metric,"scope":f"saved-prediction independent score; 603 exposed dev source groups; H{horizon}","source_file":source_path("E086"),"source_sha256":source_hash("E086"),"source_locator":f"model/mode/seed/horizon={horizon}/full28/{metric}; contrast row {contrast}","evidence_ids":"E083;E084;E085;E086","caveat":"saved-prediction score only; CoreB checkpoint fresh replay/portable integration not accepted"},False)
    ci=json.loads(rr["ci95_source_group_bootstrap_fixed_seeds"])
    if direction=="negate_source":low,high=-float(ci[1]),-float(ci[0])
    else:low,high=float(ci[0]),float(ci[1])
    sd=float(rr["sample_sd_across_three_seed_contrasts_ddof1"])
    base={"comparison_id":f"VIDEOMAE_{contrast}_H{horizon}_{metric.upper()}","claim_id":"C07","family":"VideoMAE development saved-prediction comparison","contract_id":"CB_VIDEOMAE_VJEPA_SYSTEM_LEVEL","cohort_hash":"61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78","split":"development_previously_exposed","task_scope":"Full28","horizon":"H"+horizon,"metric":metric,"metric_unit":"train-normalized error","metric_definition_hash":"independent CoreB scorer SHA64d699; V3 normalizer hash in acceptance","aggregation":"episode-equal; fixed three seeds; 603 source groups","control":control_label,"candidate":candidate_label,"control_mean":avg([x for x in control_seed.values() if x is not None]),"candidate_mean":avg([x for x in candidate_seed.values() if x is not None]),"gain_abs":gain,"gain_relative_pct":rel_pct(gain,avg([x for x in control_seed.values() if x is not None])),"training_seed_count":3,"seed_gain_sd":sd,"seed_gain_median":median(list(seedvals.values())),"n_positive_seeds":sum(x>0 for x in seedvals.values()),"n_valid_seeds":3,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"comparable_with_caveats","evidence_status":"accepted independent score of saved dev predictions; CI conditional on fixed seeds and repeatedly used dev","caveat":"system-level comparison, not causal backbone attribution; 64/20079 V-JEPA cache bridge rows differ; state-control output mismatch unresolved"}
    pvalue=getfloat(rr,"p_centered_bootstrap_add_one_two_sided")
    holm=getfloat(rr,"p_holm_existing_four_test_family")
    base["accepted_p_value"]=pvalue;base["holm_adjusted_p"]=holm
    add_comp_rows(base,[{"scope":"fixed-seed-source-group-bootstrap","lower":low,"upper":high,"method":f'{rr["bootstrap_draws"]} paired-group draws; original sign {contrast}; transformed to canonical error gain'}],["E085","E083","E084"],[f"row split=dev,horizon={horizon},region_raw=action_full28,metric={metric},contrast={contrast}"],False)
  # The interaction is a difference of gains, not an absolute prediction comparator.
  rr=next(r for r in vm_pairs if r["split"]=="dev" and r["horizon"]==horizon and r["region_raw"]=="action_full28" and r["metric"]==metric and r["contrast"]=="I")
  raw=float(rr["mean_fixed_seed_contrast"])
  ivals={s:float(rr["seed_"+s]) for s in ["20261001","20261002","20261003"]}
  for s,v in ivals.items():
    add_per({"comparison_id":f"VIDEOMAE_INTERACTION_H{horizon}_{metric.upper()}","claim_id":"C07","seed":s,"control":"G_VJ","candidate":"G_VM","control_value":None,"candidate_value":None,"gain":v,"metric":metric,"scope":"fixed-seed interaction I=G_VM-G_VJ","source_file":source_path("E085"),"source_sha256":source_hash("E085"),"source_locator":f"paired-contrast row I; seed_{s}","evidence_ids":"E083;E084;E085","caveat":"an interaction can be positive while VM own visual gain is negative"},False)
  ci=json.loads(rr["ci95_source_group_bootstrap_fixed_seeds"])
  add_comp_rows({"comparison_id":f"VIDEOMAE_INTERACTION_H{horizon}_{metric.upper()}","claim_id":"C07","family":"VideoMAE system interaction","contract_id":"CB_VIDEOMAE_VJEPA_SYSTEM_LEVEL","cohort_hash":"603 source groups; development input identity verified","split":"development_previously_exposed","task_scope":"Full28","horizon":"H"+horizon,"metric":metric+"_interaction","metric_unit":"train-normalized error interaction","metric_definition_hash":"I=G_VM-G_VJ per accepted audit","aggregation":"three fixed seeds; paired source groups","control":"V-JEPA visual gain","candidate":"VideoMAE visual gain","control_mean":None,"candidate_mean":None,"gain_abs":raw,"gain_relative_pct":None,"training_seed_count":3,"seed_gain_sd":float(rr["sample_sd_across_three_seed_contrasts_ddof1"]),"seed_gain_median":median(list(ivals.values())),"n_positive_seeds":sum(x>0 for x in ivals.values()),"n_valid_seeds":3,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"comparable_with_caveats","evidence_status":"accepted interaction; not a net-video causal attribution","caveat":"must read G_VM and G_VJ alongside I; fixed models, repeated dev"},[{"scope":"fixed-seed-source-group-bootstrap","lower":float(ci[0]),"upper":float(ci[1]),"method":"accepted paired group bootstrap"}],["E085","E083"],[f"row split=dev,horizon={horizon},region_raw=action_full28,metric={metric},contrast=I"],False)

# 7. Corrected five-task V2: per-task accepted loss summaries; task-LOO unit is task_id.
v2_root="reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/"
v2_losses=read_csv(v2_root+"results/ABSOLUTE_LOSSES.csv")
v2_dvi=read_csv(v2_root+"results/ALL_SEED_D_V_I.csv")
v2_stats=read_csv(v2_root+"results/SUMMARY_STATISTICS.csv")
v2_taskloo=read_csv(v2_root+"results/ALL_SEED_LOO.csv")
v2_ci=read_csv(v2_root+"results/CONDITIONAL_CI.csv")
v2_way=read_csv(v2_root+"results/WAYPOINT.csv")
v2_ids=["119","94","90","141","230"]
v2_seeds=["202610131","202610132","202610133"]
def v2_metric_value(model_id,seed,task,metric):
    m={"position_mae":"position_mae","safe_fd_l1":"safe_fd_l1"}[metric]
    r=next(x for x in v2_losses if x["model_id"]==model_id and x["seed"]==seed and x["task_id"]==str(task))
    return float(r[m])
def v2_model_id(program,task):
    if program in ["G-S","G-V"] or program.startswith("S-S-") or program.startswith("S-V-"):return program
    return f"{program}-{task}"
def v2_cell_for(seed,task,program,metric):
    return v2_metric_value(v2_model_id(program,task),seed,task,metric)
v2_formulas={
 "D_S":("G-S","S-S-{task}"),
 "D_V":("G-V","S-V-{task}"),
 "V_G":("G-S","G-V"),
 "V_S":("S-S-{task}","S-V-{task}"),
}
v2_macro_seed={}
for metric in ["position_mae","safe_fd_l1"]:
 for contrast in ["D_S","D_V","V_G","V_S","I"]:
    comp=f"V2_ALL5_{contrast}_{metric.upper()}"
    seed_values={};controls={};candidates={}
    for seed in v2_seeds:
        task_gains=[];task_ctrl=[];task_cand=[]
        for task in v2_ids:
            if contrast=="I":
                d_s=v2_cell_for(seed,task,"G-S",metric)-v2_cell_for(seed,task,f"S-S-{task}",metric)
                d_v=v2_cell_for(seed,task,"G-V",metric)-v2_cell_for(seed,task,f"S-V-{task}",metric)
                task_gains.append(d_v-d_s);task_ctrl.append(d_s);task_cand.append(d_v)
            else:
                cprog,candprog=v2_formulas[contrast]
                cm=v2_cell_for(seed,task,cprog.format(task=task),metric)
                nm=v2_cell_for(seed,task,candprog.format(task=task),metric)
                task_gains.append(cm-nm);task_ctrl.append(cm);task_cand.append(nm)
        seed_values[seed]=statistics.mean(task_gains)
        controls[seed]=statistics.mean(task_ctrl);candidates[seed]=statistics.mean(task_cand)
    v2_macro_seed[(metric,contrast)]=seed_values
    ss=next(x for x in v2_stats if x["scope"]=="five_task_macro" and x["task_id"] in ["ALL5_OR_AVAILABLE","ALL5"] and x["contrast"]==contrast and x["metric"]==metric)
    accepted_mean=float(ss["mean_across_seeds"])
    if abs(statistics.mean(list(seed_values.values()))-accepted_mean)>1e-8:raise RuntimeError(f"V2 macro reconstruction mismatch {metric}/{contrast}")
    # One accepted per-seed conditional interval per seed. These remain distinct rows.
    cirows=[x for x in v2_ci if x["scope"]=="five_task_macro" and x["task_id"] in ["ALL5","ALL5_OR_AVAILABLE"] and x["contrast"]==contrast and x["metric"]==metric]
    intervals=[{"scope":f"conditional_group_bootstrap_seed_{x['seed']}","lower":float(x["ci_low_2_5"]),"upper":float(x["ci_high_97_5"]),"method":f"{x['bootstrap_replicates']} source-group draws; {x['source_groups']}; session/trial independence unverified"} for x in cirows]
    for seed in v2_seeds:
        control=controls[seed];candidate=candidates[seed]
        if contrast=="I":
            control=seed_values[seed]-(0 if False else 0) # placeholder replaced below with component effects from accepted D/V/I table
            dv=statistics.mean([float(x["value"]) for x in v2_dvi if x["seed"]==seed and x["metric"]==metric and x["contrast"]=="D_V"])
            ds=statistics.mean([float(x["value"]) for x in v2_dvi if x["seed"]==seed and x["metric"]==metric and x["contrast"]=="D_S"])
            control=ds;candidate=dv
        add_per({"comparison_id":comp,"claim_id":"C14","seed":seed,"control":contrast,"candidate":contrast,"control_value":control,"candidate_value":candidate,"gain":seed_values[seed],"metric":metric,"scope":"five-task macro; equal task weights; current corrected split; 3 training seeds","source_file":source_path("E075"),"source_sha256":source_hash("E075"),"source_locator":f"derived per seed from accepted task losses; summary check E077","evidence_ids":"E073;E074;E075;E076;E077","caveat":"per-task model recipes/capacities differ; source/session independence unverified"},True)
    # derive positive directions from the fixed three seed effects and retain accepted mean/SD/median
    pos=sum(x>0 for x in seed_values.values())
    macro_control=statistics.mean(list(controls.values())) if contrast!="I" else None
    macro_candidate=statistics.mean(list(candidates.values())) if contrast!="I" else None
    # accepted standard deviation/median are copied, not reconstructed here
    add_comp_rows({"comparison_id":comp,"claim_id":"C14","family":"V2 five-task task-specialization/visual contrasts","contract_id":"CB_V2_TASK_MACRO_CORRECTED_SPLIT","cohort_hash":"tasks 119/94/90/141/230; literal corrected source-key split; fold and ID hashes in V2 acceptance","split":"development_previously_exposed","task_scope":"five-task macro ALL5_OR_AVAILABLE","horizon":"H30","metric":metric,"metric_unit":"train-normalized error; action native units unknown","metric_definition_hash":"V2 locked episode-equal metric source E074/E082","aggregation":"within-episode window mean; episode-equal per task; equal task macro; mean of 3 seeds","control":contrast,"candidate":contrast,"control_mean":macro_control,"candidate_mean":macro_candidate,"gain_abs":accepted_mean,"gain_relative_pct":rel_pct(accepted_mean,macro_control),"training_seed_count":3,"seed_gain_sd":float(ss["sample_sd_across_seeds"]),"seed_gain_median":float(ss["median_across_seeds"]),"n_positive_seeds":pos,"n_valid_seeds":3,"n_source_groups":47,"n_acquisition_clusters":None,"n_episodes":47,"n_windows":185,"comparison_status":"different_contract" if contrast in ["D_S","D_V"] else "comparable_with_caveats","evidence_status":"accepted corrected V2 development metrics; conditional per-seed group intervals only","caveat":"no session/trial independence; five-task macro is not all-episode-equal; D_S/D_V conflate capacity, LR/epoch and data sharing"},intervals,["E073","E074","E075","E076","E077","E079","E082"],[f"SUMMARY_STATISTICS.csv scope=five_task_macro task_id=ALL5_OR_AVAILABLE contrast={contrast} metric={metric}; per-seed source table E076"],True)
    # Task leave-one-out from the accepted ALL_SEED_LOO file: average the three per-seed task-LOO means.
    for task in v2_ids:
        lrows=[x for x in v2_taskloo if x["metric"]==metric and x["contrast"]==contrast and x["omitted_task_id"]==task and x["seed"] in v2_seeds]
        add_task_loo({"comparison_id":comp,"dropped_task":task,"remaining_task_count":4,"gain_mean":avg([float(x["loo_mean"]) for x in lrows]),"weighting":"equal remaining tasks within seed, then equal mean over fixed 3 seeds","scope":"true task-LOO; source ALL_SEED_LOO.csv omitted_task_id confirms deletion unit","source_file":source_path("E078"),"source_locator":f"metric={metric},contrast={contrast},omitted_task_id={task}; average of three seed rows","evidence_ids":"E073;E076;E078","caveat":"not seed-LOO; group/session independence unverified"},True)
    # True seed-LOO over the five task-macro effects.
    for dropped in v2_seeds:
        rem=[v for s,v in seed_values.items() if s!=dropped]
        add_seed_loo({"comparison_id":comp,"dropped_seed":dropped,"remaining_seed_count":2,"gain_mean":avg(rem),"scope":"true seed-LOO of five-task macro; equal task weighting then remaining seeds","source_file":source_path("E076"),"source_locator":f"ALL_SEED_D_V_I.csv accepted task/seed rows; drop seed={dropped}","evidence_ids":"E075;E076;E077","caveat":"two of three fixed development model seeds remain; not independent retraining confirmation"},True)

# task119 AUX_ONLY to GOAL_COND: B_GAIN = AUX_ONLY error - GOAL_COND error.
for metric in ["position_mae","safe_fd_l1"]:
    comp=f"V2_TASK119_GOAL_COND_VS_AUX_ONLY_{metric.upper()}"
    aux={s:v2_metric_value("B-AUX_ONLY",s,"119",metric) for s in v2_seeds}
    goal={s:v2_metric_value("B-GOAL_COND",s,"119",metric) for s in v2_seeds}
    effect={s:aux[s]-goal[s] for s in v2_seeds}
    summary=next(x for x in v2_stats if x["scope"]=="task_119" and x["task_id"]=="119" and x["contrast"]=="B_GAIN" and x["metric"]==metric)
    # Check that accepted B_GAIN rows are the same control-candidate contrast.
    for s in v2_seeds:
        accepted=next(x for x in v2_way if x["task_id"]=="119" and x["seed"]==s and x["metric"]==metric and x["contrast"]=="B_GAIN")
        if abs(float(accepted["value"])-effect[s])>1e-10:raise RuntimeError("V2 task119 goal/aux direction mismatch")
        add_per({"comparison_id":comp,"claim_id":"C14","seed":s,"control":"B-AUX_ONLY","candidate":"B-GOAL_COND","control_value":aux[s],"candidate_value":goal[s],"gain":effect[s],"metric":metric,"scope":"task119; 12 dev groups/episodes; 48 H30 anchors","source_file":source_path("E075"),"source_sha256":source_hash("E075"),"source_locator":f"ABSOLUTE_LOSSES.csv model_id=B-AUX_ONLY/B-GOAL_COND task_id=119 seed={s}; direction cross-check E080 B_GAIN","evidence_ids":"E073;E074;E075;E080;E081","caveat":"12 dev groups; seed signs mixed; episode-level session independence unverified"},True)
    cirows=[x for x in v2_ci if x["scope"] in ["task_119","task"] and x["task_id"]=="119" and x["contrast"]=="B_GAIN" and x["metric"]==metric]
    intervals=[{"scope":f"conditional_group_bootstrap_seed_{x['seed']}","lower":float(x["ci_low_2_5"]),"upper":float(x["ci_high_97_5"]),"method":f"{x['bootstrap_replicates']} source-group draws; 12 groups; conditional diagnostic"} for x in cirows]
    add_comp_rows({"comparison_id":comp,"claim_id":"C14","family":"V2 task119 waypoint contrast","contract_id":"CB_V2_TASK119_WAYPOINT","cohort_hash":"task119 corrected split; 12 development groups","split":"development_previously_exposed","task_scope":"task119 only","horizon":"H30","metric":metric,"metric_unit":"train-normalized error; native units unknown","metric_definition_hash":"V2 locked metric contract","aggregation":"12 episodes/groups; three fixed seeds; episode-equal","control":"B-AUX_ONLY","candidate":"B-GOAL_COND","control_mean":avg(list(aux.values())),"candidate_mean":avg(list(goal.values())),"gain_abs":float(summary["mean_across_seeds"]),"gain_relative_pct":rel_pct(float(summary["mean_across_seeds"]),avg(list(aux.values()))),"training_seed_count":3,"seed_gain_sd":float(summary["sample_sd_across_seeds"]),"seed_gain_median":float(summary["median_across_seeds"]),"n_positive_seeds":sum(x>0 for x in effect.values()),"n_valid_seeds":3,"n_source_groups":12,"n_acquisition_clusters":None,"n_episodes":12,"n_windows":48,"comparison_status":"comparable_with_caveats","evidence_status":"accepted task119 dev contrast; no independent confirmation","caveat":"goal endpoint metric is secondary and mixed; no model-selection claim"},intervals,["E073","E074","E075","E079","E080"],[f"SUMMARY_STATISTICS.csv task_119 B_GAIN metric={metric}; per-seed absolute losses E075"],True)
    for dropped in v2_seeds:
        rem=[v for s,v in effect.items() if s!=dropped]
        add_seed_loo({"comparison_id":comp,"dropped_seed":dropped,"remaining_seed_count":2,"gain_mean":avg(rem),"scope":"task119 B_GAIN seed-LOO; remaining fixed model seeds","source_file":source_path("E080"),"source_locator":f"WAYPOINT.csv task=119 contrast=B_GAIN metric={metric}; drop seed={dropped}","evidence_ids":"E075;E080","caveat":"small support; not a future-training uncertainty interval"},True)
    add_task_loo({"comparison_id":comp,"dropped_task":"NA","remaining_task_count":None,"gain_mean":None,"weighting":"not applicable","scope":"not_applicable_single_task_waypoint","source_file":source_path("E080"),"source_locator":"task119 only; no task macro","evidence_ids":"E080","caveat":"task LOO is not defined for the one-task waypoint"},False)

# 8. H30 Ridge: both accepted absolute summaries and fixed-model paired contrast.
for metric,control,candidate,gain,lo,hi,src in [
("position_standardized_mae",0.223135,0.230365,0.223135-0.230365,-0.03719338740126155,-0.033567723008182025,"H30 report; accepted Q0 STATE-CTRL minus Ridge interval"),
("safe_fd_normalized",0.976491,1.044448,0.976491-1.044448,-0.09436278449726086,-0.08753891454770132,"H30 report; accepted Q0 STATE-CTRL minus Ridge interval")]:
    # Persistence comparator is a deterministic reference; its interval is not asserted here.
    add_comp_rows({"comparison_id":"H30_RIDGE_VS_PERSISTENCE_"+metric.upper(),"claim_id":"C12","family":"H30 CPU Ridge horizon diagnostic","contract_id":"CB_COMMAND28_H30_RIDGE","cohort_hash":"603 accepted development source groups","split":"development_previously_exposed","task_scope":"Full28 command target","horizon":"H30","metric":metric,"metric_unit":"train-normalized error","metric_definition_hash":"H30 locked primary metric","aggregation":"episode-equal; single deterministic CPU Ridge and Persistence","control":"Persistence","candidate":"CPU residual Ridge","control_mean":control,"candidate_mean":candidate,"gain_abs":gain,"gain_relative_pct":rel_pct(gain,control),"training_seed_count":None,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":None,"n_valid_seeds":None,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"comparable_with_caveats","evidence_status":"accepted fixed-model dev summary; conditional confidence interval is for Q0-state vs Ridge, not persistence vs Ridge","caveat":"H30 Ridge worse than Persistence; different L2 objective/1,909,320 parameters; no CPU/CUDA bridge"},
       [{"scope":"no_accepted_CI_for_Ridge_vs_Persistence","lower":None,"upper":None,"method":"NA; no new resampling"}],["E047","E048"],["FINAL_REPORT table H30 dev; persistence row and Ridge CPU row"],False)
# Q0 State-control vs Ridge: point/CIs directly accepted, both error metrics.
for metric,control,candidate,gain,lo,hi in [
("position_standardized_mae",0.195021,0.230365,-0.0353441596252594,-0.03719338740126155,-0.033567723008182025),
("safe_fd_normalized",0.953524,1.044448,-0.09092396035988615,-0.09436278449726086,-0.08753891454770132)]:
    add_comp_rows({"comparison_id":"H30_STATE_Q0_VS_RIDGE_"+metric.upper(),"claim_id":"C12","family":"H30 CPU Ridge horizon diagnostic","contract_id":"CB_COMMAND28_H30_RIDGE","cohort_hash":"603 accepted dev source groups","split":"development_previously_exposed","task_scope":"Full28 command target","horizon":"H30","metric":metric,"metric_unit":"train-normalized error","metric_definition_hash":"H30 locked primary metric","aggregation":"fixed-three-seed mean Q0 STATE-CTRL vs single Ridge; paired 603 groups","control":"Q0 STATE-CTRL fixed-three-seed mean","candidate":"CPU Ridge","control_mean":control,"candidate_mean":candidate,"gain_abs":gain,"gain_relative_pct":rel_pct(gain,control),"training_seed_count":3,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":None,"n_valid_seeds":3,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"comparable_with_caveats","evidence_status":"accepted paired source-group CI; fixed models / exposed development","caveat":"negative means Ridge has higher error; does not compare against command-oracle or OBS43 task"},[{"scope":"fixed-model-source-group-bootstrap","lower":lo,"upper":hi,"method":"10,000 paired group draws; conditional on fixed models"}],["E047","E048"],["FINAL_REPORT L61-L66; FINAL_ACCEPTANCE paired_bootstrap"],False)

# 9. OBS43 command diagnostic is kept separate from command28 and from observation43 targets.
obs_p=0.16469952483390388; obs_h=0.1983193573201481; obs_o=0.1722962761399459
fd_p=1.0684457787551533; fd_h=1.238543831522864; fd_o=1.1393459383533002
for metric,control,candidate,lo,hi in [
("position_standardized_mae",obs_p,obs_h,-0.03681,-0.03047),
("safe_fd_normalized",fd_p,fd_h,-0.17731,-0.16299)]:
    gain=control-candidate
    add_comp_rows({"comparison_id":"OBS43_HOLD_VS_OBS_PERSISTENCE_"+metric.upper(),"claim_id":"C12","family":"OBS43 separate 43D observation target diagnostic","contract_id":"CB_OBS43_FULL43_NOT_COMMAND28","cohort_hash":"603 exposed dev source groups; exact OBS43 parent contract","split":"development_previously_exposed","task_scope":"Full43 observation-state target","horizon":"H30 open rollout","metric":metric,"metric_unit":"observation-train-standardized error","metric_definition_hash":"OBS43 lock, distinct from command28","aggregation":"episode-equal fixed-model comparison","control":"OBS persistence","candidate":"ARX HOLD current observed command","control_mean":control,"candidate_mean":candidate,"gain_abs":gain,"gain_relative_pct":rel_pct(gain,control),"training_seed_count":None,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":None,"n_valid_seeds":None,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"different_contract","evidence_status":"accepted OBS43 diagnostic","caveat":"negative: HOLD is worse than OBS persistence; do not rank against command28"},[{"scope":"fixed-model-paired-source-group-bootstrap","lower":lo,"upper":hi,"method":"10,000 group draws; accepted closeout"}],["E049","E051","E052"],["OBS43 FINAL_REPORT forecast views; closeout RETURN line 5-8"],False)
for metric,control,candidate,lo,hi in [
("position_standardized_mae",obs_h,obs_o,0.02496,0.02709),
("safe_fd_normalized",fd_h,fd_o,0.09516,0.10331)]:
    gain=control-candidate
    add_comp_rows({"comparison_id":"OBS43_ORACLE_VS_HOLD_"+metric.upper(),"claim_id":"C12","family":"OBS43 future command oracle diagnostic","contract_id":"CB_OBS43_FULL43_NOT_COMMAND28","cohort_hash":"603 exposed dev source groups","split":"development_previously_exposed","task_scope":"Full43 observation-state target; actual future command is privileged input","horizon":"H30","metric":metric,"metric_unit":"observation-train-standardized error","metric_definition_hash":"OBS43 locked metric","aggregation":"fixed linear model; paired groups","control":"ARX HOLD","candidate":"ARX ORACLE_COMMAND","control_mean":control,"candidate_mean":candidate,"gain_abs":gain,"gain_relative_pct":rel_pct(gain,control),"training_seed_count":None,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":None,"n_valid_seeds":None,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"different_contract","evidence_status":"accepted diagnostic-only future-command oracle","caveat":"future command is not available-at-inference; oracle remains worse than persistence"},[{"scope":"fixed-model-paired-source-group-bootstrap","lower":lo,"upper":hi,"method":"10,000 paired group draws; accepted diagnostic interval"}],["E049","E050"],["FINAL_REPORT L40-L46; accepted ORACLE/HOLD metrics"],False)

# 10. HDI direct vs increment: position gain and temporal-fidelity tradeoff.
for metric,control,candidate,gain,lo,hi,scope in [
("hand_native_level_mae",0.090962,0.088679,0.002282,None,None,"three-seed descriptive episode-equal"),
("hand_native_future_internal_fd",0.020978,0.021161,-0.000183,-0.0002192,-0.0001472,"primary crossed seed-by-episode interval; CI sign reversed from T-C")]:
    add_comp_rows({"comparison_id":"HDI_INCREMENT_VS_DIRECT_"+metric.upper(),"claim_id":"C06","family":"HDI direct/increment matched decoder","contract_id":"CB_HDI_H60_DIRECT_INCREMENT","cohort_hash":"V3 canonical train/dev IDs; source-fit tables separated","split":"development","task_scope":"Hand index slice 0:14; anatomy/physical units not asserted","horizon":"H60","metric":metric,"metric_unit":"action-command numeric / train-standardized where applicable","metric_definition_hash":"HDI locked metric definition","aggregation":"episode-equal mean across three fresh training seeds","control":"CTRL-DIRECT","candidate":"TREAT-INCREMENT","control_mean":control,"candidate_mean":candidate,"gain_abs":gain,"gain_relative_pct":rel_pct(gain,control),"training_seed_count":3,"seed_gain_sd":None,"seed_gain_median":None,"n_positive_seeds":3 if metric=="hand_native_level_mae" else 0,"n_valid_seeds":3,"n_source_groups":603,"n_acquisition_clusters":None,"n_episodes":603,"n_windows":2322,"comparison_status":"matched_accepted","evidence_status":"accepted semantic bundle and independent recompute; candidate gate fails","caveat":"increment improves level but Hand future-internal FD is consistently worse; no physical/robot-success inference"},
      ([{"scope":scope,"lower":lo,"upper":hi,"method":"accepted crossed seed-by-episode 95% interval"}] if lo is not None else []),["E034","E035"],["FORMAL_RESULTS table rows Hand native level/FD; primary T-C CI lines 46-62"],False)
# The exact three-seed Hand FD effects are printed as T-C; convert to canonical control-minus-candidate gains.
for seed,tc in [("20261071",0.0002055),("20261072",0.0001888),("20261073",0.0001547)]:
    add_per({"comparison_id":"HDI_INCREMENT_VS_DIRECT_HAND_FD","claim_id":"C06","seed":seed,"control":"CTRL-DIRECT","candidate":"TREAT-INCREMENT","control_value":None,"candidate_value":None,"gain":-tc,"metric":"hand_native_future_internal_fd","scope":"59 future transitions; development; three fresh seeds","source_file":source_path("E034"),"source_sha256":source_hash("E034"),"source_locator":f"FORMAL_RESULTS primary seed effects T-C; seed={seed}","evidence_ids":"E034;E035","caveat":"canonical gain reverses T-C; all three gains negative"},True)

# 11. Root registry: exact canonical/alias/attempt/placeholder paths from the bounded scope.
ROOT_DEFS=[
("R001","PROJECT_ROOT","root","verified_project_root","EE6008 AY26S1-10-CCA Project","project present; Python path executable","", "project AGENTS.md"),
("R002","REPO_ROOT","reference_tree","layout_policy_conflict","repo/ee6008-humanoid-vjepa","current subtree is non-empty; project AGENTS.md describes empty placeholder; git executable absent so branch/commit unknown","", "read-only filesystem and git-command preflight"),
("R003","COREA_B48_SPECIFIED_PROJECT_PATH","placeholder_or_missing_anchor","not_located_at_requested_project_relative_path","reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z","missing at project root; same-named canonical child exists under repo/","R004","E001-E005"),
("R004","COREA_B48_CANONICAL","canonical_sealed","complete_with_caveats","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z","standard 6/6; GPU3 screening 6/6; 12/12 hard gates; dev/validation only","","E001-E005"),
("R005","COREA_B48_BLOCKED_ATTEMPT","attempt","blocked_harvest","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_20260901T033748Z","blocked attempt preserved; not canonical result","R004","HARVEST blocked report"),
("R006","COREA_B48_INTERMEDIATE_HARVESTS","alias_or_recovery_attempts","superseded_intermediate","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix","aggregate copies/recovery attempts; not independent training","R004","bounded directory listing"),
("R007","COREA_FULL120_STAGE1","canonical_sealed","9_of_9_accepted_dev","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a and repo/.../CORE_A_FULL120_LORA_FACTORIAL_V1_*","8 video cells plus STATE-ONLY; one training seed; validation selected","","E006-E010"),
("R008","COREB_B2_B3_RUN_ROOT","working_and_sealed_children","sealed_child_acceptance_supersedes_old_controller","repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z","B2/B3 per-run acceptance and later analysis exist; old controller snapshot stale","","E011-E016"),
("R009","COREB_B3_PARTIAL_ANALYSIS","partial_snapshot","7_of_8_snapshot","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T060958Z_PARTIAL_B3_7OF8","partial snapshot retained; not canonical","R010","bounded archive filename inventory"),
("R010","COREB_B1_B2_B3_FINAL_ANALYSIS","canonical_sealed","B1_7of7_B2_6of6_B3_8of8_QA_pass","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8","B3 candidate gate not ready/not passed; single-seed exploratory","","E011-E016"),
("R011","D1_D2_LAB","canonical_sealed","cpu_baselines_and_learned_bundles","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z","D1/D2 reports separate sparse/dense/support repairs","","E017-E020"),
("R012","D4_RESULTS","canonical_sealed","complete_with_caveats","reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z","D4 V3 dense core 36/36; development-only; actual old D3 objective caveat","","E021-E022"),
("R013","RFSI","canonical_sealed","development_only_complete",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/RAW_FRAME_ORDER_STATE_INFO_V1_20260924T172133Z'),"30/30 training/inference; no robust raw chronology advantage","","E030-E031"),
("R014","MOTION_FIDELITY","canonical_sealed_with_erratum","corrected_development_complete_fit_unresolved",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/CORE_B_MOTION_FIDELITY_AUDIT_V1_20260925T064134Z'),"Hand/Arm naming corrected in derivative report; CPU replay gate failed; no raw-root edits","","E032-E033"),
("R015","HDI","canonical_sealed","semantic_acceptance_pass_candidate_gate_fail",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/HAND_DIRECT_INCREMENT_V1_20261006071115Z/harvest_recovery_20261006T131215Z'),"6 formal cells; source-fit separate; no candidate promotion","","E034-E035"),
("R016","D3_EQ_OLD","historical_as_run_and_repair","objective_mismatch_registered","reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z","historical original trainer kept; corrected implementation is separately named","","E023-E025"),
("R017","D3_EQ_CORRECTED_SCHEDULER","canonical_sealed","12_of_12_training_and_prediction_accepted","reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z","technical objective/scheduler contract passed; scientific gains mixed","","E026-E029"),
("R018","P1P2","canonical_sealed","H30_complete_H60_gate_not_passed","reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z","R3D visual complementarity not established","","E036"),
("R019","POST_P1P2_N0_N1","canonical_sealed","complete_with_limits","reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z","N0 interventions and N1 query fits accepted; fixed exposed dev","","E037-E038"),
("R020","V3_STATE_MATRIX_54","canonical_sealed","54_of_54_predictions_accepted","repo/ee6008-humanoid-vjepa/reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z","original multihorizon matrix; distinct from N3 18-view reuse","","E039;E105"),
("R021","N2_SAFE_EXPORT","canonical_sealed","complete","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z","12 safe models; 24 CPU/CUDA common views accepted","","E040-E043"),
("R022","N3_DIRECT_REUSE","canonical_sealed_reference","18_h30_views_directly_reused","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z","no new scoring/retraining in latest reuse handoff","","E044-E046"),
("R023","H30_RIDGE","canonical_sealed","complete_dev_cpu_ridge","reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z","13 formal CPU fits plus 1 smoke; query line closed","","E047-E048"),
("R024","COMMAND_OBSERVATION_DIAGNOSTIC","canonical_sealed","complete_dev_only","reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z","43D observation target; not a replacement for command28","","E049-E050"),
("R025","OBS43_CLOSEOUT","canonical_sealed","complete_diagnostic_only","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z","independent confirmation spec generated, not executed","","E051-E052"),
("R026","OBS43_SECOND_ROOT","placeholder","empty_unknown","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161856Z","0 files at bounded check; do not treat as acceptance","R025","N4/N5 status references"),
("R027","N4_ERROR_BUDGET","canonical_sealed","24_of_24_analysis_bundles","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z","no new model training; seed02 level and time-centered both improve","","E053-E055"),
("R028","OBS14_READINESS","canonical_sealed_data_blocked","0_of_270","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z","metadata intake only; no scoring or payload access","","E068-E072"),
("R029","N5_CLOSEOUT","canonical_sealed","complete_statistics_readiness_not_ready","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z","30 interval rows and 30 historical endpoint replays accepted","","E056-E067"),
("R030","V2_SALTED_SPLIT","invalidated_attempt","archived_exposed_not_used_for_corrected_selection","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/archive/INNER_SPLIT_DATASET_SALTED_V1_20261013T083000Z","36 prediction jobs with 6 missing; exposure retained","R031","E073-E074"),
("R031","V2_CORRECTED","canonical_sealed","complete_development_only","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z","screening 36/36; formal 36/36; waypoint 6/6; 164 total attempts","","E073-E082"),
("R032","VIDEOMAE_SOURCE_HANDOFF","accepted_with_caveats","artifact_and_saved_score_acceptance_partial","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z","CoreB saved prediction scoring accepted; fresh checkpoint replay blocked 0/12; portable integration needs_fix","","E083-E086"),
("R033","VIDEOMAE_CPU_ZERO_REPLAY","partial_attempt","replay_tolerance_failed","reports/benchmarks/VIDEOMAE_CORE_B_HANDOFF_V2_20261013T065841Z/replay/cpu_zero_20261013T065954Z","12 zero-input CPU comparisons fail exact-zero tolerance; not full fresh checkpoint replay","","E087"),
("R034","ROUTE_C","prepared_or_recovery_only","full_route_C_inference_not_located_as_accepted","reports/core_b/ROUTE_C_VIDEO_ATTRIBUTION_CONTROLS_V1_20260910T154453Z","bounded root has canonical replay correction records; no full accepted matrix in this scope","","E090"),
("R035","HISTORY_VIDEO_MATRIX","canonical_sealed","single_seed_development","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z","4/4 matrix; one seed; temporal-fidelity trade-off","","E091-E092"),
("R036","HISTORY_ENCODER_REPLICATION","canonical_sealed_partial","A_12of12_B_not_triggered","repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z","A candidate gate fail; independent B not run","","E093"),
("R037","ENCODER_ADAPTATION_B","training_complete_inference_pending_then_recovery","prediction_accepted_candidate_gate_fail","research_archive/CORE_B_ADAPTATION_HARVEST_20260919T053154Z; research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z","first harvest 0/6 predictions; later 6/6 predictions, no retraining, gate fail","","E094-E095"),
("R038","JY02_RESIDUAL","canonical_stage1_and_stage2_acceptance","dev_exploratory","repo/ee6008-humanoid-vjepa/reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z","recovered stage1/stage2 acceptance; not independent confirmation","","E103-E104"),
("R039","MOTION_EVENT_FALLBACK","canonical_sealed_exploratory","candidate_gate_fail","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z","single seed; event AUPRC support remains separate from motion forecast score","","E096"),
("R040","CALIBRATED_PROTOTYPE","canonical_sealed_screen","calibration_gate_fail_alpha_zero","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z","alpha=0; do not describe as successful calibration","","E097"),
("R041","EXPERT_ADVANTAGE_SELECTOR","canonical_sealed_screen","candidate_gate_fail","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z","single-seed/OOF selector evidence is exploratory","","E098-E100"),
("R042","DCT_DUAL_LOSS","canonical_sealed_screen","candidate_gate_fail","repo/ee6008-humanoid-vjepa/reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z","one-seed factorial; TV ratio remains low; no promotion","","E099;E101"),
("R043","OBSERVATION_DELAY","canonical_sealed","three_seed_development","repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z","delay robustness not an independent population","","E102"),
("R044","CURRENT_V3_FROZEN_SHARED_RESIDUAL","working_unsealed_root","partial_preseal_snapshot","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z","preseal main predictions PASS; latest recorded stage RGB probe; no final acceptance at snapshot; process state not queried","","E106-E111"),
("R045","COREB_COMMAND_QUERY_FINAL_HANDOFF","handoff_not_experiment","complete_no_new_query","reports/deliverables/CORE_B_COMMAND_QUERY_FINAL_HANDOFF_20261013T105945Z","organizes N2/N3/N4/N5; command28 prepared_not_executed","","E088-E089"),
("R046","COREB_BENCHMARK_CONTRACT_REPAIR","engineering_contract_repair","accepted_metadata_only","reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z","repair root is not a new scientific comparison","","reports benchmark root inventory"),
]
INV=[]
def inv(family,series,planned,actual,bundles,acceptance,independence,status,evidence,note):
    INV.append({"family_id":family,"series":series,"planned_fit_count":planned,"actual_new_training_fits":actual,"accepted_prediction_or_analysis_bundles":bundles,"acceptance_status":acceptance,"independence_status":independence,"inventory_status":status,"evidence_ids":evidence,"note":note})
inv("CORE_A_BATCH48","CoreA dynamic/static and Batch48",12,12,"standard 6/6; screening 6/6","complete_with_caveats","same 617 dev/validation cohort; 3 fixed seeds","accepted","E001-E005","Top-1 modest positive; macro-F1 interval crosses zero; no action-anticipation claim.")
inv("CORE_A_FULL120_LORA","CoreA eight video cells plus STATE-ONLY",9,9,"9/9 accepted","accepted_development_validation","single seed 20260909; same split within comparisons","accepted","E006-E009","Highest absolute video score is S-P-L static pooled LoRA; stage2 gate not passed.")
inv("CORE_A_BASELINE_FAMILY","CoreA frozen V-JEPA/R3D/nonparametric/attentive",None,None,"historical dev/validation matrix","accepted_with_caveats","protocols/caches differ; pooled sources lack episode IDs","mixed","E006;E010","Use as descriptive references only, not a causal backbone ranking.")
inv("CORE_A_VIDEOMAE","CoreA VideoMAE frozen linear probe",3,3,"3 seed train/validation replay","accepted_with_caveats","617 validation episodes; unmatched V-JEPA probe absent","different_contract","E083","Top1 0.943274, macro-F1 0.882317; separate CPU-linear probe.")
inv("CORE_B_B1","B1 baseline",7,0,"7/7 accepted inference bundles","accepted_negative_diagnostic","legacy 614 cohort; single seed","accepted","E011;E013;E016","Persistence beats B-STATE on Arm14 standardized MAE.")
inv("CORE_B_B2","B2 pooled sampling",6,4,"6/6 accepted; 2 dense controls reused","accepted_with_limits","single seed; 0.125 is proxy; 0.25/0.5 true-pixel","accepted","E014;E016","Four true-pixel cells at 0.25/0.5; do not merge proxy and true-pixel cells.")
inv("CORE_B_B3","B3 frozen/LoRA dynamic/static",8,6,"8/8 accepted; 2 exact B2 dense reuses","single_seed_exploratory","single seed; selected B2 head","candidate_gate_not_passed","E015;E016","Later sealed analysis supersedes stale controller running status.")
inv("CORE_B_D1_D2","Support/learned sparse+dense",None,None,"D1 sparse 36; D2 sparse/dense accepted bundles","development_only","603/118 group panels; multiple contracts","accepted_with_caveats","E017-E020;E021","Event support repair and dense panel are separate from primary forecast claims.")
inv("CORE_B_D4","Rolling/delay benchmark",None,0,"D4 dense core 36/36; 118 groups","3-seed descriptive","fixed dev panel, repeatedly exposed","accepted_with_caveats","E021;E022","Older D3 objective mismatch remains an explicit caveat.")
inv("CORE_B_RFSI","Raw frame order × state info",30,30,"30/30 train/inference bundles","3-seed development","603 dev groups, 2712 train groups","complete_development_only","E030;E031","No robust chronology advantage; anchor condition is not pure video.")
inv("CORE_B_MOTION_FIDELITY","Motion-fidelity correction",0,0,"30/30 dev bundles audited; no new fit","development audit","CPU/GPU replay gate 0/30; fit branch unresolved","complete_with_limitation","E032;E033","Corrected Arm/Hand label interpretation; no physical-unit semantics.")
inv("CORE_B_HDI","Direct vs increment",6,6,"18 semantic bundles accepted (12 formal + 6 source-fit)","3-seed development","source fit kept separate; dev reused","candidate_gate_fail","E034;E035","Level error falls; Hand future-internal FD fails primary/superiority gate.")
inv("CORE_B_D3_HISTORICAL","Historical D3 as-run objective/scheduler",None,None,"historical checkpoints and accepted D4 tables","historical_exposure","actual old trainer is not corrected episode-equal objective","different_contract","E020;E023;E025","Keep old as-run results separate from corrected objective/scheduler.")
inv("CORE_B_D3_EQ_SCHEDULER","Episode-equal objective + corrected scheduler",12,12,"train 12/12; dev 12/12; dense secondary 24/24","3 seeds × 2 configs × 2 horizons","603 dev groups; fixed seeds","complete_development_only","E026-E029","Implementation passed; no general scheduler or video benefit.")
inv("CORE_B_P1","R3D H30 visual complementarity",3,3,"3 primary/dense seeds accepted","3 seeds","same dev panel as prior work","mixed","E036-E038","ALT versus S-HIST gain +0.000313 CI crosses zero.")
inv("CORE_B_P2","Event onset/support",None,0,"support gates pass; dense panels scored","3-seed dev","event rows are not independent population units","mixed","E036;E037","Retain event opportunity/support denominator; no stable complementarity.")
inv("CORE_B_N0","REAL/ZERO/PERM functional sensitivity",0,0,"6/6 real replays; 12/12 interventions","3 fixed models","previously exposed dev","supported_within_scope","E037-E038","Function response does not prove visual information value or causality.")
inv("CORE_B_N1","Hadamard query geometry",6,6,"24 train/dev bundles; 6 Q1 fresh replays","3 seeds","603 exposed groups; backend deviation disclosed","seed_sensitive","E037-E038","G_TS positive point is seed-sensitive; no p-values or independent confirmation.")
inv("CORE_B_N2","Safe export/backend bridge",0,0,"12 models; 24 CPU/CUDA_COMMON views","3 seeds × 2 models × 2 horizons","same exposed 603-group dev","accepted","E040-E043","Backend drift small in accepted same-model view table.")
inv("CORE_B_N3","State-information recovery direct reuse",0,0,"18 H30 views reused; no rescore","3 seeds represented","same exposed dev","accepted_direct_reference","E044-E046","Original 54-cell matrix remains separate from the 18-view reuse.")
inv("CORE_B_H30_RIDGE","H30 residual Ridge",14,14,"12 CV fits + 1 refit + 1 smoke; train/dev predictions","CPU-only","603 exposed dev groups","complete_negative","E047-E048","Ridge is worse than persistence and Q0 STATE-CTRL on both co-primary metrics.")
inv("CORE_B_OBS43_TARGET_DIAGNOSTIC","43D observation target diagnostic",14,14,"4 forecast views; independent scoring accepted","CPU-only","603 exposed dev groups","complete_diagnostic","E049-E050","Future-command oracle is privileged; OBS43 and command28 are different contracts.")
inv("CORE_B_OBS43_ERROR_BUDGET","OBS43 fixed-model error budget",0,0,"1152 old score cells reproduced","fixed models","reused exposed dev","complete_diagnostic","E051-E052","No new fit; independent confirmation spec not executed.")
inv("CORE_B_N4","N4 command error budget",0,0,"24/24 analysis bundles; 12 historical train-only forwards","fixed 12 models","603 exposed dev groups","complete_diagnostic","E053-E055","Seed02 improves level and time-centered components; query means remain seed-sensitive.")
inv("CORE_B_OBS14","Prospective independent evaluation readiness",0,0,"0 candidate records; 0/270 clusters","no new score","metadata-only role/exposure review","data_blocked","E068-E072","No registered clean independent cluster; do not infer no future data exists.")
inv("CORE_B_N5","Statistical interval closeout/readiness",0,0,"30/30 scope rows; 30/30 historical endpoints replayed","no training/bootstrap this audit","fixed 3 seeds, 603 already-exposed groups","complete_not_confirmatory","E056-E067","Group-only, seed-only, crossed intervals kept distinct.")
inv("CORE_B_V2_SALTED","Superseded dataset-salted V2 split",78,None,"36 sealed prediction jobs; 6 missing; score values excluded","invalidated attempt","same historical dev exposure","invalidated_before_selection","E073-E074","Retained as exposure; not a separate experiment/replication.")
inv("CORE_B_V2_CORRECTED","Five-task specialization/waypoint V2",78,78,"36 screening + 36 formal + 6 waypoint fits accepted","3 seeds; five tasks","47 dev groups; session independence unverified","complete_development_only","E073-E082","Position macro specialist effect negative; visual specialist effect positive but recipe/capacity mixed.")
inv("CORE_B_V3_ORIGINAL54","V3 state-information/video matrix",54,54,"54/54 prediction bundles; 18/18 Stage0","3 seeds","603-group dev reused","complete_development_only","E039;E105","Anchor branch retains current command; no independent confirmation.")
inv("CORE_B_V3_CURRENT","V3 frozen shared residual/bias current root",None,None,"preseal main prediction receipt only; no final acceptance at snapshot","current stage unsealed","live status snapshot; no result table used","unresolved_active_unsealed","E106-E111","Do not restart, extend or claim outcome from preseal status.")
inv("CORE_B_VIDEOMAE_COREB","VideoMAE CoreB saved-prediction scoring",12,None,"24 dev prediction bundles independently scored; 24 train saved bundles rescored","3 seeds × H30/H60","603 exposed dev groups","accepted_with_caveats","E083-E086","Scoring accepted saved outputs; checkpoint replay 0/12 regenerated.")
inv("CORE_B_VIDEOMAE_PORTABLE","VideoMAE checkpoint replay/portable bridge",0,0,"12 CPU zero-input comparisons failed exact-zero tolerance","no retraining","not a matched natural video replay","incomplete_needs_fix","E083;E087","No accepted portable integration; zero-input branch does not test full CoreB video path.")
inv("CORE_B_ROUTE_C","Route C attribution controls",None,0,"lock/recovery seam QA only in checked scope","not launched full matrix","no new scored matrix found","prepared_not_executed_or_partial","E090","Do not treat locked protocol as completed inference.")
inv("CORE_B_HISTORY_VIDEO","History/video information matrix",4,4,"4/4 accepted","single seed","development only","complete_exploratory","E091-E092","History effect differs for state-only versus fusion; FD trade-off.")
inv("CORE_B_HISTORY_REPLICATION","History replication and encoder adaptation gate",12,12,"A 12/12; B trigger false","3 seeds for A","dev only","complete_A_B_not_run","E093","Adaptation branch not launched at this gate.")
inv("CORE_B_ADAPTATION_B","Independent encoder adaptation B",6,6,"first 0/6 predictions; later 6/6 inference recovery","single seed","614 old dev cohort","candidate_gate_fail","E094-E095","Level gains did not clear temporal-fidelity/low-motion gates.")
inv("CORE_B_JY02","JY02 residual temporal decoder",None,None,"Stage1/Stage2 acceptances present; one initial failure preserved","development","614 legacy dev data; not independent","complete_exploratory","E103-E104","Not used to justify another residual matrix.")
inv("CORE_B_EVENT_FALLBACK","Hand event fallback and hybrid",6,6,"6/6 matrix; 1 exploratory seed","single seed","event opportunities have separate denominator","candidate_gate_fail","E096","AUPRC and false alarms/hour stay an auxiliary event result.")
inv("CORE_B_PROTOTYPE","Calibrated prototype mixture",2,2,"2/2 screen cells; alpha selected 0","single seed","dev only","calibration_gate_fail","E097","No calibrated mixture gain.")
inv("CORE_B_SELECTOR","Expert advantage selector",4,4,"4/4 CPU/OOf cells; no candidate gate","single seed","dev only","candidate_gate_fail","E098-E100","Oracle headroom is not a usable selector.")
inv("CORE_B_DCT","DCT dual-loss factorial",4,4,"4/4 screen cells","single seed","dev only","candidate_gate_fail","E099;E101","Native TV ratio remains small.")
inv("CORE_B_OBSERVATION_DELAY","Observation delay robustness",None,None,"three-seed development analysis accepted","3 seeds","same exposed data","complete_development_only","E102","Delay robustness does not establish independent forecasting.")
inv("CORE_B_COMMAND_QUERY_HANDOFF","N2/N3/N4/N5 final handoff",0,0,"handoff checks only","not an experiment","same exposed dev","complete_no_new_query","E088-E089","COMMAND28 future protocol prepared_not_executed.")
inv("CORE_B_CONTRACT_REPAIR","Benchmark contract repair",0,0,"engineering repair receipts","no new scientific fit","schema/acceptance only","engineering_complete","E016;E082","Do not count contract fixes as scientific gains.")

# 12. Comparison-contract ledger: prohibit cross-contract rankings.
CONTRACTS={
"CA_BATCH48_MATCHED":{"scope":"CoreA 120-class task recognition; B48 dynamic vs repeated-first-frame static","data_role":"development/validation; used for early stopping/selected-best model selection","support":{"episodes_per_seed":617,"seeds":[20260825,20260826,20260827],"session_independence":"unknown"},"inputs":{"dynamic":"multi-frame video stream","control":"repeated first frame"},"target":"120-class task label","selection":"selected-best epoch on the same development/validation split","metric":["Top-1","macro-F1"],"uncertainty":"class-stratified and hierarchical bootstrap; fixed training seeds; Holm for Top-1","comparison_status":"matched_accepted","unknowns":["acquisition/session cluster hash not established in this audit","final-test not selected or evaluated"],"source_refs":["E001","E002","E003","E004","E005"]},
"CA_FULL120_LORA_SINGLE_SEED":{"scope":"CoreA pooled/attentive × dynamic/static × frozen/LoRA plus state control","data_role":"development/validation; reused for model selection","support":{"episodes":617,"seeds":1,"seed":"20260909"},"target":"120-class recognition labels","selection":"validation-selected checkpoints","metric":["Top-1","macro-F1"],"uncertainty":"paired episode bootstrap conditional on one trained seed; multiple comparisons/Holm","comparison_status":"comparable_with_caveats","unknowns":["one training seed","not independent final/test","does not identify trajectory-prediction effect"],"source_refs":["E006","E007","E008","E009"]},
"CB_B1_B2_B3_OLD_614":{"scope":"CoreB B1 baseline, B2 observation sampling, B3 video/fusion × dynamic/static × frozen/LoRA","support":{"accepted_B1_cells":7,"B2_cells":6,"B2_true_pixel_cells":4,"B3_cells":8,"B3_new_fits":6,"B3_reused_controls":2,"legacy_prediction_rows":614,"target_shape":"(614,30,28)"},"role":"development","horizon":"30 steps / 1 second, 4-second prefix for B2/B3","normalizer":"train-only; root-specific hashes retained","B2_sampling":{"0.125s":"proxy/reused dense: pixel_level_sampling=false; 32 observation frames","0.25s":"true pixel-level; 16 unique observation frames","0.5s":"true pixel-level; 8 unique observation frames"},"B3":"single-seed exploratory; later B3 candidate gate not passed","comparison_status":"different_contract_across_B1_B2_B3","unknowns":["not the later N2/N4 603-group cohort","physical channel units/URDF unknown"],"source_refs":["E013","E014","E015","E016"]},
"CB_D3_EQ_SCHEDULER_H30":{"scope":"TS11-D vs STATE-CTRL × H30/H60, old as-run vs corrected scheduler","data_role":"train 2712 groups; dev 603 groups, repeatedly exposed","objective":"corrected per-sample episode weights W/(E*n_e), weighted sum divided by actual batch size","scheduler":"one call after each optimizer update; 11,120 updates/calls per formal cell","metrics":["Full28 standardized MAE","native future-internal FD"],"selection":"epoch-40 fixed last checkpoint; no dev winner selection in corrected scheduler matrix","comparison_status":"matched_within_corrected_matrix; old D3 as-run is different_contract","unknowns":["scheduler not established as unique causal root","current units are numeric command space only"],"source_refs":["E023","E024","E026","E027","E028","E029"]},
"CB_N3_HISTORY_H30":{"scope":"FULL/OBS/ANCHOR × N/D/S; 54-view historical multi-horizon source with 18-view H30 direct reuse in N3","support":{"historical_views":54,"N3_direct_reused_views":18,"episodes":603,"source_groups":603,"seeds":3},"metric":"Full28 standardized MAE and component metrics","aggregation":"same accepted group and model rows; no N3 rescore","comparison_status":"comparable_with_caveats","unknowns":["ANCHOR retains current-command/anchor branch; not strict video-only","source groups do not prove session independence"],"source_refs":["E039","E044","E045","E046","E053"]},
"CB_N2_QUERY_CUDA_COMMON":{"scope":"TS11-D and STATE-CTRL × Q0/Q1; N2 CPU_COMMON/CUDA_COMMON bridge","support":{"seeds":[20261001,20261002,20261003],"episodes":603,"source_groups":603,"anchors":2322},"inputs":"Q0 vs Q1 accepted query definitions; same target/normalizer and task inputs","metrics":["position_standardized_mae","safe_fd_normalized"],"pairing":{"init_state_hash":"6/6 Q0/Q1 pairs pass","sample_order":"unverified; no accepted sampler/order hash"},"uncertainty":"fixed-seed group-only, exact 27-triple seed-only, and crossed paired seed×group; no p-values","comparison_status":"comparable_with_caveats","source_refs":["E041","E042","E058","E059","E061","E062","E063","E064"]},
"CB_N4_ERROR_BUDGET":{"scope":"12 fixed accepted models; N2/N3 reused dev predictions and target-blind train-only analysis forwards","support":{"dev_groups":603,"dev_rows":2322,"analysis_bundles":24},"metrics":["position_standardized_mae","safe_fd_normalized L1","secondary level/time-centered L2","future-GT oracle diagnostic"],"uncertainty":"crossed seed×source-group intervals; fixed models; repeated dev","comparison_status":"matched_within_locked_N2/N4_contract","unknowns":["GT-oracle not deployable; L2 budget not an L1 additive attribution"],"source_refs":["E053","E054","E055","E057"]},
"CB_COMMAND28_H30_RIDGE":{"scope":"28D future command target; CPU residual Ridge H30","support":{"train_groups":2712,"dev_groups":603,"dev_rows":2322},"input":"72 past values per field (32×71 packed history minus duplicated anchor) plus anchor; no video or future input","objective":"episode-equal L2 Ridge; lambda grid [0.001,0.01,0.1,1.0], selected boundary 1.0","metric":["position standardized MAE","safe-FD normalized"],"comparison_status":"comparable_with_caveats","unknowns":["different squared-error objective/capacity than neural heads","dev repeatedly exposed","physical units unknown"],"source_refs":["E047","E048"]},
"CB_OBS43_FULL43":{"scope":"43D recorded observation transition, separate from 28D command forecast","support":{"train_groups":2712,"dev_groups":603,"dev_rows":2322},"target":"next observation velocity; one-step transition rolled to H30","oracle":"future command input is privileged and not deployable","comparison_status":"different_contract_from_command28","unknowns":["command/observation pairing not asserted","availability timestamp absent; known-anchor recorded-clock assumption","physical units/anatomy unverified"],"source_refs":["E049","E050","E051","E052"]},
"CB_V2_TASK_MACRO":{"scope":"five task IDs 119/94/90/141/230; G-S/G-V/S-S/S-V/waypoint","support":{"dev_episode_groups":[12,12,11,5,7],"dev_total_episodes":47,"h30_anchors":185,"seeds":3,"task_141_groups":5,"task_230_groups":7},"aggregation":"within episode window mean; episode equal; five task macro; fixed three-seed summary","fit":{"screening":36,"formal":36,"waypoint":6,"corrected_split_attempts":78,"total_historical_attempts":164},"metrics":["position_mae","safe_fd_l1","seam_safe_fd_l1"],"comparison_status":"different_contract_between_global_and_specialist_recipes","unknowns":["model capacity/LR/epoch/data sharing differ","session/acquisition independence unverified","native action units unknown"],"source_refs":["E073","E074","E075","E076","E077","E078","E079","E082"]},
"CB_VIDEOMAE_VJEPA_SYSTEM_LEVEL":{"scope":"VideoMAE fusion vs V-JEPA fusion and each state-control, H30/H60","support":{"dev_groups":603,"dev_rows":2322,"seeds":3,"raw_regions":["action[0:14]","action[14:28]","action[0:28]"]},"common":{"target_normalizer_hashes":"match","IDs":"match","optimizer_updates":11120,"batch":64,"objective":"episode-equal per-sample"},"differences":["video resolution/pooling/backbone dimensions differ","V-JEPA cache bridge differs on 64/20079 train rows in bounded reference comparison","STATE-CTRL model outputs differ substantially although shared nonvideo init matches"],"metric":["standardized_mae","safe_fd_normalized"],"replay":{"coreA":"3/3 seeds × train/validation exact logits replay","coreB":"0/12 regenerated at original checkpoint gate; saved prediction scoring accepted","portable_integration":"needs_fix"},"comparison_status":"comparable_with_caveats","source_refs":["E083","E084","E085","E086","E087"]},
"CB_OBS14_POOL":{"scope":"prospective independent command evaluation candidate admission","support":{"required_clusters":270,"registered_candidate_records":0,"strict_independent_clusters":0,"gap":270},"role":"metadata only; no payload, target, or score access","comparison_status":"unverified_pool_not_ready","unknowns":["reserved pool status UNKNOWN","session/alias lineage not established","unregistered future intake outside checked scope"],"source_refs":["E066","E067","E068","E069","E070","E071","E072"]},
"CB_V3_CURRENT":{"scope":"CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z working root","contract":"frozen shared/task residual and bias protocol; current result table not accepted/sealed at snapshot","support":"only preseal status/readout metadata inspected; numeric predictions/targets not opened","comparison_status":"unverified_unsealed_working_root","unknowns":["final acceptance absent at snapshot","live process state not queried","phase status may change after audit snapshot"],"source_refs":["E106","E107","E108","E109","E110","E111"]}
}

# Claims C01-C16 are adjudicated independently; status does not inherit from engineering QA.
CLAIMS=[
("C01","CoreA dynamic/static and encoder claims","mixed","Batch48 matched dynamic-static Top-1 +1.891pp (95.624% vs 93.733%, 3/3 seeds, Holm p=.00240); macro-F1 +.234pp (86.990% vs 86.756%, 2/3 seeds, hierarchical CI crosses zero). Full120 LoRA is a different single-seed validation matrix: best absolute video cell S-P-L 97.083%/92.992%, not a dynamic claim.","COREA_B48_TOP1;COREA_B48_MACRO_F1;COREA_FULL120_D-P-L_vs_D-P-F_TOP1;COREA_FULL120_S-P-L_vs_S-P-F_MACRO_F1","E001-E010","Do not infer action anticipation; pooled, attentive, R3D and VideoMAE references are not all paired under one contract; final test not accessed.","Keep one matched Batch48 contrast as modest classifier evidence; stop head search per acceptance."),
("C02","B1/B2/B3, Route C, encoder adaptation","mixed","B1 accepted 7/7 but persistence Arm14 standardized MAE .126969 beats B-STATE .165614. B2 accepted 6/6, only .25/.5 s four cells are true-pixel; .125 s two are proxy/reused dense. B3 corrected 8/8 acceptance, 6 fits +2 reused, but single-seed candidate gate not passed. Route C full matrix not located as accepted; encoder adaptation B recovered 6/6 predictions but gate failed.","CB_B1_PERSISTENCE_VS_STATE_ARM14","E011-E016;E037;E090;E093-E095","B1/B2 use a distinct 614-row legacy cohort; do not pool with later 603-group command studies. B3 single seed and selected-head reuse.","No automatic B3 confirmation or Route C relaunch."),
("C03","Video gain by state/history; original 54 vs N3 18","mixed","N3 reuses 18 accepted H30 views from a separate 54-view V3 matrix. Derived fixed-seed Full28 H30 D-CHRON gain (N-video error minus D-CHRON error) is FULL +.000631, OBS -.003568, ANCHOR +.004195. The pattern is nonmonotonic; ANCHOR retains the command/anchor branch.","N3_H30_FULL_DCHRON_VS_NVIDEO_FULL28;N3_H30_OBS_DCHRON_VS_NVIDEO_FULL28;N3_H30_ANCHOR_DCHRON_VS_NVIDEO_FULL28","E039;E044-E046;E053","Three fixed seeds; 603 dev groups; session independence unknown. Anchor is not strict video-only.","Separate conditions; do not claim less history monotonically improves vision or video is useless."),
("C04","D3 actual objective and scheduler correction","mixed","Historical trainer used a reduced batch loss multiplied by mean batch weights and called the scheduler once per epoch; this is not the corrected per-sample episode-equal objective. Corrected code applies sum(w_i*l_i)/B_actual and scheduler calls after optimizer updates (11,120/11,120). Yet H30 corrected-vs-old gain is TS11-D -.003134 (CI [-.003831,-.002435]) and STATE-CTRL +.001662 ([+.001073,+.002247]); video-specific scheduler interaction -.004796 ([ -.005671,-.003923]).","D3EQ_H30_TS11_D_SCHEDULER_POSITION;D3EQ_H30_STATE_CTRL_SCHEDULER_POSITION;D3EQ_H30_VIDEO_SPECIFIC_SCHEDULER_INTERACTION","E020;E023-E029","Corrected implementation gate passed; scientific outcomes are development-only, fixed seeds, and do not identify scheduler as sole cause.","Do not merge legacy and corrected recipes or repeat scheduler sweeps."),
("C05","Under-dynamic claim by model/split/metric","supported_within_scope","For corrected TS11-D/H30, a seed was nearly motionless on both train and dev (TV ratio about 5e-6; direction accuracy .203/.209); corrected model group means commonly retain TV ratios .09-.14. HDI INCREMENT improves position but Hand TV ratio only .133 and Hand future-internal FD is worse. Magnitude/TV is not tracking accuracy.","E027;E034-E035;E037;E053","Train/dev evidence for specific model/seed and accepted strata; low TV does not prove decoder failure, irreducible noise, or multimodality. Preserve event support/opportunity denominator separately.","Use alignment, direction, FD and event denominators with amplitude; no TV-only loss change."),
("C06","Direct/increment, RFSI, HDI acceptance scope","mixed","HDI accepted six formal cells, 12 train/dev predictions plus 6 source-model predictions, and 18 semantic bundles. Increment lowers Hand native level MAE .090962→.088679, but Hand native FD rises .020978→.021161; all three seeds worsen FD and candidate gate fails. RFSI Hand/Arm names were corrected in derived analysis; units/URDF/FK/Cartesian remain unknown.","HDI_INCREMENT_VS_DIRECT_HAND_NATIVE_LEVEL_MAE;HDI_INCREMENT_VS_DIRECT_HAND_NATIVE_FUTURE_INTERNAL_FD","E030-E035","Row-wise anchor validation corrected the stale frame119/global-validator issue; CPU source replay accepted at locked 1e-6. Slice mapping is project index convention, not physical semantics.","Do not promote INCREMENT; keep raw action slices and FD/position trade-off."),
("C07","VideoMAE backbone vs net visual gain","mixed","CoreA VideoMAE frozen probe validation Top-1 .943274, macro-F1 .882317; no matched frozen V-JEPA probe in bounded references. CoreB saved-prediction independent scoring passed on 603 groups, but fresh checkpoint replay was 0/12 regenerated and portable integration needs_fix. H30 Full28 VideoMAE−V-JEPA error delta is -.004040 MAE and -.006418 safe-FD; its own G_VM state-control gains are -.000522 and -.000619, so lower error than V-JEPA does not mean net visual benefit. Team STATE-CTRL H30 SD is .0008394577, not .008839.","VIDEOMAE_Delta_backbone_H30_STANDARDIZED_MAE;VIDEOMAE_Delta_backbone_H30_SAFE_FD_NORMALIZED;VIDEOMAE_G_VM_H30_STANDARDIZED_MAE;VIDEOMAE_G_VM_H30_SAFE_FD_NORMALIZED","E083-E087","Intervals condition on fixed three seeds and repeatedly exposed dev; V-JEPA cache bridge has 64/20079 differing train rows; full CoreB fresh replay/integration not accepted.","Retain saved-score comparison with explicit status layers; resolve safe export/cache provenance before any replay claim."),
("C08","N0 REAL/ZERO/PERM","supported_within_scope","N0 accepted 6/6 real replays and 12/12 zero/permutation interventions. This shows fixed models respond to those perturbations; ZERO/PERM leave the natural joint input distribution and do not prove extra information or visual causality.","E037-E038","No new fits; no confidence interval; perturbation response is functional dependence, not net visual benefit.","Do not use intervention sensitivity to reopen query/fusion line."),
("C09","Hadamard/query seed stability","mixed","N5 G_TS fixed-three-seed means are +.001121 position and +.005342 safe-FD. Group-only fixed-seed intervals are positive; exact-27 seed-only and crossed seed×group intervals cross zero; position G_TS seed gains are -.008257,+.014806,-.003185 (1/3 positive). Dropping seed02 makes both query means negative. Q0/Q1 init hashes pair, sample-order hash is absent.","N2_position_standardized_mae_G_TS;N2_safe_fd_normalized_G_TS","E040-E043;E056-E065","Intervals cover different uncertainty axes; not confirmatory. Training pairing is nominal, not fully verified.","Keep query line closed; do not use I or rank as a causal root."),
("C10","N2 backend and N4 error-budget interpretation","supported_within_scope","N2 accepted 12 CPU_COMMON +12 CUDA_COMMON views; same-model backend drift is small. For TS11-D seed02, Q0→Q1 position MAE improves .209936→.195131 and safe-FD .976491→.956604; level L2 and time-centered L2 both fall, with centered component slightly larger. Dynamic tracking remains limited (r=.257,c=.266, skill=.071).","N4_S02_TS11D_POSITION;N4_S02_TS11D_SAFEFD","E040-E043;E053-E057","One of three seeds drives the improvement; future-GT oracle is diagnostic only; L2 budget is not an additive L1 decomposition.","Correct the old “pure level rescue” account; do not promote query."),
("C11","N5 interval scope","supported_within_scope","N5 accepted group-only, seed-only exact 27 ordered triples, and crossed paired seed×group tables. Historical N2/N4 30 endpoints replayed within max 2.29e-16. N5 group-only G_TS is positive; seed-only/crossed sensitivities cross zero. Init hashes pass 6/6, sampler/order pairing unverified.","E056-E065","No new bootstrap in this audit; N5 intervals are retrospective fixed-model/development sensitivities; no p-values.","Use group-only and seed/crossed scope labels explicitly; three seeds are not future training-population certainty."),
("C12","Ridge/horizon/OBS43 future command oracle","supported_within_scope","H30 command28 Ridge position MAE .230365 and safe-FD 1.044448 are worse than persistence .223135/.976491 and Q0 state-control .195021/.953524. In distinct OBS43 task, HOLD and ORACLE both remain worse than OBS persistence; oracle gain over HOLD is diagnostic privileged future command and cannot be deployed.","H30_RIDGE_VS_PERSISTENCE_POSITION_STANDARDIZED_MAE;OBS43_HOLD_VS_OBS_PERSISTENCE_POSITION_STANDARDIZED_MAE;OBS43_ORACLE_VS_HOLD_POSITION_STANDARDIZED_MAE","E047-E052","OBS43 is a 43D recorded observation target, not a command28 substitute; scales/tasks differ. Teacher is one-step, not H30.","Do not train D5 or rank OBS43 against command28 without a new supported driver."),
("C13","OBS14 independent replication and pool gap","supported_within_scope","Latest bounded metadata resolution has 0 registered candidate records, 0 strict clean acquisition clusters, 270 required; 91 CoreA nonfinal groups have unknown CoreB exposure and are excluded; final aggregate 614 was not enumerated.","E066-E072","Readiness is NOT_READY; no score/forecast; no claim that no unregistered/future pool exists anywhere.","Specific blocker is authenticated metadata/data intake, not another analysis of the same registry."),
("C14","Corrected five-task V2 and task119 waypoint","mixed","Five-task macro D_S position -.003717 and safe-FD -.009081; D_V +.000392/-.008059; V_G -.000783/+.011053; V_S +.003326/+.012075; I +.004109/+.001022. Task119 GOAL_COND vs AUX_ONLY B_GAIN is -.001293 position and -.000314 safe-FD with mixed per-seed signs. 47 dev groups total; task141 has 5 groups; session independence unverified.","V2_ALL5_D_S_POSITION_MAE;V2_ALL5_D_S_SAFE_FD_L1;V2_ALL5_V_S_POSITION_MAE;V2_ALL5_V_S_SAFE_FD_L1;V2_TASK119_GOAL_COND_VS_AUX_ONLY_POSITION_MAE;V2_TASK119_GOAL_COND_VS_AUX_ONLY_SAFE_FD_L1","E073-E082","Specialist recipes differ in capacity/LR/epochs/data sharing, so no pure negative-transfer causal attribution. Existing ALL_SEED_LOO drops task_id, not seed.","Keep V2 as development evidence; do not promote task119 waypoint goal gain."),
("C15","V3 frozen shared residual/bias run status","unresolved","A new 2026-10-14 root exists and has preseal main CPU prediction acceptance plus a recorded RGB-probe phase start; no final sealed acceptance was found at the audit snapshot. No result metrics were opened or used.","E106-E111","Working root is unsealed; live process/owner state was not queried; files can change after snapshot.","No duplicate or extension in this audit; review only a later terminal, hash-stable accepted package."),
("C16","Physical semantics, role and exposure boundaries","mixed","Stable references are action[0:14], action[14:28], action[0:28]. CoreB Motion-Fidelity cross-check supports project observation↔action half correspondence but not URDF joint IDs, units, FK, Cartesian position or robot success. Session/acquisition independence remains unknown across dev groups.","E030-E035;E049-E050;E066-E072;E083-E086","Current audit raw-array and final/heldout access all zero; historical target/scores were exposed in specified studies, and missing telemetry remains UNKNOWN.","Do not relabel standardized/native numbers as physical accuracy or treat current zero access as historical blindness."),
]

# contract warnings and root-registry generation
ROOTS=[]
def add_root(root_id,study_id,kind,status,path,owner="UNKNOWN",alias="",acceptance="",scope="",source=""):
    p=PROJECT/path
    exists=p.exists()
    stat=p.stat() if exists else None
    ROOTS.append({"root_id":root_id,"study_id":study_id,"root_kind":kind,"observed_status":status,"exists":exists,"root_path":path,"owner_state_reported":owner,"acceptance_or_seal_time":acceptance,"alias_or_parent":alias,"checked_scope":scope or path,"source_pointer":source,"mtime_utc":(datetime.fromtimestamp(stat.st_mtime,timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z") if stat else "NA"),"metadata_bytes":(stat.st_size if stat else "NA")})
add_root("ROOT_PROJECT","PROJECT","root","present","")
add_root("ROOT_REPO","REPO_ROOT","reference_tree","nonempty_but_git_identity_unknown","repo/ee6008-humanoid-vjepa","UNKNOWN; git executable not found","","AGENTS.md describes placeholder but current report/code trees are populated","bounded report roots","git rev-parse returned command-not-found")
add_root("ROOT_CA_B48_EXPECTED","CA_BATCH48","missing_anchor","not_located_at_project_relative_path","reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z","NA","ROOT_CA_B48_CANONICAL","NA","exact requested project-relative path","E001-E005")
add_root("ROOT_CA_B48_CANONICAL","CA_BATCH48","canonical_sealed","complete_with_caveats","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z","UNKNOWN","","acceptance in RUN_ACCEPTANCE.json; report says 12/12","selected report and aggregate/per-seed metrics","E001-E005")
add_root("ROOT_CA_B48_BLOCKED","CA_BATCH48","attempt","blocked_harvest","repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_20260901T033748Z","unknown","ROOT_CA_B48_CANONICAL","blocked report only","blocked attempt files only","bounded directory inventory")
add_root("ROOT_B3_STALE_CONTROLLER","B3","stale_snapshot","running_snapshot_superseded_by_sealed_reacceptance","repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z/results/B3_CONTROLLER_STATUS.json","stale_running","ROOT_B3_CANONICAL","later B3_REACCEPTANCE 8/8","status JSON only","E015")
add_root("ROOT_B3_CANONICAL","B3","sealed_analysis","accepted_8_of_8_single_seed","research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8","UNKNOWN","ROOT_B3_STALE_CONTROLLER","analysis QA passed 2026-09-10T07:04:28Z","selected acceptance and result summaries","E011-E016")
for rid,fid,path,stat in [
("ROOT_D1D2","D1_D2","reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z","sealed baseline/learned summaries"),
("ROOT_D4","D4","reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z","complete with caveats"),
("ROOT_RFSI","RFSI",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/RAW_FRAME_ORDER_STATE_INFO_V1_20260924T172133Z'),"development-only accepted"),
("ROOT_MOTION","MOTION_FIDELITY",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/CORE_B_MOTION_FIDELITY_AUDIT_V1_20260925T064134Z'),"corrected analysis accepted; fit unresolved"),
("ROOT_HDI","HDI",os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'core_b/HAND_DIRECT_INCREMENT_V1_20261006071115Z'),"harvest child accepted; gate fail"),
("ROOT_N2","N2","reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z","complete"),
("ROOT_N3","N3","reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z","partial parent/N3 complete"),
("ROOT_H30","H30","reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z","complete with stale owner pointer"),
("ROOT_OBS43","OBS43","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z","sealed"),
("ROOT_OBS43_EMPTY","OBS43","reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161856Z","empty placeholder; unknown"),
("ROOT_N4","N4","reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z","complete"),
("ROOT_N5","N5","reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z","complete; readiness not ready"),
("ROOT_OBS14","OBS14","reports/benchmarks/CORE_B_OBS14_PROVENANCE_POOL_RESOLUTION_V1_20261013T054028Z","data blocked"),
("ROOT_V2","V2","reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z","complete development only"),
("ROOT_VMAE_ACCEPT","VIDEOMAE","reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z","partial acceptance"),
("ROOT_VMAE_V2","VIDEOMAE","reports/benchmarks/VIDEOMAE_CORE_B_HANDOFF_V2_20261013T065841Z","incomplete CPU-zero replay"),
("ROOT_V3_CURRENT","V3","reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z", "unsealed working root at snapshot"),
("ROOT_AUDIT","CURRENT_AUDIT","reports/audits/EE6008_CONCLUSION_AUDIT_V1_20261014T061114Z","new audit root")]:
    add_root(rid,fid,"canonical_or_working",stat,path,"UNKNOWN","","acceptance pointer in selected evidence","named report/acceptance files",fid)

# The new V3 root gets an explicit phase snapshot; do not claim the process is live.
v3_root=PROJECT/"reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z"
v3_final=v3_root/"qa/FINAL_ACCEPTANCE.json"
v3_preseal=PROJECT/"reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/qa/MAIN_PREDICTION_ACCEPTANCE_PRESEAL.json"
v3_rgb=PROJECT/"reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/coordination/PHASE_STATUS_RGB_PROBE.json"
v3_phase={}
if v3_rgb.is_file():
    try:v3_phase=read_json(v3_rgb.relative_to(PROJECT).as_posix())
    except Exception:v3_phase={}
V3_FINAL_AT_SNAPSHOT=v3_final.is_file()
V3_PRESEAL_STATUS=read_json(v3_preseal.relative_to(PROJECT).as_posix()).get("status") if v3_preseal.is_file() else "absent"
v3_final_status=(read_json(v3_final.relative_to(PROJECT).as_posix()).get("status") if V3_FINAL_AT_SNAPSHOT else None)
v3_status=("sealed_final_acceptance_found_at_snapshot" if V3_FINAL_AT_SNAPSHOT else "working_root_unsealed_preseal_only")

# Exposure and outcome audit: distinguish this audit's actions from historical exposure.
AUDIT_COUNTS={
"NEW_TRAINING_FITS":0,"NEW_OPTIMIZER_STEPS":0,"MODEL_FORWARD":0,"NEW_ENCODER_EXTRACTION":0,
"NEW_SCORING":0,"NEW_RAW_ARRAY_SCORING":0,"NEW_BOOTSTRAP":0,"RAW_PREDICTION_ARRAY_READ":0,
"RAW_TARGET_ARRAY_READ":0,"NEW_POOL_NUMERIC_ACCESS":0,"FINAL_TEST_ACCESS":0,
"FINAL_HELDOUT_ACCESS":0,"FINAL_HELDOUT_PAYLOAD_ACCESS":0,"OLD_ROOT_WRITES":0,
"GT_DERIVED_STATISTICS_READ":True
}
HISTORICAL_ACCESS=[
{"study":"CoreA Batch48","known":"development/validation logits and labels used for accepted metrics/early stopping; final-test selection/evaluation recorded 0","source_refs":["E001","E004","E005"]},
{"study":"CoreA Full120 LoRA","known":"617 development/validation examples scored; final-test_access=0 in accepted matrix; dev used for model selection","source_refs":["E006","E009"]},
{"study":"CoreB N2/N3/N4","known":"603 development groups/2322 rows previously scored; N2 target access after 24-view acceptance; N5 later read 14,472 accepted GT-derived source-group loss values","source_refs":["E041","E044","E053","E064","E065"]},
{"study":"OBS43","known":"dev labels were read in the historical OBS43 diagnostic; first raw-label read time UNKNOWN in the source receipt; final heldout 0","source_refs":["E049","E051","E052"]},
{"study":"V2 task specialization","known":"three scoring attempts accumulated 16,650 dev target-row reads (5550 each); the corrected score was after 42/42 prediction seal; invalid attempt values excluded; final/heldout 0","source_refs":["E073","E074"]},
{"study":"OBS14","known":"metadata-only pool resolution; no forecast/score; final aggregate 614 groups only, IDs not enumerated","source_refs":["E068-E072"]},
{"study":"Other historical telemetry","known":"no single project-wide ledger proves all historical label exposures; unknown telemetry remains UNKNOWN","source_refs":["E065","E074","E083"]}
]
EXPOSURE={
"schema_version":1,
"observed_at_utc":utc_now(),
"access_role_scope":"only named accepted reports, manifests, acceptance records, implementation text and admitted aggregate metadata; no raw data inventory",
"protected_roles":[{"role":"final_test","count":614,"count_scope":"OBS14 legacy split aggregate only","ids_enumerated":False,"payload_read_this_audit":0,"evidence":"E070"},{"role":"unknown_role_payloads","count":None,"ids_enumerated":False,"payload_read_this_audit":0,"evidence":"not opened; source role not certified"}],
"candidate_pool":{"study":"OBS14 R2_recorded_schema_external","registered_candidate_records":0,"verified_clean_acquisition_clusters":0,"required_clusters":270,"gap":270,"reservation_status":"UNKNOWN","alias_session_lineage":"UNKNOWN","score_or_target_access":0,"status":"NOT_READY","scope":"bounded accepted metadata registries only","evidence_refs":["E066","E067","E068","E069","E071","E072"]},
"historical_access":HISTORICAL_ACCESS,
"historical_access_unknowns":["project-wide historical label-access telemetry is incomplete","OBS43 first raw target materialization time is recorded UNKNOWN","no claim of historical blind analysis is made"],
"GT_DERIVED_STATISTICS_READ":{"value":True,"current_audit_sources":["E002","E003","E004","E006","E007","E016","E018-E022","E027","E030-E046","E047-E059","E073-E086"],"note":"accepted development metric/loss summaries are outcome-derived and previously exposed; this audit did not read per-group raw scores or re-score predictions"},
"actual_this_audit":AUDIT_COUNTS,
"excluded_inputs":["all prediction NPZs","all target NPZs and V2 DEV_TARGETS_SCORER_ONLY.npz","N5 draw NPZ files","N5 accepted derived loss-sum NPZ","raw Parquet/video/frame payloads","feature caches/banks","checkpoints/model weights","final/held-out payloads","unregistered/unknown-role candidate payloads"],
"no_gpu_or_external_actions":{"GPU_commands":0,"CUDA_or_NVML_initialization":0,"scheduler_or_lease":0,"model_or_scoring_process":0,"downloads":0,"external_contacts_or_uploads":0},
"source_boundary":{"raw_roots_modified":0,"selected_source_files_verified":"SOURCE_SCOPE.csv","historical_full_manifest_verification_reported":"acceptance/manifest claims are referenced by scope; no full-root rehash in this audit"}
}

CLAIM_FIELDS=["claim_id","historical_claim","current_verdict","current_conclusion","key_comparison_ids","evidence_ids","effect_direction_and_denominator","seed_and_support","uncertainty_scope","limits","next_implication"]
CONTRACTS["CA_BATCH48_VS_BATCH32_DIFFERENT_BATCH"]={"scope":"matched task/class/dataset family; batch size 32 versus 48","support":"617 development/validation episodes; 3 standard seeds","metric":"macro-F1","uncertainty":"class-stratified plus hierarchical bootstrap; Holm correction","status":"comparable_with_caveats","caveats":["selected-best model; validation exposure","hierarchical macro-F1 interval crosses zero","same task, but differing effective batch and historical throughput"],"source_refs":["E001","E005","E006"]}
CONTRACTS["CA_FULL120_LORA_SINGLE_SEED"]={"scope":"CoreA Full120 eight video cells and STATE-ONLY control","support":"120 classes, 2757 train/617 dev; seed 20260909","metric":"Top-1 and macro-F1","status":"comparable_with_caveats","caveats":["single training seed","development/validation selected checkpoints","no action-anticipation inference"],"source_refs":["E006","E007","E008","E009"]}
CONTRACTS["CB_B1_OLD_614"]={"scope":"B1 accepted diagnostic and persistence reference","support":"614 legacy canonical IDs; H30 30x28","metric":"standardized Arm14 MAE/FD","status":"matched_accepted_within_B1","caveats":["do not rank against later N2 603-group cohort","native values are numeric command units"],"source_refs":["E013","E016"]}
CONTRACTS["CB_V3_ORIGINAL54"]={"scope":"FULL/OBS/ANCHOR × N/D/S × H30/H60","support":"54 accepted prediction bundles; seeds 20260953/54/55; primary dev 603 groups","metric":"position MAE/FD/TV and interactions","status":"accepted_development_only","caveats":["ANCHOR includes current-command/anchor token","not independent replication","54 original cells are not 18 direct-reused N3 H30 views"],"source_refs":["E039","E044","E045","E105"]}
CONTRACTS["CB_OBS14_R2"]={"scope":"prospective independent command evaluation candidate admission","support":{"required_clusters":270,"registered_candidate_records":0,"strict_independent_clusters":0,"gap":270},"role":"metadata only; no payload, target, or score access","status":"not_ready","caveats":["no payload or scores read","614 final groups aggregate only; IDs not enumerated"],"source_refs":["E066-E072"]}
CONTRACTS["CB_V2_TASK_MACRO"]={"scope":"five task macro and task119 waypoint","support":{"dev_episode_groups":[12,12,11,5,7],"dev_total_episodes":47,"h30_anchors":185,"seeds":3,"task_141_groups":5,"task_230_groups":7},"aggregation":"within-episode window mean; episode equal; five task macro; mean of 3 seeds","metric":["position_mae","safe_fd_l1","seam_safe_fd_l1"],"status":"comparable_with_caveats_within_V2; different_contract_across_models","caveats":["task macro is not global episode-equal","session/trial independence unverified","corrected split excludes salted prior outputs from selection"],"source_refs":["E073-E082"]}
CONTRACTS["CB_VIDEOMAE_COREB"]={"scope":"VideoMAE fusion vs V-JEPA fusion and each state-control","support":{"dev_groups":603,"dev_rows":2322,"seeds":3,"raw_regions":["action[0:14]","action[14:28]","action[0:28]"]},"common":{"target_normalizer_hashes":"match","IDs":"match","optimizer_updates":11120,"batch":64,"objective":"episode-equal per-sample"},"differences":["video resolution/pooling/backbone dimensions differ","V-JEPA cache bridge differs on 64/20079 train rows in bounded reference comparison","STATE-CTRL model outputs differ substantially although shared nonvideo init matches"],"metric":["standardized_mae","safe_fd_normalized"],"replay":{"coreA":"3/3 seeds x train/validation exact logits replay","coreB":"0/12 regenerated at original checkpoint gate; saved prediction scoring accepted","portable_integration":"needs_fix"},"status":"comparable_with_caveats","source_refs":["E083-E087"]}
CONTRACTS["CB_COMMAND28_VS_OBS43"]={"scope":"two distinct target definitions","command28":"future upper-body action command trajectory (28D)","OBS43":"recorded observation transition (43D)","status":"different_contract_no_ranking","caveats":["different normalizers and units","future-command oracle uses privileged unavailable-at-inference data"],"source_refs":["E047-E052"]}

def write_claims():
    write_csv("CLAIM_LEDGER.csv",CLAIM_FIELDS,[dict(zip(CLAIM_FIELDS,r)) for r in CLAIMS])
def write_contracts():
    write_json("COMPARISON_CONTRACTS.json",{"schema_version":1,"as_of_utc":utc_now(),"status_enum":["matched_accepted","comparable_with_caveats","different_contract","unverified"],"contracts":CONTRACTS})

NEXT_MD="""# 下一步决策：冻结新增模型拟合，等待当前 V3 封存后再分流

**推荐方向：P0_HOLD_NEW_FITS_AND_REVIEW_CURRENT_V3_SEAL。** 本审计观察到 2026-10-14 V3 frozen shared/task residual root 已有 preseal prediction acceptance，但仍没有 final acceptance。状态文件记录 RGB probe 阶段已开始；本审计没有检查进程/GPU、没有等待它结束，也没有读取其预测/target 数组。对同一问题再开一个 residual/gating/query/seed sweep 会重复现有工作。

## 具体瓶颈与最小可证伪问题

C05/C06/C09/C14 支持“level position loss 与变化/FD fidelity 仍存在取舍”；N4 的 S02 改善同时出现在 level 与 time-centered L2，但只有一个 seed 明显，N5 crossed/seed-only intervals跨零；HDI 的 increment 降低 Hand/Arm/Full level error，却使 Hand future-internal FD 均值恶化约 4.71%，预注册 gate 失败。竞争解释包括 task/recipe/capacity 混杂、样本意图/command availability 不足、normalizer/clock/field mapping 不完整，以及固定 dev 的重复选择。

- **最小问题（只在现有 V3 完成并被封存后评估）：** 其锁定 residual/bias 在原 co-primary 上能否同时降低误差，并且不会降低变化对齐/方向？保留 V3 已锁定的 control、task/seed/LOO与零 residual；不增加候选、seed、LR或horizon。
- **支持观察：** 正向 co-primary 同时出现在所有锁定 task 与 seed；预注册 residual 与参数/buffer invariance QA 通过；LOO 不由单一 task/seed 决定。
- **推翻观察：** co-primary 取舍、seed/task 方向混合、外层 base/fresh-zero gate 失败、或 source/metric contract 不匹配。仅 level/TV 改善不足以晋级。
- **停止条件：** V3 final acceptance 未封存、源 hash 不稳定、当前模型数值需要读 raw arrays 才能救援、或任一 co-primary/immutability gate 失败时停止，不追加拟合。

## 备选方向

1. **P1_OBS14_REGISTERED_POOL_INTAKE（信息价值高，实际模型收益未知，数据/人工成本未知）。** 目前严格准入是 0/270。下一项可做的是候选 registry metadata intake：producer/revision/许可，43D→packed-history71/action28 字段和单位，稳定 source/session/trial/episode IDs 与 alias 图，clock/video/anchor资格，以及与 CoreA/N2/N3/N4/HDI/OBS43 的历史曝光交叉表和 reservation 状态。未满足 270 个合规 cluster 前不预测/评分；不要将旧 dev 重切为独立池。
2. **P2_TARGET_CLOCK_AND_SCHEMA_REVIEW（信息价值中等，计算成本低，人工/权限成本未知）。** 取得源 schema/producer clock/command availability 的逐字段证据，保留 action[0:14]、[14:28]、[0:28] 稳定数值引用。若物理单位、通道映射或观测时可用性仍缺失，就不写 physical/Cartesian/action-success 主张。

## 既有资产、成本与分支

可复用 N5 accepted tables、N2/N3/N4 sealed references、V2 corrected root、HDI six-cell prediction acceptance；本审计没有复制模型/数组。新拟合/forward/score/bootstrap 均为 0。当前 V3 的新增模型预算在本审计没有取到一个可用的终态实测值；任何额外 fit、GPU-hour、数据购买和人工 schema work 的增量成本记 estimate_unknown。

若 V3 已完成，先只读核验最终 acceptance/源 hash并使用其真实结果，不再默认重跑V3。若未完成或封存失败，不续跑旧 owner，也不新建同合同任务。若最后瓶颈是独立数据，走 P1 并保持270门槛；这个门槛只属于 OBS14 prospective confirmation，不推广到开发任务。若数据显示幅度低且 alignment 弱，不单纯增加 TV/FD loss 权重；若 history 很强时 visual gain 没有建立，继续区分输入冗余、意图/command 缺失、融合优化和 target/clock 误差，而不宣称视觉无用。

recommendation_confidence=medium-high（关于“现在不应新增拟合”）；decision_flip_condition=当前 V3 出现完整、稳定且两 co-primary 无取舍的 sealed acceptance，或出现可审计独立数据 metadata。NEXT_DIRECTION_EXECUTED=false。
"""
write_text("NEXT_DIRECTION_DECISION.md",NEXT_MD)

def write_text(rel,text):
    p=AUDIT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding="utf-8")

CONTRACTS["CA_BATCH48_VS_BATCH32_DIFFERENT_BATCH"]={"scope":"matched task/class/dataset family; batch size 32 versus 48","support":"617 development/validation episodes; 3 standard seeds","metric":"macro-F1","uncertainty":"class-stratified plus hierarchical bootstrap; Holm correction","status":"comparable_with_caveats","caveats":["selected-best model; validation exposure","hierarchical macro-F1 interval crosses zero","same task, but differing effective batch and historical throughput"],"source_refs":["E001","E005","E006"]}
CONTRACTS["CA_FULL120_LORA_SINGLE_SEED"]={"scope":"CoreA Full120 eight video cells and STATE-ONLY control","support":"120 classes, 2757 train/617 dev; seed 20260909","metric":"Top-1 and macro-F1","status":"comparable_with_caveats","caveats":["single training seed","development/validation selected checkpoints","no action-anticipation inference"],"source_refs":["E006","E007","E008","E009"]}
CONTRACTS["CB_B1_OLD_614"]={"scope":"B1 accepted diagnostic and persistence reference","support":"614 legacy canonical IDs; H30 30x28","metric":"standardized Arm14 MAE/FD","status":"matched_accepted_within_B1","caveats":["do not rank against later N2 603-group cohort","native values are numeric command units"],"source_refs":["E013","E016"]}
CONTRACTS["CB_V3_ORIGINAL54"]={"scope":"FULL/OBS/ANCHOR × N/D/S × H30/H60","support":"54 accepted prediction bundles; seeds 20260953/54/55; primary dev 603 groups","metric":"position MAE/FD/TV and interactions","status":"accepted_development_only","caveats":["ANCHOR includes current-command/anchor token","not independent replication","54 original cells are not 18 direct-reused N3 H30 views"],"source_refs":["E039","E044","E045","E105"]}
CONTRACTS["CB_OBS14_R2"]={"scope":"prospective independent command evaluation candidate admission","support":{"required_clusters":270,"registered_candidate_records":0,"strict_independent_clusters":0,"gap":270},"role":"metadata only; no payload, target, or score access","status":"not_ready","caveats":["no payload or scores read","614 final groups aggregate only; IDs not enumerated"],"source_refs":["E066-E072"]}
CONTRACTS["CB_V2_TASK_MACRO"]={"scope":"five task macro and task119 waypoint","support":{"dev_episode_groups":[12,12,11,5,7],"dev_total_episodes":47,"h30_anchors":185,"seeds":3,"task_141_groups":5,"task_230_groups":7},"aggregation":"within-episode window mean; episode equal; five task macro; mean of 3 seeds","metric":["position_mae","safe_fd_l1","seam_safe_fd_l1"],"status":"comparable_with_caveats_within_V2; different_contract_across_models","caveats":["task macro is not global episode-equal","session/trial independence unverified","corrected split excludes salted prior outputs from selection"],"source_refs":["E073-E082"]}
CONTRACTS["CB_VIDEOMAE_COREB"]={"scope":"VideoMAE fusion vs V-JEPA fusion and each state-control","support":{"dev_groups":603,"dev_rows":2322,"seeds":3,"raw_regions":["action[0:14]","action[14:28]","action[0:28]"]},"common":{"target_normalizer_hashes":"match","IDs":"match","optimizer_updates":11120,"batch":64,"objective":"episode-equal per-sample"},"differences":["video resolution/pooling/backbone dimensions differ","V-JEPA cache bridge differs on 64/20079 train rows in bounded reference comparison","STATE-CTRL model outputs differ substantially although shared nonvideo init matches"],"metric":["standardized_mae","safe_fd_normalized"],"replay":{"coreA":"3/3 seeds x train/validation exact logits replay","coreB":"0/12 regenerated at original checkpoint gate; saved prediction scoring accepted","portable_integration":"needs_fix"},"status":"comparable_with_caveats","source_refs":["E083-E087"]}
CONTRACTS["CB_COMMAND28_VS_OBS43"]={"scope":"two distinct target definitions","command28":"future upper-body action command trajectory (28D)","OBS43":"recorded observation transition (43D)","status":"different_contract_no_ranking","caveats":["different normalizers and units","future-command oracle uses privileged unavailable-at-inference data"],"source_refs":["E047-E052"]}

def write_claims():
    write_csv("CLAIM_LEDGER.csv",CLAIM_FIELDS,[dict(zip(CLAIM_FIELDS,r)) for r in CLAIMS])
def write_contracts():
    write_json("COMPARISON_CONTRACTS.json",{"schema_version":1,"as_of_utc":utc_now(),"status_enum":["matched_accepted","comparable_with_caveats","different_contract","unverified"],"contracts":CONTRACTS})

NEXT_MD="""# 下一步决策：冻结新增模型拟合，等待当前 V3 封存后再分流

**推荐方向：P0_HOLD_NEW_FITS_AND_REVIEW_CURRENT_V3_SEAL。** 本审计观察到 2026-10-14 V3 frozen shared/task residual root 已有 preseal prediction acceptance，但仍没有 final acceptance。状态文件记录 RGB probe 阶段已开始；本审计没有检查进程/GPU、没有等待它结束，也没有读取其预测/target 数组。对同一问题再开一个 residual/gating/query/seed sweep 会重复现有工作。

## 具体瓶颈与最小可证伪问题

C05/C06/C09/C14 支持“level position loss 与变化/FD fidelity 仍存在取舍”；N4 的 S02 改善同时出现在 level 与 time-centered L2，但只有一个 seed 明显，N5 crossed/seed-only intervals 跨零；HDI 的 increment 降低 Hand/Arm/Full level error，却使 Hand future-internal FD 均值恶化约 4.71%，预注册 gate 失败。竞争解释包括 task/recipe/capacity 混杂、样本意图/command availability 不足、normalizer/clock/field mapping 不完整，以及固定 dev 的重复选择。

- **最小问题（只在现有 V3 完成并被封存后评估）：** 其锁定 residual/bias 在原 co-primary 上能否同时降低误差，并且不会降低变化对齐/方向？保留 V3 已锁定的 control、task/seed/LOO 与零 residual；不增加候选、seed、LR或horizon。
- **支持观察：** 正向 co-primary 同时出现在所有锁定 task 与 seed；预注册 residual 与参数/buffer invariance QA 通过；LOO 不由单一 task/seed 决定。
- **推翻观察：** co-primary 取舍、seed/task 方向混合、外层 base/fresh-zero gate 失败、或 source/metric contract 不匹配。仅 level/TV 改善不足以晋级。
- **停止条件：** V3 final acceptance 未封存、源 hash 不稳定、当前模型数值需要读 raw arrays 才能救援、或任一 co-primary/immutability gate 失败时停止，不追加拟合。

## 备选方向

1. **P1_OBS14_REGISTERED_POOL_INTAKE（信息价值高，实际模型收益未知，数据/人工成本未知）。** 目前严格准入是 0/270。下一项可做的是候选 registry metadata intake：producer/revision/许可，43D→packed-history71/action28 字段和单位，稳定 source/session/trial/episode IDs 与 alias 图，clock/video/anchor资格，以及与 CoreA/N2/N3/N4/HDI/OBS43 的历史曝光交叉表和 reservation 状态。未满足 270 个合规 cluster 前不预测/评分；不要将旧 dev 重切为独立池。
2. **P2_TARGET_CLOCK_AND_SCHEMA_REVIEW（信息价值中等，计算成本低，人工/权限成本未知）。** 取得源 schema/producer clock/command availability 的逐字段证据，保留 action[0:14]、[14:28]、[0:28] 稳定数值引用。若物理单位、通道映射或观测时可用性仍缺失，就不写 physical/Cartesian/action-success 主张。

## 既有资产、成本与分支

可复用 N5 accepted tables、N2/N3/N4 sealed references、V2 corrected root、HDI six-cell prediction acceptance；本审计没有复制模型/数组。新拟合/forward/score/bootstrap 均为 0。当前 V3 的新增模型预算在本审计没有取到一个可用的终态实测值；任何额外 fit、GPU-hour、数据购买和人工 schema work 的增量成本记 estimate_unknown。

若 V3 已完成，先只读核验最终 acceptance/源 hash 并使用其真实结果，不再默认重跑 V3。若未完成或封存失败，不续跑旧 owner，也不新建同合同任务。若最后瓶颈是独立数据，走 P1 并保持270门槛；这个门槛只属于 OBS14 prospective confirmation，不推广到开发任务。若数据显示幅度低且 alignment 弱，不单纯增加 TV/FD loss 权重；若 history 很强时 visual gain 没有建立，继续区分输入冗余、意图/command 缺失、融合优化和 target/clock 误差，而不宣称视觉无用。

recommendation_confidence=medium-high（关于“现在不应新增拟合”）；decision_flip_condition=当前 V3 出现完整、稳定且两 co-primary 无取舍的 sealed acceptance，或出现可审计独立数据 metadata。NEXT_DIRECTION_EXECUTED=false。
"""
write_text("NEXT_DIRECTION_DECISION.md",NEXT_MD)

SOURCE_CHANGES=recheck_sources(SRC_ROWS)
v3_changed=any(c["source_id"] in {"E106","E107","E108","E109","E110","E111"} for c in SOURCE_CHANGES)
if v3_changed:
    v3_status="unstable_live_source"
    for root_row in ROOTS:
        if root_row["root_id"]=="ROOT_V3_CURRENT":
            root_row["observed_status"]="unstable_live_source"
            root_row["acceptance_or_seal_time"]="selected phase status changed during audit; no numeric result promoted"

# Produce latest bounded V3 status text after the source snapshot.
v3_cn=("已记录 RGB probe 阶段开始，且 preseal prediction receipt 存在；未发现 final acceptance。" if not V3_FINAL_AT_SNAPSHOT else f"发现 final acceptance 状态 {v3_final_status}；其源文件仍需单独稳定性/manifest 核验。")
if v3_changed:
    v3_cn="选定 V3 状态文件在本审计期间变化，记为 unstable_live_source；不将其 preseal 数值用于确定结论。"

RETURN_MD=f"""# 回传：EE6008 全部实验结论核验

AUDIT_STATUS=complete_with_limits
观察时间：{utc_now()} UTC
审计根：{AUDIT}
覆盖：{len(INV)}/{len(INV)} 个逻辑实验/方法家族；C01–C16 均已裁决。所有性能结论都是 development/validation evidence；没有独立 final/held-out confirmation。

## 当前可靠结论

1. **Core A 最可靠的匹配证据是 Batch48 dynamic 对 repeated-first-frame static 的分类比较。** 三个固定 seed、每个 617 条 dev/validation episode：Top-1 是 95.624% 对 93.733%，gain +1.891pp，3/3 seed 为正；accepted hierarchical CI 为 [+0.756,+3.133]pp，Holm p=0.00240。macro-F1 是 86.990% 对 86.756%（+0.234pp，2/3 seed 为正），hierarchical CI [−2.577,+2.736]pp 跨零。这是有限的任务识别动态信号，不是动作 anticipation。[E001–E005]
2. **Core A Full-120 LoRA 是另一份单 seed 开发矩阵。** 最高绝对 video Top-1/macro-F1 来自 static pooled LoRA：97.083%/92.992%。Pooled dynamic LoRA 对 frozen 的 Top-1/macro-F1 gain 为 +1.297/+1.988pp，区间跨零；static LoRA 的 Top-1 gain 更大（+2.107pp）。不能把它拼成 dynamic-specific 收益。[E006–E010]
3. **Core B 的 Persistence 基线不能略过。** B1 七格验收通过，但 B-STATE Arm14 standardized MAE .165614，高于 Persistence .126969。B2 六格均验收；0.125s 两格是复用 dense proxy，0.25/0.5s 才是四格 true-pixel。B3 后续 sealed analysis 为 8/8，但只有单 seed、复用两格 B2 dense control，候选 gate 未通过。[E011–E016]
4. **D3 objective/scheduler 修正修复了 implementation contract，不是已证实的科学根因。** 老 as-run trainer 的实际 loss 是 batch loss reduction 后乘 batch weight mean，并每 epoch 推 scheduler；修正版按样本做 episode-equal weighted mean，分母为 actual batch size，optimizer/scheduler 均为 11,120 calls。H30 的 old−corrected MAE gain：TS11-D −.003134，STATE-CTRL +.001662；video-specific interaction −.004796。没有普遍收益，且 TS11-D/H30 的 seed02 在 train/dev 都近似静止。[E020,E023–E029]
5. **N2/N5 query 均值受 seed 选择影响。** G_TS 三 seed mean：position standardized MAE +.001121、safe-FD +.005342。fixed-seed group-only CI 为正；exact-27 seed-only 与 crossed seed×group CI 跨零。Position 三 seed gain 为 −.008257/+ .014806/−.003185，1/3 seed 为正；移除 seed02 后两个 LOO mean 都转负。Q0/Q1 init hashes 配对通过，sample-order hash 未记录。[E040–E043,E056–E065]
6. **N4 修正了 seed02 的旧解释。** Q0→Q1 position MAE 从 .209936 到 .195131，safe-FD 从 .976491 到 .956604；level L2 与 time-centered L2 都下降，后者略大。TV/alignment 仍有限（r=.257、c=.266、FD-MSE skill=.071）；future-GT oracle 是诊断输入，不可部署。[E053–E055]
7. **未来 command 与 observation-state 是两种目标。** Command28 H30 Ridge 的 position MAE/safe-FD 为 .230365/1.044448，差于 Persistence .223135/.976491；OBS43 HOLD 和 future-command ORACLE 也都差于 OBS persistence。Oracle 相对 HOLD 的 gain 只说明该固定模型能用输入中的未来 command，不能部署或替代 command28 排名。[E047–E052]
8. **没有独立复验池。** OBS14 bounded metadata scope 为 0 registered candidates、0/270 strict acquisition clusters；91 个 CoreA non-final groups 的 CoreB exposure unknown，final 614 仅看汇总、未枚举 IDs。[E056–E072]
9. **Newest V3 还没封存。** {v3_cn} 不从 preseal 状态推断效果，也不据此扩 residual/query/seed 矩阵。[E106–E111]

## 关键对照（不超过 12 行；完整版含每 seed 和正确 LOO）

| 对照 | control → candidate；canonical gain | seed/support | uncertainty 与证据边界 |
|---|---|---|---|
| CoreA B48 Top-1 | static 93.733% → dynamic 95.624%；+1.891pp | 3/3 正；617 episodes | hierarchical [+0.756,+3.133]pp；dev/validation |
| CoreA B48 macro-F1 | 86.756% → 86.990%；+0.234pp | 2/3 正；617 episodes | hierarchical [−2.577,+2.736]pp；跨零 |
| CoreB B1 Arm14 MAE | Persistence .126969 → B-STATE .165614；−.038645 | 1 accepted seed；614 rows | matched B1 diagnostic；Persistence 更好 |
| D3 TS11-D H30 scheduler | old .19627 → corrected .19941；−.003134 | 3 fixed seeds；603 groups | group CI [−.003831,−.002435] |
| D3 video-specific H30 interaction | STATE-CTRL scheduler gain → TS11-D gain；−.004796 | 3 seeds；603 groups | CI [−.005671,−.003923]；偏向 STATE-CTRL |
| N2 G_TS position | Q0→Q1；+.001121 standardized MAE | 1/3 seed 正；603 groups | group-only CI 正；seed-only/crossed 跨零 |
| N2 G_TS safe-FD | Q0→Q1；+.005342 normalized error | 1/3 seed 正；603 groups | group-only CI 正；seed-only/crossed 跨零 |
| N3 video gain vs N-VIDEO | FULL +.000631 / OBS −.003568 / ANCHOR +.004195 | 3 seeds；603 groups | non-monotonic；ANCHOR 不是 strict video-only |
| N4 seed02 Q0→Q1 | position +.014806；safe-FD +.019888 | 1 of 3 seeds | level 与 time-centered L2 均改善；seed-sensitive |
| VideoMAE H30 Full28 | VM−V-JEPA error delta −.004040/−.006418; own G_VM −.000522/−.000619 | 2/3 backbone seed signs; 603 groups | saved predictions scored; CoreB replay 0/12 regenerated |
| V2 five-task D_S | position −.003717；safe-FD −.009081 | 3 seeds × 5 tasks | task-LOO 是 task_id；recipes 不是纯 specialization contrast |
| HDI INCREMENT vs DIRECT Hand | level MAE gain +.002282；future-FD gain −.000183 | 3/3 seeds worsen FD | crossed FD gain CI [−.0002192,−.0001472]；candidate gate fails |

## 已修正结论、未决事项与下一步

- 不再引用“scheduler 修复了 under-dynamic”“query mean 为正就有普遍收益”“历史越少视觉越有用”“Hadamard rank 已识别根因”“HDI Increment 已成为候选”或“VideoMAE CoreB replay/portable integration 完成”。
- N5 明确区分 group-only fixed-seed、exact 27 seed-only 和 crossed seed×group uncertainty；跨零不等于零效应。V2 文件 ALL_SEED_LOO.csv 的删除单位是 task_id，本审计分别生成了 SEED_LOO.csv 与 TASK_LOO.csv。
- **首选：暂停新增拟合，只对现有 V3 的终态封存包做一次只读验收。** 只有它完成并稳定后，才据两 co-primary 是否有 trade-off 决定下一项。**备选：**按 OBS14 原合同准备 metadata intake（0/270）；或先核清 target/clock/field/command availability。三者均未在本审计执行。
- 行为计数：新 fit/optimizer/forward/encoder/scoring/bootstrap/raw-array/new-pool/final/heldout/old-root writes 全为 0；读取了已接受的 GT-derived development summary tables，故不是 blind analysis。
- 产物：ROOT_REGISTRY.csv、EXPERIMENT_INVENTORY.csv、COMPARISON_CONTRACTS.json、CLAIM_LEDGER.csv、KEY_COMPARISONS.csv、PER_SEED_EFFECTS.csv、SEED_LOO.csv、TASK_LOO.csv、CONTRADICTIONS_AND_GAPS.md、EXPOSURE_AND_DATA_READINESS.json、NEXT_DIRECTION_DECISION.md、qa/AUDIT_ACCEPTANCE.json、RESOURCE_LOG.csv。
- 最终 checksum receipt：qa/FINAL_CHECKSUM_VERIFICATION.json。审计范围仅为列出的 selected source files；未宣称全项目 checkpoint/cache/raw 数据已重验。
"""
write_text("RETURN_TO_CHATGPT.zh-CN.md",RETURN_MD)

# Machine-readable return with explicit null/UNKNOWN and evidence refs carrying file/hash/locator.
def evidence_obj(sid):
    r=SRC.get(sid,{})
    return {"file":r.get("project_relative_path"),"sha256":r.get("sha256"),"locator":r.get("locator"),"acceptance_scope":r.get("role")}
def refs(ids):
    out=[]
    for sid in ids:
        if sid in SRC:out.append(evidence_obj(sid))
    return out
claim_counts={}
for row in CLAIMS:claim_counts[row[2]]=claim_counts.get(row[2],0)+1
unique_comps=sorted(set(r["comparison_id"] for r in key_rows))
unresolved_comps=sorted(set(r["comparison_id"] for r in key_rows if r.get("comparison_status") in ["unverified","unresolved"]))
source_missing=[r["source_id"] for r in SRC_ROWS if not r.get("exists")]
source_changed_count=len(SOURCE_CHANGES)

RETURN_JSON={
 "audit_version":"EE6008_CONCLUSION_AUDIT_V1",
 "status":"complete_with_limits",
 "audit_root":str(AUDIT),
 "observed_at_utc":utc_now(),
 "coverage":{"families_required":len(INV),"families_accounted_for":len(INV),"not_located":["project-root-relative Batch48 harvest anchor; canonical repo-bound harvest exists"]},
 "key_conclusions":[
  {"id":"K01","claim":"CoreA Batch48 dynamic Top-1 is modestly positive; macro-F1 is uncertain","evidence_refs":refs(["E001","E002","E003","E004","E005"])},
  {"id":"K02","claim":"CoreB persistence remains a strong negative diagnostic; B2 proxy and true-pixel cells stay separate","evidence_refs":refs(["E011","E014","E015","E016"])},
  {"id":"K03","claim":"D3 objective/scheduler implementation repairs pass; scientific benefit is not general","evidence_refs":refs(["E023","E024","E026","E027","E028","E029"])},
  {"id":"K04","claim":"N2/N5 query average gains are seed-sensitive and dev-only","evidence_refs":refs(["E040","E058","E059","E063","E064"])},
  {"id":"K05","claim":"OBS14 metadata has 0/270 strict independent clusters; no score ran","evidence_refs":refs(["E066","E068","E069","E072"])},
  {"id":"K06","claim":"Current V3 root was unsealed at the status snapshot","evidence_refs":refs(["E106","E110","E111"])}
 ],
 "corrected_or_withdrawn_claims":[
  "Treat old D3 as-run objective as distinct from corrected per-sample episode-equal implementation.",
  "Do not label V2 ALL_SEED_LOO.csv as seed LOO; its omitted unit is task_id.",
  "Do not call N3 ANCHOR strict video-only.",
  "Do not call VideoMAE CoreB replay or portable integration complete.",
  "Do not describe standardized/native errors as physical accuracy."
 ],
 "unknowns_that_change_decision":[
  "Current V3 terminal sealed results and metric trade-offs.",
  "OBS14 reservation/alias status and any valid independent acquisition-cluster source.",
  "Training sample-order pairing for N2 Q0/Q1; init hashes alone give nominal pairing.",
  "Physical units, URDF/FK mapping, and recorded command availability where source records omit them."
 ],
 "study_status":{
  "core_a":{"status":"development_validation_complete_with_limits","evidence_refs":refs(["E001","E006","E009"])},
  "n2":{"status":"accepted_safe_exports_and_24_backend_views; query line closed","evidence_refs":refs(["E040","E041","E043"])},
  "n3":{"status":"18 H30 views directly reused; no rerun/rescore","evidence_refs":refs(["E044","E045"])},
  "n4":{"status":"24_of_24_analysis_bundles_accepted","evidence_refs":refs(["E053","E054"])},
  "n5":{"status":"complete_statistics; independent readiness NOT_READY","evidence_refs":refs(["E056","E064","E066"])},
  "command_query_line":{"status":"closed_no_new_query_training","evidence_refs":refs(["E040","E088","E089"])},
  "videomae_core_a_replay":{"status":"3_seed_train_validation_replay_accepted_with_caveats","evidence_refs":refs(["E083"])},
  "videomae_core_b_replay":{"status":"0_of_12_regenerated_at_checkpoint_replay_gate","evidence_refs":refs(["E083","E087"])},
  "videomae_portable_integration":{"status":"needs_fix","evidence_refs":refs(["E083","E087"])},
  "obs43":{"status":"error_budget_complete_independent_confirmation_not_executed","evidence_refs":refs(["E051","E052"])},
  "obs14":{"status":"data_blocked_0_of_270","evidence_refs":refs(["E068","E069","E072"])},
  "v2":{"status":"corrected_five_task_dev_complete_not_independent","evidence_refs":refs(["E073","E074","E077","E078"])},
  "v3":{"status":v3_status,"final_acceptance_present_at_snapshot":V3_FINAL_AT_SNAPSHOT,"preseal_status":V3_PRESEAL_STATUS,"evidence_refs":refs(["E106","E110","E111"])}
 },
 "direction_candidates":[
  {"id":"P0_HOLD_NEW_FITS_AND_REVIEW_CURRENT_V3_SEAL","priority":"preferred","reason":"An Oct14 V3 working root exists and has preseal output; another residual/query sweep would overlap it. Review its terminal accepted package once available.","minimal_question":"Does locked residual/bias improve position without losing FD/alignment across registered tasks and seeds?","stop_condition":"No new fitting if final acceptance is absent, unstable, or any co-primary/immutability gate fails.","incremental_cost":"estimate_unknown; this audit executes none."},
  {"id":"P1_OBS14_REGISTERED_POOL_INTAKE","priority":"alternative","reason":"Independent clusters remain the primary confirmation blocker: 0 verified/270 required.","minimal_input":"Producer/revision/license, field/unit mapping, session/trial/episode IDs, alias graph, clock/anchor rules, exposure crosswalk and reservation.","stop_condition":"No forecast/score until 270 strict acquisition clusters and role gates are satisfied.","incremental_cost":"data and human review estimate_unknown; no purchase/download authorized."},
  {"id":"P2_TARGET_CLOCK_AND_SCHEMA_REVIEW","priority":"alternative","reason":"Field/clock/availability uncertainty can change whether a future-command target is valid.","minimal_question":"Are observation, anchor command, and future command fields ordered, timed, and available as assumed?","stop_condition":"If source schema/availability remains unbound, keep physical or deployable claims unresolved.","incremental_cost":"human/source-owner review estimate_unknown; no external contact performed."}
 ],
 "preferred_direction_id":"P0_HOLD_NEW_FITS_AND_REVIEW_CURRENT_V3_SEAL",
 "recommendation_confidence":"medium-high for holding new model fits now; low for eventual model-choice direction",
 "decision_flip_condition":"A hash-stable sealed V3 result without co-primary trade-off, or accepted metadata for an independent pool.",
 "independent_pool_metadata":{"dataset_candidates":0,"strict_registered_clusters":0,"clusters_required":270,"reservation_status":"UNKNOWN","payload_read":0},
 "actual_this_audit":{"new_training_fits":0,"new_optimizer_steps":0,"model_forward":0,"new_encoder_extraction":0,"new_scoring":0,"new_raw_array_scoring":0,"derived_arithmetic_rows":len(derived_ids),"new_bootstrap":0,"raw_prediction_array_read":0,"raw_target_array_read":0,"gt_derived_statistics_read":True,"new_pool_numeric_access":0,"final_test_access":0,"final_heldout_access":0,"final_heldout_payload_access":0,"old_root_writes":0,"gt_derived_sources":["E002-E009","E016-E022","E027","E030-E086"]},
 "source_verification_scope":{"selected_sources":len(SPECS),"selected_sources_present":len(SRC_ROWS)-len(source_missing),"selected_source_hashes_rechecked":len(SRC_ROWS)-len(source_missing),"source_hash_changes":source_changed_count,"source_missing":source_missing,"current_selected_source_hash_verified":True,"historical_full_manifest_verification_reported":"per-source acceptance/manifest receipts retained; no full-root rehash"},
 "historical_access_unknowns":[x["study"]+": "+x["known"] for x in HISTORICAL_ACCESS if "UNKNOWN" in x["known"].upper() or "unknown" in x["known"].lower()],
 "checksum_verification_receipt":"qa/FINAL_CHECKSUM_VERIFICATION.json"
}
write_json("RETURN_TO_CHATGPT.json",RETURN_JSON)

ROOT_FIELDS=["root_id","study_id","root_kind","observed_status","exists","root_path","owner_state_reported","acceptance_or_seal_time","alias_or_parent","checked_scope","source_pointer","mtime_utc","metadata_bytes"]
INV_FIELDS=["family_id","series","planned_fit_count","actual_new_training_fits","accepted_prediction_or_analysis_bundles","acceptance_status","independence_status","inventory_status","evidence_ids","note"]
# Finalize all tabular deliverables before the checksum seal.
KEY_FIELDS.extend(["accepted_p_value","holm_adjusted_p"])
write_csv("ROOT_REGISTRY.csv",ROOT_FIELDS,ROOTS)
write_csv("EXPERIMENT_INVENTORY.csv",INV_FIELDS,INV)
write_csv("CLAIM_LEDGER.csv",CLAIM_FIELDS,[dict(zip(CLAIM_FIELDS,r)) for r in CLAIMS])
write_json("COMPARISON_CONTRACTS.json",{"schema_version":1,"as_of_utc":utc_now(),"status_enum":["matched_accepted","comparable_with_caveats","different_contract","unverified"],"contracts":CONTRACTS})
write_csv("KEY_COMPARISONS.csv",KEY_FIELDS,key_rows)
write_csv("PER_SEED_EFFECTS.csv",PER_FIELDS,per_rows)
write_csv("SEED_LOO.csv",SEED_LOO_FIELDS,seed_loo_rows)
write_csv("TASK_LOO.csv",TASK_LOO_FIELDS,task_loo_rows)

# Include exact protected/unknown-role path surfaces without enumerating or opening payloads.
EXPOSURE["protected_and_unknown_path_exclusions"]=[
 {"path":"data_raw/HumanoidEveryday/G1","role":"raw dataset; mixed split role","action":"not traversed/read in this audit"},
 {"path":"reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z/results/.. target inputs referenced by OUTCOME_EXPOSURE.json","role":"historical dev targets","action":"receipt metadata only; no arrays opened"},
 {"path":"reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_TARGETS_SCORER_ONLY.npz","role":"dev scorer-only payload","action":"not opened"},
 {"path":"reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/stats_input/ACCEPTED_DERIVED_LOSS_SUMS.npz","role":"accepted derived-loss payload","action":"not opened; accepted CSV/JSON summaries used"},
 {"path":"reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z/draws/*.npz","role":"bootstrap draw arrays","action":"hash/draw metadata only; arrays not opened"},
 {"path":"all checkpoint, prediction, feature-cache, video/frame, parquet and model-weight paths in selected manifests","role":"payload or unknown role","action":"filename/acceptance metadata only; payloads not opened"},
 {"path":"unregistered future intake outside accepted registry surfaces","role":"unknown role","action":"not searched; no path or candidate inferred"}
]
EXPOSURE["protected_final_test_ids"]={"enumerated":False,"payload_read":0,"aggregate_count":614,"count_scope":"OBS14 legacy aggregate only; not a project-wide final count"}
EXPOSURE["unknown_role_handling"]="Any unresolved payload role was excluded. No CSV/JSON name was treated as safe until its report/registry role was identified."

# Refresh and write source scope after all selected reads; no values are read from arrays.
source_changes_final=recheck_sources(SRC_ROWS)
if source_changes_final:
    SOURCE_CHANGES.extend([c for c in source_changes_final if c not in SOURCE_CHANGES])
    if any(c["source_id"] in {"E106","E107","E108","E109","E110","E111"} for c in source_changes_final):
        v3_changed=True
        v3_status="unstable_live_source"
        for r in ROOTS:
            if r["root_id"]=="ROOT_V3_CURRENT":
                r["observed_status"]="unstable_live_source"
                r["acceptance_or_seal_time"]="selected phase status changed during audit; no numeric result promoted"
write_csv("SOURCE_SCOPE.csv",["source_id","project_relative_path","role","locator","exists","size_bytes","mtime_utc","sha256","manifest_path","manifest_expected_sha256","manifest_match","manifest_lookup","live_source","stability","recheck_sha256","recheck_mtime_utc"],SRC_ROWS)
write_json("qa/SOURCE_HASH_RECHECK.json",{"checked_utc":utc_now(),"selected_source_count":len(SRC_ROWS),"selected_sources_present":sum(bool(r.get("exists")) for r in SRC_ROWS),"source_hash_changes":SOURCE_CHANGES,"source_scope":"selected files only; no checkpoint/cache/raw data full-root rehash","historical_full_manifest_verification":"reported acceptance receipts are referenced, not repeated"})
write_csv("ROOT_REGISTRY.csv",ROOT_FIELDS,ROOTS)
EXPOSURE["observed_at_utc"]=utc_now()
EXPOSURE["this_audit_v3_snapshot_status"]=v3_status
EXPOSURE["actual_this_audit"]["DERIVED_ARITHMETIC_ROWS"]=len(derived_ids)
write_json("EXPOSURE_AND_DATA_READINESS.json",EXPOSURE)

# Recompute machine-readable return status if the current V3 status changed during this audit.
if v3_changed:
    for item in RETURN_JSON["study_status"].values():
        pass
    RETURN_JSON["study_status"]["v3"]["status"]="unstable_live_source"
    RETURN_JSON["unknowns_that_change_decision"].append("Current V3 phase files changed during the audit window; its result remains unstable_live_source.")
    RETURN_JSON["source_verification_scope"]["source_hash_changes"]=len(SOURCE_CHANGES)
    RETURN_JSON["actual_this_audit"]["derived_arithmetic_rows"]=len(derived_ids)
    RETURN_JSON["observed_at_utc"]=utc_now()
    RETURN_MD=RETURN_MD.replace("已记录 RGB probe 阶段开始，且 preseal prediction receipt 存在；未发现 final acceptance。","V3 状态文件在审计期间变化，记为 unstable_live_source；未将其数值结果纳入结论。")
    write_text("RETURN_TO_CHATGPT.zh-CN.md",RETURN_MD)
write_json("RETURN_TO_CHATGPT.json",RETURN_JSON)
write_text("NEXT_DIRECTION_DECISION.md",NEXT_MD)
CONFLICTS="""# Contradictions and gaps

## Resolved from the checked source records

1. The supplied CoreA Batch48 project-relative anchor is absent at the project-level reports path. The canonical schema-fixed harvest is present under the populated repo-bound reports tree. The project policy text still describes that repo as an empty placeholder; the current filesystem conflicts with that text, and git is unavailable here, so commit/branch identity is UNKNOWN.
2. The early B3 controller record is a stale running snapshot. Later B3 reacceptance and the sealed unified analysis report complete 8/8, with the candidate gate not passed. Do not treat the early controller snapshot as final.
3. The historical D3 trainer did not implement sample-wise episode-equal weighting: it reduced a batch loss then multiplied by the batch mean weight, and stepped the scheduler once per epoch. The corrected D3 contract applies each row weight before reduction over actual batch size and advances the scheduler per optimizer update. The old and corrected scientific results remain separate.
4. N2 group-only and N4 crossed seed-by-group intervals have different sampling scope. N5 confirmed the distinction and reproduced 30 historical endpoints with maximum absolute difference 2.2898349882893854e-16. This audit did not rerun a bootstrap.
5. V2 ALL_SEED_LOO.csv has omitted_task_id and is a task-LOO table per seed. It is not a seed-LOO table. Separate correctly-unit seed-LOO and task-LOO tables are produced here from accepted summaries.
6. VideoMAE artifact verification, saved-prediction scoring, checkpoint replay and portable integration have different states. CoreB artifacts and saved scores were accepted with caveats; checkpoint replay regenerated 0/12 in the original gate; portable integration needs_fix; a later CPU zero-input replay failed exact-zero tolerance and is not a natural-input full checkpoint replay.
7. OBS43 has one sealed closeout and a separate empty same-named placeholder; only the sealed root informs claims.

## Claims that need narrower wording

- CoreA Batch48 Top-1 is positive and Holm-significant under its matched task-recognition contract, but macro-F1 is +0.234pp and both accepted macro-F1 intervals cross zero. It is not action-anticipation evidence.
- Full120 LoRA's highest absolute video score is static pooled LoRA. A single-seed matched LoRA gain does not establish dynamic-specific benefit.
- N2/N5 query mean effects are positive but seed-sensitive; seed-only and crossed intervals cross zero. The seed02 evidence is influential.
- N3 FULL/OBS/ANCHOR gains are nonmonotonic; ANCHOR retains a command/anchor branch and is not strict video-only.
- N4 seed02 improved both level and time-centered error, so “pure offset rescue” is outdated; the dynamic signal is still limited.
- HDI INCREMENT lowers position errors but worsens Hand future-internal FD and fails its gate.
- V2 specialization gains are task/metric dependent; training recipes/capacities differ, and session independence is unverified.
- Current V3 has only unsealed/preseal status at the audit snapshot; no result claim is made.

## Still UNKNOWN

- Project-wide historical target-access telemetry beyond the cited per-study receipts.
- Session/acquisition-cluster independence for reused development groups.
- Physical units, URDF joint identities, FK/Cartesian mapping, robot success and true command availability where not source-bound.
- OBS14 reservation/alias status and candidate data outside the exact admitted registries.
- Current live V3 process/owner state; no process or GPU query was made.
"""
write_text("CONTRADICTIONS_AND_GAPS.md",CONFLICTS)

# Resource measurements for this read-only builder. Prior wrapped source-read CPU/RSS telemetry is not available.
now=datetime.now(timezone.utc)
start_dt=datetime.fromisoformat(START_UTC.replace("Z","+00:00"))
wall_total=(now-start_dt).total_seconds()
builder_wall=time.monotonic()-BUILD_MONO
builder_cpu=time.process_time()-BUILD_CPU
usage=resource.getrusage(resource.RUSAGE_SELF)
peak_rss=int(usage.ru_maxrss*1024)
source_bytes=sum(int(r["size_bytes"]) for r in SRC_ROWS if str(r.get("size_bytes","")).isdigit())
disk_preseal=sum(p.stat().st_size for p in AUDIT.rglob("*") if p.is_file())
resource_rows=[
{"record_id":"AUDIT_TOTAL","scope":"actual_this_audit","phase":"bounded_read_only_source_review_and_report","start_utc":START_UTC,"end_utc":utc_now(),"wall_seconds":wall_total,"cpu_seconds":"NA_not_captured_for_wrapped_read_commands","peak_rss_bytes":"NA_not_captured_for_wrapped_read_commands","disk_bytes":disk_preseal,"gpu_hours":0,"gpu_actions":0,"source_bytes_hashed":source_bytes,"read_scope":"selected accepted report/metadata files listed in SOURCE_SCOPE.csv only","note":"CPU/RSS telemetry for individual wrapped source-read commands was unavailable; historical experiment resources are not added."},
{"record_id":"AUDIT_BUILDER","scope":"actual_this_audit","phase":"standard_library_extract_derived_rows_parser_QA_and_selected_hash_recheck","start_utc":BUILD_START_UTC,"end_utc":utc_now(),"wall_seconds":builder_wall,"cpu_seconds":builder_cpu,"peak_rss_bytes":peak_rss,"disk_bytes":disk_preseal,"gpu_hours":0,"gpu_actions":0,"source_bytes_hashed":source_bytes,"read_scope":"accepted aggregate CSV/JSON/report/code text only","note":"No GPU APIs, raw arrays, model weights or project code imports."},
{"record_id":"HISTORICAL_RESOURCES","scope":"historical_acceptance_records_only","phase":"not_counted_in_this_audit","start_utc":"NA","end_utc":"NA","wall_seconds":"NA","cpu_seconds":"NA","peak_rss_bytes":"NA","disk_bytes":"NA","gpu_hours":"NOT_SUMMED","gpu_actions":"NOT_THIS_AUDIT","source_bytes_hashed":"NA","read_scope":"per-source accepted resource receipts remain in original roots","note":"Prior GPU hours are not included in the audit's zero GPU use."}
]
write_csv("RESOURCE_LOG.csv",["record_id","scope","phase","start_utc","end_utc","wall_seconds","cpu_seconds","peak_rss_bytes","disk_bytes","gpu_hours","gpu_actions","source_bytes_hashed","read_scope","note"],resource_rows)

# Parse and validate only newly generated CSV/JSON artifacts; this is artifact QA, not a project test run.
required=[
"ROOT_REGISTRY.csv","EXPERIMENT_INVENTORY.csv","COMPARISON_CONTRACTS.json","CLAIM_LEDGER.csv","KEY_COMPARISONS.csv","PER_SEED_EFFECTS.csv","SEED_LOO.csv","TASK_LOO.csv","CONTRADICTIONS_AND_GAPS.md","EXPOSURE_AND_DATA_READINESS.json","NEXT_DIRECTION_DECISION.md","RETURN_TO_CHATGPT.zh-CN.md","RETURN_TO_CHATGPT.json","RESOURCE_LOG.csv","SOURCE_SCOPE.csv","qa/SOURCE_HASH_RECHECK.json"
]
missing_required=[p for p in required if not (AUDIT/p).is_file()]
parse_errors=[];parsed_csv={};parsed_json={}
for p in sorted(AUDIT.rglob("*")):
    if not p.is_file() or p.name in {"CHECKSUMS.sha256","FINAL_CHECKSUM_VERIFICATION.json"}:continue
    try:
        rel=p.relative_to(AUDIT).as_posix()
        if p.suffix.lower()==".csv":
            with p.open(newline="",encoding="utf-8-sig") as f:
                rows=list(csv.reader(f))
            if rows and any(len(row)!=len(rows[0]) for row in rows[1:]):raise ValueError("inconsistent CSV row width")
            parsed_csv[rel]=max(0,len(rows)-1)
        elif p.suffix.lower()==".json":
            json.loads(p.read_text(encoding="utf-8"));parsed_json[rel]="PASS"
    except Exception as e:parse_errors.append({"path":p.relative_to(AUDIT).as_posix(),"error":type(e).__name__+":"+str(e)})
status_counts={}
for r in key_rows:status_counts[r["comparison_status"]]=status_counts.get(r["comparison_status"],0)+1
binding_true=sum(r.get("manifest_match")=="true" for r in SRC_ROWS)
binding_false=sum(r.get("manifest_match")=="false" for r in SRC_ROWS)
binding_unverified=sum(r.get("manifest_match") not in ["true","false"] for r in SRC_ROWS)
audit_acceptance={
 "audit_version":"EE6008_CONCLUSION_AUDIT_V1","status":"complete_with_limits","observed_at_utc":utc_now(),"audit_root":str(AUDIT),
 "engineering_source_qa":{"required_files_missing":missing_required,"csv_parser":"PASS" if not parse_errors else "FAIL","json_parser":"PASS" if not parse_errors else "FAIL","csv_rows":parsed_csv,"json_files_parsed":len(parsed_json),"parse_errors":parse_errors,"selected_sources":len(SRC_ROWS),"selected_sources_present":sum(bool(r.get("exists")) for r in SRC_ROWS),"selected_source_hash_changes":SOURCE_CHANGES,"selected_manifest_entries_match":binding_true,"selected_manifest_entries_mismatch":binding_false,"selected_manifest_binding_unavailable_or_unverified":binding_unverified,"historical_full_manifest_verification_reported":False,"git_identity":"UNKNOWN: git executable not available"},
 "scientific_judgment":{"claims_adjudicated":len(CLAIMS),"claim_status_counts":claim_counts,"key_comparison_rows":len(key_rows),"key_comparison_ids":len(unique_comps),"key_comparison_status_counts":status_counts,"final_test_payload_access":0,"development_evidence_upgraded_to_confirmation":False,"cross_contract_rankings_suppressed":True},
 "actual_this_audit":{"NEW_TRAINING_FITS":0,"NEW_OPTIMIZER_STEPS":0,"MODEL_FORWARD":0,"NEW_ENCODER_EXTRACTION":0,"NEW_SCORING":0,"NEW_RAW_ARRAY_SCORING":0,"DERIVED_ARITHMETIC_ROWS":len(derived_ids),"NEW_BOOTSTRAP":0,"RAW_PREDICTION_ARRAY_READ":0,"RAW_TARGET_ARRAY_READ":0,"GT_DERIVED_STATISTICS_READ":True,"NEW_POOL_NUMERIC_ACCESS":0,"FINAL_TEST_ACCESS":0,"FINAL_HELDOUT_ACCESS":0,"FINAL_HELDOUT_PAYLOAD_ACCESS":0,"OLD_ROOT_WRITES":0},
 "known_limits":["selected source manifest binding is reported per file in SOURCE_SCOPE; some sources are self-excluded or have no accepted local manifest entry","no project-wide historical label-access ledger exists in the checked sources","V3 current root is working/unsealed or unstable as recorded; process state intentionally not queried","no git branch/commit because git executable is absent"],
 "checksum_verification_receipt":"qa/FINAL_CHECKSUM_VERIFICATION.json"
}
if missing_required or parse_errors or binding_false:
    audit_acceptance["status"]="complete_with_limits"
write_json("qa/AUDIT_ACCEPTANCE.json",audit_acceptance)
# Parse the acceptance JSON itself before checksum generation.
json.loads((AUDIT/"qa/AUDIT_ACCEPTANCE.json").read_text(encoding="utf-8"))

# Hash the current delivery last; manifest excludes itself and final receipt.
manifest=AUDIT/"CHECKSUMS.sha256"
receipt=AUDIT/"qa/FINAL_CHECKSUM_VERIFICATION.json"
manifest_entries=[]
for p in sorted(AUDIT.rglob("*")):
    if not p.is_file() or p in {manifest,receipt}:continue
    manifest_entries.append((sha(p),p.relative_to(AUDIT).as_posix()))
manifest.write_text("".join(f"{h}  {rel}\n" for h,rel in manifest_entries),encoding="utf-8")
manifest_sha=sha(manifest)
verified=missing=mismatch=0
mismatched=[];missing_paths=[]
for h,rel in manifest_entries:
    p=AUDIT/rel
    if not p.is_file():missing+=1;missing_paths.append(rel)
    elif sha(p)==h:verified+=1
    else:mismatch+=1;mismatched.append(rel)
receipt_obj={"status":"PASS" if missing==0 and mismatch==0 and verified==len(manifest_entries) else "FAIL","manifest_path":"CHECKSUMS.sha256","manifest_sha256":manifest_sha,"entries":len(manifest_entries),"verified":verified,"missing":missing,"mismatch":mismatch,"missing_paths":missing_paths,"mismatched_paths":mismatched,"excluded":[{"path":"CHECKSUMS.sha256","reason":"manifest cannot hash itself"},{"path":"qa/FINAL_CHECKSUM_VERIFICATION.json","reason":"receipt records manifest verification and is self-excluded by protocol"}],"scope":"current audit deliverables only; selected source files are separately hashed in SOURCE_SCOPE.csv; no whole-project or full checkpoint/cache rehash"}
receipt.write_text(json.dumps(receipt_obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
# Final read-back verifies the receipt itself is valid JSON and the manifest bytes are unchanged.
json.loads(receipt.read_text(encoding="utf-8"))
if sha(manifest)!=manifest_sha:raise RuntimeError("checksum manifest changed after receipt")
print(json.dumps({"audit_root":str(AUDIT),"inventory_families":len(INV),"claims":len(CLAIMS),"key_comparison_rows":len(key_rows),"key_comparison_ids":len(unique_comps),"derived_arithmetic_rows":len(derived_ids),"selected_sources":len(SRC_ROWS),"selected_sources_present":sum(bool(r.get("exists")) for r in SRC_ROWS),"manifest_matches":binding_true,"manifest_mismatches":binding_false,"checksum_entries":len(manifest_entries),"checksum_verified":verified,"checksum_status":receipt_obj["status"],"manifest_sha256":manifest_sha,"builder_cpu_seconds":builder_cpu,"builder_wall_seconds":builder_wall,"builder_peak_rss_bytes":peak_rss,"disk_preseal_bytes":disk_preseal,"v3_status":v3_status},ensure_ascii=False))
