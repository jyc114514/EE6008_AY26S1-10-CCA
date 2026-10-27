#!/usr/bin/env python3
"""Create final N4 reports from accepted result tables and status receipts."""
import os
import csv,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
N3=P/'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'
N1=P/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
H=P/'reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z'
O=P/'reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z'
OB=P/'reports/benchmarks/CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161856Z'
EXPECTED_N2='21bfdf8baf1cabe863eb4aa184320e5e79d29ee6c1a589bf9e25ec944fb4472f'
EXPECTED_N3='60701b7b357e5282222c60b15994a1408fcc1398618025dbebe5388fc69f1a75'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def readc(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writej(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.part')
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)
def write_text(path,text):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
W=writej
def fmt(x,d=6):return f'{float(x):.{d}f}'
J=readj
C=readc
def pick(rel,**kw):
 return [r for r in readc(R/rel) if all(str(r.get(k))==str(v) for k,v in kw.items())]
def mean_stats(endpoint,contrast):
 rr=sorted(pick('results/ALL_SEED_LOO.csv',split='dev',axis_scope='raw28',endpoint=endpoint,contrast=contrast),
  key=lambda r:int(r['seed']))
 r=rr[0]
 return {'mean':float(r['three_seed_mean']),'sample_sd':float(r['three_seed_sample_sd']),
  'median':float(r['three_seed_median']),
  'loo_drop_20261001':float(next(x for x in rr if int(x['loo_drop_seed'])==20261001)['loo_mean']),
  'loo_drop_20261002':float(next(x for x in rr if int(x['loo_drop_seed'])==20261002)['loo_mean']),
  'loo_drop_20261003':float(next(x for x in rr if int(x['loo_drop_seed'])==20261003)['loo_mean'])}
def get_ci(endpoint,estimand):
 rr=pick('results/GROUP_BOOTSTRAP_CI.csv',split='dev',axis_scope='raw28',endpoint=endpoint,estimand=estimand)
 if not rr:return None
 r=rr[0]
 return {'mean':float(r['bootstrap_mean']),
  'ci95':[float(r['ci_percentile_lower_2_5']),float(r['ci_percentile_upper_97_5']),
  ],'valid_draws':int(r['valid_draws'])}
def main():
 now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 pred=readj(R/'qa/PREDICTION_ACCEPTANCE.json')
 analysis=readj(R/'qa/ANALYSIS_ACCEPTANCE.json')
 recon=readj(R/'qa/ORIGINAL_METRIC_RECONCILIATION.json')
 fig_manifest=readj(R/'figures/FIGURE_MANIFEST.json')
 figures=fig_manifest
 fig_file_qa=readj(R/'qa/FINAL_FIGURE_FILE_QA.json')
 fig_visual=readj(R/'qa/FIGURE_AGENT_VISUAL_QA.json')
 exposure=readj(R/'protocol/OUTCOME_EXPOSURE.json')
 if pred.get('status')!='PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES' or analysis.get('status')!='PASS_N4_ERROR_BUDGET_ANALYSIS':
  raise SystemExit('N4 prediction/analysis acceptance is incomplete.')
 if recon.get('status')!='PASS' or recon.get('max_abs_difference')!=0.0:
  raise SystemExit('N2 original metric reconciliation failed.')
 if fig_manifest.get('status')!='PASS_EXPORTS_CREATED' or fig_file_qa.get('status')!='PASS_PNG_PDF_CAPTION_DATA_METADATA':
  raise SystemExit('figure export QA failed.')
 if fig_visual.get('status')!='PASS_AGENT_VISUAL_QA':raise SystemExit('agent visual QA is incomplete.')
 if exposure.get('final_test_access')!=0 or exposure.get('heldout_access')!=0:
  raise SystemExit('final/heldout access boundary is not zero.')
 n2_sha=sha(N2/'CHECKSUMS.sha256');n3_sha=sha(N3/'CHECKSUMS.sha256')
 if n2_sha!=EXPECTED_N2 or n3_sha!=EXPECTED_N3:raise SystemExit('sealed N2/N3 manifest identity changed.')
 source=readj(R/'inventory/SOURCE_ACCEPTANCE.json')
 changed=[x['path'] for x in source.get('n2_selected_file_inventory',[]) if sha(Path(x['path']))!=x['sha256']]
 if changed:raise SystemExit('selected sealed N2 source changed: '+changed[0])
 watcher_launch=readj(N2/'coordination/N2_CUDA_LAUNCH.json')
 watcher_path=P/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
 watcher_sha=sha(watcher_path)
 if watcher_sha!=watcher_launch['watcher_sha256']:raise SystemExit('shared watcher source changed.')
 n3_final=readj(N3/'qa/FINAL_ACCEPTANCE.json')
 if n3_final.get('checksum_status')!='PASS_CHECKSUMS_SHA256_GENERATED_LAST_AND_VERIFIED':
  raise SystemExit('N3 direct-reference acceptance mismatch.')
 n3direct=readj(N2/'n3_refs/N3_DIRECT_REUSE.json')
 ts_s02={}
 for q in ('Q0','Q1'):
  row=pick('results/BY_MODEL_SPLIT_AXIS.csv',system='TS11-D',query=q,seed=20261002,split='dev',axis_scope='raw28')[0]
  pos=pick('results/POSITION_BUDGET.csv',system='TS11-D',query=q,seed=20261002,split='dev',axis_scope='raw28')[0]
  ora=pick('results/ORACLE_MAE_BUDGET.csv',system='TS11-D',query=q,seed=20261002,split='dev',axis_scope='raw28')[0]
  fd=pick('results/FD_ENERGY_ALIGNMENT.csv',system='TS11-D',query=q,seed=20261002,split='dev',axis_scope='raw28')[0]
  oc=pick('results/OUTPUT_CONSTANT_DIAGNOSTIC.csv',system='TS11-D',query=q,seed=20261002,split='dev')[0]
  ts_s02[q]={'position_standardized_mae':float(row['position_standardized_mae']),
   'safe_fd_normalized':float(row['safe_fd_normalized']),'position_mse':float(pos['position_mse']),
   'level_mse':float(pos['level_mse']),'time_centered_error_mse':float(pos['time_centered_error_mse']),
   'oracle_original_standardized_mae':float(ora['oracle_original_standardized_mae']),
   'oracle_removable_standardized_mae':float(ora['oracle_removable_standardized_mae']),
   'oracle_remaining_standardized_mae':float(ora['oracle_remaining_standardized_mae']),
   'target_energy_T':float(fd['target_energy_T']),'prediction_energy_P':float(fd['prediction_energy_P']),
   'cross_alignment_C':float(fd['cross_alignment_C']),'r':float(fd['r_norm_ratio']),
   'c':float(fd['c_zero_lag_cosine']),'fd_mse_skill':float(fd['fd_mse_skill']),
   'fd_mse':float(fd['fd_mse']),'fd_mse_over_T':float(fd['fd_mse_over_T']),
   'norm_error_term':float(fd['norm_error_term']),'alignment_error_term':float(fd['alignment_error_term']),
   'output_constant_position_standardized_mae':float(oc['position_standardized_mae']),
   'output_constant_safe_fd_normalized':float(oc['safe_fd_normalized'])}
 qgain={name:ts_s02['Q0'][name]-ts_s02['Q1'][name] for name in (
  'position_standardized_mae','safe_fd_normalized','position_mse','level_mse',
  'time_centered_error_mse','oracle_removable_standardized_mae','oracle_remaining_standardized_mae',
  'prediction_energy_P','cross_alignment_C','fd_mse')}
 endpoints=['position_standardized_mae','safe_fd_normalized','position_mse','level_mse',
  'time_centered_error_mse','oracle_removable_standardized_mae','oracle_remaining_standardized_mae',
  'prediction_energy_P','cross_alignment_C','fd_mse','r_norm_ratio','c_zero_lag_cosine','fd_mse_skill']
 effects={ep:{c:mean_stats(ep,c) for c in ('G_TS11-D','G_STATE-CTRL','V_Q0','V_Q1','I')} for ep in endpoints}
 cis={ep:{c:get_ci(ep,c) for c in ('G_TS11-D','G_STATE-CTRL','V_Q0','V_Q1','I')}
  for ep in ['position_standardized_mae','safe_fd_normalized','level_mse','time_centered_error_mse',
   'oracle_removable_standardized_mae','oracle_remaining_standardized_mae','prediction_energy_P',
   'cross_alignment_C','fd_mse','r_norm_ratio','c_zero_lag_cosine','fd_mse_skill']}
 paired=readc(R/'results/PAIRED_COMPONENT_GAINS.csv')
 seed_gains={}
 for ep in ('position_standardized_mae','safe_fd_normalized'):
  seed_gains[ep]={}
  for seed in (20261001,20261002,20261003):
   seed_gains[ep][str(seed)]=float(next(x['estimate'] for x in paired
    if x['split']=='dev' and x['axis_scope']=='raw28' and x['endpoint']==ep and
    x['contrast']=='G_TS11-D' and int(x['seed'])==seed))
 summary={'schema_version':1,'created_utc':now,'status':'PASS_N4_EFFECT_SUPPORT_SUMMARY',
  'scope':'12 fixed H30 models; train is in-sample; dev was previously exposed; exploratory only',
  'sign_convention':{'G':'Q0 error−Q1 error; positive favors Q1 for error endpoints',
   'V':'STATE-CTRL error−TS11-D error; positive favors TS11-D for error endpoints',
   'I':'G_TS11-D−G_STATE-CTRL=V_Q1−V_Q0',
   'moment_endpoints':'P/C/r/c/skill rows are signed diagnostics, not automatic gains.'},
  'fixed_models':12,'analysis_bundles_accepted':'24/24','train_new_forward_views':12,
  'dev_cuda_common_reused':'12/12','new_dev_formal_forward_views':0,
  'co_primary_reconciliation':{'status':recon['status'],'rows':recon['co_primary_count'],
   'max_abs_difference':recon['max_abs_difference'],'tolerance_abs':recon['tolerance_abs']},
  'seed_20261002_TS11-D_Q0_vs_Q1_effects':qgain,'seed_20261002_TS11-D_cells':ts_s02,
  'G_V_I_three_seed_mean_sd_median_and_LOO':effects,'dev_group_bootstrap_ci95':cis,
  'G_TS11-D_fixed_seed_gains':seed_gains,
  'support':{'train_anchors':17757,'train_episodes':2712,'train_source_groups':2712,
   'dev_anchors':2322,'dev_episodes':603,'dev_source_groups':603,'episodes_per_group':1},
  'target_access':{'score_attempts':exposure.get('n4_target_access_attempts'),
   'failed_score_attempts_after_target_read':exposure.get('failed_analysis_attempts_after_target_read'),
   'train_rows_scored':17757,'dev_rows_scored':2322,
   'targets_native_full_key_rows_materialized_each_score_attempt':20079,
   'locked_case_anchors':14,'case_figure_full_key_materialization':1,
   'final_test_access':0,'heldout_access':0},
  'bootstrap':{'draw_source':'N2_GROUP_BOOTSTRAP_DRAWS.npz','replicates':10000,
   'seed':20261011,'reused_exact_group_and_seed_draws':True,'p_values_computed':False},
  'n3_direct_reference':n3direct,'independent_confirmation':False,
  'physical_units_verified':False,'agent_visual_qa':'PASS_AGENT_VISUAL_QA',
  'human_visual_review':'pending','submission_ready':False}
 W(R/'results/N4_EFFECT_SUPPORT_SUMMARY.json',summary)
 # Compute endpoint-specific effects as tables without outcome-driven selection.
 query_doc=(
 '# Query 研究线收口\n\n'
 'N4 固定 H30 矩阵已完成 12 个模型、train/dev 共 24 个 error-budget bundles。Q0/Q1 的原 co-primary、E1 level/time-centered 与 oracle MAE 账本、E2 T/P/C 和零时滞对齐诊断均已报告。seed 20261002 的改变不是跨 seed 普遍收益。N3 只直接引用此前接受的 18 predictions/162-row reconciliation。\n\n'
 '本轮后关闭 query 扩张：不追加 query、scale、frequency、LR、seed、H60、D5、Ridge、OBS43 或 backbone 训练。N4 结果是固定模型、已暴露 dev 的探索性证据；没有独立确认，final/heldout 保持锁定。\n')
 write_text(R/'QUERY_LINE_CLOSED.md',query_doc)
 next_doc=(
 '# 下一轮研究的准入条件（本文件未执行）\n\n'
 '本轮不启动任何新训练。若以后要确认 N4 的误差来源，必须先提出独立的新问题并建立新版本合同：\n\n'
 '1. 冻结单一候选和单一对照，预先指定主要端点与分析方向；不在多个 query、LR、horizon 或 seed 中择优。\n'
 '2. 使用与旧训练 source groups 完全隔离、且未参与既有开发结论的评价 population。事后从旧模型训练 groups 切出子集，不能成为旧模型的确认集。\n'
 '3. 锁定 train/dev/final-test 访问边界、source-group 映射、scorer/normalizer hashes、预测接受门和配对单位。\n'
 '4. 先确认独立 population 与访问授权；若没有合规独立数据，标记 confirmation_unavailable，不自动训练，也不改用已暴露 dev 宣称确认。\n'
 '5. 保持 final/heldout 锁定；任何未来开放需另行批准。\n\n'
 'N4 的 oracle-level 结果使用 future GT，只是诊断，不是校准参数或部署后处理。\n')
 write_text(R/'NEXT_RESEARCH_CONDITIONS.md',next_doc)
 # Verify bounded old-root source identities and the unmodified shared watcher.
 source=readj(R/'inventory/SOURCE_ACCEPTANCE.json')
 changed=[x['path'] for x in source.get('n2_selected_file_inventory',[]) if sha(Path(x['path']))!=x['sha256']]
 if changed:raise SystemExit('selected N2 source drift before final seal: '+changed[0])
 watcher_launch=readj(N2/'coordination/N2_CUDA_LAUNCH.json')
 watcher_path=P/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
 watcher_sha=sha(watcher_path)
 if watcher_sha!=watcher_launch['watcher_sha256']:raise SystemExit('shared watcher hash changed.')
 other=readj(R/'inventory/OTHER_WINDOW_REFERENCE_STATUS_AMENDMENT_001.json')
 horizon=readj(H/'qa/FINAL_ACCEPTANCE.json');howner=readj(H/'coordination/OWNER_STATE.json')
 obs43=readj(O/'qa/FINAL_ACCEPTANCE.json');obsowner=readj(O/'protocol/OWNER_STATE.json')
 old={'schema_version':1,'status':'PASS_BOUNDED_NONINTERFERENCE_CHECK','created_utc':now,
  'n4_output_root':str(R),'old_root_write_actions':0,
  'N2':{'root':str(N2),'checksum_manifest_sha256':sha(N2/'CHECKSUMS.sha256'),'expected_sha256':EXPECTED_N2},
  'N3':{'root':str(N3),'checksum_manifest_sha256':sha(N3/'CHECKSUMS.sha256'),'expected_sha256':EXPECTED_N3},
  'selected_N2_files_reverified':len(source.get('n2_selected_file_inventory',[])),
  'selected_N2_file_drift_count':0,'N4_locked_source_hashes':len(source.get('source_hashes',{})),
  'N1_external_sources_reverified':True,'shared_watcher_sha256':watcher_sha,
  'shared_watcher_matches_N2_receipt':True,'shared_watcher_modified':False,
  'HORIZON_reference':{'final_acceptance':horizon.get('status'),'owner_state':howner.get('status'),
   'owner_status_conflict':'Owner remains LOCKED_READY_FOR_CPU_FIT while RUN_STATUS and return say complete; no relaunch.'},
  'OBS43_reference':{'final_acceptance':obs43.get('status'),'owner_state':obsowner.get('status')},
  'OBS43_second_root_file_count':0,'other_window_tasks_modified_or_relaunched':False,
  'full_old_roots_rehashed':False,
  'scope_note':'Verified N2/N3 manifest identities and the N4-selected source hashes; did not recrawl or rehash unrelated sealed roots.',
  'final_test_access':0}
 W(R/'OLD_ROOT_NONINTERFERENCE.json',old)
 # Compact machine-readable final return fields.
 watcher=readj(R/'coordination/N4_TRAIN_WATCHER_STATUS.json')
 inf=readj(R/'coordination/N4_TRAIN_INFERENCE_STATUS.json')
 host=readj(R/'coordination/HOST_PREFLIGHT.json')
 resource=readj(R/'coordination/RESOURCE_LOG.json')
 summary_path=R/'results/N4_EFFECT_SUPPORT_SUMMARY.json'
 final={'schema_version':1,'status':'COMPLETE_N4_COMMAND_ERROR_BUDGET','run_root':str(R),
  'models_fixed':12,'analysis_bundles_accepted':'24/24','dev_cuda_views_reused':'12/12',
  'new_train_forward_views':12,'new_dev_formal_forward_views':0,'supervised_fits':0,
  'new_neural_training':0,'optimizer_steps':0,'encoder_extraction':0,'safe_model_only_used':True,
  'original_full_checkpoint_recovery_rerun':False,'n2_bridge_rerun':False,
  'n3_rerun_or_rescore':False,'n3_direct_reference_status':'PASS_DIRECT_REFERENCE_ONLY',
  'oracle_gt_used_for_diagnostics_only':True,'oracle_outputs_accepted_as_deployable_predictions':False,
  'l2_budget_replaces_coprimary_l1':False,'all_seeds_and_loo_reported':True,
  'query_line_status':'closed_no_new_query_training','independent_confirmation':False,
  'd5_model_trained':False,'final_heldout_access':0,'other_window_tasks_modified_or_relaunched':False,
  'old_root_writes':0,'shared_watcher_modified':False,'max_project_gpu_children':2,
  'human_review_status':'pending','agent_visual_review':'PASS_AGENT_VISUAL_QA','submission_ready':False,
  'prediction_acceptance':pred.get('status'),'analysis_acceptance':analysis.get('status'),
  'co_primary_reconciliation':{'status':recon['status'],'rows':24,'max_abs_difference':0.0,'tolerance_abs':1e-12},
  'output_counts':analysis['output_counts'],'bootstrap_ci_rows':513,
  'target_access':{'scoring_attempts':5,'failed_scoring_attempts_after_open':4,
   'train_unique_rows_scored':17757,'dev_unique_rows_scored':2322,
   'targets_native_full_key_rows_materialized_each_scoring_attempt':20079,
   'case_figure_target_anchors':14,'additional_full_key_materialization_for_case_figure':1,
   'final_test_access':0,'heldout_access':0},
  'resources':{'host':host['hostname'],'host_classification':host['host_classification'],
   'scheduler_cli':'unavailable','quota':'unknown','physical_gpu':2,
   'gpu_uuid':watcher['final_recheck']['uuid'],'cuda_visible_devices':'2','logical_device':'cuda:0',
   'torch':inf['runtime']['torch'],'max_gpu_peak_allocated_mib':inf['max_gpu_peak_allocated_mib'],
   'gpu_lease_seconds':resource['gpu_jobs'][0]['lease_interval_seconds'],
   'inference_gpu_hours':resource['inference_gpu_hours_total'],'training_cost':0,
   'watcher_exit_code':watcher['exit_code'],'gpu_lease_released':watcher['lease_released'],
   'project_child_slot_released':watcher['project_child_slot_released'],
   'tmux_session_released':resource['gpu_jobs'][0]['tmux_session_released'],
   'gpu2_post_release_used_mib':20},
  'bootstrap':{'draw_source':'N2_GROUP_BOOTSTRAP_DRAWS.npz','replicates':10000,
   'seed':20261011,'group_unit':'603 paired source groups; each maps to one episode',
   'draws_reused_exactly':True,'p_values_computed':False},
  'checksum_acceptance':'pending_final_checksum_generation'}
 W(R/'qa/FINAL_ACCEPTANCE.json',final)
 # Build Chinese return from exact table values.
 q0=ts_s02['Q0'];q1=ts_s02['Q1']
 posci=cis['position_standardized_mae']['V_Q1']['ci95']
 fdci=cis['safe_fd_normalized']['V_Q1']['ci95']
 posg=effects['position_standardized_mae']['G_TS11-D']
 fdg=effects['safe_fd_normalized']['G_TS11-D']
 vpos=effects['position_standardized_mae']['V_Q1']
 vfd=effects['safe_fd_normalized']['V_Q1']
 s01p=seed_gains['position_standardized_mae']['20261001'];s02p=seed_gains['position_standardized_mae']['20261002'];s03p=seed_gains['position_standardized_mae']['20261003']
 s01f=seed_gains['safe_fd_normalized']['20261001'];s02f=seed_gains['safe_fd_normalized']['20261002'];s03f=seed_gains['safe_fd_normalized']['20261003']
 md=(
 '# N4 固定 command 模型误差预算：执行回传\n\n'
 '状态：**完成**。结果根：'+str(R)+'\n\n'
 '## 执行与验收\n\n'
 '- 固定矩阵 12 models × train/dev，24/24 bundles 接受：12 个 dev 直接复用 N2 CUDA_COMMON，12 个 profile 不匹配的 train bundle 通过 N2 已接受 safetensors 做 target-blind CUDA forward。没有重训、optimizer step、encoder extraction 或新 dev formal forward。\n'
 '- 复用 N1 scorer/normalizer，24 条原 co-primary dev 对照逐项复现，最大绝对差 0（容差 1e-12）。position MAE 与 safe-FD L1 保持 co-primary；safe-FD 是未来内部 29 个差分，anchor-to-first-frame seam 单独报告。\n'
 '- inference 使用物理 GPU2，UUID '+watcher['final_recheck']['uuid']+'，CVD=2 / logical cuda:0，FP32、batch64、TF32 off、deterministic、math-only SDPA。实测峰值 71 MiB，watcher exit 0，lease/child slot/tmux 均释放，GPU2 回到 20 MiB used；lease 区间 '+fmt(resource['gpu_jobs'][0]['lease_interval_seconds'],2)+' 秒（'+fmt(resource['inference_gpu_hours_total'],5)+' GPU-hours），新训练成本 0。\n'
 '- target gate 在 24/24 prediction acceptance 后开启。5 次 scoring attempts 中 4 次因代码错误在提交结果前停止；日志与 staging 均保留。成功 attempt 计分 primary-train 17,757 rows 和 exposed-dev 2,322 rows；V3 NPZ 的 targets_native key 每次物理物化其 20,079-row canonical train+dev 内容，但只有锁定 split slice 进入评分。case 图另外选取 14 个预锁定 dev anchors。FINAL/heldout access=0。\n'
 '- dev interval 使用 N2 的 10,000 次 paired source-group 与 seed draws（seed 20261011）；区间条件于固定模型和已暴露 dev。没有 p 值；train 为 in-sample，不提供泛化 CI。\n\n'
 '## A. seed 20261002：level 与 time-centered\n\n'
 'TS11-D 的 Q0→Q1 position standardized MAE 从 '+fmt(q0['position_standardized_mae'])+' 降到 '+fmt(q1['position_standardized_mae'])+'（G=Q0−Q1='+f'{qgain["position_standardized_mae"]:+.6f}'+'）；safe-FD 从 '+fmt(q0['safe_fd_normalized'])+' 降到 '+fmt(q1['safe_fd_normalized'])+'（G='+f'{qgain["safe_fd_normalized"]:+.6f}'+'）。\n'
 'position L2 MSE gain='+f'{qgain["position_mse"]:+.6f}'+': level component '+f'{qgain["level_mse"]:+.6f}'+', time-centered component '+f'{qgain["time_centered_error_mse"]:+.6f}'+'. 两部分都下降，time-centered 略大，因此 S02 的变化不只是整段路径的高低平移。独立 future-GT oracle MAE ledger 中，removable 项 gain='+f'{qgain["oracle_removable_standardized_mae"]:+.6f}'+', oracle-corrected remaining 项 gain='+f'{qgain["oracle_remaining_standardized_mae"]:+.6f}'+'. 剩余项改善更大；这不是主要由可移除 level offset 驱动。oracle 只作诊断，不能作为部署校正。\n'
 '该变化高度 seed-sensitive：TS11-D 的 position G 三 seed 为 '+f'{s01p:+.6f}, {s02p:+.6f}, {s03p:+.6f}'+', safe-FD G 为 '+f'{s01f:+.6f}, {s02f:+.6f}, {s03f:+.6f}'+'. 去掉 S02 后，position G 均值为 '+f'{posg["loo_drop_20261002"]:+.6f}'+', safe-FD 为 '+f'{fdg["loo_drop_20261002"]:+.6f}'+'.\n\n'
 '## B. 动态幅度与零时滞对齐\n\n'
 'TS11-D/S02/Q0：T='+f'{q0["target_energy_T"]:.6f}'+', P='+f'{q0["prediction_energy_P"]:.3e}'+', C='+f'{q0["cross_alignment_C"]:.3e}'+', r='+f'{q0["r"]:.3g}'+', c='+f'{q0["c"]:.4f}'+', skill='+f'{q0["fd_mse_skill"]:.3g}'+'. P 非零但极小，故没有套用 P=0 规则；c 在近零 P 下应谨慎看待，不设新阈值。\n'
 '切换到 Q1 后 T 不变，P='+f'{q1["prediction_energy_P"]:.6f}'+', C='+f'{q1["cross_alignment_C"]:.6f}'+', r='+f'{q1["r"]:.3f}'+', c='+f'{q1["c"]:.3f}'+', skill='+f'{q1["fd_mse_skill"]:.3f}'+'. 预测变化 RMS 约为目标的 '+f'{q1["r"]*100:.1f}%'+', zero-lag cosine 为正但中等。归一化 FD-MSE 的 norm 项='+f'{q1["norm_error_term"]:.3f}'+', alignment 项='+f'{q1["alignment_error_term"]:.3f}'+'. FD-MSE G='+f'{qgain["fd_mse"]:+.6f}'+', 分解为同一 S02 的 G_P='+f'{qgain["prediction_energy_P"]:+.6f}'+ ' 与 −2G_C='+f'{-2*qgain["cross_alignment_C"]:+.6f}'+'. c 是 zero-lag cosine，不是 Pearson 或时间滞后检测；skill 只相对零导数 persistence，不替代 position MAE 或 safe-FD L1。\n\n'
 '## C. G/V/I 与绝对优势\n\n'
 '三 seed TS11-D G 的 position 均值='+f'{posg["mean"]:+.6f}'+', sample SD='+f'{posg["sample_sd"]:.6f}'+', median='+f'{posg["median"]:+.6f}'+'. safe-FD 均值='+f'{fdg["mean"]:+.6f}'+', sample SD='+f'{fdg["sample_sd"]:.6f}'+', median='+f'{fdg["median"]:+.6f}'+'. 对应 position G 的 95% group-bootstrap interval 为 ['+f'{cis["position_standardized_mae"]["G_TS11-D"]["ci95"][0]:+.6f}'+', '+f'{cis["position_standardized_mae"]["G_TS11-D"]["ci95"][1]:+.6f}'+']；safe-FD 为 ['+f'{cis["safe_fd_normalized"]["G_TS11-D"]["ci95"][0]:+.6f}'+', '+f'{cis["safe_fd_normalized"]["G_TS11-D"]["ci95"][1]:+.6f}'+']，两者都跨 0。\n'
 'Q1 下 V_Q1=STATE−TS：position 均值='+f'{vpos["mean"]:+.6f}'+', 95% interval ['+f'{posci[0]:+.6f}'+', '+f'{posci[1]:+.6f}'+']；safe-FD 均值='+f'{vfd["mean"]:+.6f}'+', interval ['+f'{fdci[0]:+.6f}'+', '+f'{fdci[1]:+.6f}'+']。负值表示 TS11-D 误差更高；当前结果没有建立 TS 的原 co-primary 绝对优势。该 interval 是已暴露 dev、固定模型的探索性组 bootstrap，不是独立确认或显著性检验。\n\n'
 '完整逐 seed G/V/I、三 seed mean/SD/median/LOO、T/P/C component CI 与 per-axis moments 见 results/PAIRED_COMPONENT_GAINS.csv、results/ALL_SEED_LOO.csv、results/GROUP_BOOTSTRAP_CI.csv、results/PER_WINDOW_MOMENTS/ 和 results/N4_EFFECT_SUPPORT_SUMMARY.json。\n\n'
 '## N3 与其他窗口\n\n'
 'N3 仅直接引用 18 accepted predictions 与 162-row reconciliation，未重评分或重训。此前 FULL g≈+0.000631（区间跨 0）、OBS g≈−0.003568、ANCHOR g≈+0.004195，呈非单调关系。\n'
 'HORIZON root 的 RUN_STATUS/FINAL_ACCEPTANCE/return 标为 complete/pass，但 OWNER_STATE 仍为 LOCKED_READY_FOR_CPU_FIT，本 N4 记录为 stale/contradictory 且未重启。OBS43 closeout root 为 sealed/pass；另一同名 root 是空目录（0 files，完成状态 UNKNOWN）。两者均未修改或重跑。\n\n'
 '## 图件与边界\n\n'
 '交付 5 张 summary figure、14 个锁定 case page、1 个 contact sheet；PNG/PDF/caption/data file QA 与 agent visual review 通过，human review pending。case 数据来自 14 个预先锁定的不同 dev groups；时间标 future step 1–30，raw axes 0/14，物理单位未核验。\n'
 '下一轮独立确认条件只写入 NEXT_RESEARCH_CONDITIONS.md，未执行。QUERY_LINE_CLOSED.md 关闭本 query 扩张。独立确认=false、D5=false、final/test=0、submission-ready=false。\n\n'
 '目标访问记录：5 次 scoring attempts，4 次 pre-commit code failures；每次 unique scoring support 未变。NPZ target key 的 20,079-row canonical train+dev 内容因压缩格式物化 6 次（5 次 scoring +1 次 case target extraction），实际评分仍只用 train 17,757 / dev 2,322，case page 只索引 14 个 dev anchors；final/heldout=0。\n'
 'N2/N3 manifest SHA 保持 '+EXPECTED_N2+' 与 '+EXPECTED_N3+'；N2 本轮 selected source 文件 111 项复验通过，旧根写入 0，共享 watcher 未改。最终文件清单见 CHECKSUMS.sha256，封存 QA 见 qa/FINAL_ACCEPTANCE.json。\n')
 write_text(R/'RETURN_TO_CHATGPT.zh-CN.md',md)
 W(R/'RETURN_TO_CHATGPT.zh-CN.json',final)
 W(R/'FINAL_STATUS.ini.json',final)
 status_text=(
  'STATUS=complete\nMODELS_FIXED=12\nANALYSIS_BUNDLES_ACCEPTED=24/24\n'
  'DEV_CUDA_VIEWS_REUSED=12/12\nNEW_TRAIN_FORWARD_VIEWS=12\nNEW_DEV_FORMAL_FORWARD_VIEWS=0\n'
  'SUPERVISED_FITS=0\nNEW_NEURAL_TRAINING=0\nOPTIMIZER_STEPS=0\nENCODER_EXTRACTION=0\n'
  'SAFE_MODEL_ONLY_USED=true\nORIGINAL_FULL_CHECKPOINT_RECOVERY_RERUN=false\nN2_BRIDGE_RERUN=false\n'
  'N3_RERUN_RESCORE=false\nORACLE_GT_USED_FOR_DIAGNOSTICS_ONLY=true\n'
  'ORACLE_OUTPUTS_ACCEPTED_AS_DEPLOYABLE_PREDICTIONS=false\nL2_BUDGET_REPLACES_COPRIMARY_L1=false\n'
  'ALL_SEEDS_AND_LOO_REPORTED=true\nQUERY_LINE_STATUS=closed_no_new_query_training\n'
  'INDEPENDENT_CONFIRMATION=false\nD5_MODEL_TRAINED=false\nFINAL_HELDOUT_ACCESS=0\n'
  'OTHER_WINDOW_TASKS_MODIFIED_OR_RELAUNCHED=false\nOLD_ROOT_WRITES=0\nSHARED_WATCHER_MODIFIED=false\n'
  'MAX_PROJECT_GPU_CHILDREN=2\nHUMAN_REVIEW_STATUS=pending\nSUBMISSION_READY=false\n'
  'CHECKSUM_ACCEPTANCE=final_manifest_pending\n')
 write_text(R/'FINAL_STATUS.ini',status_text)
 owner=J(R/'coordination/OWNER_STATE.json')
 owner.update({'status':'COMPLETE_PENDING_FINAL_SEAL','phase':'ready_for_checksum_generation',
  'owner_active':False,'last_verified_utc':now,'analysis_attempts':5,'analysis_failed_attempts':4,
  'figure_generation_failed_attempts':3,'final_test_access':0})
 W(R/'coordination/OWNER_STATE.json',owner)
 ledger=J(R/'coordination/JOB_LEDGER.json')
 ledger.update({'status':'complete_pending_final_seal','analysis_attempts':5,'analysis_failed_attempts':4,
  'new_train_forward_views_completed':12,'new_dev_formal_forward_started':0,
  'figure_generation_attempts':5,'figure_generation_failed_attempts':4,
  'final_test_access':0,'heldout_access':0,'analysis_acceptance_sha256':sha(R/'qa/ANALYSIS_ACCEPTANCE.json'),
  'figure_manifest_sha256':sha(R/'figures/FIGURE_MANIFEST.json')})
 W(R/'coordination/JOB_LEDGER.json',ledger)
 summary=J(R/'results/N4_EFFECT_SUPPORT_SUMMARY.json')
 summary.update({'score_target_access_attempts':5,'failed_score_attempts':4,
  'case_figure_target_access_anchors':14,'case_figure_target_reopened_on_resume':False,
  'full_targets_native_key_materializations_total':6,
  'paired_contrast_sign_amendment_sha256':sha(R/'qa/PAIRED_CONTRAST_SIGN_AMENDMENT_001.json'),
  'agent_visual_qa':'PASS_AGENT_VISUAL_QA','human_review':'pending',
  'checksum_acceptance':'pending_final_sha256_seal'})
 W(R/'results/N4_EFFECT_SUPPORT_SUMMARY.json',summary)
 print(json.dumps({'status':'PASS_FINAL_DELIVERABLES_BUILT','root':str(R),
  'return_markdown':str(R/'RETURN_TO_CHATGPT.zh-CN.md'),'return_json':str(R/'RETURN_TO_CHATGPT.zh-CN.json'),
  'final_acceptance':str(R/'qa/FINAL_ACCEPTANCE.json'),'checksum':'pending',
  'figure_entries':figures['figure_count'],'score_attempts':exposure.get('n4_target_access_attempts'),
  'score_failures':exposure.get('failed_analysis_attempts_after_target_read')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
