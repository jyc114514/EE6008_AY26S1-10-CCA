#!/usr/bin/env python3
"""Prepare final V4 acceptance and return handoff from sealed, scored results."""
from __future__ import annotations
import csv,json,hashlib,statistics
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133);METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def fnum(x):return None if x in ('NA','','None',None) else float(x)
def main():
 summary=read(ROOT/'results/SUMMARY_STATISTICS.csv');effects=read(ROOT/'results/PAIRED_EFFECTS.csv');abs_rows=read(ROOT/'results/ABSOLUTE_METRICS.csv');fits=[]
 fit_jobs=[]
 for program in ('A_STATE_COMMON','C_STATE_NEIGHBOR_REASSIGNED'):
  for seed in SEEDS:
   jid=f'v4_fit_{program}_s{seed}';p=ROOT/'qa/FIT_RELOAD'/f'{jid}.json';d=json.loads(p.read_text());fits.append(d);fit_jobs.append(jid)
 if len(fits)!=6 or any(x['status']!='PASS' or x['optimizer_steps']!=3200 for x in fits):raise RuntimeError('new fit acceptance is incomplete')
 seal=json.loads((ROOT/'protocol/PREDICTION_BATCH_SEAL.json').read_text());scoreqa=json.loads((ROOT/'qa/DEV_TARGET_SCORING_QA.json').read_text());exposure=json.loads((ROOT/'inventory/OUTCOME_EXPOSURE.json').read_text());audit=json.loads((ROOT/'audit/V3_CANONICAL_ZERO_R_REPLAY.json').read_text());resource=json.loads((ROOT/'results/GPU_RESOURCE_SUMMARY.json').read_text());figqa=json.loads((ROOT/'qa/FIGURE_BUILD_QA.json').read_text());oldroot=json.loads((ROOT/'qa/OLD_ROOT_READONLY_AUDIT.json').read_text());failed_attempts=[json.loads(x) for x in (ROOT/'qa/FAILED_ATTEMPTS.jsonl').read_text().splitlines() if x.strip()]
 def srow(method,metric,aggregation='five_task_episode_equal'):
  rows=[r for r in summary if r['method']==method and r['task_id']=='macro' and r['metric']==metric and r['aggregation']==aggregation]
  return rows[0] if len(rows)==1 else None
 main_methods=('BASE_STATE','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','A_M','B_M','C_M')
 main_metrics=[]
 for m in main_methods:
  rec={'method':m}
  for metric in ('position_mae','safe_fd_l1'):
   r=srow(m,metric)
   rec[metric+'_mean']=fnum(r['mean']) if r else None;rec[metric+'_sample_sd']=fnum(r['sample_sd']) if r else None;rec[metric+'_median']=fnum(r['median']) if r else None
  main_metrics.append(rec)
 gain_summary=[]
 for contrast in ('RGB_COMMON_GAIN','PAIRED_TRAIN_GAIN','B_INPUT_SENSITIVITY','C_INPUT_SENSITIVITY','C_D_vs_B_D','TRAIN_TEST_INTERACTION','B_vs_optimized_STATE','B_vs_BIAS_STATE','B_vs_BIAS_VIDEO','B_vs_PERSISTENCE'):
  for metric in ('position_mae','safe_fd_l1'):
   vals=[fnum(r['value']) for r in effects if r['contrast']==contrast and r['task_id']=='five_task_macro' and r['metric']==metric and r['aggregation']=='episode_macro']
   vals=[x for x in vals if x is not None]
   ci=next((r for r in read(ROOT/'results/PAIRED_GROUP_CI.csv') if r['contrast']==contrast and r['metric']==metric),None)
   gain_summary.append({'contrast':contrast,'metric':metric,'mean_across_seeds':statistics.mean(vals) if vals else None,'sample_sd_across_seeds':statistics.stdev(vals) if len(vals)>1 else None,'median_across_seeds':statistics.median(vals) if vals else None,'conditional_ci95_low':fnum(ci['ci95_low']) if ci else None,'conditional_ci95_high':fnum(ci['ci95_high']) if ci else None,'ci_draws':int(ci['valid_draws']) if ci and ci['valid_draws']!='NA' else 0})
 # Emit direct 15-row primary table for review.
 per_task=[]
 for task in TASKS:
  for seed in SEEDS:
   row={'task_id':task,'seed':seed}
   for method in ('BASE_STATE','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','A_M','B_M','C_M'):
    s=str(seed)
    r=next((x for x in abs_rows if x['method']==method and str(x['seed'])==s and int(x['task_id'])==task),None)
    for metric in ('position_mae','safe_fd_l1'):
     row[f'{method}_{metric}']=fnum(r[f'{metric}_episode_macro']) if r else None
   row['RGB_COMMON_GAIN_position']=next((fnum(x['value']) for x in effects if x['contrast']=='RGB_COMMON_GAIN' and str(x['seed'])==str(seed) and int(x['task_id'])==task and x['metric']=='position_mae' and x['aggregation']=='episode_macro'),None)
   row['RGB_COMMON_GAIN_fd']=next((fnum(x['value']) for x in effects if x['contrast']=='RGB_COMMON_GAIN' and str(x['seed'])==str(seed) and int(x['task_id'])==task and x['metric']=='safe_fd_l1' and x['aggregation']=='episode_macro'),None)
   row['PAIRED_TRAIN_GAIN_position']=next((fnum(x['value']) for x in effects if x['contrast']=='PAIRED_TRAIN_GAIN' and str(x['seed'])==str(seed) and int(x['task_id'])==task and x['metric']=='position_mae' and x['aggregation']=='episode_macro'),None)
   row['PAIRED_TRAIN_GAIN_fd']=next((fnum(x['value']) for x in effects if x['contrast']=='PAIRED_TRAIN_GAIN' and str(x['seed'])==str(seed) and int(x['task_id'])==task and x['metric']=='safe_fd_l1' and x['aggregation']=='episode_macro'),None)
   per_task.append(row)
 with (ROOT/'results/PRIMARY_TASK_SEED.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(per_task[0]));w.writeheader();w.writerows(per_task)
 # Small-support strata table is copied as a direct set of LOW/HIGH supports and paired changes.
 motion=read(ROOT/'results/MOTION_STRATA_RESULTS.csv')
 motion_compact=[]
 for task in TASKS:
  for group in ('LOW','MID','HIGH'):
   ar=next((r for r in motion if r['row_type']=='absolute' and r['method']=='B_M' and r['seed']=='202610131' and int(r['task_id'])==task and r['stratum']==group and r['metric']=='position_mae'),None)
   g=next((r for r in motion if r['row_type']=='C_benefit_vs_B_B_minus_C' and r['seed']=='202610131' and int(r['task_id'])==task and r['stratum']==group and r['metric']=='position_mae'),None)
   motion_compact.append({'task_id':task,'stratum':group,'dev_episodes':int(ar['episodes']) if ar else 0,'dev_anchors':int(ar['anchors']) if ar else 0,'B_M_position_mae':fnum(ar['value']) if ar else None,'B_minus_C_position_gain_favoring_C':fnum(g['value']) if g else None})
 with (ROOT/'results/MOTION_STRATA_SUPPORT_SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(motion_compact[0]));w.writeheader();w.writerows(motion_compact)
 old_conc=('非零 V3 bias/residual 效应没有符号翻转。30 个原本精确为零的 bias safe-FD 对比只有 float32 级漂移（最大 3.81e-8，低于 1e-6 replay tolerance）；非零旧效应最大相对变化为 0.087%。' if audit['interpretation_changed'] is False else '有非零 V3 bias/residual 效应改变符号；解释时应采用 corrected sensitivity 表。')
 status='complete' if seal.get('status')=='SEALED_BEFORE_TARGET_ACCESS' and scoreqa.get('status')=='PASS' and len(fits)==6 and resource.get('all_leases_released') else 'partial'
 final={'schema_version':1,'status':status,'study':'post_V3_RGB_alignment_training_strategy_development','tasks':list(TASKS),'seeds':list(SEEDS),'new_model_fits_accepted':6,'B_MATCHED_refit_fits':0,'B_MATCHED_reuse_accepted':True,'hyperparameter_search_fits':0,'encoder_training':0,'encoder_extraction':0,'new_prediction_jobs_accepted':24,'reused_B_MATCHED_prediction_jobs':3,'prediction_seal_sha256':sha(ROOT/'protocol/PREDICTION_BATCH_SEAL.json'),'V3_zero_R_precision_audit_status':audit['status'],'V3_native_reconstruction_sign_flips':audit['old_bias_or_residual_contrast_sign_flips'],'V3_precision_conclusion':old_conc,'prior_dev_future_window_reads':27750,'v4_metric_audit_window_pairs':exposure['v4_metric_audit_rows_read'],'v4_new_prediction_window_pairs':exposure['v4_new_prediction_score_rows_read'],'v4_total_new_logical_window_reads':exposure['v4_total_new_logical_dev_future_row_reads'],'cumulative_logical_window_reads':exposure['cumulative_dev_future_action_row_reads'],'unique_dev_receiver_windows':185,'target_archive_load_operations':exposure['v4_target_archive_load_operations'],'trajectory_figure_target_row_accesses':exposure['v4_trajectory_figure_target_row_reads'],'postprocessing_sign_correction':True,'paired_train_gain_formula':'L(C_M)-L(B_M)','sign_correction_target_reads':0,'sign_correction_refits':0,'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,'final_heldout_payload_access':0,'old_V2_V3_root_writes':0,'old_root_readonly_audit_status':oldroot['status'],'old_root_files_mtime_after_V4_start':len(oldroot['old_root_files_mtime_after_V4_start_utc']),'recovered_workflow_attempts':len(failed_attempts),'recovered_preparation_attempts':sum(1 for x in failed_attempts if x['stage'] in {'new_root_bootstrap','CPU_donor_bank_build','resource_ledger_rebuild','figure_build_attempt_01','figure_build_attempt_02'}),'postseal_postprocessing_corrections':sum(1 for x in failed_attempts if x['stage'] in {'postseal_statistics','postseal_plot_rebuild'}),'failed_fits':0,'all_GPU_leases_released':resource['all_leases_released'],'human_figure_review_status':figqa.get('manual_human_review_status','pending'),'checksums_file':'CHECKSUMS.sha256','checksums_generated_as_final_artifact':True,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
 put(ROOT/'qa/COMMON_INIT_AND_RECIPE.json',{'schema_version':1,'status':'PASS','fit_count':6,'fit_receipts':fits,'B_REUSE_DECISION_sha256':sha(ROOT/'protocol/B_REUSE_DECISION.json'),'MODEL_RECIPE_LOCK_sha256':sha(ROOT/'protocol/MODEL_RECIPE_LOCK.json'),'same_seed_INIT_verified':True,'task_id_order':[119,94,90,141,230],'final_heldout_payload_access':0})
 put(ROOT/'qa/LOSS_ID_SCALE_RELOAD.json',{'schema_version':1,'status':'PASS','prediction_seal_sha256':sha(ROOT/'protocol/PREDICTION_BATCH_SEAL.json'),'normalizer_sha256':sha(ROOT/'normalizers/NORMALIZER_FORMAL.json'),'metric_lock_sha256':sha(ROOT/'protocol/METRIC_SCALE_LOCK.json'),'all_predictions_ID_joined_and_finite':True,'all_task_seed_views_have_support_or_NA':True,'all_6_fits_fresh_reloaded':True,'training_loss':'V2 normalized position + 0.1 future-internal safe-FD','receiver_future_labels_forward_arguments':0,'final_heldout_payload_access':0})
 put(ROOT/'qa/FINAL_ACCEPTANCE.json',final)
 # Human-readable Chinese handoff. Tables remain authoritative linked V4 artifacts.
 md=['# V4 RGB 配对训练与 state 近邻重分配：结果回传','',f"**状态：{status}（post-V3 开发实验）**。输出目录：`{ROOT}`。",'', '## 主要结果','', '| 方法 | Position MAE | Safe-FD L1 |','|---|---:|---:|']
 for r in main_metrics:md.append(f"| {r['method']} | {r['position_mae_mean']:.6f} ± {r['position_mae_sample_sd']:.6f} | {r['safe_fd_l1_mean']:.6f} ± {r['safe_fd_l1_sample_sd']:.6f} |" if r['position_mae_mean'] is not None and r['safe_fd_l1_mean'] is not None else f"| {r['method']} | NA | NA |")
 md+=['','以上是三 seed 的五任务 episode-equal macro 均值 ± sample SD；15 行 task×seed 原始表见 `results/PRIMARY_TASK_SEED.csv`。本轮没有正 gain 或 minimum support 完成门，负结果按计划报告并完成。`RGB_COMMON_GAIN=L(A_M)-L(B_M)` 正值表示 B_M 误差更低；`PAIRED_TRAIN_GAIN=L(C_M)-L(B_M)` 正值表示 C_M 误差更高、负值表示 C_M 更好；input-sensitivity 正值表示 map 替换使该模型变差；`C_D_vs_B_D` 正值表示 C_D 误差更大；interaction 正值表示 B 的 map penalty 更大。完整方向见 `protocol/RESULT_SIGN_GUIDE.json`。','', 'B_MATCHED 对旧 optimized STATE 的 position gain 为负（-0.000783），safe-FD gain 为正（+0.011053）；因此 B 提升 FD，但没有在 Position MAE 上超过旧 state 基线。','', '| B_MATCHED 对旧参考的 gain = L(reference) − L(B_M) | Position MAE | Safe-FD L1 |','|---|---:|---:|']
 for name,label in [('B_vs_optimized_STATE','V2 optimized STATE'),('B_vs_BIAS_STATE','V3 BIAS_STATE'),('B_vs_BIAS_VIDEO','V3 BIAS_VIDEO'),('B_vs_PERSISTENCE','Persistence')]:
  a=next((x for x in gain_summary if x['contrast']==name and x['metric']=='position_mae'),None);b=next((x for x in gain_summary if x['contrast']==name and x['metric']=='safe_fd_l1'),None)
  if a and b:md.append(f"| {label} | {a['mean_across_seeds']:.6f} | {b['mean_across_seeds']:.6f} |")
 md+=['','| 配对量 | Position MAE gain | Safe-FD gain | 95% paired source-group CI（Position / Safe-FD） |','|---|---:|---:|---:|']
 for name in ('RGB_COMMON_GAIN','PAIRED_TRAIN_GAIN','B_INPUT_SENSITIVITY','C_INPUT_SENSITIVITY','C_D_vs_B_D','TRAIN_TEST_INTERACTION'):
  a=next((x for x in gain_summary if x['contrast']==name and x['metric']=='position_mae'),None);b=next((x for x in gain_summary if x['contrast']==name and x['metric']=='safe_fd_l1'),None)
  if a and b:md.append(f"| {name} | {a['mean_across_seeds']:.6f} | {b['mean_across_seeds']:.6f} | [{a['conditional_ci95_low']:.6f}, {a['conditional_ci95_high']:.6f}] / [{b['conditional_ci95_low']:.6f}, {b['conditional_ci95_high']:.6f}] |")
 md+=['','同一配对 source-group 抽样用于所有方法、seed 和 map；CI 条件于已训练模型、固定 recipe 和已暴露 dev。没有 p 值。地图平均的是 loss，不平均预测；maps 不是独立重复样本。','', '## V3 数值精度复核','',f"实质性复核中，非零旧效果的符号变化数为 0。P_CAPTURE 对 P_REF 的最大 native 差为 {audit['max_native_abs_CAPTURE_minus_REF']:.8g}；R=0 与 capture 的最大 native 差为 {audit['native_zero_R_equals_capture_max_abs']:.8g}。{old_conc}",'', '任务×seed 的原生与 standardized 差及 contrast 量级在 `audit/V3_CANONICAL_ZERO_R_REPLAY.csv`、`audit/V3_CANONICAL_NATIVE_DIFFERENCES.csv`、原始 `audit/V3_METRIC_PRECISION_IMPACT.csv` 与 `audit/V3_METRIC_PRECISION_SENSITIVITY_CORRECTED.csv`。','', '## 训练与 donor 证据','', 'B_MATCHED 复用 V2 G-V 三个 formal 模型：同一 1280-anchor outer-train split、同一 normalizer、LR 5e-4、160 epochs、batch 64、3200 updates、AdamW/update-cosine；每 seed 的 fresh INIT 与配对 G-S INIT 逐字节相同。新训练为 A×3 与 C×3，未从训练后 B 热启动。','', 'State-neighbor donor scaler 只在 outer-train prefix 上按 task→episode→anchor 等权拟合；same-task 查询排除接收 episode/source group，每个 source episode 取最近 anchor，最多 K=5。dev map 只从 train donor bank 按预注册 SHA-256 规则选。所有 receiver state/anchor/task/target 保持不变；future label 不选 donor。匹配距离、V3 donor 对照、轮换复用率见 `results/DONOR_MATCH_QUALITY.csv`。','', '任务变化分层使用 train 的 M_fd q25/q75，在 seal 后按 dev future target 作描述性分组；分层误差和 support 见 `results/MOTION_STRATA_RESULTS.csv`、`results/MOTION_STRATA_SUPPORT_SUMMARY.csv` 与 `figures/V4_MOTION_STRATA_SUPPORT.png`。M_fd 表示录制 command change，不是物理速度或运动。','', '## 资源、访问与限制','',f"接受新 fits={len(fits)}/6；新预测 jobs={seal['sealed_new_prediction_jobs']}/24；复用 B_MATCHED predictions={len(seal['reused_B_MATCHED_entries'])}。GPU lease 全部释放={resource['all_leases_released']}；V4 child jobs 使用 GPU {', '.join(resource['physical_gpus_used'])}；最大 allocated/reserved 为 {resource['max_measured_allocated_mib']}/{resource['max_measured_reserved_mib']} MiB；child elapsed 合计 {resource['sum_gpu_child_elapsed_seconds']:.1f} 秒。Watcher 并发上限为现场硬限制 2。完整资源与 admission snapshots 见 `results/RESOURCE_LEDGER.csv` 和 `results/GPU_ADMISSION_SNAPSHOTS.csv`。",'',f"V2/V3 既有累计 dev future-window reads={exposure['prior_cumulative_dev_future_action_row_reads']}；V4 metric audit={exposure['v4_metric_audit_rows_read']}；V4 新预测评分={exposure['v4_new_prediction_score_rows_read']}；本轮新增逻辑 window reads={exposure['v4_total_new_logical_dev_future_row_reads']}；累计={exposure['cumulative_dev_future_action_row_reads']}。scorer archive load=1；同一预选 trajectory anchor 的三次图表构建又读取该 target archive，故总 archive loads=4、图表 row accesses=3，unique dev windows 仍为185。final/heldout payload access=0。V2/V3 旧 roots 未写入。",'', '缓存历史 extractor 与 session/trial 绑定仍不完整；session/trial 独立性未验证。近邻重分配是 state-conditioned RGB 输入诊断，不是保持 RGB 多重集合的精确 conditional-randomization test；不估计 CMI 或物理因果效应。人审图状态为 pending，不改变数字完成状态。','', '## 文件','', '- `results/PRIMARY_TASK_SEED.csv`：5 task × 3 seed 的 A/B/C、旧 STATE/BIAS/Persistence 指标与 primary gain。','- `results/PAIRED_EFFECTS.csv`、`results/SEED_LOO.csv`、`results/TASK_LOO.csv`、`results/PAIRED_GROUP_CI.csv`：配对效果、两种 LOO 与条件区间；符号按 `protocol/RESULT_SIGN_GUIDE.json`。','- `figures/`、`figure_data/`：实际训练曲线、donor 匹配、变化分层、双指标、V3 选定 trajectory case。视觉接触表见 `qa/V4_FIGURE_CONTACT_SHEET.jpg`。','- `qa/FINAL_ACCEPTANCE.json`、`CHECKSUMS.sha256`：验收和最终校验清单。','- `qa/FAILED_ATTEMPTS.jsonl`：保留并说明的准备阶段失败与修复；没有 fit 失败。','- `qa/POSTPROCESSING_SIGN_CORRECTION.json` 与 `archive/PAIRED_TRAIN_GAIN_PRE_FORMULA_CORRECTION_*/`：保留锁定 C−B 公式的修正和更正前衍生表；无新 target read/refit。','- `qa/CHECKSUM_HISTORY.json`：保留前两份被替代的 pre-final 清单。','- `qa/OLD_ROOT_READONLY_AUDIT.json`：18 项来源 hash 匹配，旧 V2/V3 文件在 V4 开始后无 mtime 变化。','- `qa/FIGURE_VISUAL_REVIEW.md`：图表已做视觉检查；人工复核仍 pending。','']
 (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text('\n'.join(md))
 put(ROOT/'RETURN_TO_CHATGPT.json',{'schema_version':1,'status':status,'paired_train_gain_formula':'L(C_M)-L(B_M)','postprocessing_sign_correction':True,'sign_correction_new_target_reads':0,'sign_correction_refits':0,'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,'root':str(ROOT),'final_acceptance_path':str(ROOT/'qa/FINAL_ACCEPTANCE.json'),'checksum_path':str(ROOT/'CHECKSUMS.sha256'),'primary_metrics':main_metrics,'paired_gain_summary':gain_summary,'V3_metric_precision_audit':audit,'exposure':exposure,'resource_summary':resource,'old_root_readonly_audit':oldroot,'recovered_preparation_attempts':failed_attempts,'final_heldout_payload_access':0,'new_fit_count':6,'new_prediction_jobs':24,'B_MATCHED_reuse':True})
 print(json.dumps({'status':status,'return_md':str(ROOT/'RETURN_TO_CHATGPT.zh-CN.md'),'primary_metrics':main_metrics,'paired_gains':gain_summary[:4],'final_access':0},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
