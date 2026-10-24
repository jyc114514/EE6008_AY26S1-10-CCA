#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-007; source SHA-256: e9917b7d99c75a5cfd35bf1f15ca58a795c37fd607a5e5743ff093eb8a055233
from __future__ import annotations
import csv,hashlib,json,platform,socket,sys
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
import os
from pathlib import Path
ROOT=Path(os.environ.get("EE6008_V5_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT=Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
TASKS=[119,94,90,141,230];SEEDS=[202610131,202610132,202610133]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writej(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def main():
 now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 summary=json.loads((ROOT/'results/V5_METRICS_SUMMARY.json').read_text())
 exposure=json.loads((ROOT/'inventory/OUTCOME_EXPOSURE.json').read_text())
 formula_sha=sha(ROOT/'protocol/FORMULA_LOCK.json');seal_sha=sha(ROOT/'qa/PREDICTION_SEAL.json')
 prediction_seal=json.loads((ROOT/'qa/PREDICTION_SEAL.json').read_text())
 source_map=json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())
 primary=read(ROOT/'results/PRIMARY_TASK_SEED.csv');paired=read(ROOT/'results/PAIRED_EFFECTS.csv')
 macro_ci={}
 for r in read(ROOT/'results/PAIRED_GROUP_CI.csv'):
  if r['task_id']=='five_task_macro' and r['seed']=='mean_across_formal_seeds':
   macro_ci.setdefault(r['contrast'],{})[r['metric']]={'point':float(r['point_estimate']),'ci_low':float(r['ci_low']),'ci_high':float(r['ci_high'])}
 l2_rows=read(ROOT/'results/LEVEL_SHAPE_L2.csv')
 l2_summary={}
 for method in ['S','V','A','C','H_SV']:
  x=next(r for r in l2_rows if r['method']==method and r['task_id']=='five_task_macro' and r['seed']=='mean_across_formal_seeds')
  l2_summary[method]={k:float(x[k]) for k in ['level_mse','level_mse_sample_sd','shape_mse','shape_mse_sample_sd','position_mse','position_mse_sample_sd']}
 source_prov=read(ROOT/'inventory/SOURCE_INFERENCE_PROVENANCE.csv')
 paircost=read(ROOT/'results/V5_INFERENCE_COST_SUMMARY.csv')
 resource=read(ROOT/'results/RESOURCE_LEDGER.csv')
 figureqa=json.loads((ROOT/'qa/FIGURE_QA.json').read_text())
 transform=json.loads((ROOT/'qa/TRANSFORM_QA.json').read_text())
 recon=json.loads((ROOT/'qa/SOURCE_SCORE_RECONCILIATION.json').read_text())
 if prediction_seal.get('status')!='PASS' or len(prediction_seal.get('entries',[]))!=18:raise RuntimeError('prediction seal not PASS/18')
 if len(primary)!=15 or len(paired)!=120:raise RuntimeError('result matrix counts do not match locked output contract')
 if exposure.get('final_heldout_payload_access')!=0 or int(exposure.get('target_archive_load_operations_total',-1))!=4:raise RuntimeError('target/final exposure ledger incomplete')
 if len(read(ROOT/'results/PER_WINDOW_DERIVED_LOSSES.csv'))!=3330:raise RuntimeError('derived loss table incomplete')
 if figureqa.get('assistant_visual_review_status')!='PASS':raise RuntimeError('figure review status incomplete')
 # Complete exposure and owner records before the checksum manifest is written last.
 exposure.update({'status':'complete','study':'post_V4_frozen_level_shape_composition_development','formula_lock_sha256':formula_sha,'prediction_seal_sha256':seal_sha,
  'new_derived_prediction_bundles_accepted':'18/18','new_derived_dev_task_views':'90/90','new_derived_dev_future_window_logical_reads':3330,
  'final_heldout_payload_access':0,'old_root_writes':0,'complete_utc':now})
 writej(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
 owner_path=ROOT/'coordination/OWNER_STATE.json';owner=json.loads(owner_path.read_text())
 owner.update({'schema_version':1,'study':'post_V4_frozen_level_shape_composition_development','status':'complete','phase':'finalized_before_checksum',
  'active_jobs':0,'active_gpu_jobs':0,'owner_lock_acquired':False,'all_gpu_leases_released':True,'gpu_jobs_this_v5':0,
  'formula_lock_sha256':formula_sha,'prediction_seal_status':'PASS','prediction_seal_sha256':seal_sha,
  'new_prediction_bundles_accepted':18,'new_prediction_task_views':90,'target_archive_load_operations_total':4,
  'final_heldout_payload_access':0,'old_root_writes':0,'checksum_finalization_pending':False,'completed_utc':now,'updated_utc':now})
 writej(owner_path,owner)

 run_context={'schema_version':1,'run_id':ROOT.name,'started_utc':'2026-10-15T05:38:06Z','completed_utc':now,
  'working_directory':str(Path.cwd()),'project_root':str(PROJECT),'hostname':socket.gethostname(),
  'python_executable':sys.executable,'python_version':sys.version.split()[0],'platform':platform.platform(),
  'packages':{name:(version(name) if _installed(name) else 'not installed') for name in ['numpy','safetensors','matplotlib','pillow']},
  'experiment_execution':'CPU composition/scoring/plots; no GPU jobs or new fits; shared watcher not invoked',
  'current_v5_gpu_jobs':0,'final_heldout_payload_access':0,'repository_mutations':0,'repo_git_state':'not queried; experiment output root only'}
 writej(ROOT/'RUN_CONTEXT.json',run_context)

 positions=summary['absolute_five_task_episode_equal_mean_and_sample_sd']
 gains=summary['paired_gain_five_task_mean_and_sample_sd']
 task_support={int(x['task_id']):x for x in source_map['fold_membership']['task_support']}
 task_gain_rows=[]
 for task in TASKS:
  x=[r for r in paired if r['contrast'] in ['LEVEL_TRANSFER_GAIN','HYBRID_VS_OPT_STATE','HYBRID_VS_BIAS_VIDEO'] and int(r['task_id'])==task]
  means={}
  for contrast in ['LEVEL_TRANSFER_GAIN','HYBRID_VS_OPT_STATE','HYBRID_VS_BIAS_VIDEO']:
   q=[r for r in x if r['contrast']==contrast]
   means[contrast]={m:sum(float(r[m]) for r in q)/3 for m in ['position_mae','safe_fd_l1','seam_safe_fd_l1']}
  task_gain_rows.append({'task_id':task,'episodes':task_support[task]['episodes'],'anchors':task_support[task]['anchors'],
    'V_minus_HSV_position_gain':means['LEVEL_TRANSFER_GAIN']['position_mae'],
    'S_minus_HSV_position_gain':means['HYBRID_VS_OPT_STATE']['position_mae'],
    'BIAS_VIDEO_minus_HSV_position_gain':means['HYBRID_VS_BIAS_VIDEO']['position_mae']})
 writej(ROOT/'results/TASK_GAIN_SUMMARY.json',{'schema_version':1,'gain_sign':'L(reference)-L(candidate); positive favors H_SV',
  'task_means_across_three_formal_seeds':task_gain_rows,'created_utc':now})

 acceptance={
  'schema_version':1,'status':'complete','created_utc':now,'study':'post_V4_frozen_level_shape_composition_development',
  'principal_candidate':'H_SV','tasks':TASKS,'formal_seeds':SEEDS,
  'formula_lock_sha256':formula_sha,'prediction_seal_sha256':seal_sha,
  'formula_lock_precedes_new_v5_target_scoring':True,'nonempty_prediction_seal_before_target_reader':True,
  'accepted_derived_prediction_bundles':18,'expected_derived_prediction_bundles':18,'accepted_task_views':90,
  'unique_dev_receiver_windows':185,'task_support':source_map['fold_membership']['task_support'],
  'new_neural_fits':0,'optimizer_updates':0,'parameter_calibration_fits':0,'encoder_training':0,'encoder_extraction':0,'frozen_source_replays':0,
  'gt_used_to_construct_predictions':False,'post_score_target_free_regeneration_hashes_identical':len(transform['real_source_array_checks']['regeneration_after_initial_save_hashes_identical'])==18,
  'synthetic_and_real_transform_qa':'PASS','internal_fd_identity_native_max_abs':max(transform['real_source_array_checks']['internal_difference_max_native_abs'].values()),
  'internal_fd_identity_canonical_scaled_max_abs':max(transform['real_source_array_checks']['internal_difference_max_canonical_scaled_abs'].values()),
  'seam_identity_assumed':False,'source_score_reconciliation_status':recon['status'],'source_score_reconciliation_comparisons':recon['reconciled_comparisons'],
  'source_score_reconciliation_max_abs_delta':recon['max_abs_delta'],'new_derived_window_loss_evaluations':3330,
  'l2_diagnostic_window_evaluations':2775,'paired_group_bootstrap_draws':5000,'paired_group_bootstrap_seed':20261015,'p_values':False,
  'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,
  'prior_dev_future_window_logical_reads':35523,'new_dev_future_window_logical_reads':3330,
  'target_archive_path':exposure.get('target_archive_path'),'target_archive_sha256':exposure.get('target_archive_sha256'),
  'target_archive_load_operations_total':4,'target_array_window_rows_loaded_total':740,'unique_target_windows':185,
  'failed_scorer_attempts':3,'failed_attempt_target_rows_loaded':555,'failed_attempts_preserved':[
    'attempt 1: source-map prefix role resolver; 185 target windows loaded, no loss evaluation',
    'attempt 2: V4 primary table has no seam column; 185 target windows loaded, all 3330 loss rows preserved',
    'attempt 3: string/integer seed key mismatch while reusing saved scores; 185 target windows loaded, no re-scoring'],
  'final_heldout_payload_access':0,'old_root_writes':0,'source_roots_read_only':True,
  'figure_build_status':'PASS','assistant_visual_review_status':'PASS','human_figure_review_status':'pending',
  'outcome':'H_SV did not improve Position MAE over V, BIAS_VIDEO, S, or BIAS_STATE at the five-task macro. Safe-FD equality with V is inherited by formula. The fixed experiment is complete with a negative result.',
  'output_files':{'primary_task_seed':'results/PRIMARY_TASK_SEED.csv','paired_effects':'results/PAIRED_EFFECTS.csv','seed_loo':'results/SEED_LOO.csv',
    'task_loo':'results/TASK_LOO.csv','paired_group_ci':'results/PAIRED_GROUP_CI.csv','level_shape_l2':'results/LEVEL_SHAPE_L2.csv',
    'fd_identity':'qa/TRANSFORM_QA.json','resource_ledger':'results/RESOURCE_LEDGER.csv','figures':['figures/V5_DUAL_METRIC.png','figures/V5_PAIRED_GAINS_BY_TASK.png','figures/V5_LEVEL_SHAPE_L2.png','figures/V5_TRAJECTORY_EXAMPLES.png']},
 }
 writej(ROOT/'qa/FINAL_ACCEPTANCE.json',acceptance)

 metrics_table='''| Method | Position MAE, mean ± seed SD | Safe-FD L1, mean ± seed SD | Seam Safe-FD L1, mean ± seed SD |\n|---|---:|---:|---:|\n'''
 for method in ['S','V','BIAS_STATE','BIAS_VIDEO','H_SV','H_AV','H_SC','H_PV','MIX50','PERSISTENCE']:
  d=positions[method]
  def fmt(k):
   x=d[k];sd=x['sample_sd'];return f"{x['mean']:.6f} ± {sd:.6f}" if sd is not None else f"{x['mean']:.6f} (shared)"
  metrics_table+=f"| {method} | {fmt('position_mae')} | {fmt('safe_fd_l1')} | {fmt('seam_safe_fd_l1')} |\n"

 contrast_rows=[]
 for name in ['LEVEL_TRANSFER_GAIN','HYBRID_VS_OPT_STATE','HYBRID_VS_BIAS_STATE','HYBRID_VS_BIAS_VIDEO','MATCHED_SHAPE_GAIN','OPT_VS_COMMON_LEVEL_GAIN','HYBRID_VS_ANCHOR_LEVEL','HYBRID_VS_MIX50']:
  d=gains[name];ci=macro_ci[name]['position_mae']
  contrast_rows.append(f"| {name} | {d['position_mae']['mean']:+.6f} ± {d['position_mae']['sample_sd']:.6f} | [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}] | {d['safe_fd_l1']['mean']:+.6f} | {d['seam_safe_fd_l1']['mean']:+.6f} |")
 contrast_table='| 对比（reference − H_SV） | Position MAE gain, mean ± seed SD | Position MAE conditional 95% CI | Safe-FD gain | Seam gain |\n|---|---:|---:|---:|---:|\n'+'\n'.join(contrast_rows)+'\n'
 task_lines=[]
 for x in task_gain_rows:
  task_lines.append(f"| {x['task_id']} | {x['episodes']} | {x['anchors']} | {x['V_minus_HSV_position_gain']:+.6f} | {x['S_minus_HSV_position_gain']:+.6f} | {x['BIAS_VIDEO_minus_HSV_position_gain']:+.6f} |")
 task_table='| Task | Dev episodes | Anchors | V − H_SV | S − H_SV | BIAS_VIDEO − H_SV |\n|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(task_lines)+'\n'

 report='''# V5 冻结 state/video 水平—形状组合实验\n\n**状态：complete（开发实验；负结果按预注册固定矩阵收口）。**\n\nV5 生成并验收了 18/18 个固定组合预测 bundle、90 个 task views；没有新增训练或校准。主候选 H_SV 的 Safe-FD 与 V 由公式恒等继承，但它没有降低 Position MAE：五任务等权的 H_SV MAE 为 **0.159740 ± 0.000788**，高于 V、BIAS_VIDEO、S 与 BIAS_STATE。H_SV 的 seam error 为 **1.566279**，明显高于 V 的 **0.795761**。因此本轮结论是否定这条固定水平—形状组合在当前开发集上的 MAE 改善假设。\n\n## 绝对结果\n\n指标采用 V2 接受的 primary-train `target_sigma` 与 `safe_fd_scale`。MAE 和 Safe-FD 均为标准化误差；不表示物理世界精度。±值是三个固定模型 seed 间的样本 SD，Persistence 是共享参考，没有 seed SD。\n\n@@METRICS@@\n## H_SV 配对对比\n\n统一定义 `gain = L(reference) − L(candidate)`，正值才表示 H_SV 误差更低。置信区间为固定公式/固定模型下、5,000 次 within-task source-group paired bootstrap 的条件 95% 区间；它不涵盖跨训练 seed 总体推断或未观测 session 相关性。\n\n@@CONTRASTS@@\nH_SV 对 V 的 Position MAE gain 为 −0.001401，条件区间包含 0；对 BIAS_VIDEO 为 −0.001673，区间也包含 0。对 S 与 BIAS_STATE 的点估计更差，条件区间均低于 0。普通 MIX50 的 MAE 比 H_SV 低 0.004268；H_SV 不优于这一固定平均参考。\n\n逐 task、跨三个 seed 的配对 MAE gain 如下。负值表示 H_SV 更差。\n\n@@TASKS@@\n任务 141 和 94 的差值最大；119 也略差于 V，但 seed 间有方向变化；90 与 230 接近 0。Leave-one-seed-out 后，H_SV 对 V 的五任务 MAE gain 三个结果仍全为负（−0.002166、−0.001384、−0.000654）。在 15 个 task×seed 的 leave-one-task-out 汇总中，H_SV 对 S 和 BIAS_STATE 全部为负，对 V 只有 4/15 为正，对 BIAS_VIDEO 2/15 为正。所有 15 个 task×seed 绝对结果、120 条配对效应、seed LOO、task LOO 和条件区间见结果 CSV。\n\n## FD、水平与形状诊断\n\n五个组合的 29 步内部差分恒等式通过。最大 native 差为 **2.22×10⁻¹⁶**，最大按 accepted Safe-FD scale 归一后的差为 **3.26×10⁻¹⁴**，低于锁定的 `atol=1e-12, rtol=1e-10`。因此 H_SV 的内部 FD 与 V 完全重合是公式直接继承的性质，不能算作新学到的 FD 优势；相对 S 的 FD 变化也只是 V 相对 S 的既有差异。\n\nseam 没有不变性。H_SV 相对 V 的 seam error 增加 **0.770518**；相对 S 增加 **0.799336**，相对 BIAS_VIDEO 增加 **0.718335**。因此只检查未来内部 FD 会遗漏明显的 anchor→first-step 接缝代价。\n\ntrain-scale L2 诊断中，S 的 level MSE 是 **0.067307**，低于 V 的 **0.071411**；A 为 **0.070900**。H_SV 的 level MSE **0.067307** 与 S 一致，demeaned-shape MSE **0.049747** 与 V 一致，二者之和 0.117055。逐 anchor 恒等式最大误差为 `6.38e-16`。这说明预测 state 水平在此 L2 诊断上更接近目标，但不推出 L1 MAE 一定下降；真实 H_SV MAE 仍高于上述主要来源。H_SV 优于 H_AV 的 MAE gain 为 +0.001399，支持优化后的 S recipe 相对 A common recipe 有水平优势；H_SV 对 H_SC 的差异很小且条件区间跨 0，不能将其归因为纯视觉动态的因果效应。H_PV 明显更差（MAE 0.222943），说明当前 command anchor 不是足够的预测水平参考。\n\n## 计算代价\n\nV5 没有使用 GPU、没有 watcher job、没有新 encoder extraction 或模型 fit。H_SV 需要 S 与 V 两个冻结模型。V2 接受推理记录显示，每个 185-window bundle 的 S 平均 elapsed 为 **0.692481 s**，V 为 **0.758281 s**；两项历史 job elapsed 的和平均为 **1.450761 child-seconds**，三个 seed 的和为 1.957953、1.324986、1.069345 s。它是既有分开 job 的 elapsed 合计，不是 V5 测得的串行/并行端到端延迟。V5 在 CPU 上生成并序列化全部 18 个组合 bundle 用时约 **0.338 s**。组合可由冻结输出构造，不代表已集成到机器人执行接口，也不与单模型推理等成本。\n\n## 暴露与限制\n\n- 预测生成阶段 target access 为 0；18 个非空 bundle seal 通过后才打开 dev target。final/heldout payload access 为 **0**。\n- 实际 target archive 共加载 **4 次**，覆盖同一 185 个唯一 dev windows，共计 740 个 window-row loads。三次失败/恢复尝试已保留：第一次载入 185 行后在前缀角色解析处停止；第二次完成 3,330 条损失计算并保存，随后发现 V4 primary 表没有 seam 列；第三次载入后在恢复 CSV seed 类型核验处停止。最终一次读取复用已保存损失表，完成 L2 和轨迹数据；没有重复 3,330 条派生损失计算。失败记录见 `qa/SCORER_ATTEMPT_0*_FAILURE.json`。\n- 本轮唯一新派生评分是 18×185=**3,330** 个窗口损失评估；L2 另有 **2,775** 个诊断窗口评估。历史累计 dev scoring logical reads 是 35,523；本轮后主 scoring 累计 38,853，计入 L2 为 41,628，另将失败尝试实际加载的 555 个窗口单列后总访问运算数为 42,183。它们不是 42,183 个独立样本，唯一窗口仍为 185。\n- V4 per-window source 文件被读取三次做预检/恢复，累计解析 24,975 行；三次各筛选 3,885 个接受参考行，所有 B_D/C_D donor-map 行均被排除，实际独特参考为 3,885 行。source MAE/FD 与 V4 primary 表 210 个可比单元对账最大差为 0；V4 primary 表没有 seam 列，因此 seam 直接沿用已接受 per-window 表。\n- 结果只适用于这个已重复暴露的开发集；source-group CI 条件于固定模型和公式，不覆盖 seed 总体不确定性及未知 session 依赖。数值 command 的物理单位未解决。\n\n## 验收产物\n\n- 公式/对比锁：[FORMULA_LOCK.json](<@@P_FORMULA@@>)、[CONTRAST_LOCK.json](<@@P_CONTRAST@@>)；锁定 hash `@@FORMULA_SHA@@`。预测 seal：[PREDICTION_SEAL.json](<@@P_SEAL@@>)，hash `@@SEAL_SHA@@`。\n- 主结果：[PRIMARY_TASK_SEED.csv](<@@P_PRIMARY@@>)、[PAIRED_EFFECTS.csv](<@@P_PAIRED@@>)、[PAIRED_GROUP_CI.csv](<@@P_CI@@>)、[SEED_LOO.csv](<@@P_SEED_LOO@@>)、[TASK_LOO.csv](<@@P_TASK_LOO@@>)。\n- 诊断与记录：[LEVEL_SHAPE_L2.csv](<@@P_L2@@>)、[RESOURCE_LEDGER.csv](<@@P_RESOURCE@@>)、[FINAL_ACCEPTANCE.json](<@@P_ACCEPTANCE@@>)、[OUTCOME_EXPOSURE.json](<@@P_EXPOSURE@@>)。\n- 图：[双指标](<@@P_DUAL@@>)、[按 task 配对 gain](<@@P_PAIRED_FIG@@>)、[L2 level/shape](<@@P_L2_FIG@@>)、[轨迹](<@@P_TRAJ@@>). 自动/本轮视觉检查通过；人工图审仍为 pending，不是数字完成门。\n\n校验清单：[CHECKSUMS.sha256](<@@P_CHECKSUMS@@>)。\n\nV5 固定矩阵已收口；本轮未追加 alpha、路由、seed 或训练。\n'''
 report=report.replace('@@METRICS@@',metrics_table).replace('@@CONTRASTS@@',contrast_table).replace('@@TASKS@@',task_table)
 report=report.replace('@@FORMULA_SHA@@',formula_sha).replace('@@SEAL_SHA@@',seal_sha)
 report=report.replace("<@@P_DUAL@@>",'<'+str(ROOT/'figures/V5_DUAL_METRIC.png')+'>')
 report=report.replace("<@@P_PAIRED_FIG@@>",'<'+str(ROOT/'figures/V5_PAIRED_GAINS_BY_TASK.png')+'>')
 report=report.replace("<@@P_L2_FIG@@>",'<'+str(ROOT/'figures/V5_LEVEL_SHAPE_L2.png')+'>')
 report=report.replace("<@@P_TRAJ@@>",'<'+str(ROOT/'figures/V5_TRAJECTORY_EXAMPLES.png')+'>')
 for token,path in {
  '@@P_FORMULA@@':ROOT/'protocol/FORMULA_LOCK.json','@@P_CONTRAST@@':ROOT/'protocol/CONTRAST_LOCK.json',
  '@@P_SEAL@@':ROOT/'qa/PREDICTION_SEAL.json','@@P_PRIMARY@@':ROOT/'results/PRIMARY_TASK_SEED.csv',
  '@@P_PAIRED@@':ROOT/'results/PAIRED_EFFECTS.csv','@@P_CI@@':ROOT/'results/PAIRED_GROUP_CI.csv',
  '@@P_SEED_LOO@@':ROOT/'results/SEED_LOO.csv','@@P_TASK_LOO@@':ROOT/'results/TASK_LOO.csv',
  '@@P_L2@@':ROOT/'results/LEVEL_SHAPE_L2.csv','@@P_RESOURCE@@':ROOT/'results/RESOURCE_LEDGER.csv',
  '@@P_ACCEPTANCE@@':ROOT/'qa/FINAL_ACCEPTANCE.json','@@P_EXPOSURE@@':ROOT/'inventory/OUTCOME_EXPOSURE.json',
  '@@P_DUAL@@':ROOT/'figures/V5_DUAL_METRIC.png','@@P_PAIRED_FIG@@':ROOT/'figures/V5_PAIRED_GAINS_BY_TASK.png',
  '@@P_L2_FIG@@':ROOT/'figures/V5_LEVEL_SHAPE_L2.png','@@P_TRAJ@@':ROOT/'figures/V5_TRAJECTORY_EXAMPLES.png','@@P_CHECKSUMS@@':ROOT/'CHECKSUMS.sha256'}.items():
  report=report.replace(token,str(path))

 (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text(report,encoding='utf-8')

 return_json={'schema_version':1,'STATUS':'complete','STUDY':'post_V4_frozen_level_shape_composition_development','PRINCIPAL_CANDIDATE':'H_SV',
  'TASKS':TASKS,'FORMAL_SEEDS':SEEDS,'NEW_NEURAL_FITS':0,'OPTIMIZER_UPDATES':0,'PARAMETER_CALIBRATION_FITS':0,
  'NEW_DERIVED_DEV_BUNDLES_ACCEPTED':'18/18','NEW_DERIVED_DEV_TASK_VIEWS':'90/90','SOURCE_FROZEN_REPLAYS':0,
  'GT_USED_TO_CONSTRUCT_PREDICTIONS':False,'INTERNAL_FD_IDENTITY':{'native_max_abs':max(transform['real_source_array_checks']['internal_difference_max_native_abs'].values()),
   'canonical_scaled_max_abs':max(transform['real_source_array_checks']['internal_difference_max_canonical_scaled_abs'].values()),'status':'PASS'},
  'SEAM_IDENTITY_ASSUMED':False,'POSITIVE_GAIN_COMPLETION_GATE':False,'MINIMUM_SUPPORT_COMPLETION_GATE':False,
  'PRIOR_DEV_FUTURE_WINDOW_LOGICAL_READS':35523,'NEW_DEV_FUTURE_WINDOW_LOGICAL_READS':3330,
  'TARGET_ARCHIVE_LOADS':4,'TARGET_ARRAY_WINDOW_ROWS_LOADED':740,'UNIQUE_DEV_WINDOWS':185,'FINAL_HELDOUT_PAYLOAD_ACCESS':0,'OLD_ROOT_WRITES':0,
  'FORMULA_LOCK_SHA256':formula_sha,'PREDICTION_SEAL_SHA256':seal_sha,'TARGET_ARCHIVE_SHA256':exposure.get('target_archive_sha256'),
  'PRIMARY_MACRO_MEAN':{m:{metric:positions[m][metric]['mean'] for metric in ['position_mae','safe_fd_l1','seam_safe_fd_l1']} for m in positions},
  'PAIRED_GAIN_MACRO_MEAN_AND_SEED_SD':gains,'PAIRED_GROUP_CI_MEAN_ACROSS_SEEDS':macro_ci,
  'L2_LEVEL_SHAPE_MACRO':l2_summary,'SOURCE_SCORE_RECONCILIATION':{'status':recon['status'],'comparisons':recon['reconciled_comparisons'],'max_abs_delta':recon['max_abs_delta']},
  'EXPOSURE':{k:exposure.get(k) for k in ['target_archive_load_operations_total','target_array_window_rows_loaded_total','l2_diagnostic_window_evaluations','cumulative_dev_future_window_scoring_reads','cumulative_including_l2_diagnostic_evaluations','cumulative_including_l2_and_failed_target_load_rows','v4_per_window_rows_parsed_total','v4_reference_rows_selected_total','v4_per_window_file_read_operations_total','v4_unselected_rows_excluded_total']},
  'FAILED_SCORER_ATTEMPTS':3,'FIGURE_QA_STATUS':'PASS','HUMAN_FIGURE_REVIEW_STATUS':'pending',
  'RETURN_REPORT_PATH':str(ROOT/'RETURN_TO_CHATGPT.zh-CN.md'),'CHECKSUM_MANIFEST_PATH':str(ROOT/'CHECKSUMS.sha256'),'created_utc':now}
 writej(ROOT/'RETURN_TO_CHATGPT.json',return_json)
 print(json.dumps({'status':'REPORTS_WRITTEN','report':str(ROOT/'RETURN_TO_CHATGPT.zh-CN.md'),'formula_lock_sha256':formula_sha,'prediction_seal_sha256':seal_sha,
  'primary_macro_hsv_mae':positions['H_SV']['position_mae']['mean'],'target_loads':exposure.get('target_archive_load_operations_total')},indent=2))

def _installed(name):
 try:version(name);return True
 except PackageNotFoundError:return False

if __name__=='__main__':main()
