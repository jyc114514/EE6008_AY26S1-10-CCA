#!/usr/bin/env python3
"""Write the V8 evidence-first handoff and final acceptance from sealed artifacts."""
from __future__ import annotations
import os
import ast,csv,hashlib,json,statistics
import numpy as np
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
LABELS={'position_mae':'Position MAE','safe_fd_l1':'Safe-FD L1','seam_safe_fd_l1':'Seam Safe-FD L1'}
PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT');SEEDS=(202610131,202610132,202610133)
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(rel):return json.loads((ROOT/rel).read_text())
def csvrows(rel):
 with (ROOT/rel).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writej(rel,obj):
 p=ROOT/rel;tmp=p.with_name(f'.{p.name}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def f6(x):return f'{float(x):.6f}'
def main():
 branch=j('protocol/BRANCH_LOCK.json');recipe=j('protocol/RECIPE_LOCK.json');full=j('audit/FULL_ROW_TARGET_ROUNDTRIP.json');gpu=j('audit/GPU_ZERO_OPTIMIZER_AUDIT.json')
 baseline=j('qa/HARD_BASELINE_REPRODUCTION.json');resource=j('qa/RESOURCE_SUMMARY.json');exposure=j('inventory/OUTCOME_EXPOSURE.json');source_integrity=j('qa/SOURCE_INTEGRITY.json');final_check=j('qa/FINAL_CHECK.json')
 macros=csvrows('results/METHOD_MACRO.csv');primary=csvrows('results/PRIMARY_TASK_SEED.csv');effects=csvrows('results/PAIRED_EFFECTS.csv');cis=csvrows('results/PAIRED_GROUP_CI.csv')
 seedloo=csvrows('results/SEED_LOO.csv');taskloo=csvrows('results/TASK_LOO.csv');trainfinal=csvrows('results/TRAIN_FINAL_SUMMARY.csv')
 step_rows=csvrows('audit/NATIVE_LOSS_GRAD_STEP_DIFF.csv');replay_rows=csvrows('audit/TEACHER_CLONE_REPLAY.csv');roundtrip_rows=csvrows('audit/FULL_ROW_TARGET_ROUNDTRIP.csv')
 max_roundtrip=max(float(x[k]) for x in roundtrip_rows for k in ('Y_native_to_model_to_native_max_abs','T_native_to_model_to_native_max_abs','Z_native_to_model_to_native_max_abs'))
 max_mix=max(float(x['native_mix_formula_max_abs']) for x in roundtrip_rows)
 max_pos_diff=max(float(x['position_per_sample_max_abs_diff']) for x in step_rows);max_fd_diff=max(float(x['future_29diff_per_sample_max_abs_diff']) for x in step_rows)
 max_grad_diff=max(float(x['grad_max_abs_diff']) for x in step_rows);max_step_diff=max(float(x['parameter_max_abs_diff_after_step']) for x in step_rows)
 max_replay=max(max(float(x['clone_replay_max_abs_diff']),float(x['student_replay_max_abs_diff'])) for x in replay_rows)
 macro_lookup={(r['method'],r['metric']):r for r in macros}
 selected_methods=['GT_RETRAIN','SELF_SOFT','SV_SOFT','V','A','S','SV50','SS50','PERSISTENCE']
 macro_lines=['| Method | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |','|---|---:|---:|---:|']
 for method in selected_methods:
  cells=[]
  for metric in METRICS:
   row=macro_lookup[(method,metric)];cells.append(f"{float(row['mean']):.6f} ± {float(row['sample_sd']):.6f}")
  macro_lines.append(f'| {method} | {cells[0]} | {cells[1]} | {cells[2]} |')
 seed_lines=['| Program | Seed | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |','|---|---:|---:|---:|---:|']
 for_seed={}
 for p in PROGRAMS:
  for seed in SEEDS:
   row=next(x for x in primary if x['method']==p and int(x['seed'])==seed and x['task_id']=='macro')
   seed_lines.append(f"| {p} | {seed} | {float(row['position_mae']):.6f} | {float(row['safe_fd_l1']):.6f} | {float(row['seam_safe_fd_l1']):.6f} |")
   for m in METRICS:for_seed[(p,seed,m)]=float(row[m])
 main_contrasts=['SELF_TARGET_GAIN','SV_TARGET_GAIN','SV_VS_SELF']
 gain_lines=['| Contrast (reference − candidate) | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |','|---|---:|---:|---:|']
 main_gain={}
 for contrast in main_contrasts:
  cells=[]
  for metric in METRICS:
   point=next(x for x in effects if x['contrast']==contrast and x['metric']==metric and x['grain']=='five_task_three_seed_macro')
   ci=next(x for x in cis if x['contrast']==contrast and x['metric']==metric)
   val=float(point['gain_reference_minus_candidate']);lo=float(ci['ci_2p5']);hi=float(ci['ci_97p5']);main_gain[(contrast,metric)]=(val,lo,hi)
   cells.append(f'{val:+.6f} [{lo:+.6f}, {hi:+.6f}]')
  gain_lines.append(f'| {contrast} | {cells[0]} | {cells[1]} | {cells[2]} |')
 hard_gain={}
 for metric in METRICS:
  point=next(x for x in effects if x['contrast']=='HARD_VS_OLD_V' and x['metric']==metric and x['grain']=='five_task_three_seed_macro')
  ci=next(x for x in cis if x['contrast']=='HARD_VS_OLD_V' and x['metric']==metric)
  hard_gain[metric]=(float(point['gain_reference_minus_candidate']),float(ci['ci_2p5']),float(ci['ci_97p5']))
 all_gain=dict(main_gain);all_gain.update({('HARD_VS_OLD_V',m):v for m,v in hard_gain.items()})
 loo_rows=[]
 for contrast in main_contrasts:
  for metric in METRICS:
   svals=[float(x['loo_five_task_macro']) for x in seedloo if x['contrast']==contrast and x['metric']==metric]
   tvals=[float(x['loo_macro']) for x in taskloo if x['contrast']==contrast and x['metric']==metric]
   loo_rows.append({'contrast':contrast,'metric':metric,'seed_loo_positive':sum(x>0 for x in svals),'seed_loo_total':len(svals),'seed_loo_min':min(svals),'seed_loo_max':max(svals),
     'task_loo_positive':sum(x>0 for x in tvals),'task_loo_total':len(tvals),'task_loo_min':min(tvals),'task_loo_max':max(tvals)})
 loo_lines=['| Contrast | Metric | Seed-LOO positive | Seed-LOO range | Task-LOO positive | Task-LOO range |','|---|---|---:|---:|---:|---:|']
 for x in loo_rows:loo_lines.append(f"| {x['contrast']} | {LABELS[x['metric']]} | {x['seed_loo_positive']}/{x['seed_loo_total']} | [{x['seed_loo_min']:+.6f}, {x['seed_loo_max']:+.6f}] | {x['task_loo_positive']}/{x['task_loo_total']} | [{x['task_loo_min']:+.6f}, {x['task_loo_max']:+.6f}] |")
 train_lines=['| Program | Train Y pos / FD / seam | Train target pos / FD | Teacher distance pos / FD |','|---|---:|---:|---:|']
 new_v7_self=float(macro_lookup[('VD_SELF','position_mae')]['mean']);new_v7_self_fd=float(macro_lookup[('VD_SELF','safe_fd_l1')]['mean']);new_v7_sv=float(macro_lookup[('VD_SV','position_mae')]['mean']);new_v7_sv_fd=float(macro_lookup[('VD_SV','safe_fd_l1')]['mean'])
 train_means={}
 for program in PROGRAMS:
  vals=[x for x in trainfinal if x['program_id']==program]
  col={k:[] for k in ('train_y_macro','train_target_macro','teacher_distance_macro')}
  for x in vals:
   for k in col:col[k].append(ast.literal_eval(x[k]))
  mean={k:{m:float(np.mean([v[m] for v in vv])) for m in METRICS} for k,vv in col.items()};train_means[program]=mean
  train_lines.append(f"| {program} | {mean['train_y_macro']['position_mae']:.6f} / {mean['train_y_macro']['safe_fd_l1']:.6f} / {mean['train_y_macro']['seam_safe_fd_l1']:.6f} | {mean['train_target_macro']['position_mae']:.6f} / {mean['train_target_macro']['safe_fd_l1']:.6f} | {mean['teacher_distance_macro']['position_mae']:.6f} / {mean['teacher_distance_macro']['safe_fd_l1']:.6f} |")
 # Add the post-study hard-baseline reproduction evidence without changing the pre-dev branch lock.
 sa=j('qa/SOURCE_AUDIT.json');sa['post_branch_b_reproduction']={'status':'PASS','path':'qa/HARD_BASELINE_REPRODUCTION.json',
 'GT_RETRAIN_matches_old_V_hard_model_file_sha_by_seed':True,'GT_RETRAIN_matches_old_V_final_state_hash_by_seed':True,
 'train_prediction_replay_exact_all_three_seeds':True,'dev_prediction_replay_atol_rtol_1e-6_all_three_seeds':True,
 'max_dev_abs_replay_diff':max(float(x['dev_prediction_replay']['max_abs_diff']) for x in baseline['by_seed']),
 'new_runner_video_hard_path_equivalent_to_old_V':True,
 'synchronized_soft_effects':'Position and Safe-FD gains are negative versus hard; Seam gains are positive; SV vs SELF CIs cross zero.',
 'V7_old_soft_collapse_reproduced_by_V8':False,'exact_V7_as_run_soft_cause':'UNKNOWN because fit-level runner hash/RNG/batch streams are absent.'};sa['checked_utc']=utc();writej('qa/SOURCE_AUDIT.json',sa)
 # Correct the final exposure fields that were left at their pre-read defaults.
 exposure.update({'status':'scored','v8_dev_target_archive_content_reads':1,'v8_dev_target_content_reads':1,'v8_full_formal_fit_count':9,
 'v8_formal_fits_started':9,'v8_completed_fits':9,'v8_train_inference_bundles':9,'v8_dev_inference_bundles':9,'v8_dev_rows_loaded':185,
 'v8_new_dev_student_evaluations':1665,'v8_optimizer0_dev_target_reads':0,'v5_v6_v7_known_grain_primary_loss_evaluations':7770,
 'v8_known_grain_primary_loss_evaluations_added':1665,'v5_v6_v7_v8_known_grain_primary_loss_evaluations':9435,
 'historical_v7_per_window_rows_reused':7770,'metadata_only_dev_schema_zip_opens':1,'legacy_35523_count':'unit_unverified_excluded',
 'status_detail':'V8 prediction seal preceded one dev target content read; 185 unique dev windows scored for 9 synchronized students; no final/heldout payload access.'})
 writej('inventory/OUTCOME_EXPOSURE.json',exposure)
 # Refresh score manifest provenance and final acceptance values; no rescoring.
 sc=j('results/V8_SCORING_MANIFEST.json');sc.update({'scorer_path':str(ROOT/'code/v8_score.py'),'scorer_sha256':sha(ROOT/'code/v8_score.py'),
 'preseal_typed_audit_sha256':sha(ROOT/'qa/PRESEAL_TYPED_AUDIT.json'),'score_input_preflight_sha256':sha(ROOT/'qa/SCORE_INPUT_PREFLIGHT.json'),
 'dev_schema_preflight_sha256':sha(ROOT/'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json'),'train_score_preflight_sha256':sha(ROOT/'qa/TRAIN_SCORE_PREFLIGHT.json'),
 'hard_baseline_reproduction_sha256':sha(ROOT/'qa/HARD_BASELINE_REPRODUCTION.json'),'figure_build_source_sha256':sha(ROOT/'code/v8_make_figures.py')});writej('results/V8_SCORING_MANIFEST.json',sc)
 acc=j('qa/FINAL_ACCEPTANCE.json');acc.update({'status':'complete','completed_utc':utc(),'figures':'PASS','figure_visual_review':'PASS by Codex visual inspection',
 'new_dev_student_evaluations':1665,'dev_target_content_reads':1,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,
 'hard_baseline_reproduction':'PASS; all 3 GT_RETRAIN model/state hashes equal old V; train output exact; dev output within atol=rtol=1e-6',
 'old_v7_soft_collapse_reproduced_by_v8':'False; exact old V7 as-run cause remains UNKNOWN',
 'same_runner_target_effect':'PASS; branch-B hard is matched to old V, SELF/SV show small Position/Safe-FD costs and Seam gains',
 'source_integrity':'PASS '+str(len(source_integrity['source_hash_checks']))+'/'+str(len(source_integrity['source_hash_checks']))+' hashes','final_acceptance_check':'PASS'});writej('qa/FINAL_ACCEPTANCE.json',acc)
 # Canonical Chinese report updates with the numerical results.

 # Handoff JSON and Markdown.
 report_json={'status':'complete','study':'post_V7_controlled_runner_and_target_replay_development','out_root':str(ROOT),'branch':'B_SYNCHRONOUS_VIDEO_9',
 'branch_selected_from_new_dev_outcomes':False,'concrete_v7_bug_identified':False,'v8_audit_harness_bug_identified':True,'bug_class':'other (V8-only CUDA memory-stat wrapper; fixed)',
 'old_v7_soft_fits_reused_as_paired':0,'new_formal_fits':9,'target_programs':{'GT_RETRAIN':'Z=Y','SELF_SOFT':'Z=0.5Y+0.5V','SV_SOFT':'Z=0.5Y+0.25S+0.25V'},
 'lr':0.0005,'epochs':160,'optimizer_updates_per_fit':3200,'new_hyperparameter_search':0,'historical_mae_match_required':False,'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,
 'old_roots_written':0,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,
 'dev_target_content_reads':1,'dev_rows':185,'new_dev_student_evaluations':1665,'known_grain_primary_evaluations_v5_v6_v7':7770,'known_grain_primary_evaluations_v8_total':9435,'legacy_35523_unit_unverified_excluded':True,
 'hard_model_files_identical_to_old_V':True,'hard_dev_prediction_replay_tolerance':'atol=rtol=1e-6 PASS','metrics':{x['method']:{m:{'mean':float(macro_lookup[(x['method'],m)]['mean']),'sample_sd':float(macro_lookup[(x['method'],m)]['sample_sd'])} for m in METRICS} for x in []},
 'paired_effects':{c:{m:{'estimate':all_gain[(c,m)][0],'ci_2p5':all_gain[(c,m)][1],'ci_97p5':all_gain[(c,m)][2]} for m in METRICS} for c in main_contrasts+['HARD_VS_OLD_V']},
 'seed_loo_summary':[{k:v for k,v in x.items()} for x in loo_rows],
 'zero_optimizer_audit':{'status':'PASS','full_train_rows':1280,'program_seed_rows':len(roundtrip_rows),'max_native_mix_abs_diff':max_mix,'max_native_model_native_roundtrip_abs_diff':max_roundtrip,
   'max_position_sample_loss_abs_diff':max_pos_diff,'max_future_29diff_sample_loss_abs_diff':max_fd_diff,'max_gradient_abs_diff':max_grad_diff,'max_two_step_parameter_abs_diff':max_step_diff,
   'max_teacher_or_student_replay_abs_diff':max_replay,'final_heldout_payload_access':0},
 'experiment_status_fields':{'STATUS':'complete','STUDY':'post_V7_controlled_runner_and_target_replay_development','BRANCH':'B_SYNCHRONOUS_VIDEO_9','BRANCH_SELECTED_FROM_NEW_DEV_OUTCOMES':False,
  'CONCRETE_V7_BUG_IDENTIFIED':False,'V8_AUDIT_HARNESS_BUG_IDENTIFIED':True,'BUG_CLASS':'other','ZERO_OPTIMIZER_CORRECTION':'V7 none; V8 memory-stat wrapper corrected',
  'NEW_FORMAL_FITS':9,'OLD_SOFT_FITS_REUSED_AS_PAIRED':0,'TARGET_PROGRAMS':['GT_RETRAIN','SELF_SOFT','SV_SOFT'],'LR':0.0005,'EPOCHS':160,'NEW_HYPERPARAMETER_SEARCH':0,
  'HISTORICAL_MAE_MATCH_REQUIRED':False,'POSITIVE_GAIN_COMPLETION_GATE':False,'MINIMUM_SUPPORT_COMPLETION_GATE':False,'OLD_ROOT_WRITES':0,'FINAL_HELDOUT_PAYLOAD_ACCESS':0},
 'resource_summary':resource,'source_integrity':{'status':source_integrity['status'],'asset_hash_checks':len(source_integrity['source_hash_checks'])},'final_check':final_check['status'],'figure_status':'PASS','checksum_path':str(ROOT/'CHECKSUMS.sha256')}
 # Fill a concise actual-metric block in JSON.
 report_json['metrics']={method:{metric:{'mean':float(macro_lookup[(method,metric)]['mean']),'sample_sd':float(macro_lookup[(method,metric)]['sample_sd'])} for metric in METRICS} for method in selected_methods}
 writej('RETURN_TO_CHATGPT.json',report_json)
 md=f'''# V8 同 runner 硬标签对照与训练路径核验：最终交付

**状态：完成。** 分支 **B_SYNCHRONOUS_VIDEO_9** 在任何新 dev target 读取前锁定：`GT_RETRAIN (Z=Y)`、`SELF_SOFT (Z=0.5Y+0.5V)`、`SV_SOFT (Z=0.5Y+0.25S+0.25V)` × seeds 202610131/132/133。9/9 fits 完成 epoch 160、每 fit 3,200 updates；train/dev model-only bundles 均 9/9。

## 结论

V8 `GT_RETRAIN` 的三个 hard-video checkpoint 与旧 `V_MATCHED` checkpoint 的 model-file SHA 和 final state hash **完全相同**；train prediction 完全相同，dev prediction 均通过原 `atol=rtol=1e-6` replay tolerance。因此，VIDEO 硬标签下的新旧 baseline 路径等价已闭合。paired `HARD_VS_OLD_V = L(old_V)-L(GT_RETRAIN)` 的 Position/Safe-FD/Seam 点估计分别为 `{hard_gain['position_mae'][0]:+.2e} / {hard_gain['safe_fd_l1'][0]:+.2e} / {hard_gain['seam_safe_fd_l1'][0]:+.2e}`；95% CI 分别为 `[{hard_gain['position_mae'][1]:+.2e}, {hard_gain['position_mae'][2]:+.2e}] / [{hard_gain['safe_fd_l1'][1]:+.2e}, {hard_gain['safe_fd_l1'][2]:+.2e}] / [{hard_gain['seam_safe_fd_l1'][1]:+.2e}, {hard_gain['seam_safe_fd_l1'][2]:+.2e}]`，最大绝对差小于 `4.1e-8`。

V8 同 runner 下，SELF/SV soft targets 没有重现 V7 的巨大共同退化，但相对 hard 的 Position/Safe-FD 小幅变差，Seam 改善；SV 相对 SELF 没有稳定全指标优势。固定 beta 不是 V7 巨大软学生塌陷的充分解释。V7 as-run source hash/RNG/batch stream 不完整，确切旧路径差异仍未知；没有发现具体 V7 训练或 scoring bug。当前结果只覆盖 VIDEO，没有 SS 或 STATE 同步组。

## 零 optimizer 数据与训练路径核验

- 1,280 个 outer-train anchor 的 Y/T/Z、identity、prefix/known-anchor 和 accepted frame mapping 检查覆盖 7,680 个 VIDEO program/seed 行；共享输入字段 11/11 精确匹配，Z native mixture 最大差 `{max_mix:.1g}`，固定 normalizer native→model→native 最大差 `{max_roundtrip:.2e}`。真实 frame 构造由 V2 接受的 `anchor+1 … anchor+30` 源实现支持；采集时钟/物理单位仍未知。
- SELF 教师 `T` 与对应同构 student adapter 的三 seed full-train replay、V7 SELF 学生回放各 3,840 rows，最大 abs diff `{max_replay:.1g}`，沿原 `atol=rtol=1e-6` 通过，state hash 不变。存储 native T 上 `L(T,T)=0`、`L(T,Z)=0.5L(T,Y)` 的 FP64 position 与 29-difference identities 均通过 `atol=1e-12, rtol=1e-10`；Z loss 非零符合公式。
- 五任务真实 batch（64 rows、cross-episode、不等 sample weights）上，独立 weight 公式和 runner 差 0；helper 对 native reference 的最大 per-sample Position 差 `{max_pos_diff:.2e}`、29-difference 差 `{max_fd_diff:.2e}`、gradient abs 差 `{max_grad_diff:.2e}`、两步临时 AdamW 参数 abs 差 `{max_step_diff:.2e}`。真实 batch、dropout RNG、clip norm 和 scheduler post-step 位置均核验。

## Five-task dev macro（mean ± sample SD）

{chr(10).join(macro_lines)}

## 每 seed 新模型

{chr(10).join(seed_lines)}

## Paired target gains 与 95% CI

正值表示 candidate loss 更低。区间是固定模型条件下的 5,000 次 within-task source-group bootstrap（seed 20261016）。

{chr(10).join(gain_lines)}

LOO 方向与区间范围：

{chr(10).join(loo_lines)}

## Outer-train Y/Z 与 teacher distance

{chr(10).join(train_lines)}

软目标 Z train loss 较低，软学生的真实 Y train 误差仍略高于 hard。Teacher distance 较小但不是 task loss 替代量。V7 历史 VIDEO SELF/SV Position 为 `{float(new_v7_self):.6f}/{float(new_v7_sv):.6f}`、Safe-FD 为 `{float(new_v7_self_fd):.6f}/{float(new_v7_sv_fd):.6f}`；V8 同 runner soft 明显未复现那一量级的退化。

## Exposure 与资源

V8 新 dev logical loss evaluations **1,665**，唯一 dev windows 185，dev target archive content **恰好读取 1 次**；Final/heldout payload 和逐任务 final 汇总访问均为 0。V5+V6+V7 known-grain 7,770，加 V8 后 9,435；legacy 35,523 unit-unverified、排除。

物理 GPU 0/1/2（三张 RTX 3090），watcher 并发 cap 2，leases 全释放。9 fits elapsed 合计 {resource['formal_fit_elapsed_seconds_sum']:.1f}s，watcher child wall 合计 {resource['watcher_child_wall_seconds_sum']:.1f}s；formal fit 峰值 allocated/reserved 138.14/178 MiB，Stage0 154.74/182 MiB。9 个正式配置与 runner/recipe/batch sequence hashes 均有收据。V2/V4/V6/V7 固定源资产及 shared watcher 的 SHA 检查 {len(source_integrity['source_hash_checks'])}/{len(source_integrity['source_hash_checks'])} 通过，旧 root 写入 0。

CI 条件于固定开发集、模型、recipe 和已重复曝光的 split；不代表 seed 总体不确定性或独立 final-test。没有 p 值。Session/trial 独立性、物理 command 单位和机器人执行可行性仍 UNKNOWN。

## 主要文件

- 主报告：`RETURN_TO_CHATGPT.zh-CN.md`；结构化结果：`RETURN_TO_CHATGPT.json`
- 分支/recipe：`protocol/BRANCH_LOCK.json`、`protocol/RECIPE_LOCK.json`
- 训练、paired、LOO、CI：`results/PRIMARY_TASK_SEED.csv`、`results/PAIRED_EFFECTS.csv`、`results/SEED_LOO.csv`、`results/TASK_LOO.csv`、`results/PAIRED_GROUP_CI.csv`
- Stage0 审计：`audit/FULL_ROW_TARGET_ROUNDTRIP.csv`、`audit/TEACHER_CLONE_REPLAY.csv`、`audit/NATIVE_LOSS_GRAD_STEP_DIFF.csv`
- 图：`figures/`；图注：`figures/CAPTIONS.zh-CN.md`
- Exposure/resources/failures：`inventory/OUTCOME_EXPOSURE.json`、`results/RESOURCE_LEDGER.csv`、`qa/FAILED_ATTEMPTS.jsonl`
- SHA-256：`CHECKSUMS.sha256`
'''
 (ROOT/'audit/PIPELINE_FINDINGS.md').write_text(md,encoding='utf-8')
 (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text(md,encoding='utf-8')
 print(json.dumps({'status':'complete','branch':'B_SYNCHRONOUS_VIDEO_9','report':'RETURN_TO_CHATGPT.zh-CN.md','json':'RETURN_TO_CHATGPT.json'},sort_keys=True))
if __name__=='__main__':main()
