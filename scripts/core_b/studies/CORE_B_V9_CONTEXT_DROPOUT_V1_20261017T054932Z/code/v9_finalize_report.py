#!/usr/bin/env python3
from __future__ import annotations
import csv,json,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
METHODS=('A0','AD','V0','VD','MIX0','MIXD')
SEEDS=(202610131,202610132,202610133)
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rc(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def fmt(v,n=6):return f'{float(v):.{n}f}'
def fpm(mean,sd):return fmt(mean)+' ± '+fmt(sd)
def mm(rows,method,metric):return next(x for x in rows if x['method']==method and x['metric']==metric)
def pair(effects,contrast,metric):return next(x for x in effects if x['contrast']==contrast and x['metric']==metric and x['grain']=='five_task_three_seed_macro')
def cirow(rows,contrast,metric):return next(x for x in rows if x['contrast']==contrast and x['metric']==metric)
def gainfmt(e,c):return fmt(e['gain_reference_minus_candidate'])+' ['+fmt(c['ci_2_5'])+', '+fmt(c['ci_97_5'])+']'
def md_table(headers,rows):
 out=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
 out.extend('| '+' | '.join(str(v).replace('|','\\|') for v in row)+' |' for row in rows)
 return '\n'.join(out)
def contrast_rows_names():
 return ('NUMCTX_DROP_VIDEO','NUMCTX_DROP_STATE','VIDEO_GAIN_0','VIDEO_GAIN_D','INTERACTION','MIX_GAIN','VD_VS_MIX0','VD_VS_S','VD_VS_PERSISTENCE','MIXD_VS_S','MIXD_VS_V0','MIXD_VS_OLD_SV50','V0_VS_OLD_V','A0_VS_OLD_A')
def main():
 fit=rj(ROOT/'qa/FORMAL_FIT_ACCEPTANCE.json');seal=rj(ROOT/'qa/PREDICTION_SEAL.json')
 exp=rj(ROOT/'inventory/OUTCOME_EXPOSURE.json');resource=rj(ROOT/'results/RESOURCE_SUMMARY.json')
 final=rj(ROOT/'qa/FINAL_ACCEPTANCE.json');figreview=rj(ROOT/'figures/FIGURE_VISUAL_REVIEW.json')
 scoring=rj(ROOT/'results/V9_SCORING_MANIFEST.json');single=rj(ROOT/'qa/DEV_TARGET_SINGLE_READ.json')
 synth=rj(ROOT/'qa/SYNTHETIC_SCORER_QA.json');pre=rj(ROOT/'qa/FORMAL_JOB_PREFLIGHT.json')
 recovery=rj(ROOT/'results/POST_TARGET_AGGREGATION_RECOVERY.json')
 v0replay=rj(ROOT/'qa/V0_PREDICTION_REPLAY.json');oldroot=rj(ROOT/'qa/OLD_ROOT_INTEGRITY.json');mask_schedule=rj(ROOT/'protocol/MASK_SCHEDULE.json')
 macro=rc(ROOT/'results/METHOD_MACRO.csv');primary=rc(ROOT/'results/PRIMARY_TASK_SEED.csv')
 effects=rc(ROOT/'results/PAIRED_EFFECTS.csv');cis=rc(ROOT/'results/PAIRED_GROUP_CI.csv')
 masks=rc(ROOT/'results/MASKED_DIAGNOSTIC.csv')
 if any(x.get('status')!='PASS' for x in (fit,resource,final,figreview,synth,pre,recovery)) or seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or oldroot.get('status')!='PASS':
  raise RuntimeError('one or more final acceptance gates are not PASS')
 if scoring.get('dev_target_archive_content_reads')!=1 or single.get('archive_content_reads')!=1 or final.get('final_heldout_payload_access')!=0:
  raise RuntimeError('target exposure/final access receipt invalid')
 main_table=[[m]+[fpm(mm(macro,m,x)['mean'],mm(macro,m,x)['sample_sd']) for x in METRICS] for m in METHODS]
 seed_table=[]
 for method in METHODS:
  vals=[]
  for seed in SEEDS:
   row=next(x for x in primary if x['method']==method and int(x['seed'])==seed and x['task_id']=='macro')
   vals.append(fmt(row['position_mae'])+' / '+fmt(row['safe_fd_l1'])+' / '+fmt(row['seam_safe_fd_l1']))
  seed_table.append([method]+vals)
 core=('NUMCTX_DROP_VIDEO','NUMCTX_DROP_STATE','VIDEO_GAIN_0','VIDEO_GAIN_D','INTERACTION','MIX_GAIN')
 contrast_rows=[[c]+[gainfmt(pair(effects,c,m),cirow(cis,c,m)) for m in METRICS] for c in core]
 other=('VD_VS_MIX0','VD_VS_S','VD_VS_PERSISTENCE','MIXD_VS_S','MIXD_VS_V0','MIXD_VS_OLD_SV50','V0_VS_OLD_V')
 other_rows=[[c]+[gainfmt(pair(effects,c,m),cirow(cis,c,m)) for m in METRICS] for c in other]
 masked_rows=[]
 for p in ('A0','AD','V0','VD'):
  xs=[x for x in masks if x['method']==p and x['grain']=='five_task_seed_macro']
  masked_rows.append([p]+[fpm(statistics.mean(float(x[m+'_masked_minus_full']) for x in xs),
    statistics.stdev(float(x[m+'_masked_minus_full']) for x in xs)) for m in METRICS])
 exposure_rows=[
  ['Formal fits','12/12','160 epochs × 20 updates = 3,200 updates each'],
  ['Training targets','true Y only','No teacher/soft target; 2,457,600 cumulative fit presentations'],
  ['Neural predictions','36 bundles','12 train full + 12 dev full + 12 dev force-mask'],
  ['Derived mixtures','6 bundles','MIX0=0.5(S_OPT+V0), MIXD=0.5(S_OPT+VD), native float64'],
  ['Primary dev evaluation','18 bundles / 3,330 evaluations','4 single-head arms × 3 seeds + 6 fixed mixtures'],
  ['Force-mask diagnostic','12 bundles / 2,220 evaluations','All receiver numeric-context blocks forced to zero'],
  ['New dev total','30 bundles / 5,550 evaluations','185 unique windows; 150 task views'],
  ['Dev archive read','1','After prediction seal; one archive read / one in-memory numpy load'],
  ['Final/heldout','0','Payload and task-level final aggregates remained closed'],
  ['Cumulative known-grain evaluations','14,985','9,435 inherited + 5,550 V9; legacy 35,523 unit-unverified excluded']]
 resource_rows=[
  ['Formal fits','12','0 failures / 0 OOM; all watcher exit 0'],
  ['GPU pool','RTX 3090 physical GPUs 0 and 1','dynamic_all_visible; watcher cap 2; leases all released'],
  ['Measured resource gate','Stage0 reserved 508 MiB','Admission required free memory ≥1,020 MiB'],
  ['Formal peak reserved','168 MiB A0/AD; 176 MiB V0/VD','Allocated peak 124.14/137.33 MiB by model family'],
  ['GPU fit elapsed sum',fmt(resource['formal_fit_elapsed_seconds_sum'],1)+' s','Sum of 12 per-fit process elapsed times'],
  ['Inference elapsed sum',fmt(resource['inference_elapsed_seconds_sum'],1)+' s','36 fresh-process label-free bundles'],
  ['Environment','Python 3.12.14; PyTorch 2.12.1+cu130; CUDA build 13.0','RTX 3090; child logical device cuda:0']]
 v0_lines=[f"{x['dataset_role']} seed {x['seed']}: arrays exact={x['prediction_arrays_exact_equal']}, max abs diff={fmt(x['max_abs_difference'],0)}, SHA equal={x['V8_hard_prediction_sha256']==x['V9_V0_prediction_sha256']}" for x in v0replay['comparisons']]
 md='''# V9 训练期随机数值上下文屏蔽：实验交付

**状态：完成。** 范围为固定开发集与固定 3 个 seed 的受控实验；final/heldout 未访问。

## 主要结论

在完整数值上下文主场景下，VD 没有优于 V0 的单模型总体误差：V0−VD 的 Position MAE 和 Safe-FD L1 为负，Seam 指标为正。AD 对 A0 的 Position MAE 也变差。交互项没有支持数值上下文屏蔽专门提升视频作用的一致证据；Safe-FD 交互为负且 95% source-group bootstrap 区间不跨 0。

固定 mixture 对照中，MIXD 相对 MIX0 只有 seam 指标有正向点估计和区间；Position 与 Safe-FD 没有改善。MIXD 仍优于历史 S 单模型，但这不等于 VD 单 head 优于 V0。强制屏蔽推理仅作为次要诊断，不替代完整输入主结果。

V0 三个 seed 的 checkpoint 与 V8 hard-video checkpoints 文件 SHA/state hash 完全一致；train/dev V0 预测数组与 V8 accepted hard-video 输出逐字节一致，最大绝对差为 0。

## 完整上下文主结果（五任务 macro，mean ± sample SD across 3 seeds）

'''+md_table(['Program','Position MAE','Safe-FD L1','Seam Safe-FD L1'],main_table)+'''

## 每 seed 主指标

每个单元格顺序为 Position MAE / Safe-FD L1 / Seam Safe-FD L1。

'''+md_table(['Program','202610131','202610132','202610133'],seed_table)+'''

## 预注册 paired contrasts

下面均为 reference − candidate，正值表示 candidate 误差更低；区间为 5,000 次 within-task paired source-group bootstrap（seed 20261016），条件于固定模型和重复曝光的 dev 集。

'''+md_table(['Contrast','Position gain [95% CI]','Safe-FD gain [95% CI]','Seam gain [95% CI]'],contrast_rows)+'''

INTERACTION 定义为 (E(AD)-E(VD)) - (E(A0)-E(V0))。V0−VD 的 Position/FD 都是负值；VD 的 seam 改善不构成视频单 head 总体优化成功。

## 固定比较与历史参考

'''+md_table(['Contrast','Position gain [95% CI]','Safe-FD gain [95% CI]','Seam gain [95% CI]'],other_rows)+'''

V0/V8 hard-video replay：

'''+ '\n'.join('- '+x for x in v0_lines)+'''

## 强制屏蔽输入的次要诊断

单元格为 E(force-mask) − E(full)，mean ± sample SD across seeds；正值表示强制屏蔽时误差更高。

'''+md_table(['Program','Position Δ','Safe-FD Δ','Seam Δ'],masked_rows)+'''

这项输入干预诊断不等同于条件互信息或物理因果检验。

## 训练、输入与随机数值计划

- 12 个新正式 fits 均使用真实 Y；没有 teacher/soft target 或 encoder 训练/feature extraction。
- 固定 LR 5e-4、AdamW、batch 64、160 epochs、3,200 updates、last epoch；没有 p/LR/epoch 搜索。
- 屏蔽位置为 normalizer 后、state projection 前，整个 [B,32,71] receiver state block out-of-place 置零。history_mask、relative_delta、独立 anchor_command、task 与视频输入保持。
- AD/VD 每 seed 共用 SHA-256 receiver×epoch mask bank；三 seed 计划 drop 率分别为 0.499692、0.500088、0.499644，训练日志合计观测为 0.499808。未重抽以凑精确 50%；最大连续 drop 分别为 23、20、20 个 epoch。每个 seed/task/epoch、episode coverage 和 receiver run 长度见 mask_banks/MASK_TASK_EPOCH_STATS.csv、MASK_EPISODE_STATS.csv、MASK_RECEIVER_STATS.csv。
- Stage0 的真实跨任务输入 sentinel、p=0 output/loss/gradient equality、force-drop block-zero、Y/RNG isolation 均通过；另做 2 个 Stage0 optimizer updates，不计入 12 个 formal fits。

## 预测、暴露与资源

'''+md_table(['项','实际结果','说明'],exposure_rows)+'''

'''+md_table(['资源项','数值','说明'],resource_rows)+'''

正式 fit queue 按 seed 轮换 arm，watcher 动态选择物理 GPU 0/1，最多两个 child 并发；显存门来自 Stage0 实测，没有增加小时数/GiB/正收益/最低 N 完成门。完整 ledger 在 results/RESOURCE_LEDGER.csv。

## 可复现边界与异常

- 置信区间条件于固定模型/recipe 和反复曝光的 185-window 开发集；不代表训练 seed 总体不确定性。session/trial 独立性仍未知，无 p 值。
- 标准化误差与原生数值轨迹不是物理世界精度；采集单位/机器人执行可行性仍未知。
- 首次 Stage0 因未使用的模块级路径常量失败；retry 的 Stage0 通过。封存与评分出现的 schema/post-processing 错误均保留在失败账本中；paired 表从已保存 loss rows 恢复，没有重读 dev archive。
- 无 formal fit 失败或 OOM；12 fits、36 neural predictions、6 derived mixtures 完成；final/heldout access 为 0。

## 交付路径

- 主要报告：RETURN_TO_CHATGPT.zh-CN.md；结构化摘要：RETURN_TO_CHATGPT.json
- 锁与计划：protocol/STUDY_LOCK.json、protocol/MASK_FIELD_MAP.json、protocol/MASK_SCHEDULE.json、protocol/RECIPE_LOCK.json、protocol/RESOURCE_LOCK.json
- 主要结果：results/PRIMARY_TASK_SEED.csv、results/MASKED_DIAGNOSTIC.csv、results/PAIRED_EFFECTS.csv、results/PAIRED_GROUP_CI.csv、results/SEED_LOO.csv、results/TASK_LOO.csv
- 预测封存与暴露：qa/PREDICTION_SEAL.json、qa/DEV_TARGET_SINGLE_READ.json、inventory/OUTCOME_EXPOSURE.json
- 图：figures/；SHA-256 manifest: CHECKSUMS.sha256
'''
 (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text(md,encoding='utf-8')
 structured={'schema_version':1,'status':'complete','study':'post_V8_training_only_numeric_context_dropout_development',
  'programs':list(METHODS),'seeds':list(SEEDS),
  'full_context_macro':{m:{metric:{'mean':float(mm(macro,m,metric)['mean']),'sample_sd':float(mm(macro,m,metric)['sample_sd'])} for metric in METRICS} for m in METHODS},
  'per_seed_macro':{m:{str(s):{metric:float(next(x for x in primary if x['method']==m and int(x['seed'])==s and x['task_id']=='macro')[metric]) for metric in METRICS} for s in SEEDS} for m in METHODS},
  'paired_effects':{c:{metric:{'gain_reference_minus_candidate':float(pair(effects,c,metric)['gain_reference_minus_candidate']),
   'ci_2_5':float(cirow(cis,c,metric)['ci_2_5']),'ci_97_5':float(cirow(cis,c,metric)['ci_97_5'])} for metric in METRICS} for c in contrast_rows_names()},
  'masked_diagnostic':{p:{metric:{'mean_mask_minus_full':float(statistics.mean(float(x[metric+'_masked_minus_full']) for x in masks if x['method']==p and x['grain']=='five_task_seed_macro')),
   'sample_sd_across_seeds':float(statistics.stdev(float(x[metric+'_masked_minus_full']) for x in masks if x['method']==p and x['grain']=='five_task_seed_macro'))} for metric in METRICS} for p in ('A0','AD','V0','VD')},
  'counts':{'formal_fits':12,'neural_predictions':36,'derived_mixtures':6,'dev_logic_bundles':30,'dev_loss_evaluations':5550,
   'primary_bundles':18,'primary_evaluations':3330,'forced_mask_bundles':12,'forced_mask_evaluations':2220,'task_views':150,
   'train_full_context_evaluations':15360,'true_Y_training_presentations':2457600},
  'exposure':{'prior_known_grain':9435,'new_v9_dev_loss_evaluations':5550,'cumulative_known_grain':14985,
   'legacy_35523':'unit_unverified_excluded','dev_target_content_reads':1,'preseal_dev_target_reads':0,'final_heldout_payload_access':0},
  'V0_replay':v0replay,'mask_schedule_by_seed':{x['seed']:{'overall_drop_rate':x['overall_drop_rate'],'max_contiguous_drop_epochs':x['max_contiguous_drop_epochs']} for x in mask_schedule['summaries']},
  'old_root_integrity':oldroot,'resource_summary':resource,'prediction_seal_status':seal['status'],
  'post_target_recovery':recovery['status'],'figure_visual_review':figreview['status'],
  'source_map':str(ROOT/'inventory/SOURCE_MAP.json'),'output_root':str(ROOT)}
 (ROOT/'RETURN_TO_CHATGPT.json').write_text(json.dumps(structured,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','markdown':str(ROOT/'RETURN_TO_CHATGPT.zh-CN.md'),'json':str(ROOT/'RETURN_TO_CHATGPT.json')},sort_keys=True))
if __name__=='__main__':main()

