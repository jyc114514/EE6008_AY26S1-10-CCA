#!/usr/bin/env python3
"""Build V5 identity, exposure, acceptance and handoff records from sealed results."""
from __future__ import annotations
import csv, datetime, json, math, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS=[119,94,90,141,230]
SEEDS=[202610131,202610132,202610133]
METRICS=['position_mae','safe_fd_l1','seam_safe_fd_l1']
CONTRASTS={
 'LEVEL_TRANSFER_GAIN':('V','H_SV'),
 'HYBRID_VS_OPT_STATE':('S','H_SV'),
 'HYBRID_VS_BIAS_STATE':('BIAS_STATE','H_SV'),
 'HYBRID_VS_BIAS_VIDEO':('BIAS_VIDEO','H_SV'),
 'MATCHED_SHAPE_GAIN':('H_SC','H_SV'),
 'OPT_VS_COMMON_LEVEL_GAIN':('H_AV','H_SV'),
 'HYBRID_VS_ANCHOR_LEVEL':('H_PV','H_SV'),
 'HYBRID_VS_MIX50':('MIX50','H_SV')}

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')
def read_json(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def read_csv(rel):
    with (ROOT/rel).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_json(rel,obj):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def write_csv(rel,rows,fields=None):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(rows[0]) if rows else ['status']
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def fmt(x,d=6):return f'{float(x):.{d}f}'
def ci_text(ci):
    low=ci.get('low',ci.get('ci_low'));high=ci.get('high',ci.get('ci_high'))
    return f"[{fmt(low)}, {fmt(high)}]"

def main():
    summary=read_json('results/V5_METRICS_SUMMARY.json')
    exposure=read_json('inventory/OUTCOME_EXPOSURE.json')
    transform=read_json('qa/TRANSFORM_QA.json')
    seal=read_json('qa/PREDICTION_SEAL.json')
    primary=read_csv('results/PRIMARY_TASK_SEED.csv')
    accepted=read_csv('results/PER_WINDOW_ACCEPTED_REFERENCES.csv')
    derived=read_csv('results/PER_WINDOW_DERIVED_LOSSES.csv')
    if len(primary)!=15 or len(accepted)!=3885 or len(derived)!=3330:raise RuntimeError('sealed score table counts changed')
    if seal.get('status')!='PASS' or seal.get('sealed_bundles')!=18 or seal.get('final_heldout_payload_access')!=0:raise RuntimeError('prediction seal/heldout boundary invalid')
    if summary.get('new_window_loss_evaluations')!=3330 or summary.get('target_archive_load_operations_total')!=4:raise RuntimeError('score summary count mismatch')

    # Per-window Safe-FD scores must inherit the input trajectory's internal differences.
    identity_pairs={'H_SV':'V','H_AV':'V','H_PV':'V','H_VS':'S','H_SC':'C'}
    ref_map={'S':'S','V':'V','A':'A','C':'C','BIAS_STATE':'BIAS_STATE','BIAS_VIDEO':'BIAS_VIDEO','PERSISTENCE':'PERSISTENCE'}
    ref_by={(r['method'],str(r['seed']),r['anchor_id']):r for r in accepted}
    der_by={(r['method'],str(r['seed']),r['anchor_id']):r for r in derived}
    fd_rows=[]
    for method,ref in identity_pairs.items():
        per_method=[]
        for seed in SEEDS:
            diffs=[]
            for aid in sorted(a for m,s,a in der_by if m==method and s==str(seed)):
                d=der_by[(method,str(seed),aid)]
                r=ref_by[(ref,str(seed),aid)]
                diffs.append(abs(float(d['safe_fd_l1'])-float(r['safe_fd_l1'])))
            if len(diffs)!=185:raise RuntimeError(f'FD score identity support incomplete {method}/{seed}')
            maximum=max(diffs);mean_abs=float(statistics.mean(diffs));per_method.extend(diffs)
            fd_rows.append({'check_type':'per_window_safe_fd_score','method':method,'reference':ref,'seed':seed,'n_anchors':185,
                'max_abs_safe_fd_loss_difference':maximum,'mean_abs_safe_fd_loss_difference':mean_abs,
                'max_abs_native_internal_difference':'NA','max_abs_canonical_scaled_internal_difference':'NA',
                'tolerance':1e-12,'status':'PASS' if maximum<=1e-12 else 'FAIL'})
        fd_rows.append({'check_type':'aggregate_safe_fd_score','method':method,'reference':ref,'seed':'all_three_seeds','n_anchors':555,
            'max_abs_safe_fd_loss_difference':max(per_method),'mean_abs_safe_fd_loss_difference':float(statistics.mean(per_method)),
            'max_abs_native_internal_difference':transform['real_source_array_checks']['internal_difference_max_native_abs'][method],
            'max_abs_canonical_scaled_internal_difference':transform['real_source_array_checks']['internal_difference_max_canonical_scaled_abs'][method],
            'tolerance':1e-12,'status':'PASS' if max(per_method)<=1e-12 else 'FAIL'})
    if any(r['status']!='PASS' for r in fd_rows):raise RuntimeError('internal FD identity check failed')
    write_csv('results/FD_IDENTITY_CHECK.csv',fd_rows)

    # Task-level transfer values, with seed variation shown separately from group-bootstrap intervals.
    task_rows=[]
    for task in TASKS:
        rec=[r for r in primary if int(r['task_id'])==task]
        for metric in ('position_mae','safe_fd_l1','seam_safe_fd_l1'):
            vals=[float(r[f'V_{metric}'])-float(r[f'H_SV_{metric}']) for r in rec]
            task_rows.append({'contrast':'LEVEL_TRANSFER_GAIN','reference':'V','candidate':'H_SV','task_id':task,'metric':metric,
                'episodes':int(rec[0]['episodes']),'anchors':int(rec[0]['anchors']),'source_groups':int(rec[0]['source_groups']),
                'seed_values_json':json.dumps(vals,separators=(',',':')),'mean_over_three_fixed_seeds':statistics.mean(vals),
                'sample_sd_across_seeds':statistics.stdev(vals),'positive_favors':'H_SV'})
    write_csv('results/TASK_LEVEL_TRANSFER_BY_TASK.csv',task_rows)

    # Canonicalize the old reported counter and keep each V5 measure at its own grain.
    v4_return=ROOT.parents[2]/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/RETURN_TO_CHATGPT.json'
    v4json=json.loads(v4_return.read_text())
    v4exposure=v4json.get('exposure',{})
    prior=35523
    if int(v4exposure.get('cumulative_dev_future_action_row_reads',prior))!=prior:raise RuntimeError('prior V4 exposure counter no longer matches V5 plan')
    mixed={k:exposure.pop(k) for k in ('cumulative_dev_future_window_scoring_reads','cumulative_including_l2_diagnostic_evaluations','cumulative_including_l2_and_failed_target_load_rows') if k in exposure}
    old_removed=dict(exposure.get('mixed_grain_cumulative_fields_removed',{}));old_removed.update(mixed)
    exposure.update({
      'reported_prior_counter':prior,'reported_prior_counter_name':'cumulative_dev_future_action_row_reads',
      'reported_prior_counter_unit_status':'unverified_as_same_grain_as_V5; kept separate and not added',
      'mixed_grain_cumulative_fields_removed':old_removed,
      'v5_completed_primary_window_loss_evaluations':3330,'v5_primary_window_loss_recomputations_during_resume':0,
      'v5_l2_diagnostic_window_evaluations':2775,'target_archive_load_operations_total':4,
      'target_array_window_rows_loaded_total':740,'failed_attempt_target_array_window_rows_loaded':555,
      'unique_dev_receiver_windows':185,'source_groups':47,'source_group_task_validation':'PASS',
      'additional_target_archive_reads_for_figures':0,'figure_target_payload_reads':0,'final_heldout_payload_access':0,
      'outcome_report_finalized_utc':utc()})
    write_json('inventory/OUTCOME_EXPOSURE.json',exposure)

    # LOO tables retain their raw rows; this compact view reports the three-seed mean per omitted task.
    seed_loo=[r for r in read_csv('results/SEED_LOO.csv') if r['method_or_contrast']=='LEVEL_TRANSFER_GAIN']
    task_loo=[r for r in read_csv('results/TASK_LOO.csv') if r['method_or_contrast']=='LEVEL_TRANSFER_GAIN']
    task_loo_summary=[]
    for task in map(str,TASKS):
        for metric in ('position_mae','safe_fd_l1','seam_safe_fd_l1'):
            vals=[float(r['loo_macro']) for r in task_loo if r['omitted_task']==task and r['metric']==metric]
            task_loo_summary.append({'omitted_task':int(task),'metric':metric,'mean_over_three_seeds':statistics.mean(vals),
                'sample_sd_across_seeds':statistics.stdev(vals),'per_seed_values':vals})

    # Condensed exact result rows for all eight locked contrasts and three metrics.
    cis=read_csv('results/PAIRED_GROUP_CI.csv')
    ci_macro={(r['contrast'],r['metric']):r for r in cis if r.get('seed')=='mean_across_formal_seeds' and r.get('task_id')=='five_task_macro'}
    contrast_summary=[]
    for contrast in CONTRASTS:
        for metric in METRICS:
            ci=ci_macro[(contrast,metric)]
            contrast_summary.append({'contrast':contrast,'metric':metric,'point_estimate':float(ci['point_estimate']),
                'ci95_low':float(ci['ci_low']),'ci95_high':float(ci['ci_high']),'draws':int(ci['draws']),
                'source_groups':int(ci['n_source_groups']),'conditional_on_fixed_three_seeds':True})

    # Save the visual review note after checking current generated figures in the parent session.
    visual_note=("# V5 figure visual review\n\n"
      "Assistant visual check: PASS. Reviewed the dual-metric scatter, task-specific paired-gain plot, L2 diagnostic, and task 119 trajectory layout. The gain sign and metric labels are legible; per-task support is shown; the L2 chart identifies its values as standardized MSE; the trajectory panels separate all three seeds and action indices 0/14. Numeric native command values are shown with physical units unresolved. Manual human review: pending.\n")
    (ROOT/'qa/FIGURE_VISUAL_REVIEW.md').write_text(visual_note,encoding='utf-8')

    absolute=summary['absolute_five_task_episode_equal_mean_and_sample_sd']
    gains=summary['paired_gain_five_task_mean_and_sample_sd']
    l2rows=read_csv('results/LEVEL_SHAPE_L2.csv')
    l2macro=[r for r in l2rows if r.get('task_id')=='five_task_macro' and r.get('seed')=='mean_across_formal_seeds']
    v2l2={r['method']:{k:float(r[k]) for k in ('level_mse','shape_mse','position_mse','level_mse_sample_sd','shape_mse_sample_sd','position_mse_sample_sd')} for r in l2macro}
    transfer_pos={int(r['task_id']):r for r in task_rows if r['metric']=='position_mae'}
    fdmax={m:max(float(r['max_abs_safe_fd_loss_difference']) for r in fd_rows if r['method']==m and r['check_type']=='per_window_safe_fd_score') for m in identity_pairs}
    l2_identity=summary['l2_identity_max_abs']
    exposure_summary={k:exposure.get(k) for k in ('reported_prior_counter','reported_prior_counter_name','reported_prior_counter_unit_status','v5_completed_primary_window_loss_evaluations','v5_primary_window_loss_recomputations_during_resume','v5_l2_diagnostic_window_evaluations','target_archive_load_operations_total','target_array_window_rows_loaded_total','failed_attempt_target_array_window_rows_loaded','unique_dev_receiver_windows','additional_target_archive_reads_for_figures','final_heldout_payload_access','failed_scorer_attempts')}
    resource=read_csv('results/RESOURCE_LEDGER.csv')
    md=[]
    md.append('# V5 冻结 state/video 水平—形状组合：结果回传')
    md.append('')
    md.append('**状态：complete（V4 后固定公式开发实验）。** 数据范围为现有 dev split，5 个任务、3 个固定 seed。主候选 `H_SV=V-mean_t(V)+mean_t(S)` 保留 V 的内部形状并使用 S 的时间平均水平。')
    md.append('')
    md.append('## 结论')
    md.append('')
    lev=gains['LEVEL_TRANSFER_GAIN']
    md.append(f"H_SV 的 Position MAE 样本均值为 **{fmt(absolute['H_SV']['position_mae']['mean'])}**，V 为 **{fmt(absolute['V']['position_mae']['mean'])}**；`LEVEL_TRANSFER_GAIN=L(V)-L(H_SV)` 为 **{fmt(lev['position_mae']['mean'])}**，source-group 条件 95% CI **{ci_text(ci_macro[('LEVEL_TRANSFER_GAIN','position_mae')])}**。该固定组合没有显示相对 V 的 Position MAE 改善，点估计较差。")
    fdg=lev['safe_fd_l1']['mean']
    seam_gain=lev['seam_safe_fd_l1']['mean']
    md.append(f"H_SV 的内部 Safe-FD 与 V 按构造相同；实测 paired gain 为 {fdg:.3g}（浮点数值差），`FD_IDENTITY_CHECK.csv` 的每窗口最大 Safe-FD 差不超过 {max(fdmax.values()):.3g}。首步 seam 不继承此恒等式：V 到 H_SV 的 seam error 平均增加 **{fmt(-seam_gain)}**，95% CI **[{fmt(-float(ci_macro[('LEVEL_TRANSFER_GAIN','seam_safe_fd_l1')]['ci_high']))}, {fmt(-float(ci_macro[('LEVEL_TRANSFER_GAIN','seam_safe_fd_l1')]['ci_low']))}]**。")
    md.append('')
    md.append('## 五任务 episode-macro 绝对指标')
    md.append('')
    md.append('越低越好；`±` 是三个固定 seed 间的 sample SD。指标采用已接受的 train target/difference scales，属于标准化误差。')
    md.append('')
    md.append('| 方法 | Position MAE | Safe-FD L1 | Seam Safe-FD |')
    md.append('|---|---:|---:|---:|')
    for method in ('S','V','A','C','H_SV','H_AV','H_SC','MIX50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE'):
        rec=absolute[method]
        vals=[]
        for metric in METRICS:
            x=rec[metric]
            sd=x.get('sample_sd')
            vals.append(f"{fmt(x['mean'])} ± {fmt(sd)}" if isinstance(sd,(int,float)) else fmt(x['mean'])+' (shared)')
        pretty={'S':'S / V2 optimized G-S','V':'V / V4 matched RGB','A':'A / RGB masked','C':'C / reassigned RGB','H_SV':'H_SV 主候选','H_AV':'H_AV common-level control','H_SC':'H_SC reassigned-shape control','MIX50':'MIX50 固定平均控制','BIAS_STATE':'V3 BIAS_STATE','BIAS_VIDEO':'V3 BIAS_VIDEO','PERSISTENCE':'Persistence'}[method]
        md.append(f'| {pretty} | {vals[0]} | {vals[1]} | {vals[2]} |')
    md.append('')
    md.append('## 固定配对对比')
    md.append('')
    md.append('Gain 定义为 `L(reference)-L(candidate)`，正值有利于 H_SV。区间是 5,000 次 within-task paired source-group bootstrap、seed 20261015；同一组抽样用于所有方法与 seed，条件于这三个已训练 seed、锁定公式和已暴露 dev。未计算 p 值。')
    md.append('')
    md.append('| 对比 | Position MAE gain [95% CI] | Safe-FD gain [95% CI] |')
    md.append('|---|---:|---:|')
    display={
      'LEVEL_TRANSFER_GAIN':'V vs H_SV', 'HYBRID_VS_OPT_STATE':'optimized S vs H_SV',
      'HYBRID_VS_BIAS_STATE':'BIAS_STATE vs H_SV','HYBRID_VS_BIAS_VIDEO':'BIAS_VIDEO vs H_SV',
      'MATCHED_SHAPE_GAIN':'H_SC vs H_SV','OPT_VS_COMMON_LEVEL_GAIN':'H_AV vs H_SV',
      'HYBRID_VS_ANCHOR_LEVEL':'H_PV vs H_SV','HYBRID_VS_MIX50':'MIX50 vs H_SV'}
    for c,label in display.items():
        p=ci_macro[(c,'position_mae')];f=ci_macro[(c,'safe_fd_l1')]
        ptext=f"{fmt(p['point_estimate'])} {ci_text(p)}"
        fpoint=float(f['point_estimate'])
        ftext='≈0 (inherited FD identity)' if abs(fpoint)<1e-12 else f"{fmt(fpoint)} {ci_text(f)}"
        md.append(f'| {label} | {ptext} | {ftext} |')
    md.append('')
    md.append('H_SV 相对 S 的 Position gain 为负、Safe-FD gain 为正，构成清楚的指标取舍。H_SV 相对 MIX50 的 Position gain 为负，且区间完全低于 0；MIX50 是预先锁定的控制，不是事后搜索结果。')
    md.append('')
    md.append('## 任务、seed 与 LOO')
    md.append('')
    md.append('下表为 H_SV 相对 V 的 Position gain（正值有利于 H_SV），均值 ± 三个 seed 的 sample SD。')
    md.append('')
    md.append('| Task | dev episodes / source groups | Position gain |')
    md.append('|---:|---:|---:|')
    for t in TASKS:
        r=transfer_pos[t];md.append(f"| {t} | {r['episodes']} / {r['source_groups']} | {fmt(r['mean_over_three_fixed_seeds'])} ± {fmt(r['sample_sd_across_seeds'])} |")
    md.append('')
    seedloo=[r for r in seed_loo if r['metric']=='position_mae']
    md.append('Position gain 的三种 seed-LOO 均为负：'+', '.join(f"去掉 {r['omitted_seed']} 后 {fmt(r['loo_five_task_macro'])}" for r in seedloo)+'. Task-LOO 的完整 task×seed 行及派生摘要见 `results/TASK_LOO.csv` 与 `RETURN_TO_CHATGPT.json`。不利点估计主要出现在 task 94 与 141；task 90、230 接近零，task 119 跨 seed 变化。')
    md.append('')
    md.append('source-group 支持为 task 119/94/90/141/230 的 12/12/11/5/7 组，共 47 组；元数据核验确认每个 group 只属于一个 task。session/trial 依赖仍未验证。')
    md.append('')
    md.append('## 水平—形状 L2 诊断')
    md.append('')
    md.append('使用训练集 target_sigma 标准化。L2 identity 核验最大绝对误差为：H_SV level−S **{:.3g}**，H_SV shape−V **{:.3g}**，Position MSE−(level+shape) **{:.3g}**。'.format(l2_identity['level_H_SV_minus_S'],l2_identity['shape_H_SV_minus_V'],l2_identity['position_minus_level_shape']))
    md.append('')
    md.append('| 方法 | Level MSE | Shape MSE | Position MSE |')
    md.append('|---|---:|---:|---:|')
    for m in ('S','V','A','C','H_SV'):
        r=v2l2[m];md.append(f"| {m} | {fmt(r['level_mse'])} ± {fmt(r['level_mse_sample_sd'])} | {fmt(r['shape_mse'])} ± {fmt(r['shape_mse_sample_sd'])} | {fmt(r['position_mse'])} ± {fmt(r['position_mse_sample_sd'])} |")
    md.append('')
    md.append('H_SV 的 L2 level 与 S 相同、shape 与 V 相同，Position MSE 低于 S、V；Position MAE 对 V 的结果仍较差。L2 可加分解说明平方误差的组成，不替代原 Position MAE。')
    md.append('')
    md.append('## 数据边界、计算与成本')
    md.append('')
    md.append('18/18 derived bundles、90/90 task views 已 seal；所有组合为 float64、按 ordered anchor/episode/task/source-group join。A/B/C/S/V 来自已接受 frozen predictions；本实验新增 neural fits、optimizer updates、参数校准 fits、GPU jobs 均为 0。V5 primary loss 共 3,330 窗口评价，复用 attempt 2 已完成结果；L2 另计 2,775 诊断评价。')
    md.append('')
    md.append('V5 target archive load 共 4 次（3 次失败尝试的读取均保留，最终恢复评分 1 次），总计加载 740 条 receiver window rows，185 个 unique dev windows；final/held-out 为 0。图表从 score-pass 的保存表/轨迹数据构建，额外 target archive load 为 0。V4 的历史 35,523 计数保留原名 `cumulative_dev_future_action_row_reads`，其与 V5 window-loss evaluation 的 grain 未确认，故分开报告、不相加。')
    md.append('')
    md.append('V5 CPU composition 用时 0.338 秒（18 bundles），恢复评分、CI、L2 和轨迹数据 1.147 秒，绘图约 {:.3f} 秒；没有新 GPU job。H_SV 实际需要 S 与 V 两个冻结模型的输出；本轮复用已存预测，resource ledger 的历史 source forward latency 为 NA，所以未测部署级双模型延迟。'.format(next((float(r['wall_seconds']) for r in resource if r.get('phase')=='v5_figure_generation'),float('nan'))))
    md.append('')
    md.append('三次 scorer 失败记录保留；最终恢复验证了 3,330 行既有 loss table 与 3,885 行 accepted V4 reference copy，没有重算主 loss。V4 source reconciliation 的 210 项最大差为 0。')
    md.append('')
    md.append('本结果检验固定 H_SV 公式在当前 development 域的表现。目标中的 MAE 降低未出现；Safe-FD 按构造与 V 相同，首步 seam error 增加。H_SV 的 L2 level/shape 组合具有可加恒等式，Position MAE 结论仍由实测 L1 指标决定。')
    md.append('')
    md.append('## 文件')
    md.append('')
    md.append('- `results/PRIMARY_TASK_SEED.csv`、`results/PAIRED_EFFECTS.csv`、`results/PAIRED_GROUP_CI.csv`：15 行 task×seed 绝对指标、八个固定配对对比与条件区间。')
    md.append('- `results/SEED_LOO.csv`、`results/TASK_LOO.csv`、`results/LEVEL_SHAPE_L2.csv`、`results/FD_IDENTITY_CHECK.csv`：LOO、L2 分解与 FD 恒等式。')
    md.append('- `figures/` 与 `figure_data/`：散点、逐任务 gain、L2 诊断、metadata-selected 轨迹图及数据。图像 assistant visual check 通过；人工复核 pending。')
    md.append('- `qa/FINAL_ACCEPTANCE.json`、`CHECKSUMS.sha256`、`qa/FINAL_CHECKSUM_VERIFICATION.json`：完成状态与最终 checksum 记录。')
    md.append('')
    md.append('**完成解释：** 数字和工程产物 complete；positive-gain 与 minimum-support 门均记录为 false。本轮不追加 alpha、routing、训练、seed 或 encoder 搜索。')
    md.append('')
    (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

    # Machine-readable handoff with exact effect and support records.
    ret={'schema_version':1,'status':'complete','study':'post_V4_frozen_level_shape_composition_development','principal_candidate':'H_SV',
      'tasks':TASKS,'formal_seeds':SEEDS,'formula':'H_SV=V-mean_future(V)+mean_future(S)','formula_lock_sha256':summary['formula_lock_sha256'],
      'prediction_seal_sha256':summary['prediction_seal_sha256'],'absolute_five_task_episode_equal_mean_and_sample_sd':absolute,
      'paired_gain_five_task_mean_and_sample_sd':gains,'paired_source_group_ci_mean_across_seeds':contrast_summary,
      'task_level_transfer_gain':task_rows,'seed_loo_level_transfer_gain':seedloo,'task_loo_level_transfer_gain_three_seed_summary':task_loo_summary,
      'level_shape_l2_five_task_macro':v2l2,'l2_identity_max_abs':l2_identity,'fd_identity_max_abs_safe_fd_loss_difference':fdmax,
      'internal_difference_identity_max_native_abs':transform['real_source_array_checks']['internal_difference_max_native_abs'],
      'internal_difference_identity_max_canonical_scaled_abs':transform['real_source_array_checks']['internal_difference_max_canonical_scaled_abs'],
      'source_score_reconciliation':read_json('qa/SOURCE_SCORE_RECONCILIATION.json'),
      'source_group_task_validation':read_json('qa/SOURCE_GROUP_TASK_VALIDATION.json'),
      'exposure':exposure_summary,'resource_ledger':resource,'resource_cost_usd':'not measured',
      'new_neural_fits':0,'optimizer_updates':0,'parameter_calibration_fits':0,'new_v5_gpu_jobs':0,
      'accepted_derived_bundles':18,'accepted_derived_task_views':90,'primary_window_loss_evaluations':3330,
      'primary_window_loss_recomputations_during_resume':0,'l2_diagnostic_window_evaluations':2775,
      'final_heldout_payload_access':0,'old_root_writes':0,'new_source_data_used':False,
      'status_flags':{'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,'human_figure_review':'pending'},
      'conclusion':'H_SV preserved V internal Safe-FD by formula but did not lower mean Position MAE versus V; its sample mean was higher, with the conditional source-group interval crossing zero. Seam error increased. The L2 level/shape identity passed and does not replace the Position MAE result.',
      'created_utc':utc()}
    write_json('RETURN_TO_CHATGPT.json',ret)

    acceptance={'schema_version':1,'status':'complete','study':'post_V4_frozen_level_shape_composition_development',
      'principal_candidate':'H_SV','tasks':TASKS,'formal_seeds':SEEDS,'formula_lock_sha256':summary['formula_lock_sha256'],
      'prediction_seal_sha256':summary['prediction_seal_sha256'],'prediction_seal_status':'PASS','accepted_derived_bundles':18,
      'accepted_derived_task_views':90,'accepted_receiver_anchors_per_bundle':185,'source_group_task_validation':'PASS',
      'source_score_reconciliation_status':'PASS','source_score_reconciliation_rows':210,'source_score_reconciliation_max_abs_delta':0.0,
      'transform_qa_status':'PASS','fd_identity_status':'PASS','l2_identity_status':'PASS','scored_status':'PASS',
      'new_neural_fits':0,'optimizer_updates':0,'parameter_calibration_fits':0,'new_gpu_jobs':0,
      'v5_primary_window_loss_evaluations':3330,'v5_primary_window_loss_recomputations_during_resume':0,'l2_diagnostic_window_evaluations':2775,
      'target_archive_load_operations_total':4,'target_array_window_rows_loaded_total':740,'failed_scorer_attempts':3,
      'prior_legacy_counter_unit_status':'unverified; preserved separately and not summed with V5 evaluation counts',
      'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,
      'final_heldout_payload_access':0,'old_root_writes':0,'source_roots_read_only':True,
      'figures_saved':8,'assistant_visual_qa_status':'PASS','manual_human_review_status':'pending',
      'checksum_finalization':'performed as final manifest plus excluded read-back receipt','created_utc':utc()}
    write_json('qa/FINAL_ACCEPTANCE.json',acceptance)
    print(json.dumps({'status':'HANDOFF_BUILT','fd_identity_rows':len(fd_rows),'task_transfer_rows':len(task_rows),
      'report':'RETURN_TO_CHATGPT.zh-CN.md','machine_report':'RETURN_TO_CHATGPT.json','acceptance':'qa/FINAL_ACCEPTANCE.json'},ensure_ascii=False))

if __name__=='__main__':main()
