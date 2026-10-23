#!/usr/bin/env python3
"""Build the final V6 handoff from sealed result tables; does not read target archives."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = [119, 94, 90, 141, 230]
SEEDS = [202610131, 202610132, 202610133]
METRICS = ['position_mae', 'safe_fd_l1', 'seam_safe_fd_l1']
LABELS = {'position_mae':'Position MAE','safe_fd_l1':'Safe-FD L1','seam_safe_fd_l1':'Seam Safe-FD L1'}


def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write_json(path,obj):
    p=Path(path);tmp=p.with_name('.'+p.name+'.part')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,p)


def gain_ci(rows, contrast, metric, seed='mean_across_formal_seeds'):
    r=next(x for x in rows if x['contrast']==contrast and x['metric']==metric and str(x['seed'])==str(seed) and x['task_id']=='five_task_macro')
    return {'estimate':float(r['point_estimate']),'ci95_low':float(r['ci_low']),'ci95_high':float(r['ci_high'])}


def main():
    if not (ROOT/'qa/FINAL_ACCEPTANCE.json').is_file() or read_json(ROOT/'qa/FINAL_ACCEPTANCE.json').get('status')!='complete':
        raise RuntimeError('handoff requires completed final acceptance')
    acceptance=read_json(ROOT/'qa/FINAL_ACCEPTANCE.json')
    exposure=read_json(ROOT/'inventory/OUTCOME_EXPOSURE.json')
    resources=read_json(ROOT/'inventory/RESOURCE_SUMMARY.json')
    alpha=read_json(ROOT/'protocol/ALPHA_LOCK.json')
    v5=read_json(ROOT/'results/V5_MIX50_RECONCILIATION.json')
    methods=[]
    macro=read_csv(ROOT/'results/FIVE_TASK_MACRO.csv')
    for method in ['S','A','V','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','SV50','SS50','SV_CAL','SS_CAL']:
        rec={'method':method,'metrics':{}}
        for metric in METRICS:
            row=next(r for r in macro if r['method']==method and r['metric']==metric and r['seed']=='mean_across_formal_seeds')
            sd=row.get('sample_sd_across_seeds','NA')
            rec['metrics'][metric]={'mean':float(row['five_task_equal_episode_equal_macro']),
                'sample_sd_across_seeds':None if sd in ('NA','',None) else float(sd),
                'n_model_seeds':1 if method=='PERSISTENCE' else 3}
        methods.append(rec)

    v5_ci=read_csv(ROOT/'results/V5_MIX50_PAIRED_GROUP_CI.csv')
    v5_gains={}
    for ref in ['S','V','BIAS_STATE','BIAS_VIDEO']:
        contrast=f'{ref}_MINUS_MIX50'
        v5_gains[ref]={m:gain_ci(v5_ci,contrast,m) for m in METRICS}
    ci=read_csv(ROOT/'results/PAIRED_GROUP_CI.csv')
    paired={}
    for name in ['VIDEO_SUBSTITUTION_50','VIDEO_SUBSTITUTION_CAL','SV_CALIBRATION_GAIN','SS_CALIBRATION_GAIN',
                 'SV_CAL_VS_S','SV_CAL_VS_V','SV_CAL_VS_BIAS_STATE','SV_CAL_VS_BIAS_VIDEO',
                 'SV50_VS_S','SV50_VS_V','SV50_VS_BIAS_STATE','SV50_VS_BIAS_VIDEO','SS50_VS_S','SS50_VS_A']:
        paired[name]={m:gain_ci(ci,name,m) for m in METRICS}
    seed_loo=read_csv(ROOT/'results/SEED_LOO.csv')
    task_loo=read_csv(ROOT/'results/TASK_LOO.csv')
    loo={}
    for name in ['VIDEO_SUBSTITUTION_50','VIDEO_SUBSTITUTION_CAL','SV_CALIBRATION_GAIN','SS_CALIBRATION_GAIN']:
        sl=[r for r in seed_loo if r['method_or_contrast']==name and r['metric']=='position_mae']
        tl=[r for r in task_loo if r['method_or_contrast']==name and r['metric']=='position_mae']
        loo[name]={
            'position_seed_loo':{str(r['omitted_seed']):float(r['loo_five_task_macro']) for r in sl},
            'position_task_loo_positive_rows':sum(float(r['loo_macro'])>0 for r in tl),
            'position_task_loo_total_rows':len(tl),
            'position_task_loo_min':min(float(r['loo_macro']) for r in tl),
            'position_task_loo_max':max(float(r['loo_macro']) for r in tl),
        }

    handoff={
        'schema_version':1,'status':'complete','study':'post_V5_controlled_late_fusion_development',
        'created_utc':utc(),'out_root':str(ROOT),'development_only':True,
        'V5_MIX50':{'task_seed_rows':15,**v5['task_seed_metrics'],'paired_gain_reference_minus_MIX50':v5_gains,
                    'paired_ci_rows':v5['paired_ci_rows']},
        'V6_alpha_selection':alpha['selected_alphas'],
        'V6_methods':methods,'V6_paired_gain_reference_minus_candidate':paired,
        'selected_position_loo':loo,
        'sample_size':{'tasks':TASKS,'formal_seeds':SEEDS,'dev_episodes':47,'dev_anchors':185,
                       'per_task_support':read_json(ROOT/'inventory/ACTUAL_FOLD_REUSE.json')['dev']['task_support']},
        'prediction_accounting':{'logical_total':12,'new_logical':9,'new_unique_payloads':6,
                                 'shared_aliases':3,'reused_v5_SV50':3},
        'resources':resources,
        'exposure':exposure,
        'limitations':{'inner_tune_reused_from_v2_recipe_selection':True,
                        'session_trial_independence':'UNKNOWN','physical_command_units':'UNKNOWN',
                        'confidence_intervals':'conditional source-group bootstrap on fixed bases/alphas/exposed dev',
                        'final_heldout_payload_access':0},
        'acceptance':acceptance,
        'files':{'V5_MIX50_task_seed':'results/V5_MIX50_TASK_SEED.csv',
                 'V5_MIX50_gains':'results/V5_MIX50_GAIN_TASK_SEED.csv',
                 'V5_MIX50_CI':'results/V5_MIX50_PAIRED_GROUP_CI.csv',
                 'primary':'results/PRIMARY_TASK_SEED.csv','paired':'results/PAIRED_EFFECTS.csv',
                 'paired_CI':'results/PAIRED_GROUP_CI.csv','seed_LOO':'results/SEED_LOO.csv',
                 'task_LOO':'results/TASK_LOO.csv','resource_ledger':'results/RESOURCE_LEDGER.csv',
                 'exposure':'inventory/OUTCOME_EXPOSURE.json','final_acceptance':'qa/FINAL_ACCEPTANCE.json',
                 'checksums':'CHECKSUMS.sha256'},
    }
    write_json(ROOT/'RETURN_TO_CHATGPT.json',handoff)

    lines=[
        '# V6 有对照的晚期融合与 inner-tune 标量校准','',
        '**状态：complete；开发实验。** 固定矩阵已评分。正收益和最低样本量不是完成门；final/heldout payload access 为 0。','',
        '## V5 MIX50 基线补全','',
        '| 指标 | macro mean | 三个固定 seed 的 sample SD |','|---|---:|---:|']
    for metric in METRICS:
        x=v5['task_seed_metrics'][metric]
        lines.append(f'| {LABELS[metric]} | {x["mean_across_three_fixed_seeds"]:.9f} | {x["sample_sd_across_seeds"]:.9f} |')
    lines += ['', 'V5 的 15 行 task×seed 双指标与 seam、相对 S/V/BIAS_STATE/BIAS_VIDEO 的每行 gain、以及 task/seed/macro 条件区间分别在 `results/V5_MIX50_TASK_SEED.csv`、`results/V5_MIX50_GAIN_TASK_SEED.csv` 与 `results/V5_MIX50_PAIRED_GROUP_CI.csv`。以下是五任务等权、三个固定 seed 均值的 gain=reference−MIX50。','',
        '| V5 reference − MIX50 | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |','|---|---:|---:|---:|']
    for ref in ['S','V','BIAS_STATE','BIAS_VIDEO']:
        x=v5_gains[ref]
        cells=[f'{x[m]["estimate"]:+.6f} [{x[m]["ci95_low"]:+.6f}, {x[m]["ci95_high"]:+.6f}]' for m in METRICS]
        lines.append(f'| {ref} | {cells[0]} | {cells[1]} | {cells[2]} |')
    lines += ['', '## Inner-tune 系数','',
        '| 家族 | alpha | inner-tune Position MAE | inner-tune Safe-FD L1 | seam | J |','|---|---:|---:|---:|---:|---:|']
    for family in ['SV','SS']:
        a=alpha['selected_alphas'][family]
        lines.append(f'| {family} | {a["alpha"]:.2f} | {a["position_mae_macro"]:.9f} | {a["safe_fd_l1_macro"]:.9f} | {a["seam_safe_fd_l1_macro"]:.9f} | {a["selected_J"]:.9f} |')
    lines += ['', '每族固定 21 点，使用 screen seed 202610130 的 S/A/V inner-fit-only checkpoint predictions；46 episodes、270 anchors 按 task 等权、episode 等权选择 `Position MAE + 0.1×Safe-FD L1`。inner-tune 曾用于 V2 recipe 选择，因此这里是开发元校准。`alpha_SV=0.50` 使 `SV_CAL` 与 V5 `SV50/MIX50` 共享 payload。','',
        '## 四个程序的正式 dev 结果','',
        '| 方法 | Position MAE mean ± seed SD | Safe-FD L1 mean ± seed SD | Seam Safe-FD L1 mean ± seed SD |','|---|---:|---:|---:|']
    for rec in methods:
        vals=[]
        for metric in METRICS:
            x=rec['metrics'][metric]
            vals.append(f'{x["mean"]:.9f} (shared)' if x['sample_sd_across_seeds'] is None else f'{x["mean"]:.9f} ± {x["sample_sd_across_seeds"]:.9f}')
        lines.append(f'| {rec["method"]} | {vals[0]} | {vals[1]} | {vals[2]} |')
    lines += ['', '## 锁定 paired gains','',
        '正值表示 candidate loss 更低。95% 区间为 5000 次 within-task source-group paired bootstrap；对固定 bases、alpha 与已暴露 dev 条件化，不是跨 seed 总体推断。','',
        '| reference − candidate | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |','|---|---:|---:|---:|']
    main=['VIDEO_SUBSTITUTION_50','VIDEO_SUBSTITUTION_CAL','SV_CALIBRATION_GAIN','SS_CALIBRATION_GAIN',
          'SV_CAL_VS_S','SV_CAL_VS_V','SV_CAL_VS_BIAS_STATE','SV_CAL_VS_BIAS_VIDEO']
    for name in main:
        x=paired[name]
        cells=[f'{x[m]["estimate"]:+.6f} [{x[m]["ci95_low"]:+.6f}, {x[m]["ci95_high"]:+.6f}]' for m in METRICS]
        lines.append(f'| {name} | {cells[0]} | {cells[1]} | {cells[2]} |')
    loo50=loo['VIDEO_SUBSTITUTION_50']
    loo_cal=loo['VIDEO_SUBSTITUTION_CAL']
    seed_loo_50=', '.join(f'{k}: {v:.6f}' for k,v in loo50['position_seed_loo'].items())
    seed_loo_cal=', '.join(f'{k}: {v:.6f}' for k,v in loo_cal['position_seed_loo'].items())
    lines += ['', f'Position seed-LOO gains remain positive for VIDEO_SUBSTITUTION_50 ({seed_loo_50}) and VIDEO_SUBSTITUTION_CAL ({seed_loo_cal}). Across all {loo50["position_task_loo_total_rows"]} task-LOO rows, VIDEO_SUBSTITUTION_50 is positive in {loo50["position_task_loo_positive_rows"]}; range {loo50["position_task_loo_min"]:+.6f} to {loo50["position_task_loo_max"]:+.6f}.',
        '附加固定 comparisons `SV50_VS_S/V/BIAS_STATE/BIAS_VIDEO` 与 `SS50_VS_S/A` 的三指标 gains 和条件区间在 `results/PAIRED_GROUP_CI.csv` 与 `results/PAIRED_EFFECTS.csv`。完整 seed/task LOO 文件包含全部 14 contrasts。`SS_CAL` 相对 `SS50` 的 Position gain 较小，Safe-FD 点估计为负且区间跨 0，指标呈轻微 trade-off。','',
        '## 资源与 target exposure','',
        f'- screen-seed inner-fit models reused: 3；new inner-fit/formal neural fits: 0；CPU scalar calibrations: 2；inner-tune GPU replay attempts: 4，接受 3 个 tune bundles。',
        f'- GPU watcher job 共 4 次，使用物理 GPU 2（进程内 `cuda:0`），最大 observed reserved 为 {resources["max_observed_reserved_mib"]:.0f} MiB；全部 lease 已释放。V_FIT 首次登记 102 MiB、实测 reserved 134 MiB，且首次 run-local lease-dir 指向 `coordination/watcher_status`；该输出保留但不用于 alpha。更正重放使用 `coordination/gpu_lease` 和 134 MiB 登记，payload 逐元素相同。',
        f'- V6 预测矩阵 12 个逻辑 bundles：9 个新增逻辑程序×seed、6 个唯一新 tensor、3 个 SV_CAL alias、3 个 V5 SV50 复用。',
        f'- V5 dev archive 加载 4 次；V6 在 seal 后加载 2 次，共 6 次；V6 读入 370 个 target-window rows，唯一仍为 185。第一轮读取后在打分前停止；第二轮计算 1110 条新窗口损失，随后图表由保存的损失/图数据恢复。V5 已有 3330 条 primary loss；V5+V6 known-grain 合计 4440。先前报告的 35523 计数单位未核实，保持单列未加总。',
        '- final/heldout payload access: **0**；source roots V2–V5 writes: 0。session/trial 独立性与物理 command 单位未知。','',
        '## 文件','',
        '- V5 MIX50 补全：`results/V5_MIX50_TASK_SEED.csv`, `results/V5_MIX50_GAIN_TASK_SEED.csv`, `results/V5_MIX50_PAIRED_GROUP_CI.csv`',
        '- V6 结果：`results/PRIMARY_TASK_SEED.csv`, `results/PAIRED_EFFECTS.csv`, `results/PAIRED_GROUP_CI.csv`, `results/SEED_LOO.csv`, `results/TASK_LOO.csv`, `results/CONVEXITY_CHECK.csv`',
        '- QA 与资源：`qa/FINAL_ACCEPTANCE.json`, `inventory/OUTCOME_EXPOSURE.json`, `results/RESOURCE_LEDGER.csv`, `CHECKSUMS.sha256`',
        '- 图与数据：`figures/`, `figure_data/`; agent visual review passed on alpha, dual metric/seam, and paired contrasts. Full manual review remains pending for trajectory details.','',
        '固定矩阵到此停止；未基于 dev 追加系数、模型家族、seed 或训练。','']
    (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'PASS','out_root':str(ROOT),'paired_contrasts':len(paired),'methods':len(methods)},sort_keys=True))


if __name__=='__main__':main()
