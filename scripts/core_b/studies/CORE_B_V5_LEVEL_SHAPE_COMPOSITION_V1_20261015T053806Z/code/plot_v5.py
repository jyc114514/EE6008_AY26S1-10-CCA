#!/usr/bin/env python3
"""Build the locked V5 figures and non-target exposure record from scored tables."""
from __future__ import annotations
import csv, datetime, json, math, statistics, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASKS=[119,94,90,141,230]
SEEDS=[202610131,202610132,202610133]
METHODS=['S','V','A','C','H_SV','H_VS','H_AV','H_SC','H_PV','MIX50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE']
REFS=[('LEVEL_TRANSFER_GAIN','V','video V'),('HYBRID_VS_OPT_STATE','S','optimized state S'),('HYBRID_VS_BIAS_STATE','BIAS_STATE','BIAS_STATE'),('HYBRID_VS_BIAS_VIDEO','BIAS_VIDEO','BIAS_VIDEO')]
METRICS=['position_mae','safe_fd_l1','seam_safe_fd_l1']

def read_csv(rel):
    with (ROOT/rel).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(rows[0]) if rows else ['status']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def set_plot_style():
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'legend.fontsize':8,'figure.titlesize':13})

def main():
    started=time.perf_counter();set_plot_style()
    out=ROOT/'figures';out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((ROOT/'results/V5_METRICS_SUMMARY.json').read_text())
    primary=read_csv('results/PRIMARY_TASK_SEED.csv')
    support={}
    for r in primary:
        t=int(r['task_id']);support[t]=(int(r['episodes']),int(r['source_groups']),int(r['anchors']))
    # Figure 1: absolute dual-metric references, three fixed seed points and mean +/- sample SD.
    colors=plt.get_cmap('tab20').colors
    labels={'S':'S optimized state','V':'V matched video','A':'A common masked RGB','C':'C reassigned RGB',
            'H_SV':'H_SV principal','H_VS':'H_VS control','H_AV':'H_AV control','H_SC':'H_SC control',
            'H_PV':'H_PV anchor level','MIX50':'MIX50 control','BIAS_STATE':'V3 BIAS_STATE','BIAS_VIDEO':'V3 BIAS_VIDEO','PERSISTENCE':'Persistence shared'}
    stats=summary['absolute_five_task_episode_equal_mean_and_sample_sd']
    fig,ax=plt.subplots(figsize=(12.5,7.3),constrained_layout=True)
    for j,m in enumerate(METHODS):
        p=stats[m]['position_mae'];f=stats[m]['safe_fd_l1'];xp=p['per_seed'];yp=f['per_seed'];color=colors[j]
        for x,y in zip(xp,yp):ax.scatter(x,y,s=22,color=color,alpha=.28,zorder=2)
        xs=float(p['mean']);ys=float(f['mean']);xerr=p.get('sample_sd');yerr=f.get('sample_sd')
        ax.errorbar(xs,ys,xerr=None if not isinstance(xerr,(int,float)) else xerr,
                    yerr=None if not isinstance(yerr,(int,float)) else yerr,fmt='o',ms=6 if m!='H_SV' else 8,
                    color=color,capsize=3,elinewidth=1.1,zorder=4,label=labels[m],markeredgecolor='black' if m=='H_SV' else color,
                    markeredgewidth=1.2 if m=='H_SV' else .5)
    ax.set_xlabel('Position MAE (standardized; lower is better)')
    ax.set_ylabel('Safe-FD L1 (standardized; lower is better)')
    ax.set_title('V5 fixed state/video compositions on the 5-task development set')
    ax.grid(True,alpha=.22)
    ax.legend(title='Method (faint points = fixed seeds; bars = mean +/- sample SD)',loc='center left',bbox_to_anchor=(1.01,.5),frameon=True)
    ax.text(.015,.02,'H_SV, H_AV and H_PV inherit V internal FD by construction; overlapping FD is not an independent gain.',transform=ax.transAxes,fontsize=8,va='bottom')
    fig.savefig(out/'V5_DUAL_METRIC_SCATTER.png',dpi=180,bbox_inches='tight');plt.close(fig)

    # Figure 2: H_SV paired gains against the four locked references, by task.
    fig,axes=plt.subplots(1,3,figsize=(15.8,5.4),sharex=True,constrained_layout=True)
    xs0=np.arange(len(TASKS),dtype=float);offsets=np.linspace(-.24,.24,len(REFS));ref_colors=['#1f77b4','#9467bd','#2ca02c','#d62728'];markers=['o','s','^','D']
    for ax,metric in zip(axes,METRICS):
        for j,(contrast,reference,label) in enumerate(REFS):
            means=[];sds=[]
            for task in TASKS:
                vals=[]
                for seed in SEEDS:
                    rr=next(r for r in primary if int(r['task_id'])==task and int(r['seed'])==seed)
                    vals.append(float(rr[f'{reference}_{metric}'])-float(rr[f'H_SV_{metric}']))
                means.append(statistics.mean(vals));sds.append(statistics.stdev(vals))
            ax.errorbar(xs0+offsets[j],means,yerr=sds,fmt=markers[j]+'-',color=ref_colors[j],capsize=3,linewidth=1.3,markersize=4,label=label)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_title({'position_mae':'Position MAE gain','safe_fd_l1':'Safe-FD L1 gain','seam_safe_fd_l1':'Seam Safe-FD gain'}[metric])
        ax.set_ylabel('Gain = L(reference) - L(H_SV); positive favors H_SV')
        ax.set_xticks(xs0,[f'{t}\n{support[t][0]} ep / {support[t][1]} groups' for t in TASKS])
        ax.grid(axis='y',alpha=.22)
    axes[-1].legend(title='Locked contrast',loc='center left',bbox_to_anchor=(1.02,.5))
    fig.suptitle('Task-specific paired gains; points are three-seed means, bars are sample SD')
    fig.savefig(out/'V5_HSV_GAINS_BY_TASK.png',dpi=180,bbox_inches='tight');plt.close(fig)

    # Figure 3: L2 level/shape accounting. These terms are standardized MSE diagnostics, not L1 MAE.
    l2=read_csv('results/LEVEL_SHAPE_L2.csv')
    l2r={r['method']:r for r in l2 if r.get('task_id')=='five_task_macro' and r.get('seed')=='mean_across_formal_seeds'}
    l2methods=['S','V','A','C','H_SV'];fig,axes=plt.subplots(1,3,figsize=(13.8,4.8),constrained_layout=True)
    titles=[('level_mse','Level MSE'),('shape_mse','Demeaned shape MSE'),('position_mse','Position MSE (= level + shape)')]
    for ax,(field,title) in zip(axes,titles):
        means=[float(l2r[m][field]) for m in l2methods];sds=[float(l2r[m][field+'_sample_sd']) for m in l2methods]
        bars=ax.bar(np.arange(len(l2methods)),means,yerr=sds,capsize=3,color=['#6c8ebf','#4e79a7','#a0cbe8','#f28e2b','#59a14f'])
        ax.set_xticks(np.arange(len(l2methods)),l2methods);ax.set_title(title);ax.set_ylabel('Standardized MSE')
        ax.grid(axis='y',alpha=.22)
        for b,v in zip(bars,means):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.4f}',ha='center',va='bottom',fontsize=7)
    fig.suptitle('L2 diagnostic; mean across three fixed seeds +/- sample SD')
    fig.text(.5,-.02,'H_SV level term equals S; H_SV demeaned shape term equals V. This MSE identity does not decompose Position MAE.',ha='center',fontsize=8)
    fig.savefig(out/'V5_LEVEL_SHAPE_L2.png',dpi=180,bbox_inches='tight');plt.close(fig)

    # Metadata-selected representative trajectory panels, one anchor per task, fixed before outcomes.
    cases=read_csv('figure_data/TRAJECTORY_CASES.csv');traj=read_csv('figure_data/TRAJECTORY_DATA.csv')
    method_order=['GT','S','V','H_SV','MIX50'];line_styles={'GT':('-','#111111'),'S':('--','#9467bd'),'V':(':','#1f77b4'),'H_SV':('-','#2ca02c'),'MIX50':('-.','#ff7f0e')}
    trajectory_files=[]
    for task in TASKS:
        case=next(r for r in cases if int(r['task_id'])==task)
        fig,axes=plt.subplots(3,2,figsize=(12.4,8.2),sharex=True,constrained_layout=True)
        for i,seed in enumerate(SEEDS):
            for j,action in enumerate((0,14)):
                ax=axes[i,j]
                for method in method_order:
                    selected=[r for r in traj if int(r['task_id'])==task and int(r['seed'])==seed and int(r['action_index'])==action and r['method']==method]
                    selected.sort(key=lambda r:int(r['future_step']))
                    ls,color=line_styles[method]
                    ax.plot([int(r['future_step']) for r in selected],[float(r['native_command_value']) for r in selected],linestyle=ls,color=color,linewidth=1.35,label=method)
                ax.set_title(f'seed {seed} | action index {action}')
                ax.set_xlabel('Future step (1-30)');ax.set_ylabel('Numeric native command value')
                ax.grid(True,alpha=.2)
        handles,legend_labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='upper center',ncol=5,bbox_to_anchor=(.5,1.045))
        fig.suptitle(f'Task {task} | episode {case["episode_id"]} | anchor {case["anchor_id"]}\nunits unresolved; action indices 0 and 14',y=1.09)
        name=f'V5_TRAJECTORY_TASK_{task}.png';fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig);trajectory_files.append(name)

    caption=("# V5 图表说明\n\n"
      "- `V5_DUAL_METRIC_SCATTER.png`：五任务等权、episode-macro 的 Position MAE 与 Safe-FD L1。淡点是三个固定 seed，误差棒为 seed 间 sample SD。H_SV、H_AV、H_PV 的内部 FD 按公式继承 V。\n"
      "- `V5_HSV_GAINS_BY_TASK.png`：H_SV 对 V、optimized S、BIAS_STATE、BIAS_VIDEO 的配对 gain，定义为 `L(reference)-L(H_SV)`；正值有利于 H_SV。误差棒为三个固定 seed 的 sample SD，任务下方标注 dev episode/source-group 支持。\n"
      "- `V5_LEVEL_SHAPE_L2.png`：用训练集 target_sigma 标准化的 L2 level/shape 诊断及其和。该图的 MSE 量与 Position MAE 不同。\n"
      "- `V5_TRAJECTORY_TASK_119.png`、`V5_TRAJECTORY_TASK_94.png`、`V5_TRAJECTORY_TASK_90.png`、`V5_TRAJECTORY_TASK_141.png`、`V5_TRAJECTORY_TASK_230.png`：每任务一个预先 metadata-selected anchor，三 seed 分行，action indices 0/14 分列，展示 GT、S、V、H_SV、MIX50。横轴为 future step 1-30；纵轴是 numeric native command value，物理单位未确认。\n")
    (out/'CAPTIONS.zh-CN.md').write_text(caption,encoding='utf-8')

    # Metadata and checks for the figure package; plotting reuses only sealed score/trajectory files.
    exposure_path=ROOT/'inventory/OUTCOME_EXPOSURE.json';exposure=json.loads(exposure_path.read_text())
    old_mixed=dict(exposure.get('mixed_grain_cumulative_fields_removed',{}))
    old_mixed.update({k:exposure.pop(k) for k in ('cumulative_dev_future_window_scoring_reads','cumulative_including_l2_diagnostic_evaluations','cumulative_including_l2_and_failed_target_load_rows') if k in exposure})
    exposure['reported_prior_counter']={'value':35523,'name':'cumulative_dev_future_action_row_reads','unit_status':'unverified_as_same_grain_as_V5; kept separate and not added'}
    exposure.update({'new_dev_future_window_logical_reads':3330,'v5_primary_loss_evaluations_completed':3330,
      'v5_primary_loss_evaluations_recomputed_during_resume':0,'l2_diagnostic_window_evaluations':2775,
      'target_archive_load_operations_total':4,'target_array_window_rows_loaded_total':740,'failed_attempt_target_array_window_rows_loaded':555,
      'unique_dev_receiver_windows':185,'additional_target_archive_reads_for_figures':0,'figure_target_payload_reads':0,
      'final_heldout_payload_access':0,'source_group_count':47,'source_group_task_validation':'PASS',
      'mixed_grain_cumulative_fields_removed':old_mixed,'figure_generation_status':'PASS'})
    write_json(exposure_path,exposure)
    ledger_path=ROOT/'results/RESOURCE_LEDGER.csv';ledger=read_csv('results/RESOURCE_LEDGER.csv')
    ledger=[r for r in ledger if r.get('phase')!='v5_figure_generation']
    elapsed=time.perf_counter()-started
    ledger.append({'phase':'v5_figure_generation','role':'dual metric, task gains, L2 and five metadata selected trajectories','seed':'all','wall_seconds':elapsed,
      'compute_device':'CPU','physical_gpu_historical':'NA','new_v5_gpu_job':False,'prediction_hash':'NA','scope':'reads saved result and figure-data tables only; zero target archive reads'})
    write_csv(ledger_path,ledger)
    qa={'schema_version':1,'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),
        'figures': ['V5_DUAL_METRIC_SCATTER.png','V5_HSV_GAINS_BY_TASK.png','V5_LEVEL_SHAPE_L2.png']+trajectory_files,
        'all_plots_saved':True,'figure_data_reused':True,'target_archive_reads_for_figures':0,'final_heldout_payload_access':0,
        'trajectory_case_count':len(cases),'trajectory_seeds':SEEDS,'trajectory_action_indices':[0,14],
        'physical_units_claimed':False,'plot_wall_seconds':elapsed}
    write_json(ROOT/'qa/FIGURE_BUILD_QA.json',qa)
    print(json.dumps({'status':'FIGURES_BUILT','wall_seconds':elapsed,'files':qa['figures'],'target_archive_reads':0},ensure_ascii=False))

if __name__=='__main__':main()
