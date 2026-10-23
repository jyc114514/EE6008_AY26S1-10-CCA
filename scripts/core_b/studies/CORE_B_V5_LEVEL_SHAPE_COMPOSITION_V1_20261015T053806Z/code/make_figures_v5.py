#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
TASKS=[119,94,90,141,230]
SEEDS=['202610131','202610132','202610133']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#F0E442','#000000','#6A3D9A','#999999','#8C564B','#17BECF','#4D4D4D']
PALETTE=dict(zip(['S','V','A','C','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','H_SV','H_VS','H_AV','H_SC','H_PV','MIX50'],COLORS))
CONTRASTS={
 'LEVEL_TRANSFER_GAIN':('V - H_SV','#0072B2'),
 'HYBRID_VS_OPT_STATE':('S - H_SV','#D55E00'),
 'HYBRID_VS_BIAS_STATE':('BIAS_STATE - H_SV','#009E73'),
 'HYBRID_VS_BIAS_VIDEO':('BIAS_VIDEO - H_SV','#CC79A7'),
}
SEED_MARKERS={'202610131':'o','202610132':'s','202610133':'^'}

def read_csv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def write_json(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def main():
 summary=json.loads((ROOT/'results/V5_METRICS_SUMMARY.json').read_text())
 supports={int(x['task_id']):x for x in json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())['fold_membership']['task_support']}
 # Figure 1: mean and sample SD across the three fixed model seeds, not confidence intervals.
 fig,ax=plt.subplots(figsize=(11.5,7.0),layout='constrained')
 methods=['S','V','A','C','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','H_SV','H_VS','H_AV','H_SC','H_PV','MIX50']
 offsets={m:i for i,m in enumerate(methods)}
 dual=[]
 for m in methods:
  d=summary['absolute_five_task_episode_equal_mean_and_sample_sd'][m]
  x=float(d['position_mae']['mean']);y=float(d['safe_fd_l1']['mean'])
  sx=d['position_mae']['sample_sd'];sy=d['safe_fd_l1']['sample_sd']
  xerr=None if sx is None else float(sx);yerr=None if sy is None else float(sy)
  ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='none',ecolor=PALETTE[m],elinewidth=1.1,capsize=2,zorder=1,alpha=.85)
  marker='D' if m=='PERSISTENCE' else ('*' if m=='H_SV' else 'o')
  ax.scatter([x],[y],s=90 if m=='H_SV' else 52,marker=marker,color=PALETTE[m],edgecolor='black' if m in ['H_SV','PERSISTENCE'] else 'none',linewidth=.6,zorder=3,label=m)
  dual.append({'method':m,'position_mae_mean':x,'position_mae_sample_sd':sx,'safe_fd_l1_mean':y,'safe_fd_l1_sample_sd':sy,'n_formal_seeds':1 if sx is None else 3})
 ax.set_xlabel('Position MAE (train-scale standardized)')
 ax.set_ylabel('Safe-FD L1 (train-scale standardized)')
 ax.set_title('Frozen level-shape combinations: position and internal FD')
 ax.grid(True,alpha=.22)
 ax.legend(ncol=3,frameon=False,loc='upper left',fontsize=9)
 ax.text(.99,.02,'Whiskers: sample SD across three fixed seeds; Persistence is shared.\nFD overlap is inherited: H_SV/H_AV/H_PV = V; H_VS = S; H_SC = C.',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=9,bbox={'facecolor':'white','alpha':.88,'edgecolor':'#cccccc'})
 fig1=ROOT/'figures/V5_DUAL_METRIC.png';fig.savefig(fig1,dpi=180,facecolor='white');plt.close(fig)
 write_csv(ROOT/'figure_data/V5_DUAL_METRIC.csv',dual)

 # Figure 2: paired gains for four requested source comparisons, each fixed seed as a separate marker.
 pairs=read_csv(ROOT/'results/PAIRED_EFFECTS.csv')
 fig,axes=plt.subplots(3,1,figsize=(13.5,10.5),sharey=True,layout='constrained')
 metric_titles=[('position_mae','Position MAE gain'),('safe_fd_l1','Safe-FD gain'),('seam_safe_fd_l1','Seam Safe-FD gain')]
 plotrows=[]
 for ax,(metric,title) in zip(axes,metric_titles):
  for ti,task in enumerate(TASKS):
   for ci,(contrast,(label,color)) in enumerate(CONTRASTS.items()):
    baseline=ti+(ci-1.5)*.19
    for si,seed in enumerate(SEEDS):
     row=next(x for x in pairs if x['contrast']==contrast and int(x['task_id'])==task and str(x['seed'])==seed)
     value=float(row[metric]);y=baseline+(si-1)*.045
     ax.scatter(value,y,color=color,marker=SEED_MARKERS[seed],s=40,edgecolor='white',linewidth=.35,zorder=3)
     plotrows.append({'metric':metric,'task_id':task,'seed':seed,'contrast':contrast,'gain':value,'support_episodes':row['episodes'],'support_anchors':row['anchors'],'support_groups':row['source_groups']})
  ax.axvline(0,color='#333333',linewidth=1.0)
  ax.grid(axis='x',alpha=.22)
  ax.set_title(title,loc='left',fontsize=11)
  ax.set_yticks(range(len(TASKS)))
  ax.set_yticklabels([f'{t}  ({supports[t]["episodes"]} episodes, {supports[t]["anchors"]} anchors)' for t in TASKS])
  ax.invert_yaxis()
  ax.set_ylabel('Task and dev support')
  ax.set_xlabel('Gain = L(reference) − L(H_SV); positive favors H_SV')
 contrast_handles=[plt.Line2D([0],[0],marker='o',linestyle='none',color=c,markersize=7,label=l) for l,c in [v for v in CONTRASTS.values()]]
 seed_handles=[plt.Line2D([0],[0],marker=SEED_MARKERS[s],linestyle='none',color='#333333',markersize=7,label=f'seed {s}') for s in SEEDS]
 fig.legend(handles=contrast_handles+seed_handles,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
 fig.suptitle('Paired task-level gains across fixed seeds',fontsize=15)
 fig2=ROOT/'figures/V5_PAIRED_GAINS_BY_TASK.png';fig.savefig(fig2,dpi=180,facecolor='white');plt.close(fig)
 write_csv(ROOT/'figure_data/V5_PAIRED_GAINS_BY_TASK.csv',plotrows)

 # Figure 3: L2 level/shape decomposition, separated from the L1 results.
 l2=read_csv(ROOT/'results/LEVEL_SHAPE_L2.csv')
 l2rows=[x for x in l2 if x['task_id']=='five_task_macro' and x['seed']=='mean_across_formal_seeds']
 l2rows.sort(key=lambda x:['S','V','A','C','H_SV'].index(x['method']))
 fig,ax=plt.subplots(figsize=(10.5,6.5),layout='constrained')
 x=np.arange(len(l2rows));level=np.asarray([float(r['level_mse']) for r in l2rows]);shape=np.asarray([float(r['shape_mse']) for r in l2rows])
 ax.bar(x,level,color='#0072B2',label='Level MSE',width=.62)
 ax.bar(x,shape,bottom=level,color='#E69F00',label='Demeaned-shape MSE',width=.62)
 total=level+shape
 sd=np.asarray([float(r['position_mse_sample_sd']) for r in l2rows])
 ax.errorbar(x,total,yerr=sd,fmt='none',ecolor='#222222',capsize=3,linewidth=1.1,label='Position MSE sample SD')
 ax.set_xticks(x);ax.set_xticklabels([r['method'] for r in l2rows])
 ax.set_ylabel('Mean squared error after train-only target_sigma scaling')
 ax.set_title('L2 diagnostic decomposition across tasks')
 ax.legend(frameon=False,ncol=3,loc='upper left')
 ax.grid(axis='y',alpha=.2)
 fig3=ROOT/'figures/V5_LEVEL_SHAPE_L2.png';fig.savefig(fig3,dpi=180,facecolor='white');plt.close(fig)
 write_csv(ROOT/'figure_data/V5_LEVEL_SHAPE_L2.csv',[{'method':r['method'],'level_mse':r['level_mse'],'shape_mse':r['shape_mse'],
    'position_mse':r['position_mse'],'position_mse_sample_sd':r['position_mse_sample_sd']} for r in l2rows])

 # Figure 4: metadata-selected native command traces, same five methods, action 0/14, columns are seeds.
 traj=read_csv(ROOT/'figure_data/TRAJECTORY_DATA.csv');cases=read_csv(ROOT/'figure_data/TRAJECTORY_CASES.csv')
 method_order=['GT','S','V','H_SV','MIX50'];styles={'GT':('#111111','-'),'S':('#0072B2','--'),'V':('#D55E00','-.'),'H_SV':('#009E73','-'),'MIX50':('#CC79A7',':')}
 fig,axs=plt.subplots(10,3,figsize=(15,23),sharex=True,layout='constrained')
 for ti,task in enumerate(TASKS):
  case=next(c for c in cases if int(c['task_id'])==task)
  for ai,axis_index in enumerate([0,14]):
   rowidx=ti*2+ai
   for si,seed in enumerate(SEEDS):
    ax=axs[rowidx,si]
    for method in method_order:
     rr=[r for r in traj if int(r['task_id'])==task and r['seed']==seed and int(r['action_index'])==axis_index and r['method']==method]
     rr.sort(key=lambda r:int(r['future_step']))
     if len(rr)!=30:raise RuntimeError(f'trajectory panel incomplete {task}/{seed}/{axis_index}/{method}')
     xx=[int(r['future_step']) for r in rr];yy=[float(r['native_command_value']) for r in rr]
     color,ls=styles[method];ax.plot(xx,yy,color=color,linestyle=ls,linewidth=1.6 if method in ['GT','H_SV'] else 1.15,label=method)
    if ti==0 and ai==0:ax.set_title(f'seed {seed}')
    if si==0:ax.set_ylabel(f'Task {task}\nindex {axis_index}\nvalue')
    if rowidx==9:ax.set_xlabel('Future step (1–30)')
    ax.grid(alpha=.18)
    ax.text(.98,.96,f'{case["anchor_id"]}',transform=ax.transAxes,ha='right',va='top',fontsize=6,color='#555555')
 handles=[plt.Line2D([0],[0],color=styles[m][0],linestyle=styles[m][1],linewidth=1.7,label=m) for m in method_order]
 fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False)
 fig.suptitle('Metadata-selected representative forecast traces',fontsize=15)
 fig4=ROOT/'figures/V5_TRAJECTORY_EXAMPLES.png';fig.savefig(fig4,dpi=150,facecolor='white');plt.close(fig)

 captions='''# V5 Figure captions and descriptions\n\n## V5_DUAL_METRIC.png\n\nEpisode-equal, five-task macro Position MAE versus Safe-FD L1 for accepted references and fixed combinations. Points are the mean across three fixed model seeds; whiskers are sample SD, not confidence intervals. Persistence is a shared reference. The internal FD overlaps of H_SV/H_AV/H_PV with V, H_VS with S, and H_SC with C are algebraically inherited from their source trajectory shapes. Both axes use accepted train-only scales.\n\nAlt text: H_SV retains V's Safe-FD but its Position MAE lies above V, S, and BIAS_STATE; MIX50 is lower on both position and FD. H_SV is also above BIAS_VIDEO on position while matching its FD.\n\n## V5_PAIRED_GAINS_BY_TASK.png\n\nPaired source-minus-H_SV task estimates for V, optimized state S, BIAS_STATE, and BIAS_VIDEO. Positive gain favors H_SV. Each marker is one fixed seed; color identifies the reference and marker shape identifies the seed. Task labels show accepted development episode and anchor support. Intervals are not shown here; conditional paired source-group bootstrap intervals are in PAIRED_GROUP_CI.csv.\n\nAlt text: Position gains are mostly negative, including tasks 119, 94, and 141; Safe-FD gain is zero against V and BIAS_VIDEO by construction; seam gains are negative for the state/video shape substitutions.\n\n## V5_LEVEL_SHAPE_L2.png\n\nA separate diagnostic decomposition of standardized position MSE into level MSE and demeaned-shape MSE. Bars are five-task equal episode means across three fixed seeds; whiskers show sample SD for total position MSE. The additive bars are L2 quantities and are not the L1 Position MAE metric.\n\nAlt text: H_SV's level component matches S and its demeaned-shape component matches V to numerical precision.\n\n## V5_TRAJECTORY_EXAMPLES.png\n\nNative numeric command traces for one metadata-selected development anchor per task, action indices 0 and 14, and three seeds in separate columns. The task-119 anchor reuses V4's selected case; other anchors follow the locked metadata-only lowest-episode/frame/id rule. Each panel shows GT, S, V, H_SV, and MIX50 over 30 future steps. Physical units are unresolved.\n\nAlt text: Ten task/action-index rows by three seed columns; each panel compares the five specified methods on one fixed receiver window.\n'''
 (ROOT/'figures/CAPTIONS.zh-CN.md').write_text(captions,encoding='utf-8')
 info={}
 from PIL import Image
 for p in [fig1,fig2,fig3,fig4]:
  with Image.open(p) as im:info[p.name]={'width_px':im.width,'height_px':im.height,'mode':im.mode,'bytes':p.stat().st_size}
 qa={'schema_version':1,'status':'PASS_BUILD','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
    'figures':info,'figure_data_files':['V5_DUAL_METRIC.csv','V5_PAIRED_GAINS_BY_TASK.csv','V5_LEVEL_SHAPE_L2.csv','TRAJECTORY_CASES.csv','TRAJECTORY_DATA.csv'],
    'captions_file':str(ROOT/'figures/CAPTIONS.zh-CN.md'),'physical_units_claimed':False,
    'uncertainty_encoding':'sample SD across three fixed seeds where shown; conditional source-group CI is in a separate table',
    'trajectory_case_count':len(cases),'trajectory_rows':len(traj),'trajectory_axes':[0,14],
    'human_figure_review_status':'pending','assistant_visual_review_status':'pending'}
 write_json(ROOT/'qa/FIGURE_QA.json',qa)
 print(json.dumps({'status':'PASS_BUILD','figures':info,'trajectory_cases':len(cases),'trajectory_rows':len(traj)},indent=2))
if __name__=='__main__':main()
