#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
PRETTY={'position_mae':'Position MAE','safe_fd_l1':'Safe-FD L1','seam_safe_fd_l1':'Seam Safe-FD L1'}
PROGRAMS=('A0','AD','V0','VD')
COLORS={'A0':'#4C78A8','AD':'#9ECAE9','V0':'#F58518','VD':'#FFBF79','MIX0':'#54A24B','MIXD':'#B9D98C','S':'#777777','V':'#CC6677','SV50':'#AA4499','PERSISTENCE':'#333333'}
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def readnum(path):
 return rows(path)
def save(fig,name):
 path=ROOT/'figures'/name;path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig);return path
def main():
 plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.titlesize':15})
 macro=rows(ROOT/'results/METHOD_MACRO.csv')
 macro_map={(r['method'],r['metric']):r for r in macro}
 # Fig 1: actual masked-input training objective and deterministic mask rate.
 curves=rows(ROOT/'figure_data/TRAINING_CURVES.csv')
 fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
 for p in PROGRAMS:
  subsets=[[r for r in curves if r['program_id']==p and int(r['seed'])==seed] for seed in (202610131,202610132,202610133)]
  if any(len(x)!=160 for x in subsets):raise RuntimeError('training curve figure requires 160 epochs for each seed')
  x=np.arange(1,161);values=np.asarray([[float(r['masked_input_true_y_objective']) for r in sub] for sub in subsets])
  mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
  axs[0].plot(x,mean,label=p,color=COLORS[p],lw=1.8);axs[0].fill_between(x,mean-sd,mean+sd,color=COLORS[p],alpha=.14)
  mask=np.asarray([[float(r['numeric_context_drop_rate_observed']) for r in sub] for sub in subsets])
  axs[1].plot(x,mask.mean(axis=0),label=p,color=COLORS[p],lw=1.5)
 axs[0].set_ylabel('Train Y objective under training input policy');axs[0].set_title('Training objective (mean ± sample SD across seeds)')
 axs[1].set_ylabel('Observed drop rate');axs[1].set_xlabel('Epoch');axs[1].set_ylim(-.03,.58);axs[1].axhline(.5,color='black',ls='--',lw=.8,alpha=.5)
 for ax in axs:ax.grid(alpha=.2);ax.legend(ncol=4,frameon=False)
 fig.suptitle('V9 training curves and numeric-context mask rate')
 f1=save(fig,'TRAINING_CURVES_AND_MASK_RATE.png')
 # Fig 2: full-context single-head main metrics.
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,metric in zip(axs,METRICS):
  vals=[float(macro_map[(p,metric)]['mean']) for p in PROGRAMS];errs=[float(macro_map[(p,metric)]['sample_sd']) for p in PROGRAMS]
  ax.bar(np.arange(4),vals,yerr=errs,capsize=3,color=[COLORS[p] for p in PROGRAMS])
  ax.set_xticks(np.arange(4),PROGRAMS);ax.set_title(PRETTY[metric]);ax.set_ylabel('Five-task macro error')
  ax.grid(axis='y',alpha=.2)
 fig.suptitle('Full numeric context: single-head development scores (mean ± sample SD, n=3 seeds)')
 f2=save(fig,'FULL_CONTEXT_SINGLE_HEAD_METRICS.png')
 # Fig 3: paired gains, positive means the candidate is lower-error.
 ci=rows(ROOT/'results/PAIRED_GROUP_CI.csv')
 main_contrasts=('NUMCTX_DROP_VIDEO','NUMCTX_DROP_STATE','VIDEO_GAIN_0','VIDEO_GAIN_D','INTERACTION')
 labels={'NUMCTX_DROP_VIDEO':'V0 − VD','NUMCTX_DROP_STATE':'A0 − AD','VIDEO_GAIN_0':'A0 − V0','VIDEO_GAIN_D':'AD − VD','INTERACTION':'(AD − VD) − (A0 − V0)'}
 fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True)
 for ax,metric in zip(axs,METRICS):
  subset=[r for r in ci if r['metric']==metric and r['contrast'] in main_contrasts]
  subset=sorted(subset,key=lambda r:main_contrasts.index(r['contrast']))
  y=np.arange(len(subset));point=np.asarray([float(r['point_estimate']) for r in subset]);lo=np.asarray([float(r['ci_2_5']) for r in subset]);hi=np.asarray([float(r['ci_97_5']) for r in subset])
  ax.errorbar(point,y,xerr=np.vstack((point-lo,hi-point)),fmt='o',color='#2A5D7A',capsize=3)
  ax.axvline(0,color='black',lw=.8,ls='--');ax.set_title(PRETTY[metric]);ax.grid(axis='x',alpha=.2)
  ax.set_yticks(y,[labels[r['contrast']] for r in subset])
 axs[0].set_ylabel('Reference minus candidate; positive favors candidate')
 fig.suptitle('Paired source-group bootstrap intervals (95%, fixed models/dev exposure)')
 f3=save(fig,'PAIRED_GAINS_AND_INTERACTION.png')
 # Fig 4: fixed mixture comparisons.
 mix_methods=('S','V0','VD','SV50','MIX0','MIXD')
 fig,axs=plt.subplots(1,3,figsize=(14,4))
 for ax,metric in zip(axs,METRICS):
  vals=[float(macro_map[(p,metric)]['mean']) for p in mix_methods]
  err=[float(macro_map[(p,metric)]['sample_sd']) for p in mix_methods]
  ax.bar(np.arange(len(mix_methods)),vals,yerr=err,capsize=3,color=[COLORS.get(p,'#777777') for p in mix_methods])
  ax.set_xticks(np.arange(len(mix_methods)),mix_methods,rotation=30,ha='right');ax.set_title(PRETTY[metric]);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Fixed 0.5 S_OPT mixtures versus single heads (full-context dev)')
 f4=save(fig,'MIX0_MIXD_COMPARISON.png')
 # Fig 5: paired input intervention diagnostic; positive means forced mask has higher error.
 diagnostic=rows(ROOT/'results/MASKED_DIAGNOSTIC.csv')
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,metric in zip(axs,METRICS):
  sub=[r for r in diagnostic if r['task_id']=='macro']
  vals=[];errs=[]
  for p in PROGRAMS:
   v=np.asarray([float(r[metric+'_masked_minus_full']) for r in sub if r['method']==p],dtype=np.float64)
   vals.append(float(v.mean()));errs.append(float(v.std(ddof=1)))
  ax.bar(np.arange(4),vals,yerr=errs,capsize=3,color=[COLORS[p] for p in PROGRAMS])
  ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(4),PROGRAMS);ax.set_title(PRETTY[metric])
  ax.set_ylabel('Forced-mask error − full-context error');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Secondary diagnostic: forced numeric-context mask at inference (mean ± sample SD across seeds)')
 f5=save(fig,'FULL_VS_FORCED_MASK_DIAGNOSTIC.png')
 # V8-preselected metadata case plots use saved two-step/anchor rows only.
 trajectory=ROOT/'figure_data/TRAJECTORY_DATA.csv'
 f6=None
 if trajectory.is_file() and trajectory.stat().st_size:
  t=rows(trajectory);cases=rows(ROOT/'figure_data/TRAJECTORY_CASES.csv')
  combos=[(0,0),(0,14),(14,0),(14,14)]
  fig,axs=plt.subplots(5,4,figsize=(15,12),sharex=True)
  methods=('A0','AD','V0','VD','MIX0','MIXD');seed_values=(202610131,202610132,202610133)
  for ri,case in enumerate(cases):
   task=int(case['task_id']);aid=case['anchor_id']
   for ci,(action_index,step_index) in enumerate(combos):
    ax=axs[ri,ci]
    target_rows=[r for r in t if r['anchor_id']==aid and r['method']=='Y_TARGET' and int(r['action_index'])==action_index and int(r['step_index'])==step_index]
    if len(target_rows)!=1:raise RuntimeError('trajectory target point missing or duplicated')
    target_value=float(target_rows[0]['value']);ax.axhline(target_value,color='black',ls='--',lw=1.1,label='Y target')
    for mi,method in enumerate(methods):
     vals=[r for r in t if r['anchor_id']==aid and r['method']==method and int(r['action_index'])==action_index and int(r['step_index'])==step_index]
     if len(vals)!=3:raise RuntimeError(f'trajectory point support mismatch: task={task}/{method}')
     for si,row in enumerate(sorted(vals,key=lambda x:int(x['seed']))):
      jitter=(si-1)*.08
      ax.scatter(mi+jitter,float(row['value']),s=19,color=COLORS[method],alpha=.85,marker='o')
    ax.set_title(f"task {task}, dim {action_index}, step {step_index+1}",fontsize=8)
    ax.grid(axis='y',alpha=.18)
    ax.set_xticks(np.arange(len(methods)),methods,rotation=45,ha='right',fontsize=7)
  handles=[plt.Line2D([0],[0],color='black',ls='--',label='Y target')]+[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS[m],label=m,markersize=6) for m in methods]
  fig.legend(handles=handles,loc='lower center',ncol=7,frameon=False,bbox_to_anchor=(.5,.005))
  fig.text(.008,.5,'Native numeric value (physical units unresolved)',rotation=90,va='center',fontsize=9)
  fig.suptitle('V8-preselected metadata cases; action dimensions 0/14, future steps 1/15; seed points shown separately',y=.998)
  fig.tight_layout(rect=[.025,.045,1,.98])
  f6=save(fig,'PRESELECTED_TRAJECTORY_CASES.png')
 captions='''# V9 figures\n\n- TRAINING_CURVES_AND_MASK_RATE.png: true-Y objective measured on the randomized training inputs; mask rate is observed per epoch. Bands show sample SD across the three seeds.\n- FULL_CONTEXT_SINGLE_HEAD_METRICS.png: primary development results with complete numeric context restored. Values are five-task equal macros; bars show mean and sample SD across seeds.\n- PAIRED_GAINS_AND_INTERACTION.png: paired source-group bootstrap intervals conditional on the fixed models and development exposure. Positive values mean the candidate has lower error; interaction uses (AD−VD)−(A0−V0).\n- MIX0_MIXD_COMPARISON.png: fixed native-space 0.5 mixtures using the new synchronized V0/VD and accepted S_OPT.\n- FULL_VS_FORCED_MASK_DIAGNOSTIC.png: secondary input intervention; positive means force-masked error exceeds full-context error.\n- PRESELECTED_TRAJECTORY_CASES.png: V8-preselected metadata IDs; target values are reused from the accepted V8 trajectory export, while V9 model points remain separated by seed; action dimensions 0/14 and future steps 1/15.\n'''
 (ROOT/'figures/CAPTIONS.zh-CN.md').write_text(captions,encoding='utf-8')
 review={'schema_version':1,'status':'GENERATED_PENDING_VISUAL_REVIEW','created_utc':utc(),'figures':[str(p) for p in (f1,f2,f3,f4,f5)]+([str(f6)] if f6 else []),
 'source_tables':[str(ROOT/'results/METHOD_MACRO.csv'),str(ROOT/'results/PAIRED_GROUP_CI.csv'),str(ROOT/'results/MASKED_DIAGNOSTIC.csv'),str(ROOT/'figure_data/TRAINING_CURVES.csv')],
 'no_final_or_heldout_data':True}
 (ROOT/'figures/FIGURE_BUILD_QA.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','figures':review['figures']},sort_keys=True))
if __name__=='__main__':main()

