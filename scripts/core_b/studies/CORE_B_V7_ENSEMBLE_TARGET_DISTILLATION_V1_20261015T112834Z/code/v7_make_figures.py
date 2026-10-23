#!/usr/bin/env python3
"""Generate V7 training, metric, gain, imitation, and preselected trajectory figures."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PROGRAMS=('ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV')
TASKS=(119,94,90,141,230)

def read_csv(p:Path):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(p:Path,rows):
 p.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main()->int:
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':7,'figure.titlesize':13})
 curves=[]
 for program in PROGRAMS:
  for seed in (202610131,202610132,202610133):
   path=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'/'EPOCH_LOG.csv'
   for row in read_csv(path):curves.append({'program_id':program,'seed':seed,**row})
 write_csv(ROOT/'figure_data/TRAINING_CURVES.csv',curves)
 fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,constrained_layout=True)
 colors=['#4477AA','#EE6677','#228833']
 for ax,program in zip(axes.flat,PROGRAMS):
  subset=[r for r in curves if r['program_id']==program]
  epochs=np.asarray(sorted({int(r['epoch']) for r in subset}))
  for seed,color in zip((202610131,202610132,202610133),colors):
   sr=sorted([r for r in subset if int(r['seed'])==seed],key=lambda x:int(x['epoch']))
   ax.plot([int(r['epoch']) for r in sr],[float(r['soft_z_objective']) for r in sr],color=color,alpha=.45,linewidth=.8)
   ax.plot([int(r['epoch']) for r in sr],[float(r['true_y_position_plus_0.1_fd']) for r in sr],color=color,alpha=.45,linewidth=.8,linestyle='--')
  soft=np.asarray([[float(r['soft_z_objective']) for r in sorted([x for x in subset if int(x['seed'])==s],key=lambda x:int(x['epoch']))] for s in (202610131,202610132,202610133)])
  hard=np.asarray([[float(r['true_y_position_plus_0.1_fd']) for r in sorted([x for x in subset if int(x['seed'])==s],key=lambda x:int(x['epoch']))] for s in (202610131,202610132,202610133)])
  ax.plot(epochs,soft.mean(axis=0),color='#333333',linewidth=2,label='soft-Z objective')
  ax.plot(epochs,hard.mean(axis=0),color='#AA3377',linewidth=2,linestyle='--',label='true-Y diagnostic')
  ax.set_title(program);ax.set_xlabel('Epoch');ax.set_ylabel('Task/episode-equal loss');ax.grid(alpha=.25)
 axes[0,0].legend(loc='best')
 fig.suptitle('V7: fixed soft-Z objective and true-Y training diagnostic')
 fig.savefig(ROOT/'figures/TRAINING_CURVES.png',dpi=180,bbox_inches='tight');plt.close(fig)
 macro=read_csv(ROOT/'results/METHOD_MACRO.csv')
 methods=['S','A','V','SS50','SV50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE',*PROGRAMS]
 labels=[m for m in methods if any(r['method']==m for r in macro)]
 fig,axes=plt.subplots(1,3,figsize=(17,6),constrained_layout=True)
 for ax,metric,title in zip(axes,('position_mae','safe_fd_l1','seam_safe_fd_l1'),('Position MAE','Future internal safe-FD L1','Seam safe-FD L1')):
  rows={r['method']:r for r in macro if r['metric']==metric};vals=[float(rows[m]['macro_mean']) for m in labels];sd=[float(rows[m]['seed_sample_sd']) for m in labels]
  colors=['#777777' if m in {'S','A','V','SS50','SV50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE'} else ('#4477AA' if m.startswith('ST_') else '#EE6677') for m in labels]
  ax.bar(np.arange(len(labels)),vals,yerr=sd,color=colors,capsize=2,alpha=.88);ax.set_xticks(np.arange(len(labels)),labels,rotation=60,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.25)
  ax.set_ylabel('Macro loss (mean over 3 seeds; error bars = seed SD)')
 fig.suptitle('V7 students, frozen teachers, and hard-only references')
 fig.savefig(ROOT/'figures/METHOD_MACRO_DUAL_METRICS_AND_SEAM.png',dpi=180,bbox_inches='tight');plt.close(fig)
 pair=read_csv(ROOT/'results/PAIRED_TASK_SEED.csv')
 contrasts=['VD_KD_VS_HARD','VD_SV_VS_SELF','VD_VISUAL_TEACHER','ST_KD_VS_HARD','ST_SV_VS_SELF','ST_VISUAL_TEACHER','STUDENT_INPUT_GAIN']
 gain_rows=[]
 for c in contrasts:
  for metric in ('position_mae','safe_fd_l1','seam_safe_fd_l1'):
   for task in TASKS:
    vals=[float(r['gain_reference_minus_candidate']) for r in pair if r['contrast']==c and r['metric']==metric and int(r['task_id'])==task]
    if vals:gain_rows.append({'contrast':c,'metric':metric,'task_id':task,'gain_mean_over_seeds':float(np.mean(vals))})
 write_csv(ROOT/'figure_data/PAIRED_GAINS_BY_TASK.csv',gain_rows)
 fig,axes=plt.subplots(1,3,figsize=(16,7),constrained_layout=True)
 for ax,metric,title in zip(axes,('position_mae','safe_fd_l1','seam_safe_fd_l1'),('Position gain','Safe-FD gain','Seam gain')):
  mat=np.zeros((len(contrasts),len(TASKS)))
  for i,c in enumerate(contrasts):
   for j,t in enumerate(TASKS):
    rr=[r for r in gain_rows if r['contrast']==c and r['metric']==metric and r['task_id']==t]
    if rr:mat[i,j]=rr[0]['gain_mean_over_seeds']
  lim=max(float(np.max(np.abs(mat))),1e-9);im=ax.imshow(mat,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim)
  ax.set_xticks(range(len(TASKS)),TASKS);ax.set_yticks(range(len(contrasts)),contrasts);ax.set_title(title);fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='reference − candidate')
 fig.suptitle('Paired gains by task; positive means lower candidate loss')
 fig.savefig(ROOT/'figures/PAIRED_GAINS_BY_TASK.png',dpi=180,bbox_inches='tight');plt.close(fig)
 imitation=read_csv(ROOT/'results/IMITATION_ERROR.csv')
 fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
 for ax,dataset in zip(axes,('train','dev')):
  rows=[r for r in imitation if r['dataset_role']==dataset]
  labels=[f"{r['program_id']}\n{int(r['seed'])%1000}" for r in rows]
  x=np.arange(len(rows));ax.bar(x-.18,[float(r['imitation_position_mae']) for r in rows],.36,label='position');ax.bar(x+.18,[float(r['imitation_safe_fd_l1']) for r in rows],.36,label='safe-FD')
  ax.set_xticks(x,labels,rotation=75,ha='right');ax.set_title(f'{dataset}: student vs assigned teacher');ax.grid(axis='y',alpha=.25);ax.legend()
 fig.savefig(ROOT/'figures/IMITATION_ERROR.png',dpi=180,bbox_inches='tight');plt.close(fig)
 traj=read_csv(ROOT/'figure_data/TRAJECTORY_DATA.csv');fig,axes=plt.subplots(len(TASKS),2,figsize=(14,17),sharex=True,constrained_layout=True)
 for row,task in enumerate(TASKS):
  for col,action in enumerate((0,14)):
   ax=axes[row,col];sub=[r for r in traj if int(r['task_id'])==task and int(r['action_index'])==action]
   gt=next(r for r in sub if r['method']=='GT' and int(r['seed'])==202610131)
   gtv=sorted([r for r in sub if r['method']=='GT' and int(r['seed'])==202610131],key=lambda x:int(x['future_step']))
   ax.plot(range(1,31),[float(r['native_command_value']) for r in gtv],color='black',linewidth=2,label='GT')
   sv=sorted([r for r in sub if r['method']=='SV50' and int(r['seed'])==202610131],key=lambda x:int(x['future_step']))
   ax.plot(range(1,31),[float(r['native_command_value']) for r in sv],color='#AA3377',linewidth=1.5,label='SV50')
   for seed,color in zip((202610131,202610132,202610133),('#4477AA','#66AADD','#114477')):
    for method,base,ls in (('ST_SV',color,'-'),('VD_SV','#EE6677','--')):
     vals=sorted([r for r in sub if r['method']==method and int(r['seed'])==seed],key=lambda x:int(x['future_step']))
     ax.plot(range(1,31),[float(r['native_command_value']) for r in vals],color=base if method=='ST_SV' else color,alpha=.8,linewidth=.9,linestyle=ls,label=f'{method} {seed}' if row==0 and col==0 else None)
   ax.set_title(f'Task {task}; action index {action}');ax.set_xlabel('Future step');ax.set_ylabel('Numeric command value');ax.grid(alpha=.2)
 axes[0,0].legend(ncol=2,loc='best')
 fig.suptitle('V6 preselected dev cases; action indices 0 and 14; three seeds shown separately')
 fig.savefig(ROOT/'figures/TRAJECTORY_CASES.png',dpi=180,bbox_inches='tight');plt.close(fig)
 captions='''# V7 figure captions\n\n- `TRAINING_CURVES.png`: Six fixed student programs; solid curves are the beta=0.5 soft-Z objective and dashed curves are the same model's true-Y training diagnostic. Thin curves show each fixed seed; thick curves show the three-seed mean.\n- `METHOD_MACRO_DUAL_METRICS_AND_SEAM.png`: Five-task macro development losses for frozen teachers, hard-only references, and all six students; error bars show sample SD across the three fixed seeds.\n- `PAIRED_GAINS_BY_TASK.png`: Paired `reference - candidate` gains by task, averaged over the three fixed seeds; positive values mean the candidate loss is lower.\n- `IMITATION_ERROR.png`: Student predictions against their assigned frozen teacher on train and development prefixes; this is imitation error, not task loss.\n- `TRAJECTORY_CASES.png`: V6 metadata-preselected cases, numeric native command values, action indices 0 and 14, with all three seeds shown separately; physical units remain unresolved.\n'''
 (ROOT/'figures/CAPTIONS.zh-CN.md').write_text(captions,encoding='utf-8')
 qa={'schema_version':1,'status':'PASS','figure_count':5,'figures':['TRAINING_CURVES.png','METHOD_MACRO_DUAL_METRICS_AND_SEAM.png','PAIRED_GAINS_BY_TASK.png','IMITATION_ERROR.png','TRAJECTORY_CASES.png'],
     'figure_data':['TRAINING_CURVES.csv','PAIRED_GAINS_BY_TASK.csv','TRAJECTORY_DATA.csv'],'target_archive_reopened':False,'human_visual_review_status':'pending'}
 (ROOT/'qa/FIGURE_BUILD_QA.json').write_text(json.dumps(qa,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','figures':5,'training_curve_rows':len(curves),'paired_task_rows':len(gain_rows),'trajectory_rows':len(traj)},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
