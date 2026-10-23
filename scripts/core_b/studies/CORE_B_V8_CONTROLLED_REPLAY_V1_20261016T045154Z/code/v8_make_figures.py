#!/usr/bin/env python3
"""Build V8 figures only from sealed tables and figure_data; no target archive reads."""
from __future__ import annotations
import ast,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT');SEEDS=(202610131,202610132,202610133)
COLORS={'GT_RETRAIN':'#2369a8','SELF_SOFT':'#e08324','SV_SOFT':'#379168'}

def read_csv(rel):
 with (ROOT/rel).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(rel,rows):
 if not rows:raise RuntimeError(f'empty figure table {rel}')
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':160,'savefig.dpi':180})
 ROOT.joinpath('figures').mkdir(exist_ok=True);ROOT.joinpath('figure_data').mkdir(exist_ok=True)
 # Training target Z (or Y for hard) and true-Y diagnostic, three seeds kept visible.
 curves=read_csv('figure_data/TRAINING_CURVES.csv')
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
 for program in PROGRAMS:
  for seed in SEEDS:
   q=sorted([r for r in curves if r['program_id']==program and int(r['seed'])==seed],key=lambda r:int(r['epoch']))
   ep=np.asarray([int(r['epoch']) for r in q]);z=np.asarray([float(r['training_target_objective']) for r in q]);y=np.asarray([float(r['true_y_position'])+.1*float(r['true_y_safe_fd']) for r in q])
   axes[0].plot(ep,z,color=COLORS[program],alpha=.28,lw=.8)
   axes[1].plot(ep,y,color=COLORS[program],alpha=.28,lw=.8)
  q=[r for r in curves if r['program_id']==program]
  by={int(r['epoch']):[] for r in q}
  for r in q:by[int(r['epoch'])].append(float(r['training_target_objective']))
  axes[0].plot(sorted(by),[np.mean(by[e]) for e in sorted(by)],color=COLORS[program],lw=2.4,label=program)
  by_y={int(r['epoch']):[] for r in q}
  for r in q:by_y[int(r['epoch'])].append(float(r['true_y_position'])+.1*float(r['true_y_safe_fd']))
  axes[1].plot(sorted(by_y),[np.mean(by_y[e]) for e in sorted(by_y)],color=COLORS[program],lw=2.4,label=program)
 axes[0].set_title('Locked training target (Y for HARD; Z for soft)');axes[1].set_title('True-Y training diagnostic')
 for ax in axes:ax.set_xlabel('Epoch');ax.set_ylabel('Position + 0.1 × 29-difference FD');ax.grid(alpha=.2);ax.legend(frameon=False)
 fig.suptitle('V8 branch B: train objectives by program (thin: seed, thick: seed mean)');fig.tight_layout();fig.savefig(ROOT/'figures/TRAINING_CURVES.png',bbox_inches='tight');plt.close(fig)
 # Paired task/seed gains for dual metric and seam; positive means candidate loss lower.
 paired=read_csv('results/PAIRED_TASK_SEED.csv');main_contrasts=['SELF_TARGET_GAIN','SV_TARGET_GAIN','SV_VS_SELF']
 figure_rows=[r for r in paired if r['contrast'] in main_contrasts]
 write_csv('figure_data/PAIRED_GAINS_BY_TASK.csv',figure_rows)
 fig,axes=plt.subplots(1,3,figsize=(13,4.2))
 metrics=[('position_mae','Position MAE'),('safe_fd_l1','Safe-FD L1'),('seam_safe_fd_l1','Seam Safe-FD L1')]
 contrast_labels={'SELF_TARGET_GAIN':'Hard − SELF','SV_TARGET_GAIN':'Hard − SV','SV_VS_SELF':'SELF − SV'}
 offsets={202610131:-.12,202610132:0,202610133:.12}
 for ax,(metric,title) in zip(axes,metrics):
  for x,contrast in enumerate(main_contrasts):
   for seed in SEEDS:
    vals=[float(r['gain_reference_minus_candidate']) for r in figure_rows if r['contrast']==contrast and r['metric']==metric and int(r['seed'])==seed]
    value=float(np.mean(vals));ax.scatter(x+offsets[seed],value,s=32,label=str(seed) if x==0 else None)
   seed_means=[float(np.mean([float(r['gain_reference_minus_candidate']) for r in figure_rows if r['contrast']==contrast and r['metric']==metric and int(r['seed'])==seed])) for seed in SEEDS]
   mean=float(np.mean(seed_means));sd=float(np.std(seed_means,ddof=1));ax.errorbar(x,mean,yerr=sd,fmt='D',color='black',capsize=3,lw=1.2,markersize=4)
  ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(len(main_contrasts)),[contrast_labels[x] for x in main_contrasts],rotation=20,ha='right');ax.set_title(title);ax.set_ylabel('Paired gain (positive = candidate lower loss)');ax.grid(axis='y',alpha=.2)
 axes[0].legend(title='Seed',frameon=False,loc='best');fig.suptitle('V8 branch B paired development gains (points: seed × task mean; diamond: overall mean ± sample SD)');fig.tight_layout();fig.savefig(ROOT/'figures/PAIRED_DUAL_METRICS_AND_SEAM.png',bbox_inches='tight');plt.close(fig)
 # Train-Y/Z fit and zero-optimizer replay/step discrepancies.
 final=read_csv('results/TRAIN_FINAL_SUMMARY.csv');audit=json.loads((ROOT/'audit/GPU_ZERO_OPTIMIZER_AUDIT.json').read_text());teacher=read_csv('audit/TEACHER_CLONE_REPLAY.csv');steps=read_csv('audit/NATIVE_LOSS_GRAD_STEP_DIFF.csv')
 fig,axes=plt.subplots(1,2,figsize=(12,4.6))
 x=np.arange(len(PROGRAMS));width=.24
 for ix,field,label,color in [(0,'train_y_macro','True Y','#5b6b7a'),(1,'train_target_macro','Training target','#c46c2b'),(2,'teacher_distance_macro','Assigned teacher distance','#557fa6')]:
  vals=[]
  for program in PROGRAMS:
   items=[r for r in final if r['program_id']==program]
   a=[ast.literal_eval(r[field])['position_mae']+.1*ast.literal_eval(r[field])['safe_fd_l1'] for r in items]
   vals.append(float(np.mean(a)))
  axes[0].bar(x+(ix-1)*width,vals,width,label=label,color=color)
 axes[0].set_xticks(x,PROGRAMS);axes[0].set_ylabel('Position + 0.1 × FD');axes[0].set_title('Epoch-160 train evaluation');axes[0].legend(frameon=False);axes[0].grid(axis='y',alpha=.2)
 vals={'Teacher replay max abs':max(float(r['clone_replay_max_abs_diff']) for r in teacher),'Student replay max abs':max(float(r['student_replay_max_abs_diff']) for r in teacher),
 'Per-sample loss max abs':max(max(float(r['position_per_sample_max_abs_diff']),float(r['future_29diff_per_sample_max_abs_diff'])) for r in steps),
 'Gradient max abs':max(float(r['grad_max_abs_diff']) for r in steps),'AdamW step max abs':max(float(r['parameter_max_abs_diff_after_step']) for r in steps)}
 labels=list(vals);raw=np.asarray([vals[k] for k in labels]);plotv=np.maximum(raw,1e-16);axes[1].barh(np.arange(len(labels)),plotv,color=['#4b8b63','#4b8b63','#d29639','#d29639','#d29639']);axes[1].set_yticks(np.arange(len(labels)),labels);axes[1].invert_yaxis();axes[1].set_xscale('log');axes[1].set_xlabel('Maximum absolute discrepancy (log scale; exact zeros shown at 1e−16)');axes[1].set_title('Zero-optimizer replay and update QA');axes[1].grid(axis='x',which='both',alpha=.2)
 for i,value in enumerate(raw):axes[1].text(max(plotv[i]*1.1,1.2e-16),i,'0' if value==0 else f'{value:.1e}',va='center',fontsize=8)
 fig.suptitle('V8 train fit and path audit');fig.tight_layout();fig.savefig(ROOT/'figures/TRAIN_AND_REPLAY_AUDIT.png',bbox_inches='tight');plt.close(fig)
 # V7 metadata-preselected cases, action channels 0 and 14, each seed separated by line style.
 trajectory=read_csv('figure_data/TRAJECTORY_DATA.csv');cases=read_csv('figure_data/TRAJECTORY_CASES.csv')
 seed_styles={202610131:'-',202610132:'--',202610133:':'}
 fig,axes=plt.subplots(len(cases),2,figsize=(13,2.7*len(cases)),sharex=True)
 if len(cases)==1:axes=np.asarray([axes])
 legend_handles=[Line2D([0],[0],color='black',ls='--',label='Y target'),Line2D([0],[0],color='#7b4b94',ls='-',label='Old V (seed mean)')]
 for program in PROGRAMS:legend_handles.append(Line2D([0],[0],color=COLORS[program],ls='-',label=program+' (seed line style)'))
 for ri,case in enumerate(cases):
  aid=case['anchor_id']
  for ci,ch in enumerate((0,14)):
   ax=axes[ri,ci]
   q=[r for r in trajectory if r['anchor_id']==aid and int(r['action_index'])==ch]
   y=next(r for r in q if r['method']=='Y_TARGET');ax.plot(range(1,31),[float(r['native_command_value']) for r in sorted([x for x in q if x['method']=='Y_TARGET'],key=lambda r:int(r['future_step']))],color='black',ls='--',lw=1.8)
   old=[r for r in q if r['method']=='V_OLD'];oldmean=[]
   for step in range(1,31):oldmean.append(float(np.mean([float(r['native_command_value']) for r in old if int(r['future_step'])==step])))
   ax.plot(range(1,31),oldmean,color='#7b4b94',lw=1.7)
   for program in PROGRAMS:
    for seed in SEEDS:
     d=sorted([r for r in q if r['method']==program and int(r['seed'])==seed],key=lambda r:int(r['future_step']))
     ax.plot(range(1,31),[float(r['native_command_value']) for r in d],color=COLORS[program],ls=seed_styles[seed],alpha=.72,lw=1.1)
   ax.set_title(f"Task {case['task_id']} • action index {ch} • {case['case_selection']}");ax.set_ylabel('Numeric command');ax.grid(alpha=.2)
   if ri==len(cases)-1:ax.set_xlabel('Future step (1–30)')
 fig.legend(handles=legend_handles,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,1.01));fig.suptitle('V7 metadata-preselected development trajectories (seed line styles; physical units unresolved)',y=1.035);fig.tight_layout();fig.savefig(ROOT/'figures/TRAJECTORY_CASES.png',bbox_inches='tight');plt.close(fig)
 captions='''# V8 图注\n\n- `TRAINING_CURVES.png`: 每个 branch-B 程序训练的优化目标（GT_RETRAIN 为 Y；SELF_SOFT/SV_SOFT 为固定 Z）与同一模型真实 Y 诊断；细线为 seed，粗线为 seed 均值。\n- `PAIRED_DUAL_METRICS_AND_SEAM.png`: 三个预锁定主要 target contrasts 的 Position、29-difference Safe-FD 和 Seam gain；每个点为一个 seed 的五任务 paired mean，菱形和误差线为总体均值与 seed sample SD。正值表示 candidate loss 更低。\n- `TRAIN_AND_REPLAY_AUDIT.png`: epoch-160 外层 train Y/Z/teacher-distance objective 与零 optimizer 复放、独立 loss/gradient、临时 AdamW step 的最大数值差。\n- `TRAJECTORY_CASES.png`: V7 metadata 预选的五个 case，action indices 0 与 14；新模型三 seed 用不同线型，保留 V7 old-V reference。纵轴为 numeric command，物理单位未知。\n'''
 (ROOT/'figures/CAPTIONS.zh-CN.md').write_text(captions,encoding='utf-8')
 qa={'schema_version':1,'status':'PASS','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),
 'figures':['TRAINING_CURVES.png','PAIRED_DUAL_METRICS_AND_SEAM.png','TRAIN_AND_REPLAY_AUDIT.png','TRAJECTORY_CASES.png'],
 'figure_data_sources':['figure_data/TRAINING_CURVES.csv','figure_data/PAIRED_GAINS_BY_TASK.csv','figure_data/TRAJECTORY_DATA.csv','results/PRIMARY_TASK_SEED.csv','results/PAIRED_TASK_SEED.csv','audit/NATIVE_LOSS_GRAD_STEP_DIFF.csv'],
 'dev_target_archive_opened_by_figure_builder':False,'final_heldout_payload_access':0}
 (ROOT/'qa/FIGURE_BUILD_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','figures':len(qa['figures']),'case_panels':len(cases)*2,'dev_target_archive_opened':False},sort_keys=True))
if __name__=='__main__':main()
