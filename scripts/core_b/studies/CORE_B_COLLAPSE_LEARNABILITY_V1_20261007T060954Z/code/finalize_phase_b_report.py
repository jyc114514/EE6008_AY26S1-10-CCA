#!/usr/bin/env python3
"""Final train-only report and figures after head and triggered free-path runs."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/"runs/free_path"
sys.path.insert(0,str(ROOT/"code"))
import summarize_phase_b as head  # noqa:E402

COLORS={"TARGET":"#111111","TS11-D":"#0072B2","STATE-CTRL":"#D55E00","FREE_FULL":"#009E73","FREE_POS":"#CC79A7","ANCHOR":"#777777"}


def sha_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def write_json(path:Path,value:Any)->None:
 path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')


def write_csv(path:Path,rows:list[dict[str,Any]])->None:
 if not rows:raise ValueError(f'refuse empty table: {path}')
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');writer.writeheader();writer.writerows(rows)


def read_json(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding='utf-8'))


def load_freepath()->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],dict[str,dict[str,np.ndarray]]]:
 trigger=read_json(ROOT/'protocol/B_FREE_PATH_TRIGGER_LOCK.json')
 if trigger.get('status')!='LOCKED_BEFORE_FREE_PATH_CONTROLS' or trigger.get('selected_panel')!='P8':raise AssertionError('free-path trigger lock absent')
 run_summary=[];samples=[];curves=[];predictions={}
 for loss in ('FULL_LOSS','POS_ONLY'):
  run_id=f'FREEPATH-H30-P8-{loss}-S20261002';run_root=RUNS/run_id
  complete=read_json(run_root/'RUN_COMPLETE.json')
  if (complete.get('status')!='PASS' or complete.get('panel')!='P8' or complete.get('loss_name')!=loss
      or complete.get('optimizer_updates')!=2000 or complete.get('independent_residual_shape')!=[8,30,28]
      or complete.get('independent_trainable_values')!=6720 or complete.get('init_value')!='exact zeros'
      or complete.get('trigger_lock_sha256')!=sha_file(ROOT/'protocol/B_FREE_PATH_TRIGGER_LOCK.json')
      or complete.get('dev_targets_opened') is not False or complete.get('final_or_heldout_access')!=0):
   raise AssertionError(f'free-path acceptance failed: {run_id}')
  pred_path=run_root/'PREDICTIONS_FINAL.npz';checkpoint=run_root/'checkpoint_final.pt'
  if sha_file(pred_path)!=complete['prediction_sha256'] or sha_file(checkpoint)!=complete['final_checkpoint_sha256']:
   raise AssertionError(f'free-path final artifact hash mismatch: {run_id}')
  metric_rows=list(csv.DictReader((run_root/'METRICS.csv').open(newline='',encoding='utf-8')))
  sample_rows=list(csv.DictReader((run_root/'SAMPLE_METRICS.csv').open(newline='',encoding='utf-8')))
  final_samples=[r for r in sample_rows if int(r['step'])==2000]
  if len(metric_rows)!=20 or [int(x['step']) for x in metric_rows]!=list(range(100,2001,100)) or len(final_samples)!=8:
   raise AssertionError(f'free-path metrics incomplete: {run_id}')
  if not all(str(r['strong_all_metrics_pass']).lower()=='true' for r in final_samples):
   raise AssertionError(f'free-path positive control did not pass all P8 sample criteria: {run_id}')
  run_summary.append({
   'run_id':run_id,'model':'FREE_PATH','loss':loss,'panel':'P8','seed':20261002,'updates':2000,
   'elapsed_seconds':float(complete['elapsed_seconds']),'position_error_normalized_mean':float(metric_rows[-1]['position_error_normalized_mean']),
   'persistence_position_error_normalized_mean':float(metric_rows[-1]['persistence_position_error_normalized_mean']),
   'position_gain_vs_persistence_mean':float(metric_rows[-1]['position_gain_vs_persistence_mean']),
   'fd_error_safe_scale_normalized_mean':float(metric_rows[-1]['fd_error_safe_scale_normalized_mean']),
   'persistence_fd_error_safe_scale_normalized_mean':float(metric_rows[-1]['persistence_fd_error_safe_scale_normalized_mean']),
   'fd_gain_vs_persistence_mean':float(metric_rows[-1]['fd_gain_vs_persistence_mean']),
   'prediction_tv_over_target_tv_mean':float(metric_rows[-1]['tv_ratio_mean']),
   'strong_training_fit':True,'per_sample_strong_fit_count':8,'panel_rows':8,
   'independent_trainable_values':6720,'not_a_forecasting_model':True,
   'final_checkpoint_sha256':complete['final_checkpoint_sha256'],'prediction_sha256':complete['prediction_sha256']})
  for r in final_samples:
   samples.append({**{k:run_summary[-1][k] for k in ('run_id','model','loss','panel','seed')},
                   **{k:r[k] for k in r if k not in ('run_id','step')},
                   'strong_all_metrics_pass':True})
  for r in metric_rows:curves.append({'run_id':run_id,'model':'FREE_PATH','loss':loss,'panel':'P8',**r})
  with np.load(pred_path,allow_pickle=False) as z:
   predictions[run_id]={'ids':z['anchor_id'].astype('U96'),'target':z['target_native'].astype(np.float32),
                        'anchor':z['anchor_native'].astype(np.float32),'prediction':z['prediction_native'].astype(np.float32)}
 return run_summary,samples,curves,predictions


def freepath_figures(head_predictions:dict[str,dict[str,np.ndarray]],free_predictions:dict[str,dict[str,np.ndarray]],panel_lock:dict[str,Any],free_curves:list[dict[str,Any]])->list[dict[str,Any]]:
 figures=[];h=np.arange(1,31)
 full_id='FREEPATH-H30-P8-FULL_LOSS-S20261002';pos_id='FREEPATH-H30-P8-POS_ONLY-S20261002'
 state_id='TINYFIT-STATE-CTRL-H30-FULL_LOSS-P8-S20261002'
 target=free_predictions[full_id]['target'];ids=free_predictions[full_id]['ids']
 all_values=[]
 for x in (free_predictions[full_id]['target'][:,:,0],free_predictions[full_id]['prediction'][:,:,0],free_predictions[pos_id]['prediction'][:,:,0],head_predictions[state_id]['prediction'][:,:,0]):all_values.extend(np.asarray(x).ravel().tolist())
 low=min(0.0,min(all_values));high=max(0.0,max(all_values));margin=max((high-low)*.06,1e-6)
 fig,axes=plt.subplots(2,4,figsize=(13,6.8),sharex=True,sharey=True,layout='constrained')
 for i,ax in enumerate(axes.flat):
  ax.scatter([0],[float(free_predictions[full_id]['anchor'][i,0])],marker='x',s=24,color=COLORS['ANCHOR'],label='anchor (separate seam)')
  ax.plot(h,target[i,:,0],color=COLORS['TARGET'],linestyle=':',linewidth=1.5,label='train target (used in fitting)')
  ax.plot(h,head_predictions[state_id]['prediction'][i,:,0],color=COLORS['STATE-CTRL'],linestyle='--',marker='^',markevery=6,linewidth=1.25,label='STATE-CTRL / FULL_LOSS')
  ax.plot(h,free_predictions[full_id]['prediction'][i,:,0],color=COLORS['FREE_FULL'],linestyle='-',marker='o',markevery=6,linewidth=1.25,label='free path / FULL_LOSS')
  ax.plot(h,free_predictions[pos_id]['prediction'][i,:,0],color=COLORS['FREE_POS'],linestyle='-.',marker='s',markevery=6,linewidth=1.25,label='free path / POS_ONLY')
  ax.set_ylim(low-margin,high+margin);ax.grid(axis='y',alpha=.2)
  ax.set_title(f'P8 {i+1}: {ids[i]}',fontsize=8)
  if i//4==1:ax.set_xlabel('Future query index')
  if i%4==0:ax.set_ylabel('Native coordinate 0 (unit unspecified)')
  if i==0:ax.legend(fontsize=7,loc='best')
 fig.suptitle('P8 independent-output positive control · not a forecasting model · '+head.DIAG_LABEL,fontsize=12)
 path=head.OUT_FIGURES/'TINYFIT_P8_FREEPATH_VS_HEAD.png';fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
 figures.append({'path':str(path.relative_to(ROOT)),'caption':'P8 raw coordinate 0: final STATE-CTRL/FULL_LOSS head and two independent-output free-path positive controls against the same eight fitted targets.','alt_text':'Eight small multiples compare each P8 train target to the final state-control head and two free paths initialized at zero. The free paths are not forecasting models; each target was used during fitting. The anchor is shown separately, not connected to future steps.'})

 fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,layout='constrained')
 metrics=[('position_gain_vs_persistence_mean','Position gain',.90),('fd_gain_vs_persistence_mean','Safe-FD gain',.80),('tv_ratio_mean','TV ratio',None)]
 for ax,(metric,title,threshold) in zip(axes,metrics):
  for loss,color,style,marker in (('FULL_LOSS',COLORS['FREE_FULL'],'-','o'),('POS_ONLY',COLORS['FREE_POS'],'--','s')):
   run_id=f'FREEPATH-H30-P8-{loss}-S20261002';rows=sorted([r for r in free_curves if r['loss']==loss],key=lambda x:int(x['step']))
   ax.plot([int(x['step']) for x in rows],[float(x[metric]) for x in rows],color=color,linestyle=style,marker=marker,markevery=4,label=f'Free path · {loss}')
  if threshold is not None:ax.axhline(threshold,color='#444444',linestyle=':',label='strong-fit threshold')
  else:ax.axhline(.8,color='#444444',linestyle=':',label='TV lower bound');ax.axhline(1.2,color='#777777',linestyle='-.',label='TV upper bound')
  ax.grid(alpha=.2);ax.set_title(title);ax.set_xlabel('Optimizer updates')
 axes[0].set_ylabel('Panel metric');axes[0].legend(fontsize=7,loc='best')
 fig.suptitle('P8 free-path positive controls · 8 train targets used in fitting · not a forecasting model',fontsize=12)
 path=head.OUT_FIGURES/'TINYFIT_P8_FREEPATH_METRIC_CURVES.png';fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
 figures.append({'path':str(path.relative_to(ROOT)),'caption':'P8 free-path metrics over 2000 updates. The two controls use independent trainable outputs and the same locked train targets; thresholds are the predeclared diagnostic criteria.','alt_text':'Three panels show position gain, safe-FD gain, and TV ratio for FULL_LOSS and POS_ONLY free paths over 2000 updates. Both controls fit all eight training trajectories; these are positive controls, not forecast models or held-out results.'})
 return figures



def head_trajectory_figures(predictions:dict[str,dict[str,np.ndarray]],panel_lock:dict[str,Any])->list[dict[str,Any]]:
 figures=[];h=np.arange(1,31)
 p1_runs=[f'TINYFIT-{config}-H30-{loss}-P1-S20261002' for config in ('TS11-D','STATE-CTRL') for loss in ('FULL_LOSS','POS_ONLY')]
 ylims={}
 for coord in (0,14):
  values=[]
  for run_id in p1_runs:
   x=predictions[run_id];values.extend(x['target'][0,:,coord].tolist());values.extend(x['prediction'][0,:,coord].tolist());values.append(float(x['anchor'][0,coord]))
  lo=min(0.0,float(min(values)));hi=max(0.0,float(max(values)));margin=max((hi-lo)*.06,1e-7)
  ylims[coord]=(lo-margin,hi+margin)
 fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True,layout='constrained')
 for ri,run_id in enumerate(p1_runs):
  x=predictions[run_id];config='TS11-D' if 'TS11-D' in run_id else 'STATE-CTRL';loss='FULL_LOSS' if 'FULL_LOSS' in run_id else 'POS_ONLY';style='-' if loss=='FULL_LOSS' else '--';marker='o' if loss=='FULL_LOSS' else 's'
  for ci,coord in enumerate((0,14)):
   ax=axes[ri,ci];ax.axhline(0,color='#aaaaaa',linewidth=.6);ax.scatter([0],[float(x['anchor'][0,coord])],marker='x',s=28,color=COLORS['ANCHOR'],label='anchor (separate seam)')
   ax.plot(h,x['target'][0,:,coord],color=COLORS['TARGET'],linestyle=':',linewidth=1.7,label='future target (used in fitting)')
   ax.plot(h,x['prediction'][0,:,coord],color=COLORS[config],linestyle=style,marker=marker,markevery=5,linewidth=1.6,label=f'{config} / {loss}')
   ax.set_ylim(*ylims[coord]);ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(axis='y',alpha=.2);ax.set_title(f'{config} · {loss} · raw coordinate {coord}',fontsize=9)
   if ci==0:ax.set_ylabel('Native command value (unit unspecified; zero shown)')
   if ri==3:ax.set_xlabel('Future query index (anchor at 0 is not joined)')
   if ri==0 and ci==0:ax.legend(fontsize=7,loc='best')
 fig.suptitle(f'P1 · one train sample · {head.DIAG_LABEL}',fontsize=13)
 path=head.OUT_FIGURES/'TINYFIT_P1_TRAJECTORIES.png';fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
 figures.append({'path':str(path.relative_to(ROOT)),'caption':'P1 final future trajectories for four model/loss cells; raw coordinates 0 and 14. The y axes include zero; the anchor is separate and not connected to future steps.','alt_text':'Eight panels compare the fitted 30-step train target with final TS11-D or STATE-CTRL predictions under FULL_LOSS or POS_ONLY for one P1 training sample. The native y axes include zero; the anchor is shown separately. Targets were used in fitting.'})
 p8_rows=sorted([r for r in panel_lock['rows'] if r['panel']=='P8'],key=lambda x:int(x['panel_order']))
 for config in ('TS11-D','STATE-CTRL'):
  full=predictions[f'TINYFIT-{config}-H30-FULL_LOSS-P8-S20261002'];pos=predictions[f'TINYFIT-{config}-H30-POS_ONLY-P8-S20261002']
  values=[0.0];values.extend(full['target'][:,:,0].ravel().tolist());values.extend(full['prediction'][:,:,0].ravel().tolist());values.extend(pos['prediction'][:,:,0].ravel().tolist())
  lo=min(values);hi=max(values);margin=max((hi-lo)*.06,1e-6)
  fig,axes=plt.subplots(2,4,figsize=(13,6.8),sharex=True,sharey=True,layout='constrained')
  for i,ax in enumerate(axes.flat):
   ax.axhline(0,color='#aaaaaa',linewidth=.6);ax.scatter([0],[float(full['anchor'][i,0])],marker='x',s=24,color=COLORS['ANCHOR'],label='anchor (separate seam)')
   ax.plot(h,full['target'][i,:,0],color=COLORS['TARGET'],linestyle=':',linewidth=1.5,label='target (used in fitting)')
   ax.plot(h,full['prediction'][i,:,0],color=COLORS[config],linestyle='-',marker='o',markevery=7,linewidth=1.3,label='FULL_LOSS')
   ax.plot(h,pos['prediction'][i,:,0],color=COLORS[config],linestyle='--',marker='s',markevery=7,linewidth=1.3,label='POS_ONLY')
   ax.set_ylim(lo-margin,hi+margin);ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(axis='y',alpha=.2);ax.set_title(f"P8 {i+1}: {p8_rows[i]['anchor_id']}",fontsize=8)
   if i//4==1:ax.set_xlabel('Future query index')
   if i%4==0:ax.set_ylabel('Native coordinate 0 (unit unspecified; zero shown)')
   if i==0:ax.legend(fontsize=7,loc='best')
  fig.suptitle(f'P8 · {config} · eight train samples · {head.DIAG_LABEL}',fontsize=13)
  path=head.OUT_FIGURES/f"TINYFIT_P8_{config.replace('-','_')}_TRAJECTORIES.png";fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
  figures.append({'path':str(path.relative_to(ROOT)),'caption':f'P8 final trajectories for {config}, raw coordinate 0. The shared native y axis includes zero; each target was used in fitting.','alt_text':f'Eight small multiples for {config}: each locked P8 train target is compared to final FULL_LOSS and POS_ONLY outputs. The native y axis includes zero and the anchor is separate. These are fitted train trajectories.'})
 return figures

def head_metric_figures(curves:list[dict[str,Any]],samples:list[dict[str,Any]],panel_lock:dict[str,Any])->list[dict[str,Any]]:
 figures=[]
 fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharex=True,layout='constrained')
 metrics=[('position_gain_vs_persistence_mean','Position error reduction vs Persistence',.90),('fd_gain_vs_persistence_mean','FD error reduction vs Persistence',.80),('tv_ratio_mean','Prediction / target TV ratio',None)]
 for col,(metric,title,threshold) in enumerate(metrics):
  vals=np.asarray([float(r[metric]) for r in curves if np.isfinite(float(r[metric]))],dtype=float)
  ymin=min(-.15,float(vals.min())-.08) if vals.size else -.15
  ymax=max(1.15,float(vals.max())+.08) if vals.size else 1.25
  if metric=='tv_ratio_mean':ymin=max(0.,min(ymin,float(vals.min())-.08));ymax=max(1.3,float(vals.max())+.08)
  for ri,panel in enumerate(('P1','P8')):
   ax=axes[ri,col]
   for config in ('TS11-D','STATE-CTRL'):
    for loss in ('FULL_LOSS','POS_ONLY'):
     q=sorted([r for r in curves if r['panel']==panel and r['config']==config and r['loss']==loss],key=lambda x:int(x['step']))
     color=COLORS[config];marker='o' if loss=='FULL_LOSS' else 's';style='-' if config=='TS11-D' else ':'
     ax.plot([int(x['step']) for x in q],[float(x[metric]) for x in q],color=color,linestyle=style,marker=marker,markevery=4,linewidth=1.5,markersize=3,label=f'{config} · {loss}')
   if threshold is not None:ax.axhline(threshold,color='#333333',linestyle=':',linewidth=1.2,label='predeclared threshold')
   else:ax.axhline(.8,color='#555555',linestyle=':',linewidth=1.,label='TV lower bound');ax.axhline(1.2,color='#777777',linestyle='-.',linewidth=1.,label='TV upper bound')
   ax.set_ylim(ymin,ymax);ax.grid(alpha=.2);ax.set_title(f'{panel}: {title}',fontsize=9)
   if col==0:ax.set_ylabel('Fraction / ratio')
   if ri==1:ax.set_xlabel('Optimizer updates')
   if ri==0 and col==0:ax.legend(fontsize=7,loc='best')
 fig.suptitle(f'Head learning curves · train targets used in fitting · {head.DIAG_LABEL}',fontsize=13)
 path=head.OUT_FIGURES/'TINYFIT_METRIC_CURVES.png';fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
 figures.append({'path':str(path.relative_to(ROOT)),'caption':'Position and FD error reductions relative to per-sample Persistence and future-internal TV ratio over 2000 updates. P1 and P8 appear in separate rows; horizontal lines mark predeclared diagnostic thresholds.','alt_text':'A 2 by 3 grid of train-only head learning curves for P1 and P8. Solid versus dashed lines distinguish FULL_LOSS and POS_ONLY; blue versus orange distinguish TS11-D and STATE-CTRL. Targets were used in fitting.'})
 p8_rows=sorted([r for r in panel_lock['rows'] if r['panel']=='P8'],key=lambda x:int(x['panel_order']))
 ids=[r['anchor_id'] for r in p8_rows];columns=[(c,l) for c in ('TS11-D','STATE-CTRL') for l in ('FULL_LOSS','POS_ONLY')]
 matrices={k:np.full((8,4),np.nan) for k in ('position_gain_vs_persistence','fd_gain_vs_persistence')}
 for j,(config,loss) in enumerate(columns):
  run_id=f'TINYFIT-{config}-H30-{loss}-P8-S20261002'
  for i,aid in enumerate(ids):
   row=next(x for x in samples if x['run_id']==run_id and x['anchor_id']==aid)
   matrices['position_gain_vs_persistence'][i,j]=float(row['position_gain_vs_persistence'])
   matrices['fd_gain_vs_persistence'][i,j]=float(row['fd_gain_vs_persistence'])
 fig,axes=plt.subplots(1,2,figsize=(10,5.2),layout='constrained')
 for ax,(key,title) in zip(axes,(('position_gain_vs_persistence','Position gain vs Persistence'),('fd_gain_vs_persistence','FD gain vs Persistence'))):
  matrix=matrices[key];vmin=min(-.2,float(np.nanmin(matrix)));vmax=max(1.,float(np.nanmax(matrix)));im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=vmin,vcenter=0.,vmax=vmax),aspect='auto',interpolation='nearest')
  ax.set_title(title);ax.set_xticks(range(4),[f'{c}\n{l}' for c,l in columns],fontsize=8);ax.set_yticks(range(8),[f'{i+1}. {x}' for i,x in enumerate(ids)],fontsize=7);ax.set_xlabel('Locked model / loss');ax.set_ylabel('Locked train sample')
  for i in range(8):
   for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7)
  fig.colorbar(im,ax=ax,label='Gain fraction; zero = Persistence error')
 fig.suptitle(f'P8 per-sample head gains · {head.DIAG_LABEL}',fontsize=12)
 path=head.OUT_FIGURES/'TINYFIT_P8_PER_SAMPLE_GAINS.png';fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
 figures.append({'path':str(path.relative_to(ROOT)),'caption':'Final update-2000 per-sample P8 position and FD gains versus Persistence across all four head cells; color is centered at zero.','alt_text':'Two heatmaps show eight locked P8 train samples by four model/loss cells. Each value is the per-sample normalized error reduction versus Persistence; zero means no improvement. Targets were used in fitting.'})
 return figures

def main()->int:
 head_runs,head_samples,extras=head.load_runs();head_curves=extras['curves'];head_predictions=extras['predictions'];panel_lock=extras['panel_lock']
 free_runs,free_samples,free_curves,free_predictions=load_freepath()
 # Re-assert the trigger on the actual stored head outcomes.
 p1=[x for x in head_runs if x['panel']=='P1'];p8=[x for x in head_runs if x['panel']=='P8']
 p1_all_fail=all(not x['strong_training_fit'] for x in p1);p8_all_fail=all(not x['strong_training_fit'] for x in p8)
 if p1_all_fail or not p8_all_fail:raise AssertionError('stored trigger decision no longer follows the locked P1-before-P8 rule')
 head.OUT_FIGURES=ROOT/'figures/FINAL_V5';head.OUT_FIGURES.mkdir(parents=True,exist_ok=True)
 head_figures=head_trajectory_figures(head_predictions,panel_lock)+head_metric_figures(head_curves,head_samples,panel_lock)
 extra_figures=freepath_figures(head_predictions,free_predictions,panel_lock,free_curves)
 figures=head_figures+extra_figures
 write_csv(head.OUT_RESULTS/'TINYFIT_FINAL_RUN_SUMMARY.csv',head_runs+free_runs)
 write_csv(head.OUT_RESULTS/'TINYFIT_FINAL_SAMPLE_METRICS.csv',head_samples+free_samples)
 write_csv(head.OUT_RESULTS/'TINYFIT_LEARNING_CURVES.csv',head_curves+free_curves)
 total_formal=sum(float(x['elapsed_seconds']) for x in head_runs+free_runs)
 report={
  'schema_version':1,'status':'PHASE_B_COMPLETE_WITH_TRIGGERED_FREEPATH_CONTROLS',
  'formal_head_runs':len(head_runs),'head_optimizer_updates':sum(int(x['updates']) for x in head_runs),
  'free_path_runs':len(free_runs),'free_path_optimizer_updates':sum(int(x['updates']) for x in free_runs),
  'total_head_and_free_path_updates':sum(int(x['updates']) for x in head_runs+free_runs),
  'actual_cpu_wall_seconds_head_and_free_path':total_formal,'device':'cpu','torch_threads':1,
  'gpu_hours':0,'gpu_queries':0,'gpu_operations':0,
  'P1_triggered':p1_all_fail,'P8_triggered':p8_all_fail,'free_path_panel':'P8','free_path_trigger_lock_sha256':sha_file(ROOT/'protocol/B_FREE_PATH_TRIGGER_LOCK.json'),
  'free_path_both_losses_strong_fit_all_8':all(x['strong_training_fit'] for x in free_runs),
  'free_path_runs_are_forecasting_models':False,
  'head_strong_P1':[x['run_id'] for x in p1 if x['strong_training_fit']],
  'head_strong_P8':[x['run_id'] for x in p8 if x['strong_training_fit']],
  'sample_rows':len(head_samples+free_samples),'learning_curve_rows':len(head_curves+free_curves),
  'figures':figures,'protocol_sha256':sha_file(ROOT/'protocol/B_PROTOCOL_LOCKED.json'),
  'panel_lock_sha256':sha_file(ROOT/'protocol/B_PANEL_SELECTION_LOCK.json'),'matrix_sha256':sha_file(ROOT/'protocol/B_MATRIX_8.csv'),
  'source_locks':{'head':sha_file(ROOT/'protocol/B_EXECUTION_SOURCE_LOCK.json'),'free_path':sha_file(ROOT/'protocol/B_FREE_PATH_SOURCE_LOCK.json')},
  'scope':{'train_targets_used_in_fitting':True,'dev_targets_opened':False,'dev_model_selection':False,'final_or_heldout_access':0,'new_benchmark_candidate':False,'D5_training':False},
  'free_path_interpretation':'Both independent-output controls fit the eight locked train trajectories under both loss arms. This shows the objectives/optimizer can fit those targets when every output is unconstrained and independent; it does not identify whether head failures arise from shared representation, query conditioning, cross-sample interference, or another optimizer-path interaction.'}
 write_json(ROOT/'B_FINAL_RESULTS.json',report)
 lines=[
 '# Core B：近静态取证与 train-only tiny-fit 结果','',
 '状态：Phase A、8/8 个 head tiny-fit 与触发的 2/2 个 free-path 正对照完成。所有 tiny-fit 与 free-path 的未来目标都参与了对应 run 的优化；下列结果只回答固定 train 面板的可拟合性，不是 dev/test 预测、新 benchmark 候选或泛化证据。','',
 '## 结论摘要','',
 f"- Head 矩阵：{len(head_runs)}/8 格、{sum(int(x['updates']) for x in head_runs):,} optimizer updates；P1 的强拟合格：{', '.join(x['config']+' / '+x['loss'] for x in p1 if x['strong_training_fit']) or '无'}。P8 没有 head cell 满足逐样本 strong-fit 三条件。",
 f"- P8 free-path 触发是合约条件的结果：P1 四格并非全失败；P8 四格均未达 strong-fit，因此在同一 P8 上运行恰好两个控制。FULL_LOSS 与 POS_ONLY 都在 8/8 个样本上达标。",
 f"- Head 与 free-path 实际 fit 用时合计 {total_formal/60:.1f} 分钟；CPU 单线程；GPU 使用 0。dev targets、final/held-out、D5 均未访问。",
 '- Free-path 是每个样本×未来时间×raw action coordinate 的独立可训练参数，初始化全零；它不是输入条件预测模型，也不能作为 forecasting performance。它成功说明在该固定 P8 与损失/优化设置下，优化器和目标尺度可以把自由输出拟合到训练目标；它不能单独区分共享 head 容量、query conditioning、跨样本干扰或其他训练路径问题。','',
 '## Head 八格末步汇总（update 2000）','',
 '| 模型 | Loss | Panel | 位置误差降低 | FD误差降低 | TV ratio | 强拟合 | 通过数 |','|---|---|---:|---:|---:|---:|---|---:|']
 for x in head_runs:lines.append(f"| {x['config']} | {x['loss']} | {x['panel']} | {x['position_gain_vs_persistence_mean']:.3f} | {x['fd_gain_vs_persistence_mean']:.3f} | {x['prediction_tv_over_target_tv_mean']:.3f} | {'是' if x['strong_training_fit'] else '否'} | {x['per_sample_strong_fit_count']}/{x['panel_rows']} |")
 lines += ['', '## P8 free-path 正对照（非预测模型）','', '| Loss | 位置误差降低 | FD误差降低 | TV ratio | 强拟合 | 通过数 |','|---|---:|---:|---:|---|---:|']
 for x in free_runs:lines.append(f"| {x['loss']} | {x['position_gain_vs_persistence_mean']:.3f} | {x['fd_gain_vs_persistence_mean']:.3f} | {x['prediction_tv_over_target_tv_mean']:.3f} | {'是' if x['strong_training_fit'] else '否'} | {x['per_sample_strong_fit_count']}/{x['panel_rows']} |")
 lines += ['', '## 解读限制与下一步','',
 'Strong-fit 标签采用每条样本自身 Persistence 作分母，并要求 panel 中每个样本同时达到 position ≤10%、safe-FD ≤20%、TV ratio∈[0.8,1.2]；图和 CSV 同时保留均值与逐样本值。P1 是一条高运动 train 轨迹，P8 是预先锁定的 8 条不同 source group/train 轨迹，均用于优化。单 seed、重复拟合后的训练数据不能估计泛化、稳健性或置信区间。',
 'Phase A 的 query 向量无精确重复，但相邻 cosine≈0.999965、FP32 尺度有效秩 H30=4/30、H60=5/60；这只是 conditioning 风险。P1 的 STATE-CTRL/FULL_LOSS 能拟合，说明共享 head 在一种条件下可记忆单条轨迹；P8 head 全部未通过但 free-path 两 loss 全通过，说明存在需要继续区分的 shared-head/conditioning/interference 路径，不唯一指向表达能力或 loss 根因。',
 '建议停止新增训练格。唯一后续工作建议为只读核对 TS11-D 动态视频 token 到 decoder 的实际尺度/梯度路径，并据此冻结一份独立评价前的数据、目标、指标和查询编码合同；没有可证伪的实现问题时，不再追加 query-scale、LR 或 seed sweep。D5 继续不训练，action 的 raw index slices 不赋 Hand/Arm 语义。','',
 '## 图表说明','']
 for f in figures:lines.append(f"- `{f['path']}` — {f['caption']} Alt text: {f['alt_text']}")
 lines += ['', '逐样本末步结果：`results/TINYFIT_FINAL_SAMPLE_METRICS.csv`；全部 head/free-path 100-step 曲线：`results/TINYFIT_LEARNING_CURVES.csv`；各 run 的 checkpoint、预测、完整指标与 `RUN_COMPLETE.json` 位于相应 `runs/` 子目录。','']
 (ROOT/'B_FINAL_REPORT.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps({'status':report['status'],'head_runs':len(head_runs),'free_path_runs':len(free_runs),'freepath_fits':[x['loss'] for x in free_runs if x['strong_training_fit']],'actual_cpu_minutes':round(total_formal/60,1),'figures':len(figures),'report_sha256':sha_file(ROOT/'B_FINAL_REPORT.zh-CN.md')},sort_keys=True))
 return 0


if __name__=='__main__':raise SystemExit(main())
