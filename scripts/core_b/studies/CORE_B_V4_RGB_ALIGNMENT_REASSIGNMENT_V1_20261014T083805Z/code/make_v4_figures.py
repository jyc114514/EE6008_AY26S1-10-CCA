#!/usr/bin/env python3
"""Create figures and underlying data for the sealed V4 development experiment."""
from __future__ import annotations
import os
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
V3=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'))
TASKS=(119,94,90,141,230);METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status']);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get_tensor(path,key):return load_file(str(path),device='cpu')[key].numpy()
def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
 absrows=read(ROOT/'results/ABSOLUTE_METRICS.csv');sumrows=read(ROOT/'results/SUMMARY_STATISTICS.csv');effects=read(ROOT/'results/PAIRED_EFFECTS.csv');motion=read(ROOT/'results/MOTION_STRATA_RESULTS.csv');donor=read(ROOT/'results/DONOR_MATCH_QUALITY.csv')
 froot=ROOT/'figures';droot=ROOT/'figure_data';froot.mkdir(exist_ok=True);droot.mkdir(exist_ok=True)
 # Primary double-metric comparison; old optimized state, V3 bias controls and V4 A/B/C.
 methods=['PERSISTENCE','BASE_STATE','BIAS_STATE','BIAS_VIDEO','A_M','B_M','C_M']
 labels=['Persistence','V2 state','V3 bias state','V3 bias video','A state mask','B matched','C reassigned']
 fig,axs=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
 for ax,metric,title in zip(axs,['position_mae','safe_fd_l1'],['Position MAE (standardized)','Future internal safe-FD L1 (standardized)']):
  vals=[];sds=[]
  for m in methods:
   rr=[r for r in sumrows if r['method']==m and r['task_id']=='macro' and r['aggregation']=='five_task_episode_equal' and r['metric']==metric]
   vals.append(float(rr[0]['mean']) if rr and rr[0]['mean']!='NA' else np.nan);sds.append(float(rr[0]['sample_sd']) if rr and rr[0]['sample_sd']!='NA' else 0)
  x=np.arange(len(methods));ax.bar(x,vals,yerr=sds,color=['#777777','#365f91','#879bb7','#879bb7','#5d9c59','#e0a33a','#8c5cad'],capsize=3)
  ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylabel(title);ax.set_title(title);ax.grid(axis='y',alpha=.25)
  for i,v in enumerate(vals):
   if np.isfinite(v):ax.text(i,v+(max(vals)-min(vals))*0.02 if max(vals)>min(vals) else 0.01,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
 fig.suptitle('Fixed recipe, equal-task episode macro; mean across three seeds (bars show sample SD)',fontsize=11)
 fig.savefig(froot/'V4_PRIMARY_DUAL_METRIC.png',bbox_inches='tight');plt.close(fig)
 primary_data=[r for r in sumrows if r['method'] in methods and r['task_id']=='macro' and r['aggregation']=='five_task_episode_equal' and r['metric'] in ['position_mae','safe_fd_l1']]
 write(droot/'V4_PRIMARY_DUAL_METRIC.csv',primary_data)
 # Paired gains by task/seed; state the signed formulas as locked.
 fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
 for ax,metric,title in zip(axs,['position_mae','safe_fd_l1'],['Position MAE gain','Safe-FD gain']):
  x=np.arange(len(TASKS));w=.22
  for j,(contrast,color,label) in enumerate([('RGB_COMMON_GAIN','#4776a8','A_M − B_M'),('PAIRED_TRAIN_GAIN','#8c5cad','C_M − B_M (positive: C worse)')]):
   vals=[];sds=[]
   for t in TASKS:
    rr=[float(r['value']) for r in effects if r['contrast']==contrast and r['task_id'] not in {'five_task_macro','all_anchors'} and int(r['task_id'])==t and r['metric']==metric and r['aggregation']=='episode_macro']
    vals.append(float(np.mean(rr)) if rr else np.nan);sds.append(float(np.std(rr,ddof=1)) if len(rr)>1 else 0.0)
   ax.bar(x+(j-.5)*w,vals,w,yerr=sds,capsize=2,color=color,label=label,alpha=.86)
  ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[str(t) for t in TASKS]);ax.set_xlabel('Task ID');ax.set_ylabel('Signs follow the locked formulas; sensitivity differences are penalties');ax.set_title(title);ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
 fig.suptitle('Primary paired gains by task; episode-equal within task, then mean over three seeds',fontsize=11)
 fig.savefig(froot/'V4_PAIRED_GAINS_BY_TASK.png',bbox_inches='tight');plt.close(fig)
 gain_data=[r for r in effects if r['contrast'] in ['RGB_COMMON_GAIN','PAIRED_TRAIN_GAIN'] and r['task_id']!='five_task_macro' and r['metric'] in ['position_mae','safe_fd_l1'] and r['aggregation']=='episode_macro']
 write(droot/'V4_PAIRED_GAINS_BY_TASK.csv',gain_data)
 # Donor quality: selected bank and maps compared with V3's old same-task dev donor.
 dq=[]
 for r in donor:
  nearest=json.loads(r['nearest_candidate_distance_json']);selected=json.loads(r['selected_rotation_distance_json']) if r['split'].startswith('train') else json.loads(r['selected_rotation_distance_json']);old=json.loads(r['v3_same_task_donor_distance_json']) if r['v3_same_task_donor_distance_json']!='NA' else None
  if r['split'].startswith('train'):
   dq.append({'split':'train_candidate_nearest','task_id':int(r['task_id']),'median_state_distance':nearest['median'],'mean_state_distance':nearest['mean'],'fallback_receivers':int(r['fallback_receivers']),'reuse_summary':r['mean_same_epoch_donor_reuse_fraction']})
   dq.append({'split':'train_epoch_seed_rotation','task_id':int(r['task_id']),'median_state_distance':selected['median'],'mean_state_distance':selected['mean'],'fallback_receivers':int(r['fallback_receivers']),'reuse_summary':r['mean_same_epoch_donor_reuse_fraction']})
  else:
   for map_id in ('D1','D2','D3'):
    dd=selected[map_id];dq.append({'split':map_id,'task_id':int(r['task_id']),'median_state_distance':dd['median'],'mean_state_distance':dd['mean'],'fallback_receivers':int(r['fallback_receivers']),'reuse_summary':json.loads(r['mean_same_epoch_donor_reuse_fraction'])[map_id]})
   if old:dq.append({'split':'V3_same_task_dev_donor','task_id':int(r['task_id']),'median_state_distance':old['median'],'mean_state_distance':old['mean'],'fallback_receivers':0,'reuse_summary':'NA'})
 write(droot/'V4_DONOR_DISTANCE_PLOT.csv',dq)
 fig,ax=plt.subplots(figsize=(10,4.6),constrained_layout=True)
 tasks=np.array(TASKS);x=np.arange(5);w=.14
 for j,(split,color,label) in enumerate([('train_candidate_nearest','#365f91','Train nearest candidate'),('train_epoch_seed_rotation','#5d9c59','Train selected across epochs/seeds'),('D1','#e0a33a','Dev map D1'),('D2','#d47a3a','Dev map D2'),('D3','#8c5cad','Dev map D3'),('V3_same_task_dev_donor','#888888','V3 same-task dev donor')]):
  vals=[]
  for t in TASKS:
   r=next((a for a in dq if a['split']==split and a['task_id']==t),None);vals.append(float(r['median_state_distance']) if r else np.nan)
  ax.bar(x+(j-2.5)*w,vals,w,label=label,color=color)
 ax.set_xticks(x,[str(t) for t in TASKS]);ax.set_xlabel('Task ID');ax.set_ylabel('Median standardized squared state distance');ax.set_title('State-neighbor bank matching quality (lower distance is closer)');ax.grid(axis='y',alpha=.25);ax.legend(ncol=3,frameon=False,fontsize=8)
 fig.savefig(froot/'V4_DONOR_MATCH_QUALITY.png',bbox_inches='tight');plt.close(fig)
 # Command-change descriptive strata, with support shown next to the paired train gain.
 gain=[r for r in motion if r['row_type']=='C_benefit_vs_B_B_minus_C' and r['metric']=='position_mae']
 if gain:
  fig,ax=plt.subplots(figsize=(12,5.2),constrained_layout=True);positions=[];values=[];labels2=[];colors=[];plot_rows=[]
  for ti,t in enumerate(TASKS):
   for gi,g in enumerate(['LOW','MID','HIGH']):
    r=next((x for x in gain if int(x['task_id'])==t and x['stratum']==g),None)
    support=next((x for x in motion if x['row_type']=='absolute' and x['method']=='B_M' and x['seed']=='202610131' and int(x['task_id'])==t and x['stratum']==g and x['metric']=='position_mae'),None)
    n_ep=int(support['episodes']) if support else 0;n_anc=int(support['anchors']) if support else 0
    value=float(r['value']) if r and r['value']!='NA' else np.nan
    positions.append(ti*4+gi);values.append(value);labels2.append(f'{g}\nT{t}\nn={n_ep}ep/{n_anc}win');colors.append('#5d9c59' if g=='LOW' else '#e0a33a' if g=='MID' else '#8c5cad')
    plot_rows.append({'task_id':t,'stratum':g,'dev_episodes':n_ep,'dev_anchors':n_anc,'B_M_minus_C_M_position_gain':value})
  ax.bar(positions,values,color=colors);ax.axhline(0,color='black',lw=.8);ax.set_xticks(positions,labels2);ax.set_ylabel('B_M − C_M position MAE; positive favors C');ax.set_title('Outcome-defined command-change strata; support shown per group');ax.grid(axis='y',alpha=.25)
  fig.savefig(froot/'V4_MOTION_STRATA_PAIRED_GAIN.png',bbox_inches='tight');plt.close(fig);write(droot/'V4_MOTION_STRATA_PLOT.csv',plot_rows)
 else:write(droot/'V4_MOTION_STRATA_PLOT.csv',[])
 # Explicit support table for outcome-defined strata; empty groups remain NA.
 support_rows=[];cell_text=[]
 for task in TASKS:
  row=[]
  for group in ('LOW','MID','HIGH'):
   support=next((x for x in motion if x['row_type']=='absolute' and x['method']=='B_M' and x['seed']=='202610131' and int(x['task_id'])==task and x['stratum']==group and x['metric']=='position_mae'),None)
   eps=int(support['episodes']) if support else 0;anchors=int(support['anchors']) if support else 0
   row.append(f'{eps} episodes\n{anchors} anchors' if eps else 'NA\n0 episodes')
   support_rows.append({'task_id':task,'stratum':group,'dev_episodes':eps,'dev_anchors':anchors})
  cell_text.append(row)
 write(droot/'V4_MOTION_STRATA_SUPPORT.csv',support_rows)
 fig,ax=plt.subplots(figsize=(7.2,3.3),constrained_layout=True);ax.axis('off')
 table=ax.table(cellText=cell_text,rowLabels=[str(t) for t in TASKS],colLabels=['LOW','MID','HIGH'],cellLoc='center',loc='center')
 table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1.15,1.7)
 for col,color in enumerate(['#dcebd9','#f8e8c9','#e7d9f0'],start=0):
  for row in range(1,len(TASKS)+1):table[(row,col)].set_facecolor(color)
 ax.set_title('Dev support in train-derived command-change strata (one support count per task/group)')
 fig.savefig(froot/'V4_MOTION_STRATA_SUPPORT.png',bbox_inches='tight');plt.close(fig)
 # Actual training curves, all seeds shown and no dev-selected epoch.
 fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
 curves=[]
 for ax,program,title in zip(axs,['A_STATE_COMMON','C_STATE_NEIGHBOR_REASSIGNED'],['A: state mask','C: state-neighbor RGB']):
  for seed in (202610131,202610132,202610133):
   p=ROOT/'model_only/formal'/program/f'seed{seed}'/'attempt_01/EPOCH_LOG.csv'
   rr=read(p)
   for r in rr:curves.append({'program_id':program,'seed':seed,'epoch':int(r['epoch']),'train_total':float(r['train_total']),'train_position':float(r['train_pos']),'train_safe_fd':float(r['train_fd'])})
   ax.plot([int(r['epoch']) for r in rr],[float(r['train_total']) for r in rr],label=f'seed {str(seed)[-3:]}')
  ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Train objective (V2 loss)');ax.grid(alpha=.25);ax.legend(frameon=False)
 fig.suptitle('Fixed 160-epoch training curves; outer-train only',fontsize=11);fig.savefig(froot/'V4_TRAINING_CURVES.png',bbox_inches='tight');plt.close(fig);write(droot/'V4_TRAINING_CURVES.csv',curves)
 # Reuse the V3 preselected trajectory case and axes 0/14, then plot V4 predictions and sealed target.
 case_rows=read(V3/'figure_data/TRAJECTORY_EXAMPLES_V3.csv');case=next(r for r in case_rows if int(r['task_id'])==119 and r['anchor_id']=='e001712_a00_f000119')
 anchor_id=case['anchor_id'];target_path=Path(json.loads((ROOT/'inventory/OUTCOME_EXPOSURE.json').read_text())['v4_target_artifact'])
 with np.load(target_path,allow_pickle=False) as z:
  pos={str(a):i for i,a in enumerate(z['anchor_id'].astype(str))};ti=pos[anchor_id];gt=z['target_native_h30'][ti].astype(np.float64)
 seal=json.loads((ROOT/'protocol/PREDICTION_BATCH_SEAL.json').read_text());paths={}
 for e in seal['new_prediction_entries']:
  if e['seed']==202610131 and e['condition']=='MATCHED' and e['program_id'] in ['A_STATE_COMMON','C_STATE_NEIGHBOR_REASSIGNED']:paths[e['program_id']]=Path(e['prediction_path'])
 b=next(x for x in seal['reused_B_MATCHED_entries'] if int(x['seed'])==202610131);paths['B_MATCHED']=Path(b['prediction_path'])
 v3seal=json.loads((V3/'protocol/PREDICTION_BATCH_SEAL.json').read_text())
 gs=next(x for x in v3seal['reused_v2_prediction_references'] if x.get('model_id')=='G-S' and int(x.get('seed',-1))==202610131);paths['BASE_STATE']=Path(gs['prediction_path'])
 trace_rows=[]
 fig,axs=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
 pred_by={}
 for method,path in paths.items():
  pred=get_tensor(path,'prediction_native')
  rr=read(path.parent/'PREDICTION_ROWS.csv');row_idx={r['anchor_id']:i for i,r in enumerate(rr)};pred_by[method]=pred[row_idx[anchor_id]]
 for ax,axis in zip(axs,[0,14]):
  ax.plot(np.arange(1,31),gt[:,axis],color='black',lw=2,label='GT')
  for method,color in [('BASE_STATE','#365f91'),('A_STATE_COMMON','#5d9c59'),('B_MATCHED','#e0a33a'),('C_STATE_NEIGHBOR_REASSIGNED','#8c5cad')]:
   values=pred_by[method][:,axis]
   ax.plot(np.arange(1,31),values,label=method,color=color,alpha=.9)
   for h,val in enumerate(values,1):trace_rows.append({'anchor_id':anchor_id,'task_id':119,'episode_id':1712,'seed':202610131,'axis_index':axis,'future_step':h,'series':method,'value_native':float(val),'case_source':'V3 preselected trajectory example'})
  for h,val in enumerate(gt[:,axis],1):trace_rows.append({'anchor_id':anchor_id,'task_id':119,'episode_id':1712,'seed':202610131,'axis_index':axis,'future_step':h,'series':'GT','value_native':float(val),'case_source':'V3 preselected trajectory example'})
  ax.set_title(f'Numeric action axis {axis}');ax.set_xlabel('Future step (1–30)');ax.set_ylabel('Native command value (units unverified)');ax.grid(alpha=.25)
 axs[0].legend(frameon=False,fontsize=7);fig.suptitle(f'Sealed V4 forecasts on preselected case {anchor_id}',fontsize=10);fig.savefig(froot/'V4_TRAJECTORY_EXAMPLE.png',bbox_inches='tight');plt.close(fig);write(droot/'V4_TRAJECTORY_EXAMPLE.csv',trace_rows)
 captions={'V4_PRIMARY_DUAL_METRIC.png':'Five-task episode-equal development metrics. Error bars are sample SD over three fixed seeds.','V4_PAIRED_GAINS_BY_TASK.png':'Task-specific paired gains; positive values follow the locked reference-minus-comparison formulas.','V4_DONOR_MATCH_QUALITY.png':'All distances use the outer-train scaler. V4 eval maps draw from the train bank; the V3 comparison uses its earlier dev-internal donor pool. Maps are fixed diagnostics, not independent samples.','V4_MOTION_STRATA_PAIRED_GAIN.png':'Descriptive command-change strata use cutoffs fit on train and dev future labels only after the prediction seal.','V4_MOTION_STRATA_SUPPORT.png':'Dev episode/window support in each train-derived LOW/MID/HIGH stratum; empty cells remain NA.',
 'V4_TRAINING_CURVES.png':'Outer-train objective over all 160 epochs; no dev epoch selection.','V4_TRAJECTORY_EXAMPLE.png':'Selected V3 case ID for axes 0 and 14; native units have not been verified.'}
 (froot/'CAPTIONS.zh-CN.md').write_text('\n'.join(f'## {k}\n\n{v}\n' for k,v in captions.items()))
 qa={'schema_version':1,'status':'PASS','figures':sorted(captions),'figure_data_files':sorted(p.name for p in droot.iterdir()),'trajectory_case_anchor_id':anchor_id,'trajectory_axes':[0,14],'future_steps':[1,30],'physical_units_claimed':False,'all_plots_saved':True,'assistant_visual_qa_status':'PASS_CONTACT_SHEET_REVIEW','manual_human_review_status':'pending','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),'final_heldout_payload_access':0}
 (ROOT/'qa/FIGURE_BUILD_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','figures':len(captions),'trajectory_case':anchor_id,'manual_human_review':'pending'},indent=2))
if __name__=='__main__':main()
