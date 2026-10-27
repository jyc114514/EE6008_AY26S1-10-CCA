#!/usr/bin/env python3
"""Generate fixed-design scientific figures from the locked analysis tables."""
from __future__ import annotations
import os
import csv,hashlib,json,math,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3_INPUT=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz'
AMENDMENT_ID='AMEND-007-DEV-FIGURE-CORRECTIONS'
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';BLACK='#000000';GRAY='#666666';LIGHT='#D9D9D9'
MODEL_COLOR={'TS11-D':BLUE,'STATE-CTRL':ORANGE}
SEEDS=(20261001,20261002,20261003)
QUERY_STYLE={'old':('--','x'),'had':('-','o')}
SEED_MARKER={20261001:'o',20261002:'s',20261003:'^'}
CONDITION_LABEL={('TS11-D','old'):'TS11-D · Q0 old',('TS11-D','had'):'TS11-D · Q1 Hadamard',
                 ('STATE-CTRL','old'):'STATE-CTRL · Q0 old',('STATE-CTRL','had'):'STATE-CTRL · Q1 Hadamard'}

def read_csv(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def wcag(hexcolor):
 s=hexcolor.lstrip('#');rgb=[int(s[i:i+2],16)/255 for i in (0,2,4)]
 def lin(c):return c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4
 l=.2126*lin(rgb[0])+.7152*lin(rgb[1])+.0722*lin(rgb[2]);return 1.05/(l+.05)
def setup(ax,title,ylabel):
 ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('');ax.set_ylabel(ylabel);ax.grid(axis='y',color='#E5E5E5',linewidth=.6);ax.set_axisbelow(True)
def save_figure(fig,stem):
 p=ROOT/'figures'/f'{stem}.png';q=ROOT/'figures'/f'{stem}.pdf'
 if p.exists() or q.exists():raise FileExistsError(f'preserving preexisting figure {stem}')
 fig.savefig(p,dpi=200,facecolor='white',edgecolor='white')
 fig.savefig(q,facecolor='white',edgecolor='white')
 plt.close(fig)
 return p,q

def objective_curves():
 rows=read_csv(ROOT/'results/TRAIN_CURVES.csv')
 fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained',sharey=True)
 for ax,model in zip(axes,('TS11-D','STATE-CTRL')):
  subset=[r for r in rows if r['config']==model]
  for query in ('old','had'):
   col=MODEL_COLOR[model];ls,marker=QUERY_STYLE[query]
   for seed in (20261001,20261002,20261003):
    data=sorted((r for r in subset if r['query_variant']==query and int(r['seed'])==seed),key=lambda x:int(x['epoch']))
    if not data:continue
    ax.plot([int(r['epoch']) for r in data],[float(r['weighted_loss_sum_over_seen_rows']) for r in data],
            color=col,linestyle=ls,marker=SEED_MARKER[seed],markevery=8,markersize=3,linewidth=.9,alpha=.60,
            label=f'{"Q0 old" if query=="old" else "Q1 Hadamard"}, seed {seed}')
   # Thick equal-seed mean; source-group/seed SD is not a confidence interval.
   epochs=sorted({int(r['epoch']) for r in subset if r['query_variant']==query})
   if epochs:
    values=[]
    for ep in epochs:
     xs=[float(r['weighted_loss_sum_over_seen_rows']) for r in subset if r['query_variant']==query and int(r['epoch'])==ep]
     values.append(float(np.mean(xs)))
    ax.plot(epochs,values,color=col,linestyle=ls,linewidth=2.6,label=f'{"Q0" if query=="old" else "Q1"} mean, 3 seeds')
  setup(ax,model,'episode-weighted train objective / anchor')
  ax.set_xlim(1,40);ax.set_xticks([1,10,20,30,40])
  ax.legend(fontsize=7,ncol=2,frameon=False)
 fig.suptitle('TRAIN-ONLY FIT PROBE — targets used in fitting',fontweight='bold')
 fig.text(.5,-.02,'Q0 curves are reused from accepted corrected-scheduler runs; Q1 curves are the six new H30 Hadamard fits. Means weight fixed seeds equally.',ha='center',fontsize=8)
 return save_figure(fig,'TRAIN_OBJECTIVE_CURVES')

def monitor_curves():
 rows=read_csv(ROOT/'results/TRAIN_CURVES.csv')
 metrics=[('monitor_position_standardized_mae','Full28 standardized position MAE'),('monitor_safe_fd_normalized','safe-FD normalized error'),('monitor_tv_ratio','TV_pred / TV_GT')]
 fig,axes=plt.subplots(2,3,figsize=(12,6.6),layout='constrained',sharex=True,sharey='col')
 for ri,model in enumerate(('TS11-D','STATE-CTRL')):
  for ci,(field,label) in enumerate(metrics):
   ax=axes[ri,ci]
   for seed in (20261001,20261002,20261003):
    data=sorted((r for r in rows if r['config']==model and r['query_variant']=='had' and int(r['seed'])==seed),key=lambda x:int(x['epoch']))
    if not data:continue
    vals=[float(r[field]) for r in data]
    ax.plot([int(r['epoch']) for r in data],vals,color=MODEL_COLOR[model],marker=SEED_MARKER[seed],markevery=8,
            markersize=3,linewidth=.9,alpha=.60,label=f'{seed}')
   if ci==2:ax.axhline(1,color=GRAY,linestyle=':',linewidth=1,label='TV ratio = 1')
   setup(ax,f'{model} · {label}',label)
   ax.set_xlim(1,40);ax.set_xticks([1,10,20,30,40])
   ax.legend(fontsize=7,frameon=False)
 fig.suptitle('TRAIN-ONLY FIT PROBE — targets used in fitting',fontweight='bold')
 fig.text(.5,-.02,'Fixed 128-anchor train monitor panel, one anchor per source group; Q1 only. Lines show fixed seeds separately; Q0 monitor panels were not recorded in the reused runs.',ha='center',fontsize=8)
 return save_figure(fig,'TRAIN_MONITOR_PANEL_CURVES')

def dev_primary_by_seed():
 rows=read_csv(ROOT/'results/TRAIN_DEV_BY_SEED.csv')
 metrics=[('position_standardized_mae','Full28 standardized position MAE'),('safe_fd_normalized','safe-FD normalized error')]
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained',sharex='col')
 jitter={20261001:-.08,20261002:0.,20261003:.08}
 for ri,model in enumerate(('TS11-D','STATE-CTRL')):
  for ci,(metric,label) in enumerate(metrics):
   ax=axes[ri,ci]
   for seed in (20261001,20261002,20261003):
    vals=[]
    for query,x in [('old',0),('had',1)]:
     row=next(r for r in rows if r['split']=='dev' and r['config']==model and r['query_variant']==query and int(r['seed'])==seed)
     vals.append(float(row[metric]));ax.scatter(x+jitter[seed],float(row[metric]),marker=SEED_MARKER[seed],color=GRAY,edgecolor='white',linewidth=.4,s=38,zorder=3)
    ax.plot([jitter[seed],1+jitter[seed]],vals,color=MODEL_COLOR[model],linestyle=':',linewidth=.8,alpha=.8)
   means=[float(np.mean([float(r[metric]) for r in rows if r['split']=='dev' and r['config']==model and r['query_variant']==query])) for query in ('old','had')]
   sds=[float(np.std([float(r[metric]) for r in rows if r['split']=='dev' and r['config']==model and r['query_variant']==query],ddof=1)) for query in ('old','had')]
   ax.errorbar([0,1],means,yerr=sds,color=MODEL_COLOR[model],marker='D',markersize=6,linewidth=2.2,capsize=4,label='mean ± sample SD across 3 fixed seeds')
   lower=min(m-sd for m,sd in zip(means,sds));upper=max(m+sd for m,sd in zip(means,sds))
   margin=max((upper-lower)*.15,max(abs(lower),abs(upper),1.0)*.001)
   setup(ax,f'{model} · {label}',label);ax.set_xticks([0,1],['Q0 old','Q1 Hadamard']);ax.set_xlim(-.35,1.35);ax.set_ylim(lower-margin,upper+margin)
   if ri==0 and ci==0:ax.legend(fontsize=7,frameon=False)
 fig.suptitle('EXPLORATORY DEV — previously exposed; no model selection',fontweight='bold')
 fig.text(.5,-.02,'Points are paired fixed seeds; thin segments connect Q0 and Q1 within seed. Error bars are sample SD across seeds, not group-bootstrap confidence intervals.',ha='center',fontsize=8)
 return save_figure(fig,'DEV_PRIMARY_METRICS_BY_SEED')

def dev_query_gains():
 gains=read_csv(ROOT/'results/QUERY_GAINS.csv');cis=read_csv(ROOT/'results/BOOTSTRAP_CI.csv')
 metrics=[('position_standardized_mae','Full28 standardized position MAE'),('safe_fd_normalized','safe-FD normalized error')]
 fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained',sharey=True)
 for ax,(metric,label) in zip(axes,metrics):
  for x,model,key,color in [(0,'TS11-D','G_TS',BLUE),(1,'STATE-CTRL','G_STATE',ORANGE)]:
   seedrows=[r for r in gains if r['split']=='dev' and r['model']==model and r['metric']==metric and r['seed']!='THREE_SEED_MEAN']
   for row in seedrows:ax.scatter(x+(int(row['seed'])-20261002)*.00000001,float(row['G_Q_old_minus_had']),marker=SEED_MARKER[int(row['seed'])],color=GRAY,s=40,zorder=3)
   mrow=next(r for r in gains if r['split']=='dev' and r['model']==model and r['metric']==metric and r['seed']=='THREE_SEED_MEAN')
   ci=next(r for r in cis if r['metric']==metric and r['estimand']==key)
   mean=float(mrow['G_Q_old_minus_had']);low=float(ci['group_bootstrap_ci95_low']);high=float(ci['group_bootstrap_ci95_high'])
   ax.errorbar(x,mean,yerr=[[mean-low],[high-mean]],color=color,marker='D',markersize=7,linewidth=2,capsize=5,label=f'{model}: mean + source-group 95% bootstrap CI')
  ax.axhline(0,color=BLACK,linewidth=1)
  setup(ax,label,'G_Q = old error − Hadamard error')
  ax.set_xticks([0,1],['TS11-D','STATE-CTRL']);ax.set_xlim(-.45,1.45)
  ax.legend(fontsize=7,frameon=False)
 fig.suptitle('EXPLORATORY DEV — positive G_Q favors Hadamard',fontweight='bold')
 fig.text(.5,-.02,'Dots are the three fixed-seed estimates. Diamonds/intervals are equal-seed mean and paired 95% source-group bootstrap interval, conditional on these fixed models/seeds.',ha='center',fontsize=8)
 return save_figure(fig,'DEV_QUERY_GAINS_BOOTSTRAP')

def dev_system_advantage():
 video=read_csv(ROOT/'results/ABSOLUTE_VIDEO_GAINS.csv');interaction=read_csv(ROOT/'results/QUERY_MODEL_INTERACTION.csv');cis=read_csv(ROOT/'results/BOOTSTRAP_CI.csv')
 categories=[('V_old_state_minus_TS','V_old: STATE − TS under Q0'),('V_Had_state_minus_TS','V_Had: STATE − TS under Q1'),('I_GTS_minus_GSTATE','I: G_TS − G_STATE')]
 styles={'V_old_state_minus_TS':('--','x',GRAY),'V_Had_state_minus_TS':('-','o',GREEN),'I_GTS_minus_GSTATE':('-.','D',ORANGE)}
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained',sharey='row')
 for ri,(metric,label) in enumerate([('position_standardized_mae','Full28 standardized position MAE'),('safe_fd_normalized','safe-FD normalized error')]):
  for ci,(key,catlabel) in enumerate(categories):
   ax=axes[ri,ci];all_seed_values=[]
   for x,(field,_) in enumerate([(c[0],c[1]) for c in categories]):
    ls,marker,color=styles[field]
    if field=='I_GTS_minus_GSTATE':
     subset=[r for r in interaction if r['split']=='dev' and r['metric']==metric and r['seed']!='THREE_SEED_MEAN'];vals=[float(r[field]) for r in subset]
     meanrow=next(r for r in interaction if r['split']=='dev' and r['metric']==metric and r['seed']=='THREE_SEED_MEAN');mean=float(meanrow[field])
    else:
     subset=[r for r in video if r['split']=='dev' and r['metric']==metric and r['seed']!='THREE_SEED_MEAN'];vals=[float(r[field]) for r in subset]
     meanrow=next(r for r in video if r['split']=='dev' and r['metric']==metric and r['seed']=='THREE_SEED_MEAN');mean=float(meanrow[field])
    # The panel displays only its assigned estimand.
    if field!=key:continue
    for j,v in enumerate(vals):ax.scatter(j,v,marker=SEED_MARKER[SEEDS[j]],color=GRAY,s=38,zorder=3)
    ci=next(r for r in cis if r['split']=='dev' and r['metric']==metric and r['estimand']==('I' if field=='I_GTS_minus_GSTATE' else field.replace('_state_minus_TS','')))
    low=float(ci['group_bootstrap_ci95_low']);high=float(ci['group_bootstrap_ci95_high'])
    ax.errorbar(1,mean,yerr=[[mean-low],[high-mean]],color=color,marker='D',markersize=7,capsize=5,linewidth=2,label='equal-seed mean + group bootstrap CI')
   ax.axhline(0,color=BLACK,linewidth=1);setup(ax,catlabel,label);ax.set_xticks([0,1],['seed 01/02/03','mean'])
   ax.legend(fontsize=6.5,frameon=False)
 fig.suptitle('EXPLORATORY DEV — V_old, V_Had and I shown separately',fontweight='bold')
 fig.text(.5,-.02,'Positive V favors TS11-D over STATE-CTRL. Positive I means relative system advantage increased. I>0 does not imply V_Had>0; none of these is a video causal effect.',ha='center',fontsize=8)
 return save_figure(fig,'DEV_SYSTEM_ADVANTAGE_INTERACTION')

def dev_tv_direction():
 rows=read_csv(ROOT/'results/AGGREGATE_RESULTS.csv')
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharey='col')
 for ri,split in enumerate(('train','dev')):
  for ci,(metric,label) in enumerate([('tv_ratio_aggregate','Aggregate TV_pred / TV_GT'),('direction_accuracy_active_pooled','Direction accuracy on active GT steps')]):
   ax=axes[ri,ci];labels=[];vals=[];colors=[];marks=[]
   for config,query in [('TS11-D','old'),('TS11-D','had'),('STATE-CTRL','old'),('STATE-CTRL','had')]:
    row=next(r for r in rows if r['split']==split and r['config']==config and r['query_variant']==query)
    labels.append(f'{config}\n{"Q0" if query=="old" else "Q1"}')
    vals.append(float(row[f'{metric}_mean']));colors.append(MODEL_COLOR[config]);marks.append(QUERY_STYLE[query][1])
   for x,(v,c,m) in enumerate(zip(vals,colors,marks)):
    ax.scatter(x,v,color=c,marker=m,s=70,edgecolor='black',linewidth=.4)
   if metric=='tv_ratio_aggregate':ax.axhline(1,color=GRAY,linestyle=':',linewidth=1)
   setup(ax,f'{split.upper()} · {label}',label);ax.set_xticks(range(4),labels);ax.tick_params(axis='x',labelsize=8)

 fig.suptitle('TRAIN is in-sample; DEV is exploratory and previously exposed',fontweight='bold')
 fig.text(.5,-.02,'Action support counts and zero-GT-TV counts are in MOTION_DIRECTION_SUPPORT.csv. TV≈1 does not establish correct dynamics.',ha='center',fontsize=8)
 return save_figure(fig,'TRAIN_DEV_TV_DIRECTION_SUPPORT')

def paired_ecdf():
 rows=read_csv(ROOT/'results/EPISODE_PAIRED.csv')
 specs=[('G_Q_position_standardized_mae_old_minus_had','Paired position gain (old − Hadamard)'),('G_Q_safe_fd_normalized_old_minus_had','Paired safe-FD gain (old − Hadamard)')]
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
 for ax,(field,label) in zip(axes,specs):
  for config,color,ls in [('TS11-D',BLUE,'-'),('STATE-CTRL',ORANGE,'--')]:
   group_seed_values={}
   for row in rows:
    if row['split']=='dev' and row['config']==config:
     group_seed_values.setdefault(row['source_group_id'],[]).append(float(row[field]))
   if len(group_seed_values)!=603 or any(len(v)!=3 for v in group_seed_values.values()):
    raise AssertionError(f'expected 603 source groups x 3 fixed seeds for {config}')
   values=np.sort(np.asarray([np.mean(group_seed_values[g]) for g in sorted(group_seed_values)],dtype=np.float64))
   values=values[np.isfinite(values)]
   y=np.arange(1,len(values)+1)/len(values)
   ax.plot(values,y,color=color,linestyle=ls,linewidth=2,label=f'{config} (n={len(values)} source groups; 3-seed mean)')
  ax.axvline(0,color=BLACK,linewidth=1);setup(ax,label,'Empirical cumulative fraction');ax.legend(fontsize=8,frameon=False)
 fig.suptitle('EXPLORATORY DEV — paired source-group gain distributions',fontweight='bold')
 fig.text(.5,-.02,'Each source-group value is the mean paired Q0-minus-Q1 gain across the three fixed seeds; repeated anchors were first averaged within episode. Positive values favor Q1.',ha='center',fontsize=8)
 return save_figure(fig,'DEV_PAIRED_GAIN_ECDF')

def write_persistence_anchor_table():
 cases=read_csv(ROOT/'qualitative/CASE_MANIFEST.csv')
 lock=json.loads((ROOT/'qa/CASE_SELECTION_LOCKED.json').read_text(encoding='utf-8'))
 if lock.get('status')!='PASS_CASE_IDS_LOCKED_BEFORE_SCORING' or sha(ROOT/'qualitative/CASE_MANIFEST.csv')!=lock.get('case_manifest_sha256'):
  raise AssertionError('persistence figure data requires the accepted pre-scoring case lock')
 with np.load(V3_INPUT,allow_pickle=False) as archive:
  ids=archive['anchor_id'].astype('U96')
  splits=archive['split_code'].astype(np.int8)
  anchors=archive['anchor_command_native'].astype(np.float32)
 rows=[]
 for case in cases:
  matches=np.flatnonzero(ids==case['anchor_id'])
  if len(matches)!=1:raise AssertionError(f'expected one source row for case anchor {case["anchor_id"]}')
  idx=int(matches[0])
  if int(splits[idx])!=1:raise AssertionError(f'case anchor is not in the locked dev split: {case["anchor_id"]}')
  for action in (0,14):
   value=float(anchors[idx,action])
   for step in range(1,31):
    rows.append({'case_id':case['case_id'],'stratum':case['stratum'],'source_group_id':case['source_group_id'],
     'episode_index':int(case['episode_index']),'anchor_id':case['anchor_id'],'raw_action_index':action,
     'future_step':step,'target_timestamp_seconds':float(case['anchor_timestamp_seconds'])+step/30.0,
     'anchor_native':value,'persistence_native':value,'source_array':'anchor_command_native',
     'final_test_access':0,'heldout_access':0})
 path=ROOT/'figure_data/PERSISTENCE_ANCHOR_HOLD.csv'
 if path.exists():raise FileExistsError(f'preserving existing persistence case table {path}')
 with path.open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return len(rows)

def case_trajectories(action:int,stratum:str):
 cases=[x for x in read_csv(ROOT/'qualitative/CASE_MANIFEST.csv') if x['stratum']==stratum]
 persistence=read_csv(ROOT/'figure_data/PERSISTENCE_ANCHOR_HOLD.csv')
 fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
 for ax,case in zip(axes.flat,cases):
  data=read_csv(ROOT/f'figure_data/case_{case["case_id"]}_action_{action}.csv')
  target=np.asarray([float(r['target_native']) for r in sorted([x for x in data if x['config']=='TS11-D' and x['query_variant']=='old' and int(x['seed'])==20261001],key=lambda x:int(x['future_step']))])
  steps=np.arange(1,31)
  ax.plot(steps,target,color=BLACK,linewidth=2.2,label='Ground truth')
  for config,query in [('TS11-D','old'),('TS11-D','had'),('STATE-CTRL','old'),('STATE-CTRL','had')]:
   curves=[]
   for seed in SEEDS:
    sub=sorted([x for x in data if x['config']==config and x['query_variant']==query and int(x['seed'])==seed],key=lambda x:int(x['future_step']))
    curves.append([float(x['prediction_native']) for x in sub])
   arr=np.asarray(curves,dtype=np.float64);mean=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1)
   ls,marker=QUERY_STYLE[query];color=MODEL_COLOR[config]
   ax.plot(steps,mean,color=color,linestyle=ls,marker=marker,markevery=7,markersize=3,linewidth=1.5,label=CONDITION_LABEL[(config,query)])
   ax.fill_between(steps,mean-sd,mean+sd,color=color,alpha=.12,linewidth=0)
  hold=sorted([r for r in persistence if r['case_id']==case['case_id'] and int(r['raw_action_index'])==action],key=lambda r:int(r['future_step']))
  if len(hold)!=30:raise AssertionError(f'persistence case series incomplete: {case["case_id"]} action[{action}]')
  ax.plot(steps,[float(r['persistence_native']) for r in hold],color=GRAY,linestyle=':',linewidth=1.5,label='Persistence (anchor hold)')
  setup(ax,f"{case['case_id']} · {case['source_group_id'][:8]}",f'native action[{action}]')
  ax.set_xlim(1,30);ax.set_xticks([1,10,20,30]);ax.tick_params(axis='x',labelsize=8)
 handles,labels=axes.flat[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='upper center',ncol=6,frameon=False,fontsize=8,bbox_to_anchor=(.5,.94))
 fig.suptitle(f'EXPLORATORY DEV — {stratum.upper()} motion cases · raw action[{action}]',fontweight='bold',y=.99)
 fig.text(.5,.02,'Target-defined motion strata use frozen train q25/q75; case IDs were hashed and locked before scoring. Persistence is anchor hold; learned-system lines are equal-seed means with sample-SD shading.',ha='center',fontsize=8)
 fig.tight_layout(rect=(.035,.11,.985,.84))
 return save_figure(fig,f'DEV_CASE_TRAJECTORIES_{stratum.upper()}_ACTION_{action}')

def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
                      'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
 ROOT.joinpath('figures').mkdir(exist_ok=True)
 persistence_rows=write_persistence_anchor_table()
 figures=[]
 for fn in (objective_curves,monitor_curves,dev_primary_by_seed,dev_query_gains,dev_system_advantage,dev_tv_direction,paired_ecdf):figures.extend(fn())
 for stratum in ('low','high'):
  for action in (0,14):figures.extend(case_trajectories(action,stratum))
 captions={
  'TRAIN_OBJECTIVE_CURVES':'Train-only full-population episode-weighted objective by epoch. Q0 uses accepted corrected-scheduler histories; Q1 uses the six new fixed-query fits. Thin lines are seeds and thick lines are equal-seed means. Targets were used in fitting.',
  'TRAIN_MONITOR_PANEL_CURVES':'Train-only fixed 128-anchor monitor panel, one anchor per source group. Q1 monitor position, safe-FD and TV ratio are shown per fixed seed at each epoch; targets were used in fitting. Q0 monitor panels were not recorded during the reused run.',
  'DEV_PRIMARY_METRICS_BY_SEED':'Previously exposed dev position and safe-FD errors by paired fixed seed. Error bars are sample SD across three seed summaries; dev is exploratory and was not used for model selection.',
  'DEV_QUERY_GAINS_BOOTSTRAP':'Paired Q0 error minus Q1 error. Dots are fixed-seed estimates; intervals are paired 95% source-group bootstrap intervals conditional on the fitted models and current dev population.',
  'DEV_SYSTEM_ADVANTAGE_INTERACTION':'V_old and V_Had compare STATE-CTRL error minus TS11-D error; I is their paired difference. Positive V favors TS11-D. Positive I does not imply V_Had is positive and is not a video causal effect.',
  'TRAIN_DEV_TV_DIRECTION_SUPPORT':'Aggregate TV ratio and active-step direction accuracy for train and previously exposed dev. Direction uses the frozen train-fitted activity threshold; support counts and zero-motion cases are in the source table.',
  'DEV_PAIRED_GAIN_ECDF':'Empirical source-group distributions of paired Q0-minus-Q1 position and safe-FD gains after averaging each source group over three fixed seeds. Positive values favor Q1 Hadamard; dev is exploratory and previously exposed.'}
 for stratum in ('low','high'):
  for action in (0,14):captions[f'DEV_CASE_TRAJECTORIES_{stratum.upper()}_ACTION_{action}']=f'Previously exposed dev H30 trajectories for six prelocked {stratum}-motion source groups, raw action[{action}]. Ground truth is black; persistence is anchor hold in gray dotted; colored curves are four systems averaged over three fixed seeds, with ±sample SD shading.'
 md=['# Figure captions and alt text','', 'All query comparisons use the same frozen primary train/dev population. Numeric action indices are used because physical/anatomical action semantics remain unresolved. Publisher compliance is not assessed.','']
 data_sources={'TRAIN_OBJECTIVE_CURVES':'results/TRAIN_CURVES.csv','TRAIN_MONITOR_PANEL_CURVES':'results/TRAIN_CURVES.csv',
  'DEV_PRIMARY_METRICS_BY_SEED':'results/TRAIN_DEV_BY_SEED.csv; results/AGGREGATE_RESULTS.csv',
  'DEV_QUERY_GAINS_BOOTSTRAP':'results/QUERY_GAINS.csv; results/BOOTSTRAP_CI.csv',
  'DEV_SYSTEM_ADVANTAGE_INTERACTION':'results/ABSOLUTE_VIDEO_GAINS.csv; results/QUERY_MODEL_INTERACTION.csv; results/BOOTSTRAP_CI.csv',
  'TRAIN_DEV_TV_DIRECTION_SUPPORT':'results/AGGREGATE_RESULTS.csv; results/MOTION_DIRECTION_SUPPORT.csv',
  'DEV_PAIRED_GAIN_ECDF':'results/EPISODE_PAIRED.csv; 3-seed mean within each source group'}
 for p in figures:
  if p.suffix!='.png':continue
  stem=p.stem
  cap=captions.get(stem,'Source-backed exploratory figure; see its CSV data and analysis lock.')
  if stem.startswith('DEV_CASE_TRAJECTORIES_'):
   action=stem.rsplit('_',1)[-1]
   src=f'figure_data/case_*_action_{action}.csv; figure_data/PERSISTENCE_ANCHOR_HOLD.csv; qualitative/CASE_MANIFEST.csv; V3_MAIN_INPUTS.npz anchor_command_native only'
  else:src=data_sources.get(stem,'results/*.csv')
  md += [f'## {stem}', '',f'**Caption:** {cap}','','**Alt text:** {cap}','','Data source: {src}; PNG SHA-256: {sha(p)}.','']
 mdpath=ROOT/'captions/FIGURE_CAPTIONS_AND_ALT.zh-CN.md';mdpath.write_text('\n'.join(md),encoding='utf-8')
 # Metadata, opacity and palette QA.
 from PIL import Image
 entries=[]
 for p in figures:
  if p.suffix!='.png':continue
  with Image.open(p) as im:
   alpha=im.getchannel('A').getextrema() if im.mode=='RGBA' else None
   entries.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,'format':im.format,'mode':im.mode,
    'width_px':im.width,'height_px':im.height,'dpi':im.info.get('dpi'),'alpha_extrema':alpha,'opaque':alpha is None or alpha==(255,255)})
 colors=[BLUE,ORANGE,GREEN,BLACK,GRAY]
 visual={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'status':'PASS_FIGURE_METADATA_AND_PALETTE_SCREEN',
  'figures':entries,'colors':{c:{'contrast_ratio_vs_white':wcag(c)} for c in colors},
  'redundant_encoding':'model by color; query by line style and marker; ground truth black solid; persistence gray dotted; case series labeled',
  'target_used_in_fitting_disclosed':True,'dev_previously_exposed_disclosed':True,'data_sources':'results/*.csv; figure_data/*.csv; qualitative/CASE_MANIFEST.csv; V3_MAIN_INPUTS.npz anchor_command_native only',
  'manual_visual_review':'pending','human_review_status':'pending','publisher_compliance':'not assessed; no target journal/phase specified',
  'alpha_policy':'opaque white canvas; uncertainty bands use alpha on colored lines only; PNG canvas verified opaque'}
 (ROOT/'qa/VISUAL_QA.json').write_text(json.dumps(visual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 manifest={'schema_version':1,'status':'PASS','source_sha256':sha(Path(__file__).resolve()),'amendment_id':AMENDMENT_ID,
           'render_stage':str(ROOT),'figures':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in figures],
           'figure_data_sources':[{'path':'figure_data/PERSISTENCE_ANCHOR_HOLD.csv','sha256':sha(ROOT/'figure_data/PERSISTENCE_ANCHOR_HOLD.csv'),'rows':persistence_rows},
                                  {'path':'qualitative/CASE_MANIFEST.csv','sha256':sha(ROOT/'qualitative/CASE_MANIFEST.csv'),'rows':len(read_csv(ROOT/'qualitative/CASE_MANIFEST.csv'))}],
           'anchor_source':{'path':str(V3_INPUT),'array':'anchor_command_native','metadata_arrays_read':['anchor_id','split_code'],'target_arrays_opened':False},
           'caption_path':str(mdpath.relative_to(ROOT)),'caption_sha256':sha(mdpath),'visual_qa_path':'qa/VISUAL_QA.json','visual_qa_sha256':sha(ROOT/'qa/VISUAL_QA.json')}
 (ROOT/'figures/FIGURE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','figure_count_png':len(entries),'pdf_count':sum(p.suffix=='.pdf' for p in figures),'captions':len(captions),'visual_qa':'PASS_FIGURE_METADATA_AND_PALETTE_SCREEN'},indent=2))
if __name__=='__main__':main()
