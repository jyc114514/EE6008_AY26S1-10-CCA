#!/usr/bin/env python3
"""Plot N2 paired query effects and backend/profile drift from locked result tables."""
from __future__ import annotations
import csv,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
FIG=ROOT/'figures';DATA=ROOT/'figure_data';CAP=ROOT/'captions'

def read_csv(name):
 with (RESULTS/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bootstrap_map():
 return {(r['metric'],r['estimand']):r for r in read_csv('BOOTSTRAP_CI.csv')}
def allseed_map():
 return {(r['metric'],r['estimand']):r for r in read_csv('ALL_SEED_LOO.csv')}
def ci_for(metric,estimand,bm):
 r=bm[(metric,estimand)]
 return float(r['point_estimate_mean_fixed_seeds']),float(r['group_ci95_low']),float(r['group_ci95_high']),r
def add_group_effect(ax,y,mean,low,high,color,marker,label=None):
 ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt=marker,color=color,
  ecolor='#333333',elinewidth=1.2,capsize=3,markersize=7,markeredgecolor='#202020',
  markeredgewidth=0.6,label=label,zorder=3)
def main():
 FIG.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True);CAP.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
  'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
  'axes.spines.top':False,'axes.spines.right':False})
 bm=bootstrap_map();am=allseed_map()
 models=['TS11-D','STATE-CTRL'];backends=['CPU_COMMON','CUDA_COMMON'];metrics=['position_standardized_mae','safe_fd_normalized']
 colors={'TS11-D':'#0072B2','STATE-CTRL':'#D55E00'}
 markers={'CPU_COMMON':'o','CUDA_COMMON':'D'}
 labels=[]
 for model in models:
  for backend in backends:labels.append(f'{model} | {backend.replace("_COMMON","")}')
 y_map={label:i for i,label in enumerate(labels[::-1])}
 fig,axes=plt.subplots(1,2,figsize=(10.2,4.8),layout='constrained',sharey=True)
 effect_data=[]
 for ax,metric,title in zip(axes,metrics,['Position standardized MAE','Safe-FD normalized error']):
  ax.axvline(0,color='#555555',linewidth=0.9,linestyle='--',zorder=1)
  for model in models:
   for backend in backends:
    label=f'{model} | {backend.replace("_COMMON","")}'
    y=y_map[label]
    est,lo,hi,_=ci_for(metric,f'G/{model}/{backend}',bm)
    add_group_effect(ax,y,est,lo,hi,colors[model],markers[backend])
    seedrow=am[(metric,f'G/{model}/{backend}')]
    seedvals=[float(seedrow[f'seed_{s}']) for s in (20261001,20261002,20261003)]
    for off,value in zip((-0.13,0.0,0.13),seedvals):
     ax.scatter(value,y+off,s=19,marker='x',color='#222222',alpha=0.72,linewidths=0.8,zorder=4)
    effect_data.append({'metric':metric,'model':model,'backend':backend,'estimand':'G_Q0_minus_Q1',
     'mean_fixed_seeds':est,'group_ci95_low':lo,'group_ci95_high':hi,
     'seed_20261001':seedvals[0],'seed_20261002':seedvals[1],'seed_20261003':seedvals[2],
     'positive_favors':'Q1 lower error','bootstrap_replicates':10000,'bootstrap_seed':20261011,'source_groups':603})
  ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels[::-1])
  ax.set_title(title);ax.set_xlabel('G = L(Q0) - L(Q1)')
  ax.grid(axis='x',color='#DDDDDD',linewidth=0.6);ax.set_axisbelow(True)
  vals=[x for d in effect_data if d['metric']==metric for x in (d['group_ci95_low'],d['group_ci95_high'])]
  span=max(vals)-min(vals);margin=max(span*0.12,1e-4)
  ax.set_xlim(min(vals)-margin,max(vals)+margin)
 p1=FIG/'N2_QUERY_EFFECTS.png';p2=FIG/'N2_QUERY_EFFECTS.pdf'
 fig.savefig(p1,dpi=300,facecolor='white');fig.savefig(p2,facecolor='white',metadata={'Title':'N2 query effects','Creator':'EE6008 N2 analysis'})
 plt.close(fig)
 write_csv(DATA/'N2_QUERY_EFFECTS_FIGURE_DATA.csv',effect_data)

 drift_data=[]
 fig,axes=plt.subplots(2,2,figsize=(10.4,7.0),layout='constrained')
 cats=[('TS11-D','Q0'),('TS11-D','Q1'),('STATE-CTRL','Q0'),('STATE-CTRL','Q1')]
 ylabs=[f'{m} | {q}' for m,q in cats]
 for ri,metric in enumerate(metrics):
  for ci,kind in enumerate(('Backend drift B','Stored-profile drift D')):
   ax=axes[ri,ci];ax.axvline(0,color='#555555',linewidth=0.9,linestyle='--',zorder=1)
   for j,(model,query) in enumerate(cats):
    if kind.startswith('Backend'):
     estimand=f'B/{model}/{query}'
    else:
     estimand=f'DRIFT_OLD/{model}' if query=='Q0' else f'DRIFT_HAD/{model}'
    est,lo,hi,_=ci_for(metric,estimand,bm);marker='s' if query=='Q0' else 'o'
    add_group_effect(ax,j,est,lo,hi,colors[model],marker)
    drift_data.append({'metric':metric,'contrast_type':'B_CUDA_COMMON_minus_CPU_COMMON' if kind.startswith('Backend') else 'D_common_minus_N1_stored',
     'model':model,'query':query,'estimand':estimand,'mean_fixed_seeds':est,
     'group_ci95_low':lo,'group_ci95_high':hi,'positive_means':'higher error in first term',
     'bootstrap_replicates':10000,'bootstrap_seed':20261011,'source_groups':603})
   ax.set_yticks(range(4));ax.set_yticklabels(ylabs)
   ax.set_title(metric+' | '+kind)
   ax.set_xlabel('Difference in group-mean normalized loss')
   ax.grid(axis='x',color='#DDDDDD',linewidth=0.6);ax.set_axisbelow(True)
   ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
   ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
 d1=FIG/'N2_BACKEND_PROFILE_DRIFT.png';d2=FIG/'N2_BACKEND_PROFILE_DRIFT.pdf'
 fig.savefig(d1,dpi=300,facecolor='white');fig.savefig(d2,facecolor='white',metadata={'Title':'N2 backend and profile drift','Creator':'EE6008 N2 analysis'})
 plt.close(fig)
 write_csv(DATA/'N2_BACKEND_PROFILE_DRIFT_FIGURE_DATA.csv',drift_data)
 caption1='Figure N2-1. Query effect G = L(Q0) - L(Q1) under common CPU and CUDA inference profiles. Positive values indicate lower error with Q1. Points are means over the three fixed seeds; horizontal bars are 95% paired source-group percentile-bootstrap intervals (10,000 shared draws, seed 20261011, 603 development source groups). Rows name the system and backend; colors distinguish systems, markers distinguish backend, and small crosses show the three individual seed estimates. These are exploratory scores on the previously exposed development split, not independent confirmation.'
 alt1='Two-panel dot-and-whisker figure for position standardized MAE and safe-FD normalized error. It compares Q0 minus Q1 for TS11-D and STATE-CTRL under CPU_COMMON and CUDA_COMMON. The CPU and CUDA values nearly overlap. TS11-D favors Q1 on both metrics; STATE-CTRL has a negative position effect and a safe-FD interval that crosses zero.'
 caption2='Figure N2-2. Backend drift B = L(CUDA_COMMON) - L(CPU_COMMON) and stored-profile drift D = L(common profile) - L(N1 stored as-run profile), shown in separate columns. Row labels identify system and query; colors distinguish systems and markers distinguish Q0/Q1. Each point averages the three fixed seeds; bars are 95% paired source-group bootstrap intervals (10,000 shared draws, seed 20261011, 603 development groups). Historical Q0 predictions were CPU; historical Q1 predictions were CUDA on physical GPU 0. Current CUDA_COMMON views used physical GPU 2. D therefore describes the recorded as-run-to-common comparison, including its recorded device/profile context.'
 alt2='Four-panel dot-and-whisker figure separates common-profile CUDA-minus-CPU loss differences from common-minus-stored-profile differences for position MAE and safe-FD. All backend differences are very close to zero. The stored-profile differences are also small and their intervals cross zero.'
 (CAP/'N2_QUERY_EFFECTS.zh-CN.md').write_text('# N2 Figure 1\\n\\n'+caption1+'\\n\\nAlt text: '+alt1+'\\n',encoding='utf-8')
 (CAP/'N2_BACKEND_PROFILE_DRIFT.zh-CN.md').write_text('# N2 Figure 2\\n\\n'+caption2+'\\n\\nAlt text: '+alt2+'\\n',encoding='utf-8')
 manifest=[]
 for p in (p1,p2,d1,d2):
  manifest.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,
   'format':p.suffix.lstrip('.'),'purpose':'N2 common-profile query effect or backend/profile drift figure'})
 write_csv(FIG/'FIGURE_MANIFEST.csv',manifest)
 print(json.dumps({'status':'PASS','figures':[str(p1),str(p2),str(d1),str(d2)],'manifest_rows':len(manifest)},sort_keys=True))

if __name__=="__main__":
 main()
