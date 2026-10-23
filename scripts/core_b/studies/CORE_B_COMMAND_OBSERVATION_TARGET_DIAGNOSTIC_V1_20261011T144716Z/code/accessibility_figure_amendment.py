#!/usr/bin/env python3
"""Presentation-only color/line-style redundancy amendment for figures 2/3/4/6."""
import csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
VIEWS=['OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND']
STYLES={
 'OBS_PERSISTENCE':dict(color='#555555',linestyle='-',marker='o',label='OBS persistence'),
 'OBS_CONSTANT_VELOCITY':dict(color='#0072B2',linestyle='--',marker='s',label='OBS constant velocity'),
 'ARX_HOLD_COMMAND':dict(color='#009E73',linestyle='-.',marker='^',label='ARX hold command'),
 'ARX_ORACLE_COMMAND':dict(color='#D55E00',linestyle=':',marker='D',label='ARX oracle command')}
HATCH={'OBS_PERSISTENCE':'','OBS_CONSTANT_VELOCITY':'//','ARX_HOLD_COMMAND':'..','ARX_ORACLE_COMMAND':'xx'}
def rows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def load(path,keys):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def utc():return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
t0=time.perf_counter(); dev_label_read_utc=utc()
sums=rows(ROOT/'results/FORECAST_METRICS_BY_HORIZON.csv'); teachers=rows(ROOT/'results/TEACHER_LOCAL_METRICS.csv')
# Horizon line panels: color, marker, and dash encode each forecast view redundantly.
fig,axs=plt.subplots(1,2,figsize=(12,4.7))
for ax,metric,title in [(axs[0],'position_skill_vs_obs_persistence','Prefix position skill vs OBS persistence'),(axs[1],'safe_fd_skill_vs_obs_persistence','Internal safe-FD skill vs OBS persistence')]:
 for v in VIEWS:
  rr=sorted([r for r in sums if r['split']=='dev' and r['view']==v and r['block']=='Full43' and r[metric] not in ('nan','')],key=lambda r:int(r['horizon']))
  ax.plot([int(r['horizon']) for r in rr],[float(r[metric]) for r in rr],lw=1.7,markersize=5,markevery=1,**STYLES[v])
 ax.axhline(0,color='black',lw=.8); ax.set_xticks([1,3,6,10,15,30]); ax.set_xlabel('Fixed horizon (steps)'); ax.set_ylabel('Skill (1 − loss / persistence loss)'); ax.set_title(title); ax.grid(alpha=.25)
axs[1].text(.98,.03,'H=1 internal FD = NA',transform=axs[1].transAxes,ha='right',fontsize=8)
axs[0].legend(fontsize=7,loc='best'); fig.suptitle('DEV (previously exposed; exploratory) — four forecast views only'); fig.tight_layout(); fig.savefig(ROOT/'figures/FIG02_DEV_HORIZON_SKILL_AMEND001.png',dpi=180); plt.close(fig)
# H30 blocks: bars include common zero baseline; hatch/edge reinforces color.
blocks=['arm_joints_source_alias','hand_joints_source_alias','leg_joints_source_alias','Full43']; xx=np.arange(len(blocks)); width=.19
fig,axs=plt.subplots(1,2,figsize=(13,5))
for j,v in enumerate(VIEWS):
 rr=[next(r for r in sums if r['split']=='dev' and r['view']==v and r['horizon']=='30' and r['block']==b) for b in blocks]
 kw=dict(width=width,facecolor=STYLES[v]['color'],edgecolor='#222222',linewidth=.7,hatch=HATCH[v],label=STYLES[v]['label'])
 axs[0].bar(xx+(j-1.5)*width,[float(r['position_prefix_standardized_mae']) for r in rr],**kw)
 axs[1].bar(xx+(j-1.5)*width,[float(r['safe_fd_error']) if r['safe_fd_error']!='nan' else np.nan for r in rr],**{**kw,'label':None})
for ax,title in zip(axs,['H30 prefix position MAE / OBS σ','H30 future-internal safe-FD error']): ax.set_xticks(xx,blocks,rotation=15); ax.set_title(title); ax.set_ylabel('absolute error (lower is better)'); ax.grid(axis='y',alpha=.25)
axs[0].legend(fontsize=7); fig.suptitle('DEV raw source blocks; unit unknown; same eligible anchors'); fig.tight_layout(); fig.savefig(ROOT/'figures/FIG03_DEV_H30_RAW_BLOCKS_AMEND001.png',dpi=180); plt.close(fig)
# Teacher is a local, GT-reset diagnostic. Keep panel separate and use redundant markers/styles.
fig,ax=plt.subplots(figsize=(9,5))
tstyle={'TEACHER_FORCED_LOCAL':dict(color='#D55E00',linestyle='-',marker='D',label='Teacher forced local'),
 'LOCAL_PERSISTENCE':dict(color='#555555',linestyle='--',marker='o',label='Local persistence'),
 'LOCAL_CONSTANT_VELOCITY':dict(color='#0072B2',linestyle=':',marker='^',label='Local constant velocity')}
for v,sty in tstyle.items():
 rr=sorted([r for r in teachers if r['split']=='dev' and r['view']==v and r['block']=='Full43'],key=lambda r:int(r['step']))
 ax.plot([int(r['step']) for r in rr],[float(r['episode_equal_position_mae_std']) for r in rr],lw=1.5,markersize=4,markevery=5,**sty)
ax.set_xlabel('Local transition index j+1'); ax.set_ylabel('One-step position MAE / OBS σ'); ax.set_title('Teacher forcing is a GT-reset local diagnostic, not an H30 forecast'); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/'figures/FIG04_TEACHER_LOCAL_ONLY_AMEND001.png',dpi=180); plt.close(fig)
# Fixed-case native paths: redundant line styles/markers. GT black; oracle remains prominently labeled.
inp=load(ROOT/'data/dev/inputs_prefix.npz',['anchor_id']); ids=inp['anchor_id'].tolist(); idx={a:i for i,a in enumerate(ids)}
labels=load(ROOT/'data/dev/labels_obs43.npz',['obs_future'])['obs_future'];
preds=load(ROOT/'predictions/dev/FORECASTS_H30.npz',VIEWS)
cases=rows(ROOT/'data/DEV_CASES_LOCKED.csv'); channel_keys=[('arm_joints_source_alias','arm_joints_channel'),('hand_joints_source_alias','hand_joints_channel'),('leg_joints_source_alias','leg_joints_channel')]
with PdfPages(ROOT/'figures/FIG06_FIXED_DEV_NATIVE_CASE_PATHS_AMEND002.pdf') as pdf:
 for case in cases:
  i=idx[case['anchor_id']]; fig,axs=plt.subplots(1,3,figsize=(14,4.5),sharex=True)
  for ax,(block,key) in zip(axs,channel_keys):
   cidx=int(case[key]); ax.plot(np.arange(1,31),labels[i,:,cidx],label='Ground truth',color='black',lw=2,marker='o',markersize=3,markevery=5)
   for v in VIEWS:
    st=STYLES[v]; ax.plot(np.arange(1,31),preds[v][i,:,cidx],label=st['label'],color=st['color'],lw=1.2,linestyle=st['linestyle'],marker=st['marker'],markersize=2.8,markevery=5)
   ax.set_title(f'{block}: channel {cidx}'); ax.set_xlabel('step'); ax.set_ylabel('recorded numeric native value (unit unresolved)'); ax.grid(alpha=.2)
  axs[0].legend(fontsize=6,loc='best'); fig.suptitle(f"Case {case['anchor_id']} — source group {case['source_group_id'][:10]}\nDEV repeated exposure; fixed by ID hash before scores")
  fig.text(.5,.01,'ARX ORACLE COMMAND: PRIVILEGED FUTURE COMMAND / NON-DEPLOYABLE',ha='center',color='#B34700',weight='bold',fontsize=11)
  fig.tight_layout(rect=(0,.06,1,.9)); pdf.savefig(fig); plt.close(fig)
record={'status':'PASS_PRESENTATION_ONLY','changed_figures':['FIG02_DEV_HORIZON_SKILL_AMEND001.png','FIG03_DEV_H30_RAW_BLOCKS_AMEND001.png','FIG04_TEACHER_LOCAL_ONLY_AMEND001.png','FIG06_FIXED_DEV_NATIVE_CASE_PATHS_AMEND002.pdf'],'style_rule':'qualitative color is redundant with distinct marker/line style; bars also use hatching','dev_label_read_for_case_paths_utc':dev_label_read_utc,'case_selection_changed':False,'models_metrics_selection_changed':False,'all_source_groups_cases':len(cases),'case_pages':len(cases),'elapsed_wall_seconds':time.perf_counter()-t0}
(ROOT/'logs/ACCESSIBILITY_FIGURE_AMENDMENT.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,sort_keys=True))
