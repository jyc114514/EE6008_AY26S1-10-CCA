#!/usr/bin/env python3
"""N4 development/training diagnostics and locked case figures."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os,io
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from PIL import Image,ImageDraw

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
V3=P/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
SEEDS=(20261001,20261002,20261003)
COLORS={'TS11-D':'#0072B2','STATE-CTRL':'#D55E00','GT':'#000000','Persistence':'#999999',
 'level':'#6A3D9A','centered':'#CAB2D6','oracle removable':'#009E73','oracle remaining':'#A6DDBA'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def C(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def wcsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 buf=io.StringIO(newline='')
 w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows(rows)
 payload=buf.getvalue()
 if p.exists():
  with p.open(newline='',encoding='utf-8') as f:existing=list(csv.DictReader(f))
  expected=[{k:('' if r.get(k) is None else str(r.get(k))) for k in fields} for r in rows]
  if existing!=expected:raise FileExistsError('preserving semantically different existing figure data '+str(p))
  return
 p.write_text(payload,encoding='utf-8')
def wjson(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.part')
 t.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');t.replace(p)
def write_caption(stem,caption,alt):
 p=R/'captions'/f'{stem}.md'
 payload='# '+stem+'\n\nCaption: '+caption+'\n\nAlt text: '+alt+'\n'
 if p.exists():
  if p.read_text(encoding='utf-8')!=payload:raise FileExistsError('preserving non-identical existing caption '+str(p))
  return
 p.write_text(payload,encoding='utf-8')
def savefig(fig,stem,caption,alt,data_path,manifest):
 pdir=R/'figures';pdir.mkdir(parents=True,exist_ok=True)
 png=pdir/(stem+'.png');pdf=pdir/(stem+'.pdf')
 if png.exists() or pdf.exists():
  data=R/data_path;caption_path=R/'captions'/f'{stem}.md'
  if not (png.is_file() and pdf.is_file() and data.is_file() and caption_path.is_file()):
   raise FileExistsError('preserving incomplete preexisting figure '+stem)
  plt.close(fig)
  manifest.append({'figure_id':stem,'png_path':str(png),'png_sha256':sha(png),'png_bytes':png.stat().st_size,
   'pdf_path':str(pdf),'pdf_sha256':sha(pdf),'pdf_bytes':pdf.stat().st_size,
   'caption_path':str(caption_path),'caption_sha256':sha(caption_path),'alt_text':alt,
   'figure_data_path':data_path,'figure_data_sha256':sha(data),
   'development_previously_exposed':True,'final_test_access':0,'human_review':'pending',
   'reused_identical_partial_artifact':True})
  return
 fig.savefig(png,format='png',dpi=200)
 fig.savefig(pdf,format='pdf')
 plt.close(fig)
 write_caption(stem,caption,alt)
 manifest.append({'figure_id':stem,'png_path':str(png),'png_sha256':sha(png),'png_bytes':png.stat().st_size,
  'pdf_path':str(pdf),'pdf_sha256':sha(pdf),'pdf_bytes':pdf.stat().st_size,
  'caption_path':str(R/'captions'/f'{stem}.md'),'caption_sha256':sha(R/'captions'/f'{stem}.md'),
  'alt_text':alt,'figure_data_path':data_path,'figure_data_sha256':sha(R/data_path),
  'development_previously_exposed':True,'final_test_access':0,'human_review':'pending'})
def check_preconditions():
 a=J(R/'qa/ANALYSIS_ACCEPTANCE.json')
 if a.get('status')!='PASS_N4_ERROR_BUDGET_ANALYSIS':raise SystemExit('N4 analysis acceptance is required.')
 c=J(R/'protocol/CASE_SELECTION_LOCK.json')
 if c.get('status')!='LOCKED_BEFORE_N4_OUTCOME_READ' or len(c.get('case_groups',[]))!=14:
  raise SystemExit('locked 14-case selection missing.')
 if J(R/'protocol/OUTCOME_EXPOSURE.json').get('final_test_access')!=0:
  raise SystemExit('final-test gate mismatch.')
def plot_gvi(manifest):
 rows=C(R/'results/PAIRED_COMPONENT_GAINS.csv')
 loo=C(R/'results/ALL_SEED_LOO.csv')
 ci_rows=C(R/'results/GROUP_BOOTSTRAP_CI.csv')
 cats=['G_TS11-D','G_STATE-CTRL','V_Q0','V_Q1','I']
 names=['G TS11-D','G STATE-CTRL','V Q0','V Q1','I']
 seed_color={20261001:'#0072B2',20261002:'#E69F00',20261003:'#009E73'}
 fig,axs=plt.subplots(2,1,figsize=(10,7.3),layout='constrained')
 for ax,endpoint,title in zip(axs,['position_standardized_mae','safe_fd_normalized'],
  ['Position MAE (standardized)','Safe-FD L1 (accepted normalized scale)']):
  sub=[r for r in rows if r['split']=='dev' and r['axis_scope']=='raw28' and r['endpoint']==endpoint]
  stats=[r for r in loo if r['split']=='dev' and r['axis_scope']=='raw28' and r['endpoint']==endpoint]
  means={r['contrast']:float(r['three_seed_mean']) for r in stats}
  for cat_idx,c in enumerate(cats):
   for j,seed in enumerate(SEEDS):
    row=next(r for r in sub if r['contrast']==c and int(r['seed'])==seed)
    x=cat_idx+(j-1)*0.14
    ax.scatter(x,float(row['estimate']),s=42,color=seed_color[seed],marker='o',
     label=f'Seed {seed}' if cat_idx==0 else None,zorder=3)
   ciro=next(r for r in ci_rows if r['split']=='dev' and r['axis_scope']=='raw28' and r['endpoint']==endpoint and r['estimand']==c)
   mu=means[c];lo=float(ciro['ci_percentile_lower_2_5']);hi=float(ciro['ci_percentile_upper_97_5'])
   ax.errorbar(cat_idx,mu,yerr=[[mu-lo],[hi-mu]],fmt='D',markersize=5,color='black',
    capsize=3,label='Three-seed mean; 95% percentile CI' if cat_idx==0 else None,zorder=4)
  ax.axhline(0,color='#555555',linewidth=.8)
  ax.set_xticks(range(len(cats)),names);ax.set_ylabel('Q0−Q1 or system contrast')
  ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.22)
 axs[0].legend(ncol=4,frameon=False,loc='upper right')
 fig.suptitle('N4 paired G/V/I contrasts on previously exposed dev',fontsize=14)
 stem='N4_PAIRED_GVI_DEV'
 data=R/'figure_data'/f'{stem}.csv'
 selected=[r for r in rows if r['split']=='dev' and r['axis_scope']=='raw28' and
  r['endpoint'] in ('position_standardized_mae','safe_fd_normalized')]
 wcsv(data,selected)
 caption='Fixed-model development contrasts, paired by the accepted source-group unit. Positive G favors Q1; positive V favors TS11-D; I=G_TS11-D−G_STATE-CTRL. Dots show the three fixed seeds; diamonds show the equal-seed mean with the predeclared 10,000-draw source-group bootstrap percentile interval. These exploratory CIs are conditional on previously exposed development data.'
 alt='Two-panel dot plot for position MAE and safe-FD. Each panel shows seed-colored points for G TS11-D, G STATE-CTRL, V Q0, V Q1 and I, plus a black diamond and 95% bootstrap interval for the three-seed mean; a horizontal line marks zero.'
 savefig(fig,stem,caption,alt,str(data.relative_to(R)),manifest)
def plot_position_oracle(manifest):
 pos=C(R/'results/POSITION_BUDGET.csv');orc=C(R/'results/ORACLE_MAE_BUDGET.csv')
 fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained',sharex='col')
 hatches={'Q0':'','Q1':'///'}
 for row_idx,system in enumerate(('TS11-D','STATE-CTRL')):
  for q,offset in [('Q0',-.18),('Q1',.18)]:
   x=np.arange(3)+offset
   pp=[r for r in pos if r['split']=='dev' and r['axis_scope']=='raw28' and r['system']==system and r['query']==q]
   oo=[r for r in orc if r['split']=='dev' and r['axis_scope']=='raw28' and r['system']==system and r['query']==q]
   pp=sorted(pp,key=lambda r:int(r['seed']));oo=sorted(oo,key=lambda r:int(r['seed']))
   l=np.asarray([float(r['level_mse']) for r in pp]);tc=np.asarray([float(r['time_centered_error_mse']) for r in pp])
   removable=np.asarray([float(r['oracle_removable_standardized_mae']) for r in oo])
   remaining=np.asarray([float(r['oracle_remaining_standardized_mae']) for r in oo])
   axs[row_idx,0].bar(x,l,width=.34,color=COLORS['level'],hatch=hatches[q],edgecolor='#222222',label='Level MSE' if row_idx==0 and q=='Q0' else None)
   axs[row_idx,0].bar(x,tc,width=.34,bottom=l,color=COLORS['centered'],hatch=hatches[q],edgecolor='#222222',
    label='Time-centered MSE' if row_idx==0 and q=='Q0' else None)
   axs[row_idx,1].bar(x,removable,width=.34,color=COLORS['oracle removable'],hatch=hatches[q],edgecolor='#222222',
    label='Oracle-removable MAE' if row_idx==0 and q=='Q0' else None)
   axs[row_idx,1].bar(x,remaining,width=.34,bottom=removable,color=COLORS['oracle remaining'],hatch=hatches[q],edgecolor='#222222',
    label='Oracle-remaining MAE' if row_idx==0 and q=='Q0' else None)
   # Add a thin query key inside each panel without consuming data colors.
   axs[row_idx,0].text(.02,.94,'solid Q0; hatched Q1',transform=axs[row_idx,0].transAxes,fontsize=8,va='top')
   axs[row_idx,1].text(.02,.94,'solid Q0; hatched Q1',transform=axs[row_idx,1].transAxes,fontsize=8,va='top')
  axs[row_idx,0].set_title(system+' · position L2 budget',loc='left')
  axs[row_idx,1].set_title(system+' · independent oracle MAE ledger',loc='left')
  axs[row_idx,0].set_ylabel('Standardized squared error')
  axs[row_idx,1].set_ylabel('Standardized MAE')
  for ax in axs[row_idx]:
   ax.set_xticks(range(3),['20261001','20261002','20261003']);ax.grid(axis='y',alpha=.2)
 axs[0,0].legend(frameon=False,fontsize=8,loc='upper right')
 axs[0,1].legend(frameon=False,fontsize=8,loc='upper right')
 fig.suptitle('N4 dev: level/time-centered L2 and GT-oracle level-aligned L1',fontsize=14)
 stem='N4_POSITION_ORACLE_BUDGET_DEV'
 data=R/'figure_data'/f'{stem}.csv'
 data_rows=[]
 for r in pos+orc:
  if r['split']=='dev' and r['axis_scope']=='raw28':data_rows.append(r)
 wcsv(data,data_rows)
 caption='Left: POS_MSE is decomposed into axis-mean level MSE and time-centered error MSE. Right: original standardized MAE is split into the reduction from an independent per-path median-error oracle and the remaining MAE. Hatching identifies Q1. The oracle uses future ground truth and is a diagnostic, not a deployable correction. Train results are in-sample and shown in separate tables.'
 alt='Four panels: two model systems in rows; position L2 level/time-centered stacked bars and oracle MAE removable/remaining stacked bars in columns; pairs of solid and hatched bars compare Q0 and Q1 across three seeds.'
 savefig(fig,stem,caption,alt,str(data.relative_to(R)),manifest)
def plot_fd(manifest):
 rows=[r for r in C(R/'results/FD_ENERGY_ALIGNMENT.csv') if r['split']=='dev' and r['axis_scope']=='raw28']
 metrics=[('target_energy_T','Target derivative energy T'),('prediction_energy_P','Prediction derivative energy P'),
  ('cross_alignment_C','Zero-lag cross term C'),('r_norm_ratio','Norm ratio r'),
  ('c_zero_lag_cosine','Zero-lag cosine c'),('fd_mse_skill','FD-MSE skill')]
 fig,axs=plt.subplots(3,2,figsize=(12,9),layout='constrained')
 x=np.arange(6);labels=['01 Q0','01 Q1','02 Q0','02 Q1','03 Q0','03 Q1']
 system_styles={'TS11-D':('#0072B2','o'),'STATE-CTRL':('#D55E00','s')}
 q_offset={'Q0':-.06,'Q1':.06}
 for ax,(endpoint,title) in zip(axs.flat,metrics):
  for system,(color,marker) in system_styles.items():
   for q,ls in [('Q0','-'),('Q1','--')]:
    vals=[]
    for seed in SEEDS:
     rr=next(r for r in rows if r['system']==system and r['query']==q and int(r['seed'])==seed)
     vals.append(float(rr[endpoint]))
    ix=(np.arange(0,6,2) if q=='Q0' else np.arange(1,6,2)).astype(float)+q_offset[q]
    ax.plot(ix,vals,linestyle=ls,marker=marker,color=color,linewidth=1.4,
     label=f'{system} {q}' if endpoint=='target_energy_T' else None)
  ax.set_title(title,loc='left');ax.set_xticks(x,labels,rotation=0);ax.grid(axis='y',alpha=.2)
  if endpoint=='c_zero_lag_cosine':ax.set_ylabel('Inner-product cosine; not a lag test')
  elif endpoint in ('r_norm_ratio','fd_mse_skill'):ax.set_ylabel('Ratio')
  else:ax.set_ylabel('Normalized derivative energy / cross term')
  if endpoint=='target_energy_T':
   ax.text(.02,.94,'T is target-only and model-invariant',transform=ax.transAxes,va='top',fontsize=8)
 axs[0,0].legend(frameon=False,fontsize=7,ncol=2)
 fig.suptitle('N4 dev: amplitude, zero-lag alignment and derivative error budget',fontsize=14)
 stem='N4_FD_ENERGY_ALIGNMENT_DEV'
 data=R/'figure_data'/f'{stem}.csv';wcsc=rows;wcsc and wcsv(data,wcsc)
 caption='For each fixed seed and query, T is the true future-internal derivative energy, P is predicted derivative energy, and C is their same-time inner product on the accepted safe-FD scale. r, c and skill are calculated from aggregate T/P/C within each fixed model. c is a zero-lag inner-product cosine, not a Pearson coefficient or a time-lag detector. Each panel separates the quantities so their scales remain explicit.'
 alt='Six-panel line plot of target energy T, predicted energy P, cross term C, norm ratio r, zero-lag cosine c, and FD-MSE skill across three seeds and two queries, with separate colored lines for TS11-D and STATE-CTRL.'
 savefig(fig,stem,caption,alt,str(data.relative_to(R)),manifest)
def plot_axis_heatmap(manifest,metric,label,stem):
 rows=C(R/'results/BY_MODEL_SPLIT_AXIS.csv')
 rows=[r for r in rows if r['axis_scope'].startswith('axis_')]
 matrices={}
 for split in ('train','dev'):
  for system in ('TS11-D','STATE-CTRL'):
   arr=np.empty((3,28),dtype=np.float64)
   for si,seed in enumerate(SEEDS):
    for ax in range(28):
     qvals={}
     for q in ('Q0','Q1'):
      rec=next(r for r in rows if r['split']==split and r['system']==system and r['query']==q and
       int(r['seed'])==seed and r['axis_scope']==f'axis_{ax:02d}')
      qvals[q]=float(rec[metric])
     arr[si,ax]=qvals['Q0']-qvals['Q1']
   matrices[(split,system)]=arr
 vmax=max(float(np.max(np.abs(x))) for x in matrices.values())
 vmax=vmax if vmax>0 else 1.0
 fig,axs=plt.subplots(2,2,figsize=(12,6.7),layout='constrained',sharex=True,sharey=True)
 for i,split in enumerate(('train','dev')):
  for j,system in enumerate(('TS11-D','STATE-CTRL')):
   ax=axs[i,j];arr=matrices[(split,system)]
   im=ax.imshow(arr,aspect='auto',origin='lower',interpolation='nearest',
    cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax))
   ax.set_title(system+' · '+('in-sample train' if split=='train' else 'previously exposed dev'),loc='left')
   ax.set_yticks(range(3),[str(s) for s in SEEDS]);ax.set_ylabel('Fixed seed')
   ax.set_xticks(range(28),range(28),fontsize=7);ax.set_xlabel('Raw command axis (unit unverified)')
 cbar=fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.92,pad=.02);cbar.set_label('Q0 component − Q1 component; positive favors Q1')
 fig.suptitle('N4 raw-axis '+label+' gains, split shown separately',fontsize=14)
 stem_path=R/'figure_data'/f'{stem}.csv'
 data_rows=[]
 for (split,system),arr in matrices.items():
  for si,seed in enumerate(SEEDS):
   for ax in range(28):data_rows.append({'split':split,'system':system,'seed':seed,'axis':ax,
    'component':metric,'gain_Q0_minus_Q1':arr[si,ax]})
 wcsv(stem_path,data_rows)
 caption=f'Per-axis Q0−Q1 gain for {label}, shown separately for primary-train in-sample results and previously exposed development. Positive values favor Q1. Axes are raw command indices 0–27; physical units have not been verified. Each row represents one fixed seed, not an independent sample.'
 alt=f'Four-panel heatmap of raw-axis Q0-minus-Q1 {label} gains. Rows are the three fixed seeds, columns are axes 0 through 27, with separate panels for TS11-D and STATE-CTRL on train and dev. A diverging color scale centers at zero.'
 savefig(fig,stem,caption,alt,str(stem_path.relative_to(R)),manifest)
def case_figures(manifest):
 case_lock=J(R/'protocol/CASE_SELECTION_LOCK.json')
 cases=C(R/'protocol/CASE_SELECTION_LOCK.csv')
 if len(cases)!=14:raise RuntimeError('locked case list is not 14 rows')
 acceptance=J(R/'qa/PREDICTION_ACCEPTANCE.json')
 if acceptance.get('status')!='PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES':raise RuntimeError('prediction gate missing')
 # A prior post-export metadata failure may be resumed from validated locked case curves.
 case_data=R/'figure_data/CASE_CURVES.csv';case_meta=R/'figure_data/CASE_CURVE_MANIFEST.csv'
 if case_data.is_file() and case_meta.is_file():
  curve_rows=C(case_data);meta_rows=C(case_meta)
  expected_ids={c['case_id']:c for c in cases}
  series_names={'GT','Persistence','TS11-D Q0','TS11-D Q1','STATE-CTRL Q0','STATE-CTRL Q1'}
  if len(curve_rows)!=15120 or len(meta_rows)!=14:
   raise RuntimeError('existing case curve inventory is incomplete; preserve and inspect before target reuse')
  if {r['case_id'] for r in meta_rows}!=set(expected_ids):raise RuntimeError('existing case manifest ID set differs from lock')
  if {r['case_id'] for r in curve_rows}!=set(expected_ids):raise RuntimeError('existing case curve ID set differs from lock')
  for case_id,locked in expected_ids.items():
   mr=next(r for r in meta_rows if r['case_id']==case_id)
   if mr['source_group_id']!=locked['source_group_id'] or mr['anchor_id']!=locked['anchor_id']:
    raise RuntimeError('existing case manifest source identity mismatch')
   cr=[r for r in curve_rows if r['case_id']==case_id]
   if len(cr)!=1080 or {r['series'] for r in cr}!=series_names or {int(r['seed']) for r in cr}!=set(SEEDS) or {int(r['raw_axis']) for r in cr}!={0,14}:
    raise RuntimeError('existing locked case curve support mismatch')
   for seed in SEEDS:
    for axis in (0,14):
     for series in series_names:
      rr=[r for r in cr if int(r['seed'])==seed and int(r['raw_axis'])==axis and r['series']==series]
      if len(rr)!=30 or {int(r['future_step']) for r in rr}!=set(range(1,31)):
       raise RuntimeError('existing case curve path support mismatch')
      if not all(np.isfinite(float(r['native_command_value'])) for r in rr):
       raise RuntimeError('existing case curve has nonfinite value')
   stem='CASE_'+case_id;png=R/'figures/cases'/(stem+'.png');pdf=R/'figures/cases'/(stem+'.pdf')
   cap=R/'captions'/(stem+'.md')
   if not (png.is_file() and pdf.is_file() and cap.is_file()):raise RuntimeError('existing case page artifacts incomplete')
   alt=f"Six line panels for case {case_id}: two raw command axes by three fixed seeds, showing ground truth, persistence and four model-query trajectories."
   manifest.append({'figure_id':stem,'png_path':str(png),'png_sha256':sha(png),'png_bytes':png.stat().st_size,
    'pdf_path':str(pdf),'pdf_sha256':sha(pdf),'pdf_bytes':pdf.stat().st_size,
    'caption_path':str(cap),'caption_sha256':sha(cap),'alt_text':alt,
    'figure_data_path':str(case_data.relative_to(R)),'figure_data_sha256':sha(case_data),
    'case_selection_lock_sha256':sha(R/'protocol/CASE_SELECTION_LOCK.json'),
    'development_previously_exposed':True,'final_test_access':0,'human_review':'pending',
    'reused_validated_existing_case_page':True})
  contact=R/'figures/N4_CASE_CONTACT_SHEET.png';contact_cap=R/'captions/N4_CASE_CONTACT_SHEET.md'
  if not contact.is_file() or not contact_cap.is_file():raise RuntimeError('existing contact sheet artifacts missing')
  with Image.open(contact) as im:
   if im.size[0]<=0 or im.size[1]<=0:raise RuntimeError('existing contact sheet invalid')
  alt='Contact sheet of the 14 preselected distinct dev source-group cases; each tile previews the six-panel axes-0/14, three-seed trajectory page.'
  manifest.append({'figure_id':'N4_CASE_CONTACT_SHEET','png_path':str(contact),'png_sha256':sha(contact),
   'png_bytes':contact.stat().st_size,'caption_path':str(contact_cap),'caption_sha256':sha(contact_cap),
   'alt_text':alt,'figure_data_path':str(case_data.relative_to(R)),'figure_data_sha256':sha(case_data),
   'case_selection_lock_sha256':sha(R/'protocol/CASE_SELECTION_LOCK.json'),
   'development_previously_exposed':True,'final_test_access':0,'human_review':'pending',
   'reused_validated_existing_case_page':True})
  wjson(R/'qa/FIGURE_CASE_REUSE_QA.json',{'schema_version':1,'status':'PASS_REUSE_VALIDATED_LOCKED_CASE_CURVES',
   'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
   'case_selection_lock_sha256':sha(R/'protocol/CASE_SELECTION_LOCK.json'),
   'case_curve_sha256':sha(case_data),'case_curve_rows':len(curve_rows),'locked_case_groups':14,
   'case_target_reopened':False,'final_test_access':0})
  return
 # Register the locked qualitative access before opening the target key.
 when=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 ids=[c['anchor_id'] for c in cases]
 event={'attempt':1,'utc':when,'event':'N4_LOCKED_CASE_FIGURE_TARGETS_READ',
  'case_selection_lock_sha256':sha(R/'protocol/CASE_SELECTION_LOCK.json'),
  'case_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'unique_source_groups':14,
  'dev_anchor_ids_requested':14,'target_field':'targets_native[:,:30]',
  'purpose':'qualitative curves for preselected cases after accepted prediction matrix',
  'npz_target_key_rows_materialized':20079,'selected_dev_target_anchors':14,
  'final_test_access':0,'heldout_access':0}
 with (R/'qa/SCORE_TARGET_ACCESS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,sort_keys=True)+'\n')
 exp=J(R/'protocol/OUTCOME_EXPOSURE.json')
 exp['qualitative_case_target_access']={'event_utc':when,'anchors_requested':14,'source_groups':14,
  'access_log_event':'N4_LOCKED_CASE_FIGURE_TARGETS_READ','final_test_access':0,'heldout_access':0,
  'physical_note':'V3_MAIN_INPUTS is an NPZ; requesting targets_native materializes the 20,079-row canonical train+dev key, while the figure code selects only the 14 locked dev anchors.',
  'npz_target_key_rows_materialized':20079}
 wjson(R/'protocol/OUTCOME_EXPOSURE.json',exp)
 # Metadata-only fields first; target values are indexed only after this logged gate.
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  split=z['split_code'].astype(np.int8);all_ids=z['anchor_id'].astype('U96')
  all_eps=z['episode_index'].astype(np.int64)
  dev_idx=np.flatnonzero(split==1);dev_ids=all_ids[dev_idx];dev_eps=all_eps[dev_idx]
  dev_anchors=np.asarray(z['anchor_command_native'][dev_idx],dtype=np.float32).copy()
  targets=np.asarray(z['targets_native'][dev_idx,:30],dtype=np.float32).copy()
 if len(dev_ids)!=2322 or targets.shape!=(2322,30,28):raise RuntimeError('locked dev target key shape mismatch')
 row_by_id={str(a):i for i,a in enumerate(dev_ids)}
 if any(c['anchor_id'] not in row_by_id for c in cases):raise RuntimeError('case anchor is outside accepted dev support')
 # Reuse the accepted N2 common predictions, verified by the bundle table.
 n2views={(r['model'],r['query'],str(r['seed']),r['profile']):r for r in C(N2/'results/N2_VIEW_ACCEPTANCE.csv')}
 preds={}
 for system in ('TS11-D','STATE-CTRL'):
  for q in ('Q0','Q1'):
   for seed in SEEDS:
    row=n2views[(system,q,str(seed),'CUDA_COMMON')]
    path=Path(row['prediction_path'])
    if sha(path)!=row['prediction_sha256']:raise RuntimeError('N2 case prediction hash mismatch')
    with np.load(path,allow_pickle=False) as z:
     pi=np.asarray(z['predictions_native'],dtype=np.float32)
     pids=z['anchor_id'].astype('U96')
    if not np.array_equal(pids,dev_ids):raise RuntimeError('N2 case prediction ID order mismatch')
    preds[(system,q,seed)]=pi
 data_rows=[];case_manifest=[]
 for case in cases:
  aid=case['anchor_id'];i=row_by_id[aid];ep=int(dev_eps[i]);gid=case['source_group_id']
  if ep!=int(case['episode_index']):raise RuntimeError('locked case episode mismatch')
  for seed in SEEDS:
   for axis in (0,14):
    values={'GT':targets[i,:,axis],'Persistence':np.repeat(dev_anchors[i,axis],30)}
    for system in ('TS11-D','STATE-CTRL'):
     for q in ('Q0','Q1'):values[f'{system} {q}']=preds[(system,q,seed)][i,:,axis]
    for series,yv in values.items():
     for step,val in enumerate(yv,1):
      data_rows.append({'case_id':case['case_id'],'selection_origin':case['selection_origin'],
       'source_group_id':gid,'episode_index':ep,'anchor_id':aid,'seed':seed,'raw_axis':axis,
       'series':series,'future_step':step,'native_command_value':float(val),
       'physical_unit_verified':False,'final_test_access':0})
  case_manifest.append({'case_id':case['case_id'],'source_group_id':gid,'episode_index':ep,
   'anchor_id':aid,'selection_origin':case['selection_origin'],'selection_sha256':case['selection_sha256']})
 case_data=R/'figure_data/CASE_CURVES.csv';wcsv(case_data,data_rows)
 case_meta=R/'figure_data/CASE_CURVE_MANIFEST.csv';wcsv(case_meta,case_manifest)
 cmap={'TS11-D Q0':('#0072B2','-'),'TS11-D Q1':('#0072B2','--'),
  'STATE-CTRL Q0':('#D55E00','-'),'STATE-CTRL Q1':('#D55E00','--'),
  'GT':('#000000','-'),'Persistence':('#999999',':')}
 for case in cases:
  fig,axs=plt.subplots(2,3,figsize=(12,6.3),layout='constrained',sharex=True)
  selected=[r for r in data_rows if r['case_id']==case['case_id']]
  axis_lims={}
  for axis,rowi in ((0,0),(14,1)):
   vals=[r['native_command_value'] for r in selected if r['raw_axis']==axis]
   lo=min(vals);hi=max(vals);pad=max((hi-lo)*.08,1e-5);axis_lims[axis]=(lo-pad,hi+pad)
  for col,seed in enumerate(SEEDS):
   for axis,rowi in ((0,0),(14,1)):
    ax=axs[rowi,col];subset=[r for r in selected if r['raw_axis']==axis and int(r['seed'])==seed]
    for series,(color,ls) in cmap.items():
     rr=[r for r in subset if r['series']==series]
     ax.plot([r['future_step'] for r in rr],[r['native_command_value'] for r in rr],
      color=color,linestyle=ls,linewidth=1.3,label=series if (rowi==0 and col==0) else None)
    ax.set_ylim(*axis_lims[axis]);ax.set_xlim(1,30);ax.grid(alpha=.18)
    ax.set_title(f'axis {axis} · seed {seed}' if rowi==0 else f'axis {axis}',loc='left',fontsize=9)
    if col==0:ax.set_ylabel('native command value\n(unit unverified)')
    if rowi==1:ax.set_xlabel('future step (1–30)')
  fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=6,
   frameon=False,fontsize=8)
  stem=f"CASE_{case['case_id']}"
  pdir=R/'figures/cases';pdir.mkdir(parents=True,exist_ok=True)
  png=pdir/(stem+'.png');pdf=pdir/(stem+'.pdf')
  if png.exists() or pdf.exists():raise FileExistsError('case figure exists '+stem)
  fig.suptitle(f"{case['case_id']} · group {case['source_group_id'][:10]} · fixed dev case",fontsize=12)
  fig.savefig(png,dpi=180,format='png');fig.savefig(pdf,format='pdf');plt.close(fig)
  caption=f"Locked qualitative case {case['case_id']} from one distinct previously exposed development source group. Each column is a fixed seed; rows show raw axes 0 and 14. Curves compare ground truth, persistence, and the accepted TS11-D/STATE-CTRL Q0/Q1 predictions. Time is indexed by future step 1–30; physical units are unverified. Case selection used metadata-only IDs before N4 outcomes."
  alt=f"Six line panels for case {case['case_id']}: two raw command axes by three fixed seeds, showing ground truth, persistence and four model-query trajectories."
  write_caption(stem,caption,alt)
  manifest.append({'figure_id':stem,'png_path':str(png),'png_sha256':sha(png),'png_bytes':png.stat().st_size,
   'pdf_path':str(pdf),'pdf_sha256':sha(pdf),'pdf_bytes':pdf.stat().st_size,
   'caption_path':str(R/'captions'/f'{stem}.md'),'caption_sha256':sha(R/'captions'/f'{stem}.md'),
   'alt_text':alt,'figure_data_path':str(case_data.relative_to(R)),'figure_data_sha256':sha(case_data),
   'case_selection_lock_sha256':sha(R/'protocol/CASE_SELECTION_LOCK.json'),
   'development_previously_exposed':True,'final_test_access':0,'human_review':'pending'})
 # 4x4 contact sheet from all locked case pages.
 thumbs=[]
 for case in cases:
  im=Image.open(R/'figures/cases'/f"CASE_{case['case_id']}.png").convert('RGB')
  im.thumbnail((350,205),Image.Resampling.LANCZOS);thumbs.append((case,im.copy()))
 sheet=Image.new('RGB',(4*365,4*235),'white');draw=ImageDraw.Draw(sheet)
 for idx,(case,im) in enumerate(thumbs):
  x=(idx%4)*365;y=(idx//4)*235
  sheet.paste(im,(x,y+20));draw.text((x+4,y+3),f"{case['case_id']} · {case['source_group_id'][:8]}",fill='black')
 contact=R/'figures/N4_CASE_CONTACT_SHEET.png'
 if contact.exists():raise FileExistsError('contact sheet exists')
 sheet.save(contact)
 alt='Contact sheet of the 14 preselected distinct dev source-group cases; each tile previews the six-panel axes-0/14, three-seed trajectory page.'
 cap='Contact sheet for the 14 locked qualitative cases. The individual case pages contain all three seeds and GT, persistence, Q0 and Q1 outputs for both fixed heads; the contact sheet is an overview only.'
 write_caption('N4_CASE_CONTACT_SHEET',cap,alt)
 manifest.append({'figure_id':'N4_CASE_CONTACT_SHEET','png_path':str(contact),'png_sha256':sha(contact),
  'png_bytes':contact.stat().st_size,'caption_path':str(R/'captions/N4_CASE_CONTACT_SHEET.md'),
  'caption_sha256':sha(R/'captions/N4_CASE_CONTACT_SHEET.md'),'alt_text':alt,
  'figure_data_path':str(case_data.relative_to(R)),'figure_data_sha256':sha(case_data),
  'case_selection_lock_sha256':sha(R/'protocol/CASE_SELECTION_LOCK.json'),
  'development_previously_exposed':True,'final_test_access':0,'human_review':'pending'})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
 check_preconditions()
 if args.dry_run:
  print(json.dumps({'status':'PASS_PLOT_DRY_RUN','figures_planned':5,'case_pages':14,
   'targets_opened':False,'final_test_access':0},ensure_ascii=False,indent=2));return
 for d in ('figures','figure_data','captions'):
  (R/d).mkdir(parents=True,exist_ok=True)
 manifest=[]
 with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
  'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
  plot_gvi(manifest)
  plot_position_oracle(manifest)
  plot_fd(manifest)
  plot_axis_heatmap(manifest,'level_mse','level-MSE','N4_LEVEL_GAIN_RAW_AXIS')
  plot_axis_heatmap(manifest,'time_centered_error_mse','time-centered-MSE','N4_TIMECENTER_GAIN_RAW_AXIS')
  case_figures(manifest)
 wjson(R/'figures/FIGURE_MANIFEST.json',{'schema_version':1,'status':'PASS_EXPORTS_CREATED',
  'figure_count':len(manifest),'figures':manifest,'development_previously_exposed':True,
  'final_test_access':0,'human_review_status':'pending'})
 # Metadata/render checks; manual image review is a separate recorded step.
 qa=[]
 for m in manifest:
  png=Path(m['png_path'])
  data_path=Path(m['figure_data_path'])
  if not data_path.is_absolute():data_path=R/data_path
  with Image.open(png) as im:
   qa.append({'figure_id':m['figure_id'],'png_exists':png.is_file(),'png_bytes':png.stat().st_size,
    'png_dimensions':list(im.size),'caption_exists':Path(m['caption_path']).is_file(),
    'data_exists':data_path.is_file(),'data_path':str(data_path)})
 if len(qa)!=len(manifest) or not all(x['png_exists'] and x['caption_exists'] and x['data_exists'] for x in qa):
  raise RuntimeError('figure export metadata QA failed')
 wjson(R/'qa/FIGURE_EXPORT_QA.json',{'schema_version':1,'status':'PASS_EXPORT_METADATA_QA',
  'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'figure_count':len(qa),'figures':qa,'agent_visual_review':'pending',
  'human_visual_review':'pending','final_test_access':0})
 print(json.dumps({'status':'PASS_FIGURES_CREATED','figure_count':len(manifest),
  'case_pages':14,'contact_sheet':str(R/'figures/N4_CASE_CONTACT_SHEET.png'),
  'case_curve_rows':len(C(R/'figure_data/CASE_CURVES.csv')),'target_case_anchors':14,
  'final_test_access':0,'human_review':'pending'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
