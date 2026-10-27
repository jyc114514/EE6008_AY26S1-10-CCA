"""CPU-only plots for the V8.1 fixed-scope dev evidence supplement."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import scorer_adapter as sa

COLORS={'G':'#2455A4','U':'#E69F00','M':'#C43C39','T':'#238B70'}
STYLES={'G':'-','U':'--','M':':','T':'-.'}
LABELS={'G':'G · GT_RETRAIN','U':'U · SELF_SOFT','M':'M · MIX_SOFT','T':'T · MIX50 reference'}
TASKS=(119,94,90,141,230)
METRIC_LABELS={
 'position':'Position error / sigma',
 'internal_fd':'Internal safe-FD error / scale',
 'seam':'Seam safe-FD error / scale',
}


def sha(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def read_csv(path: Path) -> list[dict[str,str]]:
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def save_caption(name: str, title: str, data_path: str, metric: str, split: str='dev', selection: str='all accepted locked receivers') -> str:
 p=ROOT/'captions/generated'/f'{name}.md'; p.parent.mkdir(parents=True,exist_ok=True)
 text=(f'# {title}\n\n'
       f'- Source data: `{data_path}`; SHA-256 `{sha(ROOT/data_path)}`.\n'
       f'- Methods: G, U, M, and T; 3 fixed seeds; split: `{split}`; 185 locked windows across 47 source groups.\n'
       f'- Scale: `{metric}` uses the accepted V4 metric-scale lock.\n'
       f'- Aggregation: equal windows within episode, equal episodes within task, then equal task macro.\n'
       f'- Selection: {selection}.\n'
       '- Seed band: min–max across the three seeds; it is not a confidence interval.\n'
       '- Limits: post-result descriptive dev analysis; coordinates are indices without anatomical labels, and frame indices are not calibrated physical time. The dev labels were already exposed and are not an independent evaluation pool.\n')
 p.write_text(text,encoding='utf-8')
 return p.relative_to(ROOT).as_posix()


def line_values(rows, method: str, scope: str, metric: str, xfield: str, yfield: str='value'):
 selected=[r for r in rows if r['method']==method and r['task_or_ALL']==scope and r['metric']==metric]
 if not selected:return None
 by_seed={}
 for r in selected:
  by_seed.setdefault(int(r['seed']),{})[r[xfield]]=float(r[yfield])
 seed_ids=sorted(by_seed)
 keys=list(by_seed[seed_ids[0]])
 for k in keys:
  if any(k not in by_seed[s] for s in seed_ids):raise RuntimeError(f'incomplete plotted support {method}/{scope}/{metric}/{k}')
 if xfield=='step_or_transition':
  def xnum(k):return int(k.split('->')[0]) if '->' in k else int(k)
 else:xnum=lambda k:int(k)
 keys=sorted(keys,key=xnum)
 values=np.asarray([[by_seed[s][k] for k in keys] for s in seed_ids],dtype=np.float64)
 return np.asarray([xnum(k) for k in keys]),values.mean(axis=0),values.min(axis=0),values.max(axis=0)


def decorate(ax, xkind: str, gain: bool=False):
 ax.grid(True,axis='y',alpha=.24,linewidth=.7)
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 if gain:ax.axhline(0,color='#333333',lw=.9,alpha=.7)
 if xkind=='step':
  ax.set_xlim(1,30);ax.set_xticks([1,5,10,15,20,25,30]);ax.set_xlabel('Future-frame index (recorded frame number)')
 elif xkind=='transition':
  ax.set_xlim(1,29);ax.set_xticks([1,5,10,15,20,25,29]);ax.set_xticklabels(['1→2','5→6','10→11','15→16','20→21','25→26','29→30'])
  ax.set_xlabel('Future-internal transition')
 else:
  ax.set_xlim(-.5,27.5);ax.set_xticks([0,4,9,14,19,24,27]);ax.set_xlabel('Coordinate index (no anatomy mapping asserted)')


def draw_panel(ax, rows, scope: str, metric: str, xfield: str, gain: bool=False, show_legend: bool=False):
 xkind='step' if xfield=='step_or_transition' and metric=='position' else ('transition' if xfield=='step_or_transition' else 'axis')
 for method in sa.METHODS:
  q=line_values(rows,method,scope,metric,xfield,'gain_G_minus_method' if gain else 'value')
  if q is None:continue
  xs,mean,lo,hi=q
  ax.fill_between(xs,lo,hi,color=COLORS[method],alpha=.13,linewidth=0)
  ax.plot(xs,mean,color=COLORS[method],linestyle=STYLES[method],lw=1.8 if method!='G' else 2.1,label=LABELS[method])
 decorate(ax,xkind,gain)
 ax.set_ylabel('G loss − method loss\n(positive favors method)' if gain else METRIC_LABELS[metric])
 if show_legend:ax.legend(frameon=False,ncol=2,fontsize=8,loc='best')


def finish_figure(fig, title: str, note: str='Shading shows min–max across 3 seeds; not a 95% CI.'):
 fig.suptitle(title,fontsize=14,fontweight='bold',y=.99)
 fig.text(.5,.012,note,ha='center',va='bottom',fontsize=8,color='#4d4d4d')
 fig.tight_layout(rect=(0,.045,1,.95))


def savefig(fig, path: Path, report: list[dict], data_path: str, caption: str):
 path.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
 plt.close(fig)
 with Image.open(path) as im:
  im.verify()
 with Image.open(path) as im:
  width,height=im.size
 if width<800 or height<400:raise RuntimeError(f'figure unexpectedly small: {path}/{width}x{height}')
 report.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size,
                'width':width,'height':height,'format':'PNG','source_figure_data':data_path,
                'source_figure_data_sha256':sha(ROOT/data_path),'caption_path':caption,
                'caption_sha256':sha(ROOT/caption),'status':'DECODE_PASS'})


def make_overall_and_task_error_figures(horizon,axis,report):
 for metric in ('position','internal_fd'):
  data='figure_data/horizon_errors.csv'; field='step_or_transition'
  fig,ax=plt.subplots(figsize=(9.2,5.1))
  draw_panel(ax,horizon,'ALL',metric,field,False,True)
  finish_figure(fig,f'Overall {metric.replace("_"," ")} by future step')
  name=f'horizon_{metric}_overall'
  caption=save_caption(name,f'Overall {metric} error by future step',data,METRIC_LABELS[metric])
  savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)

  fig,axes=plt.subplots(2,3,figsize=(14.4,8.5),sharex=True)
  for ax,task in zip(axes.flat,TASKS):
   draw_panel(ax,horizon,str(task),metric,field,False,False);ax.set_title(f'Task {task}',fontsize=10)
  legend_ax=axes.flat[-1];legend_ax.axis('off')
  handles,labels=axes.flat[0].get_legend_handles_labels()
  legend_ax.legend(handles,labels,frameon=False,ncol=2,fontsize=8,loc='center',title='Method')
  finish_figure(fig,f'{metric.replace("_"," ")} error by task and future step')
  name=f'horizon_{metric}_by_task'
  caption=save_caption(name,f'{metric} error by task and future step',data,METRIC_LABELS[metric])
  savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)
 for metric in ('position','internal_fd','seam'):
  data='figure_data/axis_errors.csv'; field='axis'
  fig,ax=plt.subplots(figsize=(9.2,5.1))
  draw_panel(ax,axis,'ALL',metric,field,False,True)
  finish_figure(fig,f'Overall {metric.replace("_"," ")} error by coordinate')
  name=f'axis_{metric}_overall'
  caption=save_caption(name,f'Overall {metric} error by coordinate',data,METRIC_LABELS[metric])
  savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)

  fig,axes=plt.subplots(2,3,figsize=(14.4,8.5),sharex=True)
  for ax,task in zip(axes.flat,TASKS):
   draw_panel(ax,axis,str(task),metric,field,False,False);ax.set_title(f'Task {task}',fontsize=10)
  legend_ax=axes.flat[-1];legend_ax.axis('off')
  handles,labels=axes.flat[0].get_legend_handles_labels()
  legend_ax.legend(handles,labels,frameon=False,ncol=2,fontsize=8,loc='center',title='Method')
  finish_figure(fig,f'{metric.replace("_"," ")} error by task and coordinate')
  name=f'axis_{metric}_by_task'
  caption=save_caption(name,f'{metric} error by task and coordinate',data,METRIC_LABELS[metric])
  savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)


def make_gain_figures(horizon_gains,axis_gains,report):
 data='figure_data/horizon_gains.csv'
 fig,axes=plt.subplots(1,2,figsize=(14,5.2),sharex=False)
 for ax,metric in zip(axes,('position','internal_fd')):
  draw_panel(ax,horizon_gains,'ALL',metric,'step_or_transition',True,False)
  ax.set_title(metric.replace('_',' '),fontsize=11)
  ax.legend(frameon=False,ncol=2,fontsize=8,loc='best')
 finish_figure(fig,'Overall candidate gain by future step','Gain = G error − candidate error; positive favors candidate. Band = 3-seed min–max, not CI.')
 name='horizon_gains_overall';caption=save_caption(name,'Overall candidate gain by future step',data,'position sigma and safe-FD scale')
 savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)

 fig,axes=plt.subplots(5,2,figsize=(14,15),sharex=True)
 for i,task in enumerate(TASKS):
  for j,metric in enumerate(('position','internal_fd')):
   ax=axes[i,j];draw_panel(ax,horizon_gains,str(task),metric,'step_or_transition',True,False)
   ax.set_title(f'Task {task} · {metric.replace("_"," ")}',fontsize=9)
   if i==0 and j==0:ax.legend(frameon=False,ncol=2,fontsize=7,loc='best')
 finish_figure(fig,'Candidate gains by task and future step','Gain = G error − candidate error; positive favors candidate. Bands are 3-seed min–max, not CIs.')
 name='horizon_gains_by_task';caption=save_caption(name,'Candidate gains by task and future step',data,'position sigma and safe-FD scale')
 savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)

 data='figure_data/axis_gains.csv'
 fig,axes=plt.subplots(1,3,figsize=(15,5.1),sharex=True)
 for ax,metric in zip(axes,('position','internal_fd','seam')):
  draw_panel(ax,axis_gains,'ALL',metric,'axis',True,False);ax.set_title(metric.replace('_',' '),fontsize=10)
  ax.legend(frameon=False,ncol=2,fontsize=7,loc='best')
 finish_figure(fig,'Overall candidate gain by coordinate','Gain = G error − candidate error; positive favors candidate. Bands are 3-seed min–max, not CIs.')
 name='axis_gains_overall';caption=save_caption(name,'Overall candidate gain by coordinate',data,'position sigma and safe-FD scale')
 savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)

 fig,axes=plt.subplots(5,3,figsize=(17,16),sharex=True)
 for i,task in enumerate(TASKS):
  for j,metric in enumerate(('position','internal_fd','seam')):
   ax=axes[i,j];draw_panel(ax,axis_gains,str(task),metric,'axis',True,False)
   ax.set_title(f'Task {task} · {metric.replace("_"," ")}',fontsize=9)
   if i==0:ax.legend(frameon=False,ncol=2,fontsize=7,loc='best')
 finish_figure(fig,'Candidate gains by task and coordinate','Gain = G error − candidate error; positive favors candidate. Bands are 3-seed min–max, not CIs.')
 name='axis_gains_by_task';caption=save_caption(name,'Candidate gains by task and coordinate',data,'position sigma and safe-FD scale')
 savefig(fig,ROOT/'figures'/f'{name}.png',report,data,caption)


def make_trajectories(report):
 cases=read_csv(ROOT/'protocol/CASE_SELECTION.csv')
 data_path='figure_data/trajectory_data.csv';data=read_csv(ROOT/data_path)
 if len(cases)!=15 or len(data)!=4530:raise RuntimeError(f'trajectory source support mismatch: {len(cases)}/{len(data)}')
 for case in cases:
  pid=case['plot_id'];case_data=[r for r in data if r['plot_id']==pid]
  fig,axes=plt.subplots(1,2,figsize=(13.6,6.4),sharey=False)
  for ax,coordinate in zip(axes,(int(case['axis_0_14']),int(case['axis_14_28']))):
   subset=[r for r in case_data if int(r['axis'])==coordinate]
   known=[r for r in subset if r['series']=='known_anchor']
   if len(known)!=1:raise RuntimeError(f'{pid}/{coordinate}: expected one known anchor point')
   ax.scatter([0],[float(known[0]['native_value'])],marker='D',s=42,color='#222222',label='Known last-input anchor',zorder=4)
   for method in ('Y','G','U','M','T'):
    line=sorted((r for r in subset if r['series']==method),key=lambda r:int(r['future_frame']))
    if len(line)!=30:raise RuntimeError(f'{pid}/{coordinate}/{method}: expected 30 future points')
    xs=np.asarray([int(r['future_frame']) for r in line]);ys=np.asarray([float(r['native_value']) for r in line])
    color='#111111' if method=='Y' else COLORS[method]
    ax.plot(xs,ys,color=color,lw=2.1 if method in ('Y','G') else 1.6,
            linestyle='-' if method!='T' else '--',marker='o' if method=='Y' else None,
            markersize=2.5,label={'Y':'Y · dev target','G':LABELS['G'],'U':LABELS['U'],'M':LABELS['M'],'T':LABELS['T']}[method])
   ax.set_title(f'Coordinate index {coordinate}',fontsize=10)
   ax.set_xlabel('Future-frame index; 0 is last known input, 1–30 are future frames')
   ax.set_xlim(-.5,30.5);ax.set_xticks([0,5,10,15,20,25,30]);ax.grid(True,alpha=.22)
   ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
   ax.set_ylabel('Native recorded output value')
  post='posthoc failure example' if case['posthoc_failure_example']=='True' else 'hash-selected regular example'
  fig.suptitle(f"Task {case['task_id']} · {case['anchor_id']} · {post}",fontsize=12,fontweight='bold',y=.99)
  handles,labels=axes[0].get_legend_handles_labels()
  fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,.925))
  fig.text(.5,.022,'Seed 202610131 · Native values · recorded frame index, not calibrated physical time · descriptive already-exposed dev Y',ha='center',fontsize=8,color='#4d4d4d')
  fig.tight_layout(rect=(0,.09,1,.86))
  name=f'trajectory_{pid}'
  caption=ROOT/'captions/generated'/f'{name}.md';caption.parent.mkdir(parents=True,exist_ok=True)
  caption.write_text(
   f"# {case['plot_id']} trajectory: task {case['task_id']}\n\n"
   f"- Anchor: `{case['anchor_id']}`; fixed seed: `{case['seed']}`; source group: `{case['source_group']}`.\n"
   f"- Series: known last-input anchor, real dev target Y, G, U, M, and accepted T/MIX50 prediction.\n"
   f"- Coordinates: indices `{case['axis_0_14']}` and `{case['axis_14_28']}`, selected from the train-only equal-task teacher-difference scores; no anatomy mapping is asserted.\n"
   f"- Selection: `{case['selection_reasons']}`; V8 fixed case ledger; posthoc flag `{case['posthoc_failure_example']}`.\n"
   f"- Figure data: `{data_path}`; SHA-256 `{sha(ROOT/data_path)}`. Values are native recorded output coordinates; x-axis is an index, not calibrated physical time. The target is already exposed dev evidence, not an independent pool.\n",encoding='utf-8')
  savefig(fig,ROOT/'figures/trajectories'/f'{pid}.png',report,data_path,caption.relative_to(ROOT).as_posix())


def contact_sheet(paths: list[Path], out_path: Path, cols: int=4, tile_w: int=480, tile_h: int=300):
 pad=16;rows=(len(paths)+cols-1)//cols
 canvas=Image.new('RGB',(cols*tile_w+(cols+1)*pad,rows*tile_h+(rows+1)*pad),'#e9edf2')
 draw=ImageDraw.Draw(canvas)
 for i,path in enumerate(paths):
  im=Image.open(path).convert('RGB');im.thumbnail((tile_w-12,tile_h-42),Image.Resampling.LANCZOS)
  x=pad+(i%cols)*(tile_w+pad);y=pad+(i//cols)*(tile_h+pad)
  bg=Image.new('RGB',(tile_w,tile_h),'white');bg.paste(im,((tile_w-im.width)//2,8))
  canvas.paste(bg,(x,y))
  draw.text((x+5,y+tile_h-27),path.stem,fill='#16202a')
 out_path.parent.mkdir(parents=True,exist_ok=True);canvas.save(out_path,optimize=True)


def main():
 started=time.perf_counter()
 horizon=read_csv(ROOT/'figure_data/horizon_errors.csv')
 axis=read_csv(ROOT/'figure_data/axis_errors.csv')
 horizon_gains=read_csv(ROOT/'figure_data/horizon_gains.csv')
 axis_gains=read_csv(ROOT/'figure_data/axis_gains.csv')
 if (len(horizon),len(axis),len(horizon_gains),len(axis_gains))!=(4248,6048,4248,6048):
  raise RuntimeError('plot input coverage does not match complete fixed support')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                      'axes.titlesize':10,'legend.fontsize':8,'figure.facecolor':'white',
                      'axes.facecolor':'white','savefig.facecolor':'white'})
 report=[]
 make_overall_and_task_error_figures(horizon,axis,report)
 make_gain_figures(horizon_gains,axis_gains,report)
 make_trajectories(report)
 generated=[ROOT/x['path'] for x in report]
 aggregate=[p for p in generated if '/trajectories/' not in p.as_posix()]
 trajectory=[p for p in generated if '/trajectories/' in p.as_posix()]
 contact_sheet(aggregate,ROOT/'qa/AGGREGATE_FIGURE_CONTACT_SHEET.png',cols=4,tile_w=480,tile_h=300)
 contact_sheet(trajectory,ROOT/'qa/TRAJECTORY_FIGURE_CONTACT_SHEET.png',cols=3,tile_w=500,tile_h=330)
 contact_records=[]
 for p in (ROOT/'qa/AGGREGATE_FIGURE_CONTACT_SHEET.png',ROOT/'qa/TRAJECTORY_FIGURE_CONTACT_SHEET.png'):
  with Image.open(p) as im: size=im.size
  contact_records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'size':list(size)})
 review={'schema_version':1,'status':'BUILT_REQUIRES_AGENT_VISUAL_REVIEW','generated_figure_count':len(report),
         'aggregate_figure_count':len(aggregate),'trajectory_figure_count':len(trajectory),
         'methods_in_aggregate_and_gain_figures':list(sa.METHODS),'fixed_case_ids':[p.stem for p in trajectory],
         'position_horizon_frames':30,'internal_fd_transitions':29,'coordinates':28,
         'error_rows_by_method_seed_scope':{'horizon':4248,'axis':6048},
         'gain_sign':'G loss minus candidate loss; positive favors candidate',
         'seed_band':'min-max across 3 seeds; never labeled as CI','no_new_bootstrap':True,
         'human_review_status':'PENDING','agent_visual_review_status':'PENDING',
         'contact_sheets':contact_records,'figures':report,
         'elapsed_seconds':time.perf_counter()-started}
 (ROOT/'qa/FIGURE_REVIEW.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 build={'schema_version':1,'status':'PASS','figures_generated':len(report),'aggregate_figures':len(aggregate),
        'trajectory_figures':len(trajectory),'contact_sheets':len(contact_records),
        'all_pngs_decode_and_size_check_pass':True,'all_figure_caption_paths_exist':all((ROOT/x['caption_path']).is_file() for x in report),
        'all_methods_present':list(sa.METHODS),'split':'locked dev','target_values_written_to_figure_data':True,
        'figure_data_role':'PRIVATE_ALREADY_EXPOSED_DEV_DIAGNOSTIC','elapsed_seconds':time.perf_counter()-started}
 (ROOT/'qa/FIGURE_BUILD_QA.json').write_text(json.dumps(build,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':build['status'],'figure_count':len(report),'aggregate':len(aggregate),
                   'trajectories':len(trajectory),'review_status':review['status'],'elapsed_seconds':build['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
