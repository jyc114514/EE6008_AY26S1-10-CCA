from __future__ import annotations
import os
import csv,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiment_contracts import legal_frame_indices

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
RAW=PROJECT/'data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d'
SEEDS=(20260950,20260951,20260952)

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def load_predictions(model,ids):
 vals=[]
 for seed in SEEDS:
  p=V3/'predictions'/f'V3-{model}-H30-S{seed}'/'primary.npz'
  with np.load(p,allow_pickle=False) as z:
   found=z['anchor_id'].astype('U64');pred=z['predictions_native'].copy()
  lookup={x:i for i,x in enumerate(found)}
  if set(lookup)!=set(ids):raise RuntimeError(f'reference bundle IDs differ: {model}/{seed}')
  vals.append(pred[np.asarray([lookup[x] for x in ids])])
 return np.mean(np.stack(vals),axis=0)
def read_past_rgb(rel_path,anchor_frame):
 path=(RAW/rel_path).resolve()
 if not path.is_relative_to(RAW.resolve()) or not path.is_file():raise FileNotFoundError(path)
 idx=legal_frame_indices(anchor_frame);wanted={int(idx[0]),int(idx[len(idx)//2]),int(idx[-1])}
 cap=cv2.VideoCapture(str(path))
 if not cap.isOpened():raise RuntimeError(f'cannot open allowlisted dev video: {path}')
 out={};frame=0
 try:
  while frame<=max(wanted):
   ok,bgr=cap.read()
   if not ok:raise RuntimeError(f'decode failed at past frame {frame}: {path}')
   if frame in wanted:out[frame]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
   frame+=1
 finally:cap.release()
 if set(out)!=wanted:raise RuntimeError('qualitative past-frame selection incomplete')
 return [(i,out[i]) for i in sorted(wanted)]

def main()->int:
 case_path=EXP/'qualitative/CASE_MANIFEST.csv';manifest=read_csv(case_path)
 if not manifest:raise RuntimeError('case manifest empty')
 acceptance=read_csv(EXP/'predictions/PRIMARY_MANIFEST.csv')
 accepted=[r for r in acceptance if r.get('condition')=='ALT-POOL-R3D18' and r.get('horizon_steps')=='30' and r.get('status')=='accepted']
 if {int(r['seed']) for r in accepted}!=set(SEEDS) or len(accepted)!=3:raise RuntimeError('visual render blocked until all three ALT H30 predictions are accepted')
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  ids=z['anchor_id'].astype('U64').copy();episodes=z['episode_index'].copy();split=z['split_code'].copy();targets=z['targets_native'][:,:30].copy();anchors=z['anchor_command_native'].copy();frames=z['anchor_frame'].copy()
 dev_rows=np.flatnonzero(split==1);dev_ids=ids[dev_rows].tolist();id_to_row={aid:int(r) for aid,r in zip(dev_ids,dev_rows)}
 if len(dev_ids)!=2322:raise RuntimeError('unexpected canonical development population')
 sigma=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
 alt=[]
 for seed in SEEDS:
  r=next(x for x in accepted if int(x['seed'])==seed);path=EXP/r['bundle_path']
  with np.load(path,allow_pickle=False) as z:
   found=z['anchor_id'].astype('U64');pred=z['predictions_native'].copy()
  lookup={x:i for i,x in enumerate(found)}
  if set(lookup)!=set(dev_ids):raise RuntimeError(f'ALT case prediction IDs differ seed={seed}')
  alt.append(pred[np.asarray([lookup[x] for x in dev_ids])])
 alt_mean=np.mean(np.stack(alt),axis=0)
 pred_hist=load_predictions('S-HIST',dev_ids);pred_vj=load_predictions('F-POOL-D',dev_ids)
 persistence=np.broadcast_to(anchors[dev_rows,None,:],(len(dev_rows),30,28)).copy()
 preds={'Persistence':persistence,'History':pred_hist,'V-JEPA':pred_vj,'R3D-18':alt_mean}
 case_row_lookup={x:i for i,x in enumerate(dev_ids)}
 channels=json.loads((EXP/'qualitative/CHANNEL_SELECTION.json').read_text())['selected_indices']
 figures=EXP/'qualitative/figures_layoutfix1';data_dir=EXP/'qualitative/figure_data_layoutfix1';captions=EXP/'qualitative/captions_layoutfix1'
 outdata=[];qa=[]
 for case in manifest:
  aid=case['anchor_id'];i=case_row_lookup[aid];episode=int(episodes[id_to_row[aid]]);frame=int(frames[id_to_row[aid]])
  pop_row=next(r for r in read_csv(EXP/'inventory/ANCHOR_MANIFEST.csv') if r['anchor_id']==aid)
  past=read_past_rgb(pop_row['video_path'],frame)
  gt=targets[id_to_row[aid]];anchor=anchors[id_to_row[aid]]
  errors={name:np.abs(pred[i]-gt)/sigma[None,:] for name,pred in preds.items()}
  vmax=max(float(x.max()) for x in errors.values())
  fig=plt.figure(figsize=(18,13),constrained_layout=True)
  grid=fig.add_gridspec(4,4,height_ratios=[1.0,1.1,1.0,1.0])
  for j,(frame_id,rgb) in enumerate(past):
   ax=fig.add_subplot(grid[0,j]);ax.imshow(rgb);ax.set_title(f'Legal past RGB frame {frame_id}');ax.axis('off')
  legend_ax=fig.add_subplot(grid[0,3]);legend_ax.axis('off')
  channel_handles=[Line2D([0],[0],color=color,linewidth=2,label=f'channel{ch:02d}') for ch,color in zip(channels,['#2b6cb0','#2f855a','#c05621','#805ad5'])]
  model_handles=[Line2D([0],[0],color='#333333',linestyle='-',linewidth=2.0,label='GT'),Line2D([0],[0],color='#333333',linestyle='--',linewidth=1.1,alpha=.75,label='Persistence'),Line2D([0],[0],color='#333333',linestyle='-.',linewidth=1.1,alpha=.75,label='History'),Line2D([0],[0],color='#333333',linestyle=':',linewidth=1.1,alpha=.75,label='V-JEPA'),Line2D([0],[0],color='#333333',linestyle='-',linewidth=1.1,alpha=.75,label='R3D-18 mean of 3 seeds')]
  model_legend=legend_ax.legend(handles=model_handles,title='Line style = model',loc='upper left',fontsize=7,title_fontsize=7,frameon=False,handlelength=2.2,labelspacing=.25,borderaxespad=.1)
  legend_ax.add_artist(model_legend)
  legend_ax.legend(handles=channel_handles,title='Color = channel',loc='lower left',fontsize=7,title_fontsize=7,frameon=False,handlelength=2.2,labelspacing=.25,borderaxespad=.1)
  ax=fig.add_subplot(grid[1,:]);t=np.arange(1,31)/30.0
  colors=['#2b6cb0','#2f855a','#c05621','#805ad5']
  for ch,color in zip(channels,colors):
   ax.plot(t,gt[:,ch],color=color,linewidth=2,label=f'GT Full28.channel{ch:02d}')
   for name,style in [('Persistence','--'),('History','-.'),('V-JEPA',':'),('R3D-18','-')]:
    ax.plot(t,preds[name][i,:,ch],linestyle=style,color=color,alpha=.75,linewidth=1.1,label=f'{name} ch{ch:02d}')
  ax.set_title('Selected train-locked native command channels; units unresolved')
  ax.set_xlabel('Forecast lead (s)');ax.set_ylabel('Source-native command value');ax.grid(alpha=.25)
  for j,name in enumerate(('Persistence','History','V-JEPA','R3D-18')):
   row=2+j//2;col=(j%2)*2
   ax=fig.add_subplot(grid[row,col:col+2]);im=ax.imshow(errors[name].T,aspect='auto',origin='lower',vmin=0,vmax=vmax,cmap='magma')
   ax.set_title(f'{name}: standardized absolute error, all 28 channels');ax.set_xlabel('Future step');ax.set_ylabel('Neutral channel index')
   fig.colorbar(im,ax=ax,fraction=.022,pad=.02)
  tags=case['tags'];fig.suptitle(f"{aid} | {tags} | dev-only, outcome-exposed reference selection",fontsize=14)
  stem=f"CASE_{int(case['render_slot']):02d}_{aid}"
  png=figures/f'{stem}.png'
  if png.exists():raise FileExistsError(png)
  fig.savefig(png,dpi=160);plt.close(fig)
  for local_i,(frame_id,_) in enumerate(past):outdata.append({'anchor_id':aid,'episode_index':episode,'tag':tags,'kind':'rgb_frame','frame_index':frame_id,'row':None,'channel':None,'value':None,'final_test_access':0})
  for name,pred in preds.items():
   for h in range(30):
    for ch in range(28):
     outdata.append({'anchor_id':aid,'episode_index':episode,'tag':tags,'kind':f'{name}_target_prediction_error','frame_index':None,'row':h+1,'channel':ch,'value':float(errors[name][h,ch]),'final_test_access':0})
  for ch in channels:
   for h in range(30):
    outdata.append({'anchor_id':aid,'episode_index':episode,'tag':tags,'kind':'native_trajectory','frame_index':None,'row':h+1,'channel':ch,'value':float(gt[h,ch]),'final_test_access':0})
    for name,pred in preds.items():outdata.append({'anchor_id':aid,'episode_index':episode,'tag':tags,'kind':f'{name}_native_trajectory','frame_index':None,'row':h+1,'channel':ch,'value':float(pred[i,h,ch]),'final_test_access':0})
  caption=f"Development-only qualitative case {aid}. Tags: {tags}. The anchor was selected from exposed V3 references by the locked SHA-256 ordering and train-defined D1/M strata. R3D predictions are the mean of the three fixed H30 seeds. Channels are neutral Full-28 indices; physical names and units are unresolved. Frames shown are the first, middle, and last of the legal 32-frame past clip; no post-anchor frame is used."
  caption_path=captions/f'{stem}.md'
  if caption_path.exists():raise FileExistsError(caption_path)
  caption_path.write_text(caption+'\n')
  qa.append({'anchor_id':aid,'image_path':str(png.relative_to(EXP)),'width_inches':18,'height_inches':13,'raw_past_frames':3,'channels':channels,'models':list(preds),'status':'generated_pending_visual_inspection','final_test_access':0})
 figdata=data_dir/'CASE_FIGURE_DATA.csv'
 if figdata.exists():raise FileExistsError(figdata)
 with figdata.open('w',newline='',encoding='utf-8') as f:
  fields=['anchor_id','episode_index','tag','kind','frame_index','row','channel','value','final_test_access'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(outdata)
 vis=EXP/'qa/VISUAL_QA.json'
 if vis.exists():
  tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');shutil.copyfile(vis,vis.with_name(vis.name+'.bak_'+tag))
 vis.write_text(json.dumps({'status':'generated_pending_visual_inspection','cases':qa,'final_test_access':0},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'GENERATED_PENDING_VISUAL_INSPECTION','unique_cases':len(qa),'figure_data_rows':len(outdata),'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
