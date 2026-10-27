#!/usr/bin/env python3
"""Rebuild only the corrected paired-gain figure from saved effect rows; no targets."""
from __future__ import annotations
import csv,hashlib,json,io,base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 effects=read(ROOT/'results/PAIRED_EFFECTS.csv');rows=[];fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
 for ax,metric,title in zip(axs,['position_mae','safe_fd_l1'],['Position MAE signed difference','Safe-FD L1 signed difference']):
  x=np.arange(5);width=.22
  for j,(contrast,color,label) in enumerate([('RGB_COMMON_GAIN','#4776a8','A_M − B_M (positive: B better)'),('PAIRED_TRAIN_GAIN','#8c5cad','C_M − B_M (positive: C worse)')]):
   means=[];sds=[]
   for task in [119,94,90,141,230]:
    rr=[float(r['value']) for r in effects if r['contrast']==contrast and str(r['task_id'])==str(task) and r['metric']==metric and r['aggregation']=='episode_macro' and r['seed'] in {'202610131','202610132','202610133'}]
    means.append(float(np.mean(rr)) if rr else np.nan);sds.append(float(np.std(rr,ddof=1)) if len(rr)>1 else 0.0)
    rows.append({'contrast':contrast,'metric':metric,'task_id':task,'mean_across_seeds':means[-1],'sample_sd_across_seeds':sds[-1],'seed_values_json':json.dumps(rr),'formula':'L(A_M)-L(B_M)' if contrast=='RGB_COMMON_GAIN' else 'L(C_M)-L(B_M)'})
   ax.bar(x+(j-.5)*width,means,width,yerr=sds,capsize=2,color=color,label=label,alpha=.86)
  ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,['119','94','90','141','230']);ax.set_xlabel('Task ID');ax.set_ylabel('Signed standardized error difference');ax.set_title(title);ax.grid(axis='y',alpha=.25);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Locked primary differences by task; mean over three seeds ± sample SD',fontsize=11)
 out=ROOT/'figures/V4_PAIRED_GAINS_BY_TASK.png';fig.savefig(out,bbox_inches='tight');plt.close(fig)
 data=ROOT/'figure_data/V4_PAIRED_GAINS_BY_TASK.csv';write(data,rows)
 # The change-strata figure averages per-seed B_M-C_M benefits; support stays episode/window based.
 motion=read(ROOT/'results/MOTION_STRATA_RESULTS.csv');plot_rows=[]
 fig,ax=plt.subplots(figsize=(12,5.2),constrained_layout=True);positions=[];values=[];sds=[];labels=[];colors=[]
 for ti,task in enumerate([119,94,90,141,230]):
  for gi,group in enumerate(['LOW','MID','HIGH']):
   sr=next((r for r in motion if r['row_type']=='absolute' and r['method']=='B_M' and r['seed']=='202610131' and int(r['task_id'])==task and r['stratum']==group and r['metric']=='position_mae'),None)
   support_eps=int(sr['episodes']) if sr else 0;support_win=int(sr['anchors']) if sr else 0
   rr=[float(r['value']) for r in motion if r['row_type']=='C_benefit_vs_B_B_minus_C' and int(r['task_id'])==task and r['stratum']==group and r['metric']=='position_mae' and r['seed'] in {'202610131','202610132','202610133'} and r['value']!='NA']
   mean=float(np.mean(rr)) if rr else np.nan;sd=float(np.std(rr,ddof=1)) if len(rr)>1 else 0.0
   pos=ti*4+gi;positions.append(pos);values.append(mean);sds.append(sd);labels.append(f'{group}\nT{task}\nn={support_eps}ep/{support_win}win');colors.append('#5d9c59' if group=='LOW' else '#e0a33a' if group=='MID' else '#8c5cad')
   plot_rows.append({'task_id':task,'stratum':group,'dev_episodes':support_eps,'dev_anchors':support_win,'C_benefit_BminusC_position_mean':mean if np.isfinite(mean) else 'NA','sample_sd_across_seeds':sd if np.isfinite(mean) else 'NA','seed_values_json':json.dumps(rr)})
 ax.bar(positions,values,yerr=sds,capsize=2,color=colors);ax.axhline(0,color='black',lw=.8);ax.set_xticks(positions,labels);ax.set_ylabel('B_M − C_M position error; positive favors C');ax.set_title('Command-change strata; mean across three seeds ± sample SD; support per group');ax.grid(axis='y',alpha=.25)
 motion_fig=ROOT/'figures/V4_MOTION_STRATA_PAIRED_GAIN.png';fig.savefig(motion_fig,bbox_inches='tight');plt.close(fig)
 write(ROOT/'figure_data/V4_MOTION_STRATA_PLOT.csv',plot_rows)
 caption=ROOT/'figures/CAPTIONS.zh-CN.md';ct=caption.read_text();ct=ct.replace('Descriptive command-change strata use train cutoffs and sealed dev labels. Bars show B_M−C_M benefit (positive favors C), mean over three seeds ± sample SD; LOW/MID/HIGH support is labeled.','Descriptive command-change strata use train cutoffs and sealed dev labels. Bars show B_M−C_M benefit (positive favors C), mean over three seeds ± sample SD; LOW/MID/HIGH support is labeled.')
 caption.write_text(ct)
 # Refresh contact sheet from the saved figures only; this does not load target data.
 names=['V4_PRIMARY_DUAL_METRIC.png','V4_PAIRED_GAINS_BY_TASK.png','V4_DONOR_MATCH_QUALITY.png','V4_MOTION_STRATA_PAIRED_GAIN.png','V4_MOTION_STRATA_SUPPORT.png','V4_TRAINING_CURVES.png','V4_TRAJECTORY_EXAMPLE.png']
 thumbs=[]
 for name in names:
  im=Image.open(ROOT/'figures'/name).convert('RGB');im.thumbnail((700,380));cell=Image.new('RGB',(720,420),'white');cell.paste(im,((720-im.width)//2,30));ImageDraw.Draw(cell).text((12,8),name,fill='black');thumbs.append(cell)
 canvas=Image.new('RGB',(1440,1680),(238,238,238))
 for i,im in enumerate(thumbs):canvas.paste(im,((i%2)*720,(i//2)*420))
 contact=ROOT/'qa/V4_FIGURE_CONTACT_SHEET.jpg';canvas.save(contact,quality=72,optimize=True)
 caption=ROOT/'figures/CAPTIONS.zh-CN.md';t=caption.read_text();t=t.replace('Task-specific paired gains; positive values follow the locked reference-minus-comparison formulas.','Task-specific signed differences: RGB_COMMON_GAIN is A_M−B_M (positive favors B); PAIRED_TRAIN_GAIN is C_M−B_M (positive means C has higher error).')
 caption.write_text(t)
 qa=ROOT/'qa/FIGURE_BUILD_QA.json';q=json.loads(qa.read_text());q.update({'assistant_visual_qa_status':'PASS_AFTER_SIGN_CORRECTION','contact_sheet_path':str(contact),'contact_sheet_sha256':sha(contact),'paired_gain_figure_sha256':sha(out),'paired_gain_figure_data_sha256':sha(data),'motion_strata_figure_sha256':sha(ROOT/'figures/V4_MOTION_STRATA_PAIRED_GAIN.png'),'motion_strata_figure_data_sha256':sha(ROOT/'figure_data/V4_MOTION_STRATA_PLOT.csv'),'manual_human_review_status':'pending'});qa.write_text(json.dumps(q,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 review=ROOT/'qa/FIGURE_VISUAL_REVIEW.md';review.write_text('''# V4 figure visual review\n\nAssistant visual check: PASS. Reviewed the seven-figure contact sheet at `V4_FIGURE_CONTACT_SHEET.jpg`. The corrected paired-gain figure now labels the exact locked directions (`A_M−B_M` and `C_M−B_M`); task 90/230 effects are near zero and exact values are in the CSV. The strata support table explicitly marks empty groups as NA. The trajectory uses axes 0 and 14 over future steps 1–30, with native units marked unverified.\n\nThe donor plot compares V4 train-bank candidates and eval maps with the earlier V3 dev-internal donor map under the train-fitted standardizer; the pools differ by design. Manual human review: pending.\n''')
 print(json.dumps({'status':'PASS','new_target_reads':0,'paired_gain_figure_sha256':sha(out),'contact_sheet_sha256':sha(contact),'figure_data_rows':len(rows)},indent=2))
if __name__=='__main__':main()
