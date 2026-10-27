#!/usr/bin/env python3
"""Fit six deterministic CPU train-only weighted-median task/axis bias tables."""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from v3_common import ROOT,V2,TASK_IDS,FORMAL_SEEDS,read_json,sha

def weighted_median(x,w):
 x=np.asarray(x,dtype=np.float64);w=np.asarray(w,dtype=np.float64)
 if x.ndim!=1 or w.shape!=x.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('invalid weighted median inputs')
 order=np.argsort(x,kind='mergesort');xs=x[order];ws=w[order];total=float(ws.sum())
 if not np.isfinite(total) or total<=0:raise ValueError('invalid weight sum')
 i=int(np.searchsorted(np.cumsum(ws),0.5*total,side='left'))
 return float(xs[min(i,len(xs)-1)]),total

def main():
 train_path=V2/'data/TRAIN_SELECTED_H30.npz'
 with np.load(train_path,allow_pickle=False) as a:
  ids=a['anchor_id'].astype(str).copy();eps=a['episode_id'].astype(np.int64).copy();tasks=a['task_id'].astype(np.int64).copy()
  anchors=a['anchor_native'].astype(np.float64).copy();targets=a['target_native_h30'].astype(np.float64).copy();source_groups=a['source_cluster'].astype(str).copy()
 if targets.shape!=(1280,30,28):raise RuntimeError('accepted V2 train labels have unexpected shape')
 rows=[];summaries=[];model_receipts=[]
 base_map=json.loads((ROOT/'inventory/BASE_STAGE_MAP.json').read_text())
 for base in base_map['formal_bases']:
  model=base['model_id'];seed=int(base['seed'])
  fr=ROOT/'features/formal'/f"{model.replace('-','_')}_s{seed}"/'attempt_01'
  comp=read_json(fr/'FEATURES_COMPLETE.json');ft=load_file(str(fr/'TRAIN_FEATURES.safetensors'),device='cpu')
  feature_rows=[]
  with (fr/'TRAIN_ROWS.csv').open(newline='',encoding='utf-8') as f:feature_rows=list(csv.DictReader(f))
  fids=[r['anchor_id'] for r in feature_rows]
  if comp.get('status')!='PASS' or fids!=ids.tolist() or ft['z_base'].shape!=(1280,30,28):raise RuntimeError(f'formal base train outputs do not align: {model}/{seed}')
  normalizer=read_json(Path(base['normalizer_path']));sigma=np.asarray(normalizer['target_sigma'],dtype=np.float64)
  z=ft['z_base'].numpy().astype(np.float64)
  pred_native=anchors[:,None,:]+sigma[None,None,:]*z
  if not np.isfinite(pred_native).all():raise FloatingPointError('nonfinite formal-base train output')
  task_summaries=[]
  for task in TASK_IDS:
   ix=np.flatnonzero(tasks==task);unique_eps=sorted(set(int(eps[i]) for i in ix));E=len(unique_eps)
   if E==0:raise RuntimeError(f'train-only bias task missing: {task}')
   per_ep=Counter(int(eps[i]) for i in ix);roww=np.asarray([1.0/(E*per_ep[int(eps[i])]) for i in ix],dtype=np.float64)
   timew=np.repeat(roww/30.0,30)
   if abs(float(timew.sum())-1.0)>1e-12:raise RuntimeError(f'bias weights do not sum to 1: {task}')
   res=(targets[ix]-pred_native[ix]).transpose(0,1,2).reshape(-1,28)
   group_count=len(set(source_groups[ix].tolist()))
   axis_bias=[]
   for axis in range(28):
    bias,ws=weighted_median(res[:,axis],timew)
    axis_bias.append(bias)
    rows.append({'model_id':model,'modality':'STATE' if model=='G-S' else 'VIDEO','seed':seed,'task_id':task,
      'action_axis_index':axis,'bias_native':f'{bias:.17g}','weighted_mass':f'{ws:.17g}',
      'train_episodes':E,'train_source_groups':group_count,'train_anchors':len(ix),'future_values':len(ix)*30,
      'base_model_sha256':base['model_sha256'],'base_train_features_sha256':sha(fr/'TRAIN_FEATURES.safetensors'),
      'train_targets_source_sha256':sha(train_path),'normalizer_sha256':base['normalizer_sha256']})
   b=np.asarray(axis_bias,dtype=np.float64)[None,None,:]
   before_fd=np.diff(pred_native[ix],axis=1)
   after_fd=np.diff(pred_native[ix]+b,axis=1)
   invariant=float(np.max(np.abs(before_fd-after_fd)))
   if invariant>1e-12:raise RuntimeError(f'constant train bias changed future-internal FD beyond FP64 tolerance: {model}/{seed}/{task} {invariant}')
   task_summaries.append({'task_id':task,'train_episodes':E,'train_source_groups':group_count,'train_anchors':len(ix),
     'weight_sum':float(timew.sum()),'bias_vector':axis_bias,'future_fd_max_abs_change_fp64':invariant})
  model_receipts.append({'model_id':model,'seed':seed,'status':'PASS','base_model_sha256':base['model_sha256'],
    'normalizer_sha256':base['normalizer_sha256'],'train_feature_sha256':sha(fr/'TRAIN_FEATURES.safetensors'),
    'target_source_sha256':sha(train_path),'calibrated_task_count':5,'bias_parameters':5*28,'tasks':task_summaries})
 p=ROOT/'calibration/BIAS_CONSTANTS.csv'
 import io
 b=io.StringIO(newline='');writer=csv.DictWriter(b,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);csv_content=b.getvalue()
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes().decode('utf-8')!=csv_content:raise RuntimeError(f'preserving differing bias table: {p}')
 else:p.write_text(csv_content,encoding='utf-8')
 receipt={'schema_version':1,'status':'PASS','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),
  'calibration_count':len(model_receipts),'planned_calibration_count':6,'total_bias_constants':len(rows),'bias_constants_per_calibration':140,
  'method':'weighted median of native (target - frozen base) residual over episodes/anchors/future time',
  'weight_each_future_value':'1/(E_task * windows_in_episode * H30)','median_rule':'stable sort; first value whose cumulative weight reaches 0.5; lower value selected at exact boundary',
  'source_role':'full corrected outer-train only (220 episodes, 1280 anchors); no inner-tune-only or dev targets used',
  'labels_entered_base_forward':False,'labels_used_only_for_cpu_bias_fit':True,'bias_is_per_task_axis_constant':True,
  'constant_bias_safe_fd_identity_atol':1e-12,'all_future_fd_identity_checks_pass':True,'calibrations':model_receipts,
  'bias_csv':str(p),'bias_csv_sha256':sha(p),'final_heldout_payload_access':0}
 out=ROOT/'calibration/BIAS_CALIBRATION_COMPLETE.json'
 content=json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if out.exists():
  if out.read_bytes().decode('utf-8')!=content:raise RuntimeError(f'preserving differing bias receipt: {out}')
 else:out.write_text(content,encoding='utf-8')
 print(json.dumps({'status':'PASS','calibrations':len(model_receipts),'bias_constants':len(rows),'safe_fd_invariance':True,
  'dev_targets_read':False,'final_heldout_payload_access':0,'bias_csv_sha256':sha(p)},indent=2))
if __name__=='__main__':main()
