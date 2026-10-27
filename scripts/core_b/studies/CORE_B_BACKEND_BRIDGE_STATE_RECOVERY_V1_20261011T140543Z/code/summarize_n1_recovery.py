#!/usr/bin/env python3
"""Recover accepted N1 train metrics and all fixed-seed contrasts."""
from __future__ import annotations
import os
import csv
from pathlib import Path
import numpy as np
ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'))
SEEDS=(20261001,20261002,20261003)
def read(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(path,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def summarize(x):
 x=np.asarray(x,dtype=float);out={'mean_over_fixed_seeds':float(x.mean()),'sample_sd_over_seeds':float(x.std(ddof=1)),'median_over_seeds':float(np.median(x))}
 for i,s in enumerate(SEEDS):out[f'loo_mean_excluding_{s}']=float(np.delete(x,i).mean())
 return out
n1=read(ROOT/'n1_refs/TRAIN_DEV_BY_SEED.csv')
dev=[r for r in n1 if r['split']=='dev' and r['seed'] in {str(s) for s in SEEDS}]
lookup={(r['config'],r['query_variant'],int(r['seed'])):r for r in dev}
metric_map={'position_standardized_mae':'position_standardized_mae','safe_fd_normalized':'safe_fd_normalized'}
source_gains=read(ROOT/'n1_refs/QUERY_GAINS.csv')
gain_lookup={(r['model'],int(r['seed']),r['metric']):float(r['G_Q_old_minus_had']) for r in source_gains if r['split']=='dev' and r['seed'] in {str(s) for s in SEEDS}}
source_i=read(ROOT/'n1_refs/QUERY_MODEL_INTERACTION.csv')
i_lookup={(int(r['seed']),r['metric']):float(r['I_GTS_minus_GSTATE']) for r in source_i if r['split']=='dev' and r['seed'] in {str(s) for s in SEEDS}}
boot=read(ROOT/'n1_refs/BOOTSTRAP_CI.csv')
boot_lookup={(r['metric'],r['estimand']):r for r in boot if r['split']=='dev'}
contrasts=[]
for metric in metric_map:
 vals={name:[] for name in ('G_TS','G_STATE','V_old','V_Had','I')}
 for seed in SEEDS:
  q0ts=float(lookup[('TS11-D','old',seed)][metric]);q1ts=float(lookup[('TS11-D','had',seed)][metric])
  q0st=float(lookup[('STATE-CTRL','old',seed)][metric]);q1st=float(lookup[('STATE-CTRL','had',seed)][metric])
  gts=q0ts-q1ts;gst=q0st-q1st;vold=q0st-q0ts;vhad=q1st-q1ts;i=gts-gst
  if abs(i-(vhad-vold))>1e-9:raise SystemExit(f'identity failed {metric} seed {seed}')
  if abs(gts-gain_lookup[('TS11-D',seed,metric)])>1e-9 or abs(gst-gain_lookup[('STATE-CTRL',seed,metric)])>1e-9 or abs(i-i_lookup[(seed,metric)])>1e-9:raise SystemExit(f'source gain table mismatch {metric} seed {seed}')
  for k,v in zip(vals,(gts,gst,vold,vhad,i)):vals[k].append(v)
 for name,arr in vals.items():
  b=boot_lookup.get((metric,name),{})
  mean=float(np.mean(arr))
  if b and abs(mean-float(b['point_estimate_mean_fixed_seeds']))>1e-9:raise SystemExit(f'bootstrap point mismatch {metric} {name}')
  contrasts.append({'metric':metric,'estimand':name,**{f'seed_{s}':v for s,v in zip(SEEDS,arr)},**summarize(arr),
                    'source_group_bootstrap_ci95_low':b.get('group_bootstrap_ci95_low',''),
                    'source_group_bootstrap_ci95_high':b.get('group_bootstrap_ci95_high',''),
                    'historical_profiles':'Q0 CPU; Q1 CUDA physical GPU 0','final_test_access':0})
write(ROOT/'results/N1_SEED_MEDIAN_ALL_LOO.csv',contrasts)
train=[r for r in read(ROOT/'inventory/N1_TRAIN_METRIC_RECOVERY.csv')]
train_rows=[]
for model in ('TS11-D','STATE-CTRL'):
 for query in ('old','had'):
  group=[r for r in train if r['config']==model and r['query_variant']==query]
  for metric in ('position_standardized_mae','safe_fd_normalized','native_future_internal_fd','tv_ratio_window','direction_accuracy_active'):
   xs=[float(r[metric]) for r in sorted(group,key=lambda z:int(z['seed']))]
   train_rows.append({'model':model,'query_variant':query,'split':'train_in_sample',
                      'metric':metric,**{f'seed_{s}':v for s,v in zip(SEEDS,xs)},**summarize(xs),
                      'n_anchors':group[0]['n_anchors'],'n_episodes':group[0]['n_episodes'],
                      'n_source_groups':group[0]['n_source_groups'],'final_test_access':0})
write(ROOT/'inventory/N1_TRAIN_METRIC_SUMMARY.csv',train_rows)
curve=read(ROOT/'n1_refs/TRAIN_CURVES.csv')
curve_rows=[r for r in curve if int(r['epoch']) in (1,10,20,30,40)]
write(ROOT/'inventory/N1_TRAIN_CURVE_ENDPOINTS.csv',curve_rows)
print({'status':'PASS_N1_METRICS_RECOVERED','train_rows':len(train),'contrast_rows':len(contrasts),'curve_endpoints':len(curve_rows)})
