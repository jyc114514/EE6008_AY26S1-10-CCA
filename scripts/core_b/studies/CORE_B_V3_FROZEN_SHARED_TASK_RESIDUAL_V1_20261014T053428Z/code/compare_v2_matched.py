#!/usr/bin/env python3
"""Verify new V2 hidden/output capture reproduces accepted V2 baseline predictions."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from v3_common import ROOT,V2,read_json,sha

def rows(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
 stage=read_json(ROOT/'inventory/BASE_STAGE_MAP.json');rec=[]
 dev_path=V2/'data/DEV_PREFIXES_NO_LABELS.npz'
 with np.load(dev_path,allow_pickle=False) as a:
  ids=a['anchor_id'].astype(str);anchors=a['anchor_native'].astype(np.float32);tasks=a['task_id'].astype(np.int64)
 ref=stage['reused_v2_accepted_prediction_references']
 for model in ('G-S','G-V'):
  for seed in (202610131,202610132,202610133):
   key=f'{model}:{seed}';old=ref[key]
   base=next(x for x in stage['formal_bases'] if x['model_id']==model and int(x['seed'])==seed)
   froot=ROOT/'features/formal'/f"{model.replace('-','_')}_s{seed}"/'attempt_01'
   ft=load_file(str(froot/'DEV_FEATURES.safetensors'),device='cpu')
   frows=rows(froot/'DEV_ROWS.csv');orows=rows(Path(old['ids_path']))
   fids=[x['anchor_id'] for x in frows];oids=[x['anchor_id'] for x in orows]
   if len(set(fids))!=len(fids) or len(set(oids))!=len(oids) or set(fids)!=set(oids) or set(fids)!=set(ids.tolist()):
    raise RuntimeError(f'accepted V2/new hidden anchor ID sets differ for {key}')
   if ft['h'].shape!=(185,30,128) or ft['z_base'].shape!=(185,30,28):raise RuntimeError(f'feature shape mismatch for {key}')
   # V2 inference sorts rows by anchor_id; feature extraction preserves archive order. Join by ID explicitly.
   feature_index={anchor:i for i,anchor in enumerate(fids)}
   prefix_index={anchor:i for i,anchor in enumerate(ids.tolist())}
   order=[feature_index[x] for x in oids]
   prefix_order=[prefix_index[x] for x in oids]
   norm=read_json(Path(base['normalizer_path']));sigma=np.asarray(norm['target_sigma'],dtype=np.float32)
   native=anchors[prefix_order,None,:]+sigma[None,None,:]*ft['z_base'][order].numpy()
   old_tensor=load_file(old['prediction_path'],device='cpu')['prediction_native'].numpy()
   if old_tensor.shape!=native.shape:raise RuntimeError(f'old/new prediction shape differs for {key}')
   delta=np.abs(native-old_tensor)
   max_abs=float(delta.max());max_rel=float((delta/np.maximum(np.abs(old_tensor),1e-8)).max())
   allclose=bool(np.allclose(native,old_tensor,rtol=1e-6,atol=1e-6))
   if not allclose:raise RuntimeError(f'V2 matched replay differs beyond atol=1e-6/rtol=1e-6 for {key}: max_abs={max_abs}')
   old_tasks=np.asarray([int(x['task_id']) for x in orows],dtype=np.int64)
   per_task={str(t):float(delta[old_tasks==t].max()) for t in sorted(set(old_tasks.tolist()))}
   rec.append({'model_id':model,'seed':seed,'job_id':old['job_id'],'rows':len(ids),'prediction_ids_sha256':old['ids_sha256'],
    'v2_prediction_sha256':old['prediction_sha256'],'v3_feature_bundle_sha256':sha(froot/'DEV_FEATURES.safetensors'),
    'hidden_shape':list(ft['h'].shape),'max_abs_native_output_difference':max_abs,'max_rel_native_output_difference':max_rel,
    'atol':1e-6,'rtol':1e-6,'allclose':allclose,'per_task_max_abs':per_task,
    'dev_targets_opened':False,'final_heldout_payload_access':0})
 receipt={'schema_version':1,'status':'PASS','comparisons':rec,'formal_model_count':6,'task_views':30,
  'comparator':'native V2 accepted prediction versus anchor + V2 formal target_sigma * captured V3 z_base',
  'dev_target_artifact_opened':False,'dev_target_rows_read':0,'final_heldout_payload_access':0}
 p=ROOT/'qa/V2_MATCHED_BASE_REPLAY.json'
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','models':len(rec),'task_views':sum(x['rows'] for x in rec),
  'max_abs_native_output_difference':max(x['max_abs_native_output_difference'] for x in rec),'dev_targets_opened':False,
  'final_heldout_payload_access':0},indent=2))
if __name__=='__main__':main()
