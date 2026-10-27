#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V8=PROJECT/'reports/benchmarks/CORE_B_V8_CONTROLLED_REPLAY_V1_20261016T045154Z'
SEEDS=(202610131,202610132,202610133);METHODS=('A0','AD','V0','VD','MIX0','MIXD')
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part')
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 seal=json.loads((ROOT/'qa/PREDICTION_SEAL.json').read_text())
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('dev_target_archive_opened') is not False:raise RuntimeError('trajectory reuse requires pre-target prediction seal')
 if (ROOT/'figure_data/TRAJECTORY_DATA.csv').exists():raise RuntimeError('trajectory data already exists')
 cases=readcsv(V8/'figure_data/TRAJECTORY_CASES.csv')
 if len(cases)!=5 or len({int(x['task_id']) for x in cases})!=5:raise RuntimeError('V8 metadata case set invalid')
 with np.load(V4/'data/DEV_PREFIXES_NO_LABELS.npz',allow_pickle=False) as z:
  ids=z['anchor_id'].astype(str).tolist();eps=z['episode_id'];tasks=z['task_id'];groups=z['source_cluster'].astype(str)
 if len(ids)!=185 or len(set(ids))!=185:raise RuntimeError('V4 dev prefix support mismatch')
 index={aid:i for i,aid in enumerate(ids)}
 for case in cases:
  aid=case['anchor_id'];i=index[aid]
  if (int(tasks[i]),int(eps[i]),str(groups[i]))!=(int(case['task_id']),int(case['episode_id']),str(case['source_group'])):
   raise RuntimeError('V8 preselected case metadata differs from V9 dev prefix')
 old_target=readcsv(V8/'figure_data/TRAJECTORY_DATA.csv')
 target={}
 for r in old_target:
  if r['method']!='Y_TARGET':continue
  key=(r['anchor_id'],int(r['action_index']),int(r['future_step']))
  target[key]=float(r['native_command_value'])
 pred_arrays={}
 for method in METHODS:
  for seed in SEEDS:
   if method in ('MIX0','MIXD'):pred_path=ROOT/'predictions/dev_derived'/method/f'seed{seed}'/'PREDICTIONS.safetensors'
   else:pred_path=ROOT/'predictions/dev/full_context'/method/f'seed{seed}'/'PREDICTIONS.safetensors'
   pred=load_file(str(pred_path),device='cpu')['prediction_native']
   if tuple(pred.shape)!=(185,30,28):raise RuntimeError('prediction shape mismatch for selected case')
   pred_arrays[(method,seed)]=pred.numpy()
 output=[];case_out=[]
 for ci,case in enumerate(cases):
  aid=case['anchor_id'];task=int(case['task_id']);episode=int(case['episode_id']);group=case['source_group'];idx=index[aid]
  case_out.append({'case_order':ci,'task_id':task,'episode_id':episode,'anchor_id':aid,'source_group':group,
    'case_selection':case['case_selection'],'reused_v8_case_id':case.get('reused_v5_case_id',''),
    'selection_rule':'V8 preselected metadata ID; no V9 outcome selection'})
  for step_index in (0,14):
   future_step=step_index+1
   for action_index in (0,14):
    key=(aid,action_index,future_step)
    if key not in target:raise RuntimeError('V8 target trajectory export missing selected ID/index')
    target_value=target[key]
    base={'task_id':task,'episode_id':episode,'anchor_id':aid,'source_group':group,'step_index':step_index,
      'future_step':future_step,'action_index':action_index,'case_order':ci}
    output.append({**base,'seed':'shared','method':'Y_TARGET','series':'Y_TARGET','value':target_value,
      'source':'accepted_V8_trajectory_target_export'})
    for method in METHODS:
     for seed in SEEDS:
      pred=pred_arrays[(method,seed)]
      val=float(pred[idx,step_index,action_index])
      output.append({**base,'seed':seed,'method':method,'series':method,'value':val,'source':'V9 sealed full-context prediction'})
 writecsv(ROOT/'figure_data/TRAJECTORY_CASES.csv',case_out)
 writecsv(ROOT/'figure_data/TRAJECTORY_DATA.csv',output)
 rec={'schema_version':1,'status':'PASS','created_utc':utc(),'case_count':len(case_out),'selected_future_steps_0_based':[0,14],
  'selected_action_indices':[0,14],'new_v9_prediction_points':len(output)-20,'reused_target_points_from_V8_export':20,
  'V8_trajectory_data_path':str(V8/'figure_data/TRAJECTORY_DATA.csv'),'V8_trajectory_data_sha256':sha(V8/'figure_data/TRAJECTORY_DATA.csv'),
  'V8_case_manifest_path':str(V8/'figure_data/TRAJECTORY_CASES.csv'),'V8_case_manifest_sha256':sha(V8/'figure_data/TRAJECTORY_CASES.csv'),
  'V9_dev_target_archive_opened_or_reopened':False,'additional_dev_loss_evaluations':0,'final_heldout_payload_access':0}
 (ROOT/'qa/TRAJECTORY_CASE_REUSE.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 exposure_path=ROOT/'inventory/OUTCOME_EXPOSURE.json';exposure=json.loads(exposure_path.read_text())
 exposure.update({'v9_trajectory_cases':5,'v9_trajectory_case_target_values_reused_from_V8_export':20,
  'v9_trajectory_case_target_new_archive_reads':0,'v9_dev_target_archive_content_reads':1,'final_heldout_payload_access':0})
 exposure_path.write_text(json.dumps(exposure,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','cases':5,'points':len(output),'V9_dev_target_reads':0},sort_keys=True))
if __name__=='__main__':main()

