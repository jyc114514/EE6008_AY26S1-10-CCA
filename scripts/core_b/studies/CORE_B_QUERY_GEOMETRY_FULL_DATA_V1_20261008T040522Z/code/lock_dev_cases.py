#!/usr/bin/env python3
"""Choose the locked 6-low/6-high dev trajectory cases before model scoring."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 pa=json.loads((ROOT/'qa/PREDICTION_ACCEPTANCE.json').read_text(encoding='utf-8'))
 if pa.get('status')!='PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES':raise RuntimeError('case selection requires all 12 train and dev prediction bundles accepted')
 if pa.get('dev_bundles_accepted')!='12/12' or pa.get('train_bundles_accepted')!='12/12':raise RuntimeError('prediction bundle count gate failed')
 if any(int(x.get('final_test_access',-1))!=0 or int(x.get('heldout_access',-1))!=0 for x in pa.get('bundle_records',[])):
  raise RuntimeError('prediction bundle records include forbidden evaluation access')
 with (ROOT/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:map_rows=list(csv.DictReader(f))
 episode_group={int(x['episode_index']):(x['split'],x['source_group_id']) for x in map_rows}
 ids=[]
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  split=z['split_code'].astype(np.int8,copy=True);episode=z['episode_index'].astype(np.int64,copy=True)
  ids=z['anchor_id'].astype('U96',copy=True);anchor_frame=z['anchor_frame'].astype(np.int64,copy=True);anchor_time=z['anchor_timestamp'].astype(np.float64,copy=True)
  dev_idx=np.flatnonzero(split==1)
  # First new dev target-array open: limited to the authorized primary-dev scoring population.
  targets=z['targets_native'][dev_idx,:30].astype(np.float32,copy=True)
  sigma=json.loads((V3/'data/V3_NORMALIZERS.json').read_text(encoding='utf-8'))['target_sigma']
  sigma=np.asarray(sigma,dtype=np.float32)
  delta=np.diff(targets,axis=1)
  anchor_motion=np.abs(delta/sigma[None,None,:]).mean(axis=(1,2))
  groups=np.asarray([episode_group[int(episode[i])][1] for i in dev_idx],dtype='U128')
  if len(targets)!=2322 or len(np.unique(episode[dev_idx]))!=603 or len(set(groups.tolist()))!=603:
   raise RuntimeError('dev population/source-group manifest mismatch during case selection')
  q=json.loads((ROOT/'protocol/MOTION_ACTIVITY_THRESHOLD.json').read_text(encoding='utf-8'))['strata_thresholds']['30']
  q25=float(q['q25']);q75=float(q['q75'])
  group_score={g:float(np.mean(anchor_motion[groups==g])) for g in np.unique(groups)}
  group_episode={g:int(episode[dev_idx[np.flatnonzero(groups==g)[0]]]) for g in group_score}
  selected_groups={}
  for stratum,predicate in [('low',lambda x:x<=q25),('high',lambda x:x>=q75)]:
   eligible=[g for g,v in group_score.items() if predicate(v)]
   group_rows=[]
   for g in eligible:
    positions=np.flatnonzero(groups==g)
    ranked=[]
    for pos in positions:
     ix=int(dev_idx[pos]);ep=int(episode[ix]);aid=str(ids[ix])
     digest=hashlib.sha256(f'qg-full-v1|{ep}|{aid}'.encode()).hexdigest()
     ranked.append((digest,ix,ep,aid))
    digest,ix,ep,aid=min(ranked)
    group_rows.append((digest,g,ix,ep,aid))
   group_rows.sort(key=lambda x:x[0])
   if len(group_rows)<6:raise RuntimeError(f'{stratum} stratum has only {len(group_rows)} eligible dev groups')
   selected_groups[stratum]=group_rows[:6]
  cases=[]
  for stratum,items in selected_groups.items():
   for rank,(digest,g,ix,ep,aid) in enumerate(items,1):
    cases.append({'case_id':f'{stratum.upper()}-{rank:02d}','stratum':stratum,'stratum_rank':rank,'source_group_id':g,
     'episode_index':ep,'anchor_id':aid,'selection_hash':digest,'dev_group_motion_score':group_score[g],
     'train_q25':q25,'train_q75':q75,'anchor_frame':int(anchor_frame[ix]),'anchor_timestamp_seconds':float(anchor_time[ix]),
     'target_first_frame':int(anchor_frame[ix])+1,'target_last_frame':int(anchor_frame[ix])+30,
     'target_timestamp_formula':'anchor_timestamp + step/30; frame=anchor_frame+step; step=1..30','final_test_access':0})
  if len(cases)!=12 or len({x['source_group_id'] for x in cases})!=12:raise RuntimeError('case list must contain 12 unique source groups')
  out=ROOT/'qualitative/CASE_MANIFEST.csv'
  if out.exists():raise FileExistsError(out)
  with out.open('w',newline='',encoding='utf-8') as f:
   fields=list(cases[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(cases)
  now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
  record={'schema_version':1,'status':'PASS_CASE_IDS_LOCKED_BEFORE_SCORING','created_utc':now,'case_manifest_path':str(out),
   'case_manifest_sha256':sha(out),'case_count':12,'low_cases':6,'high_cases':6,'unique_source_groups':12,
   'selection_algorithms':['group motion score uses dev H30 target only for target-blind case stratification','train q25/q75 thresholds are reused from the accepted train-only threshold record',
    'one anchor per source group chosen by SHA256(qg-full-v1|episode_id|anchor_id)','six groups per low/high stratum ordered by the selected anchor hash'],
   'all_predictions_accepted_before_case_selection':True,'model_prediction_arrays_opened':False,'model_scores_computed':False,
   'dev_target_access_purpose':'pre-scoring case stratification only; dev was previously exposed; exploratory evaluation',
   'dev_target_rows_read':2322,'final_test_access':0,'heldout_access':0,'independent_confirmation':False}
  (ROOT/'qa/CASE_SELECTION_LOCKED.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  access={'schema_version':1,'phase':'case_selection_before_scoring','created_utc':now,'dev_target_rows_read':2322,
   'reason':'select the prelocked 6-low/6-high descriptive case panels using train-fitted thresholds after all target-blind prediction bundles were accepted',
   'dev_previously_exposed':True,'final_test_access':0,'heldout_access':0}
  (ROOT/'qa/DEV_TARGET_ACCESS.json').write_text(json.dumps(access,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print(json.dumps({'status':record['status'],'case_manifest_sha256':record['case_manifest_sha256'],'case_count':len(cases),'low':6,'high':6,'scores_computed':False},indent=2))
if __name__=='__main__':main()
