from __future__ import annotations
import os
import csv,json
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'

def main()->int:
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  anchor=z['anchor_command_native'].copy();targets=z['targets_native'][:,:30].copy();episodes=z['episode_index'].copy();split=z['split_code'].copy();ids=z['anchor_id'].copy();tasks=z['original_task_index'].copy()
 if set(np.unique(split).tolist())-{0,1}:raise RuntimeError('unexpected split code')
 sigma=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
 magnitude=np.mean(np.abs(targets.astype(np.float64)-anchor[:,None,:])/sigma[None,None,:],axis=(1,2))
 train=split==0;dev=split==1
 per_episode=[]
 for e in np.unique(episodes[train]):per_episode.append(float(np.mean(magnitude[train&(episodes==e)])))
 q25=float(np.quantile(per_episode,.25,method='linear'));q75=float(np.quantile(per_episode,.75,method='linear'))
 if q75<=q25:raise RuntimeError('train motion quantiles degenerate')
 group={}
 with (EXP/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):group[int(r['episode_index'])]=r['source_group_id']
 out=EXP/'analysis/P2_MOTION.csv'
 if out.exists():raise FileExistsError(out)
 fields=['anchor_id','episode_index','task','source_group_id','M_h30','stratum','split','final_test_access']
 with out.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for i in np.flatnonzero(dev):
   val=float(magnitude[i]);w.writerow({'anchor_id':str(ids[i]),'episode_index':int(episodes[i]),'task':int(tasks[i]),'source_group_id':group[int(episodes[i])],'M_h30':val,'stratum':'stable' if val<=q25 else ('large' if val>=q75 else 'moderate'),'split':'dev','final_test_access':0})
 summary={'status':'LOCKED_TRAIN_THRESHOLDS','definition':'M=mean(|future-anchor|/train_only_target_sigma) over H30 time and Full-28 channels','train_episodes':int(len(per_episode)),'train_windows':int(train.sum()),'dev_episodes':int(len(set(episodes[dev].tolist()))),'dev_windows':int(dev.sum()),'train_episode_equal_q25':q25,'train_episode_equal_q75':q75,'dev_targets_used_for_diagnostic_strata_only':True,'dev_threshold_tuning':False,'final_test_access':0}
 p=EXP/'analysis/P2_MOTION_LOCK.json'
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
