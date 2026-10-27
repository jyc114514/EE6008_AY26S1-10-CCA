from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
def main()->int:
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  target=z['targets_native'][:,:30].copy();episode=z['episode_index'].copy();split=z['split_code'].copy()
 sigma=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
 train=split==0;per=[]
 for e in np.unique(episode[train]):
  rows=train&(episode==e)
  value=np.mean(np.abs(np.diff(target[rows].astype(np.float64),axis=1))/sigma[None,None,:],axis=(0,1))
  per.append(value)
 score=np.mean(np.asarray(per),axis=0)
 nonconstant=[i for i,v in enumerate(score) if np.isfinite(v) and v>1e-12]
 if len(nonconstant)<4:raise RuntimeError('fewer than four nonconstant train-derived channel variation scores')
 ranked=sorted(nonconstant,key=lambda i:(score[i],i));low=ranked[:2];high=sorted(nonconstant,key=lambda i:(-score[i],i))[:2]
 payload={'status':'LOCKED_TRAIN_ONLY','definition':'episode-equal mean absolute H30 target first difference divided by V3 train-only target_sigma; nonconstant threshold 1e-12','fit_episodes':int(len(per)),'fit_train_windows':int(train.sum()),'channel_index_order':'source action array order preserved; physical labels and units unknown','bottom_two_indices':low,'top_two_indices':high,'selected_indices':low+high,'all_scores_standardized_first_difference':score.tolist(),'final_test_access':0}
 out=EXP/'qualitative/CHANNEL_SELECTION.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','selected_channels':low+high,'low':low,'high':high,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
