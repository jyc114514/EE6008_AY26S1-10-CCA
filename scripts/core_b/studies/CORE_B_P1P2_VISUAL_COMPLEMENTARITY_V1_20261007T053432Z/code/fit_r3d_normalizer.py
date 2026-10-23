from __future__ import annotations
import os
import hashlib,json
from pathlib import Path
import numpy as np
from experiment_contracts import episode_equal_moments

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/"repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"

def main()->int:
 with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:
  split=z["split_code"].copy();episodes=z["episode_index"].copy();anchor_ids=z["anchor_id"].copy()
 if set(np.unique(split).tolist())-{0,1}:raise RuntimeError("input includes a split outside canonical train/dev")
 index=np.load(EXP/"cache/R3D_PRIMARY_INDEX.npy",allow_pickle=False)
 union=np.load(EXP/"cache/R3D_FEATURES_UNION.npy",mmap_mode="r")
 features=np.asarray(union[index],dtype=np.float32)
 if features.shape!=(len(split),512) or not np.isfinite(features).all():raise RuntimeError("primary feature bank schema/finite check failed")
 train=split==0;mean,std=episode_equal_moments(features,episodes,train)
 fit_ids=sorted(str(x) for x in anchor_ids[train])
 ids_hash=hashlib.sha256("\n".join(fit_ids).encode()).hexdigest()
 payload={'status':'LOCKED','fit_scope':'primary train only; episode-equal moments, equal anchor weight within episode','fit_episodes':len(np.unique(episodes[train])),'fit_windows':int(train.sum()),'fit_anchor_ids_sha256':ids_hash,'mean':mean.tolist(),'safe_std':std.tolist(),'epsilon':1e-6,'feature_dim':512,'feature_cache_sha256':hashlib.sha256((EXP/"cache/R3D_FEATURES_UNION.npy").read_bytes()).hexdigest(),'dev_used':False,'final_test_access':0}
 out=EXP/"cache/R3D_NORMALIZER.json"
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','fit_episodes':payload['fit_episodes'],'fit_windows':payload['fit_windows'],'fit_ids_sha256':ids_hash,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
