#!/usr/bin/env python3
"""CPU-only acceptance for all V3 prediction bundles."""

from __future__ import annotations
import os
import csv,json
from pathlib import Path
import numpy as np
import torch

EXP=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
CELLS=("S-HIST","F-POOL-D","F-TOK-D","F-TOK-S","V-TOK-D-A","V-TOK-S-A")
SEEDS=(20260950,20260951,20260952); HORIZONS=(30,60)


def main()->int:
 with np.load(EXP/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z: split=z["split_code"].copy(); ids=z["anchor_id"].copy(); primary_dev=np.flatnonzero(split==1)
 with np.load(EXP/"data/V3_SUPPLEMENTARY_1S_INPUTS.npz",allow_pickle=False) as z: supp_ids=z["anchor_id"].copy()
 rows=[]; accepted=0
 for seed in SEEDS:
  for horizon in HORIZONS:
   for cell in CELLS:
    run_id=f"V3-{cell}-H{horizon}-S{seed}"; pred=EXP/f"predictions/{run_id}"; meta=json.loads((pred/"PREDICTION_EXPORT_COMPLETE.json").read_text()); checkpoint=EXP/f"runs/{run_id}/{'formal_pilot_retry1' if seed==20260950 else 'formal'}/checkpoint_last.pt"; payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    with np.load(pred/"primary.npz",allow_pickle=False) as z: p={k:z[k].copy() for k in z.files}
    primary_ok=meta.get("status")=="PASS" and payload.get("epoch")==40 and payload.get("optimizer_steps")==11120 and p["predictions_native"].shape==(len(primary_dev),horizon,28) and len(np.unique(p["anchor_id"]))==len(primary_dev) and np.array_equal(p["anchor_id"],ids[primary_dev]) and np.isfinite(p["predictions_native"]).all() and meta.get("state_unchanged") is True
    supp_ok=True; supp_n=0
    if horizon==30:
     with np.load(pred/"supplementary_1s.npz",allow_pickle=False) as z: s={k:z[k].copy() for k in z.files}
     supp_n=len(supp_ids); supp_ok=s["predictions_native"].shape==(supp_n,30,28) and len(np.unique(s["anchor_id"]))==supp_n and np.array_equal(s["anchor_id"],supp_ids) and np.isfinite(s["predictions_native"]).all()
    ok=bool(primary_ok and supp_ok); accepted+=int(ok); rows.append({"run_id":run_id,"seed":seed,"horizon_steps":horizon,"cell":cell,"primary_rows":len(primary_dev),"supplementary_rows":supp_n,"checkpoint_epoch":payload.get("epoch"),"optimizer_steps":payload.get("optimizer_steps"),"primary_ok":primary_ok,"supplementary_ok":supp_ok,"state_unchanged":meta.get("state_unchanged"),"status":"PASS" if ok else "FAIL","final_test_access":0})
 (EXP/"analysis").mkdir(parents=True,exist_ok=True)
 with (EXP/"analysis/PREDICTION_ACCEPTANCE.csv").open("w",encoding="utf-8",newline="") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 value={"status":"PASS" if accepted==36 else "FAIL","accepted":f"{accepted}/36","primary_dev_windows":int(len(primary_dev)),"supplementary_1s_windows":int(len(supp_ids)),"final_test_access":0}; (EXP/"analysis/PREDICTION_ACCEPTANCE.json").write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(value,sort_keys=True)); return 0 if accepted==36 else 1


if __name__=="__main__": raise SystemExit(main())
