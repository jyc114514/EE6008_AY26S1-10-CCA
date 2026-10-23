#!/usr/bin/env python3
"""Validate JY-02 Stage0/Stage1 artifacts without launching computation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[0].parent
GROUP = "stage1_retry_20260912T101001Z"
CELLS = ["JY02-S-A-D", "JY02-S-R-D", "JY02-S-A-T", "JY02-S-R-T", "JY02-F-A-D", "JY02-F-R-D", "JY02-F-A-T", "JY02-F-R-T"]


def main() -> int:
    checks=[]; accepted=[]
    smoke=json.loads((ROOT/"STAGE0_SMOKE_RESULTS.json").read_text()); checks.append({"check":"stage0_8_of_8","status":"PASS" if smoke.get("status")=="PASS" and smoke.get("cells")==8 else "FAIL"})
    for cell in CELLS:
        run=ROOT/"runs"/GROUP/cell; status=json.loads((run/"RUN_STATUS.json").read_text()); complete=json.loads((run/"RUN_COMPLETE.json").read_text()); config=json.loads((run/"RUN_CONFIG_LOCKED.json").read_text());
        with np.load(run/"predictions_best.npz",allow_pickle=False) as a: shape=list(a["predictions_standardized"].shape); finite=bool(np.isfinite(a["predictions_standardized"]).all()); n=int(len(a["episode_index"]))
        ok=status.get("status")=="complete" and complete.get("status")=="complete" and shape==[614,30,28] and finite and config.get("raw_video") is False and config.get("encoder_or_lora_modified") is False and config.get("final_test_access")==0 and int(status.get("optimizer_steps",0))>0
        checks.append({"check":cell,"status":"PASS" if ok else "FAIL","N":n,"shape":shape,"finite":finite,"optimizer_steps":status.get("optimizer_steps")});
        if ok: accepted.append({"cell_id":cell,"status":"accepted","N":n,"shape":str(tuple(shape)),"path":str(run),"best_epoch":json.loads((run/"RUN_METRICS.json").read_text())["best_epoch"]})
    decision=json.loads((ROOT/"STAGE1_CANDIDATE_DECISION.json").read_text()); checks.append({"check":"candidate_gate","status":"PASS" if decision.get("candidate_gate")=="pass" and decision.get("selected_candidate")=="JY02-F-R-D" else "FAIL"})
    stage_status=json.loads((ROOT/"runs"/GROUP/"STAGE1_STATUS.json").read_text())
    result={"status":"PASS" if len(accepted)==8 and all(item["status"]=="PASS" for item in checks) else "FAIL","accepted_cells":len(accepted),"expected_cells":8,"checks":checks,"candidate":decision.get("selected_candidate"),"stage2_status":"blocked_pending_explicit_launch_approval","optimizer_steps":int(stage_status.get("optimizer_steps",0)),"final_test_access":0}
    (ROOT/"STAGE1_RUN_ACCEPTANCE.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (ROOT/"STAGE1_RUN_ACCEPTANCE.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["cell_id","status","N","shape","path","best_epoch"]); writer.writeheader(); writer.writerows(accepted)
    print(json.dumps({"status":result["status"],"accepted_cells":len(accepted),"candidate":decision.get("selected_candidate")},sort_keys=True)); return 0 if result["status"]=="PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
