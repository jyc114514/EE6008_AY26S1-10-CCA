"""Verify full-order Core A logits with fixed fresh-feature substitutions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
MIXED_ROOT=RUN_ROOT/"replay/core_a/mixed_dev_bridge_attempt_01"
BRIDGE_ROOT=RUN_ROOT/"features/core_a/bridge_attempt_01/artifacts"
ATOL=1e-5;RTOL=1e-5


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def read_csv(path:Path)->list[dict[str,str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main()->int:
    inputs=json.loads((MIXED_ROOT/"MIXED_BRIDGE_INPUTS.json").read_text(encoding="utf-8"))
    receipt=json.loads((MIXED_ROOT/"watcher_artifacts/MIXED_BRIDGE_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt.get("status")!="PREDICTIONS_COMPLETE_PENDING_CPU_COMPARISON" or inputs.get("status")!="PREPARED":raise ValueError("Core A mixed bridge receipts are incomplete")
    matrix=read_csv(OUT_ROOT/"protocol/FINAL_MODEL_MATRIX.csv")
    checks=[]
    for row in matrix:
        if row["core"]!="A" or row["program"] not in {"A_DYNAMIC","A_STATIC"}:continue
        program=row["program"];seed=int(row["seed"])
        output=MIXED_ROOT/"watcher_artifacts"/program/f"seed{seed}.npz"
        reference=Path(row["checkpoint_path"]).with_name("predictions_best.npz")
        with np.load(output,allow_pickle=False) as new, np.load(reference,allow_pickle=False) as old:
            new_ids=new["episode_index"].astype(np.int64);new_logits=new["logits"].astype(np.float64)
            old_ids=old["episode_index"].astype(np.int64);old_logits=old["logits"].astype(np.float64);old_top=old["predictions"].astype(np.int64)
        if not np.array_equal(new_ids,old_ids) or new_logits.shape!=(617,120):raise ValueError(f"Core A mixed bridge order/shape mismatch: {program}/{seed}")
        close=bool(np.allclose(new_logits,old_logits,atol=ATOL,rtol=RTOL))
        top_match=bool(np.array_equal(np.argmax(new_logits,axis=1),old_top))
        result={"program":program,"seed":seed,"rows":617,"fresh_feature_substitutions":32,"raw_logits_allclose":close,"top1_exact_match":top_match,"max_abs_difference":float(np.max(np.abs(new_logits-old_logits))),"new_output_sha256":sha256_file(output),"accepted_output_sha256":sha256_file(reference)}
        checks.append(result)
        if not close or not top_match:raise ValueError(f"Core A full-order mixed bridge mismatch: {program}/{seed}")
    if len(checks)!=6:raise ValueError("Core A mixed bridge must include six cells")
    result={"schema_version":1,"status":"PASS","created_utc":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),"attempt":"mixed_dev_bridge_attempt_01","input_receipt_sha256":sha256_file(MIXED_ROOT/"MIXED_BRIDGE_INPUTS.json"),"prediction_receipt_sha256":sha256_file(MIXED_ROOT/"watcher_artifacts/MIXED_BRIDGE_RECEIPT.json"),"tolerance":{"atol":ATOL,"rtol":RTOL},"checks":checks,"final_payload_read":False,"labels_read":False}
    path=RUN_ROOT/"replay/CORE_A_MIXED_BRIDGE_ACCEPTANCE.json";temp=path.with_name(f".{path.name}.{os.getpid()}.part");temp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(temp,path)
    print(json.dumps({"status":result["status"],"checks":len(checks),"max_abs_difference":max(row["max_abs_difference"] for row in checks)},sort_keys=True),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
