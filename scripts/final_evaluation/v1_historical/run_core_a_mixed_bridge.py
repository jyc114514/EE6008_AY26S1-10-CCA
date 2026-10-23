"""Replay all six Core A heads on 617 dev rows with 32 fresh feature substitutions."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
CODE=OUT_ROOT/"code"
MIXED_ROOT=RUN_ROOT/"replay/core_a/mixed_dev_bridge_attempt_01"


def sha256_file(path:Path)->str:
    import hashlib
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def atomic_json(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(temp,path)


def main()->int:
    physical=os.environ.get("GPU_WATCHER_PHYSICAL_GPU","")
    if physical!="3":raise RuntimeError(f"Core A mixed bridge requires physical GPU3; got {physical!r}")
    artifact_root=MIXED_ROOT/"watcher_artifacts"
    artifact_root.mkdir(parents=True,exist_ok=False)
    input_receipt=json.loads((MIXED_ROOT/"MIXED_BRIDGE_INPUTS.json").read_text(encoding="utf-8"))
    if input_receipt.get("status")!="PREPARED" or input_receipt.get("dev_rows")!=617 or input_receipt.get("fresh_dev_substitutions")!=32:
        raise ValueError("mixed dev bridge input receipt is invalid")
    with (OUT_ROOT/"protocol/FINAL_MODEL_MATRIX.csv").open("r",encoding="utf-8-sig",newline="") as stream:
        matrix=list(csv.DictReader(stream))
    cells=sorted([r for r in matrix if r["core"]=="A" and r["program"] in {"A_DYNAMIC","A_STATIC"}],key=lambda r:(r["program"],int(r["seed"])))
    if len(cells)!=6:raise ValueError("Core A mixed bridge must replay six fixed cells")
    record={"schema_version":1,"status":"RUNNING","protocol_id":"EE6008_FINAL_TEST_V1_20261019T041642Z","attempt":"mixed_dev_bridge_attempt_01","started_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"physical_gpu":3,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","unknown"),"logical_device":"cuda:0","input_receipt_sha256":sha256_file(MIXED_ROOT/"MIXED_BRIDGE_INPUTS.json"),"new_training_runs":0,"final_payload_read":False,"prediction_records":[]}
    status=artifact_root/"STATUS.json";atomic_json(status,record)
    try:
        for cell in cells:
            program=cell["program"];seed=int(cell["seed"]);condition="dynamic" if program=="A_DYNAMIC" else "static_repeated_first_frame"
            manifest=Path(input_receipt["conditions"][condition]["manifest_path"])
            output=artifact_root/program/f"seed{seed}.npz"
            command=[sys.executable,str(CODE/"core_a_predict.py"),"--manifest",str(manifest),"--checkpoint",cell["checkpoint_path"],"--checkpoint-sha256",cell["checkpoint_sha256"],"--run-id",f"EE6008_FINAL_TEST_V1_20261019T041642Z_{program}_{seed}_MIXED_BRIDGE617","--output",str(output),"--output-root",str(RUN_ROOT),"--device","cuda","--expected-rows","617"]
            started=time.perf_counter();subprocess.run(command,cwd=PROJECT,check=True,env=os.environ.copy())
            sidecar=output.with_suffix(".json");meta=json.loads(sidecar.read_text(encoding="utf-8"))
            record["prediction_records"].append({"program":program,"seed":seed,"path":str(output),"sha256":meta["output_sha256"],"sidecar_sha256":sha256_file(sidecar),"feature_manifest_sha256":meta["feature_manifest_sha256"],"rows":617,"forward_rows":617,"elapsed_seconds":time.perf_counter()-started})
            atomic_json(status,record)
        if len(record["prediction_records"])!=6:raise ValueError("Core A mixed predictor replay incomplete")
        record["status"]="PREDICTIONS_COMPLETE_PENDING_CPU_COMPARISON";record["completed_utc"]=datetime.now(timezone.utc).isoformat().replace("+00:00","Z");record["classifier_forward_rows"]=6*617
        receipt=artifact_root/"MIXED_BRIDGE_RECEIPT.json";atomic_json(receipt,record);atomic_json(status,record)
        print(json.dumps({"status":record["status"],"prediction_cells":6,"fresh_feature_substitutions_per_condition":32,"physical_gpu":3},sort_keys=True),flush=True)
        return 0
    except Exception as exc:
        record["status"]="FAILED";record["failed_utc"]=datetime.now(timezone.utc).isoformat().replace("+00:00","Z");record["error_type"]=type(exc).__name__;record["error"]=str(exc);atomic_json(status,record);raise


if __name__=="__main__":raise SystemExit(main())
