"""Verify the fixed 64-row Core A fresh features against accepted train/dev caches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import core_a_predict as core_a


PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
RUN_ROOT=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
BRIDGE_ROOT=RUN_ROOT/"features/core_a/bridge_attempt_01/artifacts"
FEATURE_SOURCES={
    "dynamic":PROJECT/"features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet",
    "static_repeated_first_frame":PROJECT/"features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z/feature_manifest.parquet",
}
ATOL=1e-5;RTOL=1e-5


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main()->int:
    with (RUN_ROOT/"replay/core_a/core_a_bridge_manifest.csv").open("r",encoding="utf-8-sig",newline="") as stream:
        bridge=list(csv.DictReader(stream))
    selected=[int(row["episode_index"]) for row in bridge]
    if len(selected)!=64 or len(set(selected))!=64:raise ValueError("registered Core A bridge IDs invalid")
    results={}
    for condition,source_path in FEATURE_SOURCES.items():
        source=pd.read_parquet(source_path,columns=["episode_index","feature_path"])
        old={int(row.episode_index):Path(str(row.feature_path)) for row in source.itertuples(index=False)}
        fresh=core_a.read_feature_manifest(BRIDGE_ROOT/"features"/f"feature_manifest_{condition}.csv")
        if [row.episode_index for row in fresh]!=selected:raise ValueError(f"fresh {condition} feature IDs/order differ")
        errors=[]
        for row in fresh:
            if row.episode_index not in old:raise ValueError(f"accepted {condition} cache misses bridge ID {row.episode_index}")
            a,_=core_a.load_feature_tokens(row)
            b,_=core_a.load_feature_tokens(core_a.FeatureRow(row.episode_index,old[row.episode_index]))
            if tuple(a.shape)!=tuple(b.shape) or not np.allclose(a.numpy(),b.numpy(),atol=ATOL,rtol=RTOL):raise ValueError(f"{condition} feature mismatch at {row.episode_index}")
            errors.append(float(np.max(np.abs(a.float().numpy()-b.float().numpy()))))
        results[condition]={"rows":64,"allclose":True,"max_abs_difference":max(errors,default=0.0),"fresh_manifest_sha256":sha256_file(BRIDGE_ROOT/"features"/f"feature_manifest_{condition}.csv")}
    receipt={"schema_version":1,"status":"PASS","attempt":"bridge_attempt_01","selection_manifest_sha256":sha256_file(RUN_ROOT/"replay/core_a/core_a_bridge_manifest.csv"),"tolerance":{"atol":ATOL,"rtol":RTOL},"conditions":results,"final_payload_read":False}
    path=RUN_ROOT/"replay/CORE_A_FEATURE_BRIDGE_ACCEPTANCE.json";path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f".{path.name}.{os.getpid()}.part");temp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(temp,path)
    print(json.dumps(receipt,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
