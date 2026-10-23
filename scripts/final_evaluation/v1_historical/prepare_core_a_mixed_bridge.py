"""Prepare a 617-row dev replay with 32 preselected fresh-feature substitutions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
BRIDGE_ROOT=RUN_ROOT/"features/core_a/bridge_attempt_01/artifacts"
DEV_IDENTITY=RUN_ROOT/"replay/core_a/core_a_dev_identity.csv"
FEATURE_SOURCES={
    "dynamic":PROJECT/"features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet",
    "static_repeated_first_frame":PROJECT/"features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z/feature_manifest.parquet",
}


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def read_csv(path:Path)->list[dict[str,str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def atomic_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:raise ValueError("empty Core A mixed bridge manifest")
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0]);temp=path.with_name(f".{path.name}.{os.getpid()}.part")
    with temp.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="raise");writer.writeheader();writer.writerows(rows);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def main()->int:
    bridge=read_csv(RUN_ROOT/"replay/core_a/core_a_bridge_manifest.csv")
    identity=read_csv(DEV_IDENTITY)
    dev_order=[int(row["episode_id"]) for row in identity]
    dev_set=set(dev_order)
    fresh_ids={int(row["episode_index"]) for row in bridge if int(row["episode_index"]) in dev_set}
    if len(dev_order)!=617 or len(dev_set)!=617 or len(fresh_ids)!=32:raise ValueError("fixed Core A bridge must include 32 unique dev IDs")
    attempt=RUN_ROOT/"replay/core_a/mixed_dev_bridge_attempt_01"
    if attempt.exists():raise FileExistsError(attempt)
    attempt.mkdir(parents=True)
    result={"schema_version":1,"status":"PREPARED","created_utc":pd.Timestamp.now(tz="UTC").isoformat(),"dev_rows":617,"fresh_dev_substitutions":32,"selection_manifest_sha256":sha256_file(RUN_ROOT/"replay/core_a/core_a_bridge_manifest.csv"),"identity_manifest_sha256":sha256_file(DEV_IDENTITY),"conditions":{}}
    for condition,source_path in FEATURE_SOURCES.items():
        source=pd.read_parquet(source_path,columns=["episode_index","feature_path"])
        old={int(row.episode_index):str(row.feature_path) for row in source.itertuples(index=False)}
        fresh_rows=read_csv(BRIDGE_ROOT/"features"/f"feature_manifest_{'static_repeated_first_frame' if condition=='static_repeated_first_frame' else 'dynamic'}.csv")
        fresh={}
        for row in fresh_rows:
            feature_path=Path(row["feature_path"])
            if not feature_path.is_absolute():feature_path=BRIDGE_ROOT/"features"/feature_path
            fresh[int(row["episode_index"])]=str(feature_path.resolve())
        rows=[]
        for episode in dev_order:
            path=fresh.get(episode,old.get(episode))
            if path is None:raise ValueError(f"{condition} source lacks dev episode {episode}")
            rows.append({"episode_index":episode,"feature_path":path})
        manifest=attempt/f"{condition}_dev_mixed_manifest.csv"
        atomic_csv(manifest,rows)
        result["conditions"][condition]={"manifest_path":str(manifest),"manifest_sha256":sha256_file(manifest),"rows":len(rows),"fresh_substitutions":sum(int(row["episode_index"]) in fresh_ids for row in rows)}
    result["fresh_dev_ids_sha256"]=hashlib.sha256("\n".join(map(str,sorted(fresh_ids))).encode()).hexdigest()
    receipt=attempt/"MIXED_BRIDGE_INPUTS.json";temp=receipt.with_name(f".{receipt.name}.{os.getpid()}.part");temp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(temp,receipt)
    print(json.dumps({"status":result["status"],"rows":617,"fresh_dev_substitutions":32,"attempt":str(attempt)},sort_keys=True),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
