"""Verify 64 fresh Core A features and six dev-subset predictor replays."""

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


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
BRIDGE_ROOT = RUN_ROOT / "features/core_a/bridge_attempt_01/artifacts"
ATOL = 1e-5
RTOL = 1e-5


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def accepted_features(path: Path) -> dict[int,Path]:
    frame=pd.read_parquet(path,columns=["episode_index","feature_path"])
    if frame["episode_index"].duplicated().any(): raise ValueError("duplicate IDs in accepted Core A feature manifest")
    return {int(row.episode_index):Path(str(row.feature_path)) for row in frame.itertuples(index=False)}


def main()->int:
    bridge_rows=read_csv(RUN_ROOT/"replay/core_a/core_a_bridge_manifest.csv")
    expected_ids=[int(row["episode_index"]) for row in bridge_rows]
    if len(expected_ids)!=64 or len(set(expected_ids))!=64: raise ValueError("Core A bridge input must have 64 unique IDs")
    extraction=json.loads((BRIDGE_ROOT/"features/EXTRACTION_RECEIPT.json").read_text(encoding="utf-8"))
    if extraction.get("status")!="FEATURE_EXTRACTION_COMPLETE" or extraction.get("mode")!="dev" or extraction.get("row_count")!=64:
        raise ValueError("Core A bridge feature extraction receipt invalid")
    accepted={
        "dynamic":accepted_features(PROJECT/"features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet"),
        "static_repeated_first_frame":accepted_features(PROJECT/"features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z/feature_manifest.parquet"),
    }
    feature_results={}
    for condition,reference_map in accepted.items():
        manifest=BRIDGE_ROOT/"features"/f"feature_manifest_{condition}.csv"
        rows=core_a.read_feature_manifest(manifest)
        if [row.episode_index for row in rows]!=expected_ids: raise ValueError(f"fresh Core A {condition} ID order differs")
        diffs=[]
        for row in rows:
            if row.episode_index not in reference_map: raise ValueError(f"accepted Core A {condition} feature missing ID {row.episode_index}")
            fresh,_=core_a.load_feature_tokens(row)
            old,_=core_a.load_feature_tokens(core_a.FeatureRow(row.episode_index,reference_map[row.episode_index]))
            if tuple(fresh.shape)!=tuple(old.shape) or not np.allclose(fresh.numpy(),old.numpy(),atol=ATOL,rtol=RTOL):
                raise ValueError(f"Core A {condition} fresh feature mismatch at episode {row.episode_index}")
            diffs.append(float(np.max(np.abs(fresh.float().numpy()-old.float().numpy()))))
        feature_results[condition]={"rows":64,"allclose":True,"max_abs_difference":max(diffs,default=0.0),"fresh_manifest_sha256":sha256_file(manifest)}

    dev_identity=read_csv(RUN_ROOT/"replay/core_a/core_a_dev_identity.csv")
    dev_ids={int(row["episode_id"]) for row in dev_identity}
    dev_bridge_ids=[episode for episode in expected_ids if episode in dev_ids]
    if len(dev_bridge_ids)!=32: raise ValueError("Core A bridge dev subset must contain 32 IDs")
    bridge_index={episode:i for i,episode in enumerate(expected_ids)}
    matrix=read_csv(OUT_ROOT/"protocol/FINAL_MODEL_MATRIX.csv")
    predictor_results=[]
    for model_row in matrix:
        if model_row["core"]!="A" or model_row["program"] not in {"A_DYNAMIC","A_STATIC"}: continue
        program=model_row["program"];seed=int(model_row["seed"])
        fresh_path=BRIDGE_ROOT/"predictions"/program/f"seed{seed}.npz"
        with np.load(fresh_path,allow_pickle=False) as fresh:
            ids=fresh["episode_index"].astype(np.int64);logits=fresh["logits"].astype(np.float64)
        if ids.tolist()!=expected_ids or logits.shape!=(64,120): raise ValueError(f"Core A bridge prediction shape/order invalid: {program}/{seed}")
        accepted_path=Path(model_row["checkpoint_path"]).with_name("predictions_best.npz")
        with np.load(accepted_path,allow_pickle=False) as old:
            old_ids=old["episode_index"].astype(np.int64);old_logits=old["logits"].astype(np.float64);old_top=old["predictions"].astype(np.int64)
        old_order={int(value):i for i,value in enumerate(old_ids.tolist())}
        if any(episode not in old_order for episode in dev_bridge_ids): raise ValueError(f"accepted Core A dev logits miss bridge IDs: {program}/{seed}")
        fresh_idx=np.asarray([bridge_index[episode] for episode in dev_bridge_ids],dtype=np.int64)
        old_idx=np.asarray([old_order[episode] for episode in dev_bridge_ids],dtype=np.int64)
        close=bool(np.allclose(logits[fresh_idx],old_logits[old_idx],atol=ATOL,rtol=RTOL))
        top_match=bool(np.array_equal(np.argmax(logits[fresh_idx],axis=1),old_top[old_idx]))
        record={"program":program,"seed":seed,"dev_bridge_rows":32,"raw_logits_allclose":close,"top1_exact_match":top_match,"max_abs_difference":float(np.max(np.abs(logits[fresh_idx]-old_logits[old_idx]))),"fresh_output_sha256":sha256_file(fresh_path),"accepted_output_sha256":sha256_file(accepted_path)}
        predictor_results.append(record)
        if not close or not top_match: raise ValueError(f"Core A fresh predictor bridge mismatch: {program}/{seed}")
    if len(predictor_results)!=6: raise ValueError("Core A bridge predictor check must cover six cells")
    output={"schema_version":1,"status":"PASS","created_utc":pd.Timestamp.now(tz="UTC").isoformat(),"attempt":"bridge_attempt_01","split_support":{"train":32,"dev":32},"tolerance":{"atol":ATOL,"rtol":RTOL},"features":feature_results,"predictor_replay":predictor_results,"feature_encoder_forward_rows":128,"classifier_forward_rows":6*64,"dev_label_values_read":False,"final_payload_read":False}
    path=RUN_ROOT/"replay/CORE_A_BRIDGE_ACCEPTANCE.json";path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f".{path.name}.{os.getpid()}.part");temp.write_text(json.dumps(output,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(temp,path)
    print(json.dumps({"status":output["status"],"features":feature_results,"predictor_cells":len(predictor_results)},sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
