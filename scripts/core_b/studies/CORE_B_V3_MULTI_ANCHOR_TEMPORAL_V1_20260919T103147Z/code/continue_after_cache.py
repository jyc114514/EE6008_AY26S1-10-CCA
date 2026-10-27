#!/usr/bin/env python3
"""Wait for full cache, accept it, fit normalization, then launch pilot Stage0."""

from __future__ import annotations
import os

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
EXP=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
PYTHON=PROJECT/"envs/jyc/bin/python"


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    part=path.with_name(f".{path.name}.part"); part.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"); part.replace(path)


def main()->int:
    full_manifest=EXP/"cache/CACHE_EXTRACTION_COMPLETE.json"; watcher=EXP/"logs/watcher_V3-FEATURE-FULL.json"; cache=EXP/"cache/V3_TEMPORAL_FEATURES.npy"
    while not full_manifest.is_file() or not watcher.is_file() or json.loads(watcher.read_text()).get("status")!="complete": time.sleep(60)
    atomic_json(EXP/"logs/CACHE_CONTINUATION_STATUS.json",{"status":"cache_complete_running_acceptance","final_test_access":0})
    subprocess.run([str(PYTHON),str(EXP/"code/fit_video_normalizer.py")],cwd=str(PROJECT),check=True)
    features=np.load(cache,mmap_mode="r")
    if tuple(features.shape)!=(20143,2,16,768) or not np.isfinite(features).all(): raise RuntimeError(f"cache shape/finite failed: {features.shape}")
    with np.load(EXP/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as main_data, np.load(EXP/"data/V3_SUPPLEMENTARY_1S_INPUTS.npz",allow_pickle=False) as supp_data:
        expected_ids=np.concatenate((main_data["episode_index"],supp_data["episode_index"]))
    if len(expected_ids)!=features.shape[0]: raise RuntimeError("cache/input row count seam failed")
    d_minus_s=float(np.mean(np.abs(features[:,0].astype(np.float64)-features[:,1].astype(np.float64))))
    disk_bytes=cache.stat().st_size
    started=json.loads(watcher.read_text()).get("started_utc"); completed=json.loads(watcher.read_text()).get("completed_utc"); gpu_hours=None
    if started and completed:
        gpu_hours=(datetime.fromisoformat(completed.replace("Z","+00:00"))-datetime.fromisoformat(started.replace("Z","+00:00"))).total_seconds()/3600.0
    manifest={"status":"PASS","feature_cache":str(cache),"feature_cache_sha256":digest(cache),"shape":list(features.shape),"row_count":int(features.shape[0]),"episode_id_order_sha256":hashlib.sha256(expected_ids.astype(np.int64).tobytes()).hexdigest(),"D_S_mean_abs_difference":d_minus_s,"disk_bytes":disk_bytes,"disk_gib":disk_bytes/(1024**3),"feature_gpu_hours":gpu_hours,"token_layout":{"raw_tokens":[9216,768],"temporal_tokens":[16,768],"spatial_grid":[24,24],"temporal_slots":16},"final_test_access":0}
    atomic_json(EXP/"cache/CACHE_MANIFEST.json",manifest); atomic_json(EXP/"cache/CACHE_ACCEPTED.json",{"status":"PASS","cache_manifest_sha256":digest(EXP/"cache/CACHE_MANIFEST.json"),"disk_gib":manifest["disk_gib"],"feature_gpu_hours":gpu_hours,"final_test_access":0})
    atomic_json(EXP/"logs/CACHE_CONTINUATION_STATUS.json",{"status":"cache_accepted_launching_pilot_stage0","manifest":manifest,"final_test_access":0})
    command=[str(PYTHON),str(EXP/"code/launch_v3_queue.py"),"--pattern","V3-*-S20260950-STAGE0.json","--controller","V3_PILOT_STAGE0"]
    result=subprocess.run(command,cwd=str(PROJECT),check=False)
    atomic_json(EXP/"logs/CACHE_CONTINUATION_STATUS.json",{"status":"pilot_stage0_controller_finished","pilot_returncode":result.returncode,"cache_manifest":manifest,"final_test_access":0})
    return result.returncode


if __name__ == "__main__": raise SystemExit(main())
