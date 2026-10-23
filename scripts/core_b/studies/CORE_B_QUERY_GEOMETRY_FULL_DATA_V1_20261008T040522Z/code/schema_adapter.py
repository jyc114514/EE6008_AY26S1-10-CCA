"""Typed adapters for accepted EE6008 source receipts; preserves source field names."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

@dataclass(frozen=True)
class CanonicalRun:
    run_id: str
    config: str
    seed: int
    horizon_steps: int
    status: str
    epochs: int
    optimizer_updates: int
    scheduler_step_calls: int
    parameter_count: int
    init_hash: str
    checkpoint_sha256: str

@dataclass(frozen=True)
class CanonicalPrediction:
    run_id: str
    split: str
    rows: int
    status: str
    prediction_sha256: str
    checkpoint_sha256: str
    ids_sha256: str


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected regular non-symlink JSON file: {path}")
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def canonical_run(run_id: str, value: dict[str, Any]) -> CanonicalRun:
    """Normalize RUN_COMPLETE aliases without hiding the original schema."""
    config=value.get("config",value.get("model"))
    horizon=value.get("horizon_steps",value.get("horizon"))
    updates=value.get("optimizer_updates",value.get("optimizer_steps"))
    checkpoint=value.get("checkpoint_sha256",value.get("checkpoint_last_sha256"))
    required=(config,value.get("seed"),horizon,value.get("status"),value.get("epochs"),updates,
              value.get("scheduler_step_calls"),value.get("parameter_count"),value.get("init_hash"),checkpoint)
    if any(x is None for x in required):
        raise ValueError(f"RUN_COMPLETE missing required typed fields for {run_id}: {list(value)}")
    return CanonicalRun(run_id=str(run_id),config=str(config),seed=int(value["seed"]),horizon_steps=int(horizon),
                        status=str(value["status"]),epochs=int(value["epochs"]),optimizer_updates=int(updates),
                        scheduler_step_calls=int(value["scheduler_step_calls"]),parameter_count=int(value["parameter_count"]),
                        init_hash=str(value["init_hash"]),checkpoint_sha256=str(checkpoint))


def canonical_prediction(value: dict[str, Any]) -> CanonicalPrediction:
    required=(value.get("run_id"),value.get("split"),value.get("rows"),value.get("status"),
              value.get("prediction_sha256"),value.get("checkpoint_sha256"),value.get("ids_sha256"))
    if any(x is None for x in required):
        raise ValueError(f"prediction sidecar missing typed fields: {list(value)}")
    return CanonicalPrediction(run_id=str(value["run_id"]),split=str(value["split"]),rows=int(value["rows"]),
                               status=str(value["status"]),prediction_sha256=str(value["prediction_sha256"]),
                               checkpoint_sha256=str(value["checkpoint_sha256"]),ids_sha256=str(value["ids_sha256"]))


def canonical_source_entries(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the accepted SOURCE_HASHES.json sources[] mapping."""
    sources=value.get("sources")
    if not isinstance(sources,list):
        raise TypeError("SOURCE_HASHES.sources must be a list")
    normalized=[]
    for index,item in enumerate(sources):
        if not isinstance(item,dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"invalid source entry index={index}")
        normalized.append({'path':str(item['path']),'sha256':str(item['sha256']),
                           'role':str(item.get('role','')),'bytes':int(item.get('bytes',0))})
    return normalized
