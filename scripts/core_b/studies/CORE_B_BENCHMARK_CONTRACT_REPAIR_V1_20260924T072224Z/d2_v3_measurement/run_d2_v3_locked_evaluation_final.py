"""Run the locked D2_V3 evaluator with provenance-checked constants and CSV adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate_d2_v3_panel as evaluator


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


matcher = json.loads((ROOT / "D2_V3_MATCHER_LOCKED.json").read_text())
runner_lock = json.loads((ROOT / "D2_V3_RUNNER_LOCKED.json").read_text())
if runner_lock.get("status") != "LOCKED_BEFORE_D2_V3_PREDICTION_EVALUATION":
    raise RuntimeError("runner code was not locked before evaluation")
if runner_lock.get("runner_code_sha256") != sha256_file(Path(__file__)):
    raise RuntimeError("runner source changed after its lock")
if runner_lock.get("matcher_protocol_sha256") != sha256_file(ROOT / "D2_V3_MATCHER_LOCKED.json"):
    raise RuntimeError("matcher lock changed after runner lock")
if matcher.get("evaluator_code_sha256") != sha256_file(ROOT / "evaluate_d2_v3_panel.py"):
    raise RuntimeError("core evaluator source changed after matcher lock")
if matcher.get("threshold_protocol_sha256") != sha256_file(ROOT / "D2_V3_PROTOCOL_LOCKED.json"):
    raise RuntimeError("train threshold lock changed after matcher lock")

evaluator.SIGMA = np.asarray(
    json.loads((evaluator.V3 / "data/V3_NORMALIZERS.json").read_text())["target_sigma"],
    dtype=np.float64,
)
old_v2 = json.loads((evaluator.RESULT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
evaluator.D2_V2_ON = {"Hand": float(old_v2["theta_on"][0]), "Arm": float(old_v2["theta_on"][1])}
evaluator.D2_V2_OFF = {"Hand": float(old_v2["theta_off"][0]), "Arm": float(old_v2["theta_off"][1])}

_write_csv = evaluator.write_csv


def _write_union_csv(path: Path, rows, fields=None):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fields or [])
    return _write_csv(path, rows, fieldnames)


evaluator.write_csv = _write_union_csv
evaluator.main()
