"""Bind constants and run the pre-hashed D2_V3 evaluator without changing its method."""

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


runner_lock = json.loads((ROOT / "D2_V3_RUNNER_LOCKED.json").read_text())
matcher_lock = json.loads((ROOT / "D2_V3_MATCHER_LOCKED.json").read_text())
if runner_lock.get("status") != "LOCKED_BEFORE_D2_V3_PREDICTION_EVALUATION":
    raise RuntimeError("runner code was not locked before evaluation")
if runner_lock.get("runner_code_sha256") != sha256_file(Path(__file__)):
    raise RuntimeError("runner source changed after its lock")
if runner_lock.get("matcher_protocol_sha256") != sha256_file(ROOT / "D2_V3_MATCHER_LOCKED.json"):
    raise RuntimeError("matcher lock changed after runner lock")
if matcher_lock.get("evaluator_code_sha256") != sha256_file(ROOT / "evaluate_d2_v3_panel.py"):
    raise RuntimeError("evaluator source changed after matcher lock")

v3_root = evaluator.V3
evaluator.SIGMA = np.asarray(json.loads((v3_root / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
old = json.loads((evaluator.RESULT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
evaluator.D2_V2_ON = {"Hand": float(old["theta_on"][0]), "Arm": float(old["theta_on"][1])}
evaluator.D2_V2_OFF = {"Hand": float(old["theta_off"][0]), "Arm": float(old["theta_off"][1])}
evaluator.main()
