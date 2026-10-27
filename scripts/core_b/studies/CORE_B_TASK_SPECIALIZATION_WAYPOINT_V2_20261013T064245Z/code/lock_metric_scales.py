#!/usr/bin/env python3
"""Bind V2 evaluation scales to the accepted V3 primary-train normalizer."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    source = Path(resolved["v3_root"]) / "data/V3_NORMALIZERS.json"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "LOCKED" or int(payload.get("final_test_access", -1)) != 0:
        raise RuntimeError("canonical V3 metric normalizer is not locked/final0")
    if "primary train windows only" not in str(payload.get("fit_scope", "")):
        raise RuntimeError("canonical metric scales are not bound to primary-train windows")
    sigma = np.asarray(payload["target_sigma"], dtype=np.float32)
    safe_fd = np.asarray(payload["safe_fd_scale"], dtype=np.float32)
    if sigma.shape != (28,) or safe_fd.shape != (28,) or not np.isfinite(sigma).all() or not np.isfinite(safe_fd).all():
        raise RuntimeError("canonical Full-28 metric scales fail shape/finite validation")
    if np.any(sigma <= 0) or np.any(safe_fd <= 0):
        raise RuntimeError("canonical Full-28 metric scales must be positive")
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "LOCKED",
        "metric_contract": "Full-28 absolute position MAE and future-internal 29-difference safe-FD L1; seam reported separately",
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_status": payload["status"],
        "source_fit_scope": payload["fit_scope"],
        "source_fit_ids_sha256": payload["fit_ids_sha256"],
        "source_train_episodes": int(payload["train_episodes"]),
        "source_train_windows": int(payload["train_windows"]),
        "target_sigma": sigma.tolist(),
        "safe_fd_scale": safe_fd.tolist(),
        "native_inverse": "pred_native=anchor_native+training_target_sigma*predicted_residual",
        "training_scale_is_not_metric_scale": True,
        "final_heldout_payload_access": 0,
    }
    path = ROOT / "protocol/METRIC_SCALE_LOCK.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old.pop("created_utc", None)
        comparable = dict(record)
        comparable.pop("created_utc", None)
        if old != comparable:
            raise RuntimeError("existing metric scale lock differs; preserving it for inspection")
    else:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "source_sha256": source_sha,
        "position_scale_dim": len(sigma),
        "safe_fd_scale_dim": len(safe_fd),
        "final_heldout_payload_access": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
