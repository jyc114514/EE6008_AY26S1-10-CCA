"""Pin source-group proof and exact pair-support audit into the D3-EQ D4 lock."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.pairlock.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    status = json.loads((EQ_ROOT / "STATUS.json").read_text(encoding="utf-8"))
    if status.get("gpu_training_started") is not False:
        raise AssertionError("D4 input protocol can only be finalized before GPU runs")
    group_audit = json.loads((EQ_ROOT / "qa/D3_EQ_D4_SOURCE_GROUP_AUDIT.json").read_text(encoding="utf-8"))
    pair_audit = json.loads((EQ_ROOT / "qa/D3_EQ_D4_PAIR_PROTOCOL_AUDIT.json").read_text(encoding="utf-8"))
    if group_audit.get("status") != "PASS" or pair_audit.get("status") != "PASS":
        raise AssertionError("D4 pair/source-group audit must pass")
    d4_path = EQ_ROOT / "protocol/D3_EQ_D4_PROTOCOL_LOCKED.json"
    d4 = json.loads(d4_path.read_text(encoding="utf-8"))
    d4["source_group"] = {
        "key": "video_sha256_fingerprint",
        "source_group_mapping_sha256": group_audit["dense_panel"]["episode_group_mapping_sha256"],
        "pair_source_group_audit_sha256": sha256_file(EQ_ROOT / "qa/D3_EQ_D4_SOURCE_GROUP_AUDIT.json"),
        "episode_to_source_group_bijection": True,
        "group_count": 118,
    }
    d4["pair_support_audit_sha256"] = sha256_file(EQ_ROOT / "qa/D3_EQ_D4_PAIR_PROTOCOL_AUDIT.json")
    d4["outcome_blind_representative_selection_sha256"] = sha256_file(EQ_ROOT / "protocol/D3_EQ_TRAJECTORY_SELECTION.json")
    d4.pop("canonical_protocol_sha256", None)
    d4["canonical_protocol_sha256"] = sha256_json(d4)
    atomic_json(d4_path, d4)

    main_path = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
    main_lock = json.loads(main_path.read_text(encoding="utf-8"))
    main_lock["d4_rolling_protocol_sha256"] = sha256_file(d4_path)
    main_lock["d4_pair_source_group_audit_sha256"] = sha256_file(EQ_ROOT / "qa/D3_EQ_D4_SOURCE_GROUP_AUDIT.json")
    main_lock["canonical_protocol_sha256"] = "pending_recompute"
    main_lock.pop("canonical_protocol_sha256")
    main_lock["canonical_protocol_sha256"] = sha256_json(main_lock)
    atomic_json(main_path, main_lock)

    report = {
        "status": "PASS",
        "d4_protocol_sha256": sha256_file(d4_path),
        "d3_eq_protocol_sha256": sha256_file(main_path),
        "pair_source_group_mapping_pinned": True,
        "common_absolute_time_support_pinned": True,
        "old_d4_root_written": False,
        "final_test_access": 0,
        "finalized_utc": utc_now(),
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_D4_PROTOCOL_FINALIZATION.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
