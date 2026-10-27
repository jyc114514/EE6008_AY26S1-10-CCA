"""Final pre-execution source registry and source-group/bootstrap lock refresh."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = EQ_ROOT / "protocol/D3_EQ_SOURCE_HASHES.json"
LOCK_PATH = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
FINALIZATION_AUDIT = EQ_ROOT / "protocol/D3_EQ_LOCK_FINALIZATION_AUDIT.json"
STATUS_PATH = EQ_ROOT / "STATUS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.final_registry.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def add(entries: dict[str, Any], key: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries[key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> int:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    if status.get("gpu_training_started") is not False:
        raise AssertionError("final source lock refresh is permitted only before Stage0")
    group_audit = json.loads((EQ_ROOT / "qa/D3_EQ_SOURCE_GROUP_AUDIT.json").read_text(encoding="utf-8"))
    dense_audit = json.loads((EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json").read_text(encoding="utf-8"))
    pair_audit = json.loads((EQ_ROOT / "qa/D3_EQ_D4_PAIR_PROTOCOL_AUDIT.json").read_text(encoding="utf-8"))
    job_audit = json.loads((EQ_ROOT / "qa/D3_EQ_STAGE0_JOB_SPEC_AUDIT.json").read_text(encoding="utf-8"))
    if any(item.get("status") != "PASS" for item in (group_audit, dense_audit, pair_audit, job_audit)):
        raise AssertionError("one or more pre-execution source/identity gates are not PASS")

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    primary_groups = group_audit["primary_dev"]
    dense_groups = group_audit["dense_panel"]
    lock["source_group_identity"] = {
        "key": "video_sha256_fingerprint from duplicate-safe V3 split manifest",
        "primary_dev_episode_count": primary_groups["episodes"],
        "primary_dev_source_group_count": primary_groups["source_fingerprint_groups"],
        "primary_dev_episode_to_source_group_bijection": primary_groups["episode_to_source_group_bijection"],
        "primary_dev_mapping_sha256": primary_groups["episode_group_mapping_sha256"],
        "dense_episode_count": dense_groups["episodes"],
        "dense_source_group_count": dense_groups["source_fingerprint_groups"],
        "dense_episode_to_source_group_bijection": dense_groups["episode_to_source_group_bijection"],
        "dense_mapping_sha256": dense_groups["episode_group_mapping_sha256"],
        "evaluation_reduction": "window metrics averaged within episode; episode means averaged within source group; source groups equally weighted",
        "groups_with_multiple_episodes_in_these_panels": 0,
    }
    lock["evaluation"]["bootstrap_unit"] = "video_sha256_fingerprint source group; exact episode/source-group bijection verified for this primary-dev panel"
    lock["evaluation"]["primary_holm_family"] = "4 contrasts x 2 horizons x 2 body regions; pooled paired source-fingerprint-group bootstrap; seed-specific intervals also reported"
    lock["evaluation"]["primary_p_value_procedure"] = "two-sided centered paired source-group bootstrap with add-one smoothing; Holm across the 16 primary tests"
    lock["evaluation"]["parametric_normality_assumption"] = "not used; inference uses paired source-group bootstrap intervals and centered bootstrap p-values"
    d4_lock_path = EQ_ROOT / "protocol/D3_EQ_D4_PROTOCOL_LOCKED.json"
    d4_lock = json.loads(d4_lock_path.read_text(encoding="utf-8"))
    d4_lock["source_group_mapping_sha256"] = dense_groups["episode_group_mapping_sha256"]
    d4_lock["dense_cache_reuse_audit_sha256"] = sha256_file(EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json")
    d4_lock["pair_protocol_audit_sha256"] = sha256_file(EQ_ROOT / "qa/D3_EQ_D4_PAIR_PROTOCOL_AUDIT.json")
    d4_lock.pop("canonical_protocol_sha256", None)
    d4_lock["canonical_protocol_sha256"] = sha256_json(d4_lock)
    atomic_json(d4_lock_path, d4_lock)

    source_doc = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    entries = source_doc["sources"]
    excluded_protocol = {"D3_EQ_PROTOCOL_LOCKED.json", "D3_EQ_SOURCE_HASHES.json", "D3_EQ_LOCK_FINALIZATION_AUDIT.json"}
    excluded_qa = {"D3_EQ_SOURCE_REGISTRY_AUDIT.json", "D3_EQ_HOST_APPROVAL.json", "D3_EQ_RESOURCE_GATE_PRE_FORMAL.json"}
    for path in sorted((EQ_ROOT / "code").glob("*.py")):
        add(entries, f"code/{path.name}", path)
    for folder_name, excluded in (("qa", excluded_qa), ("configs", set()), ("protocol", excluded_protocol)):
        folder = EQ_ROOT / folder_name
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name not in excluded and path.suffix.lower() in {".json", ".csv"}:
                add(entries, path.relative_to(EQ_ROOT).as_posix(), path)
    add(entries, "RESOURCE_ESTIMATE.csv", EQ_ROOT / "RESOURCE_ESTIMATE.csv")
    source_doc["generated_utc"] = utc_now()
    source_doc["coverage"] = {
        "all_code_python_files": True,
        "pre_execution_qa_configs_and_protocols": True,
        "source_group_identity_pinned": True,
        "volatile_host_approval_and_runtime_status_excluded": True,
        "gpu_execution_started": False,
    }
    atomic_json(SOURCE_PATH, source_doc)

    lock["d4_rolling_protocol_path"] = "D3_EQ_D4_PROTOCOL_LOCKED.json"
    lock["d4_rolling_protocol_sha256"] = sha256_file(d4_lock_path)
    lock["source_hashes_sha256"] = sha256_file(SOURCE_PATH)
    lock.pop("canonical_protocol_sha256", None)
    lock["canonical_protocol_sha256"] = sha256_json(lock)
    atomic_json(LOCK_PATH, lock)

    audit = json.loads(FINALIZATION_AUDIT.read_text(encoding="utf-8"))
    audit.update({
        "source_group_identity_pinned": True,
        "dense_and_d4_input_protocols_pinned": True,
        "source_registry_refreshed_utc": utc_now(),
        "source_registry_entries": len(entries),
        "source_hashes_sha256_matches": lock["source_hashes_sha256"] == sha256_file(SOURCE_PATH),
        "canonical_protocol_sha256_valid": lock["canonical_protocol_sha256"] == sha256_json({key: value for key, value in lock.items() if key != "canonical_protocol_sha256"}),
        "final_lock_sha256": sha256_file(LOCK_PATH),
        "training_started": False,
    })
    atomic_json(FINALIZATION_AUDIT, audit)
    status.update({
        "updated_utc": utc_now(),
        "source_hashes_complete": True,
        "source_group_mapping": "PASS_603_PRIMARY_AND_118_DENSE_FINGERPRINT_GROUPS",
        "d4_pair_manifest": "PASS_7200_PAIRS_3600_PER_HORIZON",
        "stage0_job_specs": "PASS_12_SPECS_DRY_RUN_PASS",
        "gpu_state_query": "not_run_requires_explicit_host_approval",
        "gpu_training_started": False,
        "stage0": "0/12",
        "formal_training": "0/36",
        "primary_predictions": "0/36",
        "dense_predictions": "0/36_not_started",
        "final_heldout_access": 0,
        "protocol_file_sha256": sha256_file(LOCK_PATH),
        "next_gate": "explicit host approval, node role check, fresh resource/GPU lease gate before Stage0",
    })
    atomic_json(STATUS_PATH, status)
    report = {
        "status": "PASS",
        "source_registry_entries": len(entries),
        "primary_source_groups": primary_groups["source_fingerprint_groups"],
        "dense_source_groups": dense_groups["source_fingerprint_groups"],
        "d4_pair_rows": pair_audit["pair_rows"],
        "stage0_job_specs": job_audit["stage0_job_count"],
        "lock_sha256": sha256_file(LOCK_PATH),
        "source_hashes_sha256": lock["source_hashes_sha256"],
        "gpu_query_performed": False,
        "training_started": False,
        "final_test_access": 0,
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_SOURCE_REGISTRY_AUDIT.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
