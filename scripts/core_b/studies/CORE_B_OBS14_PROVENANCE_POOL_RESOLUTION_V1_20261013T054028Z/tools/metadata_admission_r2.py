#!/usr/bin/env python3
"""Metadata-only R2 admission. Never opens prefix or target payloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    "candidate_record_id", "dataset_id", "dataset_revision", "dataset_role",
    "proposed_split", "allowed_use", "dataset_registry_entry_id",
    "registry_entry_sha256", "explicit_nonfinal_pool_id", "registration_evidence_ref",
    "source_group_id", "source_group_aliases", "session_id", "session_aliases",
    "independence_evidence_ref", "session_independence_verified", "exposure_fit",
    "exposure_normalizer", "exposure_cv", "exposure_dev", "related_dev_selection",
    "final_test", "heldout", "record_signature_sha256", "producer_code_sha256",
    "source_schema_sha256", "conversion_sha256", "timestamp_basis",
    "timestamp_evidence_ref", "physical_config_status", "frame_index_path",
    "frame_index_sha256", "prefix_capture_path", "prefix_capture_sha256",
    "future_labels_path", "future_labels_sha256", "hard_qa_anchor_count",
]
FALSE_VALUES = {"false", "0", "no"}
HEX64 = set("0123456789abcdef")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_aliases(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    try:
        decoded = json.loads(value)
        if isinstance(decoded, list):
            return [str(x).strip() for x in decoded if str(x).strip()]
        if isinstance(decoded, str):
            return [decoded.strip()] if decoded.strip() else []
    except json.JSONDecodeError:
        pass
    return [x.strip() for x in value.replace(";", "|").split("|") if x.strip()]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def row_reasons(row: dict[str, str], signature_sha: str,
                prior_identities: set[str]) -> list[str]:
    reasons: list[str] = []
    required_text = (
        "candidate_record_id", "dataset_id", "dataset_revision", "dataset_registry_entry_id",
        "registry_entry_sha256", "explicit_nonfinal_pool_id", "registration_evidence_ref",
        "source_group_id", "session_id", "independence_evidence_ref", "timestamp_evidence_ref",
        "frame_index_path", "frame_index_sha256", "prefix_capture_path", "prefix_capture_sha256",
        "future_labels_path", "future_labels_sha256",
    )
    for field in required_text:
        if not (row.get(field) or "").strip():
            reasons.append(f"missing_{field}")
    if row.get("dataset_role") != "prospective_nonfinal":
        reasons.append("dataset_role_not_project_prospective_nonfinal")
    if row.get("proposed_split") != "prospective_eval":
        reasons.append("split_not_explicit_prospective_eval")
    if "prospective_eval" not in (row.get("allowed_use") or ""):
        reasons.append("project_authority_does_not_allow_prospective_eval")
    for field in ("registry_entry_sha256", "frame_index_sha256", "prefix_capture_sha256", "future_labels_sha256",
                  "producer_code_sha256", "source_schema_sha256", "conversion_sha256"):
        value = (row.get(field) or "").lower()
        if len(value) != 64 or not set(value) <= HEX64:
            reasons.append(f"invalid_{field}")
    if (row.get("record_signature_sha256") or "").lower() != signature_sha:
        reasons.append("record_signature_mismatch")
    if not (row.get("source_schema_sha256") or "").lower() == "a5754b47bb819533b2268a38159416f9e19d4e0a6c6f4a47254315d2fbb42cb0":
        reasons.append("source_schema_field_map_mismatch")
    if not (row.get("conversion_sha256") or "").lower() == "9afc1f6b143e6e25c59423be6056ffd196a87f61de8fd35e10af991d7a3b5bed":
        reasons.append("numeric_conversion_contract_mismatch")
    if row.get("timestamp_basis") != "nominal_30hz_frame_grid":
        reasons.append("timestamp_basis_not_locked_nominal_frame_grid")
    if not (row.get("physical_config_status") or "").strip():
        reasons.append("physical_configuration_status_missing")
    if (row.get("session_independence_verified") or "").strip().lower() not in {"true", "1", "yes"}:
        reasons.append("session_independence_not_verified")
    for field in ("exposure_fit", "exposure_normalizer", "exposure_cv", "exposure_dev",
                  "related_dev_selection", "final_test", "heldout"):
        if (row.get(field) or "").strip().lower() not in FALSE_VALUES:
            reasons.append(f"{field}_not_verified_clear")
    try:
        if int(row.get("hard_qa_anchor_count", "0")) < 1:
            reasons.append("no_metadata_eligible_anchor")
    except (TypeError, ValueError):
        reasons.append("invalid_hard_qa_anchor_count")
    identity_values = {row.get("source_group_id", ""), row.get("session_id", "")}
    identity_values.update(parse_aliases(row.get("source_group_aliases", "")))
    identity_values.update(parse_aliases(row.get("session_aliases", "")))
    if (identity_values - {""}) & prior_identities:
        reasons.append("prior_exposure_or_alias_overlap")

    # The registration evidence is a metadata-only custody/provenance receipt.
    # It may be read as JSON metadata; payload paths are never opened here.
    ref = (row.get("registration_evidence_ref") or "").strip()
    ref_path = (ROOT / ref).resolve() if ref and not Path(ref).is_absolute() else Path(ref).resolve() if ref else None
    if not ref or ref_path is None:
        reasons.append("registration_evidence_file_missing")
    elif not ref_path.is_relative_to(ROOT.resolve()):
        reasons.append("registration_evidence_outside_work_root")
    elif not ref_path.is_file():
        reasons.append("registration_evidence_file_missing")
    else:
        try:
            evidence = json.loads(ref_path.read_text(encoding="utf-8"))
            if evidence.get("r2_compatibility_verified") is not True:
                reasons.append("candidate_record_compatibility_not_verified")
            if evidence.get("nonfinal_authority_verified") is not True:
                reasons.append("project_nonfinal_authority_not_verified")
            if evidence.get("prefix_label_custody") != "separated_prefix_only_and_scorer_only":
                reasons.append("prefix_label_custody_not_verified")
            if evidence.get("session_independence_verified") is not True:
                reasons.append("independence_evidence_not_verified")
        except (OSError, json.JSONDecodeError):
            reasons.append("registration_evidence_not_readable_json_metadata")
    return sorted(set(reasons))


def run_admission(root: Path) -> dict:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    table_path = root / "catalog/CANDIDATE_REGISTRATION.csv"
    identities_path = root / "catalog/PRIOR_EXPOSURE_IDENTITIES.csv"
    signature_path = root / "provenance/LEGACY_RECORD_SIGNATURE.json"
    with table_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    signature_sha = sha256(signature_path)
    prior: set[str] = set()
    with identities_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prior.add(row.get("source_group_id", ""))
            if row.get("episode_index"):
                prior.add(row["episode_index"])
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(header))
    row_decisions = []
    uf = UnionFind()
    eligible_records = []
    reasons_count: Counter[str] = Counter()
    if not missing_columns:
        for row in rows:
            reasons = row_reasons(row, signature_sha, prior)
            reasons_count.update(reasons)
            key = row.get("candidate_record_id", "").strip()
            if key:
                uf.find(key)
            ids = [key, row.get("source_group_id", "").strip(), row.get("session_id", "").strip()]
            ids += parse_aliases(row.get("source_group_aliases", ""))
            ids += parse_aliases(row.get("session_aliases", ""))
            ids = [x for x in ids if x]
            for alias in ids[1:]:
                uf.union(ids[0], alias)
            row_decisions.append({"candidate_record_id": key or "UNKNOWN", "eligible": not reasons,
                                  "reason_codes": reasons})
            if not reasons:
                eligible_records.append(row)
    else:
        reasons_count["registration_header_missing_columns"] += len(missing_columns)

    eligible_components = {uf.find(r["candidate_record_id"].strip()) for r in eligible_records}
    strict_clusters = len(eligible_components)
    parent_lock = json.loads((root / "protocol/INHERITED_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS_METADATA_ONLY" if not missing_columns else "FAIL_REGISTRATION_SCHEMA",
        "decision": "NO_STRICT_POOL" if strict_clusters < 270 else "STRICT_POOL_AVAILABLE",
        "active_scope": "R2_recorded_schema_external",
        "candidate_records": len(rows),
        "eligible_records": len(eligible_records),
        "strict_independent_clusters": strict_clusters,
        "n_required": 270,
        "n_gap": max(0, 270 - strict_clusters),
        "new_target_outcomes_read": 0,
        "prefix_payloads_read": 0,
        "protected_payloads_read": 0,
        "final_heldout_access": 0,
        "registration_csv_sha256": sha256(table_path),
        "prior_identity_manifest_sha256": sha256(identities_path),
        "legacy_record_signature_sha256": signature_sha,
        "inherited_protocol_lock_sha256": sha256(root / "protocol/INHERITED_PROTOCOL_LOCK.json"),
        "parent_model_sha256": parent_lock["frozen_model_sha256"],
        "missing_registration_columns": missing_columns,
        "reason_counts": dict(sorted(reasons_count.items())),
        "candidate_decisions": row_decisions,
        "method": "CSV and referenced JSON metadata only; no prefix/future-label paths opened or hashed",
        "process_cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "catalog/ADMISSION_RESULT_R2.json")
    args = parser.parse_args()
    root = args.root.resolve()
    result = run_admission(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_METADATA_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
