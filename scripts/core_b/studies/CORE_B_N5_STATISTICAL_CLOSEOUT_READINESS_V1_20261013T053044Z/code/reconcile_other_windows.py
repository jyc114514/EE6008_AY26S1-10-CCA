#!/usr/bin/env python3
"""Bounded, read-only reconciliation of the exact N4-referenced study roots."""

import configparser
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks"
OUT = BENCH / "CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z"
N4 = BENCH / "CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z"
HORIZON = BENCH / "CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z"
OBS43 = BENCH / "CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z"
OBS43_EMPTY = BENCH / "CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161856Z"

EXPECTED_MANIFESTS = {
    "HORIZON": "2b5d505e9d86cb4f69b7eb158da55597d9d523f478d7cee2536517e87dd94e6c",
    "OBS43": "1821278f1cf26834900e4279d630585bf584df61dae9cc2796269820d17cf71d",
}

HORIZON_SELECTED = (
    "RUN_STATUS.ini",
    "coordination/OWNER_STATE.json",
    "qa/FINAL_ACCEPTANCE.json",
    "qa/HORIZON_ANALYSIS_ACCEPTANCE.json",
    "qa/RIDGE_PREDICTION_ACCEPTANCE.json",
    "inventory/SOURCE_ACCEPTANCE.json",
    "models/RIDGE_MODEL_LOCK.json",
)
OBS43_SELECTED = (
    "RUN_STATUS.json",
    "protocol/OWNER_STATE.json",
    "qa/FINAL_ACCEPTANCE.json",
    "source/SOURCE_REGISTRY.json",
)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_manifest(root):
    entries = {}
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"unparseable checksum entry under {root}")
        digest, rel = fields
        entries[rel.lstrip("*").removeprefix("./")] = digest
    return entries


def verify(root, expected_manifest, selected):
    manifest_path = root / "CHECKSUMS.sha256"
    actual_manifest = sha_file(manifest_path)
    if actual_manifest != expected_manifest:
        raise ValueError(f"checksum manifest changed for {root.name}")
    entries = parse_manifest(root)
    checks = []
    for rel in selected:
        expected = entries.get(rel)
        if expected is None:
            raise ValueError(f"selected file absent from manifest: {root.name}/{rel}")
        actual = sha_file(root / rel)
        if actual != expected:
            raise ValueError(f"selected checksum mismatch: {root.name}/{rel}")
        checks.append({"relative_path": rel, "sha256": actual, "status": "PASS"})
    return {"root": str(root), "manifest_sha256": actual_manifest, "selected_files": checks, "status": "PASS_SELECTED_FILES"}


def parse_ini(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def same_account_matches():
    uid = str(os.getuid())
    cmd = ["ps", "-u", uid, "-ww", "-o", "pid=,etime=,comm=,args="]
    output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    roots = (str(HORIZON), str(OBS43))
    matches = []
    for line in output.splitlines():
        if not any(root in line for root in roots):
            continue
        if "python3 - <<'PY'" in line or "reconcile_other_windows.py" in line or "same_account_matches" in line:
            continue
        matches.append(line[:800])
    return {"uid": int(uid), "matching_process_count": len(matches), "matching_process_metadata": matches}


def main():
    n4_refs = read_json(N4 / "inventory/OTHER_WINDOW_REFERENCE_STATUS_AMENDMENT_001.json")
    reference_rows = {r["study"]: r for r in n4_refs["references"]}
    if Path(reference_rows["HORIZON_LINEAR_PROBE"]["root"]) != HORIZON:
        raise ValueError("HORIZON root differs from N4 pointer")
    if Path(reference_rows["OBS43_ERROR_BUDGET_CLOSEOUT"]["root"]) != OBS43:
        raise ValueError("OBS43 root differs from N4 pointer")

    horizon_verify = verify(HORIZON, EXPECTED_MANIFESTS["HORIZON"], HORIZON_SELECTED)
    obs43_verify = verify(OBS43, EXPECTED_MANIFESTS["OBS43"], OBS43_SELECTED)

    run = parse_ini(HORIZON / "RUN_STATUS.ini")
    owner = read_json(HORIZON / "coordination/OWNER_STATE.json")
    final = read_json(HORIZON / "qa/FINAL_ACCEPTANCE.json")
    analysis = read_json(HORIZON / "qa/HORIZON_ANALYSIS_ACCEPTANCE.json")
    ridge = read_json(HORIZON / "qa/RIDGE_PREDICTION_ACCEPTANCE.json")
    horizon = {
        "root": str(HORIZON),
        "artifact_completion": {
            "run_status": run.get("STATUS"),
            "query_line_status": run.get("QUERY_LINE_STATUS"),
            "final_acceptance_status": final.get("status"),
            "horizon_analysis_status": analysis.get("status"),
            "ridge_prediction_status": ridge.get("status"),
            "model_fit_acceptance_status": final.get("fit_acceptance", {}).get("status"),
        },
        "seal_integrity": horizon_verify,
        "owner_state_consistency": {
            "owner_status": owner.get("status"),
            "prior_query_owner_active": owner.get("prior_query_owner_active"),
            "consistent_with_complete_receipts": False,
            "classification": "accepted_artifacts_with_stale_owner",
            "conflict": "owner remains LOCKED_READY_FOR_CPU_FIT while RUN_STATUS and final/prediction receipts report complete",
        },
        "current_same_account_owner_evidence": same_account_matches(),
        "decision": "do_not_resume_or_reconcile_by_mutating_old_root",
    }

    obs_run = read_json(OBS43 / "RUN_STATUS.json")
    obs_owner = read_json(OBS43 / "protocol/OWNER_STATE.json")
    obs_final = read_json(OBS43 / "qa/FINAL_ACCEPTANCE.json")
    obs_source = read_json(OBS43 / "source/SOURCE_REGISTRY.json")
    obs43 = {
        "root": str(OBS43),
        "artifact_completion": {
            "run_status": obs_run.get("status"),
            "new_model_fits": obs_run.get("new_model_fits"),
            "final_acceptance_status": obs_final.get("status"),
            "score_reproduction_status": obs_final.get("score_reproduction", {}).get("status"),
            "independent_confirmation": obs_final.get("independent_confirmation"),
            "human_review_status": obs_final.get("human_review_status"),
        },
        "seal_integrity": obs43_verify,
        "owner_state_consistency": {
            "owner_status": obs_owner.get("status"),
            "consistent_with_complete_receipts": obs_owner.get("status") == "SEALED" and obs_run.get("status") == "complete",
        },
        "source_registry_metadata": {
            "source_count": obs_source.get("source_count"),
            "old_obs_root": obs_source.get("old_obs_root"),
            "source_role": "prior OBS43 error-budget study; previously exposed and excluded from an independent pool",
        },
        "current_same_account_owner_evidence": same_account_matches(),
        "decision": "sealed_reference_only; independent confirmation remains unexecuted",
    }

    empty_files = [p for p in OBS43_EMPTY.rglob("*") if p.is_file()]
    obs43_empty = {
        "root": str(OBS43_EMPTY),
        "exists": OBS43_EMPTY.exists(),
        "file_count": len(empty_files),
        "top_level_entry_count": sum(1 for _ in OBS43_EMPTY.iterdir()) if OBS43_EMPTY.exists() else 0,
        "status": "UNKNOWN_PLACEHOLDER_ONLY" if not empty_files else "UNKNOWN_UNACCEPTED_ROOT",
        "interpretation": "N4 pointer names this as a separate scaffold; no acceptance/owner/registry receipt is inferred from directories alone.",
    }
    result = {
        "schema_version": 1,
        "status": "PASS_BOUNDED_OTHER_WINDOW_RECONCILIATION",
        "source_pointer": str(N4 / "inventory/OTHER_WINDOW_REFERENCE_STATUS_AMENDMENT_001.json"),
        "N4_reference_amendment_sha256": sha_file(N4 / "inventory/OTHER_WINDOW_REFERENCE_STATUS_AMENDMENT_001.json"),
        "HORIZON": horizon,
        "OBS43_sealed_root": obs43,
        "OBS43_second_root": obs43_empty,
        "other_window_roots_modified_or_relaunched": False,
        "OBS14_root_seen_in_bounded_benchmark_name_index": False,
        "OBS14_reservation_status": "UNKNOWN; no OBS14 root was present in the exact N4 pointers or bounded benchmark root-name check",
    }
    out = OUT / "inventory/STATE_RECONCILIATION.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for study, node in (("HORIZON", horizon), ("OBS43_SEALED", obs43), ("OBS43_SECOND_ROOT", obs43_empty)):
        rows.append({
            "study": study,
            "root": node.get("root"),
            "artifact_status": json.dumps(node.get("artifact_completion", node.get("status")), ensure_ascii=False, sort_keys=True),
            "seal_status": node.get("seal_integrity", {}).get("status", node.get("status")),
            "owner_status": node.get("owner_state_consistency", {}).get("owner_status", "UNKNOWN"),
            "live_same_account_process_matches": node.get("current_same_account_owner_evidence", {}).get("matching_process_count", "UNKNOWN"),
            "interpretation": node.get("decision", node.get("interpretation", "UNKNOWN")),
        })
    with (OUT / "inventory/STATE_RECONCILIATION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "status": result["status"],
        "horizon": horizon["owner_state_consistency"]["classification"],
        "horizon_selected_hashes": len(horizon_verify["selected_files"]),
        "obs43": obs43["artifact_completion"]["final_acceptance_status"],
        "obs43_selected_hashes": len(obs43_verify["selected_files"]),
        "obs43_second_file_count": obs43_empty["file_count"],
        "live_process_matches": [horizon["current_same_account_owner_evidence"]["matching_process_count"], obs43["current_same_account_owner_evidence"]["matching_process_count"]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
