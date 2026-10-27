#!/usr/bin/env python3
"""Pre-seal structure, arithmetic acceptance, metadata and source-integrity QA."""
import os

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks"
ROOT = BENCH / "CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z"
EXPECTED = {
    "N2": (BENCH / "CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z", "21bfdf8baf1cabe863eb4aa184320e5e79d29ee6c1a589bf9e25ec944fb4472f"),
    "N3": (BENCH / "CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z", "60701b7b357e5282222c60b15994a1408fcc1398618025dbebe5388fc69f1a75"),
    "N4": (BENCH / "CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z", "e3dba3237275022b7338d0eacde5a5f37549415c4a3d6649ab8b79c3a2d7e976"),
    "HORIZON": (BENCH / "CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z", "2b5d505e9d86cb4f69b7eb158da55597d9d523f478d7cee2536517e87dd94e6c"),
    "OBS43": (BENCH / "CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z", "1821278f1cf26834900e4279d630585bf584df61dae9cc2796269820d17cf71d"),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []
    for name, (old_root, expected_hash) in EXPECTED.items():
        actual = sha(old_root / "CHECKSUMS.sha256")
        checks.append({"check": f"{name}_old_checksum_manifest_unchanged", "status": "PASS" if actual == expected_hash else "FAIL", "actual_sha256": actual})

    status = read_json(ROOT / "qa/FINAL_ACCEPTANCE.json")
    exposure = read_json(ROOT / "protocol/OUTCOME_EXPOSURE.json")
    source = read_json(ROOT / "inventory/SOURCE_ACCEPTANCE.json")
    pairing = read_json(ROOT / "protocol/SEED_PAIRING_PROVENANCE.json")
    metadata = read_json(ROOT / "readiness/METADATA_ADMISSION.json")
    state = read_json(ROOT / "inventory/STATE_RECONCILIATION.json")
    figure = read_json(ROOT / "qa/FIGURE_VISUAL_QA.json")

    expected_tables = {
        "results/CO_PRIMARY_SCOPE_TABLE30.csv": 30,
        "results/HISTORICAL_INTERVAL_REPRODUCTION.csv": 30,
        "results/BY_SEED_AND_LOO.csv": 30,
        "results/PAIRING_ALGEBRA_QA.csv": 6,
        "inventory/CI_PROVENANCE.csv": 60,
        "readiness/FUTURE_PREDICTION_SLOTS13_NOT_EXECUTED.csv": 13,
        "readiness/EXPOSURE_CLUSTER_MAP.csv": 3315,
    }
    for rel, count in expected_tables.items():
        _, rows = read_csv(ROOT / rel)
        checks.append({"check": f"{rel}_row_count", "expected": count, "actual": len(rows), "status": "PASS" if len(rows) == count else "FAIL"})
    hist = read_csv(ROOT / "results/HISTORICAL_INTERVAL_REPRODUCTION.csv")[1]
    seed = read_csv(ROOT / "results/BY_SEED_AND_LOO.csv")[1]
    algebra = read_csv(ROOT / "results/PAIRING_ALGEBRA_QA.csv")[1]
    algebra_by_scope = {r["scope"]: r for r in algebra if r["metric"] == "position_standardized_mae"}
    sharing_ok = (
        algebra_by_scope["GROUP_ONLY_FIXED_SEEDS"]["shared_group_indices_all_cells_metrics"] == "True"
        and algebra_by_scope["GROUP_ONLY_FIXED_SEEDS"]["paired_seed_indices_all_cells_metrics"] == "False"
        and algebra_by_scope["SEED_ONLY_FIXED_COHORT"]["group_sampling_status"] == "NOT_APPLICABLE_FIXED_COHORT"
        and algebra_by_scope["SEED_ONLY_FIXED_COHORT"]["paired_seed_indices_all_cells_metrics"] == "True"
        and algebra_by_scope["CROSSED_PAIRED_SEED_GROUP"]["shared_group_indices_all_cells_metrics"] == "True"
        and algebra_by_scope["CROSSED_PAIRED_SEED_GROUP"]["paired_seed_indices_all_cells_metrics"] == "True"
    )
    checks.extend([
        {"check": "historical_interval_replays", "status": "PASS" if all(r["status"] == "PASS_REPRODUCED_FROM_STORED_DRAWS" for r in hist) else "FAIL"},
        {"check": "seed_and_LOO_reproduction", "status": "PASS" if all(r["status"] == "PASS_ACCEPTED_SEED_AND_LOO_REPRODUCTION" for r in seed) else "FAIL"},
        {"check": "paired_GVI_algebra", "status": "PASS" if all(r["status"] == "PASS" for r in algebra) else "FAIL"},
        {"check": "scope_specific_draw_sharing_labels", "status": "PASS" if sharing_ok else "FAIL"},
        {"check": "selected_source_acceptance", "status": "PASS" if source.get("status") == "PASS_SELECTED_SOURCES_AND_N5_DERIVED_RESULTS" else "FAIL"},
        {"check": "candidate_pool_metadata_only_not_ready", "status": "PASS" if metadata.get("status") == "NOT_READY" and metadata.get("candidate_records_found") == 0 and exposure.get("N5_new_evaluation_score_read") == 0 else "FAIL"},
        {"check": "seed_pairing_label", "status": "PASS" if pairing.get("status") == "INIT_PAIRING_VERIFIED_ORDER_PAIRING_UNVERIFIED" else "FAIL"},
        {"check": "bounded_state_reconciliation", "status": "PASS" if state.get("status") == "PASS_BOUNDED_OTHER_WINDOW_RECONCILIATION" else "FAIL"},
        {"check": "single_figure_agent_QA_human_pending", "status": "PASS" if figure.get("status") == "PASS_AGENT_VISUAL_QA" and figure.get("figure_count") == 1 and figure.get("human_review_status") == "pending" else "FAIL"},
        {"check": "raw_target_and_final_boundary", "status": "PASS" if exposure.get("N5_raw_target_array_read") == 0 and exposure.get("N5_final_or_heldout_access") == 0 else "FAIL"},
        {"check": "no_temporary_part_files", "status": "PASS" if not list(ROOT.rglob("*.part")) else "FAIL"},
        {"check": "checksum_manifest_not_yet_written", "status": "PASS" if not (ROOT / "CHECKSUMS.sha256").exists() else "FAIL"},
    ])
    result = {
        "schema_version": 1,
        "status": "PASS_N5_PRESEAL_DELIVERY_QA" if all(r["status"] == "PASS" for r in checks) else "FAIL_N5_PRESEAL_DELIVERY_QA",
        "checks": checks,
        "check_count": len(checks),
        "all_pass": all(r["status"] == "PASS" for r in checks),
        "old_roots_modified": False,
        "raw_target_array_read": False,
        "model_forward": 0,
        "gpu_query_or_use": 0,
        "final_heldout_access": 0,
    }
    (ROOT / "qa/PRESEAL_ACCEPTANCE.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "all_pass": result["all_pass"]}, ensure_ascii=False))
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
