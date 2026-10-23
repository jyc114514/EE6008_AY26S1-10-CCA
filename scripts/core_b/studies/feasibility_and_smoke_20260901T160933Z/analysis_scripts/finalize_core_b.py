"""Final CPU-only QA for the Core B feasibility packet."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED = (
    "README.md", "CORE_B_SOURCE_INVENTORY.csv", "CORE_B_DATA_SCHEMA_AUDIT.md", "CORE_B_DATA_SCHEMA_AUDIT.json",
    "CORE_B_TARGET_CANDIDATES.md", "CORE_B_TARGET_CANDIDATES.json", "CORE_B_TARGET_CANDIDATES.csv",
    "CORE_B_TEMPORAL_ALIGNMENT_QA.csv", "CORE_B_ELIGIBILITY_AND_COVERAGE.csv", "CORE_B_ELIGIBILITY_AND_COVERAGE.csv",
    "CORE_B_SPLIT_AND_LEAKAGE_AUDIT.md", "CORE_B_SPLIT_AND_LEAKAGE_AUDIT.json", "CORE_B_TARGET_CONTRACT_DRAFT.md", "CORE_B_TARGET_CONTRACT_DRAFT.json",
    "CORE_B_METRIC_CONTRACT_DRAFT.md", "CORE_B_METRIC_CONTRACT_DRAFT.json", "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json",
    "CORE_B_FEASIBILITY_DECISION.md", "CORE_B_FEASIBILITY_DECISION.json", "CORE_B_SMOKE_PROTOCOL.md", "CORE_B_SMOKE_PROTOCOL.json",
    "CORE_B_SMOKE_RUN_ACCEPTANCE.csv", "CORE_B_SMOKE_RUN_ACCEPTANCE.json", "CORE_B_SMOKE_RESULTS.md", "CORE_B_SMOKE_RESULTS.json", "CORE_B_SMOKE_RESULTS.csv",
    "CORE_B_GPU_PROFILE.csv", "CORE_B_GPU_PROFILE.json", "CORE_B_FULL_BASELINE_EXECUTION_PROMPT.md",
    "PROFESSOR_MEETING_CORE_A_FREEZE_AND_CORE_B_DECISIONS.md", "ANALYSIS_COMMAND_AUDIT.md", "CHECKSUMS.sha256",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_checksums(root: Path) -> None:
    checksum = root / "CHECKSUMS.sha256"
    checksum.write_text("".join(f"{sha256_file(path)}  {path}\n" for path in sorted(path for path in root.rglob("*") if path.is_file() and path != checksum)), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core-a-harvest", type=Path, required=True)
    parser.add_argument("--core-a-freeze-md", type=Path, required=True)
    parser.add_argument("--core-a-freeze-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "status": "pass" if passed else "FAIL", "evidence": evidence})

    missing = sorted({name for name in REQUIRED if not (root / name).is_file()})
    check("required_files", not missing, {"missing": missing})
    check("core_a_freeze_files", args.core_a_freeze_md.is_file() and args.core_a_freeze_json.is_file(), {"md": str(args.core_a_freeze_md), "json": str(args.core_a_freeze_json)})
    core_a_check = subprocess.run(["sha256sum", "-c", str(args.core_a_harvest / "CHECKSUMS.sha256")], capture_output=True, text=True, check=False)
    check("core_a_canonical_checksum", core_a_check.returncode == 0, {"exit_code": core_a_check.returncode, "ok_count": sum(line.endswith(": OK") for line in core_a_check.stdout.splitlines())})

    qa = read_json(root / "CORE_B_FEASIBILITY_QA.json")
    decision = read_json(root / "CORE_B_FEASIBILITY_DECISION.json")
    gates = read_json(root / "CORE_B_FEASIBILITY_GATES.json")
    schema = read_json(root / "CORE_B_DATA_SCHEMA_AUDIT.json")
    leakage = read_json(root / "CORE_B_SPLIT_AND_LEAKAGE_AUDIT.json")
    smoke = read_json(root / "CORE_B_SMOKE_PROTOCOL.json")
    smoke_acceptance = read_json(root / "CORE_B_SMOKE_RUN_ACCEPTANCE.json")
    smoke_results = read_json(root / "CORE_B_SMOKE_RESULTS.json")
    target_cache_inventory = read_json(root / "CORE_B_TARGET_CACHE_INVENTORY.json")
    normalization = read_json(root / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json")
    target_candidates = read_json(root / "CORE_B_TARGET_CANDIDATES.json")
    coverage = read_csv(root / "CORE_B_ELIGIBILITY_AND_COVERAGE.csv")
    temporal = read_csv(root / "CORE_B_TEMPORAL_ALIGNMENT_QA.csv")
    naive = read_csv(root / "CORE_B_NAIVE_BASELINES.csv")
    source = read_csv(root / "CORE_B_SOURCE_INVENTORY.csv")

    check("g1_only", qa.get("g1_only") is True and schema.get("dataset", {}).get("robot_type") == "g1", {"g1_only": qa.get("g1_only"), "robot_type": schema.get("dataset", {}).get("robot_type")})
    check("temporal_50_of_50", qa.get("spot_check", {}).get("sampled") == 50 and qa.get("spot_check", {}).get("passed") == 50 and len(temporal) == 50 and all(row.get("row_pass") == "True" for row in temporal), {"sampled": qa.get("spot_check", {}).get("sampled"), "passed": qa.get("spot_check", {}).get("passed")})
    check("coverage_and_task_retention", len(coverage) == 6 and all(float(row["eligible_percentage"]) >= 0.90 for row in coverage) and all(int(row["primary_dev_task_loss"]) == 0 for row in coverage if row["horizon"] in {"0.5s", "1.0s"}), {"coverage_rows": len(coverage), "two_second_dev_task_loss": [row["primary_dev_task_loss"] for row in coverage if row["horizon"] == "2.0s" and row["split"] == "validation"]})
    check("split_and_leakage", leakage.get("train_dev_episode_overlap") == 0 and leakage.get("train_dev_duplicate_ids") == 0 and leakage.get("future_leakage_in_spot_checks") == 0 and leakage.get("final_test_target_extraction") == 0 and leakage.get("normalization_final_test_rows") == 0, {key: leakage.get(key) for key in ("train_dev_episode_overlap", "train_dev_duplicate_ids", "future_leakage_in_spot_checks", "final_test_target_extraction", "normalization_final_test_rows")})
    check("action_mapping_and_unit_gate", target_candidates.get("action_stream", {}).get("mapping_source_verified") is True and target_candidates.get("action_stream", {}).get("units_source_verified") is False, {"mapping_source_verified": target_candidates.get("action_stream", {}).get("mapping_source_verified"), "units_source_verified": target_candidates.get("action_stream", {}).get("units_source_verified")})
    check("normalization_train_only", qa.get("normalization_train_only") is True and all(item.get("source_split") == "train only" and item.get("final_test_targets_accessed") == 0 for item in normalization.values()), {"horizons": list(normalization)})
    cache_bad = [item for item in target_cache_inventory.get("caches", []) if item.get("final_test_rows") != 0 or item.get("validation_rows", 0) <= 0]
    check("target_caches_train_dev_only", target_cache_inventory.get("train_dev_only") is True and target_cache_inventory.get("final_test_target_files_created") == 0 and not cache_bad, {"cache_count": len(target_cache_inventory.get("caches", [])), "bad": len(cache_bad)})
    finite_bad = []
    for row in naive:
        for key, value in row.items():
            if key in {"horizon", "horizon_seconds", "target_steps", "method", "split", "episodes", "target_shape", "units_status", "input_history_legal"}:
                continue
            try:
                if not math.isfinite(float(value)):
                    finite_bad.append((row.get("horizon"), row.get("method"), key))
            except ValueError:
                finite_bad.append((row.get("horizon"), row.get("method"), key))
    check("naive_baselines_numeric_finite", len(naive) == 12 and not finite_bad, {"rows": len(naive), "bad": finite_bad[:10]})
    check("smoke_not_started", decision.get("gpu_smoke_started") is False and decision.get("full_core_b_benchmark_started") is False and smoke.get("physical_gpus_used") == [] and smoke_acceptance.get("status") == "FEASIBILITY_BLOCKED" and smoke_results.get("gpu_smokes_started") is False, {"decision_status": decision.get("status"), "gpu_smoke_started": decision.get("gpu_smoke_started"), "physical_gpus_used": smoke.get("physical_gpus_used")})
    check("final_test_target_lock", qa.get("final_test_targets_accessed") == 0 and decision.get("final_test_targets_accessed") == 0 and schema.get("final_test_membership_only", {}).get("target_values_loaded") == 0, {"qa": qa.get("final_test_targets_accessed"), "decision": decision.get("final_test_targets_accessed"), "schema": schema.get("final_test_membership_only", {}).get("target_values_loaded")})
    check("source_inventory", len(source) >= 8 and all(row.get("exists") == "True" and row.get("final_test_access") == "False" for row in source), {"rows": len(source)})
    check("core_b_plot_present", (root / "plots/coverage_by_horizon.png").is_file() and (root / "plots/coverage_by_horizon.png").stat().st_size > 0, {"path": str(root / "plots/coverage_by_horizon.png")})

    report = {
        "status": "pass_with_feasibility_blocker" if all(item["status"] == "pass" for item in checks) and decision.get("status") == "blocked" else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": checks,
        "summary": {"checks_total": len(checks), "checks_passed": sum(item["status"] == "pass" for item in checks), "checks_failed": sum(item["status"] == "FAIL" for item in checks)},
        "feasibility_status": decision.get("status"),
        "gpu_smoke_started": False,
        "full_benchmark_started": False,
        "final_test_targets_accessed": 0,
        "visual_inspection": "pending",
        "production_or_old_artifacts_modified": False,
    }
    write_json(root / "CORE_B_FINAL_QA_REPORT.json", report)
    lines = ["# Core B final QA report", "", f"Status: `{report['status']}`", "", f"Checks: `{report['summary']['checks_passed']}/{report['summary']['checks_total']}`", "", "| Check | Status |", "|---|---|"]
    lines.extend(f"| {item['name']} | `{item['status']}` |" for item in checks)
    lines.extend(["", "Feasibility is blocked for formal GPU smoke by unresolved action-unit semantics; this is not a model-performance failure.", "Visual inspection is recorded separately after opening the coverage plot.", ""])
    (root / "CORE_B_FINAL_QA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_checksums(root)
    print(json.dumps({"status": report["status"], "checks_passed": report["summary"]["checks_passed"], "checks_total": report["summary"]["checks_total"], "root": str(root)}, sort_keys=True))
    return 0 if report["status"] == "pass_with_feasibility_blocker" else 1


if __name__ == "__main__":
    raise SystemExit(main())
