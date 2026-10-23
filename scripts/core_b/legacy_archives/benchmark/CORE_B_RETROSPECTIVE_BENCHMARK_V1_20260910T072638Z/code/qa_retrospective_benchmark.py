#!/usr/bin/env python3
"""Deterministic structural and numerical QA for the retrospective benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / "analysis" / "CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(check_id: str, passed: bool, evidence: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    required = [
        "BENCHMARK_PROTOCOL_LOCKED.json", "OUTCOME_EXPOSURE_STATEMENT.json", "ACCEPTANCE_AND_SCOPE.json",
        "CORE_B_RETROSPECTIVE_MAIN_RESULTS.csv", "CORE_B_PAIRED_EFFECTS.csv", "CORE_B_PROXY_RESULTS_SEPARATE.csv",
        "CORE_B_PER_EPISODE_METRICS.csv", "CORE_B_PER_TIMESTEP_METRICS.csv", "CORE_B_PER_JOINT_TIME_METRICS.csv",
        "CORE_B_PER_TASK_METRICS.csv", "CORE_B_MOTION_STRATIFIED_METRICS.csv", "CORE_B_FAILURE_TAXONOMY.csv",
        "CORE_B_TEMPORAL_ALIGNMENT_DIAGNOSTICS.json", "VALIDITY_SCORECARD.json", "FIGURE_MANIFEST.json",
        "data/COMMON_EVALUATION_POPULATION.json", "data/TARGET_NORMALIZATION_AUDIT.json",
        "data/MODEL_SOURCE_REGISTRY.csv", "data/SOURCE_ARTIFACT_REVALIDATION.json", "data/REPRESENTATIVE_EPISODES.csv",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    check("required_artifacts", not missing, "missing=" + repr(missing))

    protocol = json.loads((ROOT / "BENCHMARK_PROTOCOL_LOCKED.json").read_text())
    check("protocol_locked", protocol.get("status") == "locked_before_score_exposure", str(protocol.get("status")))
    check("protocol_dimension_map", protocol["metric_contract"].get("dimension_mapping") == {"hand14": "0:14", "arm14": "14:28", "full28": "0:28"}, str(protocol["metric_contract"].get("dimension_mapping")))
    check("protocol_final_test_lock", protocol["execution_boundary"]["final_test_access"] == 0, str(protocol["execution_boundary"]))

    status = json.loads((ROOT / "BENCHMARK_STATUS.json").read_text())
    check("status_common_population", status.get("common_population_N") == 614, str(status))
    check("status_target_shape", status.get("target_shape") == [614, 30, 28], str(status.get("target_shape")))
    check("status_no_gpu_jobs", status.get("gpu_jobs_started_by_analysis") is False, str(status.get("gpu_jobs_started_by_analysis")))

    acceptance = {row["stage"]: row for row in read_csv("ACCEPTANCE_AND_SCOPE.csv")}
    check("b1_coverage", acceptance["B1"]["coverage"] == "7/7" and acceptance["B1"]["status"] == "PASS", str(acceptance["B1"]))
    check("b2_coverage", acceptance["B2"]["coverage"] == "6/6" and acceptance["B2"]["status"] == "PASS", str(acceptance["B2"]))
    check("b3_coverage", acceptance["B3"]["coverage"] == "8/8" and acceptance["B3"]["status"] == "PASS", str(acceptance["B3"]))
    check("proxy_separation", acceptance["B2_PROXY"]["status"] == "AUXILIARY_EXCLUDED", str(acceptance["B2_PROXY"]))

    main_rows = read_csv("CORE_B_RETROSPECTIVE_MAIN_RESULTS.csv")
    proxy_rows = read_csv("CORE_B_PROXY_RESULTS_SEPARATE.csv")
    check("main_row_count", len(main_rows) == 24, f"rows={len(main_rows)}")
    check("proxy_row_count", len(proxy_rows) == 4, f"rows={len(proxy_rows)}")
    check("main_no_proxy", not any("PROXY" in row["model"] for row in main_rows), "proxy models absent from main table")
    check("main_common_N", all(int(row["N"]) == 614 for row in main_rows), "all main rows N=614")
    check("main_final_test_zero", all(int(row["final_test_access"]) == 0 for row in main_rows), "all main rows final_test_access=0")
    check("main_finite", all(np.isfinite(float(row["std_arm_mae"])) for row in main_rows), "all primary means finite")

    formal_source = {}
    with (SOURCE / "CORE_B_UNIFIED_RESULTS.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stage"] in {"B1", "B2", "B3"}:
                formal_source[row["cell"]] = float(row["std_arm_mae"])
    independent = {row["model"]: float(row["std_arm_mae"]) for row in main_rows if row["model"] in formal_source}
    max_diff = max(abs(independent[name] - formal_source[name]) for name in independent)
    check("independent_recompute_matches_formal", max_diff <= 2e-6, f"max_abs_diff={max_diff}")

    audit = json.loads((ROOT / "data/TARGET_NORMALIZATION_AUDIT.json").read_text())
    check("normalization_audit", audit.get("status") == "PASS" and audit.get("canonical_expected_vs_stored_max_abs", 1.0) <= 2e-6, str(audit))
    source_audit = json.loads((ROOT / "data/SOURCE_ARTIFACT_REVALIDATION.json").read_text())
    check("source_artifact_audit", source_audit.get("status") == "PASS", str(source_audit))

    population = json.loads((ROOT / "data/COMMON_EVALUATION_POPULATION.json").read_text())
    check("population_id_count", population.get("N") == 614 and len(population.get("episode_ids", [])) == 614, str(population.get("N")))
    representative = read_csv("data/REPRESENTATIVE_EPISODES.csv")
    counts = {stratum: sum(row["motion_stratum"] == stratum for row in representative) for stratum in ("low", "mid", "high")}
    check("representative_episodes", len(representative) == 12 and counts == {"low": 4, "mid": 4, "high": 4}, str(counts))

    effects = read_csv("CORE_B_PAIRED_EFFECTS.csv")
    check("paired_effects", len(effects) == 20 and all(int(row["N"]) == 614 for row in effects), f"rows={len(effects)}")
    check("holm_values", all(0 <= float(row["holm_p_value"]) <= 1 for row in effects), "Holm values in [0,1]")
    lag = json.loads((ROOT / "CORE_B_TEMPORAL_ALIGNMENT_DIAGNOSTICS.json").read_text())
    check("lag_synthetic_validation", lag["synthetic_validation"]["pass"] is True and lag["synthetic_validation"]["best_lag"] == -1, str(lag["synthetic_validation"]))

    per_episode = read_csv("CORE_B_PER_EPISODE_METRICS.csv")
    check("per_episode_rows", len(per_episode) == 24 * 614, f"rows={len(per_episode)}")
    per_time = read_csv("CORE_B_PER_TIMESTEP_METRICS.csv")
    check("per_time_rows", len(per_time) == 24 * 30, f"rows={len(per_time)}")
    trajectory = read_csv("data/TRAJECTORY_REPRESENTATIVE_EPISODES.csv")
    check("trajectory_rows", len(trajectory) == 12 * 5 * 30 * 14, f"rows={len(trajectory)}")

    figure_manifest = json.loads((ROOT / "FIGURE_MANIFEST.json").read_text())
    figure_failures = []
    for item in figure_manifest["figures"]:
        for key in ("data", "png", "svg", "pdf", "caption"):
            if not (ROOT / item[key]).exists():
                figure_failures.append(f"{item['figure']}:{key}")
    check("figure_bundle", len(figure_manifest["figures"]) == 10 and not figure_failures, repr(figure_failures))

    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "code/run_retrospective_benchmark.py")], capture_output=True, text=True)
    check("benchmark_script_compiles", compile_result.returncode == 0, compile_result.stderr.strip())

    passed = sum(row["status"] == "PASS" for row in checks)
    failed = len(checks) - passed
    payload = {"status": "PASS" if failed == 0 else "FAIL", "checks": checks, "passed": passed, "failed": failed, "final_test_access": 0, "visual_qa": "pending_manual_view_image"}
    write_json(ROOT / "qa/QA_REPORT.json", payload)
    lines = ["# Retrospective benchmark QA", "", f"Status: **{payload['status']}** ({passed} passed, {failed} failed)", "", "| Check | Status | Evidence |", "|---|---|---|"]
    lines.extend(f"| {row['check_id']} | {row['status']} | {row['evidence']} |" for row in checks)
    (ROOT / "qa/QA_REPORT.md").write_text("\n".join(lines) + "\n")
    if failed:
        raise SystemExit(1)
    print(json.dumps({"status": payload["status"], "passed": passed, "failed": failed}))


if __name__ == "__main__":
    main()
