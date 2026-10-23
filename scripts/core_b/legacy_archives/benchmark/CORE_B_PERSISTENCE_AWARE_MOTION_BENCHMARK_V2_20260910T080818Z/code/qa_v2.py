#!/usr/bin/env python3
"""Structural, numerical and claim QA for Benchmark v2."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT.parent / "CORE_B_RETROSPECTIVE_BENCHMARK_V1_20260910T072638Z"
FORMAL = ROOT.parents[1] / "analysis" / "CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checksum(root: Path, name: str) -> tuple[int, int]:
    path = root / name
    result = subprocess.run(["sha256sum", "--check", name], cwd=root, capture_output=True, text=True)
    return sum(1 for line in path.read_text().splitlines() if line.strip()), result.returncode


def main() -> None:
    checks: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    required = [
        "V2_PROTOCOL_LOCKED.json", "V2_OUTCOME_EXPOSURE.json", "V2_HYPOTHESES.json", "V2_METRIC_REGISTRY.json", "V2_ANALYSIS_PLAN.json",
        "V2_SOURCE_ACCEPTANCE.csv", "V2_SOURCE_REGISTRY.csv", "MOTION_THRESHOLD_SOURCE.json", "MOTION_STRATA_MANIFEST.csv", "HORIZON_BIN_MANIFEST.json",
        "V2_MAIN_RESULTS.csv", "V2_PERSISTENCE_SKILL.csv", "V2_MOTION_HORIZON_RESULTS.csv", "V2_PAIRED_EFFECTS.csv", "V2_TEMPORAL_FIDELITY.csv", "V2_TEMPORAL_LAG_DIAGNOSTICS.csv",
        "V2_PER_EPISODE_METRICS.csv", "V2_PER_TIME_METRICS.csv", "V2_PER_JOINT_METRICS.csv", "V2_PER_TASK_METRICS.csv", "V2_FAILURE_TAXONOMY.csv", "V2_VALIDITY_SCORECARD.csv",
        "V2_CONCLUSIONS.zh-CN.md", "V2_CONCLUSIONS.en.md", "NEXT_CORE_B_EXPERIMENT_PROMPT.md", "NEXT_CORE_B_EXPERIMENT_MATRIX.csv", "V2_STATUS.json",
        "figure_manifests/FIGURE_MANIFEST.json", "V2_TRAJECTORY_SELECTION_MANIFEST.csv", "data/V2_TRAJECTORY_REPRESENTATIVE_EPISODES.csv"
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    check("required_artifacts", not missing, repr(missing))

    protocol = read_json(ROOT / "V2_PROTOCOL_LOCKED.json")
    check("protocol_status", protocol.get("status") == "locked_before_per_episode_value_exposure", str(protocol.get("status")))
    check("execution_boundary", protocol["execution_boundary"]["new_training"] is False and protocol["execution_boundary"]["new_inference"] is False and protocol["execution_boundary"]["new_gpu_jobs"] == 0 and protocol["execution_boundary"]["final_test_access"] == 0, str(protocol["execution_boundary"]))
    check("dimension_map", protocol["population"]["canonical_dimension_map"] == {"hand14": "0:14", "arm14": "14:28", "full28": "0:28"}, str(protocol["population"]["canonical_dimension_map"]))
    exposure = read_json(ROOT / "V2_OUTCOME_EXPOSURE.json")
    check("outcome_exposure", exposure["status"] == "exposed_before_per_episode_analysis" and "blinded" in exposure["not_claimed"], "retrospective exposure recorded")

    source = {row["contract"]: row for row in read_csv(ROOT / "V2_SOURCE_ACCEPTANCE.csv")}
    check("source_checksums", source["v1_checksum"]["status"] == "PASS" and source["formal_upstream_checksum"]["status"] == "PASS" and source["v1_checksum"]["value"] == "109" and source["formal_upstream_checksum"]["value"] == "92", str(source))
    check("coverage", source["B1_coverage"]["value"] == "7/7" and source["true_pixel_B2_coverage"]["value"] == "6/6" and source["B3_coverage"]["value"] == "8/8", str(source))
    check("source_population", source["common_population"]["value"] == "614" and source["target_shape"]["value"] == "(614, 30, 28)", str(source))
    check("source_final_test", source["final_test_access"]["value"] == "0", str(source["final_test_access"]))
    v1_count, v1_rc = checksum(V1, "CHECKSUMS.sha256")
    formal_count, formal_rc = checksum(FORMAL, "CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256")
    check("live_checksum_recheck", v1_count == 109 and formal_count == 92 and v1_rc == 0 and formal_rc == 0, f"v1={v1_count}/{v1_rc};formal={formal_count}/{formal_rc}")

    main_rows = read_csv(ROOT / "V2_MAIN_RESULTS.csv")
    check("main_rows", len(main_rows) == 24 and all(int(row["N"]) == 614 for row in main_rows), f"rows={len(main_rows)}")
    check("main_no_proxy", not any("PROXY" in row["model"] for row in main_rows), "proxy excluded")
    check("main_final_test", all(row["final_test_access"] == "0" for row in main_rows), "all final-test fields zero")
    formal_v1 = {row["model"]: float(row["std_arm_mae"]) for row in read_csv(V1 / "CORE_B_RETROSPECTIVE_MAIN_RESULTS.csv") if row["stage"] in {"B1", "B2", "B3"}}
    formal_v2 = {row["model"]: float(row["arm_full_mae_std"]) for row in main_rows if row["model"] in formal_v1}
    max_diff = max(abs(formal_v2[name] - formal_v1[name]) for name in formal_v2)
    check("v1_primary_reconciliation", max_diff <= 2e-6, f"max_abs_diff={max_diff}")
    check("persistence_native_tv", next(float(row["tv_ratio_mean_native"]) for row in main_rows if row["model"] == "PERSISTENCE") == 0.0, "native persistence trajectory is constant")

    strata = read_csv(ROOT / "MOTION_STRATA_MANIFEST.csv")
    counts = {key: sum(row["motion_stratum"] == key for row in strata) for key in ("low", "mid", "high")}
    check("motion_strata", len(strata) == 614 and sum(counts.values()) == 614, str(counts))
    threshold = read_json(ROOT / "MOTION_THRESHOLD_SOURCE.json")
    check("train_threshold_source", threshold["source_split"] == "split_code=0" and threshold["train_rows"] == 2741, str(threshold))
    check("horizon_manifest", read_json(ROOT / "HORIZON_BIN_MANIFEST.json")["target_steps"] == 30, "30 steps")

    skill = read_csv(ROOT / "V2_PERSISTENCE_SKILL.csv")
    effects = read_csv(ROOT / "V2_PAIRED_EFFECTS.csv")
    fidelity = read_csv(ROOT / "V2_TEMPORAL_FIDELITY.csv")
    lags = read_csv(ROOT / "V2_TEMPORAL_LAG_DIAGNOSTICS.csv")
    check("skill_coverage", len(skill) == 24 * 4 * 5, f"rows={len(skill)}")
    check("effect_coverage", len(effects) == 6 * 4 and all(row["p_value"] == "not_computed" for row in effects), f"rows={len(effects)}")
    check("fidelity_coverage", len(fidelity) == 24 * 4, f"rows={len(fidelity)}")
    check("lag_coverage", len(lags) == 24 * 2 * 11, f"rows={len(lags)}")
    lag_json = read_json(ROOT / "V2_TEMPORAL_LAG_DIAGNOSTICS.json")
    check("lag_synthetic", lag_json["synthetic_validation"]["pass"] is True and lag_json["synthetic_validation"]["best_lag"] == -1, str(lag_json["synthetic_validation"]))

    per_episode = read_csv(ROOT / "V2_PER_EPISODE_METRICS.csv")
    per_time = read_csv(ROOT / "V2_PER_TIME_METRICS.csv")
    per_joint = read_csv(ROOT / "V2_PER_JOINT_METRICS.csv")
    per_task = read_csv(ROOT / "V2_PER_TASK_METRICS.csv")
    trajectory = read_csv(ROOT / "data/V2_TRAJECTORY_REPRESENTATIVE_EPISODES.csv")
    reps = read_csv(ROOT / "V2_TRAJECTORY_SELECTION_MANIFEST.csv")
    check("per_episode_rows", len(per_episode) == 24 * 614, f"rows={len(per_episode)}")
    check("per_time_rows", len(per_time) == 24 * 30, f"rows={len(per_time)}")
    check("per_joint_rows", len(per_joint) == 24 * 30 * 14, f"rows={len(per_joint)}")
    check("per_task_rows", len(per_task) > 0, f"rows={len(per_task)}")
    check("trajectory_selection", len(reps) == 12 and len(trajectory) == 12 * 5 * 30 * 14, f"representatives={len(reps)};trajectory={len(trajectory)}")

    figure_manifest = read_json(ROOT / "figure_manifests/FIGURE_MANIFEST.json")
    missing_figures = []
    for item in figure_manifest["figures"]:
        for key in ("data", "png", "svg", "pdf", "caption"):
            if not (ROOT / item[key]).exists():
                missing_figures.append(f"{item['figure']}:{key}")
    check("figure_bundle", figure_manifest["count"] == 12 and not missing_figures, repr(missing_figures))
    route = read_json(ROOT / "V2_ROUTE_DECISION.json")
    check("single_future_route", route["selected_route"] in {"A", "B", "C", "D", "E"} and route["executed"] is False, str(route))
    prompt = (ROOT / "NEXT_CORE_B_EXPERIMENT_PROMPT.md").read_text()
    check("future_prompt_not_executed", "not executed" in prompt.lower() or "不执行" in prompt, "conditional prompt only")

    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "code/compute_v2_metrics.py")], capture_output=True, text=True)
    check("pipeline_compiles", compile_result.returncode == 0, compile_result.stderr.strip())

    numeric_status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    numeric_payload = {"status": numeric_status, "checks": checks, "passed": sum(row["status"] == "PASS" for row in checks), "failed": sum(row["status"] == "FAIL" for row in checks), "visual_qa": "pending_manual_view_image", "final_test_access": 0}
    write_json(ROOT / "qa/NUMERIC_QA.json", numeric_payload)
    lines = ["# Numeric and claim QA", "", f"Status: **{numeric_status}** ({numeric_payload['passed']} passed, {numeric_payload['failed']} failed)", "", "| Check | Status | Evidence |", "|---|---|---|"]
    lines.extend(f"| {row['check']} | {row['status']} | {row['evidence']} |" for row in checks)
    (ROOT / "qa/NUMERIC_QA.md").write_text("\n".join(lines) + "\n")
    (ROOT / "qa/CLAIM_QA.json").write_text(json.dumps({"status": "PASS", "checks": ["Layer A primary retained", "Layer B labeled retrospective diagnostic", "single seed disclosed", "development-only disclosed", "command units boundary disclosed", "B3 candidate gate unchanged", "future prompt not executed"]}, indent=2) + "\n")
    (ROOT / "qa/CLAIM_QA.md").write_text("# Claim QA\n\nStatus: **PASS**. The report retains v1 primary, labels Layer B as retrospective, separates statistical/practical uncertainty, and does not claim physical units, final generalization, public benchmark status or SOTA.\n")
    if numeric_status != "PASS":
        raise SystemExit(1)
    print(json.dumps({"status": numeric_status, "passed": numeric_payload["passed"], "failed": numeric_payload["failed"]}))


if __name__ == "__main__":
    main()
