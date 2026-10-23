#!/usr/bin/env python3
"""Fail-closed final QA for the Hand temporal-fidelity diagnostic."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
OLD = REPO / "reports/core_b"
EXPECTED_OLD = {
    "sel_u_audit": (OLD / "SEL_U_PROSPECTIVE_CONFIRMATION_V1_20260914T142514Z/qa/CHECKSUMS.sha256", "268b520f1b19808928cc0c11f969bf3585e332e506fc3898b8132040b6ccd735"),
    "selector": (OLD / "HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z/qa/CHECKSUMS.sha256", "66a71d453ae0d0f67826c2affe92018c3cfc9126c02e3492328706fa9bd1f2a7"),
    "hand_event": (OLD / "HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/qa/CHECKSUMS.sha256", "060283a9e62f21d4c30ea9d3746c8843b5765b2b5a73ce4f5d0b766b45fd0999"),
    "postfreeze": (OLD / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/qa/CHECKSUMS.sha256", "f12b2f1a0797705c9a9043c2b38cab82d650a8a0b9a01a9cfd81e9d58ec63197"),
    "freeze": (OLD / "JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z/qa/CHECKSUMS.sha256", "e6922a170abe6a5cc97ca4cf1a3701fcddc41aaf62f12b365838674cad23c3e8"),
    "benchmark_v2": (PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/qa/CHECKSUMS.sha256", "bb059ba7ed645c0dea7fae9bf39ecab7195692a1ef3f7ae4c739ab9bfaece976"),
}
WATCHER = REPO / "tools/gpu_lease_watcher.py"
POLICY = REPO / "docs/operations/GPU_WATCHER_POLICY.md"
WATCHER_BACKUP = REPO / "tools/gpu_lease_watcher.py.pre-v2-20260914T151719Z.bak"
POLICY_BACKUP = REPO / "docs/operations/GPU_WATCHER_POLICY.md.pre-v2-20260914T151719Z.bak"
EXPECTED_WATCHER = "89106fa8a0c65699d288851d92655e0d973f6990ecfbfad49dc8c5ebbd2feacb"
EXPECTED_POLICY = "c00e1791e2ce521375c17b1bb45dc839ab4563daf6365dc413602f25a489871a"
EXPECTED_BACKUP_WATCHER = "d082adffb0e3219c84557ebdb11da10695e369c9965f371d09f0725db45bb507"
EXPECTED_BACKUP_POLICY = "5dd0fd77c0baab3e9e50dcc469ac760acf0e880678e29eaddab6d3a2c0dbaad6"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nonfinite(value: Any, path: str = "root") -> list[str]:
    bad: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            bad.extend(nonfinite(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            bad.extend(nonfinite(child, f"{path}[{index}]"))
    return bad


def main() -> int:
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def record(name: str, passed: bool, details: Any) -> None:
        checks[name] = {"pass": bool(passed), "details": details}
        if not passed:
            failures.append(name)

    required = [
        "FINAL_STATUS.ini", "PRELAUNCH_INTEGRITY_AUDIT.md", "PRELAUNCH_INTEGRITY_AUDIT.json",
        "GPU_RESOURCE_POLICY_V2.md", "GPU_RESOURCE_POLICY_V2.json", "GPU_WATCHER_CHANGELOG.md",
        "TEMPORAL_FIDELITY_PROTOCOL_LOCKED.md", "TEMPORAL_FIDELITY_PROTOCOL_LOCKED.json",
        "SIGNED_DECOMPOSITION_DEFINITIONS.md", "SIGNED_DECOMPOSITION_DEFINITIONS.json",
        "COUNTERFACTUAL_MATRIX_LOCKED.csv", "COUNTERFACTUAL_MATRIX_LOCKED.json", "DCT_K_REGISTRY.json",
        "ROUTE_GATE_B.json", "ROUTE_GATE_B_DECISION.md", "ROUTE_GATE_B_DECISION.json",
        "OUTCOME_EXPOSURE.md", "OUTCOME_EXPOSURE.json", "CLAIM_BOUNDARY.md", "CLAIM_BOUNDARY.json",
        "MATRIX.csv", "MATRIX.json", "SEED_REGISTRY_SNAPSHOT.json", "UNTRIGGERED_BRANCHES.json",
        "EXPERIMENT_DECISION.zh-CN.md", "EXPERIMENT_DECISION.en.md",
        "BENCHMARK_NON_INTERFERENCE.md", "BENCHMARK_NON_INTERFERENCE.json",
        "NEXT_TEMPORAL_FIDELITY_MULTI_SEED_PROMPT.md", "NEXT_HAND_PHASE_CONDITIONAL_PROTOTYPE_PROMPT.md",
        "analysis/FD_SIGNED_DECOMPOSITION.csv", "analysis/FD_SIGNED_DECOMPOSITION.json", "analysis/FD_SIGNED_DECOMPOSITION.md",
        "analysis/GATE_SWITCH_DIAGNOSTICS.csv", "analysis/GATE_SWITCH_DIAGNOSTICS.json", "analysis/TASK_JOINT_SUPPORT.csv", "analysis/TASK_JOINT_SUPPORT.json",
        "analysis/COUNTERFACTUAL_RESULTS.csv", "analysis/COUNTERFACTUAL_RESULTS.json", "analysis/COUNTERFACTUAL_RESULTS.md",
        "analysis/COUNTERFACTUAL_PAIRED_EFFECTS.csv", "analysis/COUNTERFACTUAL_PAIRED_EFFECTS.json",
        "analysis/MAIN_RESULTS.csv", "analysis/MAIN_RESULTS.json", "analysis/MAIN_RESULTS.md",
        "analysis/PAIRED_EFFECTS.csv", "analysis/PAIRED_EFFECTS.json", "analysis/FACTORIAL_EFFECTS.csv", "analysis/FACTORIAL_EFFECTS.json",
        "analysis/TRAJECTORY_METRICS.csv", "analysis/TRAJECTORY_METRICS.json", "analysis/RESULTS_SUMMARY.json", "analysis/RESULTS_SUMMARY.md",
        "data/COUNTERFACTUAL_PREDICTIONS.npz", "FIGURE_GENERATION.json", "figures/FIGURE_DATA.csv",
        "captions/FIGURE_MANIFEST.csv", "captions/FIGURE_MANIFEST.json", "code/temporal_fidelity_analysis.py", "code/final_qa_temporal.py",
        "logs/temporal_fidelity_analysis.log", "logs/LOG_ATTEMPT_INDEX.md",
        "qa/GPU_POLICY_TESTS.json", "qa/GPU_POLICY_TESTS.md", "qa/RUN_ACCEPTANCE.csv", "qa/RUN_ACCEPTANCE.json",
        "qa/LEAKAGE_QA.json", "qa/LEAKAGE_QA.md", "qa/NUMERIC_QA.json", "qa/NUMERIC_QA.md",
        "qa/PLOT_VISUAL_QA.json", "qa/PLOT_VISUAL_QA.md", "qa/CLAIM_QA.json", "qa/CLAIM_QA.md",
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    record("required_artifacts", not missing, {"missing": missing, "count": len(required)})

    json_files = [path for path in ROOT.rglob("*.json") if path.name != "CHECKSUMS.sha256"]
    parse_errors: list[str] = []
    nonfinite_files: list[str] = []
    for path in json_files:
        try:
            value = load_json(path)
            bad = nonfinite(value)
            if bad:
                nonfinite_files.append(f"{path.relative_to(ROOT)}:{bad[:3]}")
        except Exception as exc:
            parse_errors.append(f"{path.relative_to(ROOT)}:{type(exc).__name__}:{exc}")
    record("json_parse_and_finite", not parse_errors and not nonfinite_files, {"json_count": len(json_files), "parse_errors": parse_errors, "nonfinite": nonfinite_files})

    csv_files = [path for path in ROOT.rglob("*.csv")]
    csv_errors: list[str] = []
    csv_rows: dict[str, int] = {}
    for path in csv_files:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                csv_rows[str(path.relative_to(ROOT))] = sum(1 for _ in csv.DictReader(handle))
        except Exception as exc:
            csv_errors.append(f"{path.relative_to(ROOT)}:{type(exc).__name__}:{exc}")
    record("csv_parse", not csv_errors, {"csv_count": len(csv_files), "rows": csv_rows, "errors": csv_errors})

    try:
        route = load_json(ROOT / "ROUTE_GATE_B.json")
        results = load_json(ROOT / "analysis/COUNTERFACTUAL_RESULTS.json")
        decomposition = load_json(ROOT / "analysis/FD_SIGNED_DECOMPOSITION.json")
        dct = load_json(ROOT / "DCT_K_REGISTRY.json")
        figures = load_json(ROOT / "FIGURE_GENERATION.json")
        visual = load_json(ROOT / "qa/PLOT_VISUAL_QA.json")
        gpu_tests = load_json(ROOT / "qa/GPU_POLICY_TESTS.json")
    except Exception as exc:
        record("analysis_json_contract", False, f"{type(exc).__name__}: {exc}")
        route = results = decomposition = dct = figures = visual = gpu_tests = {}
    else:
        record("analysis_json_contract", True, "analysis JSON contracts loaded")

    corrected = route.get("corrected_cells_pass", {})
    record("gate_b_and_route", route.get("status") == "PASS" and route.get("gate_b") == "fail" and route.get("route") == "NONE" and corrected == {"CF-D0-G1": False, "CF-D1-G0": False, "CF-D1-G1": False} and route.get("training_started") is False, {"gate_b": route.get("gate_b"), "route": route.get("route"), "corrected_cells_pass": corrected})
    record("replay_and_signed_identity", decomposition.get("native_replay_max_abs", 1.0) < 5e-6 and decomposition.get("standardized_identity_max_abs", 1.0) < 5e-6 and decomposition.get("native_identity_max_abs", 1.0) < 5e-6, {"native_replay_max_abs": decomposition.get("native_replay_max_abs"), "standardized_identity_max_abs": decomposition.get("standardized_identity_max_abs"), "native_identity_max_abs": decomposition.get("native_identity_max_abs")})
    record("dct_registry_locked", dct.get("selected_K") == 16 and dct.get("development_results_read_before_selection") is False and dct.get("median_cumulative_energy", {}).get("12", 1.0) < 0.95 <= dct.get("median_cumulative_energy", {}).get("16", 0.0), {"selected_K": dct.get("selected_K"), "median_cumulative_energy": dct.get("median_cumulative_energy")})
    record("figures_and_captions", figures.get("status") == "PASS" and figures.get("figure_count") == 14 and visual.get("status") == "PASS" and visual.get("rendered_file_count") == 42, {"figure_generation": figures, "visual_qa": visual})
    record("gpu_policy_tests", gpu_tests.get("status") == "PASS_WITH_TELEMETRY_CAVEAT" and gpu_tests.get("mock_watcher_tests") == {"passed": 16, "failed": 0} and gpu_tests.get("non_child_process_signaled") is False, gpu_tests)

    old_hashes: dict[str, Any] = {}
    for name, (path, expected) in EXPECTED_OLD.items():
        actual = sha256(path)
        old_hashes[name] = {"path": str(path), "entries": sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else None, "expected": expected, "actual": actual, "match": actual == expected}
    record("old_roots_unchanged", all(item["match"] for item in old_hashes.values()), old_hashes)
    current_hashes = {"watcher": sha256(WATCHER), "policy": sha256(POLICY), "watcher_backup": sha256(WATCHER_BACKUP), "policy_backup": sha256(POLICY_BACKUP)}
    record("gpu_policy_backup_and_hashes", current_hashes == {"watcher": EXPECTED_WATCHER, "policy": EXPECTED_POLICY, "watcher_backup": EXPECTED_BACKUP_WATCHER, "policy_backup": EXPECTED_BACKUP_POLICY}, current_hashes)
    watcher_text = WATCHER.read_text(encoding="utf-8") if WATCHER.is_file() else ""
    policy_text = POLICY.read_text(encoding="utf-8") if POLICY.is_file() else ""
    record("dynamic_policy_source_contract", "GPU_POOL_MODE = \"dynamic_all_visible\"" in watcher_text and "ALLOWED_GPUS" not in watcher_text and "GPU2 and GPU3 only" not in policy_text and "idle gate" not in policy_text.lower(), {"pool_mode_present": "GPU_POOL_MODE = \"dynamic_all_visible\"" in watcher_text, "fixed_allowlist_symbol_present": "ALLOWED_GPUS" in watcher_text, "legacy_policy_phrase_present": "GPU2 and GPU3 only" in policy_text, "idle_gate_phrase_present": "idle gate" in policy_text.lower()})

    unactivated = load_json(ROOT / "UNTRIGGERED_BRANCHES.json")
    empty_execution = {directory: [str(path) for path in (ROOT / directory).rglob("*") if path.is_file()] for directory in ("oof", "runs", "predictions", "configs")}
    record("no_training_branch", unactivated.get("status") == "not_applicable_gate_b_fail" and all(not files for files in empty_execution.values()) and load_json(ROOT / "MATRIX.json").get("training_started") is False, empty_execution)
    outcome = load_json(ROOT / "OUTCOME_EXPOSURE.json")
    claim = load_json(ROOT / "CLAIM_BOUNDARY.json")
    record("scope_and_final_boundary", outcome.get("final_test_target_access") == 0 and outcome.get("final_test_explicitly_authorized") is False and claim.get("old_selector_gate_rewritten") is False and claim.get("frozen_candidate_changed") is False, {"outcome": outcome, "claim": claim})
    final_log = (ROOT / "logs/temporal_fidelity_analysis.log").read_text(encoding="utf-8", errors="replace")
    record("final_log_clean", "Traceback" not in final_log and '"route": "NONE"' in final_log and '"figure_status": "PASS"' in final_log, {"bytes": len(final_log.encode("utf-8"))})

    report = {
        "status": "PASS_WITH_CAVEAT" if not failures else "FAIL",
        "qa_status": "PASS" if not failures else "FAIL",
        "gate_b": route.get("gate_b"),
        "route": route.get("route"),
        "failures": failures,
        "checks": checks,
        "final_test_access": 0,
        "training_started": False,
        "gpu_used": False,
        "old_selector_gate_rewritten": False,
        "frozen_candidate_changed": False,
        "benchmark_v2_changed": False,
        "note": "A QA pass confirms a correctly stopped Gate B failure; it does not promote a temporal-fidelity candidate."
    }
    (ROOT / "qa/FINAL_QA.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Final QA", "", f"Status: `{report['status']}`.", "",
        f"- QA checks: `{report['qa_status']}`; failed checks: `{len(failures)}`.",
        f"- Gate B: `{route.get('gate_b')}`; Route: `{route.get('route')}`; corrected cells: `3/3` evaluated, `0/3` passed.",
        "- Native signed identity, train-only DCT registry, dynamic watcher mock tests and plot QA were verified.",
        "- No GPU Stage0, single-seed training, multi-seed confirmation or final/held-out access occurred.",
        "- Old report manifests, SEL-U Gate A, frozen JY02 candidate and Benchmark v2 remain unchanged.",
    ]
    if failures:
        lines += ["", "Failures:", ""] + [f"- `{failure}`" for failure in failures]
    (ROOT / "qa/FINAL_QA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "qa_status": report["qa_status"], "gate_b": report["gate_b"], "route": report["route"], "failures": failures}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
