#!/usr/bin/env python3
"""Fail-closed final QA for the SEL-U Gate A audit."""

from __future__ import annotations
import os

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
OLD_ROOT = REPO / "reports/core_b"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
POLICY = REPO / "docs/operations/GPU_WATCHER_POLICY.md"
WATCHER = REPO / "tools/gpu_lease_watcher.py"

OLD_MANIFESTS = {
    "selector": (
        OLD_ROOT / "HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z/qa/CHECKSUMS.sha256",
        "66a71d453ae0d0f67826c2affe92018c3cfc9126c02e3492328706fa9bd1f2a7",
    ),
    "hand_event": (
        OLD_ROOT / "HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/qa/CHECKSUMS.sha256",
        "060283a9e62f21d4c30ea9d3746c8843b5765b2b5a73ce4f5d0b766b45fd0999",
    ),
    "postfreeze": (
        OLD_ROOT / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/qa/CHECKSUMS.sha256",
        "f12b2f1a0797705c9a9043c2b38cab82d650a8a0b9a01a9cfd81e9d58ec63197",
    ),
    "freeze": (
        OLD_ROOT / "JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z/qa/CHECKSUMS.sha256",
        "e6922a170abe6a5cc97ca4cf1a3701fcddc41aaf62f12b365838674cad23c3e8",
    ),
    "benchmark_v2": (
        PROJECT
        / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/qa/CHECKSUMS.sha256",
        "bb059ba7ed645c0dea7fae9bf39ecab7195692a1ef3f7ae4c739ab9bfaece976",
    ),
}

EXPECTED_POLICY_SHA = "5dd0fd77c0baab3e9e50dcc469ac760acf0e880678e29eaddab6d3a2c0dbaad6"
EXPECTED_WATCHER_SHA = "d082adffb0e3219c84557ebdb11da10695e369c9965f371d09f0725db45bb507"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite_tree(value: Any, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            failures.extend(finite_tree(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(finite_tree(child, f"{path}[{index}]"))
    return failures


def parse_ini(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> int:
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def record(name: str, passed: bool, details: Any) -> None:
        checks[name] = {"pass": bool(passed), "details": details}
        if not passed:
            failures.append(name)

    required = [
        "FINAL_STATUS.ini",
        "PRELAUNCH_INTEGRITY_AUDIT.json",
        "CLAIM_BOUNDARY.md",
        "CLAIM_BOUNDARY.json",
        "BENCHMARK_NON_INTERFERENCE.md",
        "BENCHMARK_NON_INTERFERENCE.json",
        "GPU_RESOURCE_POLICY_V2.md",
        "GPU_RESOURCE_POLICY_V2.json",
        "GPU_WATCHER_CHANGELOG.md",
        "PROSPECTIVE_CONFIRMATION_PROTOCOL_LOCKED.md",
        "PROSPECTIVE_CONFIRMATION_PROTOCOL_LOCKED.json",
        "PROSPECTIVE_CANDIDATE_GATE.md",
        "PROSPECTIVE_CANDIDATE_GATE.json",
        "DEVELOPMENT_CANDIDATE_PROPOSAL.md",
        "DEVELOPMENT_CANDIDATE_PROPOSAL.json",
        "EXPERIMENT_DECISION.zh-CN.md",
        "EXPERIMENT_DECISION.en.md",
        "SEL_U_ELIGIBILITY_PROTOCOL_LOCKED.md",
        "SEL_U_ELIGIBILITY_PROTOCOL_LOCKED.json",
        "SEL_U_ELIGIBILITY_GATE.json",
        "SEL_U_ELIGIBILITY_GATE_EVALUATED.json",
        "SEL_U_ELIGIBILITY_RESULTS.json",
        "SEL_U_ELIGIBILITY_RESULTS.csv",
        "SEL_U_ELIGIBILITY_RESULTS.md",
        "PHASE1_RUN_COMPLETE.json",
        "SEED_REGISTRY_SNAPSHOT.json",
        "MATRIX.json",
        "MATRIX.csv",
        "NEXT_HAND_EXPERT_MULTIMODAL_DECODER_PROMPT_REVIEWED.md",
        "UNTRIGGERED_BRANCHES.json",
        "analysis/NOT_APPLICABLE_GATE_A_FAIL.json",
        "captions/NOT_APPLICABLE_GATE_A_FAIL.md",
        "FIGURE_GENERATION.json",
        "code/run_sel_u_eligibility.py",
        "code/final_qa_sel_u.py",
        "logs/SEL_U_ELIGIBILITY.log",
        "qa/GPU_POLICY_TESTS.json",
        "qa/GPU_POLICY_TESTS.md",
        "qa/LEAKAGE_QA.json",
        "qa/LEAKAGE_QA.md",
        "qa/NUMERIC_QA.json",
        "qa/NUMERIC_QA.md",
        "qa/PLOT_VISUAL_QA.json",
        "qa/PLOT_VISUAL_QA.md",
        "qa/CLAIM_QA.json",
        "qa/CLAIM_QA.md",
        "qa/RUN_ACCEPTANCE.csv",
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    record("required_artifacts", not missing, {"missing": missing, "count": len(required)})

    try:
        results = read_json(ROOT / "SEL_U_ELIGIBILITY_RESULTS.json")
        evaluated = read_json(ROOT / "SEL_U_ELIGIBILITY_GATE_EVALUATED.json")
        phase = read_json(ROOT / "PHASE1_RUN_COMPLETE.json")
        outcome = read_json(ROOT / "OUTCOME_EXPOSURE.json")
        gpu = read_json(ROOT / "GPU_RESOURCE_POLICY_V2.json")
        seed = read_json(ROOT / "SEED_REGISTRY_SNAPSHOT.json")
        matrix = read_json(ROOT / "MATRIX.json")
        figure_manifest = read_json(ROOT / "FIGURE_GENERATION.json")
        claim = read_json(ROOT / "CLAIM_BOUNDARY.json")
        claim_qa = read_json(ROOT / "qa/CLAIM_QA.json")
        prospective_gate = read_json(ROOT / "PROSPECTIVE_CANDIDATE_GATE.json")
        branch_marker = read_json(ROOT / "UNTRIGGERED_BRANCHES.json")
    except Exception as exc:  # pragma: no cover - fail-closed reporting path
        record("json_contracts_parse", False, f"{type(exc).__name__}: {exc}")
        results = evaluated = phase = outcome = gpu = seed = matrix = figure_manifest = claim = claim_qa = prospective_gate = branch_marker = {}
    else:
        record("json_contracts_parse", True, "all required JSON objects parsed")

    metric_names = [
        "hand_mae",
        "persistence_hand_mae",
        "full28_mae",
        "fixed_hybrid_full28_mae",
        "hand_relative_improvement",
        "full_relative_improvement_vs_hybrid",
        "oracle_step_group_hand_mae",
        "oracle_step_group_headroom",
        "hand_first_difference_relative_change",
        "native_tv_relative_change",
        "q_collapse_fraction",
    ]
    nonfinite = finite_tree(results)
    record("reported_metrics_finite", not nonfinite, {"nonfinite_paths": nonfinite})
    record(
        "result_contract",
        results.get("status") == "PASS"
        and results.get("selector") == "SEL-U"
        and results.get("development_count") == 614
        and results.get("bootstrap_replicates") == 10_000
        and all(name in results for name in metric_names),
        {"status": results.get("status"), "selector": results.get("selector"), "development_count": results.get("development_count")},
    )

    expected_metrics = {
        "hand_mae": 0.23714446009631326,
        "persistence_hand_mae": 0.2574218404031712,
        "full28_mae": 0.17069720639983438,
        "fixed_hybrid_full28_mae": 0.18083589655326332,
        "hand_relative_improvement": 0.07877101754497484,
        "full_relative_improvement_vs_hybrid": 0.05606569462519681,
        "oracle_headroom_closure": 0.5227222686890611,
        "hand_first_difference_relative_change": 0.09260069956313664,
        "native_tv_relative_change": -0.1570844446998001,
    }
    metric_values = {
        name: results.get(name)
        for name in expected_metrics
        if name != "oracle_headroom_closure"
    }
    closure_value = results.get("oracle_headroom_closure")
    if isinstance(closure_value, dict):
        metric_values["oracle_headroom_closure"] = closure_value.get("effect")
    metric_diffs = {
        name: abs(float(metric_values.get(name, float("nan"))) - expected)
        for name, expected in expected_metrics.items()
        if metric_values.get(name) is not None
    }
    record("metric_values_current", all(value <= 1e-12 for value in metric_diffs.values()), metric_diffs)

    gate_checks = results.get("gate_checks", {})
    false_gate_keys = sorted(key for key, value in gate_checks.items() if value is False)
    record(
        "gate_a_exact_evaluation",
        results.get("gate") == "fail"
        and false_gate_keys == ["hand_first_difference_relative_change_le_1pct"]
        and results.get("hand_first_difference_relative_change", 0) > 0.01,
        {"gate": results.get("gate"), "false_gate_keys": false_gate_keys},
    )
    record(
        "gate_evaluation_matches_locked_copy",
        evaluated.get("evaluated", {}).get("gate") == "fail"
        and evaluated.get("locked", {}).get("status") == "LOCKED_BEFORE_RESULTS",
        {"locked_status": evaluated.get("locked", {}).get("status"), "evaluated_gate": evaluated.get("evaluated", {}).get("gate")},
    )

    record(
        "phase_stop_contract",
        phase.get("status") == "complete"
        and phase.get("gate_a") == "fail"
        and phase.get("fresh_seed_training_started") is False
        and phase.get("final_test_access") == 0,
        phase,
    )
    record(
        "conditional_branch_not_activated",
        seed.get("status") == "not_applicable_gate_a_fail"
        and seed.get("fresh_seed_training_started") is False
        and matrix.get("status") == "not_applicable_gate_a_fail"
        and matrix.get("cells") == [],
        {"seed": seed, "matrix": matrix},
    )
    record(
        "not_applicable_markers",
        prospective_gate.get("status") == "not_applicable_gate_a_fail"
        and prospective_gate.get("gate") == "not_run"
        and claim_qa.get("status") == "PASS"
        and branch_marker.get("status") == "not_applicable_gate_a_fail",
        {"prospective_gate": prospective_gate, "claim_qa": claim_qa, "branch_marker": branch_marker},
    )

    figure_paths = figure_manifest.get("files", [])
    figure_checks = {
        str(path): Path(path).is_file() and Path(path).stat().st_size > 0
        for path in figure_paths
    }
    expected_figure_names = {
        "SEL_U_ELIGIBILITY_MAE.png",
        "SEL_U_ELIGIBILITY_GATES.png",
        "SEL_U_Q_DISTRIBUTION.png",
    }
    actual_figure_names = {Path(path).name for path in figure_paths}
    record(
        "figures",
        figure_manifest.get("status") == "PASS"
        and actual_figure_names == expected_figure_names
        and all(figure_checks.values()),
        {"count": len(figure_paths), "files_nonempty": figure_checks},
    )

    try:
        with np.load(INPUT, allow_pickle=False) as source:
            target_shape = tuple(int(value) for value in source["targets_standardized"].shape)
            split = source["split_code"]
            dev_count = int(np.sum(split == 1))
            train_count = int(np.sum(split == 0))
            finite_input = bool(np.isfinite(source["targets_standardized"]).all())
        input_details = {
            "target_shape": target_shape,
            "development_count": dev_count,
            "train_count": train_count,
            "finite_targets": finite_input,
        }
        record(
            "input_contract",
            target_shape == (3355, 30, 28)
            and dev_count == 614
            and train_count == 2741
            and finite_input,
            input_details,
        )
    except Exception as exc:  # pragma: no cover - fail-closed reporting path
        record("input_contract", False, f"{type(exc).__name__}: {exc}")

    inherited_selector = OLD_ROOT / "HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z"
    try:
        old_final = read_json(inherited_selector / "qa/FINAL_QA.json")
        old_leakage = read_json(inherited_selector / "qa/LEAKAGE_QA.json")
        oof_details = {
            "final_qa_status": old_final.get("status"),
            "final_qa_failures": old_final.get("failures"),
            "leakage_status": old_leakage.get("status"),
            "coverage": old_leakage.get("oof_coverage"),
            "overlap": old_leakage.get("oof_fold_overlap"),
        }
        record(
            "inherited_oof_and_leakage",
            old_final.get("status") in {"PASS", "PASS_WITH_CAVEAT"}
            and old_final.get("failures") == []
            and old_leakage.get("status") == "PASS"
            and old_leakage.get("oof_coverage") == 2741
            and old_leakage.get("oof_fold_overlap") == 0,
            oof_details,
        )
    except Exception as exc:  # pragma: no cover - fail-closed reporting path
        record("inherited_oof_and_leakage", False, f"{type(exc).__name__}: {exc}")

    manifest_details: dict[str, Any] = {}
    for name, (path, expected) in OLD_MANIFESTS.items():
        actual = sha256(path) if path.is_file() else None
        manifest_details[name] = {"path": str(path), "expected": expected, "actual": actual, "match": actual == expected}
    record("old_manifests_unchanged", all(item["match"] for item in manifest_details.values()), manifest_details)
    policy_actual = sha256(POLICY) if POLICY.is_file() else None
    watcher_actual = sha256(WATCHER) if WATCHER.is_file() else None
    record(
        "shared_gpu_policy_unchanged",
        policy_actual == EXPECTED_POLICY_SHA and watcher_actual == EXPECTED_WATCHER_SHA,
        {"policy_sha256": policy_actual, "watcher_sha256": watcher_actual},
    )

    current_root_files = []
    for directory in (ROOT / "runs", ROOT / "predictions", ROOT / "data", ROOT / "configs", ROOT / "tests"):
        if directory.is_dir():
            current_root_files.extend(str(path) for path in directory.rglob("*") if path.is_file())
    record("no_prospective_execution_artifacts", not current_root_files, {"files": current_root_files})

    log_text = (ROOT / "logs/SEL_U_ELIGIBILITY.log").read_text(encoding="utf-8", errors="replace")
    record("analysis_log_clean", "traceback" not in log_text.lower() and '"status": "PASS"' in log_text, {"log_bytes": len(log_text.encode("utf-8"))})

    record(
        "final_test_and_scope_boundary",
        outcome.get("final_test_explicitly_authorized") is False
        and outcome.get("final_test_target_access") == 0
        and outcome.get("old_candidate_gate_rewritten") is False
        and outcome.get("professor_contact_started") is False
        and claim.get("final_or_heldout_evaluation") is False,
        {"outcome": outcome, "claim": claim},
    )
    record(
        "gpu_boundary",
        gpu.get("dynamic_policy_change_applied") is False
        and gpu.get("gpu1_used") is False
        and gpu.get("non_child_process_signal") is False
        and gpu.get("final_test_target_access") == 0,
        gpu,
    )
    stray_root = PROJECT / "repo/core_b/SEL_U_PROSPECTIVE_CONFIRMATION_V1_20260914T142514Z"
    stray_files = [str(path) for path in stray_root.rglob("*") if path.is_file()] if stray_root.is_dir() else []
    record("no_stray_duplicate_files", not stray_files, {"files": stray_files})

    qa_status = "PASS" if not failures else "FAIL"
    final_status = "PASS_WITH_CAVEAT" if qa_status == "PASS" else "FAIL"
    report = {
        "status": final_status,
        "qa_status": qa_status,
        "gate_a": results.get("gate"),
        "failures": failures,
        "checks": checks,
        "final_test_access": 0,
        "fresh_seed_training_started": False,
        "gpu_used": False,
        "frozen_candidate_changed": False,
        "benchmark_v2_changed": False,
        "note": "QA pass confirms a correctly stopped Gate A failure; it does not convert SEL-U into a prospective or final model.",
    }
    qa_json = ROOT / "qa/FINAL_QA.json"
    qa_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Final QA",
        "",
        f"Status: `{final_status}`.",
        "",
        f"- QA checks: `{qa_status}`; failed checks: `{len(failures)}`.",
        f"- Gate A: `{results.get('gate')}`; only failed substantive gate: Hand first-difference relative change `+9.26%` > `+1%`.",
        "- SEL-U Hand and Full-28 MAE improvements, oracle closure, native-TV, support and inherited OOF/leakage checks were verified.",
        "- Fresh-seed training, GPU Stage0 and prospective matrix: not started.",
        "- Final-test target access: `0`; old manifests, frozen candidate and Benchmark v2: unchanged.",
    ]
    if failures:
        lines.extend(["", "Failures:", ""] + [f"- `{failure}`" for failure in failures])
    (ROOT / "qa/FINAL_QA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": final_status, "qa_status": qa_status, "gate_a": results.get("gate"), "failures": failures}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
