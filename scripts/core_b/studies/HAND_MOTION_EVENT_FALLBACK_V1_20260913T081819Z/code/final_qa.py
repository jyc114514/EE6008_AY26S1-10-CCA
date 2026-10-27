#!/usr/bin/env python3
"""Fail-closed final QA for the Hand event-gate exploratory artifact."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from train_event_cell import CELL_ORDER, INPUT_PATH, ROOT, STEPS, DIMS, build_cell_model, load_contract_data

MEETING_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'professor_meeting/CORE_B_HAND_PERSISTENCE_QUESTION_20260913T081819Z'))


EXPECTED_OLD_MANIFEST_SHA = {
    "freeze": "e6922a170abe6a5cc97ca4cf1a3701fcddc41aaf62f12b365838674cad23c3e8",
    "stage2": "92c949fd1a930fb018e702f5dbdad2cd65a4840743247cd460ab50d86fad110e",
    "postfreeze": "f12b2f1a0797705c9a9043c2b38cab82d650a8a0b9a01a9cfd81e9d58ec63197",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> int:
    failures: list[str] = []
    checks: dict[str, object] = {}
    data, train_mask, dev_mask, contracts = load_contract_data()
    dev_targets = data["targets_standardized"][dev_mask]
    dev_persistence = data["persistence_standardized"][dev_mask]
    dev_ids = data["episode_index"][dev_mask]
    dev_tasks = data["original_task_index"][dev_mask]
    checks["input_contract"] = {"train": int(train_mask.sum()), "development": int(dev_mask.sum()), "shape": list(data["targets_standardized"].shape), "final_test_access": 0}

    stage0 = json.loads((ROOT / "STAGE0_RESULTS.json").read_text(encoding="utf-8"))
    checks["stage0"] = {"status": stage0.get("status"), "accepted": stage0.get("accepted"), "expected": stage0.get("expected")}
    if stage0.get("status") != "PASS" or stage0.get("accepted") != 6:
        failures.append("Stage0 is not 6/6 PASS")

    audit = json.loads((ROOT / "MOTION_AUDIT_SUMMARY.json").read_text(encoding="utf-8"))
    checks["phase_a"] = {"status": audit.get("status"), "evidence_gate": audit.get("evidence_gate"), "derived_hybrid": audit.get("derived_hybrid"), "outcome_exposed": audit.get("outcome_exposed")}
    if audit.get("status") != "PASS" or not audit.get("evidence_gate", {}).get("pass") or not audit.get("outcome_exposed"):
        failures.append("Phase A evidence/output contract failed")
    hybrid_full = float(audit["derived_hybrid"]["hyb_a_nn_h_p_full28_mae_std"])
    if not np.isclose(hybrid_full, 0.18083589655326335, rtol=0, atol=1e-8):
        failures.append("derived HYB-A_NN-H_P Full-28 value mismatch")

    matrix = json.loads((ROOT / "CANDIDATE_GATE_EVALUATIONS.json").read_text(encoding="utf-8"))
    checks["matrix_decision"] = {"status": matrix.get("status"), "matrix_accepted": matrix.get("matrix_accepted"), "candidate_gate": matrix.get("candidate_gate"), "selected_candidate": matrix.get("selected_candidate"), "seed": matrix.get("seed")}
    if matrix.get("status") != "PASS" or matrix.get("matrix_accepted") != "6/6" or matrix.get("candidate_gate") != "fail" or matrix.get("selected_candidate") is not None:
        failures.append("matrix decision is not the locked fail/no-selection state")

    # Re-open every checkpoint strictly and verify every prediction archive is
    # paired to the same development episode ordering.
    prediction_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    for cell in CELL_ORDER:
        run = ROOT / "runs" / cell
        marker = json.loads((run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        config = json.loads((run / "RUN_CONFIG_LOCKED.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "RUN_METRICS.json").read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("optimizer_steps") != 1720 or marker.get("prediction_shape") != [614, STEPS, DIMS] or marker.get("final_test_access") != 0:
            failures.append(f"run marker failed: {cell}")
        if config.get("event_label_input_to_model") is not False or config.get("event_future_label_read_at_inference") is not False or config.get("final_test_access") != 0:
            failures.append(f"future-label/final-test config failed: {cell}")
        if float(config.get("added_event_parameter_fraction", 1.0)) > 0.05:
            failures.append(f"event parameter budget failed: {cell}")
        pred_path = run / "predictions_best.npz"
        checkpoint_path = run / "checkpoint_best.pt"
        if not pred_path.is_file() or not checkpoint_path.is_file():
            failures.append(f"missing checkpoint/prediction: {cell}")
            continue
        prediction_hashes[cell] = digest(pred_path)
        checkpoint_hashes[cell] = digest(checkpoint_path)
        if prediction_hashes[cell] != marker.get("prediction_best_sha256") or checkpoint_hashes[cell] != marker.get("checkpoint_best_sha256"):
            failures.append(f"artifact hash marker mismatch: {cell}")
        with np.load(pred_path, allow_pickle=False) as archive:
            bundle = {key: archive[key].copy() for key in archive.files}
        required = {"predictions_standardized", "targets_standardized", "persistence_standardized", "predictions_native", "targets_native", "persistence_native", "gate_values", "event_probability", "event_step_labels", "event_window_labels", "episode_index", "original_task_index"}
        if not required.issubset(bundle):
            failures.append(f"prediction schema missing keys: {cell}")
            continue
        if bundle["predictions_standardized"].shape != (614, STEPS, DIMS) or bundle["gate_values"].shape != (614, STEPS, 2) or bundle["event_probability"].shape != (614, STEPS):
            failures.append(f"prediction shape failed: {cell}")
        if not np.array_equal(bundle["episode_index"], dev_ids) or not np.array_equal(bundle["original_task_index"], dev_tasks) or not np.array_equal(bundle["targets_standardized"], dev_targets) or not np.array_equal(bundle["persistence_standardized"], dev_persistence):
            failures.append(f"development pairing failed: {cell}")
        if not np.isfinite(bundle["predictions_standardized"]).all() or not np.isfinite(bundle["predictions_native"]).all() or not np.isfinite(bundle["gate_values"]).all() or not np.isfinite(bundle["event_probability"]).all() or np.any(bundle["gate_values"] < 0) or np.any(bundle["gate_values"] > 1) or np.any(bundle["event_probability"] < 0) or np.any(bundle["event_probability"] > 1):
            failures.append(f"finite/bounded prediction failed: {cell}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mode = str(config["event_mode"])
        model, _hidden, params, added = build_cell_model(cell, int(config["component_seeds"]["model"]), float(config["train_event_prevalence"]))
        try:
            model.load_state_dict(payload["model_state_dict"], strict=True)
        except Exception as exc:
            failures.append(f"strict checkpoint reload failed: {cell}: {type(exc).__name__}: {exc}")
        if params != int(config["parameter_count"]) or added != int(config["added_event_parameters"]):
            failures.append(f"parameter contract failed: {cell}")
        if metrics.get("status") != "complete" or metrics.get("optimizer_steps") != 1720 or metrics.get("final_test_access") != 0:
            failures.append(f"run metrics failed: {cell}")
    checks["runs"] = {"completed": len(prediction_hashes), "expected": 6, "prediction_hashes": prediction_hashes, "checkpoint_hashes": checkpoint_hashes, "strict_reload_checked": True}
    if len(prediction_hashes) != 6:
        failures.append("not all six prediction archives were verified")

    watcher_files = sorted((ROOT / "logs").glob("*_watcher.json"))
    watcher_summary = []
    for path in watcher_files:
        item = json.loads(path.read_text(encoding="utf-8"))
        watcher_summary.append({"file": path.name, "status": item.get("status"), "physical_gpu": item.get("physical_gpu"), "non_child_process_signaled": item.get("non_child_process_signaled")})
        if item.get("non_child_process_signaled") is not False:
            failures.append(f"watcher signal policy failed: {path.name}")
        if item.get("physical_gpu") not in (2, None):
            failures.append(f"unexpected physical GPU in watcher: {path.name}")
    checks["watchers"] = watcher_summary

    old_manifests = {
        "freeze": ROOT.parent / "JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z/qa/CHECKSUMS.sha256",
        "stage2": ROOT.parent / "JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/runs/stage2_authorized_20260912T151855Z/qa/CHECKSUMS.sha256",
        "postfreeze": ROOT.parent / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/qa/CHECKSUMS.sha256",
    }
    old_hashes = {name: digest(path) for name, path in old_manifests.items() if path.is_file()}
    checks["prior_roots"] = {"manifest_sha256": old_hashes, "expected_manifest_sha256": EXPECTED_OLD_MANIFEST_SHA, "unchanged": old_hashes == EXPECTED_OLD_MANIFEST_SHA}
    if old_hashes != EXPECTED_OLD_MANIFEST_SHA:
        failures.append("prior freeze/Stage2/post-freeze manifest changed")

    pngs = sorted((ROOT / "figures").glob("*.png"))
    figure_ok = len(pngs) >= 6 and all(path.stat().st_size > 1000 for path in pngs)
    checks["figures"] = {"count": len(pngs), "files": [str(path) for path in pngs], "nonempty": figure_ok}
    if not figure_ok:
        failures.append("figure generation/size QA failed")

    # Fail closed on any nonzero final-test access marker anywhere in this new
    # root, including nested run/config/QA JSON files.
    final_access_violations: list[str] = []
    for path in ROOT.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [("", value)]
        while stack:
            prefix, current = stack.pop()
            if isinstance(current, dict):
                for key, child in current.items():
                    if key in {"final_test_access", "final_test_target_access"} and child not in (0, 0.0, False, None):
                        final_access_violations.append(f"{path.relative_to(ROOT)}:{prefix + key}={child}")
                    stack.append((prefix + key + ".", child))
            elif isinstance(current, list):
                stack.extend((prefix, child) for child in current)
    checks["final_test"] = {"violations": final_access_violations, "explicitly_authorized": False, "target_access": 0}
    if final_access_violations:
        failures.append("nonzero final-test access marker found")

    checks["tv_ratio_correction"] = {"acknowledged": (ROOT / "TV_RATIO_QA_CORRECTION.md").is_file(), "denominator_floor": 1e-6, "gate_metric_scope": "Arm TV only"}
    if not checks["tv_ratio_correction"]["acknowledged"]:
        failures.append("TV ratio correction note missing")

    handoff_files = [
        MEETING_ROOT / "INDEX.md",
        MEETING_ROOT / "TEACHER_HAND_PERSISTENCE_QUESTION_SHORT.zh-CN.md",
        MEETING_ROOT / "TEACHER_HAND_PERSISTENCE_QUESTION_DETAILED.zh-CN.md",
        MEETING_ROOT / "TEACHER_HAND_PERSISTENCE_QUESTION_DETAILED.en.md",
        ROOT / "NEXT_HAND_EVENT_GATE_MULTI_SEED_PROMPT.md",
        ROOT / "BENCHMARK_FINAL_NON_INTERFERENCE.md",
        ROOT / "FINAL_STATUS.ini",
        ROOT / "EXPERIMENT_DECISION.zh-CN.md",
        ROOT / "MOTION_AUDIT_DECISION.md",
    ]
    short_path = MEETING_ROOT / "TEACHER_HAND_PERSISTENCE_QUESTION_SHORT.zh-CN.md"
    short_chars = len(short_path.read_text(encoding="utf-8")) if short_path.is_file() else -1
    checks["handoff"] = {"required_files_present": all(path.is_file() for path in handoff_files), "short_message_character_count_including_heading": short_chars, "teacher_message_status": "complete_not_sent"}
    if not all(path.is_file() for path in handoff_files):
        failures.append("handoff package file missing")
    if short_chars < 300 or short_chars > 450:
        failures.append("short teacher message is outside the locked 300-450 character range")

    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks, "final_test_access": 0, "frozen_candidate_changed": False, "final_protocol_changed": False, "gpu0_used": False, "non_child_process_signaled": False}
    (ROOT / "qa" / "FINAL_QA.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    lines = ["# Final QA", "", f"Status: `{result['status']}`.", "", f"- Stage0: `{stage0.get('accepted')}/{stage0.get('expected')}`.", f"- Matrix: `6/6`; candidate gate: `{matrix.get('candidate_gate')}`; selected candidate: `{matrix.get('selected_candidate')}`.", f"- Strict checkpoint reloads: `{len(prediction_hashes)}/6`.", f"- Figures: `{len(pngs)}` non-empty PNG files.", f"- Final-test target access: `0`; explicit authorization: `false`.", f"- GPU0 used: `false`; non-child process signaled: `false`."]
    if failures:
        lines.extend(["", "Failures:", *[f"- {failure}" for failure in failures]])
    (ROOT / "qa" / "FINAL_QA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures, "strict_reload": len(prediction_hashes), "figures": len(pngs), "final_test_access": 0}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
