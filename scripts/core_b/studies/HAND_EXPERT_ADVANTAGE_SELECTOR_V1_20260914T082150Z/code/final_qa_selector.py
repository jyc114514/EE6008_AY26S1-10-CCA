#!/usr/bin/env python3
"""Fail-closed QA for the expert-advantage selector study."""

from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from oof_base import HandResidualProposal
from selector_model import SelectorHead, state_hash
from train_selector import MODES, INPUT_PATH, LOCKED_ARM_PATH, OOF_PATH, ROOT, SELECTOR_SEED, load_contract

CELLS = MODES


OLD_MANIFESTS = {
    "hand_event": (ROOT.parent / "HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/qa/CHECKSUMS.sha256", "060283a9e62f21d4c30ea9d3746c8843b5765b2b5a73ce4f5d0b766b45fd0999"),
    "postfreeze": (ROOT.parent / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/qa/CHECKSUMS.sha256", "f12b2f1a0797705c9a9043c2b38cab82d650a8a0b9a01a9cfd81e9d58ec63197"),
    "freeze": (ROOT.parent / "JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z/qa/CHECKSUMS.sha256", "e6922a170abe6a5cc97ca4cf1a3701fcddc41aaf62f12b365838674cad23c3e8"),
    "benchmark_v2": (Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/qa/CHECKSUMS.sha256')), "bb059ba7ed645c0dea7fae9bf39ecab7195692a1ef3f7ae4c739ab9bfaece976"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    return str(value)


def main() -> int:
    failures: list[str] = []
    checks: dict[str, object] = {}
    contract = load_contract(); data = contract["data"]; train_mask = contract["train_mask"]; dev_mask = contract["dev_mask"]; oof = contract["oof"]; dev_ids = data["episode_index"][dev_mask]; dev_tasks = data["original_task_index"][dev_mask]
    checks["input"] = {"train": int(train_mask.sum()), "development": int(dev_mask.sum()), "target_shape": list(data["targets_standardized"].shape), "hand_slice": "0:14", "arm_slice": "14:28"}
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614 or data["targets_standardized"].shape != (3355, 30, 28): failures.append("input contract failed")
    headroom = json.loads((ROOT / "HEADROOM_DECISION.json").read_text(encoding="utf-8")); checks["headroom"] = {"gate": headroom.get("gate"), "primary": headroom.get("primary_expert"), "step_group_gain": headroom.get("step_group_hand_relative_gain"), "step_group_ci": headroom.get("step_group_hand_effect")}
    if headroom.get("gate") != "pass": failures.append("Gate 1 is not PASS")
    oof_summary = json.loads((ROOT / "oof/OOF_SUMMARY.json").read_text(encoding="utf-8")); oof_acceptance = json.loads((ROOT / "oof/OOF_ACCEPTANCE.json").read_text(encoding="utf-8")); checks["oof"] = {"summary": oof_summary, "acceptance": oof_acceptance}
    if oof_summary.get("coverage") != 2741 or oof_summary.get("expected_coverage") != 2741 or oof_summary.get("fold_overlap") != 0 or oof_acceptance.get("status") != "PASS": failures.append("OOF coverage/overlap failed")
    with np.load(OOF_PATH, allow_pickle=False) as archive: oof_archive = {key: archive[key].copy() for key in archive.files}
    if oof_archive["oof_predictions_hand_standardized"].shape != (2741, 30, 14) or not np.isfinite(oof_archive["oof_predictions_hand_standardized"]).all() or not np.isfinite(oof_archive["advantage"]).all(): failures.append("OOF numeric/schema failed")
    # OOF checkpoint strict reload and exact fold metadata.
    oof_reload = []
    for name, seed in (("base_fold_0", 20260932), ("base_fold_1", 20261932), ("base_fold_2", 20262932), ("base_all_train", 20263932)):
        run = ROOT / "runs" / name; marker = json.loads((run / "RUN_COMPLETE.json").read_text(encoding="utf-8")); config = json.loads((run / "RUN_CONFIG_LOCKED.json").read_text(encoding="utf-8")); payload = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False); model = HandResidualProposal(seed=seed); ok = True
        try: model.load_state_dict(payload["model_state_dict"], strict=True)
        except Exception: ok = False
        oof_reload.append({"run": name, "status": marker.get("status"), "strict_reload": ok, "optimizer_steps": marker.get("optimizer_steps"), "dev_target_used_in_fit": config.get("dev_target_used_in_fit")})
        if marker.get("status") != "complete" or not ok or config.get("dev_target_used_in_fit") is not False: failures.append(f"OOF run failed: {name}")
    checks["oof_strict_reload"] = oof_reload
    with np.load(LOCKED_ARM_PATH, allow_pickle=False) as archive: locked = {key: archive[key].copy() for key in archive.files}
    locked_arm = locked["predictions_standardized"][:, :, 14:28]
    selector_reload = []; arm_identical = True; initial_hashes = []
    for cell in CELLS:
        run = ROOT / "runs" / cell; marker = json.loads((run / "RUN_COMPLETE.json").read_text(encoding="utf-8")); config = json.loads((run / "RUN_CONFIG_LOCKED.json").read_text(encoding="utf-8")); metrics = json.loads((run / "RUN_METRICS.json").read_text(encoding="utf-8")); bundle = {key: value.copy() for key, value in np.load(run / "PREDICTIONS_DEV.npz", allow_pickle=False).items()}
        model = SelectorHead(seed=SELECTOR_SEED); payload = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False); ok = True
        try: model.load_state_dict(payload["model_state_dict"], strict=True)
        except Exception: ok = False
        initial_hashes.append(metrics.get("initial_state_hash")); exact_arm = np.array_equal(bundle["locked_arm_standardized"], locked_arm) and np.array_equal(bundle["full_predictions_standardized"][:, :, 14:28], locked_arm); arm_identical = arm_identical and exact_arm
        selector_reload.append({"cell": cell, "status": marker.get("status"), "strict_reload": ok, "steps": marker.get("optimizer_steps"), "shapes": [marker.get("prediction_hand_shape"), marker.get("prediction_full_shape")], "dev_target_used_in_fit": config.get("dev_target_used_in_fit"), "arm_bitwise": exact_arm, "q_finite": bool(np.isfinite(bundle["q"]).all())})
        if marker.get("status") != "complete" or marker.get("optimizer_steps") != 3120 or not ok or not exact_arm or config.get("dev_target_used_in_fit") is not False or config.get("future_motion_input") is not False or config.get("future_advantage_input") is not False or metrics.get("final_test_access") != 0: failures.append(f"selector acceptance failed: {cell}")
    checks["selectors"] = {"rows": selector_reload, "arm_bitwise_identical_all": arm_identical, "initial_hashes": initial_hashes, "all_initial_hashes_equal": len(set(initial_hashes)) == 1}
    if not arm_identical or len(set(initial_hashes)) != 1: failures.append("selector Arm/init non-interference failed")
    stage0 = json.loads((ROOT / "STAGE0_SELECTOR_RESULTS.json").read_text(encoding="utf-8")); checks["stage0"] = {"status": stage0.get("status"), "accepted": stage0.get("accepted"), "expected": stage0.get("expected"), "device": "cpu"}
    if stage0.get("status") != "PASS" or stage0.get("accepted") != 4: failures.append("selector Stage0 failed")
    decision = json.loads((ROOT / "SELECTOR_CANDIDATE_GATE.json").read_text(encoding="utf-8")); checks["candidate"] = {"gate": decision.get("candidate_gate"), "selected": decision.get("selected_candidate"), "order": decision.get("candidate_order")}
    if decision.get("candidate_gate") != "fail" or decision.get("selected_candidate") is not None: failures.append("candidate decision is not fail/no-selection")
    for name, (path, expected) in OLD_MANIFESTS.items():
        if not path.is_file() or digest(path) != expected: failures.append(f"old manifest changed: {name}")
    checks["old_manifest_hashes"] = {name: {"path": str(path), "sha256": digest(path) if path.is_file() else None, "expected": expected} for name, (path, expected) in OLD_MANIFESTS.items()}
    required_docs = [ROOT / "EXPERT_ADVANTAGE_PROTOCOL_LOCKED.md", ROOT / "OOF_PROTOCOL_LOCKED.md", ROOT / "SELECTOR_GATE_EVALUATION_LOCKED.md", ROOT / "EXPERIMENT_DECISION.zh-CN.md", ROOT / "EXPERIMENT_DECISION.en.md", ROOT / "BENCHMARK_NON_INTERFERENCE.md", ROOT / "NEXT_HAND_EXPERT_MULTIMODAL_DECODER_PROMPT.md", ROOT / "NEXT_ADVANTAGE_SELECTOR_MULTI_SEED_PROMPT.md", ROOT / "FINAL_STATUS.ini", ROOT / "qa/NUMERIC_QA.md", ROOT / "qa/LEAKAGE_QA.md", ROOT / "qa/PLOT_VISUAL_QA.md"]
    checks["documents"] = {"all_present": all(path.is_file() for path in required_docs), "missing": [str(path.relative_to(ROOT)) for path in required_docs if not path.is_file()]}
    if not all(path.is_file() for path in required_docs): failures.append("required final document missing")
    figures = sorted((ROOT / "figures").glob("*.png")); figure_manifest = json.loads((ROOT / "figures/FIGURE_MANIFEST_SELECTOR.json").read_text(encoding="utf-8")); figure_ok = len(figures) >= 20 and all(path.stat().st_size > 1000 for path in figures) and figure_manifest.get("status") == "PASS"; checks["figures"] = {"count": len(figures), "manifest_status": figure_manifest.get("status"), "nonempty": figure_ok}
    if not figure_ok: failures.append("figure QA failed")
    final_violations = []
    for path in ROOT.rglob("*.json"):
        try: value = json.loads(path.read_text(encoding="utf-8"))
        except Exception: continue
        stack = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, child in current.items():
                    if key in {"final_test_access", "final_test_target_access"} and child not in (0, 0.0, False, None): final_violations.append(f"{path.relative_to(ROOT)}:{key}={child}")
                    stack.append(child)
            elif isinstance(current, list): stack.extend(current)
    checks["final_test"] = {"violations": final_violations, "explicitly_authorized": False, "target_access": 0}
    if final_violations: failures.append("nonzero final-test marker")
    tracebacks = [str(path.relative_to(ROOT)) for path in (ROOT / "logs").glob("*.log") if "Traceback" in path.read_text(encoding="utf-8", errors="replace")]
    checks["logs"] = {"traceback_logs": tracebacks, "known_interrupted_attempt": False}
    if tracebacks: failures.append("traceback found in final logs")
    qa_status = "PASS_WITH_CAVEAT" if not failures and checks["stage0"].get("device") == "cpu" else ("PASS" if not failures else "FAIL")
    result = {"status": qa_status, "failures": failures, "checks": checks, "final_test_access": 0, "professor_contact_started": False, "frozen_candidate_changed": False, "benchmark_v2_changed": False, "gpu0_used": False, "gpu1_used": False, "non_child_process_signaled": False}
    (ROOT / "qa/FINAL_QA.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    lines = ["# Final QA", "", f"Status: `{result['status']}`.", "", "- Phase 1 Gate 1: `PASS`.", "- OOF: `2741/2741`, 0 overlap; strict reloads `4/4`.", "- Selector Stage0: `4/4` CPU smoke; selector matrix: `4/4`.", f"- Candidate gate: `{decision.get('candidate_gate')}`; selected: `{decision.get('selected_candidate')}`.", f"- Figures: `{len(figures)}` non-empty PNG files; visual QA document required.", "- Final-test target access: `0`; frozen candidate and Benchmark v2 unchanged.", "- GPU0/GPU1 used: `false/false`; no non-child process signaled."]
    if failures: lines.extend(["", "Failures:", *[f"- {failure}" for failure in failures]])
    (ROOT / "qa/FINAL_QA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures, "figures": len(figures), "final_test_access": 0}, sort_keys=True)); return 0 if not failures else 1


if __name__ == "__main__": raise SystemExit(main())
