#!/usr/bin/env python3
"""Archive a corrected-split zero-prediction seal/scoring attempt before inference retry."""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/CORRECTED_SPLIT_ZERO_PREDICTION_SCORING_ATTEMPT_20261013T092800Z"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def move(rel: str, moved: list[str]) -> None:
    source = ROOT / rel
    if not source.exists():
        return
    target = ARCHIVE / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(source), str(target))
    moved.append(rel)


def main() -> None:
    if ARCHIVE.exists():
        raise FileExistsError(ARCHIVE)
    owner_lock = (ROOT / "coordination/OWNER.lock").open("a+")
    try:
        fcntl.flock(owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("V2 owner is active; refuse to archive current scorer attempt") from exc
    seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if int(seal.get("sealed_prediction_jobs", -1)) != 0 or len(seal.get("missing_prediction_jobs", [])) != 42:
        raise RuntimeError("active seal is not the expected 0/42 corrected-split inference attempt")
    target_path = ROOT / "data/DEV_TARGETS_SCORER_ONLY.npz"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    exposure_path = ROOT / "inventory/OUTCOME_EXPOSURE.json"
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    scoring_qa_path = ROOT / "qa/DEV_TARGET_SCORING_QA.json"
    scoring_qa = json.loads(scoring_qa_path.read_text(encoding="utf-8"))
    if scoring_qa.get("status") != "PASS" or int(scoring_qa.get("selected_dev_windows", -1)) != 185:
        raise RuntimeError("scorer-only dev target access receipt differs from observed V2 support")
    result_path = ROOT / "results/RESULTS_SUMMARY.json"
    result_status = json.loads(result_path.read_text()).get("status") if result_path.exists() else "missing"
    formal_queue = json.loads((ROOT / "coordination/queues/FORMAL_DEV.json").read_text())
    job_ids = []
    for raw in formal_queue["jobs"]:
        watcher = json.loads(Path(raw).read_text())
        job_ids.append(str(watcher["job_id"]))
    current_statuses = []
    for job_id in job_ids:
        path = ROOT / "coordination/watcher_status" / f"{job_id}.json"
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "running":
                raise RuntimeError(f"watcher still active during zero-prediction archive: {job_id}")
            current_statuses.append(state)
    if sum(1 for state in current_statuses if state.get("child_pid")) < 1:
        raise RuntimeError("no inference child launch receipts found; expected reset_peak_memory_stats failures")

    ARCHIVE.mkdir(parents=True)
    moved: list[str] = []
    for rel in (
        "protocol/PREDICTION_BATCH_SEAL.json",
        "data/DEV_TARGETS_SCORER_ONLY.npz",
        "qa/DEV_TARGET_SCORING_QA.json", "qa/FIGURE_BUILD_QA.json", "qa/FINAL_ACCEPTANCE.json",
        "results", "figures", "figure_data", "predictions/dev",
        "coordination/JOB_LEDGER.json", "coordination/PHASE_STATUS_FORMAL_DEV.json",
        "coordination/OWNER_STATE.json", "RETURN_TO_CHATGPT.zh-CN.md", "RETURN_TO_CHATGPT.json",
    ):
        move(rel, moved)
    for job_id in job_ids:
        for rel_dir in (
            "coordination/watcher_status", "coordination/watcher_status/console", "logs/watcher_children",
        ):
            folder = ROOT / rel_dir
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.name in {f"{job_id}.json", f"{job_id}.log"} or path.name.startswith(f"{job_id}.attempt_"):
                    move(path.relative_to(ROOT).as_posix(), moved)

    event = {
        "event_id": "corrected_split_zero_prediction_seal_attempt",
        "split": "literal V2 canonical source key",
        "prediction_seal_sha256": sha(ARCHIVE / "protocol/PREDICTION_BATCH_SEAL.json"),
        "sealed_prediction_jobs": 0,
        "missing_prediction_jobs": 42,
        "watcher_jobs_scheduled": len(job_ids),
        "watcher_child_receipts": sum(1 for state in current_statuses if state.get("child_pid")),
        "inference_error": "torch.cuda.reset_peak_memory_stats(torch.device('cuda:0')) raised RuntimeError: Invalid device argument before model/data load",
        "dev_target_artifact_sha256": sha(ARCHIVE / "data/DEV_TARGETS_SCORER_ONLY.npz"),
        "dev_target_windows_loaded_by_scorer": int(scoring_qa["selected_dev_windows"]),
        "dev_future_action_rows_loaded_by_scorer": int(scoring_qa["future_action_rows_read"]),
        "target_access_order": "after the zero-prediction seal, but the seal lacked any valid model prediction; invalid scoring attempt",
        "result_summary_status": result_status,
        "score_outputs_used_for_model_selection": False,
        "human_read_of_score_values": False,
        "final_heldout_payload_access": 0,
        "archive_root": str(ARCHIVE),
        "moved_artifact_count": len(moved),
        "moved_paths": moved,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (ARCHIVE / "ARCHIVE_SUMMARY.json").write_text(json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    backup = exposure_path.with_name("OUTCOME_EXPOSURE.json.pre-empty-prediction-score-20261013T092800Z.bak")
    if not backup.exists():
        shutil.copy2(exposure_path, backup)
    historical = exposure.get("dev_target_access_history", [])
    if not any(row.get("event_id") == event["event_id"] for row in historical):
        historical.append(event)
    exposure.update({
        "dev_target_access_history": historical,
        "dev_target_labels_read": True,
        "dev_predictions_or_errors_read": True,
        "dev_target_labels_read_after_prediction_seal": True,
        "current_split_dev_targets_pending_after_valid_prediction_seal": True,
        "current_split_dev_metrics_written": False,
        "current_split_prediction_seal_sha256": None,
        "prior_split_scores_used_for_corrected_selection": False,
        "empty_prediction_scoring_attempts": 1,
        "final_heldout_payload_access": 0,
        "new_final_aggregate_counts_read_or_printed": False,
    })
    exposure_path.write_text(json.dumps(exposure, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "archived_sealed_predictions": 0, "missing": 42,
        "dev_target_rows_in_invalid_attempt": scoring_qa["future_action_rows_read"],
        "active_dev_target_path_exists": target_path.exists(),
        "final_heldout_payload_access": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
