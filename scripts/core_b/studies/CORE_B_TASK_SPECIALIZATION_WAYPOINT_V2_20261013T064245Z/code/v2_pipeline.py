#!/usr/bin/env python3
"""Resumable V2 experiment coordinator. All GPU work goes through the shared watcher."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = str(PROJECT / "envs/jyc/bin/python")
CODE = ROOT / "code"
LOG_PATH = ROOT / "logs/V2_PIPELINE.log"
OWNER_PATH = ROOT / "coordination/OWNER_STATE.json"
LOCK_PATH = ROOT / "coordination/OWNER.lock"
SESSION = "ee6008-v2-20261013"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_owner(phase: str, *, status: str = "running", active_jobs: int = 0, next_gate: str = "") -> None:
    prior = json.loads(OWNER_PATH.read_text(encoding="utf-8")) if OWNER_PATH.exists() else {}
    payload = {
        "schema_version": 1,
        "created_utc": prior.get("created_utc", utc_now()),
        "updated_utc": utc_now(),
        "out_root": str(ROOT),
        "owner_lock_acquired": status == "running",
        "owner_lock_path": str(LOCK_PATH),
        "owner_pid": os.getpid(),
        "tmux_session": SESSION,
        "phase": phase,
        "status": status,
        "active_jobs": active_jobs,
        "parent_root_read_only": True,
        "final_heldout_payload_access": 0,
        "next_gate": next_gate,
    }
    atomic_json(OWNER_PATH, payload)


def log(message: str) -> None:
    stamp = utc_now()
    text = f"{stamp} {message}"
    print(text, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def run_command(argv: list[str], *, phase: str, required: bool = True) -> int:
    log(f"START phase={phase} argv={json.dumps(argv, ensure_ascii=False)}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as output:
        result = subprocess.run(argv, cwd=str(PROJECT), stdout=output, stderr=subprocess.STDOUT, check=False)
    log(f"END phase={phase} returncode={result.returncode}")
    if required and result.returncode:
        raise RuntimeError(f"phase {phase} command failed with exit {result.returncode}: {argv}")
    return result.returncode


def run_queue(queue_name: str, phase: str, expected: int, *, require_all: bool) -> dict[str, Any]:
    queue_path = ROOT / "coordination/queues" / queue_name
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    write_owner(phase, active_jobs=min(len(queue["jobs"]), 2), next_gate=f"waiting for {phase} watcher leases")
    run_command([PYTHON, str(CODE / "run_v2_queue.py"), "--queue", str(queue_path)], phase=phase, required=False)
    status_path = ROOT / "coordination" / f"PHASE_STATUS_{str(queue.get('phase')).upper()}.json"
    summary = json.loads(status_path.read_text(encoding="utf-8"))
    accepted = int(summary.get("accepted_fits", 0))
    scheduled = int(summary.get("scheduled_jobs", 0))
    if scheduled != expected:
        raise RuntimeError(f"{phase} scheduled job count mismatch: expected {expected}, found {scheduled}")
    if require_all and accepted != expected:
        raise RuntimeError(f"{phase} did not finish all required jobs: {accepted}/{expected}")
    return summary


def check_pass_receipt(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
    except Exception:
        return False


def main() -> int:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as owner_lock:
        try:
            fcntl.flock(owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another V2 owner holds the run lock") from exc
        write_owner("pilot_screening_train", active_jobs=2, next_gate="real 160-epoch task-119 screening pilots")
        try:
            run_command([PYTHON, str(CODE / "build_v2_jobs.py"), "--phase", "screening"], phase="build_screening")
            pilot_train = run_queue("SCREENING_PILOT.json", "pilot_screening_train", 2, require_all=True)

            run_command([PYTHON, str(CODE / "build_inference_jobs.py"), "--phase", "pilot_reload"], phase="build_pilot_reload")
            pilot_infer = run_queue("PILOT_RELOAD.json", "pilot_fresh_process_reload", 2, require_all=True)
            pilot_receipt = ROOT / "qa/FIRST_REAL_PILOT.json"
            if not check_pass_receipt(pilot_receipt):
                run_command([PYTHON, str(CODE / "accept_first_pilot.py")], phase="accept_first_real_pilot")
            if not check_pass_receipt(pilot_receipt):
                raise RuntimeError("first real pilot reload/prediction did not receive PASS")
            log(f"FIRST_REAL_PILOT PASS train={pilot_train['accepted_fits']}/2 inference={pilot_infer['accepted_fits']}/2")

            remaining_queue = json.loads((ROOT / "coordination/queues/SCREENING_REMAINING.json").read_text())
            remaining_expected = int(remaining_queue["fit_count"])
            run_queue("SCREENING_REMAINING.json", "screening_remaining", remaining_expected, require_all=False)
            selection_path = ROOT / "protocol/LR_EPOCH_SELECTION.json"
            if not selection_path.exists():
                run_command([PYTHON, str(CODE / "select_lr_epoch.py")], phase="lock_lr_epoch")
            selection = json.loads(selection_path.read_text())
            if selection.get("status") not in {"LOCKED", "LOCKED_WITH_PARTIAL_SEARCH"} or len(selection.get("logical_model_choices", [])) != 12:
                raise RuntimeError(f"inner-tune selection cannot support all 12 logical models: {selection.get('status')}")
            run_command([PYTHON, str(CODE / "build_v2_jobs.py"), "--phase", "formal_waypoint"], phase="build_formal_waypoint")

            run_queue("FORMAL.json", "formal_training", 36, require_all=False)
            run_queue("WAYPOINT.json", "waypoint_training", 6, require_all=False)
            run_command([PYTHON, str(CODE / "build_inference_jobs.py"), "--phase", "formal_dev"], phase="build_formal_dev_inference")
            infer_queue = json.loads((ROOT / "coordination/queues/FORMAL_DEV.json").read_text())
            run_queue("FORMAL_DEV.json", "fresh_process_dev_inference", int(infer_queue["inference_job_count"]), require_all=False)

            seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
            if not seal_path.exists():
                run_command([PYTHON, str(CODE / "seal_predictions_before_scoring.py")], phase="seal_predictions")
            run_command([PYTHON, str(CODE / "score_v2_results.py")], phase="dev_scoring_and_paired_stats", required=False)
            run_command([PYTHON, str(CODE / "make_v2_figures.py")], phase="build_figures", required=False)
            run_command([PYTHON, str(CODE / "finalize_v2.py")], phase="final_acceptance_and_checksums", required=False)
            final_path = ROOT / "qa/FINAL_ACCEPTANCE.json"
            final_status = json.loads(final_path.read_text()).get("status") if final_path.exists() else "PARTIAL"
            write_owner("complete" if final_status == "PASS" else "partial", status="complete" if final_status == "PASS" else "partial",
                        active_jobs=0, next_gate="human figure review pending" if final_status == "PASS" else "review acceptance receipts and resume failed/incomplete jobs")
            log(f"PIPELINE_END status={final_status}")
            return 0 if final_status == "PASS" else 2
        except Exception as exc:
            write_owner("technical_failure", status="technical_failure", active_jobs=0, next_gate=f"inspect V2 failure receipt/log and resume: {type(exc).__name__}: {exc}")
            log(f"PIPELINE_FAILURE {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
