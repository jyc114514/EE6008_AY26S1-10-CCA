#!/usr/bin/env python3
"""Prepare and execute bounded VideoMAE GPU replays through the shared project watcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


STAGE0_CELLS = [
    "TS11-D_H30_S20261001", "TS11-D_H60_S20261001",
    "STATE-CTRL_H30_S20261001", "STATE-CTRL_H60_S20261001"
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--v2-root", type=Path, required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--lease-dir", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return p


def queue_paths(audit: Path, phase: str, cell: str):
    job_dir = audit / "status/jobs" / phase
    job_path = job_dir / f"{cell}.json"
    status_path = audit / "status" / phase / f"{cell}.json"
    log_path = audit / "logs" / phase / f"{cell}.log"
    return job_path, status_path, log_path


def historical_peak(v2: Path) -> tuple[int, list[dict]]:
    archive = read_json(v2 / "ARCHIVE.json")
    records = []
    for cell in archive.get("cells", []):
        manifest = read_json(v2 / cell["manifest"])
        checkpoint_path = Path(manifest["original_checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = v2 / checkpoint_path
        status = checkpoint_path.parent / "RUN_STATUS.json"
        # Only the sibling RUN_STATUS.json is parsed; the checkpoint is never opened here.
        run_status = read_json(status)
        peak = run_status.get("peak_reserved_mib")
        if not isinstance(peak, int) or peak <= 0:
            raise RuntimeError(f"missing historical peak_reserved_mib for {cell['name']}")
        records.append({"cell": cell["name"], "peak_reserved_mib": peak, "source": str(status)})
    if len(records) != 12:
        raise RuntimeError(f"expected 12 historical run resource records, found {len(records)}")
    return max(int(r["peak_reserved_mib"]) for r in records), records


def build_job(project: Path, v2: Path, audit: Path, cell: str, phase: str,
              peak_mib: int, preferred_gpu: int | None = None) -> tuple[dict, Path, Path, Path]:
    python = project / "envs/jyc/bin/python"
    runner = audit / "source/replay_one_cell.py"
    job_path, status_path, log_path = queue_paths(audit, phase, cell)
    kind = "gpu-stage0" if phase == "stage0" else "gpu-full"
    command = [
        str(python), str(runner), "--project-root", str(project), "--v2-root", str(v2),
        "--audit-root", str(audit), "--cell", cell, "--device", "cuda", "--run-kind", kind
    ]
    output = audit / "replay" / ("gpu_stage0" if phase == "stage0" else "gpu") / cell
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(audit / "tmp"),
        "XDG_CACHE_HOME": str(audit / ".cache"),
        "MPLCONFIGDIR": str(audit / ".mplconfig"),
        "TORCH_HOME": str(audit / ".torch"),
        "TORCHINDUCTOR_CACHE_DIR": str(audit / ".torchinductor"),
        "TRITON_CACHE_DIR": str(audit / ".triton_cache"),
        "CUDA_CACHE_PATH": str(audit / ".cuda_cache"),
        "AUDIT_GPU_WATCHER_STATUS_PATH": str(status_path)
    }
    job = {
        "job_id": f"videomae_acceptance_v2_{phase}_{cell}",
        "protocol_id": "VIDEOMAE_CORE_B_HANDOFF_ACCEPTANCE_V2",
        "output_root": str(output),
        "command": command,
        "environment": environment,
        "resource_estimate": {
            "measured_peak_mib": int(peak_mib),
            "pre_registration_source": "historical_RUN_STATUS.peak_reserved_mib" if phase == "stage0" else "same-runner GPU Stage0 peak_reserved_mib",
            "historical_training_peak_not_same_runner": phase == "stage0",
            "purpose": "64-row Stage0 admission estimate only" if phase == "stage0" else "formal replay admitted using same-runner Stage0 measured peak"
        },
        "max_oom_requeues": 0,
        "log_path": str(log_path)
    }
    if preferred_gpu is not None:
        job["preferred_gpu"] = int(preferred_gpu)
    return job, job_path, status_path, log_path


def watcher_command(project: Path, job_path: Path, status_path: Path, lease_dir: Path,
                    *, dry_run: bool, max_wait_seconds: int) -> list[str]:
    return [
        str(project / "envs/jyc/bin/python"),
        str(project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"),
        "--job-json", str(job_path), "--lease-dir", str(lease_dir),
        "--status-path", str(status_path), "--poll-seconds", "30",
        "--max-wait-seconds", str(max_wait_seconds), "--max-oom-requeues", "0"
    ] + (["--dry-run"] if dry_run else [])


def stage0_queue(audit: Path, project: Path, v2: Path):
    peak, source_records = historical_peak(v2)
    jobs = []
    for cell in STAGE0_CELLS:
        job, job_path, status_path, log_path = build_job(project, v2, audit, cell, "stage0", peak)
        write_json(job_path, job)
        dry_output = subprocess.run(
            watcher_command(project, job_path, status_path, project / "reports/locks/gpu_lease",
                            dry_run=True, max_wait_seconds=300),
            cwd=str(audit / "tmp"), capture_output=True, text=True, check=True
        )
        result = json.loads(dry_output.stdout.strip().splitlines()[-1])
        jobs.append({"cell": cell, "job_path": str(job_path), "status_path": str(status_path),
                     "log_path": str(log_path), "dry_run": result})
    receipt = {
        "status": "PASS_WATCHER_DRY_RUN",
        "mode": "dry-run; no GPU query, allocation, or child launch",
        "visible_gpu_pool": "dynamic_all_visible",
        "old_fixed_gpu_number_inherited": False,
        "pre_registered_stage0_peak_mib": peak,
        "historical_peak_source_records": source_records,
        "stage0_jobs": jobs,
        "max_wait_seconds": 300,
        "max_oom_requeues": 0,
        "final_test_access": 0
    }
    write_json(audit / "GPU_WATCHER_DRY_RUN.json", receipt)
    print(json.dumps({"status": receipt["status"], "stage0_jobs": len(jobs), "peak_mib": peak}, ensure_ascii=False))


def run_one(project: Path, v2: Path, audit: Path, lease: Path, cell: str, phase: str,
            peak: int, preferred_gpu: int | None):
    job, job_path, status_path, log_path = build_job(project, v2, audit, cell, phase, peak, preferred_gpu)
    write_json(job_path, job)
    command = watcher_command(project, job_path, status_path, lease, dry_run=False, max_wait_seconds=300)
    result = subprocess.run(command, cwd=str(audit / "tmp"), check=False)
    status = read_json(status_path) if status_path.is_file() else {"status": "missing_status"}
    return result.returncode, status, job, status_path, log_path



def update_replay_rows(audit: Path, status: str, reason: str, *, completed_cell: str | None = None,
                       receipt: dict | None = None, reference_map: dict | None = None) -> None:
    path = audit / "CHECKPOINT_REPLAY.csv"
    if not path.is_file():
        return
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if completed_cell is not None and receipt is not None:
        outputs = {item["split"]: item for item in receipt.get("outputs", [])}
        for row in rows:
            if row["cell"] != completed_cell:
                continue
            split = row["split"]
            out = outputs.get(split, {})
            ref = (reference_map or {}).get((completed_cell, split), {})
            runtime = receipt.get("runtime", {})
            gpu = runtime.get("physical_gpu_uuid_receipt") or {}
            row.update({
                "status": "generated_unvalidated", "executed": "1", "numeric_accepted": "0",
                "device": runtime.get("device", ""), "physical_gpu": gpu.get("physical_gpu", ""),
                "logical_device": gpu.get("logical_device", ""), "gpu_uuid": gpu.get("watcher_uuid", ""),
                "prediction_sha256": out.get("sha256", ""), "reference_sha256": ref.get("sha256", ""),
                "dtype": out.get("dtype", row.get("dtype", "")),
                "block_reason": "prediction generated; waiting for complete 24-output structure gate and seal"
            })
    else:
        for row in rows:
            if row.get("executed") == "0" and row.get("status", "").startswith("awaiting") or row.get("executed") == "0" and row.get("status") == "not_tested_running":
                row["status"] = status
                row["block_reason"] = reason
    temp = path.with_name("." + path.name + ".part")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)

def write_stage_status(audit: Path, status: str, phase: str, cell: str, watcher_status: dict, details: dict) -> None:
    write_json(audit / "GPU_RESOURCE_STATUS.json", {
        "status": status, "phase": phase, "cell": cell, "utc": utc_now(),
        "watcher_status": watcher_status, "details": details,
        "external_process_identity_recorded": False, "non_child_process_signaled": False,
        "new_training": 0, "optimizer_updates": 0, "new_encoder_extraction": 0,
        "final_test_access": 0
    })


def run_all(project: Path, v2: Path, audit: Path, lease: Path) -> int:
    if not os.environ.get("TMUX"):
        raise RuntimeError("GPU replay orchestrator must run inside tmux")
    host_receipt_path = audit / "GPU_HOST_PREFLIGHT.json"
    if not host_receipt_path.is_file():
        raise RuntimeError("run the user-approved host_preflight.py first")
    host_receipt = read_json(host_receipt_path)
    if host_receipt.get("status") != "PASS_AGGREGATE_HOST_PREFLIGHT" or not host_receipt.get("gpus"):
        raise RuntimeError("host aggregate resource preflight is incomplete")
    if host_receipt.get("external_process_identity_recorded") is not False:
        raise RuntimeError("host preflight must not contain other-process identity data")
    if not (audit / "ARCHIVE_CPU_FIXTURE_SUMMARY.json").is_file():
        raise RuntimeError("four CPU archive integration fixtures must pass first")
    cpu_fixtures = read_json(audit / "ARCHIVE_CPU_FIXTURE_SUMMARY.json")
    if cpu_fixtures.get("status") != "PASS":
        raise RuntimeError("CPU archive fixture gate has not passed")

    session = subprocess.run(["tmux", "display-message", "-p", "#S"], check=True, capture_output=True, text=True).stdout.strip()
    run_receipt = {
        "status": "running", "started_utc": utc_now(), "tmux_session": session,
        "working_directory": os.getcwd(), "command": [str(audit / "REPRODUCE_V2.sh"), "--gpu-replay"],
        "project_root": str(project), "v2_root": str(v2), "audit_root": str(audit),
        "lease_dir": str(lease), "visible_gpu_pool": "dynamic_all_visible",
        "new_training": 0, "optimizer_updates": 0, "new_encoder_extraction": 0,
        "unrestricted_pickle_loads": 0, "final_test_access": 0, "old_root_writes": 0
    }
    write_json(audit / "RUN_RECEIPT.json", run_receipt)
    update_replay_rows(audit, "not_tested_running", "host approval obtained; waiting for GPU admission")
    provisional_peak, historical_records = historical_peak(v2)

    first_cell = STAGE0_CELLS[0]
    stage0_receipts = []
    preferred_gpu = None
    for cell in STAGE0_CELLS:
        returncode, watcher_status, job, status_path, log_path = run_one(
            project, v2, audit, lease, cell, "stage0", provisional_peak, preferred_gpu)
        if returncode != 0 or watcher_status.get("status") != "complete":
            kind = watcher_status.get("status")
            resource_status = "resource_waiting" if kind in {
                "capacity_timeout", "waiting_for_memory_capacity", "waiting_for_project_child_slot",
                "project_child_slot_timeout", "local_gpu_lease_or_recheck_timeout",
                "waiting_final_memory_recheck_race", "waiting_for_local_gpu_lease"
            } else "technical_failure"
            write_stage_status(audit, resource_status, "stage0", cell, watcher_status,
                               {"exit_code": returncode, "log_path": str(log_path), "job": job})
            update_replay_rows(audit, resource_status, f"Stage0 stopped before formal replay: watcher={kind}")
            run_receipt.update({"status": resource_status, "completed_utc": utc_now(), "stage0_completed": len(stage0_receipts),
                                "formal_cells_executed": 0, "blocked_at_cell": cell})
            write_json(audit / "RUN_RECEIPT.json", run_receipt)
            print(json.dumps({"status": resource_status, "blocked_at": cell, "watcher_status": kind}, ensure_ascii=False))
            return 0 if resource_status == "resource_waiting" else returncode or 1
        output = audit / "replay/gpu_stage0" / cell / "REPLAY_RECEIPT.json"
        if not output.is_file():
            raise RuntimeError(f"GPU Stage0 child receipt missing: {output}")
        receipt = read_json(output)
        if receipt.get("status") != "PASS_GPU_STAGE0" or receipt.get("target_arrays_loaded") is not False:
            raise RuntimeError(f"GPU Stage0 scope/status mismatch: {cell}")
        if preferred_gpu is None:
            profile = read_json(audit / "PROFILE_LOCK.json")["new_current_profile"]
            preferred_gpu = int(profile["physical_gpu"])
        if receipt["runtime"]["physical_gpu_uuid_receipt"]["watcher_uuid"] != read_json(audit / "PROFILE_LOCK.json")["new_current_profile"]["gpu_uuid"]:
            raise RuntimeError(f"GPU Stage0 UUID mismatch: {cell}")
        stage0_receipts.append({"cell": cell, "receipt": str(output),
                                "peak_allocated_mib": receipt["runtime"]["peak_allocated_mib"],
                                "peak_reserved_mib": receipt["runtime"]["peak_reserved_mib"],
                                "gpu_uuid": receipt["runtime"]["physical_gpu_uuid_receipt"]["watcher_uuid"],
                                "physical_gpu": receipt["runtime"]["physical_gpu_uuid_receipt"]["physical_gpu"]})

    same_runner_peak = max(int(r["peak_reserved_mib"]) for r in stage0_receipts)
    if same_runner_peak <= 0:
        raise RuntimeError("Stage0 returned no positive reserved-memory measurement")
    write_json(audit / "GPU_RESOURCE_RECEIPT.json", {
        "status": "PASS_SAME_RUNNER_STAGE0_MEMORY_MEASURED",
        "pre_registered_stage0_peak_mib": provisional_peak,
        "pre_registered_source": "maximum historical formal RUN_STATUS.peak_reserved_mib over twelve verified cells",
        "historical_source_records": historical_records,
        "same_runner_stage0_peak_reserved_mib": same_runner_peak,
        "same_runner_stage0_peak_allocated_mib": max(int(r["peak_allocated_mib"]) for r in stage0_receipts),
        "formal_required_peak_mib": same_runner_peak,
        "stage0_receipts": stage0_receipts,
        "same_gpu_uuid": read_json(audit / "PROFILE_LOCK.json")["new_current_profile"]["gpu_uuid"],
        "final_test_access": 0
    })

    expected = list(csv.DictReader((audit / "EXPECTED_CELLS.csv").open(newline="", encoding="utf-8")))
    formal_results = []
    for row in expected:
        cell = row["cell"]
        returncode, watcher_status, job, status_path, log_path = run_one(
            project, v2, audit, lease, cell, "gpu", same_runner_peak, preferred_gpu)
        if returncode != 0 or watcher_status.get("status") != "complete":
            kind = watcher_status.get("status")
            resource_status = "resource_waiting" if kind in {
                "capacity_timeout", "waiting_for_memory_capacity", "waiting_for_project_child_slot",
                "project_child_slot_timeout", "local_gpu_lease_or_recheck_timeout",
                "waiting_final_memory_recheck_race", "waiting_for_local_gpu_lease"
            } else "technical_failure"
            write_stage_status(audit, resource_status, "formal", cell, watcher_status,
                               {"exit_code": returncode, "log_path": str(log_path), "job": job,
                                "formal_cells_completed": len(formal_results)})
            update_replay_rows(audit, resource_status, f"formal replay stopped at {cell}: watcher={kind}")
            run_receipt.update({"status": resource_status, "completed_utc": utc_now(),
                                "stage0_completed": len(stage0_receipts), "formal_cells_completed": len(formal_results),
                                "blocked_at_cell": cell})
            write_json(audit / "RUN_RECEIPT.json", run_receipt)
            print(json.dumps({"status": resource_status, "blocked_at": cell, "watcher_status": kind}, ensure_ascii=False))
            return 0 if resource_status == "resource_waiting" else returncode or 1
        output = audit / "replay/gpu" / cell / "REPLAY_RECEIPT.json"
        if not output.is_file():
            raise RuntimeError(f"formal GPU child receipt missing: {output}")
        receipt = read_json(output)
        if receipt.get("status") != "PASS_GPU_FROZEN_REPLAY":
            raise RuntimeError(f"formal GPU child did not complete: {cell}")
        child_uuid = receipt["runtime"]["physical_gpu_uuid_receipt"]["watcher_uuid"]
        locked_uuid = read_json(audit / "PROFILE_LOCK.json")["new_current_profile"]["gpu_uuid"]
        if child_uuid != locked_uuid:
            raise RuntimeError(f"formal replay GPU UUID differs from profile lock: {cell}")
        formal_results.append({"cell": cell, "receipt": str(output), "gpu_uuid": child_uuid,
                               "physical_gpu": receipt["runtime"]["physical_gpu_uuid_receipt"]["physical_gpu"],
                               "outputs": receipt["outputs"]})
        contract = read_json(audit / "REPLAY_CONTRACT.json")
        reference_map = {(r["cell"], r["split"]): r for r in contract.get("references", [])}
        update_replay_rows(audit, "generated_unvalidated", "", completed_cell=cell,
                           receipt=receipt, reference_map=reference_map)

    validate_cmd = [str(project / "envs/jyc/bin/python"), str(audit / "source/validate_predictions.py"),
                    "--project-root", str(project), "--v2-root", str(v2), "--audit-root", str(audit)]
    validation = subprocess.run(validate_cmd, cwd=str(audit / "tmp"), check=False)
    if validation.returncode != 0:
        run_receipt.update({"status": "technical_failure", "completed_utc": utc_now(),
                            "stage0_completed": len(stage0_receipts), "formal_cells_completed": len(formal_results),
                            "prediction_validation_exit_code": validation.returncode})
        write_json(audit / "RUN_RECEIPT.json", run_receipt)
        return validation.returncode

    scorer_cmd = [str(project / "envs/jyc/bin/python"), str(audit / "source/score_predictions.py"),
                  "--project-root", str(project), "--v2-root", str(v2), "--v1-root", str(project / "reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z"),
                  "--audit-root", str(audit)]
    scoring = subprocess.run(scorer_cmd, cwd=str(audit / "tmp"), check=False)
    run_receipt.update({"status": "complete" if scoring.returncode == 0 else "technical_failure",
                        "completed_utc": utc_now(), "stage0_completed": len(stage0_receipts),
                        "formal_cells_completed": len(formal_results), "prediction_validation_exit_code": validation.returncode,
                        "scoring_exit_code": scoring.returncode, "gpu_uuid": read_json(audit / "PROFILE_LOCK.json")["new_current_profile"]["gpu_uuid"],
                        "same_runner_stage0_peak_reserved_mib": same_runner_peak})
    write_json(audit / "RUN_RECEIPT.json", run_receipt)
    print(json.dumps({"status": run_receipt["status"], "stage0_completed": len(stage0_receipts),
                      "formal_cells_completed": len(formal_results), "scoring_exit_code": scoring.returncode},
                     ensure_ascii=False))
    return scoring.returncode


def main():
    args = build_parser().parse_args()
    project, v2, audit, lease = [p.resolve() for p in (args.project_root, args.v2_root, args.audit_root, args.lease_dir)]
    if args.prepare_dry_run:
        stage0_queue(audit, project, v2)
        return 0
    return run_all(project, v2, audit, lease)


if __name__ == "__main__":
    raise SystemExit(main())

