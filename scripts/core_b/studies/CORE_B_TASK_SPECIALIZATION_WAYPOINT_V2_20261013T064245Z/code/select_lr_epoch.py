#!/usr/bin/env python3
"""Select each logical model's LR/checkpoint only from inner-tune canonical metrics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = (119, 94, 90, 141, 230)
LR_PRIORITY = {1e-3: 0, 5e-4: 1, 2e-3: 2}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_fit_configs() -> list[tuple[dict[str, Any], Path]]:
    queue = json.loads((ROOT / "coordination/queues/SCREENING_ALL.json").read_text(encoding="utf-8"))
    out: list[tuple[dict[str, Any], Path]] = []
    for watcher_path in queue["jobs"]:
        watcher = json.loads(Path(watcher_path).read_text(encoding="utf-8"))
        fit_path = Path(watcher["command"][-1])
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        out.append((fit, Path(fit["job_root"])))
    return out


def retry_incomplete_screening_fits() -> int:
    queue_path = ROOT / "coordination/queues/SCREENING_ALL.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    incomplete: list[Path] = []
    for watcher_raw in queue["jobs"]:
        watcher_path = Path(watcher_raw)
        watcher = json.loads(watcher_path.read_text(encoding="utf-8"))
        fit = json.loads(Path(watcher["command"][-1]).read_text(encoding="utf-8"))
        out = Path(fit["job_root"])
        complete = out / "RUN_COMPLETE.json"
        if complete.is_file() and json.loads(complete.read_text()).get("status") == "PASS":
            continue
        incomplete.append(watcher_path)
        job_id = str(fit["job_id"])
        run_status = out / "RUN_STATUS.json"
        watcher_status = ROOT / "coordination/watcher_status" / f"{job_id}.json"
        child_log = ROOT / "logs/watcher_children" / f"{job_id}.log"
        if run_status.exists():
            current = json.loads(run_status.read_text(encoding="utf-8"))
            if current.get("status") == "failed":
                old_attempts = sorted(out.glob("RUN_STATUS.failed-attempt-*.json"))
                same = False
                for old in old_attempts:
                    try:
                        if json.loads(old.read_text(encoding="utf-8")) == current:
                            same = True
                            break
                    except Exception:
                        continue
                if not same:
                    target = out / f"RUN_STATUS.failed-attempt-{len(old_attempts)+1:02d}.json"
                    shutil.copy2(run_status, target)
                gpu = current.get("physical_gpu")
                if gpu not in (None, "", "unknown"):
                    log_path = ROOT / "qa/FAILED_ATTEMPTS.jsonl"
                    seen = False
                    if log_path.exists():
                        for line in log_path.read_text(encoding="utf-8").splitlines():
                            try:
                                item = json.loads(line)
                            except Exception:
                                continue
                            if item.get("job_id") == job_id and item.get("physical_gpu") == int(gpu) and item.get("error") == current.get("error"):
                                seen = True
                                break
                    if not seen:
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "attempt": f"{job_id}/retry_before_selection", "phase": "screening",
                                "job_id": job_id, "status": "failed_before_first_optimizer_update",
                                "error": current.get("error"), "physical_gpu": int(gpu),
                                "optimizer_updates_completed": 0, "watcher_lease_released": True,
                                "final_heldout_payload_access": 0, "dev_targets_read": False,
                                "utc": current.get("failed_utc", "unknown"),
                            }, ensure_ascii=False, sort_keys=True) + "\n")
        for source, pattern in (
            (watcher_status, f"{job_id}.attempt_*.json"),
            (child_log, f"{job_id}.attempt_*.log"),
        ):
            if source.exists():
                existing = sorted(source.parent.glob(pattern))
                same = False
                for old in existing:
                    try:
                        if old.read_bytes() == source.read_bytes():
                            same = True
                            break
                    except Exception:
                        continue
                if not same:
                    target = source.with_name(f"{job_id}.attempt_{len(existing)+1:02d}{source.suffix}")
                    if not target.exists():
                        shutil.copy2(source, target)
    if not incomplete:
        return 0
    retry_path = ROOT / "coordination/queues/SCREENING_RETRY_AFTER_GATE.json"
    retry_queue = {
        "schema_version": 1,
        "phase": "screening_retry_after_gate",
        "jobs": sorted(str(path) for path in incomplete),
        "fit_count": len(incomplete),
        "reason": "retry incomplete pre-registered screening fit slots once after primary queue, preserving effective-batch and LR/epoch contract",
    }
    content = json.dumps(retry_queue, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if retry_path.exists() and retry_path.read_text(encoding="utf-8") != content:
        raise RuntimeError("existing screening retry queue differs; preserving it")
    if not retry_path.exists():
        retry_path.write_text(content, encoding="utf-8")
    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    python = str(resolved["python_bin"])
    runner = ROOT / "code/run_v2_queue.py"
    subprocess.run([python, str(runner), "--queue", str(retry_path)], cwd=resolved["project_root"], check=False)
    return len(incomplete)


def append_failure_records(failures: list[dict[str, Any]]) -> None:
    if not failures:
        return
    path = ROOT / "qa/FAILED_ATTEMPTS.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    lock_path = ROOT / "protocol/LR_EPOCH_SELECTION.json"
    if lock_path.exists():
        raise FileExistsError(f"selection lock already exists; do not overwrite: {lock_path}")
    configs = read_fit_configs()
    if len(configs) != 36:
        raise RuntimeError(f"screening queue must contain 36 jobs, got {len(configs)}")
    retried_count = retry_incomplete_screening_fits()
    if retried_count:
        print(json.dumps({"event": "screening_retry_after_primary_queue", "jobs": retried_count}, sort_keys=True), flush=True)
        configs = read_fit_configs()
    candidates: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    for fit, out_dir in configs:
        complete_path = out_dir / "RUN_COMPLETE.json"
        manifest_path = out_dir / "MODEL_MANIFEST.json"
        job_record = {
            "job_id": fit["job_id"], "model_id": fit["model_id"], "lr": fit["lr"],
            "seed": fit["seed"], "output_root": str(out_dir),
        }
        if not complete_path.is_file() or not manifest_path.is_file():
            status = json.loads((out_dir / "RUN_STATUS.json").read_text()) if (out_dir / "RUN_STATUS.json").is_file() else {}
            job_record.update({"status": "FAILED_OR_INCOMPLETE", "error": status.get("error", "no RUN_COMPLETE/MODEL_MANIFEST")})
            job_rows.append(job_record)
            failures.append({
                "attempt": "screening_fit", "job_id": fit["job_id"], "phase": "screening",
                "status": "fit_failed_or_incomplete", "error": job_record["error"],
                "gpu_started": bool(status.get("physical_gpu", "unknown") not in {"unknown", None}),
                "final_heldout_payload_access": 0,
                "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            })
            continue
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if complete.get("status") != "PASS" or manifest.get("status") != "PASS":
            job_record.update({"status": "FAILED_OR_INCOMPLETE", "error": "completion/manifest status not PASS"})
            job_rows.append(job_record)
            continue
        job_record.update({
            "status": "PASS", "epochs_completed": complete["epochs_completed"],
            "optimizer_steps": complete["optimizer_steps"], "elapsed_seconds": complete["elapsed_seconds"],
            "physical_gpu": manifest["physical_gpu"], "peak_memory_reserved_mib": manifest["peak_memory_reserved_mib"],
        })
        job_rows.append(job_record)
        curve_path = out_dir / "EPOCH_LOG.csv"
        for row in read_csv(curve_path):
            curves.append({"job_id": fit["job_id"], "model_id": fit["model_id"], "lr": fit["lr"], "seed": fit["seed"], **row})
        metric_path = out_dir / "TUNE_CHECKPOINT_METRICS.csv"
        if not metric_path.is_file():
            raise RuntimeError(f"accepted screening fit lacks checkpoint metrics: {fit['job_id']}")
        for row in read_csv(metric_path):
            candidates.append({
                "job_id": fit["job_id"], "model_id": fit["model_id"], "scope": fit["scope"],
                "task_id": fit.get("task_id") if fit.get("task_id") is not None else "all",
                "video_mode": fit["video_mode"], "lr": float(fit["lr"]),
                "checkpoint_epoch": int(row["checkpoint_epoch"]), "metric_scope": row["metric_scope"],
                "metric_task_id": row["task_id"], "position_mae": row["position_mae"],
                "safe_fd_l1": row["safe_fd_l1"], "seam_safe_fd_l1": row.get("seam_safe_fd_l1", ""),
                "native_mae": row["native_mae"], "j_tune": row["j_tune"],
                "tune_episodes": row["tune_episodes"], "selected_for_model": False,
                "fit_status": "PASS", "model_manifest_sha256": sha_file(manifest_path),
            })

    selected_rows: list[dict[str, Any]] = []
    model_ids = ["G-S", "G-V"] + [f"S-{mode}-{task}" for task in TASK_IDS for mode in ("S", "V")]
    for model_id in model_ids:
        metric_scope = "macro" if model_id in {"G-S", "G-V"} else "selection"
        valid = []
        for row in candidates:
            if row["model_id"] != model_id or row["metric_scope"] != metric_scope:
                continue
            try:
                j_value = float(row["j_tune"])
                if math.isfinite(j_value):
                    valid.append(row)
            except (TypeError, ValueError):
                continue
        if not valid:
            continue
        best = min(float(row["j_tune"]) for row in valid)
        tied = [row for row in valid if abs(float(row["j_tune"]) - best) <= 1e-12]
        chosen = min(tied, key=lambda row: (
            int(row["checkpoint_epoch"]),
            LR_PRIORITY[float(row["lr"])],
        ))
        chosen["selected_for_model"] = True
        selected_rows.append({
            "model_id": model_id,
            "selected_lr": float(chosen["lr"]),
            "selected_epoch": int(chosen["checkpoint_epoch"]),
            "selected_j_tune": float(chosen["j_tune"]),
            "selected_position_mae": float(chosen["position_mae"]),
            "selected_safe_fd_l1": float(chosen["safe_fd_l1"]),
            "selected_seam_safe_fd_l1": float(chosen["seam_safe_fd_l1"]),
            "selected_native_mae": float(chosen["native_mae"]),
            "selected_tune_episodes": chosen["tune_episodes"],
            "selection_metric_scope": metric_scope,
            "valid_lr_epoch_candidates": len(valid),
            "planned_lr_epoch_candidates": 9,
            "search_complete": len(valid) == 9,
            "tie_rule": "J within 1e-12: shorter epoch, then LR 1e-3, 5e-4, 2e-3",
            "screening_seed": 202610130,
        })
    if len(selected_rows) != 12:
        status = "PARTIAL_NO_CHOICE_FOR_ONE_OR_MORE_MODELS"
    else:
        status = "LOCKED" if len(job_rows) == 36 and all(row["status"] == "PASS" for row in job_rows) and all(row["search_complete"] for row in selected_rows) else "LOCKED_WITH_PARTIAL_SEARCH"

    candidate_fields = [
        "job_id", "model_id", "scope", "task_id", "video_mode", "lr", "checkpoint_epoch",
        "metric_scope", "metric_task_id", "position_mae", "safe_fd_l1", "seam_safe_fd_l1",
        "native_mae", "j_tune", "tune_episodes", "selected_for_model", "fit_status",
        "model_manifest_sha256",
    ]
    write_csv(ROOT / "screening/ALL_CANDIDATES.csv", candidate_fields, candidates)
    curve_fields = ["job_id", "model_id", "lr", "seed", "epoch", "optimizer_steps", "lr_after_epoch",
                    "train_pos", "train_fd", "train_seam_safe_fd", "train_goal", "train_total",
                    "tune_j", "tune_pos_macro", "tune_fd_macro", "elapsed_seconds"]
    write_csv(ROOT / "screening/TRAIN_TUNE_CURVES.csv", curve_fields, curves)
    write_csv(ROOT / "screening/FIT_ACCEPTANCE.csv",
              ["job_id", "model_id", "lr", "seed", "status", "epochs_completed", "optimizer_steps",
               "elapsed_seconds", "physical_gpu", "peak_memory_reserved_mib", "error"], job_rows)
    append_failure_records(failures)
    lock = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "selection_source": "inner_tune only; canonical accepted primary-train metric scales",
        "selection_metric": "episode-equal Full-28 normalized position MAE + 0.1 future-internal 29-difference safe-FD L1",
        "excluded_from_selection": ["exposed_dev", "final_heldout"],
        "tie_rule": "J within 1e-12: shorter epoch, then LR 1e-3, 5e-4, 2e-3",
        "screening_seed": 202610130,
        "lr_candidates": [5e-4, 1e-3, 2e-3],
        "checkpoint_epochs": [40, 80, 160],
        "planned_fits": 36,
        "accepted_fits": sum(row["status"] == "PASS" for row in job_rows),
        "failed_or_incomplete_fits": sum(row["status"] != "PASS" for row in job_rows),
        "logical_model_choices": selected_rows,
        "missing_model_choices": [model for model in model_ids if model not in {row["model_id"] for row in selected_rows}],
        "candidates_csv_sha256": sha_file(ROOT / "screening/ALL_CANDIDATES.csv"),
        "curves_csv_sha256": sha_file(ROOT / "screening/TRAIN_TUNE_CURVES.csv"),
        "fit_acceptance_csv_sha256": sha_file(ROOT / "screening/FIT_ACCEPTANCE.csv"),
        "failed_attempts_appended": len(failures),
        "retry_jobs_attempted_once_after_primary_queue": retried_count,
        "final_heldout_payload_access": 0,
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "accepted_fits": lock["accepted_fits"],
                      "failed_fits": lock["failed_or_incomplete_fits"],
                      "choices": len(selected_rows), "selection_lock": str(lock_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
