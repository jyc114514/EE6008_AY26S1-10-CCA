from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V8 = Path(__file__).resolve().parents[1]
PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
REPLAY = V8 / "protocol/PREDICTION_REUSE_AND_REPLAY.json"
PREFLIGHT = V8 / "coordination/PREFLIGHT_COMPLETE.json"
GATE = V8 / "coordination/GPU_RESOURCE_GATE.json"
STATUS_ROOT = V8 / "coordination/status"
LEASE_ROOT = V8 / "coordination/gpu_lease"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_completion(job: dict, output_root: Path) -> dict:
    marker = output_root / "PREDICTION_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"watcher returned without a completion marker: {marker}")
    receipt = read(marker)
    if (
        receipt.get("status") != "PASS"
        or receipt.get("job_id") != job["job_id"]
        or receipt.get("program_id") != job["program"]
        or int(receipt.get("seed", -1)) != int(job["seed"])
        or int(receipt.get("receiver_rows", -1)) != 1280
        or receipt.get("prediction_shape") != [1280,30,28]
        or receipt.get("prediction_dtype") != "float64"
        or receipt.get("optimizer_updates") != 0
        or receipt.get("future_target_values_read") != 0
        or receipt.get("dev_target_values_read") != 0
        or receipt.get("final_heldout_payload_access") != 0
    ):
        raise RuntimeError(f"frozen forward completion receipt failed: {marker}")
    pred = Path(receipt["prediction_path"])
    rows = Path(receipt["prediction_rows_path"])
    if sha(pred) != receipt["prediction_sha256"] or sha(rows) != receipt["prediction_rows_sha256"]:
        raise RuntimeError(f"frozen forward output digest mismatch: {output_root}")
    return {
        "completion_path": str(marker), "completion_sha256": sha(marker),
        "prediction_path": str(pred), "prediction_sha256": receipt["prediction_sha256"],
        "prediction_rows_path": str(rows), "prediction_rows_sha256": receipt["prediction_rows_sha256"],
        "physical_gpu": receipt["physical_gpu"], "logical_device": receipt["logical_device"],
        "peak_memory_allocated_mib": receipt["peak_memory_allocated_mib"],
        "peak_memory_reserved_mib": receipt["peak_memory_reserved_mib"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "model_state_hash_before": receipt["model_state_hash_before"],
        "model_state_hash_after": receipt["model_state_hash_after"],
        "ordered_receiver_ids_sha256": receipt["ordered_receiver_ids_sha256"],
    }


def main() -> int:
    pre = read(PREFLIGHT)
    gate = read(GATE)
    replay = read(REPLAY)
    if pre.get("status") != "PASS" or gate.get("status") != "PASS":
        raise RuntimeError("preflight or host resource gate is not PASS")
    if int(replay.get("planned_frozen_student_train_forward_jobs", -1)) != 6:
        raise RuntimeError("replay lock does not authorize exactly six frozen train forwards")
    jobs = replay["train_forward_jobs"]
    if len(jobs) != 6:
        raise RuntimeError("replay lock job count mismatch")
    if replay.get("new_teacher_forward_jobs") != 0 or replay.get("new_dev_prediction_jobs") != 0:
        raise RuntimeError("replay lock violates zero-teacher / zero-dev-prediction boundary")

    begun = utc_now()
    execution = {
        "schema_version":1, "status":"RUNNING", "started_utc":begun,
        "host":os.environ.get("EE6008_HOST_LABEL", "shared-host"), "working_directory":str(PROJECT),
        "runner_path":str(Path(__file__).resolve()),
        "jobs_expected":6, "jobs_accepted":0, "jobs_failed":0,
        "new_fit_count":0, "new_optimizer_updates":0, "new_teacher_forward_jobs":0,
        "new_dev_prediction_jobs":0, "encoder_extraction":0,
        "final_heldout_payload_access":0, "old_root_writes":0,
        "jobs":[]
    }
    write(V8/"coordination/FROZEN_FORWARD_EXECUTION.json", execution)
    replay["status"] = "RUNNING"
    replay["execution_started_utc"] = begun
    write(REPLAY, replay)

    for item in jobs:
        job_id = item["job_id"]
        watch_config = Path(item["watcher_config_path"])
        output_root = Path(item["output_root"])
        status_path = Path(item["status_path"])
        console_path = V8/"logs"/"console"/f"{job_id}.txt"
        if output_root.exists():
            if (output_root/"PREDICTION_COMPLETE.json").exists():
                raise FileExistsError(f"refusing to overwrite existing completion: {output_root}")
            if any(output_root.iterdir()):
                raise FileExistsError(f"refusing to overwrite partial prior output: {output_root}")
        command = [
            str(PYTHON), str(WATCHER), "--job-json", str(watch_config),
            "--lease-dir", str(LEASE_ROOT), "--status-path", str(status_path),
        ]
        env = os.environ.copy()
        env.update({
            "MKL_NUM_THREADS":"4", "OMP_NUM_THREADS":"4",
            "OPENBLAS_NUM_THREADS":"4", "PYTHONDONTWRITEBYTECODE":"1",
            "PYTHONUNBUFFERED":"1", "TMPDIR":str(V8/"tmp"),
            "TMP":str(V8/"tmp"), "TEMP":str(V8/"tmp"),
        })
        started = utc_now()
        t0 = time.perf_counter()
        proc = subprocess.Popen(command, cwd=str(PROJECT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output, _ = proc.communicate()
        elapsed = time.perf_counter()-t0
        console_path.parent.mkdir(parents=True,exist_ok=True)
        console_path.write_text(output,encoding="utf-8")
        watcher_status = read(status_path) if status_path.is_file() else {}
        result = {
            "job_id":job_id, "program":item["program"], "seed":item["seed"],
            "started_utc":started, "completed_utc":utc_now(),
            "watcher_pid":proc.pid, "command":command, "cwd":str(PROJECT),
            "exit_code":proc.returncode, "wall_seconds":elapsed,
            "watcher_status_path":str(status_path), "watcher_status":watcher_status,
            "console_path":str(console_path),
            "output_root":str(output_root),
        }
        if proc.returncode == 0:
            try:
                result["accepted_outputs"] = validate_completion(item,output_root)
                result["status"]="PASS"
                execution["jobs_accepted"]+=1
            except Exception as exc:
                result["status"]="FAILED_ACCEPTANCE"
                result["acceptance_error"]=f"{type(exc).__name__}: {exc}"
                execution["jobs_failed"]+=1
        else:
            result["status"]="FAILED"
            execution["jobs_failed"]+=1
        execution["jobs"].append(result)
        execution["updated_utc"]=utc_now()
        write(V8/"coordination/FROZEN_FORWARD_EXECUTION.json",execution)
        if result["status"]!="PASS":
            execution["status"]="PARTIAL"
            execution["completed_utc"]=utc_now()
            write(V8/"coordination/FROZEN_FORWARD_EXECUTION.json",execution)
            replay["status"]="PARTIAL"
            replay["execution_result_path"]=str(V8/"coordination/FROZEN_FORWARD_EXECUTION.json")
            replay["execution_result_sha256"]=sha(V8/"coordination/FROZEN_FORWARD_EXECUTION.json")
            write(REPLAY,replay)
            print(json.dumps({"status":"PARTIAL","stopped_after_job":job_id,"exit_code":proc.returncode},sort_keys=True),flush=True)
            return 1

    execution["status"]="PASS"
    execution["completed_utc"]=utc_now()
    write(V8/"coordination/FROZEN_FORWARD_EXECUTION.json",execution)
    replay["status"]="PASS"
    replay["new_frozen_student_train_forward_jobs"]=6
    replay["accepted_train_forward_outputs"]=[
        j["accepted_outputs"]|{"program":j["program"],"seed":j["seed"],"job_id":j["job_id"]}
        for j in execution["jobs"]
    ]
    replay["execution_result_path"]=str(V8/"coordination/FROZEN_FORWARD_EXECUTION.json")
    replay["execution_result_sha256"]=sha(V8/"coordination/FROZEN_FORWARD_EXECUTION.json")
    replay["new_teacher_forward_jobs"]=0
    replay["new_dev_prediction_jobs"]=0
    replay["new_optimizer_updates"]=0
    write(REPLAY,replay)
    print(json.dumps({
        "status":"PASS","accepted_frozen_train_forward_jobs":execution["jobs_accepted"],
        "new_fit_count":0,"optimizer_updates":0,"new_dev_predictions":0,
        "physical_gpus":[j["accepted_outputs"]["physical_gpu"] for j in execution["jobs"]],
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
