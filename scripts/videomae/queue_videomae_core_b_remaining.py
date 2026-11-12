"""Prepare or execute eleven remaining formal cells serially via shared watcher."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--reference-job-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    project, root = args.project_root.resolve(), args.queue_root.resolve()
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(root != scratch and root.is_relative_to(scratch), "Personal queue directory required")
    watcher = project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
    lease = project / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"
    queue_path = root / "QUEUE.json"

    def watcher_command(job, dry=False):
        command = [sys.executable, str(watcher), "--job-json", str(job),
                   "--lease-dir", str(lease), "--status-path", str(job.parent / "watcher_status.json")]
        return command + (["--dry-run"] if dry else [])

    if not args.execute:
        require(args.reference_job_root is not None, "Reference job required for preparation")
        reference = args.reference_job_root.resolve()
        require(reference.is_relative_to(scratch), "Personal reference required")
        status = json.loads((reference / "watcher_status.json").read_text())
        complete = json.loads((reference / "run/RUN_STATUS.json").read_text())
        config = json.loads((reference / "run/CONFIG.json").read_text())
        require(status["status"] == "complete" and status["exit_code"] == 0 and status.get("lease_released_utc"), "Reference watcher not complete")
        require(complete["status"] == "PASS_TRAIN_AND_EXPORT" and complete["epochs"] == 40
                and complete["optimizer_steps"] == 11120, "Reference training incomplete")
        require((config["mode"], config["horizon"], config["seed"]) == ("TS11-D", 60, 20261001), "Reference cell mismatch")
        require(digest(reference / "run/checkpoint_last.pt") == complete["checkpoint_sha256"], "Reference checkpoint mismatch")
        original = json.loads((reference / "job.json").read_text())
        command = original["command"]
        require(command[1] == "python" and "--resume" not in command and "--resume-check" not in command, "Reference command mismatch")
        runner = Path(command[2])
        require(digest(runner) == config["runner_sha256"] == original["runner_sha256"], "Runner changed")
        root.mkdir(parents=True, exist_ok=False)
        cells = []
        for mode in ("TS11-D", "STATE-CTRL"):
            for horizon in (30, 60):
                for seed in (20261001, 20261002, 20261003):
                    if (mode, horizon, seed) == ("TS11-D", 60, 20261001):
                        continue
                    name = f"{mode}_H{horizon}_S{seed}"
                    cell = root / name
                    cell.mkdir()
                    argv = list(command)
                    for flag, value in (("--mode", mode), ("--horizon", str(horizon)),
                                        ("--seed", str(seed)), ("--output-root", str(cell / "run"))):
                        argv[argv.index(flag) + 1] = value
                    job = {"job_id": f"videomae-{root.name}-{name}",
                           "protocol_id": "VIDEOMAE_CORE_B_CORRECTED_FORMAL_V1", "output_root": str(cell),
                           "command": argv, "resource_estimate": original["resource_estimate"],
                           "max_oom_requeues": 0, "runner_sha256": config["runner_sha256"]}
                    path = cell / "job.json"
                    write(path, job)
                    subprocess.run(watcher_command(path, dry=True), check=True)
                    cells.append({"mode": mode, "horizon": horizon, "seed": seed,
                                  "job": str(path), "job_sha256": digest(path)})
        write(queue_path, {"status": "PREPARED_DRY_RUN_ONLY", "cells": cells,
                          "runner": str(runner), "runner_sha256": config["runner_sha256"],
                          "reference_job_root": str(reference), "own_concurrency": 1,
                          "prepared_utc": datetime.now(timezone.utc).isoformat(), "final_test_access": 0})
        print(json.dumps({"status": "PREPARED_11_CELLS_DRY_RUN_ONLY", "queue": str(queue_path), "gpu_launched": False}))
        return
    queue = json.loads(queue_path.read_text())
    require(len(queue["cells"]) == 11 and digest(Path(queue["runner"])) == queue["runner_sha256"], "Queue runner mismatch")
    for number, entry in enumerate(queue["cells"], 1):
        path = Path(entry["job"]).resolve()
        require(path.is_relative_to(root) and digest(path) == entry["job_sha256"], "Queue job changed")
        require(digest(Path(queue["runner"])) == queue["runner_sha256"], "Runner changed during queue")
        complete_path = path.parent / "run/RUN_STATUS.json"
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if complete.get("status") == "PASS_TRAIN_AND_EXPORT":
                saved_config = json.loads((path.parent / "run/CONFIG.json").read_text())
                require((saved_config["mode"], saved_config["horizon"], saved_config["seed"]) ==
                        (entry["mode"], entry["horizon"], entry["seed"]), "Completed cell identity mismatch")
                require(saved_config["runner_sha256"] == queue["runner_sha256"], "Completed runner mismatch")
                for name, key in (("checkpoint_last.pt", "checkpoint_sha256"),
                                  ("predictions_train.npz", "train_prediction_sha256"),
                                  ("predictions_dev.npz", "dev_prediction_sha256")):
                    require(digest(path.parent / "run" / name) == complete[key], "Completed artifact hash mismatch")
                continue
        require(not (path.parent / "run").exists(), "Partial run found; stop for explicit resume handling")
        write(root / "QUEUE_STATUS.json", {"status": "running", "cell_number": number, "cell": entry})
        result = subprocess.run(watcher_command(path))
        status_path = path.parent / "watcher_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        complete = json.loads(complete_path.read_text()) if complete_path.exists() else {}
        if (result.returncode != 0 or status.get("status") != "complete" or status.get("exit_code") != 0
                or not status.get("lease_released_utc") or complete.get("status") != "PASS_TRAIN_AND_EXPORT"):
            write(root / "QUEUE_STATUS.json", {"status": "STOPPED_ON_FAILURE", "cell": entry,
                                               "watcher_status": status, "final_test_access": 0})
            raise RuntimeError("Cell failed or waiting watcher exited; queue stopped, do not blindly relaunch")
        print(json.dumps({"completed_cell": entry, "queue_position": number}), flush=True)
    write(root / "QUEUE_STATUS.json", {"status": "PASS_11_REMAINING_TRAIN_EXPORT", "cells": 11,
                                       "metrics_evaluated": False, "final_test_access": 0})
    print((root / "QUEUE_STATUS.json").read_text())


if __name__ == "__main__":
    main()
