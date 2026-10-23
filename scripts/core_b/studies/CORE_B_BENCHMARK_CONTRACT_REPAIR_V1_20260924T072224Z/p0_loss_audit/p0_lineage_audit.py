"""Read-only verification of old D3's 36 run/checkpoint/prediction lineage."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/p0_loss_audit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def status_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    matrix = list(csv.DictReader((BENCH / "protocol/D3_MATRIX_36.csv").open(encoding="utf-8", newline="")))
    expected_acceptance = status_record(BENCH / "d3/D3_PREDICTION_ACCEPTANCE.json")
    trainer = BENCH / "code/d3_train.py"
    trainer_sha = sha256_file(trainer)
    manifest_match = any(
        len(parts := line.split(maxsplit=1)) == 2 and parts[0] == trainer_sha and parts[1] == "code/d3_train.py"
        for line in (BENCH / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    )
    output_rows: list[dict[str, object]] = []
    init_by_seed: dict[str, set[str]] = defaultdict(set)

    for row in matrix:
        run_id = row["run_id"]
        config_path = BENCH / "configs" / f"{run_id}-FORMAL.json"
        cfg = status_record(config_path)
        command = list(cfg.get("command", []))
        try:
            entrypoint = Path(command[1]).resolve()
            run_root = Path(cfg["output_root"]).resolve()
        except (IndexError, KeyError, TypeError):
            entrypoint = Path()
            run_root = BENCH / "d3/runs" / f"{run_id}-FORMAL"
        complete = status_record(run_root / "RUN_COMPLETE.json")
        run_status = status_record(run_root / "RUN_STATUS.json")
        checkpoint = run_root / "checkpoint_last.pt"
        pred_stem = BENCH / "d3/predictions" / f"{run_id}-FORMAL"
        pred_npz = pred_stem.with_suffix(".npz")
        pred_sidecar = pred_stem.with_suffix(".json")
        if not pred_npz.is_file():
            retry_stem = BENCH / "d3/predictions" / f"{run_id}-FORMAL-RETRY1"
            if retry_stem.with_suffix(".npz").is_file():
                pred_stem = retry_stem
                pred_npz = retry_stem.with_suffix(".npz")
                pred_sidecar = retry_stem.with_suffix(".json")
        pred_meta = status_record(pred_sidecar)
        watcher = status_record(BENCH / "logs" / f"watcher_{run_id}-FORMAL.json")
        seed = str(row["seed"])
        init_hash = str(complete.get("init_hash", ""))
        if init_hash:
            init_by_seed[seed].add(init_hash)
        output_rows.append({
            "run_id": run_id,
            "config_sha256": sha256_file(config_path) if config_path.is_file() else "",
            "entrypoint": str(entrypoint) if entrypoint else "",
            "entrypoint_matches_locked_path": bool(entrypoint and entrypoint == trainer.resolve()),
            "entrypoint_current_sha256": trainer_sha,
            "current_hash_matches_bench_checksum": manifest_match,
            "execution_time_code_hash_recorded": False,
            "run_status": complete.get("status", run_status.get("status", "missing")),
            "epochs": complete.get("epochs"),
            "optimizer_steps": complete.get("optimizer_steps"),
            "seed_init_hash": init_hash,
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else "",
            "prediction_npz_exists": pred_npz.is_file(),
            "prediction_npz_sha256": sha256_file(pred_npz) if pred_npz.is_file() else "",
            "prediction_sidecar_status": pred_meta.get("status", "missing"),
            "physical_gpu": watcher.get("physical_gpu", "not_recorded"),
            "watcher_status": watcher.get("status", "missing"),
            "watcher_exit_code": watcher.get("exit_code"),
            "lease_released": watcher.get("lease_released"),
            "final_test_access": 0,
        })

    failures: list[dict[str, object]] = []
    error_re = re.compile(r"Traceback|RuntimeError|CUDA out of memory|AssertionError|ERROR:", re.IGNORECASE)
    for log_path in sorted((BENCH / "logs").glob("child_D3-*-FORMAL*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = [line.strip() for line in text.splitlines() if error_re.search(line)]
        if matches:
            failures.append({
                "attempt_id": log_path.stem.removeprefix("child_"),
                "log_path": str(log_path.relative_to(BENCH)),
                "error_excerpt": " | ".join(matches[:4]),
                "preserved_in_old_root": True,
            })
    for watcher_path in sorted((BENCH / "logs").glob("watcher_D3-*-FORMAL*.json")):
        value = status_record(watcher_path)
        if value and (value.get("status") != "complete" or value.get("exit_code") != 0):
            failures.append({
                "attempt_id": watcher_path.stem.removeprefix("watcher_"),
                "log_path": str(watcher_path.relative_to(BENCH)),
                "error_excerpt": f"watcher status={value.get('status')} exit_code={value.get('exit_code')}",
                "preserved_in_old_root": True,
            })

    summary = {
        "matrix_rows": len(matrix),
        "run_complete_pass": sum(row["run_status"] == "PASS" for row in output_rows),
        "last_checkpoints_present": sum(bool(row["checkpoint_exists"]) for row in output_rows),
        "prediction_npz_present": sum(bool(row["prediction_npz_exists"]) for row in output_rows),
        "prediction_sidecar_pass": sum(row["prediction_sidecar_status"] == "PASS" for row in output_rows),
        "prediction_acceptance": expected_acceptance,
        "all_rows_use_same_trainer_path": all(bool(row["entrypoint_matches_locked_path"]) for row in output_rows),
        "trainer_sha256_current": trainer_sha,
        "trainer_sha256_matches_sealed_checksum": manifest_match,
        "launch_time_trainer_hash_recorded_per_run": False,
        "initial_hashes_per_seed": {seed: len(values) for seed, values in sorted(init_by_seed.items())},
        "preserved_formal_failure_records": len(failures),
        "failure_records": failures,
        "final_test_access": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "P0_OLD_D3_RUN_LINEAGE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with (OUT / "P0_OLD_D3_FAILURE_RECORDS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["attempt_id", "log_path", "error_excerpt", "preserved_in_old_root"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    (OUT / "P0_OLD_D3_RUN_LINEAGE.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "matrix": summary["matrix_rows"],
        "run_complete_pass": summary["run_complete_pass"],
        "checkpoints": summary["last_checkpoints_present"],
        "predictions": summary["prediction_sidecar_pass"],
        "same_trainer_path": summary["all_rows_use_same_trainer_path"],
        "failure_records": summary["preserved_formal_failure_records"],
        "init_hashes_per_seed": summary["initial_hashes_per_seed"],
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
