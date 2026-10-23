"""Prediction-gated acceptance and independent metrics for Core-B B1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LANES = (
    "B-STATE",
    "B-D-P-F",
    "B-S-P-F",
    "B-D-A-F",
    "B-S-A-F",
    "B-D-FUS-P-F",
    "B-S-FUS-P-F",
)
SEED = 20260903
DEV_COUNT = 614
STEPS = 30
DIMENSIONS = 28
ARM = slice(14, 28)
HAND = slice(0, 14)
PHYSICAL_BATCH = 8
ACCUMULATION = 4
EFFECTIVE_BATCH = 32
MIN_EPOCHS = 5
PATIENCE = 5
MIN_DELTA = 0.0001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def check_marker(run_dir: Path) -> None:
    marker = load_json(run_dir / "RUN_COMPLETE.json")
    if marker.get("status") != "complete" or marker.get("exit_code") != 0:
        raise ValueError(f"completion marker failed: {run_dir}")
    if marker.get("final_test_accessed", 0) != 0:
        raise ValueError(f"final-test access marker: {run_dir}")
    hashes = marker.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise TypeError(f"artifact hashes missing: {run_dir}")
    for name, expected in hashes.items():
        path = run_dir / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")


def replay(run_dir: Path, metrics: dict[str, Any]) -> dict[str, object]:
    with (run_dir / "epoch_curves.csv").open(encoding="utf-8", newline="") as handle:
        curves = list(csv.DictReader(handle))
    if not curves or len(curves) > 20:
        raise ValueError(f"invalid curve count: {run_dir}")
    if [int(row["epoch"]) for row in curves] != list(range(1, len(curves) + 1)):
        raise ValueError(f"non-contiguous epochs: {run_dir}")
    train_count = int(load_json(run_dir / "RUN_CONFIG_LOCKED.json")["train_count"])
    expected_steps = math.ceil(math.ceil(train_count / PHYSICAL_BATCH) / ACCUMULATION)
    if any(int(row["optimizer_steps"]) != expected_steps for row in curves):
        raise ValueError(f"optimizer-step replay failed: {run_dir}")
    if any(row["partial_final_cycle"] != "True" for row in curves):
        raise ValueError(f"partial-cycle evidence missing: {run_dir}")
    best_epoch = None
    best_value = math.inf
    bad = 0
    stop_epoch = None
    for row in curves:
        epoch = int(row["epoch"])
        value = float(row["validation_standardized_mae"])
        improved = value < best_value - MIN_DELTA
        if improved:
            best_epoch, best_value, bad = epoch, value, 0
        else:
            bad += 1
        if epoch >= MIN_EPOCHS and bad >= PATIENCE:
            stop_epoch = epoch
            break
    expected_last = stop_epoch if stop_epoch is not None else len(curves)
    if int(metrics["best_epoch"]) != int(best_epoch) or int(metrics["last_epoch"]) != expected_last:
        raise ValueError(f"early stopping replay failed: {run_dir}")
    return {
        "curve_rows": len(curves),
        "replayed_best_epoch": int(best_epoch),
        "replayed_last_epoch": expected_last,
        "early_stopping_replay": "pass",
    }


def preflight(run_dir: Path, lane: str) -> dict[str, object]:
    required = (
        "RUN_CONFIG_LOCKED.json",
        "MODEL_CONTRACT.json",
        "RUN_STATUS.json",
        "RUN_METRICS.json",
        "RUN_COMPLETE.json",
        "checkpoint_best.pt",
        "checkpoint_last.pt",
        "predictions_best.npz",
        "predictions_last.npz",
        "epoch_curves.csv",
        "epoch_curves.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: {missing}")
    check_marker(run_dir)
    config = load_json(run_dir / "RUN_CONFIG_LOCKED.json")
    status = load_json(run_dir / "RUN_STATUS.json")
    metrics = load_json(run_dir / "RUN_METRICS.json")
    if config.get("cell") != lane or config.get("seed") != SEED:
        raise ValueError(f"lane/seed identity failed: {run_dir}")
    if config.get("physical_gpu") != 3:
        raise ValueError(f"Core-B GPU role failed: {run_dir}")
    if config.get("physical_batch") != PHYSICAL_BATCH or config.get("gradient_accumulation") != ACCUMULATION or config.get("nominal_effective_batch") != EFFECTIVE_BATCH:
        raise ValueError(f"batch contract failed: {run_dir}")
    if config.get("final_test_accessed") != 0 or config.get("target_physical_unit_status") != "unresolved":
        raise ValueError(f"target/final-test contract failed: {run_dir}")
    if config.get("task_labels_used_as_model_input") is not False:
        raise ValueError(f"task-label input contract failed: {run_dir}")
    if config.get("activation_checkpointing_or_offload_used") is not False:
        raise ValueError(f"offload/checkpointing contract failed: {run_dir}")
    if status.get("status") != "complete" or status.get("exit_code") != 0:
        raise ValueError(f"run status incomplete: {run_dir}")
    if metrics.get("status") != "complete" or metrics.get("final_test_accessed") != 0:
        raise ValueError(f"run metrics incomplete: {run_dir}")
    from checkpoint_utils import load_checkpoint

    for checkpoint_name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        payload = load_checkpoint(run_dir / checkpoint_name, map_location="cpu")
        identity = payload["run_identity"]
        if identity.get("cell") != lane or identity.get("seed") != SEED:
            raise ValueError(f"checkpoint identity failed: {run_dir}/{checkpoint_name}")
        if payload["final_test_count"] != 0:
            raise ValueError(f"checkpoint final-test contract failed: {run_dir}")
    return {
        "lane": lane,
        "run_dir": str(run_dir),
        "physical_gpu": 3,
        "preflight": "pass",
        "predictions_opened": False,
    } | replay(run_dir, metrics)


def open_prediction(run_dir: Path, name: str) -> dict[str, np.ndarray]:
    allowed = {
        "episode_index",
        "original_task_index",
        "targets_standardized",
        "predictions_standardized",
        "targets_native",
        "predictions_native",
        "last_observed_action",
    }
    with np.load(run_dir / name, allow_pickle=False) as archive:
        if set(archive.files) != allowed:
            raise ValueError(f"prediction schema failed: {run_dir}/{name}")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if arrays["episode_index"].shape != (DEV_COUNT,) or arrays["original_task_index"].shape != (DEV_COUNT,):
        raise ValueError(f"prediction ID shape failed: {run_dir}/{name}")
    for key in ("targets_standardized", "predictions_standardized", "targets_native", "predictions_native"):
        if arrays[key].shape != (DEV_COUNT, STEPS, DIMENSIONS):
            raise ValueError(f"prediction tensor shape failed: {run_dir}/{name}/{key}")
    if arrays["last_observed_action"].shape != (DEV_COUNT, DIMENSIONS):
        raise ValueError(f"persistence shape failed: {run_dir}/{name}")
    if len(np.unique(arrays["episode_index"])) != DEV_COUNT:
        raise ValueError(f"duplicate dev IDs: {run_dir}/{name}")
    if not all(np.isfinite(arrays[key]).all() for key in allowed if key != "episode_index" and key != "original_task_index"):
        raise ValueError(f"non-finite predictions: {run_dir}/{name}")
    return arrays


def metric_row(
    lane: str,
    checkpoint: str,
    arrays: dict[str, np.ndarray],
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, object]:
    error_std = arrays["predictions_standardized"] - arrays["targets_standardized"]
    error_native = arrays["predictions_native"] - arrays["targets_native"]
    persistence_native = np.repeat(
        arrays["last_observed_action"][:, None, :], STEPS, axis=1
    )
    persistence_std = (persistence_native - target_mean) / target_std
    return {
        "lane": lane,
        "checkpoint": checkpoint,
        "physical_gpu": 3,
        "standardized_mae": float(np.abs(error_std).mean()),
        "standardized_rmse": float(np.sqrt(np.mean(error_std**2))),
        "standardized_arm_mae": float(np.abs(error_std[:, :, ARM]).mean()),
        "standardized_hand_mae": float(np.abs(error_std[:, :, HAND]).mean()),
        "standardized_final_step_mae": float(np.abs(error_std[:, -1, :]).mean()),
        "native_command_value_mae": float(np.abs(error_native).mean()),
        "persistence_standardized_mae": float(
            np.abs(persistence_std - arrays["targets_standardized"]).mean()
        ),
        "persistence_native_command_value_mae": float(
            np.abs(persistence_native - arrays["targets_native"]).mean()
        ),
        "final_test_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-b-root", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    retry_root = args.retry_root.resolve()
    manifest = load_json(args.manifest.resolve())
    jobs = {str(job["lane"]): job for job in manifest["jobs"]}
    if set(jobs) != set(LANES):
        raise ValueError("B1 lane set mismatch")

    # Phase 1: no prediction arrays are opened until every lane passes.
    rows = []
    for lane in LANES:
        rows.append(preflight(Path(str(jobs[lane]["output_root"])).resolve(), lane))

    target_norm_all = load_json(
        args.feasibility_root.resolve() / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
    )
    target_norm = target_norm_all["1.0s"]
    target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
    target_std = np.asarray(target_norm["safe_std"], dtype=np.float32)
    metric_rows = []
    for row in rows:
        run_dir = Path(str(row["run_dir"]))
        best = open_prediction(run_dir, "predictions_best.npz")
        last = open_prediction(run_dir, "predictions_last.npz")
        for arrays in (best, last):
            if not np.allclose(
                arrays["predictions_native"],
                arrays["predictions_standardized"] * target_std + target_mean,
                atol=2e-5,
                rtol=2e-5,
            ):
                raise ValueError(f"prediction inverse-transform mismatch: {run_dir}")
            if not np.allclose(
                arrays["targets_native"],
                arrays["targets_standardized"] * target_std + target_mean,
                atol=2e-5,
                rtol=2e-5,
            ):
                raise ValueError(f"target inverse-transform mismatch: {run_dir}")
        if not np.array_equal(best["episode_index"], last["episode_index"]) or not np.array_equal(best["original_task_index"], last["original_task_index"]):
            raise ValueError(f"best/last ID mismatch: {run_dir}")
        row["predictions_opened"] = True
        metric_rows.extend(
            [
                metric_row(row["lane"], "best", best, target_mean, target_std),
                metric_row(row["lane"], "last", last, target_mean, target_std),
            ]
        )
    output_root = args.output_root.resolve()
    for name in (
        "BASELINE_RUN_ACCEPTANCE.csv",
        "BASELINE_RUN_ACCEPTANCE.json",
        "BASELINE_METRICS.csv",
        "BASELINE_METRICS.json",
        "CORE_B_BASELINE_MATRIX.csv",
        "CORE_B_BASELINE_MATRIX.json",
    ):
        if (output_root / name).exists():
            raise FileExistsError(f"refusing existing B1 artifact: {output_root / name}")
    with (output_root / "BASELINE_RUN_ACCEPTANCE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    acceptance = {
        "status": "complete",
        "accepted": len(rows),
        "required": len(LANES),
        "rows": rows,
        "retry_root": str(retry_root),
        "prediction_gate": "all_7_preflight_checks_passed_before_npz_open",
        "final_test_accessed": 0,
    }
    (output_root / "BASELINE_RUN_ACCEPTANCE.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / "BASELINE_METRICS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (output_root / "BASELINE_METRICS.json").write_text(
        json.dumps({"status": "complete", "rows": metric_rows, "final_test_accessed": 0}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    best_rows = [row for row in metric_rows if row["checkpoint"] == "best"]
    best_rows.sort(key=lambda row: float(row["standardized_arm_mae"]))
    with (output_root / "CORE_B_BASELINE_MATRIX.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)
    (output_root / "CORE_B_BASELINE_MATRIX.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "ranking_metric": "best standardized arm MAE",
                "best_lane": best_rows[0]["lane"],
                "rows": best_rows,
                "final_test_accessed": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "accepted": len(rows), "required": len(LANES), "best_lane": best_rows[0]["lane"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
