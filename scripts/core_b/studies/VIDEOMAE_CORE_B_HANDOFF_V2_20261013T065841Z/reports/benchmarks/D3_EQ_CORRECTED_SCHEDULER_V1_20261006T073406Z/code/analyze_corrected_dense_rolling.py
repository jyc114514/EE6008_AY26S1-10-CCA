#!/usr/bin/env python3
"""CPU-only matched D3-EQ scheduler effects on the locked D4 dense rolling panel."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
D4_ROOT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
DENSE_ROOT = TASK_ROOT / "dense_extension"
OUTPUT_ROOT = DENSE_ROOT / "results"
LOCK_PATH = DENSE_ROOT / "DENSE_EXTENSION_PROTOCOL_LOCKED.json"
PREDICTION_ACCEPTANCE = DENSE_ROOT / "PREDICTION_ACCEPTANCE.json"
PRIMARY_PREDICTION_ACCEPTANCE = TASK_ROOT / "PREDICTION_ACCEPTANCE.json"
MATRIX_PATH = TASK_ROOT / "protocol/MATRIX_12.csv"
SEEDS = (20261001, 20261002, 20261003)
CONFIGS = ("TS11-D", "STATE-CTRL")
HORIZONS = (30, 60)
BODIES = ("Hand", "Arm", "Full28")
METRICS = ("U", "L_new", "FD_new")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260930


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{__import__('os').getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty result table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_d4_module():
    code_root = D4_ROOT / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return runpy.run_path(str(code_root / "analyze_d3_d4_dense.py"), run_name="d3_dense_readonly_module")


def load_predictions(lock: dict[str, Any], module: dict[str, Any], context: dict[str, Any]):
    accepted = read_json(PREDICTION_ACCEPTANCE)
    if (accepted.get("status") != "PASS" or int(accepted.get("accepted_bundles", -1)) != 24
            or accepted.get("final_test_access") != 0):
        raise AssertionError("target scoring blocked until target-blind dense prediction acceptance is PASS 24/24")
    expected_ids = context["dense_ids"]
    expected_ids_sha = hashlib.sha256(expected_ids.tobytes()).hexdigest()
    if accepted.get("dense_ids_sha256") != expected_ids_sha:
        raise AssertionError("accepted dense IDs differ from D4 rolling-analysis panel")
    rows = []
    with MATRIX_PATH.open(newline="", encoding="utf-8") as stream:
        matrix = list(csv.DictReader(stream))
    predictions: dict[tuple[str, str, int, int], np.ndarray] = {}
    for row in matrix:
        config, seed, horizon = row["config"], int(row["seed"]), int(row["horizon_steps"])
        cell = next(item for item in lock["cells"] if (item["config"], item["seed"], item["horizon_steps"]) == (config, seed, horizon))
        for role, run_id in (("old_as_run", cell["old_run_id"]), ("corrected_scheduler", cell["corrected_run_id"])):
            sidecar_path = DENSE_ROOT / "predictions" / role / f"{run_id}.json"
            prediction_path = sidecar_path.with_suffix(".npz")
            sidecar = read_json(sidecar_path)
            if (sidecar.get("status") != "PASS" or sidecar.get("role") != role
                    or sidecar.get("run_id") != run_id or sidecar.get("final_test_access") != 0
                    or sidecar.get("target_arrays_loaded") is not False
                    or sidecar.get("dense_extension_protocol_sha256") != sha_file(LOCK_PATH)
                    or sidecar.get("prediction_sha256") != sha_file(prediction_path)):
                raise AssertionError(f"prediction provenance failed before target scoring: {run_id}/{role}")
            with np.load(prediction_path, allow_pickle=False) as bundle:
                ids = bundle["dense_id"].astype("U96")
                value = bundle["predictions_native"].astype(np.float32, copy=False)
                key = (role, config, seed, horizon)
                if (key in predictions or not np.array_equal(ids, expected_ids)
                        or value.shape != (3718, horizon, 28) or not np.isfinite(value).all()):
                    raise AssertionError(f"prediction ID/shape/finiteness failure: {run_id}/{role}")
                for field in ("episode_index", "original_task_index", "anchor_frame"):
                    if not np.array_equal(bundle[field], context["data"][field]):
                        raise AssertionError(f"prediction metadata join failed for {field}: {run_id}/{role}")
                if not np.allclose(bundle["anchor_timestamp"], context["data"]["anchor_timestamp"], atol=1e-6, rtol=0):
                    raise AssertionError(f"prediction timestamp join failed: {run_id}/{role}")
                predictions[key] = value
                rows.append({"role": role, "config": config, "seed": seed, "horizon_steps": horizon,
                             "run_id": run_id, "checkpoint_sha256": sidecar["checkpoint_sha256"],
                             "prediction_sha256": sidecar["prediction_sha256"]})
    if len(predictions) != 24:
        raise AssertionError(f"expected 24 accepted prediction arrays, found {len(predictions)}")
    return predictions, rows


def main() -> int:
    lock = read_json(LOCK_PATH)
    if lock.get("status") != "LOCKED" or lock.get("final_test_access") != 0:
        raise AssertionError("dense protocol lock is invalid")
    if lock.get("analysis_source_sha256") != sha_file(Path(__file__).resolve()):
        raise AssertionError("dense analysis code changed after lock")
    if lock.get("parent_matrix_sha256") != sha_file(MATRIX_PATH):
        raise AssertionError("parent matrix changed after dense lock")
    if lock.get("parent_prediction_acceptance_sha256") != sha_file(PRIMARY_PREDICTION_ACCEPTANCE):
        raise AssertionError("primary train/dev prediction acceptance changed after dense lock")
    if lock.get("d4_analysis_source_sha256") != sha_file(D4_ROOT / "code/analyze_d3_d4_dense.py"):
        raise AssertionError("D4 rolling analysis source changed after dense lock")
    if OUTPUT_ROOT.exists() or (DENSE_ROOT / "DENSE_ROLLING_ANALYSIS_QA.json").exists():
        raise FileExistsError("dense analysis outputs already exist; refusing overwrite")
    module = load_d4_module()
    context = module["build_context"]()
    predictions, prediction_rows = load_predictions(lock, module, context)
    # Targets are first materialized by the D4 source-group join below, after the 24/24 target-blind gate.
    group_by_h = {
        h: sorted(set(context["pair_groups"][row["pair_id"]] for row in context["pairs_by_h"][h]))
        for h in HORIZONS
    }
    if any(len(group_by_h[h]) != 118 or len(context["pairs_by_h"][h]) != 3600 for h in HORIZONS):
        raise AssertionError("locked D4 dense pair/group support changed")
    bootstrap_by_h = {h: module["bootstrap_counts"](118, BOOTSTRAP_SEED) for h in HORIZONS}
    pair_groups_by_h = {
        h: np.asarray([context["pair_groups"][r["pair_id"]] for r in context["pairs_by_h"][h]], dtype="U64")
        for h in HORIZONS
    }
    values_by_cell: dict[tuple[str, int, int, str], dict[str, np.ndarray]] = {}
    group_values: dict[tuple[str, str, int, int, str, str], np.ndarray] = {}
    model_rows: list[dict[str, Any]] = []
    for role in ("old_as_run", "corrected_scheduler"):
        for config in CONFIGS:
            for seed in SEEDS:
                for horizon in HORIZONS:
                    pred = predictions[(role, config, seed, horizon)]
                    metrics = module["metric_arrays"](context, (config, seed, horizon), pred)
                    values_by_cell[(role, config, seed, horizon)] = metrics
                    pair_groups = pair_groups_by_h[horizon]
                    ordered = group_by_h[horizon]
                    for body in BODIES:
                        for metric in METRICS:
                            group_values[(role, config, seed, horizon, body, metric)] = module["group_means"](
                                metrics[f"{body}|{metric}"], pair_groups, ordered
                            )
                    for body in BODIES:
                        for metric in METRICS:
                            grouped = group_values[(role, config, seed, horizon, body, metric)]
                            estimate, low, high = module["summarize_groups"](grouped, bootstrap_by_h[horizon])
                            model_rows.append({
                                "checkpoint_role": role, "config": config, "seed": seed,
                                "horizon_steps": horizon, "body": body, "metric": metric,
                                "estimate_source_group_equal": estimate, "ci95_group_bootstrap_low": low,
                                "ci95_group_bootstrap_high": high, "pairs": 3600,
                                "source_groups": 118, "bootstrap_replicates": BOOTSTRAPS,
                                "bootstrap_seed": BOOTSTRAP_SEED, "final_test_access": 0,
                            })
    schedule_by_seed: list[dict[str, Any]] = []
    group_schedule_rows: list[dict[str, Any]] = []
    schedule_groups: dict[tuple[str, int, int, str, str], np.ndarray] = {}
    for config in CONFIGS:
        for seed in SEEDS:
            for horizon in HORIZONS:
                for body in BODIES:
                    for metric in METRICS:
                        effect = (group_values[("corrected_scheduler", config, seed, horizon, body, metric)]
                                  - group_values[("old_as_run", config, seed, horizon, body, metric)])
                        schedule_groups[(config, seed, horizon, body, metric)] = effect
                        estimate, low, high = module["summarize_groups"](effect, bootstrap_by_h[horizon])
                        schedule_by_seed.append({
                            "config": config, "seed": seed, "horizon_steps": horizon, "body": body,
                            "metric": metric, "effect_corrected_minus_old": estimate,
                            "ci95_group_bootstrap_low": low, "ci95_group_bootstrap_high": high,
                            "pairs": 3600, "source_groups": 118, "bootstrap_replicates": BOOTSTRAPS,
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "positive_U_favors_larger_update_gain": metric == "U",
                            "negative_L_new_or_FD_new_favors_lower_error": metric != "U",
                            "final_test_access": 0,
                        })
                        for group, value in zip(group_by_h[horizon], effect, strict=True):
                            group_schedule_rows.append({"config": config, "seed": seed, "horizon_steps": horizon,
                                                        "body": body, "metric": metric,
                                                        "source_group_id": group, "effect_corrected_minus_old": float(value),
                                                        "final_test_access": 0})
    fixed3_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    group_video_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        groups = group_by_h[horizon]
        counts = bootstrap_by_h[horizon]
        for config in CONFIGS:
            for body in BODIES:
                for metric in METRICS:
                    per_seed = np.vstack([schedule_groups[(config, seed, horizon, body, metric)] for seed in SEEDS])
                    fixed_group = per_seed.mean(axis=0)
                    estimate, low, high = module["summarize_groups"](fixed_group, counts)
                    fixed3_rows.append({
                        "config": config, "horizon_steps": horizon, "body": body, "metric": metric,
                        "effect_equal_fixed_seed_then_source_group": estimate,
                        "ci95_source_group_bootstrap_low": low, "ci95_source_group_bootstrap_high": high,
                        "fixed_seeds": "20261001;20261002;20261003", "source_groups": 118,
                        "pairs_per_seed": 3600, "bootstrap_replicates": BOOTSTRAPS,
                        "bootstrap_seed": BOOTSTRAP_SEED, "cross_seed_uncertainty": "not represented by this interval",
                        "p_value": "not computed", "final_test_access": 0,
                    })
        for body in BODIES:
            for metric in METRICS:
                interaction_by_seed = np.vstack([
                    schedule_groups[("TS11-D", seed, horizon, body, metric)]
                    - schedule_groups[("STATE-CTRL", seed, horizon, body, metric)]
                    for seed in SEEDS
                ])
                fixed_group = interaction_by_seed.mean(axis=0)
                estimate, low, high = module["summarize_groups"](fixed_group, counts)
                video_rows.append({
                    "contrast": "TS11-D_scheduler_gain_minus_STATE-CTRL_scheduler_gain",
                    "horizon_steps": horizon, "body": body, "metric": metric,
                    "effect_equal_fixed_seed_then_source_group": estimate,
                    "ci95_source_group_bootstrap_low": low, "ci95_source_group_bootstrap_high": high,
                    "fixed_seeds": "20261001;20261002;20261003", "source_groups": 118,
                    "pairs_per_seed": 3600, "bootstrap_replicates": BOOTSTRAPS,
                    "bootstrap_seed": BOOTSTRAP_SEED, "p_value": "not computed",
                    "final_test_access": 0,
                })
                for group, value in zip(groups, fixed_group, strict=True):
                    group_video_rows.append({"horizon_steps": horizon, "body": body, "metric": metric,
                                             "source_group_id": group,
                                             "video_specific_effect_corrected_minus_old": float(value),
                                             "final_test_access": 0})
    write_csv(OUTPUT_ROOT / "DENSE_MODEL_METRICS_BY_SEED.csv", model_rows)
    write_csv(OUTPUT_ROOT / "DENSE_SCHEDULER_GAINS_BY_SEED.csv", schedule_by_seed)
    write_csv(OUTPUT_ROOT / "DENSE_GROUP_LEVEL_SCHEDULER_GAINS.csv", group_schedule_rows)
    write_csv(OUTPUT_ROOT / "DENSE_SCHEDULER_GAINS_FIXED3.csv", fixed3_rows)
    write_csv(OUTPUT_ROOT / "DENSE_VIDEO_SPECIFIC_GAINS_FIXED3.csv", video_rows)
    write_csv(OUTPUT_ROOT / "DENSE_GROUP_LEVEL_VIDEO_SPECIFIC_GAINS.csv", group_video_rows)
    qa = {
        "status": "PASS", "study_id": "D3_EQ_CORRECTED_SCHEDULER_DENSE_SECONDARY_V1",
        "dense_predictions": "24/24 exact old_as_run and corrected_scheduler matched checkpoints",
        "dense_queries": 3718, "episodes": 118, "source_groups": 118,
        "pairs_h30": 3600, "pairs_h60": 3600,
        "common_absolute_frame_timestamp_join": "reused locked D4 join; validated by source analysis module",
        "target_scoring_started_after_prediction_acceptance": "PASS 24/24",
        "primary_estimand": "corrected scheduler minus exact matched old as-run checkpoint on D4 rolling metrics",
        "video_specific_estimand": "TS11-D scheduler effect minus STATE-CTRL scheduler effect",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "118 source groups; the fixed three model seeds stay paired within each group",
        "p_values": "not computed; secondary development diagnostic",
        "independent_confirmation": False, "training": "none", "inference_device": "cpu",
        "final_test_access": 0, "heldout_access": 0,
        "input_manifest_sha256": lock["dense_input_sha256"],
        "feature_cache_manifest_sha256": lock["feature_cache_manifest_sha256"],
        "d4_analysis_source_sha256": lock["d4_analysis_source_sha256"],
        "d4_metric_source_sha256": lock["d4_metric_source_sha256"],
        "prediction_acceptance_sha256": sha_file(PREDICTION_ACCEPTANCE),
        "prediction_bundles": prediction_rows,
    }
    write_json(DENSE_ROOT / "DENSE_ROLLING_ANALYSIS_QA.json", qa)
    print(json.dumps({"status": "PASS", "scheduler_effect_rows": len(schedule_by_seed),
                      "fixed3_rows": len(fixed3_rows), "video_specific_rows": len(video_rows),
                      "source_groups": 118, "pairs_per_horizon": 3600, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
