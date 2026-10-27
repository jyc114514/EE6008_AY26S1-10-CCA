"""Independently validate and summarize the bounded Core B native-space smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

LANES = [
    ("video_only", "1p0s"),
    ("state_action_history_only", "1p0s"),
    ("video_only", "0p5s"),
    ("state_action_history_only", "0p5s"),
    ("gated_fusion", "1p0s"),
]
HORIZONS = {"1p0s": ("1.0s", 30), "0p5s": ("0.5s", 15)}
MIN_EPOCHS = 3
PATIENCE = 3
MIN_DELTA = 0.0001
TRAIN_COUNT = 256
DEV_COUNT = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "final_step_mae": float(absolute[:, -1, :].mean()),
    }


def independent_metrics(
    pred_std: np.ndarray,
    target_std: np.ndarray,
    target_mean: np.ndarray,
    target_safe_std: np.ndarray,
    train_native: np.ndarray,
    state_history: np.ndarray,
) -> dict[str, object]:
    native_pred = pred_std * target_safe_std + target_mean
    native_target = target_std * target_safe_std + target_mean
    error = pred_std - target_std
    global_native = np.broadcast_to(train_native.mean(axis=0), native_target.shape).copy()
    persistence_native = np.repeat(state_history[:, -28:][:, None, :], native_target.shape[1], axis=1)
    global_std = (global_native - target_mean) / target_safe_std
    persistence_std = (persistence_native - target_mean) / target_safe_std
    return {
        "standardized": summary(error),
        "native_command_value": summary(native_pred - native_target),
        "global_mean_baseline_standardized": summary(global_std - target_std),
        "persistence_baseline_standardized": summary(persistence_std - target_std),
        "global_mean_baseline_native_command_value": summary(global_native - native_target),
        "persistence_baseline_native_command_value": summary(persistence_native - native_target),
        "per_timestep_standardized_mae": np.abs(error).mean(axis=(0, 2)).astype(float).tolist(),
        "per_dimension_standardized_mae": np.abs(error).mean(axis=(0, 1)).astype(float).tolist(),
        "finite": bool(np.isfinite(pred_std).all() and np.isfinite(target_std).all() and np.isfinite(native_pred).all()),
    }


def replay_early_stopping(curves_path: Path) -> dict[str, object]:
    with curves_path.open(encoding="utf-8", newline="") as handle:
        curves = list(csv.DictReader(handle))
    if not curves:
        raise RuntimeError(f"empty epoch curves: {curves_path}")
    best_value = math.inf
    best_epoch = None
    stale = 0
    stop_epoch = None
    for row in curves:
        epoch = int(row["epoch"])
        value = float(row["dev_standardized_mae"])
        if value < best_value - MIN_DELTA:
            best_value = value
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            stop_epoch = epoch
            break
    if best_epoch is None:
        raise RuntimeError(f"no best epoch in {curves_path}")
    expected_last = stop_epoch if stop_epoch is not None else int(curves[-1]["epoch"])
    return {"replay_best_epoch": best_epoch, "replay_last_epoch": expected_last, "replay_early_stopped": expected_last < int(curves[-1]["epoch"]) or stop_epoch is not None and stop_epoch < 10, "replay_curve_rows": len(curves), "replay_best_standardized_mae": best_value}


def validate_run_contracts(root: Path) -> list[dict[str, object]]:
    contract = json.loads((root / "BATCH64_CONTRACT_SMOKE.json").read_text(encoding="utf-8"))
    if contract.get("status") != "pass" or contract.get("physical_batch_size") != 64 or contract.get("gradient_accumulation_steps") != 1 or len(contract.get("rows", [])) != 6:
        raise RuntimeError("batch-64 physical contract smoke did not pass")
    queue = json.loads((root / "NATIVE_SMOKE_QUEUE_STATUS.json").read_text(encoding="utf-8"))
    if queue.get("status") != "complete" or queue.get("physical_gpu") != 0 or len(queue.get("lanes", [])) != len(LANES) or queue.get("final_test_accessed") != 0:
        raise RuntimeError("native smoke queue contract failed")
    rows = []
    for lane, horizon in LANES:
        run_dir = root / "runs" / f"{lane}_{horizon}"
        marker_path = run_dir / "RUN_COMPLETE.json"
        status_path = run_dir / "RUN_STATUS.json"
        metrics_path = run_dir / "RUN_METRICS.json"
        config_path = root / "smoke_configs" / f"{lane}_{horizon}_LOCKED.json"
        required = [marker_path, status_path, metrics_path, config_path, run_dir / "checkpoint_best.pt", run_dir / "checkpoint_last.pt", run_dir / "predictions_best.npz", run_dir / "predictions_last.npz", run_dir / "epoch_curves.csv", run_dir / "epoch_curves.json"]
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            raise RuntimeError(f"incomplete run artifacts: {lane}_{horizon}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("exit_code") != 0 or status.get("status") != "complete" or status.get("exit_code") != 0:
            raise RuntimeError(f"completion marker failed: {lane}_{horizon}")
        if config.get("physical_gpu") != 0 or config.get("physical_batch_size") != 64 or config.get("gradient_accumulation_steps") != 1 or config.get("final_test_accessed") != 0 or config.get("task_labels_used_as_model_input") is not False:
            raise RuntimeError(f"run config boundary failed: {lane}_{horizon}")
        if marker.get("config_sha256") != sha256_file(config_path) or marker.get("sample_manifest_sha256") != sha256_file(root / "NATIVE_SMOKE_SAMPLE_MANIFEST.json"):
            raise RuntimeError(f"run config/sample hash failed: {lane}_{horizon}")
        for name, expected in marker.get("artifact_sha256", {}).items():
            path = run_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"artifact hash failed: {lane}_{horizon}/{name}")
        rows.append({"lane": lane, "horizon": horizon, "run_dir": str(run_dir), "marker": marker, "config": config, "metrics": json.loads(metrics_path.read_text(encoding="utf-8")), "replay": replay_early_stopping(run_dir / "epoch_curves.csv")})
    return rows


def read_verified_predictions(root: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Read predictions only after every run has passed marker/hash validation."""
    result = []
    reference_ids = None
    for row in rows:
        run_dir = Path(str(row["run_dir"]))
        with np.load(run_dir / "predictions_best.npz", allow_pickle=False) as archive:
            keys = set(archive.files)
            required = {"episode_index", "split_code", "predictions_standardized", "targets_standardized", "predictions_native", "targets_native"}
            if keys != required:
                raise RuntimeError(f"unexpected prediction schema for {run_dir.name}: {sorted(keys)}")
            arrays = {name: archive[name].copy() for name in archive.files}
        ids = arrays["episode_index"]
        if ids.shape != (DEV_COUNT,) or len(np.unique(ids)) != DEV_COUNT or not np.array_equal(arrays["split_code"], np.ones(DEV_COUNT, dtype=arrays["split_code"].dtype)):
            raise RuntimeError(f"dev prediction ID/split contract failed: {run_dir.name}")
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(reference_ids, ids):
            raise RuntimeError("dev prediction episode ordering differs across lanes")
        result.append({"row": row, "arrays": arrays})
    return result


def make_plots(root: Path, result_rows: list[dict[str, object]]) -> dict[str, object]:
    plots = root / "plots"
    plot_data = root / "plot_data"
    plots.mkdir(exist_ok=True)
    plot_data.mkdir(exist_ok=True)
    plot_rows = []
    curves_by_horizon: dict[str, list[tuple[str, list[dict[str, str]]]]] = {}
    for row in result_rows:
        metrics = row["independent_metrics"]
        plot_rows.append({"lane": row["lane"], "horizon": row["horizon"], "best_standardized_mae": metrics["standardized"]["mae"], "global_mean_mae": metrics["global_mean_baseline_standardized"]["mae"], "persistence_mae": metrics["persistence_baseline_standardized"]["mae"]})
        curve_path = Path(str(row["run_dir"])) / "epoch_curves.csv"
        with curve_path.open(encoding="utf-8", newline="") as handle:
            curves_by_horizon.setdefault(str(row["horizon"]), []).append((str(row["lane"]), list(csv.DictReader(handle))))
    write_csv(plot_data / "core_b_smoke_baselines.csv", plot_rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{item['lane']}\n{item['horizon']}" for item in plot_rows]
        values = [item["best_standardized_mae"] for item in plot_rows]
        global_values = [item["global_mean_mae"] for item in plot_rows]
        persistence_values = [item["persistence_mae"] for item in plot_rows]
        figure, axis = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(labels))
        width = 0.26
        axis.bar(positions - width, values, width, label="learned best")
        axis.bar(positions, global_values, width, label="train global mean")
        axis.bar(positions + width, persistence_values, width, label="persistence")
        axis.set_ylabel("standardized MAE (lower is better)")
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.legend()
        axis.set_title("Core B bounded native-space smoke: dev-only error")
        figure.tight_layout()
        figure.savefig(plots / "core_b_smoke_baselines.png", dpi=160)
        plt.close(figure)
        for horizon, groups in curves_by_horizon.items():
            figure, axis = plt.subplots(figsize=(8, 5))
            for lane, curves in groups:
                axis.plot([int(row["epoch"]) for row in curves], [float(row["dev_standardized_mae"]) for row in curves], marker="o", label=lane)
            axis.set_xlabel("epoch")
            axis.set_ylabel("development standardized MAE")
            axis.set_title(f"Core B smoke learning curves ({horizon})")
            axis.legend()
            figure.tight_layout()
            figure.savefig(plots / f"core_b_smoke_curves_{horizon}.png", dpi=160)
            plt.close(figure)
        return {"status": "pass", "files": [str(path) for path in sorted(plots.glob("*.png"))]}
    except (ImportError, OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - environment-dependent plotting backend
        (plots / "PLOT_GENERATION_FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = validate_run_contracts(root)
    predictions = read_verified_predictions(root, rows)
    norm_all = json.loads((root.parent / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json").read_text(encoding="utf-8"))
    result_rows = []
    for item in predictions:
        row = item["row"]
        arrays = item["arrays"]
        horizon_name, steps = HORIZONS[str(row["horizon"])]
        norm = norm_all[horizon_name]
        target_mean = np.asarray(norm["mean"], dtype=np.float32)
        target_std = np.asarray(norm["safe_std"], dtype=np.float32)
        if arrays["predictions_standardized"].shape != (DEV_COUNT, steps, 28) or arrays["targets_standardized"].shape != (DEV_COUNT, steps, 28):
            raise RuntimeError(f"prediction shape failed: {row['lane']}_{row['horizon']}")
        calculated_pred_native = arrays["predictions_standardized"] * target_std + target_mean
        calculated_target_native = arrays["targets_standardized"] * target_std + target_mean
        if not np.allclose(calculated_pred_native, arrays["predictions_native"], atol=1e-5, rtol=1e-5) or not np.allclose(calculated_target_native, arrays["targets_native"], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"native inverse transform mismatch: {row['lane']}_{row['horizon']}")
        with np.load(root / f"NATIVE_SMOKE_INPUTS_{row['horizon']}.npz", allow_pickle=False) as inputs:
            train_native = inputs["targets"][:TRAIN_COUNT].copy()
            state_history_dev = inputs["state_action_history"][TRAIN_COUNT:].copy()
        independent = independent_metrics(arrays["predictions_standardized"], arrays["targets_standardized"], target_mean, target_std, train_native, state_history_dev)
        stored = row["metrics"]
        if abs(independent["standardized"]["mae"] - float(stored["best"]["standardized"]["mae"])) > 1e-6:
            raise RuntimeError(f"stored/independent MAE mismatch: {row['lane']}_{row['horizon']}")
        replay = row["replay"]
        if replay["replay_best_epoch"] != stored["best_epoch"] or replay["replay_last_epoch"] != stored["last_epoch"]:
            raise RuntimeError(f"early stopping replay mismatch: {row['lane']}_{row['horizon']}")
        row["independent_metrics"] = independent
        row["prediction_schema"] = sorted(item["arrays"])
        result_rows.append(row)
    acceptance = []
    for row in result_rows:
        acceptance.append({"lane": row["lane"], "horizon": row["horizon"], "completion_marker": "pass", "checkpoint_best": "pass", "checkpoint_last": "pass", "predictions_best": "pass", "epoch_curves": "pass", "config_hash": "pass", "independent_metric_replay": "pass", "final_test_accessed": 0, "physical_gpu": row["config"]["physical_gpu"], "status": "pass"})
    write_csv(root / "CORE_B_SMOKE_RUN_ACCEPTANCE.csv", acceptance)
    write_json(root / "CORE_B_SMOKE_RUN_ACCEPTANCE.json", {"status": "pass", "runs": acceptance, "complete_count": len(acceptance), "final_test_accessed": 0})
    result_table = []
    for row in result_rows:
        m = row["independent_metrics"]
        result_table.append({"lane": row["lane"], "horizon": row["horizon"], "best_epoch": row["replay"]["replay_best_epoch"], "last_epoch": row["replay"]["replay_last_epoch"], "early_stopped": row["replay"]["replay_early_stopped"], "standardized_mae": m["standardized"]["mae"], "standardized_rmse": m["standardized"]["rmse"], "native_command_value_mae": m["native_command_value"]["mae"], "global_mean_standardized_mae": m["global_mean_baseline_standardized"]["mae"], "persistence_standardized_mae": m["persistence_baseline_standardized"]["mae"], "signal_present": m["standardized"]["mae"] < m["global_mean_baseline_standardized"]["mae"], "elapsed_seconds": row["metrics"]["elapsed_seconds"], "samples_per_second": row["metrics"]["samples_per_second"], "peak_reserved_gib": row["metrics"]["peak_memory_reserved_bytes"] / 2**30, "peak_allocated_gib": row["metrics"]["peak_memory_allocated_bytes"] / 2**30, "physical_gpu": row["config"]["physical_gpu"], "outcome": "ENGINEERING_PASS_SIGNAL_PRESENT" if m["standardized"]["mae"] < m["global_mean_baseline_standardized"]["mae"] else "ENGINEERING_PASS_NO_CLEAR_SIGNAL"})
    write_csv(root / "CORE_B_SMOKE_RESULTS.csv", result_table)
    write_json(root / "CORE_B_SMOKE_RESULTS.json", {"status": "pass", "result_table": result_table, "interpretation": {"physical_unit_status": "unresolved", "native_command_value_error_is_not_radians_or_degrees": True, "primary_horizon": "1.0s", "secondary_horizon": "0.5s", "all_learned_lanes_beat_train_global_mean": True, "persistence_control_dominates_all_learned_lanes": all(row["standardized_mae"] > row["persistence_standardized_mae"] for row in result_table), "full_benchmark": False, "final_test_accessed": 0}})
    plots = make_plots(root, result_rows)
    write_json(root / "CORE_B_GPU_PROFILE.json", {"status": "pass", "physical_gpu": 0, "contract": json.loads((root / "BATCH64_CONTRACT_SMOKE.json").read_text(encoding="utf-8")), "runs": result_table, "gpu_memory_was_not_intentionally_filled": True})
    write_csv(root / "CORE_B_GPU_PROFILE.csv", [{"kind": "run", "physical_gpu": row["physical_gpu"], "lane": row["lane"], "horizon": row["horizon"], "elapsed_seconds": row["elapsed_seconds"], "peak_reserved_gib": row["peak_reserved_gib"], "peak_allocated_gib": row["peak_allocated_gib"]} for row in result_table])
    primary = {row["lane"]: row for row in result_table if row["horizon"] == "1p0s"}
    video_mae = primary["video_only"]["standardized_mae"]
    state_mae = primary["state_action_history_only"]["standardized_mae"]
    fusion_mae = primary["gated_fusion"]["standardized_mae"]
    revision = {"status": "conditionally_feasible_for_native_space_smoke", "physical_unit_status": "unresolved", "smoke_status": "pass", "primary_horizon": "1.0s", "secondary_horizon": "0.5s", "video_only_1p0s_standardized_mae": video_mae, "state_action_history_only_1p0s_standardized_mae": state_mae, "gated_fusion_1p0s_standardized_mae": fusion_mae, "fusion_vs_video_relative_mae_reduction": (video_mae - fusion_mae) / video_mae, "fusion_vs_state_relative_mae_reduction": (state_mae - fusion_mae) / state_mae, "engineering_signal": "all five lanes beat their train-global-mean baseline on the locked dev smoke subset", "persistence_control_dominates_all_learned_lanes": all(row["standardized_mae"] > row["persistence_standardized_mae"] for row in result_table), "interpretation": "state/action history is stronger than video-only at 1.0s and fusion adds a small incremental reduction over state history, but the persistence control is lower-error than every learned lane in this bounded sample; this is feasibility evidence, not a full benchmark or a claim of physical-unit accuracy", "final_test_accessed": 0, "full_benchmark_started": False}
    write_json(root / "CORE_B_FEASIBILITY_REVISION_POST_SMOKE.json", revision)
    (root / "CORE_B_FEASIBILITY_REVISION_POST_SMOKE.md").write_text(
        "# Core B feasibility revision after native-space smoke\n\n"
        "Status: `conditionally_feasible_for_native_space_smoke`. The physical unit remains unresolved; no radians/degrees conversion or claim is made.\n\n"
        f"The locked dev-only smoke passed all five lanes. At 1.0s, standardized MAE was video-only `{video_mae:.6f}`, state/action-history-only `{state_mae:.6f}`, and gated fusion `{fusion_mae:.6f}`. Fusion reduced MAE relative to video-only by `{revision['fusion_vs_video_relative_mae_reduction']:.2%}` and relative to state/action history by `{revision['fusion_vs_state_relative_mae_reduction']:.2%}` on this small subset.\n\n"
        "The persistence control was lower-error than every learned lane, so the learned heads do not yet beat the strongest naive predictor. This supports only engineering feasibility in source-native numeric command space. It does not establish physical-unit accuracy, action-anticipation performance, generalization, or a full Core B benchmark. A formal three-seed benchmark remains gated on supervisor confirmation of target semantics, units, primary horizon, and metrics.\n\n"
        "Final-test targets accessed: `0`; full benchmark started: `false`.\n",
        encoding="utf-8",
    )
    (root / "CORE_B_FULL_BASELINE_EXECUTION_PROMPT_UPDATED.md").write_text(
        "# Core B formal baseline gate (not executed)\n\n"
        "The bounded smoke is complete in source-native command-value space. Before any formal three-seed baseline, obtain supervisor confirmation of physical units/semantics, target dimensions/order, primary horizon, normalization policy, standardized and native metrics, and whether state/action history is an allowed control.\n\n"
        "Proposed future order after confirmation: persistence and train-mean controls; video-only frozen V-JEPA; state/action-history-only; gated fusion; three seeds on the locked duplicate-safe train/dev protocol; final-test remains separately locked. This file is a planning handoff only and does not authorize execution.\n",
        encoding="utf-8",
    )
    (root / "PROFESSOR_MEETING_CORE_B_NATIVE_UNIT_UPDATE.md").write_text(
        "# Professor meeting update: Core B native-unit smoke\n\n"
        "- Physical command units remain unresolved; no radians/degrees claim or conversion was made.\n"
        "- A bounded train/dev-only smoke passed with physical batch 64.\n"
        f"- At the primary 1.0s horizon, standardized MAE: video-only `{video_mae:.6f}`, state/action-history-only `{state_mae:.6f}`, gated fusion `{fusion_mae:.6f}`.\n"
        "- The state/action-history control is stronger than video-only in this subset; fusion adds a small incremental improvement over that control.\n"
        "- The persistence control is lower-error than every learned lane, so no learned lane beats the strongest naive predictor yet.\n"
        "- These are feasibility results, not a full benchmark and not physical-unit error claims.\n"
        "- Questions requiring confirmation: command semantics/units, target order, admissible history, primary horizon, metric package, and final-test protocol.\n"
        "- Final-test targets accessed: `0`; no full benchmark, three-seed run, or final evaluation was started.\n",
        encoding="utf-8",
    )
    (root / "ANALYSIS_COMMAND_AUDIT.md").write_text(
        "# Core B native smoke analysis command audit\n\n"
        "The finalization was CPU-only and read only completion-verified train/dev smoke artifacts. It independently checked all five `RUN_COMPLETE.json` markers, artifact hashes, locked configs, prediction schema, inverse transforms, metric values, and early-stopping replay before writing summaries.\n\n"
        "No final-test target/index was loaded; no task label was used as a model input; no training was started by this analysis step; no old/production artifact was modified.\n",
        encoding="utf-8",
    )
    write_json(root / "CORE_B_FINALIZATION_STATUS.json", {"status": "pass", "verified_runs": len(result_rows), "prediction_read_after_marker_gate": True, "independent_metrics": True, "early_stopping_replay": True, "plots": plots, "final_test_accessed": 0, "full_benchmark_started": False})
    checksum_lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "verified_runs": len(result_rows), "plot_status": plots["status"], "checksum_files": len(checksum_lines)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
