"""Recompute all D1 seed/horizon comparisons from accepted V3 dev predictions."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/p2_seed_audit"
MODELS = ("S-HIST", "F-TOK-D", "F-TOK-S", "F-POOL-D")
SEEDS = (20260950, 20260951, 20260952)
HORIZONS = (30, 60)
STRATA = ("natural_overall", "supported_overall", "low_dispersion_supported", "middle_dispersion_supported", "high_dispersion_supported")
BODY = {"Hand": np.arange(0, 14), "Arm": np.arange(14, 28), "Full28": np.arange(0, 28)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def group_values(values: np.ndarray, mask: np.ndarray, group_ids: np.ndarray, tasks: np.ndarray | None = None):
    buckets: dict[object, list[np.ndarray]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        key: object = str(group_ids[index]) if tasks is None else (int(tasks[index]), str(group_ids[index]))
        buckets[key].append(np.asarray(values[index]))
    return {key: np.mean(np.stack(items), axis=0) for key, items in buckets.items()}


def average_buckets(buckets: dict[object, np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(list(buckets.values())), axis=0) if buckets else np.full((), np.nan)


def task_joint_mean(values: np.ndarray, mask: np.ndarray, group_ids: np.ndarray, tasks: np.ndarray, task_order: list[int]) -> np.ndarray:
    buckets = group_values(values, mask, group_ids, tasks)
    out = np.full((len(task_order), values.shape[-1]), np.nan, dtype=np.float64)
    for row, task in enumerate(task_order):
        vals = [value for key, value in buckets.items() if key[0] == task]
        if vals:
            out[row] = np.mean(np.stack(vals), axis=0)
    return out


def strata_masks(ids: np.ndarray, query: dict[str, dict[str, str]]) -> dict[str, np.ndarray]:
    rows = [query[str(value)] for value in ids]
    low = np.asarray([row["low_h60"] == "True" for row in rows])
    high = np.asarray([row["high_h60"] == "True" for row in rows])
    supported = np.asarray([row["supported"] == "True" for row in rows])
    return {
        "natural_overall": np.ones(len(ids), dtype=bool),
        "supported_overall": supported,
        "low_dispersion_supported": low & supported,
        "middle_dispersion_supported": (~low) & (~high) & supported,
        "high_dispersion_supported": high & supported,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_path = V3 / "data/V3_MAIN_INPUTS.npz"
    with np.load(input_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in (
            "anchor_id", "episode_index", "original_task_index", "split_code",
            "anchor_frame", "anchor_timestamp", "history_selected_timestamps",
            "targets_native", "anchor_command_native",
        )}
    if set(np.unique(data["split_code"]).tolist()) != {0, 1}:
        raise AssertionError("V3 primary input unexpectedly includes an unapproved split")
    dev_indices = np.flatnonzero(data["split_code"] == 1)
    dev_ids = data["anchor_id"][dev_indices].astype("U64")
    if len(dev_ids) != 2322 or len(set(dev_ids.tolist())) != len(dev_ids):
        raise AssertionError("locked V3 primary dev ID cardinality failed")
    id_to_index = {str(value): int(index) for index, value in enumerate(data["anchor_id"])}

    source_rows = read_csv(RESULT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv")
    query = {row["anchor_id"]: row for row in source_rows}
    if set(dev_ids.tolist()) != set(query):
        raise AssertionError("D1 source-group table does not exactly cover V3 dev IDs")
    group_by_id = {key: value["source_group_id"] for key, value in query.items()}
    groups = np.asarray([group_by_id[str(value)] for value in dev_ids], dtype="U64")
    masks = strata_masks(dev_ids, query)
    dev_ep = data["episode_index"][dev_indices]
    dev_task = data["original_task_index"][dev_indices]
    dev_anchor = data["anchor_command_native"][dev_indices]
    max_history_future = float(np.max(data["history_selected_timestamps"][dev_indices] - data["anchor_timestamp"][dev_indices, None]))
    if max_history_future > 1e-5:
        raise AssertionError(f"V3 input history includes timestamp after anchor: {max_history_future}")

    normalizer_path = V3 / "data/V3_NORMALIZERS.json"
    video_normalizer_path = V3 / "data/V3_VIDEO_NORMALIZER.json"
    normalizers = json.loads(normalizer_path.read_text())
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float64)
    if normalizers.get("final_test_access") != 0 or normalizers.get("fit_scope") != "primary train windows only; episode equal":
        raise AssertionError("V3 target normalizer train-only contract failed")
    sigma_hash = sha256_file(normalizer_path)
    video_norm_hash = sha256_file(video_normalizer_path)
    cache_path = V3 / "cache/V3_TEMPORAL_FEATURES.npy"
    cache_hash = sha256_file(cache_path)
    model_path = V3 / "code/model.py"
    model_hash = sha256_file(model_path)
    input_hash = sha256_file(input_path)
    train_code = (V3 / "code/train_v3.py").read_text(encoding="utf-8")
    infer_code = (V3 / "code/infer_v3.py").read_text(encoding="utf-8")
    matched_order_contract = "torch.Generator().manual_seed(args.seed+epoch)" in train_code.replace(" ", "")
    id_join_contract = "anchor_id=data[\"anchor_id\"][indices]" in infer_code

    summary_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    lineage_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    run_values: dict[tuple[str, int, int], dict[str, object]] = {}
    task_joint_cache: dict[tuple[str, int, int, str, str], np.ndarray] = {}
    dynamic_rows: list[dict[str, object]] = []
    curves_by_cell: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    stored_metrics: list[dict[str, object]] = []
    pred_arrays_for_figures: dict[tuple[str, int, int], np.ndarray] = {}
    task_order = sorted(int(x) for x in np.unique(dev_task))

    for seed in SEEDS:
        for horizon in HORIZONS:
            target = data["targets_native"][dev_indices, :horizon].astype(np.float64)
            anchor = dev_anchor.astype(np.float64)
            persistence = np.repeat(anchor[:, None, :], horizon, axis=1)
            for model in MODELS:
                run_id = f"V3-{model}-H{horizon}-S{seed}"
                run_root = V3 / "runs" / run_id
                marker_path = V3 / "predictions" / run_id / "PREDICTION_EXPORT_COMPLETE.json"
                marker = json.loads(marker_path.read_text())
                pred_path = V3 / "predictions" / run_id / "primary.npz"
                with np.load(pred_path, allow_pickle=False) as bundle:
                    pred_ids = bundle["anchor_id"].astype("U64")
                    pred = bundle["predictions_native"].astype(np.float64)
                    pred_ep = bundle["episode_index"]
                    pred_task = bundle["original_task_index"]
                    pred_frame = bundle["anchor_frame"]
                    pred_ts = bundle["anchor_timestamp"]
                    pred_horizon = int(bundle["horizon_steps"])
                if marker.get("status") != "PASS" or marker.get("final_test_access") != 0:
                    raise AssertionError(f"prediction sidecar not accepted: {run_id}")
                if pred_horizon != horizon or len(pred_ids) != len(dev_ids) or len(set(pred_ids.tolist())) != len(pred_ids):
                    raise AssertionError(f"prediction shape/horizon/ID uniqueness failed: {run_id}")
                if set(pred_ids.tolist()) != set(dev_ids.tolist()) or pred.shape != (len(dev_ids), horizon, 28):
                    raise AssertionError(f"prediction target ID/shape mismatch: {run_id}")
                order = np.asarray([int(np.where(pred_ids == value)[0][0]) for value in dev_ids], dtype=np.int64)
                pred = pred[order]
                source = np.asarray([id_to_index[str(value)] for value in dev_ids], dtype=np.int64)
                if np.any(data["split_code"][source] != 1):
                    raise AssertionError("P2 attempted non-dev prediction join")
                if not (
                    np.array_equal(pred_ep[order], dev_ep)
                    and np.array_equal(pred_task[order], dev_task)
                    and np.array_equal(pred_frame[order], data["anchor_frame"][dev_indices])
                    and np.allclose(pred_ts[order], data["anchor_timestamp"][dev_indices], atol=1e-5, rtol=0)
                ):
                    raise AssertionError(f"prediction ID join metadata mismatch: {run_id}")
                if not np.isfinite(pred).all():
                    raise AssertionError(f"nonfinite prediction values: {run_id}")

                checkpoint = Path(str(marker["checkpoint"]))
                checkpoint_sha = sha256_file(checkpoint)
                checkpoint_match = checkpoint_sha == marker.get("checkpoint_sha256")
                cfg_path = run_root / ("formal_pilot_retry1" if seed == 20260950 else "formal") / "RUN_CONFIG_LOCKED.json"
                complete_path = cfg_path.parent / "RUN_COMPLETE.json"
                curves_path = cfg_path.parent / "EPOCH_CURVES.json"
                cfg = json.loads(cfg_path.read_text())
                complete = json.loads(complete_path.read_text())
                curve_payload = json.loads(curves_path.read_text())
                curves = list(curve_payload.get("rows", []))
                if cfg.get("input_sha256") != input_hash or cfg.get("cache_sha256") != cache_hash or cfg.get("model_code_sha256") != model_hash:
                    raise AssertionError(f"V3 run data/cache/model-code hash mismatch: {run_id}")
                if cfg.get("batch_size") != 64 or cfg.get("accumulation") != 1 or cfg.get("epochs") != 40 or complete.get("optimizer_steps") != 11120:
                    raise AssertionError(f"V3 run training contract differs from lock: {run_id}")
                if not checkpoint_match or complete.get("status") != "PASS" or complete.get("final_test_access") != 0:
                    raise AssertionError(f"V3 checkpoint/complete marker mismatch: {run_id}")
                lineage_rows.append({
                    "run_id": run_id,
                    "model": model,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_hash_matches_export": checkpoint_match,
                    "initial_model_hash": cfg.get("initial_model_hash"),
                    "epochs": complete.get("epoch"),
                    "optimizer_steps": complete.get("optimizer_steps"),
                    "train_windows": cfg.get("train_windows"),
                    "train_episodes": cfg.get("train_episodes"),
                    "input_sha256": cfg.get("input_sha256"),
                    "cache_sha256": cfg.get("cache_sha256"),
                    "model_code_sha256": cfg.get("model_code_sha256"),
                    "normalizer_sha256": sigma_hash,
                    "video_normalizer_sha256": video_norm_hash,
                    "prediction_rows": len(pred_ids),
                    "prediction_id_set_matches_dev": True,
                    "prediction_sidecar_state_unchanged": marker.get("state_unchanged"),
                    "final_test_access": 0,
                })
                curves_by_cell[(model, seed, horizon)] = curves

                truth = data["targets_native"][source, :horizon].astype(np.float64)
                sigma_for_joint = sigma[None, None, :]
                std_error = np.abs(pred - truth) / sigma_for_joint
                fd_error = np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1))
                persist_error = np.abs(persistence - truth) / sigma_for_joint
                std_joint = np.mean(std_error, axis=1)
                fd_joint = np.mean(fd_error, axis=1)
                persist_joint = np.mean(persist_error, axis=1)
                body_arrays: dict[str, dict[str, np.ndarray]] = {}
                for body, joint_indices in BODY.items():
                    body_arrays[body] = {
                        "mae": np.mean(std_joint[:, joint_indices], axis=1),
                        "fd": np.mean(fd_joint[:, joint_indices], axis=1),
                        "persistence": np.mean(persist_joint[:, joint_indices], axis=1),
                    }
                    early_end = horizon // 2
                    late_std = [
                        np.mean(std_error[:, :early_end, joint_indices]),
                        np.mean(std_error[:, early_end:, joint_indices]),
                    ]
                    pred_fd = np.diff(pred, axis=1)
                    true_fd = np.diff(truth, axis=1)
                    early_fd_end = max(1, early_end - 1)
                    late_fd_start = early_end
                    late_fd = np.mean(np.abs(pred_fd[:, late_fd_start:, joint_indices] - true_fd[:, late_fd_start:, joint_indices]))
                    early_fd = np.mean(np.abs(pred_fd[:, :early_fd_end, joint_indices] - true_fd[:, :early_fd_end, joint_indices]))
                    for segment, mae_value, fd_value in (("first_half", early_std[0] if False else late_std[0], early_fd), ("second_half", late_std[1], late_fd)):
                        segment_rows.append({
                            "model": model,
                            "seed": seed,
                            "horizon_steps": horizon,
                            "body": body,
                            "segment": segment,
                            "N_source_groups": len(set(groups.tolist())),
                            "N_windows": len(dev_ids),
                            "standardized_mae": float(mae_value),
                            "native_fd": float(fd_value),
                            "final_test_access": 0,
                        })

                # Distribution/output-scale diagnostics remain descriptive.
                dynamic_rows.append({
                    "model": model,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "prediction_min": float(pred.min()),
                    "prediction_max": float(pred.max()),
                    "prediction_abs_p99": float(np.quantile(np.abs(pred), 0.99)),
                    "prediction_std": float(pred.std()),
                    "target_min": float(truth.min()),
                    "target_max": float(truth.max()),
                    "target_std": float(truth.std()),
                    "final_test_access": 0,
                })
                run_values[(model, seed, horizon)] = {
                    "pred": pred,
                    "truth": truth,
                    "anchor": anchor,
                    "mae_joint": std_joint,
                    "fd_joint": fd_joint,
                    "body": body_arrays,
                    "source": source,
                }
                if (seed, horizon) in ((20260951, 30), (20260952, 60), (20260950, 30)):
                    pred_arrays_for_figures[(model, seed, horizon)] = pred

                for stratum in STRATA:
                    mask = masks[stratum]
                    for body in BODY:
                        body_values = body_arrays[body]
                        g_mae = group_values(body_values["mae"], mask, groups)
                        g_fd = group_values(body_values["fd"], mask, groups)
                        g_persist = group_values(body_values["persistence"], mask, groups)
                        summary_rows.append({
                            "model": model,
                            "seed": seed,
                            "horizon_steps": horizon,
                            "body": body,
                            "stratum": stratum,
                            "N_source_groups": len(g_mae),
                            "N_windows": int(mask.sum()),
                            "standardized_mae_episode_equal": float(average_buckets(g_mae)),
                            "native_fd_episode_equal": float(average_buckets(g_fd)),
                            "persistence_standardized_mae_episode_equal": float(average_buckets(g_persist)),
                            "skill_vs_persistence": float(1.0 - average_buckets(g_mae) / average_buckets(g_persist)) if g_persist else None,
                            "final_test_access": 0,
                        })

                    # Source-group means nested within task, then task-wise
                    # descriptive values. No p-values or seed deletion.
                    for task in task_order:
                        task_mask = mask & (dev_task == task)
                        if not task_mask.any():
                            continue
                        for body in BODY:
                            group_mae_task = group_values(body_arrays[body]["mae"], task_mask, groups)
                            group_fd_task = group_values(body_arrays[body]["fd"], task_mask, groups)
                            task_rows.append({
                                "model": model,
                                "seed": seed,
                                "horizon_steps": horizon,
                                "stratum": stratum,
                                "task": task,
                                "body": body,
                                "N_source_groups": len(group_mae_task),
                                "N_windows": int(task_mask.sum()),
                                "standardized_mae": float(average_buckets(group_mae_task)),
                                "native_fd": float(average_buckets(group_fd_task)),
                                "final_test_access": 0,
                            })

                    for metric_name, values in (("standardized_mae", std_joint), ("native_fd", fd_joint)):
                        by_task_joint = task_joint_mean(values, mask, groups, dev_task, task_order)
                        task_joint_cache[(model, seed, horizon, stratum, metric_name)] = by_task_joint
                        for task_row, task in enumerate(task_order):
                            task_mask = mask & (dev_task == task)
                            task_group_count = len(group_values(values[:, :1].mean(axis=1), task_mask, groups))
                            for joint in range(28):
                                value = by_task_joint[task_row, joint]
                                if not np.isfinite(value):
                                    continue
                                joint_rows.append({
                                    "model": model,
                                    "seed": seed,
                                    "horizon_steps": horizon,
                                    "stratum": stratum,
                                    "task": task,
                                    "joint": f"j{joint:02d}",
                                    "metric": metric_name,
                                    "N_source_groups": task_group_count,
                                    "N_windows": int(task_mask.sum()),
                                    "estimate": float(value),
                                    "final_test_access": 0,
                                })

    # Paired visual gains and task/joint deltas use identical source-group sets.
    for seed in SEEDS:
        for horizon in HORIZONS:
            for stratum in STRATA:
                for body in BODY:
                    candidate = run_values[("F-TOK-D", seed, horizon)]["body"][body]
                    for control in ("S-HIST", "F-TOK-S", "F-POOL-D"):
                        control_values = run_values[(control, seed, horizon)]["body"][body]
                        control_groups = group_values(control_values["mae"], masks[stratum], groups)
                        candidate_groups = group_values(candidate["mae"], masks[stratum], groups)
                        common = sorted(set(control_groups) & set(candidate_groups))
                        gain = np.asarray([control_groups[key] - candidate_groups[key] for key in common], dtype=np.float64)
                        gain_rows.append({
                            "seed": seed,
                            "horizon_steps": horizon,
                            "body": body,
                            "stratum": stratum,
                            "control": control,
                            "candidate": "F-TOK-D",
                            "gain_control_minus_candidate": float(gain.mean()) if len(gain) else None,
                            "source_group_sd": float(gain.std(ddof=1)) if len(gain) > 1 else None,
                            "N_source_groups": len(common),
                            "N_windows": int(masks[stratum].sum()),
                            "interval_note": "descriptive SD across paired source groups; not a confidence interval",
                            "final_test_access": 0,
                        })
                for control in ("S-HIST", "F-TOK-S", "F-POOL-D"):
                    for metric_name in ("standardized_mae", "native_fd"):
                        control_matrix = task_joint_cache[(control, seed, horizon, stratum, metric_name)]
                        candidate_matrix = task_joint_cache[("F-TOK-D", seed, horizon, stratum, metric_name)]
                        delta = control_matrix - candidate_matrix
                        for row, task in enumerate(task_order):
                            for joint in range(28):
                                if np.isfinite(delta[row, joint]):
                                    joint_rows.append({
                                        "model": f"F-TOK-D_minus_{control}",
                                        "seed": seed,
                                        "horizon_steps": horizon,
                                        "stratum": stratum,
                                        "task": task,
                                        "joint": f"j{joint:02d}",
                                        "metric": f"gain_{metric_name}_control_minus_candidate",
                                        "N_source_groups": int(np.sum(masks[stratum] & (dev_task == task))),
                                        "N_windows": int(np.sum(masks[stratum] & (dev_task == task))),
                                        "estimate": float(delta[row, joint]),
                                        "final_test_access": 0,
                                    })

    # Compare the new ID-joined recomputation with the previously sealed D1
    # table to catch accidental row-order and metric-definition drift.
    stored = read_csv(BENCH / "d1/D1_LEARNED_RESULTS.csv")
    stored_lookup = {(r["model"], int(r["seed"]), int(r["horizon_steps"]), r["stratum"], r["body"]): r for r in stored}
    agree_rows = []
    for row in summary_rows:
        key = (row["model"], row["seed"], row["horizon_steps"], row["stratum"].replace("_overall", "").replace("_supported", ""), row["body"])
        match = stored_lookup.get(key)
        if match:
            stored_mae = float(match["standardized_mae_episode_equal"])
            stored_fd = float(match["native_fd_episode_equal"])
            agree_rows.append({
                "model": row["model"], "seed": row["seed"], "horizon_steps": row["horizon_steps"],
                "body": row["body"], "stratum": row["stratum"],
                "mae_delta_recomputed_minus_stored": row["standardized_mae_episode_equal"] - stored_mae,
                "fd_delta_recomputed_minus_stored": row["native_fd_episode_equal"] - stored_fd,
            })

    # Fixed representative is selected before model errors: smallest stable
    # hash of an ID. The worst trajectory is a separate, explicitly post-hoc
    # failure diagnostic selected by maximum F-TOK-D Arm sMAE for H30/seed51.
    fixed_id = min(dev_ids.tolist(), key=lambda value: hashlib.sha256(f"{value}|P2_FIXED_REPRESENTATIVE_V1".encode()).hexdigest())
    anomaly_key = ("F-TOK-D", 20260951, 30)
    anomaly_values = run_values[anomaly_key]
    arm_joint = BODY["Arm"]
    worst_local = int(np.argmax(np.mean(np.abs(anomaly_values["pred"] - anomaly_values["truth"])[:, :, arm_joint] / sigma[None, None, arm_joint], axis=(1, 2))))
    worst_id = str(dev_ids[worst_local])

    # Uncertainty and dynamic-range table is intentionally descriptive.
    agreement_max = max((max(abs(row["mae_delta_recomputed_minus_stored"]), abs(row["fd_delta_recomputed_minus_stored"])) for row in agree_rows), default=None)
    for key in (("F-TOK-D", 20260951, 30), ("F-TOK-D", 20260952, 60)):
        model, seed, horizon = key
        for selected_id, selection_kind in ((fixed_id, "fixed_hash_selected"), (worst_id, "posthoc_worst_arm_error_for_seed51_h30")):
            if selected_id not in set(dev_ids.tolist()):
                continue
            local = int(np.flatnonzero(dev_ids == selected_id)[0])
            for model_name in MODELS:
                plot_pred = run_values[(model_name, seed, horizon)]["pred"][local]
                for joint in (0, 14):
                    for step in range(horizon):
                        target_value = run_values[(model_name, seed, horizon)]["truth"][local, step, joint]
                        dynamic_rows.append({
                            "record_type": "trajectory_point",
                            "selection": selection_kind,
                            "anchor_id": selected_id,
                            "model": model_name,
                            "seed": seed,
                            "horizon_steps": horizon,
                            "joint": f"j{joint:02d}",
                            "step": step + 1,
                            "seconds": (step + 1) / 30.0,
                            "target_native": float(target_value),
                            "prediction_native": float(plot_pred[step, joint]),
                            "final_test_access": 0,
                        })

    write_csv(OUT / "P2_RECOMPUTED_SEED_HORIZON.csv", summary_rows, list(summary_rows[0]))
    write_csv(OUT / "P2_TASK_DECOMPOSITION.csv", task_rows, list(task_rows[0]))
    write_csv(OUT / "P2_TASK_JOINT_SEED_DECOMPOSITION.csv", joint_rows, list(joint_rows[0]))
    write_csv(OUT / "P2_HORIZON_SEGMENT.csv", segment_rows, list(segment_rows[0]))
    write_csv(OUT / "P2_GAIN_BY_STRATUM.csv", gain_rows, list(gain_rows[0]))
    write_csv(OUT / "P2_RECOMPUTATION_AGREEMENT.csv", agree_rows, list(agree_rows[0]))
    write_csv(OUT / "P2_TRAJECTORY_DATA.csv", dynamic_rows, list(dynamic_rows[0]))
    write_csv(OUT / "P2_CHECKPOINT_AND_OUTPUT_LINEAGE.csv", lineage_rows, list(lineage_rows[0]))

    # Figures: diverging heatmaps use a shared zero-centered scale; trajectories
    # use common native-unit axes within each joint and redundant line styles.
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    heatmaps = []
    for ax, (seed, horizon, title) in zip(axes, ((20260951, 30, "H30 / seed 20260951"), (20260952, 60, "H60 / seed 20260952"))):
        ctrl = task_joint_cache[("S-HIST", seed, horizon, "high_dispersion_supported", "standardized_mae")]
        cand = task_joint_cache[("F-TOK-D", seed, horizon, "high_dispersion_supported", "standardized_mae")]
        gain = ctrl - cand
        heatmaps.append(gain)
    max_abs = max(float(np.nanmax(np.abs(value))) for value in heatmaps)
    norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    image = None
    for ax, gain, title in zip(axes, heatmaps, ("H30 / seed 20260951", "H60 / seed 20260952")):
        image = ax.imshow(np.ma.masked_invalid(gain), aspect="auto", cmap="coolwarm", norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Joint index (0–27)")
        ax.set_ylabel("Original task index")
        ax.set_xticks(np.arange(0, 28, 2))
        ax.set_yticks(np.arange(len(task_order)))
        ax.set_yticklabels(task_order)
    fig.colorbar(image, ax=axes, label="Standardized MAE gain: S-HIST − F-TOK-D (positive favors F-TOK-D)")
    fig.suptitle("High-dispersion supported source-group mean; exploratory D1 seed diagnostic")
    fig.savefig(OUT / "figures/FIG_P2_SEED_TASK_JOINT_GAIN.png", dpi=180, facecolor="white")
    fig.savefig(OUT / "figures/FIG_P2_SEED_TASK_JOINT_GAIN.pdf", facecolor="white")
    plt.close(fig)

    def draw_trajectory(anchor_id: str, seed: int, horizon: int, filename: str, label: str) -> None:
        local = int(np.flatnonzero(dev_ids == anchor_id)[0])
        target_values = run_values[("F-TOK-D", seed, horizon)]["truth"][local]
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, layout="constrained")
        palette = {"S-HIST": "#0072B2", "F-TOK-D": "#D55E00", "F-TOK-S": "#009E73", "F-POOL-D": "#CC79A7"}
        styles = {"S-HIST": ("--", "s"), "F-TOK-D": ("-", "o"), "F-TOK-S": ("-.", "^"), "F-POOL-D": (":", "D")}
        for ax, joint, body in zip(axes, (0, 14), ("Hand joint 0", "Arm joint 14")):
            t = (np.arange(horizon) + 1) / 30.0
            ax.plot(t, target_values[:, joint], color="#222222", linewidth=1.5, label="Ground truth", marker="", linestyle="-")
            for model_name in MODELS:
                prediction = run_values[(model_name, seed, horizon)]["pred"][local, :, joint]
                linestyle, marker = styles[model_name]
                ax.plot(t, prediction, color=palette[model_name], linestyle=linestyle, marker=marker, markevery=max(1, horizon // 10), markersize=3, label=model_name)
            ax.set_ylabel(f"{body}\nnative command")
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Time after anchor (s)")
        axes[0].legend(ncol=3, fontsize=7, loc="best")
        fig.suptitle(f"{label}; {anchor_id}; seed={seed}, H={horizon}; dev diagnostic")
        fig.savefig(OUT / f"figures/{filename}.png", dpi=180, facecolor="white")
        fig.savefig(OUT / f"figures/{filename}.pdf", facecolor="white")
        plt.close(fig)

    draw_trajectory(fixed_id, 20260950, 30, "FIG_P2_FIXED_HASH_REPRESENTATIVE", "Preselected by anchor-ID hash before outcome inspection")
    draw_trajectory(worst_id, 20260951, 30, "FIG_P2_WORST_FTOKD_ARM_DIAGNOSTIC", "Post-hoc worst F-TOK-D Arm sMAE; failure diagnostic, not representative")

    completion = {
        "status": "PASS_RECOMPUTED_D1_AUDIT",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "accepted_cells_recomputed": len(run_values),
        "dev_episodes": len(set(dev_ep.tolist())),
        "dev_windows": len(dev_ids),
        "dev_source_groups": len(set(groups.tolist())),
        "source_group_table_sha256": sha256_file(RESULT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv"),
        "normalizer_sha256": sigma_hash,
        "video_normalizer_sha256": video_norm_hash,
        "history_timestamp_max_future_seconds": max_history_future,
        "all_predictions_id_joined": True,
        "all_checkpoint_hashes_match_export_sidecars": all(row["checkpoint_hash_matches_prediction"] for row in lineage_rows),
        "matched_seed_initializations": {str(seed): len({row["initial_model_hash"] for row in lineage_rows if row["seed"] == seed}) for seed in SEEDS},
        "all_runs_use_effective_batch64_epoch40": all(row["epochs"] == 40 and row["optimizer_steps"] == 11120 for row in lineage_rows),
        "same_window_order_rule_source_checked": matched_order_contract,
        "prediction_writer_id_rule_source_checked": id_join_contract,
        "max_absolute_difference_vs_locked_D1_table": agreement_max,
        "fixed_representative_anchor_id": fixed_id,
        "posthoc_worst_seed51_h30_anchor_id": worst_id,
        "figures": [
            "figures/FIG_P2_SEED_TASK_JOINT_GAIN.png",
            "figures/FIG_P2_FIXED_HASH_REPRESENTATIVE.png",
            "figures/FIG_P2_WORST_FTOKD_ARM_DIAGNOSTIC.png",
        ],
        "interpretation_status": "Exploratory; dev previously exposed; preserve all seeds; no confirmatory claims",
        "final_test_access": 0,
    }
    (OUT / "P2_SEED_AUDIT_SUMMARY.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": completion["status"],
        "cells": completion["accepted_cells_recomputed"],
        "episodes": completion["dev_episodes"],
        "windows": completion["dev_windows"],
        "max_abs_difference_vs_d1": completion["max_absolute_difference_vs_locked_D1_table"],
        "fixed_representative": fixed_id,
        "posthoc_worst": worst_id,
        "checkpoint_hashes_match": completion["all_checkpoint_hashes_match_export_sidecars"],
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
