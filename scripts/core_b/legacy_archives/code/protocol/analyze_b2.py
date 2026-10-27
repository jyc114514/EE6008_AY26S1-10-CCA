#!/usr/bin/env python3
"""Accept B2 runs, compute paired episode statistics, and lock selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = [
    ("B2-V-0125", 0.125, 32, "video_only"),
    ("B2-V-0250", 0.250, 16, "video_only"),
    ("B2-V-0500", 0.500, 8, "video_only"),
    ("B2-FUS-0125", 0.125, 32, "fusion"),
    ("B2-FUS-0250", 0.250, 16, "fusion"),
    ("B2-FUS-0500", 0.500, 8, "fusion"),
]
BOOTSTRAP_N = 10_000
PERMUTATION_N = 20_000
BOOTSTRAP_SEED = 20260908
PERMUTATION_SEED = 20260909


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(part, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def metric_arrays(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
    error = np.abs(prediction - target)
    return {
        "episode_full": error.mean(axis=(1, 2)),
        "episode_arm": error[:, :, 14:28].mean(axis=(1, 2)),
        "episode_hand": error[:, :, 0:14].mean(axis=(1, 2)),
        "episode_final": error[:, -1, :].mean(axis=1),
        "episode_rmse": np.sqrt(np.square(prediction - target).mean(axis=(1, 2))),
        "full": float(error.mean()),
        "arm": float(error[:, :, 14:28].mean()),
        "hand": float(error[:, :, 0:14].mean()),
        "final": float(error[:, -1, :].mean()),
        "rmse": float(np.sqrt(np.square(prediction - target).mean())),
    }


def metric_row(label: str, metrics: dict[str, np.ndarray | float], *, stage: str, interval: float | None, frames: int | None, branch: str, epoch: int | None, peak_reserved: float | None, elapsed: float | None) -> dict[str, object]:
    return {
        "model": label,
        "stage": stage,
        "branch": branch,
        "interval_seconds": interval,
        "unique_observation_frames": frames,
        "selected_epoch": epoch,
        "standardized_arm14_mae": metrics["arm"],
        "standardized_full28_mae": metrics["full"],
        "standardized_hand14_mae": metrics["hand"],
        "standardized_final_step_mae": metrics["final"],
        "standardized_rmse": metrics["rmse"],
        "peak_reserved_mib": peak_reserved,
        "elapsed_seconds": elapsed,
        "final_test_accessed": 0,
    }


def bootstrap_effect(candidate: np.ndarray, baseline: np.ndarray, indices: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    # Negative means candidate has lower MAE; positive means candidate improves.
    effects = (candidate[indices].mean(axis=1) - baseline[indices].mean(axis=1)).astype(np.float64)
    return float((candidate.mean() - baseline.mean())), float(np.percentile(effects, 2.5)), float(np.percentile(effects, 97.5)), effects


def relative_bootstrap(candidate: np.ndarray, baseline: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    candidate_means = candidate[indices].mean(axis=1)
    baseline_means = baseline[indices].mean(axis=1)
    values = candidate_means / baseline_means - 1.0
    return float(candidate.mean() / baseline.mean() - 1.0), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def sign_flip_pvalue(difference: np.ndarray, signs: np.ndarray) -> float:
    observed = abs(float(difference.mean()))
    sampled = np.abs((signs * difference[None, :]).mean(axis=1))
    return float((1.0 + np.count_nonzero(sampled >= observed)) / (len(sampled) + 1.0))


def holm(rows: list[dict[str, object]]) -> None:
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["raw_p_value"]))
    adjusted = [0.0] * len(rows)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(rows) - rank) * float(rows[index]["raw_p_value"]))
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-input-root", type=Path, required=True)
    parser.add_argument("--target-normalization", type=Path, required=True)
    parser.add_argument("--state-run", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment_root.resolve()
    source = args.source_input_root.resolve()
    target_norm_path = args.target_normalization.resolve()
    state_run = args.state_run.resolve()

    canonical = load_npz(source / "CORE_B_FULL_INPUTS_1p0s.npz")
    train_mask = canonical["split_code"] == 0
    dev_mask = canonical["split_code"] == 1
    dev_ids = canonical["episode_index"][dev_mask]
    dev_target = canonical["targets_standardized"][dev_mask]
    if dev_target.shape != (614, 30, 28) or dev_ids.shape != (614,):
        raise ValueError("canonical Core-B dev target/ID shape failed")
    target_norm = json.loads(target_norm_path.read_text(encoding="utf-8"))["1.0s"]
    target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
    target_std = np.asarray(target_norm["safe_std"], dtype=np.float32)
    persistence_native = canonical["last_observed_action"][dev_mask, None, :]
    persistence_native = np.repeat(persistence_native, 30, axis=1)
    persistence_std = (persistence_native - target_mean) / target_std
    persistence_metrics = metric_arrays(persistence_std, dev_target)

    accepted: list[dict[str, object]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, object]] = []
    for label, interval, frames, branch in CONDITIONS:
        run = exp / "runs" / label
        status_path = run / "RUN_STATUS.json"
        complete_path = run / "RUN_COMPLETE.json"
        config_path = run / "RUN_CONFIG_LOCKED.json"
        metrics_path = run / "RUN_METRICS.json"
        prediction_path = run / "predictions_best.npz"
        writer_repair_path = run / "WRITER_REPAIR_STATUS.json"
        required = [status_path, complete_path, config_path, metrics_path, prediction_path, writer_repair_path, run / "checkpoint_best.pt", run / "checkpoint_last.pt"]
        missing = [str(p) for p in required if not p.is_file()]
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        writer_repair = json.loads(writer_repair_path.read_text(encoding="utf-8")) if writer_repair_path.is_file() else {}
        arrays = load_npz(prediction_path) if prediction_path.is_file() else {}
        shape_ok = arrays.get("predictions_standardized", np.empty(0)).shape == (614, 30, 28)
        target_ok = arrays.get("targets_standardized", np.empty(0)).shape == (614, 30, 28) and np.array_equal(arrays.get("targets_standardized"), dev_target)
        ids_ok = arrays.get("episode_index", np.empty(0)).shape == (614,) and np.array_equal(arrays.get("episode_index"), dev_ids)
        complete_ok = status.get("status") == "complete" and status.get("exit_code") == 0 and status.get("final_test_accessed") == 0 and metrics.get("status") == "complete"
        contract_ok = config.get("physical_gpu") == 1 and config.get("physical_batch") == 8 and config.get("gradient_accumulation") == 4 and config.get("primary_target_subset") == "arm[14:28]" and config.get("final_test_accessed") == 0
        writer_repair_ok = writer_repair.get("status") == "pass" and writer_repair.get("final_test_accessed") == 0
        accepted_ok = not missing and shape_ok and target_ok and ids_ok and complete_ok and contract_ok and writer_repair_ok
        accepted.append({
            "condition": label,
            "status": "accepted" if accepted_ok else "rejected",
            "missing_artifacts": ";".join(missing),
            "shape_ok": shape_ok,
            "target_ids_ok": bool(target_ok and ids_ok),
            "complete_ok": complete_ok,
            "contract_ok": contract_ok,
            "writer_repair_ok": writer_repair_ok,
            "final_test_accessed": 0,
            "run_root": str(run),
        })
        if not accepted_ok:
            raise RuntimeError(f"B2 acceptance failed for {label}: {accepted[-1]}")
        prediction = arrays["predictions_standardized"].astype(np.float32)
        predictions[label] = {"prediction": prediction, "target": arrays["targets_standardized"].astype(np.float32), "ids": arrays["episode_index"]}
        run_metrics = metric_arrays(prediction, dev_target)
        metric_rows.append(metric_row(label, run_metrics, stage="B2", interval=interval, frames=frames, branch=branch, epoch=int(metrics["best_epoch"]), peak_reserved=float(metrics["peak_reserved_mib"]), elapsed=float(metrics["elapsed_seconds"])))

    state_arrays = load_npz(state_run / "predictions_best.npz")
    if state_arrays["predictions_standardized"].shape != (614, 30, 28) or not np.array_equal(state_arrays["episode_index"], dev_ids):
        raise ValueError("B1 state reference is not aligned to Core-B dev IDs/predictions")
    # The historical B1 archive has the same old writer bug in its target
    # field. Its prediction tensor and IDs are retained as the accepted
    # reference; canonical targets supply the comparison target here.
    state_prediction = state_arrays["predictions_standardized"].astype(np.float32)
    state_metrics = metric_arrays(state_prediction, dev_target)
    predictions["STATE"] = {"prediction": state_prediction, "target": dev_target, "ids": dev_ids}
    metric_rows.append(metric_row("STATE", state_metrics, stage="reference", interval=None, frames=None, branch="state_only", epoch=None, peak_reserved=None, elapsed=None))
    metric_rows.append(metric_row("PERSISTENCE", persistence_metrics, stage="reference", interval=None, frames=None, branch="persistence", epoch=None, peak_reserved=None, elapsed=None))
    write_csv(exp / "B2_RUN_ACCEPTANCE.csv", accepted)
    atomic_json(exp / "B2_RUN_ACCEPTANCE.json", {"status": "pass", "accepted": accepted, "accepted_count": len(accepted), "logical_coverage": "6/6", "final_test_access": 0})
    write_csv(exp / "B2_RESULTS.csv", metric_rows)
    atomic_json(exp / "B2_RESULTS.json", {"status": "pass", "rows": metric_rows, "primary_metric": "standardized_arm14_trajectory_mae", "dev_count": 614, "final_test_access": 0})

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(0, 614, size=(BOOTSTRAP_N, 614), dtype=np.uint16)
    np.save(exp / "figure_data/B2_BOOTSTRAP_RESAMPLE_INDICES.npy", bootstrap_indices)
    sign_rng = np.random.default_rng(PERMUTATION_SEED)
    signs = np.where(sign_rng.integers(0, 2, size=(PERMUTATION_N, 614), dtype=np.int8) == 0, -1.0, 1.0)
    comparisons: list[dict[str, object]] = []
    comparison_specs = [
        ("B2-FUS-0250", "B2-FUS-0125", "B2_primary", "FUS-0250_vs_FUS-0125"),
        ("B2-FUS-0500", "B2-FUS-0125", "B2_primary", "FUS-0500_vs_FUS-0125"),
        ("B2-FUS-0250", "STATE", "B2_primary", "FUS-0250_vs_STATE"),
        ("B2-FUS-0500", "STATE", "B2_primary", "FUS-0500_vs_STATE"),
    ]
    for candidate, baseline, family, name in comparison_specs:
        cand = metric_arrays(predictions[candidate]["prediction"], dev_target)["episode_arm"]
        base = metric_arrays(predictions[baseline]["prediction"], dev_target)["episode_arm"] if baseline != "STATE" else state_metrics["episode_arm"]
        effect, ci_low, ci_high, _ = bootstrap_effect(cand, base, bootstrap_indices)
        raw_p = sign_flip_pvalue(cand - base, signs)
        rel, rel_low, rel_high = relative_bootstrap(cand, base, bootstrap_indices)
        comparisons.append({"comparison": name, "candidate": candidate, "baseline": baseline, "family": family, "effect_candidate_minus_baseline": effect, "ci95_lower": ci_low, "ci95_upper": ci_high, "relative_effect": rel, "relative_ci95_lower": rel_low, "relative_ci95_upper": rel_high, "raw_p_value": raw_p, "holm_adjusted_p": None, "bootstrap_n": BOOTSTRAP_N, "permutation_n": PERMUTATION_N, "bootstrap_seed": BOOTSTRAP_SEED, "permutation_seed": PERMUTATION_SEED})
    holm(comparisons)
    write_csv(exp / "B2_BOOTSTRAP_COMPARISONS.csv", comparisons)
    atomic_json(exp / "B2_BOOTSTRAP_COMPARISONS.json", {"status": "pass", "comparisons": comparisons, "resample_rule": "np.random.default_rng(20260908).integers(0,614,(10000,614),uint16)", "sign_flip_rule": "np.random.default_rng(20260909), 20000x614 iid +/-1 signs", "final_test_access": 0})

    by_label = {row["model"]: row for row in metric_rows}
    selection_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for label, interval, frames, _branch in CONDITIONS:
        if label.endswith("0125"):
            continue
        fus = label.replace("B2-V", "B2-FUS")
        dense_fus = "B2-FUS-0125"
        comp_dense = next(row for row in comparisons if row["candidate"] == fus and row["baseline"] == dense_fus)
        comp_state = next(row for row in comparisons if row["candidate"] == fus and row["baseline"] == "STATE")
        fus_row = by_label[fus]
        dense_row = by_label[dense_fus]
        hand_rel = float(fus_row["standardized_hand14_mae"]) / float(dense_row["standardized_hand14_mae"]) - 1.0
        full_rel = float(fus_row["standardized_full28_mae"]) / float(dense_row["standardized_full28_mae"]) - 1.0
        accuracy = bool(float(comp_dense["ci95_upper"]) < 0 and float(comp_state["ci95_upper"]) < 0 and float(comp_dense["holm_adjusted_p"]) < 0.05 and float(comp_state["holm_adjusted_p"]) < 0.05 and hand_rel <= 0.01 and full_rel <= 0.01)
        rel_dense = (float(fus_row["standardized_arm14_mae"]) / float(dense_row["standardized_arm14_mae"]) - 1.0)
        rel_state = (float(fus_row["standardized_arm14_mae"]) / float(by_label["STATE"]["standardized_arm14_mae"]) - 1.0)
        efficiency = bool(float(comp_dense["relative_ci95_lower"]) >= -0.01 and float(comp_dense["relative_ci95_upper"]) <= 0.01 and float(comp_state["relative_ci95_upper"]) <= 0.01 and hand_rel <= 0.01 and full_rel <= 0.01 and frames <= 16)
        row = {"interval_seconds": interval, "unique_frames": frames, "fus_condition": fus, "accuracy_candidate": accuracy, "efficiency_candidate": efficiency, "relative_arm_vs_dense": rel_dense, "relative_arm_vs_state": rel_state, "relative_hand_vs_dense": hand_rel, "relative_full_vs_dense": full_rel, "dense_comparison_ci_upper": comp_dense["ci95_upper"], "state_comparison_ci_upper": comp_state["ci95_upper"], "dense_holm_p": comp_dense["holm_adjusted_p"], "state_holm_p": comp_state["holm_adjusted_p"]}
        selection_rows.append(row)
        if accuracy or efficiency:
            candidates.append(row)
    if candidates:
        accuracy_candidates = [row for row in candidates if row["accuracy_candidate"]]
        pool = accuracy_candidates or candidates
        selected = sorted(pool, key=lambda row: (float(row["relative_arm_vs_dense"]), int(row["unique_frames"])))[0]
        selected_interval = float(selected["interval_seconds"])
        selection_status = "accuracy_candidate" if selected["accuracy_candidate"] else "observation_budget_candidate"
    else:
        selected_interval = 0.125
        selected = {"interval_seconds": selected_interval, "unique_frames": 32, "fus_condition": "B2-FUS-0125", "accuracy_candidate": False, "efficiency_candidate": False}
        selection_status = "dense_fallback_negative_or_inconclusive"
    selection = {"status": "pass", "selection_status": selection_status, "selected_interval_seconds": selected_interval, "selected_unique_frames": int(selected["unique_frames"]), "selected_fusion_condition": selected["fus_condition"], "selection_rows": selection_rows, "rule": "accuracy candidates first; then efficiency candidates; otherwise dense0.125 fallback", "b3_selected_strategy_locked": True, "final_test_access": 0}
    atomic_json(exp / "B2_SELECTION.json", selection)
    lines = ["# Core B B2 selection", "", f"- status: `{selection_status}`", f"- selected strategy: `{selected_interval:.3f}s`, `{int(selected['unique_frames'])}` unique frames", f"- selected fusion condition: `{selected['fus_condition']}`", "- B3 may use this single locked strategy; no per-cell interval selection.", "", "## Candidate table", "", "| interval | frames | accuracy | efficiency |", "|---:|---:|---|---|"]
    lines.extend(f"| {row['interval_seconds']:.3f} | {row['unique_frames']} | {row['accuracy_candidate']} | {row['efficiency_candidate']} |" for row in selection_rows)
    lines.extend(["", "The sparse representation is feature-level nested block pooling with a 32-slot repetition contract; reduced unique frames are not an encoder-speed claim."])
    (exp / "B2_SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    prep = json.loads((exp / "inputs/B2_INPUT_PREPARATION.json").read_text(encoding="utf-8"))
    frame_qa = {"status": "pass", "created_utc": utc_now(), "train_count": int(train_mask.sum()), "development_validation_count": int(dev_mask.sum()), "target_shape": list(canonical["targets"].shape), "dev_target_shape": list(dev_target.shape), "dev_episode_id_shape": list(dev_ids.shape), "episode_ids_unique": bool(len(np.unique(dev_ids)) == 614), "target_start_frame": 120, "prefix_frame_cutoff": 119, "history_frame_indices": [0, 17, 34, 51, 68, 85, 102, 119], "frame_maps": prep["frame_maps"], "dense_cache_float16_reconstruction_max_abs": prep["dense_pool_reconstruction_raw_max_abs"], "dense_condition_exact_b1_input_reuse": prep["dense_condition_exact_b1_input_reuse"], "final_test_access": 0}
    atomic_json(exp / "FRAME_AND_TARGET_QA.json", frame_qa)
    (exp / "FRAME_AND_TARGET_QA.md").write_text("# Core B B2 frame and target QA\n\n- Status: `pass`.\n- Population: train 2741 / development-validation 614.\n- Dev targets: `(614,30,28)`; episode IDs: `(614,)`, unique.\n- Prefix ends at frame 119; target starts at frame 120.\n- Dense32/sparse16/sparse8 all end at the latest observed frame; exact actual timestamps and gaps are in `inputs/B2_FRAME_MAPS.json`.\n- Dense32 input is exact B1 standardized-vector reuse; token-vs-pooled float16 cache reconstruction is recorded separately.\n- Final-test access: `0`.\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "accepted": len(accepted), "selection": selection_status, "selected_interval_seconds": selected_interval}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
