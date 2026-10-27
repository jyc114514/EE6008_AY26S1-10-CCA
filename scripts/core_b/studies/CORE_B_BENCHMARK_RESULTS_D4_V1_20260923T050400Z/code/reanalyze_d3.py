"""Paired source-group analysis of the accepted D3 6x2x3 prediction grid."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from pathlibs import BENCH_ROOT, NEXT_ROOT, REPO_ROOT, V3_ROOT

SPLIT = REPO_ROOT / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260925
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
PRIMARY_PAIRS = (
    ("TS11-D", "TS10-D", "TS11_vs_TS10", "spatial_addition_given_temporal_preserved"),
    ("TS11-D", "TS01-D", "TS11_vs_TS01", "temporal_addition_given_spatial_preserved"),
    ("TS11-D", "TS11-S", "TS11D_vs_TS11S", "dynamic_vs_static_video"),
    ("TS11-D", "STATE-CTRL", "TS11_vs_STATE", "full_video_vs_state_only"),
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_group_map():
    table = pq.read_table(SPLIT, columns=["episode_index", "split", "video_sha256_fingerprint"])
    return {int(r["episode_index"]): str(r["video_sha256_fingerprint"] or f"episode:{r['episode_index']}") for r in table.to_pylist() if r["split"] in {"train", "validation"}}


def bootstrap_counts(n_groups: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n_groups, size=(BOOTSTRAPS, n_groups), dtype=np.int32)
    counts = np.zeros((BOOTSTRAPS, n_groups), dtype=np.uint16)
    row_ids = np.broadcast_to(np.arange(BOOTSTRAPS, dtype=np.int32)[:, None], draws.shape)
    np.add.at(counts, (row_ids, draws), 1)
    return counts


def estimate_ci(values: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    estimate = float(np.mean(values))
    reps = counts @ values / values.size
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def metric_windows(prediction: np.ndarray, truth: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for body, sl in (("Hand", slice(0, 14)), ("Arm", slice(14, 28)), ("Full28", slice(0, 28))):
        out[f"{body}|mae"] = np.mean(np.abs(prediction[:, :, sl] - truth[:, :, sl]) / sigma[None, None, sl], axis=(1, 2), dtype=np.float64)
        out[f"{body}|fd"] = np.mean(np.abs(np.diff(prediction[:, :, sl], axis=1) - np.diff(truth[:, :, sl], axis=1)), axis=(1, 2), dtype=np.float64)
    return out


def main() -> None:
    with np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        split_code = archive["split_code"]
        if set(np.unique(split_code).tolist()) != {0, 1}:
            raise ValueError("primary input archive unexpectedly includes non-development rows")
        dev = np.flatnonzero(split_code == 1)
        dev_ids = archive["anchor_id"][dev].astype("U64")
        dev_episode = archive["episode_index"][dev].astype(np.int64)
        dev_task = archive["original_task_index"][dev].astype(np.int64)
        truth_all = archive["targets_native"][dev, :60]
    sigma = np.asarray(json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
    groups = source_group_map()
    dev_group_ids = np.asarray([groups[int(ep)] for ep in dev_episode], dtype="U64")
    unique_groups = sorted(set(dev_group_ids.tolist()))
    group_index = {group: i for i, group in enumerate(unique_groups)}
    if len(unique_groups) != 603 or len(set(dev_episode.tolist())) != 603:
        raise ValueError("D3 primary development population is not 603 episode-distinct exact source groups")
    window_group_idx = np.asarray([group_index[g] for g in dev_group_ids], dtype=np.int32)
    counts = bootstrap_counts(len(unique_groups))

    accepted: dict[tuple[str, int, int], tuple[str, np.ndarray, str]] = {}
    for sidecar in sorted((BENCH_ROOT / "d3/predictions").glob("D3-*.json")):
        marker = json.loads(sidecar.read_text())
        if marker.get("status") != "PASS":
            continue
        config, seed, horizon = str(marker["config"]), int(marker["seed"]), int(marker["horizon_steps"])
        key = (config, seed, horizon)
        previous = accepted.get(key)
        if previous is not None and not ("RETRY" in str(marker["run_id"]) and "RETRY" not in previous[0]):
            continue
        pred_path = sidecar.with_suffix(".npz")
        with np.load(pred_path, allow_pickle=False) as bundle:
            ids = bundle["anchor_id"].astype("U64")
            predictions = bundle["predictions_native"]
            bundle_horizon = int(bundle["horizon_steps"])
        if set(ids.tolist()) != set(dev_ids.tolist()) or predictions.shape != (2322, horizon, 28) or bundle_horizon != horizon or not np.isfinite(predictions).all():
            raise ValueError(f"D3 prediction bundle join/shape/finiteness failed: {pred_path.name}")
        lookup = {str(anchor_id): i for i, anchor_id in enumerate(ids)}
        aligned = predictions[np.asarray([lookup[str(anchor_id)] for anchor_id in dev_ids])]
        accepted[key] = (str(marker["run_id"]), aligned, str(pred_path))
    expected = {(c, s, h) for c in CONFIGS for s in (20261001, 20261002, 20261003) for h in (30, 60)}
    if set(accepted) != expected or len(accepted) != 36:
        raise ValueError(f"accepted D3 grid differs from exact 6x2x3: {len(accepted)} cells")

    group_values: dict[tuple[str, int, int, str, str], np.ndarray] = {}
    seed_rows, task_rows, bundle_rows = [], [], []
    for (config, seed, horizon), (run_id, prediction, path) in sorted(accepted.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
        metrics = metric_windows(prediction, truth_all[:, :horizon], sigma)
        for metric_key, window_values in metrics.items():
            body, metric = metric_key.split("|")
            by_group = np.empty(len(unique_groups), dtype=np.float64)
            for gi in range(len(unique_groups)):
                selected = window_group_idx == gi
                by_group[gi] = float(np.mean(window_values[selected]))
            group_values[(config, seed, horizon, body, metric)] = by_group
            estimate, low, high = estimate_ci(by_group, counts)
            seed_rows.append({"config": config, "seed": seed, "horizon_steps": horizon, "body": body, "metric": metric, "estimate": estimate, "ci95_low": low, "ci95_high": high, "episodes": 603, "source_groups": len(unique_groups), "windows": len(dev), "bootstrap_unit": "exact video_sha256_fingerprint source group", "final_test_access": 0})
            for task in sorted(set(dev_task.tolist())):
                task_groups = sorted(set(dev_group_ids[dev_task == task].tolist()))
                idx = np.asarray([group_index[g] for g in task_groups], dtype=np.int32)
                task_rows.append({"config": config, "seed": seed, "horizon_steps": horizon, "task": int(task), "body": body, "metric": metric, "estimate": float(np.mean(by_group[idx])) if len(idx) else None, "episodes": len(idx), "source_groups": len(idx), "status": "descriptive_task_layer_no_taskwise_CI", "final_test_access": 0})
        bundle_rows.append({"config": config, "seed": seed, "horizon_steps": horizon, "run_id": run_id, "prediction_bundle": path, "episodes": 603, "source_groups": len(unique_groups), "windows": len(dev), "bundle_rows": 2322, "bundle_status": "PASS_exact_dev_ID_shape_finite", "final_test_access": 0})

    contrast_rows = []
    for horizon in (30, 60):
        for seed in (20261001, 20261002, 20261003):
            for candidate, control, contrast, direction in PRIMARY_PAIRS:
                for body in ("Hand", "Arm", "Full28"):
                    for metric in ("mae", "fd"):
                        diff = group_values[(candidate, seed, horizon, body, metric)] - group_values[(control, seed, horizon, body, metric)]
                        est, low, high = estimate_ci(diff, counts)
                        contrast_rows.append({"contrast": contrast, "interpretation": direction, "seed": seed, "horizon_steps": horizon, "body": body, "metric": metric, "estimand": "candidate_minus_control; negative favors candidate", "estimate": est, "ci95_low": low, "ci95_high": high, "episodes": 603, "source_groups": len(unique_groups), "final_test_access": 0})
            # Paired 2x2 factorial interaction and marginal temporal/spatial effects.
            for body in ("Hand", "Arm", "Full28"):
                for metric in ("mae", "fd"):
                    l00 = group_values[("TS00-D", seed, horizon, body, metric)]
                    l10 = group_values[("TS10-D", seed, horizon, body, metric)]
                    l01 = group_values[("TS01-D", seed, horizon, body, metric)]
                    l11 = group_values[("TS11-D", seed, horizon, body, metric)]
                    derived = (
                        ("factorial_interaction_TS11_TS10_TS01_TS00", (l11 - l10) - (l01 - l00), "negative favors synergistic reduction in loss"),
                        ("spatial_marginal_main_effect", ((l01 - l00) + (l11 - l10)) / 2.0, "negative favors spatial-bin preservation averaged over temporal factor"),
                        ("temporal_marginal_main_effect", ((l10 - l00) + (l11 - l01)) / 2.0, "negative favors temporal preservation averaged over spatial factor"),
                    )
                    for label, effect, interpretation in derived:
                        est, low, high = estimate_ci(effect, counts)
                        contrast_rows.append({"contrast": label, "interpretation": interpretation, "seed": seed, "horizon_steps": horizon, "body": body, "metric": metric, "estimand": "paired source-group factorial contrast; negative favors lower loss", "estimate": est, "ci95_low": low, "ci95_high": high, "episodes": 603, "source_groups": len(unique_groups), "final_test_access": 0})
    write_csv(NEXT_ROOT / "results/D3_SOURCE_GROUP_RESULTS.csv", seed_rows)
    write_csv(NEXT_ROOT / "results/D3_SOURCE_GROUP_CONTRASTS.csv", contrast_rows)
    write_csv(NEXT_ROOT / "results/D3_TASK_DESCRIPTIVE_RESULTS.csv", task_rows)
    write_csv(NEXT_ROOT / "results/D3_ACCEPTED_BUNDLES.csv", bundle_rows)

    # Reconcile full-grid point values against the sealed analysis table.
    source_results = read_csv(BENCH_ROOT / "d3/analysis/D3_RESULTS.csv")
    source_index = {(r["config"], int(r["seed"]), int(r["horizon_steps"]), r["body"], r["metric"]): float(r["estimate"]) for r in source_results}
    reconciliation = []
    for row in seed_rows:
        key = (row["config"], row["seed"], row["horizon_steps"], row["body"], row["metric"])
        source_value = source_index.get(key)
        delta = float(row["estimate"] - source_value) if source_value is not None else None
        reconciliation.append({**{k: row[k] for k in ("config", "seed", "horizon_steps", "body", "metric")}, "source_estimate": source_value, "recomputed_estimate": row["estimate"], "delta": delta, "status": "PASS" if delta is not None and abs(delta) <= 1e-7 else "MISMATCH_OR_MISSING", "final_test_access": 0})
    write_csv(NEXT_ROOT / "results/D3_SOURCE_RECONCILIATION.csv", reconciliation)
    summary = {"status": "D3_SOURCE_GROUP_ANALYSIS_COMPLETE", "accepted_bundles": len(bundle_rows), "result_rows": len(seed_rows), "contrast_rows": len(contrast_rows), "task_descriptive_rows": len(task_rows), "source_groups": len(unique_groups), "episodes": 603, "windows": len(dev), "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED, "source_point_reconciliation_pass": sum(r["status"] == "PASS" for r in reconciliation), "source_point_reconciliation_total": len(reconciliation), "training_objective_conformance": "NONCONFORMANT_AS_IMPLEMENTED; actual-run outputs are exploratory and no retraining was performed", "final_test_access": 0}
    (NEXT_ROOT / "results/D3_SOURCE_GROUP_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
