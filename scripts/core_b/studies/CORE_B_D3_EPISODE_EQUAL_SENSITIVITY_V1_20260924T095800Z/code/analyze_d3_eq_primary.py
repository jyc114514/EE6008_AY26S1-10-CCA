"""CPU-only common scoring for OLD_D3_ACTUAL and D3_EQ_V1 primary dev predictions."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OLD_PRED_ROOT = BENCH_ROOT / "d3/predictions"
NEW_PRED_ROOT = EQ_ROOT / "d3/predictions"
OUT = EQ_ROOT / "d3/analysis"
SEEDS = (20261001, 20261002, 20261003)
HORIZONS = (30, 60)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260924
BODIES = {"Hand": slice(0, 14), "Arm": slice(14, 28), "Full28": slice(0, 28)}
PRIMARY_CONTRASTS = (
    ("TS11-D", "TS10-D", "TS11D_vs_TS10D"),
    ("TS11-D", "TS01-D", "TS11D_vs_TS01D"),
    ("TS11-D", "TS11-S", "TS11D_vs_TS11S"),
    ("TS11-D", "STATE-CTRL", "TS11D_vs_STATE"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing empty result CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_indices(n_groups: int, horizon: int) -> np.ndarray:
    # Shared group resamples are reused across all matched conditions/contrasts.
    rng = np.random.default_rng(BOOTSTRAP_SEED + horizon)
    return rng.integers(0, n_groups, size=(BOOTSTRAPS, n_groups), dtype=np.int32)


def summarize_groups(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be a finite group-level vector")
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    p_low = (int(np.count_nonzero(samples <= 0.0)) + 1) / (len(samples) + 1)
    p_high = (int(np.count_nonzero(samples >= 0.0)) + 1) / (len(samples) + 1)
    p_two_sided = min(1.0, 2.0 * min(p_low, p_high))
    return float(values.mean()), float(low), float(high), float(p_two_sided)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pvalue))
        adjusted[key] = running
    return adjusted


def reduce_to_groups(values: np.ndarray, episode_ids: np.ndarray, ordered_groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episode_ids, dtype=np.int64)
    grouped = np.empty(len(ordered_groups), dtype=np.float64)
    for index, episode in enumerate(ordered_groups):
        selected = values[episodes == int(episode)]
        if not len(selected):
            raise AssertionError(f"dev episode group missing from score vector: {episode}")
        grouped[index] = selected.mean()
    return grouped


def per_anchor_metrics(pred: np.ndarray, truth: np.ndarray, sigma: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != 28:
        raise ValueError("prediction/target must share [anchor,horizon,28] shape")
    result: dict[tuple[str, str], np.ndarray] = {}
    for body, sl in BODIES.items():
        error = np.abs(pred[:, :, sl] - truth[:, :, sl]) / sigma[None, None, sl]
        result[(body, "standardized_mae")] = error.mean(axis=(1, 2), dtype=np.float64)
        fd_error = np.abs(np.diff(pred[:, :, sl], axis=1) - np.diff(truth[:, :, sl], axis=1))
        result[(body, "native_future_internal_fd")] = fd_error.mean(axis=(1, 2), dtype=np.float64)
    return result


def accepted_old_predictions() -> dict[tuple[str, int, int], tuple[dict[str, Any], Path]]:
    selected: dict[tuple[str, int, int], tuple[dict[str, Any], Path]] = {}
    for sidecar in sorted(OLD_PRED_ROOT.glob("D3-*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            continue
        key = (str(record["config"]), int(record["seed"]), int(record["horizon_steps"]))
        prior = selected.get(key)
        if prior is None or ("RETRY" in str(record["run_id"]) and "RETRY" not in str(prior[0]["run_id"])):
            selected[key] = (record, sidecar.with_suffix(".npz"))
    if len(selected) != 36:
        raise AssertionError(f"expected 36 accepted historical D3 prediction slots, found {len(selected)}")
    return selected


def load_predictions(
    prediction_root: Path,
    expected_ids: np.ndarray,
    *,
    is_new: bool,
) -> dict[tuple[str, int, int], np.ndarray]:
    selected: dict[tuple[str, int, int], np.ndarray] = {}
    for sidecar in sorted(prediction_root.glob("*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            continue
        npz_path = sidecar.with_suffix(".npz")
        if not npz_path.is_file() or sha256_file(npz_path) != record.get("prediction_sha256"):
            raise AssertionError(f"prediction file/checksum mismatch: {npz_path.name}")
        with np.load(npz_path, allow_pickle=False) as archive:
            ids = archive["anchor_id"].astype("U64")
            pred = archive["predictions_native"].astype(np.float32, copy=True)
            config = str(archive["config"].item())
            seed = int(archive["seed"].item())
            horizon = int(archive["horizon_steps"])
        key = (config, seed, horizon)
        if len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(expected_ids.tolist()):
            raise AssertionError(f"prediction IDs differ from the locked primary dev set: {npz_path.name}")
        if pred.shape != (len(expected_ids), horizon, 28) or not np.isfinite(pred).all():
            raise AssertionError(f"prediction shape/finite check failed: {npz_path.name}")
        order = {str(value): index for index, value in enumerate(ids)}
        pred = pred[np.asarray([order[str(value)] for value in expected_ids], dtype=np.int64)]
        if key in selected:
            raise AssertionError(f"duplicate accepted prediction slot in {prediction_root}: {key}")
        selected[key] = pred
    if len(selected) != 36:
        raise AssertionError(f"expected 36 accepted prediction bundles in {prediction_root}, found {len(selected)}")
    return selected


def build_group_scores(predictions: dict[tuple[str, int, int], np.ndarray], targets: np.ndarray, sigma: np.ndarray, episode_ids: np.ndarray, groups: np.ndarray) -> dict[tuple[str, int, int, str, str], np.ndarray]:
    ordered_groups = np.sort(np.unique(groups))
    scores: dict[tuple[str, int, int, str, str], np.ndarray] = {}
    for (config, seed, horizon), pred in predictions.items():
        metrics = per_anchor_metrics(pred, targets[:, :horizon], sigma)
        for (body, metric), values in metrics.items():
            scores[(config, seed, horizon, body, metric)] = reduce_to_groups(values, episode_ids, ordered_groups)
    return scores


def main() -> int:
    protocol = json.loads((EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED" or protocol.get("final_test_access") != 0:
        raise AssertionError("D3-EQ protocol is not locked/final-test denied")
    # This evaluator is a post-training step and is the first code path allowed
    # to access dev target values. The V3 archive contains train+dev only.
    archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    split = archive["split_code"].astype(np.int8, copy=True)
    if set(np.unique(split).tolist()) != {0, 1}:
        raise AssertionError("V3 primary input archive contains an unexpected split code")
    dev_idx = np.flatnonzero(split == 1)
    ids = archive["anchor_id"][dev_idx].astype("U64", copy=True)
    episodes = archive["episode_index"][dev_idx].astype(np.int64, copy=True)
    targets = archive["targets_native"][dev_idx].astype(np.float32, copy=True)
    if len(ids) != 2322 or len(set(ids.tolist())) != 2322 or len(np.unique(episodes)) != 603:
        raise AssertionError("primary dev row/group counts differ from locked 2,322/603")
    sigma = np.asarray(json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))["target_sigma"], dtype=np.float32)
    if targets.shape != (2322, 60, 28) or sigma.shape != (28,) or not np.isfinite(targets).all():
        raise AssertionError("primary dev target shape/finite check failed")
    new_predictions = load_predictions(NEW_PRED_ROOT, ids, is_new=True)
    old_predictions = load_predictions(OLD_PRED_ROOT, ids, is_new=False)
    expected_slots = {(config, seed, horizon) for config in CONFIGS for seed in SEEDS for horizon in HORIZONS}
    if set(new_predictions) != expected_slots or set(old_predictions) != expected_slots:
        raise AssertionError("old/new prediction matrices do not match the locked 36-cell design")

    ordered_groups = np.sort(np.unique(episodes))
    index_by_horizon = {horizon: bootstrap_indices(len(ordered_groups), horizon) for horizon in HORIZONS}
    new_scores = build_group_scores(new_predictions, targets, sigma, episodes, ordered_groups)
    old_scores = build_group_scores(old_predictions, targets, sigma, episodes, ordered_groups)

    model_rows = []
    for label, scores in (("OLD_D3_ACTUAL", old_scores), ("D3_EQ_V1", new_scores)):
        for (config, seed, horizon, body, metric), group_values in sorted(scores.items()):
            estimate, low, high, _ = summarize_groups(group_values, index_by_horizon[horizon])
            model_rows.append({
                "study": label,
                "config": config,
                "seed": seed,
                "horizon_steps": horizon,
                "body": body,
                "metric": metric,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "anchors": len(ids),
                "source_episode_groups": len(ordered_groups),
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED + horizon,
                "bootstrap_unit": "episode_index; windows averaged equally within episode",
                "final_test_access": 0,
            })
    write_csv(OUT / "D3_EQ_RESULTS_BY_SEED.csv", model_rows)

    seed_contrast_rows = []
    pooled_primary = []
    pooled_p = {}
    for horizon in HORIZONS:
        bidx = index_by_horizon[horizon]
        for candidate, control, contrast in PRIMARY_CONTRASTS:
            for body in ("Hand", "Arm", "Full28"):
                for metric in ("standardized_mae", "native_future_internal_fd"):
                    seed_deltas = {}
                    for seed in SEEDS:
                        delta = new_scores[(candidate, seed, horizon, body, metric)] - new_scores[(control, seed, horizon, body, metric)]
                        seed_deltas[seed] = delta
                        estimate, low, high, p = summarize_groups(delta, bidx)
                        seed_contrast_rows.append({
                            "contrast": contrast,
                            "candidate": candidate,
                            "control": control,
                            "seed": seed,
                            "horizon_steps": horizon,
                            "body": body,
                            "metric": metric,
                            "estimand": "candidate_minus_control; negative favors lower loss",
                            "estimate": estimate,
                            "ci95_low": low,
                            "ci95_high": high,
                            "p_unadjusted_bootstrap": p,
                            "source_episode_groups": len(ordered_groups),
                            "bootstrap_replicates": BOOTSTRAPS,
                            "bootstrap_seed": BOOTSTRAP_SEED + horizon,
                            "final_test_access": 0,
                        })
                    pooled_delta = np.mean(np.stack([seed_deltas[seed] for seed in SEEDS]), axis=0)
                    estimate, low, high, p = summarize_groups(pooled_delta, bidx)
                    key = f"{contrast}|H{horizon}|{body}"
                    row = {
                        "contrast": contrast,
                        "candidate": candidate,
                        "control": control,
                        "horizon_steps": horizon,
                        "body": body,
                        "metric": metric,
                        "estimate_three_seed_mean": estimate,
                        "ci95_low_descriptive": low,
                        "ci95_high_descriptive": high,
                        "p_unadjusted_bootstrap": p,
                        "seed_estimates": json.dumps({str(seed): float(seed_deltas[seed].mean()) for seed in SEEDS}, sort_keys=True),
                        "seed_mean": float(np.mean([seed_deltas[seed].mean() for seed in SEEDS])),
                        "seed_sd": float(np.std([seed_deltas[seed].mean() for seed in SEEDS], ddof=1)),
                        "source_episode_groups": len(ordered_groups),
                        "bootstrap_replicates": BOOTSTRAPS,
                        "bootstrap_unit": "paired episode groups; seed-averaged group contrast, descriptive across seeds",
                        "final_test_access": 0,
                    }
                    pooled_primary.append(row)
                    if metric == "standardized_mae" and body in ("Hand", "Arm"):
                        pooled_p[key] = p
    write_csv(OUT / "D3_EQ_CONTRASTS_BY_SEED.csv", seed_contrast_rows)
    holm = holm_adjust(pooled_p)
    for row in pooled_primary:
        if row["metric"] == "standardized_mae" and row["body"] in ("Hand", "Arm"):
            key = f"{row['contrast']}|H{row['horizon_steps']}|{row['body']}"
            row["p_holm_16_primary_family"] = holm[key]
        else:
            row["p_holm_16_primary_family"] = "not_in_primary_family"
    write_csv(OUT / "D3_EQ_PRIMARY_POOLED.csv", pooled_primary)

    factorial_rows = []
    for horizon in HORIZONS:
        bidx = index_by_horizon[horizon]
        for body in BODIES:
            for metric in ("standardized_mae", "native_future_internal_fd"):
                per_seed = {}
                for seed in SEEDS:
                    l00 = new_scores[("TS00-D", seed, horizon, body, metric)]
                    l10 = new_scores[("TS10-D", seed, horizon, body, metric)]
                    l01 = new_scores[("TS01-D", seed, horizon, body, metric)]
                    l11 = new_scores[("TS11-D", seed, horizon, body, metric)]
                    per_seed[seed] = {
                        "spatial_main": ((l01 - l00) + (l11 - l10)) / 2.0,
                        "temporal_main": ((l10 - l00) + (l11 - l01)) / 2.0,
                        "interaction": (l11 - l10) - (l01 - l00),
                    }
                for effect_name in ("spatial_main", "temporal_main", "interaction"):
                    group_effect = np.mean(np.stack([per_seed[seed][effect_name] for seed in SEEDS]), axis=0)
                    estimate, low, high, p = summarize_groups(group_effect, bidx)
                    factorial_rows.append({
                        "effect": effect_name,
                        "horizon_steps": horizon,
                        "body": body,
                        "metric": metric,
                        "estimate_three_seed_mean": estimate,
                        "ci95_low_descriptive": low,
                        "ci95_high_descriptive": high,
                        "p_unadjusted_bootstrap": p,
                        "seed_estimates": json.dumps({str(seed): float(per_seed[seed][effect_name].mean()) for seed in SEEDS}, sort_keys=True),
                        "seed_mean": float(np.mean([per_seed[seed][effect_name].mean() for seed in SEEDS])),
                        "seed_sd": float(np.std([per_seed[seed][effect_name].mean() for seed in SEEDS], ddof=1)),
                        "bootstrap_replicates": BOOTSTRAPS,
                        "bootstrap_unit": "paired episode groups; descriptive seed-average effect",
                        "holm_adjustment": "not applied; predeclared factorial/secondary family reported with effect and CI",
                        "final_test_access": 0,
                    })
    write_csv(OUT / "D3_EQ_FACTORIAL_EFFECTS.csv", factorial_rows)

    objective_rows = []
    for config in CONFIGS:
        for horizon in HORIZONS:
            bidx = index_by_horizon[horizon]
            for body in BODIES:
                for metric in ("standardized_mae", "native_future_internal_fd"):
                    per_seed = {
                        seed: new_scores[(config, seed, horizon, body, metric)] - old_scores[(config, seed, horizon, body, metric)]
                        for seed in SEEDS
                    }
                    pooled = np.mean(np.stack([per_seed[seed] for seed in SEEDS]), axis=0)
                    estimate, low, high, p = summarize_groups(pooled, bidx)
                    objective_rows.append({
                        "comparison": "D3_EQ_V1_minus_OLD_D3_ACTUAL",
                        "config": config,
                        "horizon_steps": horizon,
                        "body": body,
                        "metric": metric,
                        "estimate_three_seed_mean": estimate,
                        "ci95_low_descriptive": low,
                        "ci95_high_descriptive": high,
                        "p_unadjusted_bootstrap": p,
                        "seed_estimates": json.dumps({str(seed): float(per_seed[seed].mean()) for seed in SEEDS}, sort_keys=True),
                        "seed_mean": float(np.mean([per_seed[seed].mean() for seed in SEEDS])),
                        "seed_sd": float(np.std([per_seed[seed].mean() for seed in SEEDS], ddof=1)),
                        "source_episode_groups": len(ordered_groups),
                        "bootstrap_replicates": BOOTSTRAPS,
                        "interpretation": "development-only objective sensitivity; not an independent seed/holdout",
                        "final_test_access": 0,
                    })
    write_csv(OUT / "D3_EQ_OBJECTIVE_SENSITIVITY.csv", objective_rows)

    # Directly compare each predeclared architecture contrast under the two
    # training objectives on the same seed/group cells.
    contrast_sensitivity_rows = []
    for horizon in HORIZONS:
        bidx = index_by_horizon[horizon]
        for candidate, control, contrast in PRIMARY_CONTRASTS:
            for body in BODIES:
                for metric in ("standardized_mae", "native_future_internal_fd"):
                    deltas = []
                    by_seed = {}
                    for seed in SEEDS:
                        d_new = new_scores[(candidate, seed, horizon, body, metric)] - new_scores[(control, seed, horizon, body, metric)]
                        d_old = old_scores[(candidate, seed, horizon, body, metric)] - old_scores[(control, seed, horizon, body, metric)]
                        delta = d_new - d_old
                        by_seed[seed] = delta
                        deltas.append(delta)
                    pooled = np.mean(np.stack(deltas), axis=0)
                    estimate, low, high, p = summarize_groups(pooled, bidx)
                    contrast_sensitivity_rows.append({
                        "contrast": contrast,
                        "horizon_steps": horizon,
                        "body": body,
                        "metric": metric,
                        "estimand": "(EQ candidate-control) - (OLD candidate-control)",
                        "estimate_three_seed_mean": estimate,
                        "ci95_low_descriptive": low,
                        "ci95_high_descriptive": high,
                        "p_unadjusted_bootstrap": p,
                        "seed_estimates": json.dumps({str(seed): float(by_seed[seed].mean()) for seed in SEEDS}, sort_keys=True),
                        "source_episode_groups": len(ordered_groups),
                        "bootstrap_replicates": BOOTSTRAPS,
                        "interpretation": "objective-sensitivity change in paired architecture effect",
                        "final_test_access": 0,
                    })
    write_csv(OUT / "D3_EQ_CONTRAST_OBJECTIVE_SENSITIVITY.csv", contrast_sensitivity_rows)

    qa = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "old_prediction_slots": len(old_predictions),
        "new_prediction_slots": len(new_predictions),
        "primary_dev_anchors": int(len(ids)),
        "source_episode_groups": int(len(ordered_groups)),
        "task_categories_not_bootstrap_units": int(len(np.unique(archive["original_task_index"][dev_idx]))),
        "dev_targets_loaded_after_training": True,
        "final_test_access": 0,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_holm_tests": len(holm),
        "paired_prediction_id_joins": "PASS_OLD_AND_NEW_36_OF_36",
        "seed_aggregate_intervals": "descriptive; group-level resampling; seeds not treated as independent sample units",
    }
    atomic_json(OUT / "D3_EQ_ANALYSIS_QA.json", qa)
    print(json.dumps(qa, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
