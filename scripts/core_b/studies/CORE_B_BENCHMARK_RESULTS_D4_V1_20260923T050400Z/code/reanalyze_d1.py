"""Source-video-group-aware D1 reanalysis; development split only."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pathlibs import NEXT_ROOT, REPO_ROOT, V3_ROOT

BENCH_ROOT = NEXT_ROOT.parent / "CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
DATA_PATH = V3_ROOT / "data/V3_MAIN_INPUTS.npz"
SIGMA = np.asarray(json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
SPLIT = REPO_ROOT / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
K = 20
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260923


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def quantile_episode_equal(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def make_bootstrap_counts(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups), dtype=np.int32)
    counts = np.zeros((n_boot, n_groups), dtype=np.uint16)
    row_idx = np.broadcast_to(np.arange(n_boot, dtype=np.int32)[:, None], draws.shape)
    np.add.at(counts, (row_idx, draws), 1)
    return counts


def boot_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    numerator = counts @ np.nan_to_num(values, nan=0.0)
    denominator = counts @ valid.astype(np.uint16)
    return np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)


def summarize_group_metric(values: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    estimate = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    reps = boot_mean(values, counts)
    finite = reps[np.isfinite(reps)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return estimate, float(lo), float(hi)


def load_group_map() -> dict[int, str]:
    table = pq.read_table(SPLIT, columns=["episode_index", "split", "video_sha256_fingerprint"])
    mapping = {}
    for row in table.to_pylist():
        if row["split"] not in {"train", "validation"}:
            continue
        episode = int(row["episode_index"])
        fingerprint = row["video_sha256_fingerprint"]
        mapping[episode] = str(fingerprint) if fingerprint else f"episode:{episode}"
    return mapping


def ordered_indices(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return indices[np.lexsort((data["anchor_id"][indices], data["anchor_frame"][indices], data["episode_index"][indices]))]


def select_bank(data: dict[str, np.ndarray], train: np.ndarray) -> np.ndarray:
    ordered = ordered_indices(data, train)
    counts: dict[int, int] = defaultdict(int)
    selected = []
    for index in ordered:
        episode = int(data["episode_index"][index])
        if counts[episode] < 8:
            counts[episode] += 1
            selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def select_train_queries(data: dict[str, np.ndarray], train: np.ndarray) -> np.ndarray:
    ordered = ordered_indices(data, train)
    seen: set[int] = set()
    selected = []
    for index in ordered:
        episode = int(data["episode_index"][index])
        if episode not in seen:
            selected.append(int(index))
            seen.add(episode)
    return np.asarray(selected, dtype=np.int64)


def nearest_group_neighbors(
    queries: np.ndarray,
    bank: np.ndarray,
    state: np.ndarray,
    episode: np.ndarray,
    group: np.ndarray,
    ids: np.ndarray,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact stable nearest rows, retaining only the closest row per source group."""
    bank_state = state[bank]
    bank_group = group[bank]
    bank_episode = episode[bank]
    # select_bank already locks episode/frame/anchor-ID order; preserve it for ties.
    bank_norm = np.einsum("ij,ij->i", bank_state, bank_state, dtype=np.float64)
    result_idx = np.empty((len(queries), k), dtype=np.int64)
    result_dist = np.empty((len(queries), k), dtype=np.float64)
    for start in range(0, len(queries), 32):
        qidx = queries[start : start + 32]
        qstate = state[qidx]
        qnorm = np.einsum("ij,ij->i", qstate, qstate, dtype=np.float64)
        dists = (bank_norm[:, None] - 2.0 * (bank_state @ qstate.T).astype(np.float64) + qnorm[None, :]) / state.shape[1]
        dists = np.maximum(dists, 0.0)
        for j, query_index in enumerate(qidx):
            same_source = bank_group == group[query_index]
            dists[same_source, j] = np.inf
            # lexsort gives deterministic ties: distance, then stable bank row.
            order = np.lexsort((np.arange(len(bank)), dists[:, j]))
            chosen: list[int] = []
            seen: set[str] = set()
            for candidate in order:
                candidate_group = str(bank_group[candidate])
                if candidate_group in seen:
                    continue
                seen.add(candidate_group)
                chosen.append(int(candidate))
                if len(chosen) == k:
                    break
            if len(chosen) != k:
                raise ValueError(f"query {ids[query_index]} has {len(chosen)} distinct eligible source groups; K={k} required")
            out_row = start + j
            result_idx[out_row] = bank[np.asarray(chosen, dtype=np.int64)]
            result_dist[out_row] = dists[np.asarray(chosen, dtype=np.int64), j]
    return result_idx, result_dist


def neighbor_outputs(data: dict[str, np.ndarray], query_indices: np.ndarray, bank: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    state = data["history_standardized"].reshape(len(data["anchor_id"]), -1).astype(np.float32)
    episode = data["episode_index"]
    neighbor_index, distances = nearest_group_neighbors(query_indices, bank, state, episode, groups, data["anchor_id"])
    residuals = (data["targets_native"][neighbor_index, :60] - data["anchor_command_native"][neighbor_index, None, :]) / SIGMA[None, None, :]
    median = np.median(residuals, axis=1)
    anchor = data["anchor_command_native"][query_indices]
    output = {
        "neighbor_index": neighbor_index,
        "neighbor_distance": distances,
        "nn_prediction_h30": anchor[:, None, :] + median[:, :30, :] * SIGMA[None, None, :],
        "nn_prediction_h60": anchor[:, None, :] + median * SIGMA[None, None, :],
        "dispersion_h30": np.abs(residuals[:, :, :][:, :, :]).mean(axis=(1, 2), dtype=np.float64),
    }
    for horizon in (30, 60):
        centered = residuals[:, :, :] - median[:, None, :, :]
        output[f"dispersion_h{horizon}"] = np.abs(centered[:, :, :horizon, :]).mean(axis=(1, 2, 3), dtype=np.float64)
    return output


def window_metrics(pred: np.ndarray, truth: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    rows = {}
    for body, sl in (("Hand", slice(0, 14)), ("Arm", slice(14, 28)), ("Full28", slice(0, 28))):
        rows[f"{body}|mae"] = np.abs(pred[:, :, sl] - truth[:, :, sl]) / sigma[None, None, sl]
        rows[f"{body}|fd"] = np.abs(np.diff(pred[:, :, sl], axis=1) - np.diff(truth[:, :, sl], axis=1))
    return {key: val.mean(axis=(1, 2), dtype=np.float64) for key, val in rows.items()}


def group_values(window_values: np.ndarray, group_idx: np.ndarray, mask: np.ndarray, n_groups: int) -> np.ndarray:
    result = np.full(n_groups, np.nan, dtype=np.float64)
    for g in range(n_groups):
        local = mask & (group_idx == g)
        if local.any():
            result[g] = float(np.mean(window_values[local]))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="verify joins/neighbors on 20 train and 20 dev queries only")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    group_map = load_group_map()
    episode = data["episode_index"]
    missing = sorted(set(map(int, episode)) - set(group_map))
    if missing:
        # The source NPZ contains final-test rows too; only train/dev rows require groups.
        relevant = np.isin(data["split_code"], [0, 1])
        missing = sorted(set(map(int, episode[relevant])) - set(group_map))
    if missing:
        raise ValueError(f"source-group manifest misses {len(missing)} train/dev episodes; first={missing[:8]}")
    groups = np.asarray([group_map.get(int(ep), f"final_test:{int(ep)}") for ep in episode], dtype="U64")
    train_all = np.flatnonzero(data["split_code"] == 0)
    dev_all = np.flatnonzero(data["split_code"] == 1)
    bank = select_bank(data, train_all)
    train_q = select_train_queries(data, train_all)
    if args.smoke:
        train_q = train_q[:20]
        dev_all = dev_all[:20]
    train_out = neighbor_outputs(data, train_q, bank, groups)
    if args.smoke:
        dev_out = neighbor_outputs(data, dev_all, bank, groups)
        assert train_out["neighbor_index"].shape == (20, K)
        assert dev_out["neighbor_index"].shape == (20, K)
        assert all(len(set(groups[row].tolist())) == K for row in dev_out["neighbor_index"])
        print(json.dumps({"smoke": "PASS", "bank_rows": len(bank), "train_queries": len(train_q), "dev_queries": len(dev_all), "unique_neighbor_groups": K, "final_test_access": 0}))
        return

    # Freeze train-calibrated thresholds before reading any learned prediction arrays.
    q25 = quantile_episode_equal(train_out["dispersion_h60"], 0.25)
    q75 = quantile_episode_equal(train_out["dispersion_h60"], 0.75)
    support_q95 = quantile_episode_equal(train_out["neighbor_distance"][:, -1], 0.95)
    dev_out = neighbor_outputs(data, dev_all, bank, groups)
    low = dev_out["dispersion_h60"] <= q25
    high = dev_out["dispersion_h60"] >= q75
    supported = dev_out["neighbor_distance"][:, -1] <= support_q95
    strata = {
        "overall": np.ones(len(dev_all), dtype=bool),
        "supported_overall": supported,
        "low_dispersion_all": low,
        "high_dispersion_all": high,
        "low_dispersion_supported": low & supported,
        "high_dispersion_supported": high & supported,
    }
    dev_groups = groups[dev_all]
    ordered_group_ids = sorted(set(dev_groups.tolist()))
    group_to_index = {value: i for i, value in enumerate(ordered_group_ids)}
    window_group = np.asarray([group_to_index[x] for x in dev_groups], dtype=np.int32)
    n_groups = len(ordered_group_ids)
    if n_groups != len(np.unique(episode[dev_all])):
        raise ValueError(f"primary-dev source-group and episode counts disagree: groups={n_groups}, episodes={len(np.unique(episode[dev_all]))}")
    counts = make_bootstrap_counts(n_groups, BOOTSTRAPS, BOOTSTRAP_SEED)
    truth = data["targets_native"][dev_all, :60]
    anchor = data["anchor_command_native"][dev_all]
    predictions: dict[tuple[str, int, int], np.ndarray] = {}
    predictions[("PERSISTENCE", 0, 30)] = np.repeat(anchor[:, None, :], 30, axis=1)
    predictions[("PERSISTENCE", 0, 60)] = np.repeat(anchor[:, None, :], 60, axis=1)
    predictions[("NN_MEDIAN", 0, 30)] = dev_out["nn_prediction_h30"]
    predictions[("NN_MEDIAN", 0, 60)] = dev_out["nn_prediction_h60"]
    for marker_path in sorted((V3_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "PASS":
            continue
        run_id = marker_path.parent.name
        cell = str(marker.get("cell", "UNKNOWN"))
        seed = int(marker["seed"])
        with np.load(marker_path.parent / "primary.npz", allow_pickle=False) as bundle:
            ids = bundle["anchor_id"].astype("U64")
            pred = bundle["predictions_native"]
            horizon = int(bundle["horizon_steps"])
        dev_lookup = {str(v): i for i, v in enumerate(data["anchor_id"][dev_all])}
        bundle_lookup = {str(v): i for i, v in enumerate(ids)}
        if len(ids) != len(dev_all) or set(ids.tolist()) != set(dev_lookup):
            raise ValueError(f"D1 prediction ID set mismatch: {run_id}")
        pred_aligned = pred[np.asarray([bundle_lookup[str(aid)] for aid in data["anchor_id"][dev_all]])]
        predictions[(cell, seed, horizon)] = pred_aligned

    row_output = []
    group_metric: dict[tuple[str, int, int, str, str, str], np.ndarray] = {}
    metric_cache: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for (model, seed, horizon), pred in predictions.items():
        metric_cache[(model, seed, horizon)] = window_metrics(pred, truth[:, :horizon], SIGMA)
        for metric_name, values in metric_cache[(model, seed, horizon)].items():
            body, metric = metric_name.split("|")
            for stratum, mask in strata.items():
                gv = group_values(values, window_group, mask, n_groups)
                group_metric[(model, seed, horizon, body, metric, stratum)] = gv
                est, lo, hi = summarize_group_metric(gv, counts)
                contributing = int(np.isfinite(gv).sum())
                row_output.append({"model": model, "seed": seed if seed else "NOT_APPLICABLE", "horizon_steps": horizon, "body": body, "metric": metric, "stratum": stratum, "N_source_group": contributing, "N_episode": contributing, "N_window": int(mask.sum()), "estimate": est, "ci95_low": lo, "ci95_high": hi, "final_test_access": 0})
    write_csv(NEXT_ROOT / "results/D1_SOURCE_GROUP_RESULTS.csv", row_output)

    interactions = []
    for seed in (20260950, 20260951, 20260952):
        for horizon in (30, 60):
            for body in ("Hand", "Arm", "Full28"):
                for control in ("S-HIST", "F-POOL-D", "F-TOK-S"):
                    fusion = "F-TOK-D"
                    keys = [(cell, seed, horizon, body, "mae", stratum) for cell, stratum in ((control, "low_dispersion_supported"), (control, "high_dispersion_supported"), (fusion, "low_dispersion_supported"), (fusion, "high_dispersion_supported"))]
                    if any(key not in group_metric for key in keys):
                        continue
                    c_low, c_high, f_low, f_high = (group_metric[key] for key in keys)
                    gain_low = c_low - f_low
                    gain_high = c_high - f_high
                    estimate = float(np.nanmean(gain_high) - np.nanmean(gain_low))
                    reps = boot_mean(gain_high, counts) - boot_mean(gain_low, counts)
                    finite = reps[np.isfinite(reps)]
                    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                    interactions.append({"seed": seed, "horizon_steps": horizon, "body": body, "control": control, "fusion": fusion, "contrast": f"{control}-minus-{fusion}: high_gain-minus-low_gain", "gain_low": float(np.nanmean(gain_low)), "gain_high": float(np.nanmean(gain_high)), "high_minus_low": estimate, "ci95_low": float(lo), "ci95_high": float(hi), "N_low_groups": int(np.isfinite(gain_low).sum()), "N_high_groups": int(np.isfinite(gain_high).sum()), "bootstrap_source_groups": n_groups, "final_test_access": 0})
    write_csv(NEXT_ROOT / "results/D1_SOURCE_GROUP_INTERACTIONS.csv", interactions)

    query_rows = []
    for i, source in enumerate(dev_all):
        query_rows.append({"anchor_id": str(data["anchor_id"][source]), "episode_index": int(episode[source]), "source_group_id": str(groups[source]), "task": int(data["original_task_index"][source]), "anchor_frame": int(data["anchor_frame"][source]), "dispersion_h30": float(dev_out["dispersion_h30"][i]), "dispersion_h60": float(dev_out["dispersion_h60"][i]), "neighbor_distance_k20": float(dev_out["neighbor_distance"][i, -1]), "low_h60": bool(low[i]), "high_h60": bool(high[i]), "supported": bool(supported[i]), "final_test_access": 0})
    write_csv(NEXT_ROOT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv", query_rows)
    np.savez_compressed(NEXT_ROOT / "results/D1_SOURCE_GROUP_NEIGHBORS.npz", dev_anchor_ids=data["anchor_id"][dev_all], neighbor_anchor_ids=data["anchor_id"][dev_out["neighbor_index"]], neighbor_distances=dev_out["neighbor_distance"], nn_prediction_h30=dev_out["nn_prediction_h30"], nn_prediction_h60=dev_out["nn_prediction_h60"])
    summary = {"status": "COMPLETE_SOURCE_GROUP_REANALYSIS", "protocol_id": "CORE_B_D1_SOURCE_GROUP_REANALYSIS_V2", "train_bank_rows": int(len(bank)), "train_bank_episodes": int(len(np.unique(episode[bank]))), "train_calibration_episodes": int(len(train_q)), "dev_rows": int(len(dev_all)), "dev_episodes": int(len(np.unique(episode[dev_all]))), "dev_source_groups": n_groups, "thresholds_train_only": {"dispersion_q25_h60": q25, "dispersion_q75_h60": q75, "neighbor_distance_k20_q95": support_q95}, "dev_support": {"low_supported_windows": int((low & supported).sum()), "high_supported_windows": int((high & supported).sum()), "supported_windows": int(supported.sum()), "all_windows": len(dev_all)}, "accepted_v3_prediction_bundles": sum(1 for key in predictions if key[0] not in {"PERSISTENCE", "NN_MEDIAN"}), "result_rows": len(row_output), "interaction_rows": len(interactions), "bootstrap_replicates": BOOTSTRAPS, "bootstrap_unit": "exact video_sha256_fingerprint source group", "final_test_access": 0}
    (NEXT_ROOT / "results/D1_SOURCE_GROUP_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
