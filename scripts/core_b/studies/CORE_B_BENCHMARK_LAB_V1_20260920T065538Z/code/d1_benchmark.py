"""CPU D1 state-neighborhood future-dispersion benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, sha256_json, utc_now, write_csv


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
DATA = V3 / "data/V3_MAIN_INPUTS.npz"
NORMALIZERS = V3 / "data/V3_NORMALIZERS.json"


def lock_protocol() -> dict[str, object]:
    protocol = {
        "protocol_id": "CORE_B_BENCHMARK_D1_NEIGHBOR_DISPERSION_V1",
        "generated_utc": utc_now(),
        "train_bank": "canonical V3 primary train rows, at most first 8 anchors per episode ordered by anchor_frame then anchor_id",
        "train_calibration_queries": "one earliest primary anchor per train episode, fixed before dev target inspection, episode-equal calibration",
        "distance": "mean squared distance over 32x71 V3-standardized state history values; no video/task/future/model error",
        "same_source_exclusion": "episode_index because duplicate_group_id is absent from the V3 manifest; this limitation is explicit",
        "neighbors": 20,
        "neighbor_tie_break": "stable bank order: episode_index, anchor_frame, anchor_id",
        "dispersion": "coordinatewise median residual to anchor, sigma_y scaled, mean absolute deviation over neighbors/time/joints",
        "strata": "train calibration Q25/Q75; 2s A defines common strata for both horizons; 1s A supplement also emitted",
        "support": "train calibration 20th-neighbor distance Q95; dev over-support retained in natural overall and marked distant_neighbor",
        "primary_claim_status": "visual model gain remains pending until sealed V3 predictions are available",
        "final_test_access": 0,
    }
    protocol["protocol_hash"] = sha256_json(protocol)
    atomic_write_json(BENCH_ROOT / "protocol/D1_PROTOCOL_LOCKED.json", protocol)
    return protocol


def stable_order(episode: np.ndarray, frame: np.ndarray, ids: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # Numeric episode/frame plus string ids are deterministic and do not inspect targets.
    return indices[np.lexsort((ids[indices], frame[indices], episode[indices]))]


def select_bank(data: dict[str, np.ndarray]) -> np.ndarray:
    train = np.flatnonzero(data["split_code"] == 0)
    ordered = stable_order(data["episode_index"], data["anchor_frame"], data["anchor_id"], train)
    selected: list[int] = []
    counts: dict[int, int] = {}
    for index in ordered:
        episode = int(data["episode_index"][index])
        if counts.get(episode, 0) >= 8:
            continue
        counts[episode] = counts.get(episode, 0) + 1
        selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def select_calibration_queries(data: dict[str, np.ndarray]) -> np.ndarray:
    train = np.flatnonzero(data["split_code"] == 0)
    ordered = stable_order(data["episode_index"], data["anchor_frame"], data["anchor_id"], train)
    seen: set[int] = set()
    selected = []
    for index in ordered:
        episode = int(data["episode_index"][index])
        if episode not in seen:
            seen.add(episode)
            selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def nearest_unique(query: np.ndarray, bank: np.ndarray, bank_episode: np.ndarray, query_episode: int, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    distances = np.mean((bank - query[None, :]) ** 2, axis=1, dtype=np.float64)
    distances[bank_episode == query_episode] = np.inf
    candidate_count = min(len(distances), max(1024, k * 64))
    candidates = np.argpartition(distances, candidate_count - 1)[:candidate_count]
    candidates = candidates[np.argsort(distances[candidates], kind="mergesort")]
    chosen: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        group = int(bank_episode[candidate])
        if group in seen:
            continue
        seen.add(group)
        chosen.append(int(candidate))
        if len(chosen) == k:
            break
    if len(chosen) < k:
        # This is exact fallback, not silent K reduction.
        for candidate in np.argsort(distances, kind="mergesort"):
            group = int(bank_episode[candidate])
            if group in seen:
                continue
            seen.add(group)
            chosen.append(int(candidate))
            if len(chosen) == k:
                break
    if len(chosen) < k:
        raise ValueError(f"query episode {query_episode} has only {len(chosen)} independent neighbors; K={k} required")
    selected = np.asarray(chosen, dtype=np.int64)
    return selected, distances[selected].astype(np.float32)


def neighbor_outputs(data: dict[str, np.ndarray], query_indices: np.ndarray, bank_indices: np.ndarray, sigma: np.ndarray, horizons: tuple[int, ...] = (30, 60)) -> dict[str, np.ndarray]:
    state = data["history_standardized"].reshape(len(data["history_standardized"]), -1).astype(np.float32)
    bank_state = state[bank_indices]
    bank_episode = data["episode_index"][bank_indices]
    q = len(query_indices)
    neighbor_index = np.empty((q, 20), dtype=np.int64)
    neighbor_distance = np.empty((q, 20), dtype=np.float32)
    outputs: dict[str, np.ndarray] = {}
    for horizon in horizons:
        outputs[f"nn_prediction_h{horizon}"] = np.empty((q, horizon, 28), dtype=np.float32)
        outputs[f"dispersion_h{horizon}"] = np.empty(q, dtype=np.float32)
    for row, source_index in enumerate(query_indices):
        selected, distances = nearest_unique(state[source_index], bank_state, bank_episode, int(data["episode_index"][source_index]))
        neighbor_index[row] = bank_indices[selected]
        neighbor_distance[row] = distances
        anchor = data["anchor_command_native"][source_index]
        for horizon in horizons:
            residual = (data["targets_native"][neighbor_index[row], :horizon] - data["anchor_command_native"][neighbor_index[row], None, :]) / sigma[None, None, :]
            center = np.median(residual, axis=0)
            outputs[f"nn_prediction_h{horizon}"][row] = anchor[None, :] + sigma[None, :] * center
            outputs[f"dispersion_h{horizon}"][row] = np.abs(residual - center[None, :, :]).mean(dtype=np.float64)
    outputs["neighbor_index"] = neighbor_index
    outputs["neighbor_distance"] = neighbor_distance
    return outputs


def metric_rows(data: dict[str, np.ndarray], query_indices: np.ndarray, predictions: dict[str, np.ndarray], strata: dict[str, np.ndarray], sigma: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    targets = data["targets_native"][query_indices]
    anchors = data["anchor_command_native"][query_indices]
    episodes = data["episode_index"][query_indices]
    for horizon in (30, 60):
        target = targets[:, :horizon]
        persistence = np.repeat(anchors[:, None, :], horizon, axis=1)
        model_predictions = {"PERSISTENCE": persistence, "NN_MEDIAN": predictions[f"nn_prediction_h{horizon}"]}
        for stratum, mask in strata.items():
            selected = np.flatnonzero(mask)
            for model, pred in model_predictions.items():
                for body, sl in (("Hand", slice(0, 14)), ("Arm", slice(14, 28)), ("Full28", slice(0, 28))):
                    window_mae = np.mean(np.abs(pred[selected, :, sl] - target[selected, :, sl]) / sigma[sl][None, None, :], axis=(1, 2)) if len(selected) else np.asarray([], dtype=np.float32)
                    fd_target = np.diff(target[selected, :, sl], axis=1) if len(selected) else np.empty((0, max(horizon - 1, 0), len(range(*sl.indices(28)))), dtype=np.float32)
                    fd_pred = np.diff(pred[selected, :, sl], axis=1) if len(selected) else fd_target.copy()
                    window_fd = np.mean(np.abs(fd_pred - fd_target), axis=(1, 2)) if len(selected) else np.asarray([], dtype=np.float32)
                    ep_values = []
                    ep_fd = []
                    for episode in np.unique(episodes[selected]):
                        local = np.flatnonzero(episodes[selected] == episode)
                        ep_values.append(float(np.mean(window_mae[local])))
                        ep_fd.append(float(np.mean(window_fd[local])))
                    rows.append({"horizon_steps": horizon, "stratum": stratum, "model": model, "body": body, "N_episode": len(ep_values), "N_window": len(selected), "standardized_mae_episode_equal": float(np.mean(ep_values)) if ep_values else None, "native_fd_episode_equal": float(np.mean(ep_fd)) if ep_fd else None, "support_fraction": float(np.mean(strata["supported"][selected])) if len(selected) else None})
    return rows


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    protocol = lock_protocol()
    with np.load(DATA, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    normalizers = read_json(NORMALIZERS)
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    bank = select_bank(data)
    calibration = select_calibration_queries(data)
    dev = np.flatnonzero(data["split_code"] == 1)
    train_out = neighbor_outputs(data, calibration, bank, sigma)
    dev_out = neighbor_outputs(data, dev, bank, sigma)
    q25 = float(np.quantile(train_out["dispersion_h60"], 0.25))
    q75 = float(np.quantile(train_out["dispersion_h60"], 0.75))
    support_q95 = float(np.quantile(train_out["neighbor_distance"][:, -1], 0.95))
    strata = {
        "overall": np.ones(len(dev), dtype=bool),
        "low_dispersion": dev_out["dispersion_h60"] <= q25,
        "mid_dispersion": (dev_out["dispersion_h60"] > q25) & (dev_out["dispersion_h60"] < q75),
        "high_dispersion": dev_out["dispersion_h60"] >= q75,
        "supported": dev_out["neighbor_distance"][:, -1] <= support_q95,
    }
    strata["low_supported"] = strata["low_dispersion"] & strata["supported"]
    strata["high_supported"] = strata["high_dispersion"] & strata["supported"]
    rows = metric_rows(data, dev, dev_out, strata, sigma)
    write_csv(BENCH_ROOT / "d1/D1_RESULTS.csv", rows, list(rows[0].keys()) if rows else ["horizon_steps"])
    query_rows = []
    for i, source_index in enumerate(dev):
        query_rows.append({"anchor_id": data["anchor_id"][source_index], "episode_index": int(data["episode_index"][source_index]), "task": int(data["original_task_index"][source_index]), "anchor_frame": int(data["anchor_frame"][source_index]), "dispersion_h30": float(dev_out["dispersion_h30"][i]), "dispersion_h60": float(dev_out["dispersion_h60"][i]), "neighbor_distance_k20": float(dev_out["neighbor_distance"][i, -1]), "low_h60": bool(strata["low_dispersion"][i]), "high_h60": bool(strata["high_dispersion"][i]), "supported": bool(strata["supported"][i]), "final_test_access": 0})
    write_csv(BENCH_ROOT / "d1/D1_QUERY_TABLE.csv", query_rows, list(query_rows[0].keys()) if query_rows else ["anchor_id"])
    np.savez_compressed(BENCH_ROOT / "d1/D1_NEIGHBOR_OUTPUTS.npz", dev_indices=dev, bank_indices=bank, calibration_indices=calibration, dev_neighbor_index=dev_out["neighbor_index"], dev_neighbor_distance=dev_out["neighbor_distance"], dev_dispersion_h30=dev_out["dispersion_h30"], dev_dispersion_h60=dev_out["dispersion_h60"], nn_prediction_h30=dev_out["nn_prediction_h30"], nn_prediction_h60=dev_out["nn_prediction_h60"])
    summary = {
        "status": "CPU_COMPLETE",
        "protocol_hash": protocol["protocol_hash"],
        "bank_rows": int(len(bank)), "bank_episodes": int(len(np.unique(data["episode_index"][bank]))),
        "calibration_rows": int(len(calibration)), "dev_rows": int(len(dev)), "dev_episodes": int(len(np.unique(data["episode_index"][dev]))),
        "thresholds": {"q25_h60": q25, "q75_h60": q75, "support_q95_distance": support_q95},
        "support": {"low": int(strata["low_supported"].sum()), "high": int(strata["high_supported"].sum()), "overall": int(strata["supported"].sum())},
        "visual_disambiguation": "INCONCLUSIVE_AWAITING_SEALED_V3_PREDICTIONS",
        "nn_baseline": "produced; compare to learned state/fusion after prediction bundle acceptance",
        "final_test_access": 0,
    }
    atomic_write_json(BENCH_ROOT / "d1/D1_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
