"""Cluster-level CPU statistics for the completed D1/D2 baseline work."""

from __future__ import annotations

import json

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, read_csv, write_csv


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def bootstrap(values: np.ndarray, seed: int = 20260920, draws: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> int:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    sigma = np.asarray(read_json(V3 / "data/V3_NORMALIZERS.json")["target_sigma"], dtype=np.float32)
    saved = np.load(BENCH_ROOT / "d1/D1_NEIGHBOR_OUTPUTS.npz", allow_pickle=False)
    dev = saved["dev_indices"]
    predictions = {"PERSISTENCE": np.repeat(data["anchor_command_native"][dev, None, :], 30, axis=1), "NN_MEDIAN": saved["nn_prediction_h30"]}
    predictions["PERSISTENCE60"] = np.repeat(data["anchor_command_native"][dev, None, :], 60, axis=1)
    predictions["NN_MEDIAN60"] = saved["nn_prediction_h60"]
    d1_summary = read_json(BENCH_ROOT / "d1/D1_SUMMARY.json")
    q25 = float(d1_summary["thresholds"]["q25_h60"])
    q75 = float(d1_summary["thresholds"]["q75_h60"])
    support = float(d1_summary["thresholds"]["support_q95_distance"])
    dispersion = saved["dev_dispersion_h60"]
    strata = {"overall": np.ones(len(dev), dtype=bool), "low_dispersion": dispersion <= q25, "mid_dispersion": (dispersion > q25) & (dispersion < q75), "high_dispersion": dispersion >= q75, "low_supported": (dispersion <= q25) & (saved["dev_neighbor_distance"][:, -1] <= support), "high_supported": (dispersion >= q75) & (saved["dev_neighbor_distance"][:, -1] <= support)}
    stats = []
    for horizon in (30, 60):
        target = data["targets_native"][dev, :horizon]
        nn = saved[f"nn_prediction_h{horizon}"]
        persistence = np.repeat(data["anchor_command_native"][dev, None, :], horizon, axis=1)
        for stratum, mask in strata.items():
            selected = np.flatnonzero(mask)
            for body, sl in (("Hand", slice(0, 14)), ("Arm", slice(14, 28)), ("Full28", slice(0, 28))):
                if len(selected) == 0:
                    continue
                difference = np.mean(np.abs(persistence[selected, :, sl] - target[selected, :, sl]) / sigma[sl][None, None, :], axis=(1, 2)) - np.mean(np.abs(nn[selected, :, sl] - target[selected, :, sl]) / sigma[sl][None, None, :], axis=(1, 2))
                episode_values = []
                episodes = data["episode_index"][dev[selected]]
                for episode in np.unique(episodes):
                    episode_values.append(float(np.mean(difference[episodes == episode])))
                mean, low, high = bootstrap(np.asarray(episode_values), seed=20260920 + horizon)
                stats.append({"horizon_steps": horizon, "stratum": stratum, "body": body, "contrast": "PERSISTENCE_minus_NN_MEDIAN_standardized_MAE", "N_episode": len(episode_values), "N_window": len(selected), "estimate": mean, "ci_low": low, "ci_high": high, "support_fraction": float(np.mean(saved["dev_neighbor_distance"][selected, -1] <= support))})
    write_csv(BENCH_ROOT / "d1/D1_BOOTSTRAP_CLUSTER_STATS.csv", stats, list(stats[0].keys()))
    d2_sparse = read_csv(BENCH_ROOT / "d2/D2_SPARSE_RESULTS.csv")
    d2_dense = read_csv(BENCH_ROOT / "d2/D2_DENSE_BASELINE_RESULTS.csv")
    d2_rows = []
    for row in d2_dense:
        d2_rows.append({"track": "dense", "model": row["model"], "horizon_steps": row["horizon_steps"], "episodes": row["episodes"], "queries": row["queries"], "hits": row["hits"], "false_alarms": row["false_alarms"], "misses": row["misses"], "false_alarms_per_hour": row["false_alarms_per_hour"], "status": "baseline_only"})
    for model in sorted({row["model"] for row in d2_sparse}):
        for horizon in ("30", "60"):
            subset = [row for row in d2_sparse if row["model"] == model and row["horizon_steps"] == horizon]
            d2_rows.append({"track": "sparse", "model": model, "horizon_steps": horizon, "episodes": len({row["episode_index"] for row in subset}), "queries": len(subset), "hits": sum(int(row["hits"]) for row in subset), "false_alarms": sum(int(row["false_alarms"]) for row in subset), "misses": sum(int(row["misses"]) for row in subset), "false_alarms_per_hour": "NA", "status": "baseline_only"})
    write_csv(BENCH_ROOT / "d2/D2_AGGREGATE_BASELINE_STATS.csv", d2_rows, list(d2_rows[0].keys()))
    atomic_write_json(BENCH_ROOT / "qa/CPU_ANALYSIS_QA.json", {"status": "PASS", "d1_bootstrap_draws": 10000, "d1_rows": len(stats), "d2_rows": len(d2_rows), "learned_model_results": "not_claimed", "final_test_access": 0})
    print(json.dumps({"status": "PASS", "d1_stats": len(stats), "d2_stats": len(d2_rows), "bootstrap": 10000}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
