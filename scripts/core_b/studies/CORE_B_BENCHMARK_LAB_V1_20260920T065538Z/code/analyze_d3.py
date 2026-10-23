"""CPU D3 metric, paired contrast, and factorial interaction analysis."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, write_csv


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def bootstrap(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> int:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    sigma = np.asarray(read_json(V3 / "data/V3_NORMALIZERS.json")["target_sigma"], dtype=np.float32)
    dev = np.flatnonzero(data["split_code"] == 1)
    per_model: dict[tuple[str, int, int], dict[int, dict[str, float]]] = {}
    result_rows = []
    for path in sorted((BENCH_ROOT / "d3/predictions").glob("D3-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["anchor_id"].astype("U64"); pred = archive["predictions_native"]; horizon = int(archive["horizon_steps"]); config = str(archive["config"].item()); seed = int(archive["seed"].item())
        source_map = {str(value): int(index) for index, value in enumerate(data["anchor_id"])}
        episode_metrics = defaultdict(lambda: defaultdict(list))
        for row_index, anchor_id in enumerate(ids):
            source = source_map[str(anchor_id)]
            if source not in set(dev): continue
            truth = data["targets_native"][source, :horizon]
            prediction = pred[row_index]
            episode = int(data["episode_index"][source])
            for body, sl in (("Hand", slice(0, 14)), ("Arm", slice(14, 28)), ("Full28", slice(0, 28))):
                mae = float(np.mean(np.abs(prediction[:, sl] - truth[:, sl]) / sigma[sl][None, :]))
                fd = float(np.mean(np.abs(np.diff(prediction[:, sl], axis=0) - np.diff(truth[:, sl], axis=0))))
                episode_metrics[episode][f"{body}_mae"].append(mae); episode_metrics[episode][f"{body}_fd"].append(fd)
        key = (config, seed, horizon); per_model[key] = {}
        for episode, metrics in episode_metrics.items():
            per_model[key][episode] = {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for body in ("Hand", "Arm", "Full28"):
            for metric in ("mae", "fd"):
                values = [item[f"{body}_{metric}"] for item in per_model[key].values()]
                result_rows.append({"config": config, "seed": seed, "horizon_steps": horizon, "body": body, "metric": metric, "N_episode": len(values), "estimate": float(np.mean(values)), "final_test_access": 0})
    write_csv(BENCH_ROOT / "d3/analysis/D3_RESULTS.csv", result_rows, list(result_rows[0].keys()))
    contrasts = []
    pairs = [("TS11-D", "TS10-D", "TS11_vs_TS10"), ("TS11-D", "TS01-D", "TS11_vs_TS01"), ("TS11-D", "TS11-S", "TS11D_vs_TS11S"), ("TS11-D", "STATE-CTRL", "TS11_vs_STATE")]
    for horizon in (30, 60):
        for seed in (20261001, 20261002, 20261003):
            for candidate, control, label in pairs:
                for body in ("Hand", "Arm", "Full28"):
                    key_c, key_o = (candidate, seed, horizon), (control, seed, horizon)
                    if key_c not in per_model or key_o not in per_model: continue
                    common = sorted(set(per_model[key_c]) & set(per_model[key_o]))
                    if not common: continue
                    diff = np.asarray([per_model[key_c][episode][f"{body}_mae"] - per_model[key_o][episode][f"{body}_mae"] for episode in common])
                    estimate, low, high = bootstrap(diff, seed + horizon)
                    contrasts.append({"contrast": label, "seed": seed, "horizon_steps": horizon, "body": body, "metric": "standardized_mae_candidate_minus_control", "N_episode": len(diff), "estimate": estimate, "ci_low": low, "ci_high": high, "final_test_access": 0})
    for horizon in (30, 60):
        for seed in (20261001, 20261002, 20261003):
            for body in ("Hand", "Arm", "Full28"):
                keys = [("TS11-D", seed, horizon), ("TS10-D", seed, horizon), ("TS01-D", seed, horizon), ("TS00-D", seed, horizon)]
                if not all(key in per_model for key in keys): continue
                common = sorted(set.intersection(*(set(per_model[key]) for key in keys)))
                values = np.asarray([per_model[keys[0]][e][f"{body}_mae"] - per_model[keys[1]][e][f"{body}_mae"] - per_model[keys[2]][e][f"{body}_mae"] + per_model[keys[3]][e][f"{body}_mae"] for e in common])
                estimate, low, high = bootstrap(values, seed + 1000 + horizon)
                contrasts.append({"contrast": "factorial_interaction_TS11_TS10_TS01_TS00", "seed": seed, "horizon_steps": horizon, "body": body, "metric": "(L11-L10)-(L01-L00)", "N_episode": len(values), "estimate": estimate, "ci_low": low, "ci_high": high, "final_test_access": 0})
    write_csv(BENCH_ROOT / "d3/analysis/D3_CONTRASTS.csv", contrasts, list(contrasts[0].keys()))
    summary = {"status": "D3_ANALYSIS_COMPLETE", "prediction_files": len(per_model), "result_rows": len(result_rows), "contrast_rows": len(contrasts), "bootstrap_draws": 10000, "final_test_access": 0}
    atomic_write_json(BENCH_ROOT / "d3/analysis/D3_ANALYSIS_SUMMARY.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__": raise SystemExit(main())
