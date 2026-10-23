"""Join and evaluate sealed V3 predictions on the locked D1/D2 sparse populations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, write_csv


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def load_data():
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def rows_for_prediction(data: dict[str, np.ndarray], prediction_path: Path, sigma: np.ndarray, strata_by_id: dict[str, dict[str, bool]]) -> list[dict[str, object]]:
    with np.load(prediction_path, allow_pickle=False) as archive:
        ids = archive["anchor_id"].astype("U64")
        preds = archive["predictions_native"]
        horizon = int(archive["horizon_steps"])
    id_to_source = {str(value): index for index, value in enumerate(data["anchor_id"])}
    if len(set(ids.tolist())) != len(ids) or any(str(value) not in id_to_source for value in ids):
        raise ValueError(f"prediction ID join failed: {prediction_path}")
    target = data["targets_native"]
    output = []
    for pred_index, anchor_id in enumerate(ids):
        key = str(anchor_id)
        source = id_to_source[key]
        if int(data["split_code"][source]) != 1:
            continue
        prediction = preds[pred_index]
        truth = target[source, :horizon]
        for body, sl in (("Hand", slice(0, 14)), ("Arm", slice(14, 28)), ("Full28", slice(0, 28))):
            mae = float(np.mean(np.abs(prediction[:, sl] - truth[:, sl]) / sigma[sl][None, :]))
            fd = float(np.mean(np.abs(np.diff(prediction[:, sl], axis=0) - np.diff(truth[:, sl], axis=0))))
            for stratum, included in strata_by_id[key].items():
                if included:
                    output.append({"anchor_id": key, "episode_index": int(data["episode_index"][source]), "horizon_steps": horizon, "body": body, "stratum": stratum, "mae_window": mae, "native_fd_window": fd})
    return output


def aggregate(rows: list[dict[str, object]], model: str, seed: int) -> list[dict[str, object]]:
    output = []
    for horizon in sorted({int(row["horizon_steps"]) for row in rows}):
        for stratum in ("overall", "low_dispersion", "mid_dispersion", "high_dispersion", "low_supported", "high_supported"):
            for body in ("Hand", "Arm", "Full28"):
                subset = [row for row in rows if int(row["horizon_steps"]) == horizon and row["stratum"] == stratum and row["body"] == body]
                ep_mae = []
                ep_fd = []
                for episode in sorted({row["episode_index"] for row in subset}):
                    values = [row for row in subset if row["episode_index"] == episode]
                    ep_mae.append(float(np.mean([row["mae_window"] for row in values])))
                    ep_fd.append(float(np.mean([row["native_fd_window"] for row in values])))
                output.append({"model": model, "seed": seed, "horizon_steps": horizon, "stratum": stratum, "body": body, "N_episode": len(ep_mae), "N_window": len(subset), "standardized_mae_episode_equal": float(np.mean(ep_mae)) if ep_mae else None, "native_fd_episode_equal": float(np.mean(ep_fd)) if ep_fd else None})
    return output


def main() -> int:
    data = load_data()
    query = list(__import__("csv").DictReader((BENCH_ROOT / "d1/D1_QUERY_TABLE.csv").open(encoding="utf-8")))
    strata_by_id = {row["anchor_id"]: {"overall": True, "low_dispersion": row["low_h60"] == "True", "mid_dispersion": row["low_h60"] == "False" and row["high_h60"] == "False", "high_dispersion": row["high_h60"] == "True", "low_supported": row["low_h60"] == "True" and row["supported"] == "True", "high_supported": row["high_h60"] == "True" and row["supported"] == "True"} for row in query}
    sigma = np.asarray(read_json(V3 / "data/V3_NORMALIZERS.json")["target_sigma"], dtype=np.float32)
    all_rows = []
    interaction_rows = []
    for bundle in sorted((V3 / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = read_json(bundle)
        if marker.get("status") != "PASS":
            continue
        run_id = bundle.parent.name
        parts = run_id.split("-")
        if len(parts) < 5:
            continue
        cell = str(marker.get("cell") or ("-".join(parts[1:4]) if parts[0] == "V3" else "UNKNOWN"))
        seed = int(marker["seed"])
        row_values = rows_for_prediction(data, bundle.parent / "primary.npz", sigma, strata_by_id)
        all_rows.extend(aggregate(row_values, cell, seed))
    write_csv(BENCH_ROOT / "d1/D1_LEARNED_RESULTS.csv", all_rows, list(all_rows[0].keys()) if all_rows else ["model"])
    for seed in sorted({row["seed"] for row in all_rows}):
        for horizon in (30, 60):
            for body in ("Hand", "Arm", "Full28"):
                vals = {}
                for model in ("S-HIST", "F-POOL-D", "F-TOK-D", "F-TOK-S"):
                    match = [row for row in all_rows if row["seed"] == seed and row["horizon_steps"] == horizon and row["body"] == body and row["stratum"] in {"low_supported", "high_supported"} and row["model"] == model]
                    vals[model] = {row["stratum"]: row["standardized_mae_episode_equal"] for row in match}
                for control, fusion in (("S-HIST", "F-TOK-D"), ("F-POOL-D", "F-TOK-D"), ("F-TOK-S", "F-TOK-D")):
                    if control not in vals or fusion not in vals or any(key not in vals[control] or key not in vals[fusion] for key in ("low_supported", "high_supported")):
                        continue
                    if any(vals[model].get(key) is None for model in (control, fusion) for key in ("low_supported", "high_supported")):
                        continue
                    low_gain = vals[control]["low_supported"] - vals[fusion]["low_supported"]
                    high_gain = vals[control]["high_supported"] - vals[fusion]["high_supported"]
                    interaction_rows.append({"seed": seed, "horizon_steps": horizon, "body": body, "control": control, "fusion": fusion, "gain_low": low_gain, "gain_high": high_gain, "high_minus_low": high_gain - low_gain})
    if interaction_rows:
        write_csv(BENCH_ROOT / "d1/D1_LEARNED_INTERACTIONS.csv", interaction_rows, list(interaction_rows[0].keys()))
    summary = {"status": "LEARNED_SPARSE_COMPLETE", "accepted_prediction_bundles": len({(row["model"], row["seed"], row["horizon_steps"]) for row in all_rows}), "rows": len(all_rows), "interaction_rows": len(interaction_rows), "visual_disambiguation": "LEARNED_SPARSE_EVALUATED; DENSE_PENDING", "dense_status": "not_yet_evaluated", "final_test_access": 0}
    atomic_write_json(BENCH_ROOT / "d1/D1_LEARNED_SUMMARY.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
