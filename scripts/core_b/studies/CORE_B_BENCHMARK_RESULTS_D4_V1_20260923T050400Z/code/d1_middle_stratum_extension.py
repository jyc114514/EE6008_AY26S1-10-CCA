"""Post-hoc D1 supported-middle stratum, reusing locked thresholds and groups."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import reanalyze_d1 as d1


ROOT = Path(__file__).resolve().parents[1]
QUERY_CSV = ROOT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv"
NEIGHBOR_NPZ = ROOT / "results/D1_SOURCE_GROUP_NEIGHBORS.npz"
SUMMARY_JSON = ROOT / "results/D1_SOURCE_GROUP_SUMMARY.json"
OUT_CSV = ROOT / "results/D1_MIDDLE_STRATUM_RESULTS.csv"
OUT_JSON = ROOT / "results/D1_MIDDLE_STRATUM_SUMMARY.json"
PROTOCOL = ROOT / "protocol/D1_MIDDLE_STRATUM_EXTENSION_V1.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with np.load(d1.DATA_PATH, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if set(np.unique(data["split_code"]).tolist()) != {0, 1}:
        raise ValueError("V3 primary input archive is not train/dev-only")

    dev_all = np.flatnonzero(data["split_code"] == 1)
    dev_ids = data["anchor_id"][dev_all].astype("U64")
    query_rows = d1.csv_rows(QUERY_CSV)
    query_by_id = {row["anchor_id"]: row for row in query_rows}
    if len(query_by_id) != len(query_rows) or set(query_by_id) != set(dev_ids.tolist()):
        raise ValueError("D1 query IDs do not exactly match V3 primary-dev IDs")

    parent = read_json(SUMMARY_JSON)
    q25 = float(parent["thresholds_train_only"]["dispersion_q25_h60"])
    q75 = float(parent["thresholds_train_only"]["dispersion_q75_h60"])
    q95 = float(parent["thresholds_train_only"]["neighbor_distance_k20_q95"])
    dispersion = np.asarray([float(query_by_id[aid]["dispersion_h60"]) for aid in dev_ids], dtype=np.float64)
    neighbor_distance = np.asarray([float(query_by_id[aid]["neighbor_distance_k20"]) for aid in dev_ids], dtype=np.float64)
    supported = np.asarray([query_by_id[aid]["supported"].lower() == "true" for aid in dev_ids], dtype=bool)
    expected_supported = neighbor_distance <= q95
    if not np.array_equal(supported, expected_supported):
        raise ValueError("query supported flag does not reproduce the locked train-only Q95 rule")
    middle = (dispersion > q25) & (dispersion < q75)
    selected = middle & supported

    with np.load(NEIGHBOR_NPZ, allow_pickle=False) as archive:
        neighbor_ids = archive["dev_anchor_ids"].astype("U64")
        nn_h30 = archive["nn_prediction_h30"].astype(np.float64)
        nn_h60 = archive["nn_prediction_h60"].astype(np.float64)
    if not np.array_equal(neighbor_ids, dev_ids):
        raise ValueError("D1 NN prediction rows are not in exact primary-dev ID order")

    groups = np.asarray([query_by_id[aid]["source_group_id"] for aid in dev_ids], dtype="U64")
    episodes = data["episode_index"][dev_all]
    if len(set(groups.tolist())) != 603 or len(np.unique(episodes)) != 603:
        raise ValueError("primary-dev episode/source-group support differs from the locked 603/603 contract")
    ordered_groups = sorted(set(groups.tolist()))
    group_index_map = {value: i for i, value in enumerate(ordered_groups)}
    group_index = np.asarray([group_index_map[value] for value in groups], dtype=np.int32)
    counts = d1.make_bootstrap_counts(len(ordered_groups), d1.BOOTSTRAPS, d1.BOOTSTRAP_SEED)

    anchors = data["anchor_command_native"][dev_all]
    truth_all = data["targets_native"][dev_all, :60]
    predictions: dict[tuple[str, int, int], np.ndarray] = {
        ("PERSISTENCE", 0, 30): np.repeat(anchors[:, None, :], 30, axis=1),
        ("PERSISTENCE", 0, 60): np.repeat(anchors[:, None, :], 60, axis=1),
        ("NN_MEDIAN", 0, 30): nn_h30,
        ("NN_MEDIAN", 0, 60): nn_h60,
    }
    for marker_path in sorted((d1.V3_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = read_json(marker_path)
        if marker.get("status") != "PASS" or marker.get("final_test_access") != 0:
            continue
        with np.load(marker_path.parent / "primary.npz", allow_pickle=False) as archive:
            ids = archive["anchor_id"].astype("U64")
            values = archive["predictions_native"].astype(np.float64)
            horizon = int(archive["horizon_steps"])
        if len(ids) != len(dev_ids) or set(ids.tolist()) != set(dev_ids.tolist()):
            raise ValueError(f"prediction ID mismatch: {marker_path.parent.name}")
        lookup = {value: index for index, value in enumerate(ids.tolist())}
        aligned = values[np.asarray([lookup[value] for value in dev_ids.tolist()], dtype=np.int64)]
        predictions[(str(marker["cell"]), int(marker["seed"]), horizon)] = aligned

    rows = []
    for (model, seed, horizon), prediction in sorted(predictions.items()):
        if prediction.shape != (len(dev_ids), horizon, 28):
            raise ValueError(f"prediction shape mismatch: {model}/{seed}/{horizon}")
        metrics = d1.window_metrics(prediction, truth_all[:, :horizon], d1.SIGMA)
        for metric_name, values in metrics.items():
            body, metric = metric_name.split("|")
            group_values = d1.group_values(values, group_index, selected, len(ordered_groups))
            estimate, low, high = d1.summarize_group_metric(group_values, counts)
            rows.append({
                "model": model,
                "seed": seed if seed else "NOT_APPLICABLE",
                "horizon_steps": horizon,
                "body": body,
                "metric": metric,
                "stratum": "middle_dispersion_supported",
                "N_source_group": int(np.isfinite(group_values).sum()),
                "N_episode": int(np.isfinite(group_values).sum()),
                "N_window": int(selected.sum()),
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "final_test_access": 0,
            })

    fields = list(rows[0]) if rows else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "COMPLETE_POSTHOC_DESCRIPTIVE_EXTENSION",
        "protocol_id": "CORE_B_D1_MIDDLE_STRATUM_EXTENSION_V1",
        "protocol_sha256": sha256(PROTOCOL),
        "parent_protocol_id": parent["protocol_id"],
        "parent_protocol_hash": "a0718dbec118c27d4211354f5888b58578c4d656645ed2e8eb8bfee5b3f67d2b",
        "parent_thresholds_reused_train_only": {"dispersion_q25_h60": q25, "dispersion_q75_h60": q75, "neighbor_distance_k20_q95": q95},
        "primary_dev_windows": int(len(dev_ids)),
        "middle_supported_windows": int(selected.sum()),
        "middle_supported_episodes": int(len(set(episodes[selected].tolist()))),
        "middle_supported_source_groups": int(len(set(groups[selected].tolist()))),
        "prediction_cells": int(sum(key[0] not in {"PERSISTENCE", "NN_MEDIAN"} for key in predictions)),
        "result_rows": len(rows),
        "bootstrap_replicates": d1.BOOTSTRAPS,
        "bootstrap_seed": d1.BOOTSTRAP_SEED,
        "bootstrap_unit": "exact video_sha256_fingerprint source group",
        "outcome_exposure": "posthoc; parent D1 outcomes were already inspected",
        "final_test_access": 0,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
