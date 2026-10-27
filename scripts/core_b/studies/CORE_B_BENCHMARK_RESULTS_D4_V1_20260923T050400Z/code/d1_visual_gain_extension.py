"""Paired source-group bootstrap of D1 visual gains across train-defined strata."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import reanalyze_d1 as d1


ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv"
NEIGHBORS = ROOT / "results/D1_SOURCE_GROUP_NEIGHBORS.npz"
SUMMARY = ROOT / "results/D1_SOURCE_GROUP_SUMMARY.json"
OUTPUT = ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS.csv"
OUTPUT_SUMMARY = ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS_SUMMARY.json"
PROTOCOL = ROOT / "protocol/D1_VISUAL_GAIN_EXTENSION_V1.json"
MODELS = ("S-HIST", "F-TOK-D", "F-POOL-D", "F-TOK-S")
CONTROLS = ("S-HIST", "F-POOL-D", "F-TOK-S")
STRATA_ORDER = ("natural_overall", "supported_overall", "low_dispersion_supported", "middle_dispersion_supported", "high_dispersion_supported")


def file_hash(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with np.load(d1.DATA_PATH, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if set(np.unique(data["split_code"]).tolist()) != {0, 1}:
        raise ValueError("V3 input archive must contain only train/dev rows")
    dev = np.flatnonzero(data["split_code"] == 1)
    ids = data["anchor_id"][dev].astype("U64").tolist()
    query_rows = d1.csv_rows(QUERY)
    query = {row["anchor_id"]: row for row in query_rows}
    if len(query) != len(query_rows) or set(query) != set(ids):
        raise ValueError("D1 query IDs do not exactly match primary-dev IDs")

    parent = json.loads(SUMMARY.read_text(encoding="utf-8"))
    q25 = float(parent["thresholds_train_only"]["dispersion_q25_h60"])
    q75 = float(parent["thresholds_train_only"]["dispersion_q75_h60"])
    q95 = float(parent["thresholds_train_only"]["neighbor_distance_k20_q95"])
    dispersion = np.asarray([float(query[aid]["dispersion_h60"]) for aid in ids])
    distance = np.asarray([float(query[aid]["neighbor_distance_k20"]) for aid in ids])
    supported = np.asarray([query[aid]["supported"].lower() == "true" for aid in ids])
    low = np.asarray([query[aid]["low_h60"].lower() == "true" for aid in ids])
    high = np.asarray([query[aid]["high_h60"].lower() == "true" for aid in ids])
    if not np.array_equal(supported, distance <= q95) or not np.array_equal(low, dispersion <= q25) or not np.array_equal(high, dispersion >= q75):
        raise ValueError("D1 query flags do not reproduce the locked train-only thresholds")
    middle = (dispersion > q25) & (dispersion < q75)
    masks = {
        "natural_overall": np.ones(len(ids), dtype=bool),
        "supported_overall": supported,
        "low_dispersion_supported": low & supported,
        "middle_dispersion_supported": middle & supported,
        "high_dispersion_supported": high & supported,
    }

    source_groups = np.asarray([query[aid]["source_group_id"] for aid in ids], dtype="U64")
    episodes = data["episode_index"][dev]
    if len(np.unique(episodes)) != 603 or len(set(source_groups.tolist())) != 603:
        raise ValueError("D1 primary-dev support must remain 603 episodes/source groups")
    ordered_groups = sorted(set(source_groups.tolist()))
    group_map = {group: index for index, group in enumerate(ordered_groups)}
    group_index = np.asarray([group_map[group] for group in source_groups], dtype=np.int32)
    bootstrap = d1.make_bootstrap_counts(len(ordered_groups), d1.BOOTSTRAPS, d1.BOOTSTRAP_SEED)
    truth = data["targets_native"][dev, :60]

    prediction_map = {}
    for marker_path in sorted((d1.V3_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "PASS" or marker.get("final_test_access") != 0:
            continue
        model = str(marker["cell"])
        if model not in MODELS:
            continue
        with np.load(marker_path.parent / "primary.npz", allow_pickle=False) as archive:
            bundle_ids = archive["anchor_id"].astype("U64").tolist()
            predictions = archive["predictions_native"].astype(np.float64)
            horizon = int(archive["horizon_steps"])
        if len(bundle_ids) != len(ids) or set(bundle_ids) != set(ids):
            raise ValueError(f"D1 bundle ID mismatch: {marker_path.parent.name}")
        row_lookup = {anchor_id: index for index, anchor_id in enumerate(bundle_ids)}
        predictions = predictions[np.asarray([row_lookup[anchor_id] for anchor_id in ids], dtype=np.int64)]
        prediction_map[(model, int(marker["seed"]), horizon)] = predictions
    required_cells = {(model, seed, horizon) for model in MODELS for seed in (20260950, 20260951, 20260952) for horizon in (30, 60)}
    if set(prediction_map) != required_cells:
        raise ValueError(f"D1 gain extension requires 36 accepted V3 cells; found {len(prediction_map)}")

    grouped = {}
    for key, prediction in prediction_map.items():
        grouped[key] = {}
        for metric_key, values in d1.window_metrics(prediction, truth[:, :key[2]], d1.SIGMA).items():
            body, metric = metric_key.split("|")
            grouped[key][(body, metric)] = {
                stratum: d1.group_values(values, group_index, mask, len(ordered_groups))
                for stratum, mask in masks.items()
            }

    output = []
    for seed in (20260950, 20260951, 20260952):
        for horizon in (30, 60):
            for body in ("Hand", "Arm", "Full28"):
                for metric in ("mae", "fd"):
                    for control in CONTROLS:
                        control_values = grouped[(control, seed, horizon)][(body, metric)]
                        video_values = grouped[("F-TOK-D", seed, horizon)][(body, metric)]
                        gains = {}
                        for stratum in STRATA_ORDER:
                            gain = control_values[stratum] - video_values[stratum]
                            gains[stratum] = gain
                            estimate, low_ci, high_ci = d1.summarize_group_metric(gain, bootstrap)
                            output.append({
                                "contrast": "visual_gain_control_minus_F_TOK_D",
                                "control": control,
                                "candidate": "F-TOK-D",
                                "seed": seed,
                                "horizon_steps": horizon,
                                "body": body,
                                "metric": metric,
                                "stratum": stratum,
                                "estimate": estimate,
                                "ci95_low": low_ci,
                                "ci95_high": high_ci,
                                "N_source_groups": int(np.isfinite(gain).sum()),
                                "N_windows": int(masks[stratum].sum()),
                                "bootstrap_replicates": d1.BOOTSTRAPS,
                                "bootstrap_seed": d1.BOOTSTRAP_SEED,
                                "final_test_access": 0,
                            })
                        gain_low = gains["low_dispersion_supported"]
                        gain_high = gains["high_dispersion_supported"]
                        reps = d1.boot_mean(gain_high, bootstrap) - d1.boot_mean(gain_low, bootstrap)
                        finite = reps[np.isfinite(reps)]
                        low_ci, high_ci = np.quantile(finite, [0.025, 0.975])
                        output.append({
                            "contrast": "high_minus_low_visual_gain",
                            "control": control,
                            "candidate": "F-TOK-D",
                            "seed": seed,
                            "horizon_steps": horizon,
                            "body": body,
                            "metric": metric,
                            "stratum": "high_gain_minus_low_gain",
                            "estimate": float(np.nanmean(gain_high) - np.nanmean(gain_low)),
                            "ci95_low": float(low_ci),
                            "ci95_high": float(high_ci),
                            "N_source_groups": int(np.isfinite(gain_low).sum()),
                            "N_windows": int(masks["low_dispersion_supported"].sum() + masks["high_dispersion_supported"].sum()),
                            "bootstrap_replicates": d1.BOOTSTRAPS,
                            "bootstrap_seed": d1.BOOTSTRAP_SEED,
                            "final_test_access": 0,
                        })

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "status": "COMPLETE_POSTHOC_PAIRED_GAIN_EXTENSION",
        "protocol_id": "CORE_B_D1_VISUAL_GAIN_EXTENSION_V1",
        "protocol_sha256": file_hash(PROTOCOL),
        "parent_protocol_id": parent["protocol_id"],
        "parent_protocol_hash": "a0718dbec118c27d4211354f5888b58578c4d656645ed2e8eb8bfee5b3f67d2b",
        "train_only_thresholds_reused": {"A60_Q25": q25, "A60_Q75": q75, "k20_distance_Q95": q95},
        "strata_support": {key: {"windows": int(mask.sum()), "source_groups": int(len(set(source_groups[mask].tolist())))} for key, mask in masks.items()},
        "accepted_v3_cells": len(prediction_map),
        "rows": len(output),
        "bootstrap_replicates": d1.BOOTSTRAPS,
        "bootstrap_seed": d1.BOOTSTRAP_SEED,
        "bootstrap_unit": "exact video_sha256_fingerprint source group",
        "effect_sign": "positive gain means control error minus F-TOK-D error and favors F-TOK-D",
        "outcome_exposure": "posthoc descriptive extension; no p-values or confirmatory claims",
        "final_test_access": 0,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
