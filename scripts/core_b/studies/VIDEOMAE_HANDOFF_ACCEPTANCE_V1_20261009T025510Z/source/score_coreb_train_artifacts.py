#!/usr/bin/env python3
"""Independently score the already-saved target-blind Core B train prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from independent_core_b_metrics import (HORIZONS, METRICS, MODES, REGIONS, SEEDS,
                                        activity_threshold, episode_group_map, grouped_scores,
                                        read_bundle, sha256, window_metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "audit-root", "handoff-root", "team-root", "v3-root", "split-manifest", "train-target-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    project, audit, handoff, team, v3, split, target_cache = [p.resolve() for p in
        (args.project_root, args.audit_root, args.handoff_root, args.team_root, args.v3_root,
         args.split_manifest, args.train_target_cache)]
    index = json.loads((handoff / "HANDOFF_INDEX.json").read_text(encoding="utf-8"))
    inputs_path = v3 / "data/V3_MAIN_INPUTS.npz"
    normalizer_path = v3 / "data/V3_NORMALIZERS.json"
    norm = json.loads(normalizer_path.read_text(encoding="utf-8"))
    sigma = np.asarray(norm["target_sigma"], dtype=np.float32)
    safe_fd = np.asarray(norm["safe_fd_scale"], dtype=np.float32)
    with np.load(inputs_path, allow_pickle=False) as z:
        split_codes = z["split_code"].copy()
        all_ids = z["anchor_id"].astype("U64", copy=True)
        all_eps = z["episode_index"].astype(np.int64, copy=True)
        train_rows = np.flatnonzero(split_codes == 0)
        train_ids_input = all_ids[train_rows]
        train_eps_input = all_eps[train_rows]
    if (len(train_rows), len(np.unique(train_eps_input))) != (17757, 2712):
        raise AssertionError("Core B train population differs from locked source")
    with np.load(target_cache, allow_pickle=False) as z:
        train_ids = z["anchor_id"].astype("U64", copy=True)
        train_eps = z["episode_index"].astype(np.int64, copy=True)
        anchors = np.asarray(z["anchor_command_native"], dtype=np.float32)
        targets60 = np.asarray(z["targets_native_60"], dtype=np.float32)
    if not np.array_equal(train_ids, train_ids_input) or not np.array_equal(train_eps, train_eps_input):
        raise AssertionError("train-only target cache order differs from primary V3 inputs")
    if targets60.shape != (17757, 60, 28) or anchors.shape != (17757, 28):
        raise AssertionError("train-only target cache shape mismatch")
    mapping = episode_group_map(split)
    groups = np.asarray([mapping[int(ep)][1] for ep in train_eps], dtype="U128")
    if len(np.unique(groups)) != 2712 or any(mapping[int(ep)][0] != "train" for ep in np.unique(train_eps)):
        raise AssertionError("train source-group/split mapping mismatch")
    threshold, fallback = activity_threshold(targets60)

    # Inference completed in prior runs; this separate scorer reads targets and saved predictions only.
    records = []
    old_cell_metrics = []
    cells = {(c["mode"], int(c["horizon"]), int(c["seed"])): c for c in index["core_b"]["cells"]}
    for mode in MODES:
        for horizon in HORIZONS:
            target = targets60[:, :horizon]
            for seed in SEEDS:
                cell = cells[(mode, horizon, seed)]
                run = Path(cell["run_root"])
                status = json.loads((run / "RUN_STATUS.json").read_text(encoding="utf-8"))
                prediction_path = run / "predictions_train.npz"
                if status.get("status") != "PASS_TRAIN_AND_EXPORT" or sha256(prediction_path) != status.get("train_prediction_sha256"):
                    raise AssertionError(f"VideoMAE train prediction hash/status mismatch: {prediction_path}")
                pred = read_bundle(prediction_path, expected_ids=train_ids,
                                   expected_episodes=train_eps, prediction_key="predictions_native")
                if pred.shape != (17757, horizon, 28):
                    raise AssertionError(f"VideoMAE train prediction shape mismatch {pred.shape}")
                model_name = "VideoMAE" if mode == "TS11-D" else "VM-STATE-CTRL"
                eval_report = json.loads(Path(cell["metrics"]).read_text(encoding="utf-8"))
                train_models = [(model_name, pred, eval_report["panels"]["train"])]
                team_run_id = f"D3EQ-CS-{mode}-H{horizon}-S{seed}"
                team_path = team / f"predictions/train/{team_run_id}.npz"
                team_sidecar_path = team_path.with_suffix(".json")
                team_sidecar = json.loads(team_sidecar_path.read_text(encoding="utf-8"))
                if (team_sidecar.get("status") != "PASS" or team_sidecar.get("split") != "train"
                        or team_sidecar.get("rows") != 17757 or team_sidecar.get("final_test_access") != 0
                        or team_sidecar.get("target_arrays_loaded") is not False
                        or team_sidecar.get("prediction_sha256") != sha256(team_path)):
                    raise AssertionError(f"team train prediction sidecar/hash/split check failed: {team_path}")
                team_pred = read_bundle(team_path, expected_ids=train_ids, expected_episodes=train_eps,
                                        prediction_key="predictions_native")
                if team_pred.shape != (17757, horizon, 28):
                    raise AssertionError(f"team train prediction shape mismatch: {team_pred.shape}")
                team_name = "V-JEPA" if mode == "TS11-D" else "TEAM-STATE-CTRL"
                train_models.append((team_name, team_pred, None))
                for score_name, score_pred, stored_panel in train_models:
                    for region, sl in REGIONS.items():
                        per_window = window_metrics(score_pred[:, :, sl], target[:, :, sl], anchors[:, sl],
                                                    sigma[sl], safe_fd[sl], threshold[sl])
                        for metric in METRICS:
                            aggregate, _ = grouped_scores(per_window[metric], train_eps, groups)
                            for weighting, value in aggregate.items():
                                records.append({"model": score_name, "mode": mode, "horizon": horizon, "seed": seed,
                                                "region_raw": region, "region_slice": f"{sl.start}:{sl.stop}",
                                                "anatomical_label": "unresolved", "split": "train",
                                                "aggregation": weighting, "metric": metric, "value": value,
                                                "n_windows": 17757, "n_episodes": 2712, "n_source_groups": 2712})
                            if stored_panel is not None and metric in ("standardized_mae", "native_future_internal_fd"):
                                original_label = "VideoMAE" if mode == "TS11-D" else "STATE-CTRL"
                                original = float(stored_panel[region][original_label][metric])
                                independent = aggregate["source_group_equal"]
                                old_cell_metrics.append(abs(float(independent) - original))

    path = audit / "CORE_B_PER_SEED_METRICS.csv"
    prior = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    prior = [r for r in prior if r.get("split") != "train"]
    columns = list(prior[0]) if prior else list(records[0])
    for row in records:
        for key in row:
            if key not in columns: columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore"); w.writeheader(); w.writerows(prior + records)

    qa = {"status": "PASS_SAVED_CORE_B_TRAIN_ARTIFACT_SCORE" if max(old_cell_metrics, default=float("inf")) <= 1e-7 else "NEEDS_REVIEW",
          "scorer_sha256": sha256(Path(__file__).resolve()), "train_prediction_artifacts_scored": 24,
          "combined_saved_train_dev_prediction_bundles_scored": 48,
          "rows_per_bundle": 17757, "episodes": 2712, "source_groups": 2712,
          "group_equals_episode_support": True, "train_target_cache_sha256": sha256(target_cache),
          "target_fit_source": "primary train only", "activity_threshold_fit_source": "same primary train-only H60 targets",
          "max_abs_vs_saved_handoff_train_metrics": max(old_cell_metrics), "metric_comparisons": len(old_cell_metrics),
          "checkpoint_replayed": False, "final_test_access": 0,
          "note": "These are scores of existing train prediction NPZs from both models; they do not count as checkpoint replay."}
    out = audit / "metrics/core_b/CORE_B_TRAIN_ARTIFACT_SCORE_QA.json"
    out.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(qa, allow_nan=False))
    return 0 if qa["status"] == "PASS_SAVED_CORE_B_TRAIN_ARTIFACT_SCORE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
