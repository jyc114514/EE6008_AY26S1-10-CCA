#!/usr/bin/env python3
"""Create compact six-config, three-seed train/dev and scheduler summaries."""
from __future__ import annotations
import os

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z'))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, values: list[dict[str, Any]]) -> None:
    if not values:
        raise ValueError(f"no rows to write: {path}")
    cols = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in values:
            w.writerow({k: ("NA" if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()})


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def relabel_source_table(source: Path, destination: Path, *, event_legacy: bool = False) -> None:
    data = rows(source)
    labels = {"Hand": "action[0:14]", "Arm": "action[14:28]", "Full28": "action[0:28]"}
    for row in data:
        if "body" not in row:
            continue
        previous = row["body"]
        if event_legacy:
            row["legacy_body_label"] = previous
            row["body_semantics_status"] = "legacy event evaluator group label; raw action slice mapping unresolved"
            del row["body"]
        else:
            row["body_legacy_label"] = previous
            row["body"] = labels.get(previous, previous)
            row["body_semantics_status"] = "numeric raw slice based on D3-EQ analysis helper; anatomy mapping unresolved"
    write(destination, data)


def main() -> int:
    raw = rows(OUT / "TRAIN_DEV_METRICS.csv")
    natural = [r for r in raw if r["stratum"] == "natural"]
    grouped: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in natural:
        grouped[(row["split"], row["config"], int(row["horizon_steps"]), row["region_raw"])].append(row)
    names = [
        "native_mae", "standardized_mae", "native_rmse", "standardized_rmse",
        "endpoint_native_mae", "endpoint_standardized_mae", "native_future_internal_fd",
        "safe_fd_normalized", "seam_error_native", "tv_pred_mean", "tv_gt_mean",
        "tv_ratio_group_weighted_means", "tv_ratio_window_p25", "tv_ratio_window_p50",
        "tv_ratio_window_p75", "tv_ratio_window_valid_count", "zero_gt_tv_windows",
        "pred_step_abs_mean",
        "gt_step_abs_mean", "pred_anchor_net_abs", "gt_anchor_net_abs",
        "pred_anchor_net_signed", "gt_anchor_net_signed", "direction_accuracy_active",
        "active_step_count_total", "near_zero_gt_step_count_total",
        "small_nonzero_gt_step_count_total", "early_native_mae", "mid_native_mae", "late_native_mae",
    ]
    summaries: list[dict[str, Any]] = []
    for (split, config, horizon, region), cell_rows in sorted(grouped.items()):
        by_seed = {int(r["seed"]): r for r in cell_rows}
        if set(by_seed) != {20261001, 20261002, 20261003}:
            raise AssertionError(f"seed coverage failure: {(split,config,horizon,region)} {sorted(by_seed)}")
        row: dict[str, Any] = {"split": split, "config": config, "horizon_steps": horizon,
                               "region_raw": region, "region_slice": by_seed[20261001]["region_slice"],
                               "n_windows": int(by_seed[20261001]["n_windows"]),
                               "n_episodes": int(by_seed[20261001]["n_episodes"]),
                               "n_source_groups": int(by_seed[20261001]["n_source_groups"]),
                               "region_semantics": "raw action coordinates; Hand/Arm mapping unresolved"}
        for metric in names:
            col = f"model_{metric}"
            vals = [float(by_seed[s][col]) for s in sorted(by_seed)]
            row[f"{metric}_seed_mean"] = mean(vals)
            row[f"{metric}_seed_min"] = min(vals)
            row[f"{metric}_seed_max"] = max(vals)
            row[f"{metric}_seed_20261001"] = vals[0]
            row[f"{metric}_seed_20261002"] = vals[1]
            row[f"{metric}_seed_20261003"] = vals[2]
        for baseline in ("persistence", "slope"):
            for metric in ("native_mae", "standardized_mae", "native_future_internal_fd", "safe_fd_normalized",
                           "tv_pred_mean", "tv_gt_mean", "tv_ratio_group_weighted_means",
                           "pred_step_abs_mean", "gt_step_abs_mean", "direction_accuracy_active"):
                vals = [float(by_seed[s][f"{baseline}_{metric}"]) for s in sorted(by_seed)]
                spread = max(vals) - min(vals)
                if spread > 1e-7:
                    raise AssertionError(f"shared {baseline} baseline differs across seeds: {(split,horizon,region,metric)} spread={spread}")
                row[f"{baseline}_{metric}"] = mean(vals)
        for metric in ("native_mae", "standardized_mae", "native_future_internal_fd", "safe_fd_normalized"):
            gains = [float(by_seed[s][f"gain_vs_persistence_{metric}"]) for s in sorted(by_seed)]
            row[f"gain_vs_persistence_{metric}_seed_mean"] = mean(gains)
            row[f"gain_vs_persistence_{metric}_seed_min"] = min(gains)
            row[f"gain_vs_persistence_{metric}_seed_max"] = max(gains)
            gains_slope = [float(by_seed[s][f"gain_vs_slope_{metric}"]) for s in sorted(by_seed)]
            row[f"gain_vs_slope_{metric}_seed_mean"] = mean(gains_slope)
        summaries.append(row)
    if len(summaries) != 72:
        raise AssertionError(f"expected 2 splits x 6 configs x 2 horizons x 3 raw regions = 72 rows, got {len(summaries)}")
    write(OUT / "TRAIN_DEV_SIX_CONFIG_SUMMARY.csv", summaries)
    primary = [r for r in summaries if r["config"] in {"TS11-D", "STATE-CTRL"}]
    write(OUT / "TRAIN_DEV_TS11D_STATECTRL_SUMMARY.csv", primary)

    result_root = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/analysis'))
    relabel_source_table(result_root / "D3_EQ_PRIMARY_POOLED.csv", OUT / "D3_EQ_PRIMARY_CONTRASTS_RELABELED_V3.csv")
    relabel_source_table(result_root / "D3_EQ_FACTORIAL_EFFECTS.csv", OUT / "D3_EQ_FACTORIAL_EFFECTS_RELABELED_V3.csv")
    relabel_source_table(result_root / "D3_EQ_RESULTS_BY_SEED.csv", OUT / "D3_EQ_RESULTS_BY_SEED_RELABELED_V3.csv")
    relabel_source_table(result_root / "D3_EQ_DENSE_RESULTS_BY_SEED.csv", OUT / "D3_EQ_DENSE_RESULTS_BY_SEED_RELABELED_V3.csv")
    relabel_source_table(result_root / "D3_EQ_DENSE_CONTRASTS_BY_SEED.csv", OUT / "D3_EQ_DENSE_CONTRASTS_BY_SEED_RELABELED_V3.csv")
    relabel_source_table(result_root / "D3_EQ_D4_ROLLING_PAIRED_EFFECTS.csv", OUT / "D3_EQ_D4_ROLLING_PAIRED_EFFECTS_RELABELED_V3.csv")
    relabel_source_table(result_root / "D3_EQ_D4_ROLLING_RESULTS_BY_SEED.csv", OUT / "D3_EQ_D4_ROLLING_RESULTS_BY_SEED_RELABELED_V3.csv")
    relabel_source_table(result_root / "D3_EQ_D2V3_EVENT_SECONDARY.csv", OUT / "D3_EQ_EVENT_SECONDARY_LEGACY_LABELS_V3.csv", event_legacy=True)

    curves = rows(OUT / "TRAINING_CURVE_CHECKPOINTS.csv")
    by_curve: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for r in curves:
        if r.get("status") == "present":
            by_curve[(r["config"], int(r["horizon_steps"]), int(r["epoch"]))].append(r)
    curve_summary = []
    for (config, horizon, epoch), group in sorted(by_curve.items()):
        lr = [float(r["lr"]) for r in group]
        weighted = [float(r["weighted_loss_sum_over_seen_rows"]) for r in group]
        unweighted = [float(r["unweighted_mean_loss"]) for r in group]
        curve_summary.append({"config": config, "horizon_steps": horizon, "epoch": epoch,
                              "n_seeds": len(group), "lr_mean": mean(lr), "lr_min": min(lr), "lr_max": max(lr),
                              "weighted_loss_mean": mean(weighted), "weighted_loss_min": min(weighted),
                              "weighted_loss_max": max(weighted), "unweighted_loss_mean": mean(unweighted),
                              "unweighted_loss_min": min(unweighted), "unweighted_loss_max": max(unweighted)})
    write(OUT / "TRAINING_CURVE_SUMMARY.csv", curve_summary)
    print(json.dumps({"status": "PASS", "natural_summary_rows": len(summaries),
                      "ts11d_statectrl_rows": len(primary), "curve_summary_rows": len(curve_summary),
                      "all_seed_rows_present": len(curves) == 144}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
