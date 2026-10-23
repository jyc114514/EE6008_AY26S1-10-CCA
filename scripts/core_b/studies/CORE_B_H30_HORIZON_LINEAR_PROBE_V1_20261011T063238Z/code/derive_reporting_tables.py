from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/HORIZON_METRICS_BY_SEED.csv"
OUT = ROOT / "results/HORIZON_METRICS_MEAN_BY_SYSTEM.csv"
FIG = ROOT / "figure_data/HORIZON_MEAN_BY_SYSTEM.csv"
METRICS = [
    "endpoint_position_standardized_mae", "prefix_position_standardized_mae",
    "endpoint_position_native_mae", "prefix_position_native_mae",
    "position_endpoint_skill_vs_persistence", "position_prefix_skill_vs_persistence",
    "future_internal_fd_native_mae", "future_internal_fd_safe_normalized",
    "future_internal_fd_native_skill_vs_persistence", "future_internal_fd_safe_skill_vs_persistence",
    "seam_error_native", "seam_error_safe_normalized", "seam_pred_step_abs_native",
    "seam_gt_step_abs_native", "tv_pred_episode_equal_mean", "tv_gt_episode_equal_mean",
    "tv_ratio_aggregate", "direction_accuracy_active_episode_equal",
    "positive_recall_episode_equal", "negative_recall_episode_equal",
    "balanced_direction_accuracy_episode_equal",
    "active_components", "positive_components", "negative_components",
    "direction_episodes_active", "direction_episodes_both_signs",
]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, fields, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def main():
    rows = read_rows(INPUT)
    if len(rows) != 168:
        raise AssertionError(f"unexpected summary row count: {len(rows)}")
    groups = {}
    for r in rows:
        key = (r["split"], r["system"], r["family"], r["query"], int(r["horizon"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (split, system, family, query, horizon), items in sorted(groups.items()):
        seeds = [r for r in items if r["seed"] in ("20261001", "20261002", "20261003")]
        single = [r for r in items if r["seed"] == "NA"]
        if seeds and len(seeds) != 3:
            raise AssertionError(f"fixed seed count is not three: {split}/{system}/H{horizon}")
        if (seeds and single) or (not seeds and len(single) != 1):
            raise AssertionError(f"unexpected seed/single-fit mix: {split}/{system}/H{horizon}")
        use = seeds if seeds else single
        row = {
            "split": split, "system": system, "family": family, "query": query,
            "horizon": horizon, "seed_count": len(seeds),
            "seed_sample_sd_reported": bool(seeds),
            "actual_target_delta_ms_mean": float(np.mean([float(r["actual_target_delta_ms_mean"]) for r in use])),
            "n_episodes": int(use[0]["episodes"]), "n_groups": int(use[0]["source_groups"]),
            "aggregation": "mean of seed-level losses; prediction arrays not ensembled" if seeds else "single deterministic model; no seed SD",
        }
        for metric in METRICS:
            vals = np.asarray([float(r[metric]) for r in use], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_seed_sample_sd"] = float(np.std(finite, ddof=1)) if seeds and len(finite) == 3 else float("nan")
        out.append(row)
    fields = list(out[0].keys())
    write_rows(OUT, fields, out)
    write_rows(FIG, fields, out)
    print(json.dumps({"status": "PASS_SEED_LOSS_AGGREGATES", "rows": len(out), "metrics": len(METRICS)}, sort_keys=True))


if __name__ == "__main__":
    main()

