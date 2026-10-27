"""Post-hoc, source-group-aware descriptive D4 summary across three seeds."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "d4/D4_PAIR_METRICS.csv"
PROTOCOL = ROOT / "protocol/D4_CROSS_SEED_DESCRIPTIVE_V1.json"
OUTPUT = ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE.csv"
SUMMARY = ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE_SUMMARY.json"
SEEDS = ["20260950", "20260951", "20260952"]
CONTRASTS = {
    "video": ("F-TOK-D", "S-HIST"),
    "temporal": ("F-TOK-D", "F-POOL-D"),
    "dynamic": ("F-TOK-D", "F-TOK-S"),
}
ABS_METRICS = ["L_old", "L_new", "U", "R", "FD_old", "FD_new"]
DIFF_METRICS = ["delta_U", "delta_L_new", "delta_FD_new"]
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260927


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_matrix(frame: pd.DataFrame, value_col: str, groups: list[str]) -> np.ndarray:
    # The task contract is pair mean -> episode mean -> source-group mean.
    episode_values = frame.groupby(["seed", "source_group_id", "episode_index"], sort=True)[value_col].mean()
    source_values = episode_values.groupby(level=[0, 1], sort=True).mean()
    matrix = np.full((len(SEEDS), len(groups)), np.nan, dtype=np.float64)
    seed_index = {seed: i for i, seed in enumerate(SEEDS)}
    group_index = {group: i for i, group in enumerate(groups)}
    for (seed, group), value in source_values.items():
        matrix[seed_index[str(seed)], group_index[str(group)]] = float(value)
    if not np.isfinite(matrix).all():
        raise ValueError("cross-seed/source-group matrix has missing values")
    return matrix


def summarize(matrix: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    if matrix.shape != (3, 118):
        raise ValueError(f"expected seed x source-group matrix (3,118), got {matrix.shape}")
    estimate_by_seed = matrix.mean(axis=1)
    reps = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 250):
        stop = min(start + 250, BOOTSTRAPS)
        seed_draws = rng.integers(0, len(SEEDS), size=(stop - start, len(SEEDS)))
        group_draws = rng.integers(0, matrix.shape[1], size=(stop - start, matrix.shape[1]))
        selected = matrix[seed_draws[:, :, None], group_draws[:, None, :]]
        reps[start:stop] = selected.mean(axis=(1, 2))
    low, high = np.quantile(reps, [0.025, 0.975])
    return {
        "estimate_equal_seed_group_mean": float(estimate_by_seed.mean()),
        "seed_mean_sd_n3": float(estimate_by_seed.std(ddof=1)),
        "ci95_descriptive_low": float(low),
        "ci95_descriptive_high": float(high),
        "seed_20260950_estimate": float(estimate_by_seed[0]),
        "seed_20260951_estimate": float(estimate_by_seed[1]),
        "seed_20260952_estimate": float(estimate_by_seed[2]),
    }


def main() -> None:
    data = pd.read_csv(INPUT, dtype={"pair_id": "string", "model": "string", "seed": "string", "source_group_id": "string"})
    if set(data["final_test_access"].astype(int)) != {0}:
        raise ValueError("D4 input contains nonzero final-test access")
    data = data[data["seed"].isin(SEEDS)].copy()
    if set(data["seed"].unique()) != set(SEEDS):
        raise ValueError("expected exactly the three locked D4 seeds")
    expected = {"pair_id", "model", "seed", "horizon_steps", "body", "episode_index", "source_group_id", *ABS_METRICS}
    if not expected.issubset(data.columns):
        raise ValueError(f"D4 pair metrics lack required fields: {sorted(expected-set(data.columns))}")

    output_rows = []
    diagnostics = {"model_cells": 0, "contrast_cells": 0, "pair_join_checks": []}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for horizon in (30, 60):
        horizon_data = data[data["horizon_steps"].astype(int) == horizon]
        canonical = horizon_data[horizon_data["model"] == "F-TOK-D"]
        if len(canonical) == 0:
            raise ValueError(f"no candidate rows for H{horizon}")
        pair_meta = canonical[["pair_id", "episode_index", "source_group_id"]].drop_duplicates()
        if len(pair_meta) != 3600 or pair_meta["source_group_id"].nunique() != 118 or pair_meta["episode_index"].nunique() != 118:
            raise ValueError(f"H{horizon} pair/group support differs from 3600/118/118")
        if pair_meta.groupby("source_group_id")["episode_index"].nunique().max() != 1:
            raise ValueError("dense source group maps to multiple episodes; nested episode averaging required audit")
        groups = sorted(pair_meta["source_group_id"].astype(str).unique().tolist())

        for body in ("Arm", "Hand"):
            base = horizon_data[(horizon_data["horizon_steps"].astype(int) == horizon) & (horizon_data["body"] == body)]
            for model in ("F-TOK-D", "S-HIST", "F-POOL-D", "F-TOK-S"):
                for metric in ABS_METRICS:
                    frame = base[base["model"] == model]
                    rows_per_seed = frame.groupby("seed")["pair_id"].nunique().to_dict()
                    if any(int(rows_per_seed.get(seed, 0)) != 3600 for seed in SEEDS):
                        raise ValueError(f"model pair coverage mismatch: {model}/H{horizon}/{body}/{metric}")
                    matrix = group_matrix(frame, metric, groups)
                    output_rows.append({
                        "analysis": "per_model", "contrast": "", "model": model, "candidate": "", "control": "",
                        "horizon_steps": horizon, "body": body, "metric": metric,
                        **summarize(matrix, rng), "pair_count_per_seed": 3600,
                        "episode_count": 118, "source_group_count": 118,
                    })
                    diagnostics["model_cells"] += 1

            for contrast, (candidate, control) in CONTRASTS.items():
                left = base[base["model"] == candidate]
                right = base[base["model"] == control]
                keys = ["pair_id", "seed", "horizon_steps", "body", "episode_index", "source_group_id"]
                left = left[keys + ["U", "L_new", "FD_new"]].rename(columns={m: f"{m}_candidate" for m in ("U", "L_new", "FD_new")})
                right = right[keys + ["U", "L_new", "FD_new"]].rename(columns={m: f"{m}_control" for m in ("U", "L_new", "FD_new")})
                joined = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
                if len(joined) != 10_800 or not (joined["_merge"] == "both").all():
                    raise ValueError(f"paired D4 row join failed: {contrast}/H{horizon}/{body}")
                diagnostics["pair_join_checks"].append({"contrast": contrast, "horizon_steps": horizon, "body": body, "pairs_per_seed": 3600, "joined_rows": len(joined), "status": "PASS"})
                for output_metric, raw_metric in (("delta_U", "U"), ("delta_L_new", "L_new"), ("delta_FD_new", "FD_new")):
                    joined[output_metric] = joined[f"{raw_metric}_candidate"] - joined[f"{raw_metric}_control"]
                    matrix = group_matrix(joined, output_metric, groups)
                    output_rows.append({
                        "analysis": "paired_contrast", "contrast": contrast, "model": "", "candidate": candidate, "control": control,
                        "horizon_steps": horizon, "body": body, "metric": output_metric,
                        **summarize(matrix, rng), "pair_count_per_seed": 3600,
                        "episode_count": 118, "source_group_count": 118,
                    })
                    diagnostics["contrast_cells"] += 1

    fields = list(output_rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "status": "COMPLETE_POSTHOC_DESCRIPTIVE_EXTENSION",
        "protocol_id": "CORE_B_D4_CROSS_SEED_DESCRIPTIVE_V1",
        "protocol_sha256": sha256(PROTOCOL),
        "input_sha256": sha256(INPUT),
        "output_rows": len(output_rows),
        "per_model_cells": diagnostics["model_cells"],
        "paired_contrast_cells": diagnostics["contrast_cells"],
        "pair_join_checks": diagnostics["pair_join_checks"],
        "seeds": SEEDS,
        "independent_source_groups_per_horizon": 118,
        "pairs_per_seed_horizon": 3600,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_interpretation": "descriptive hierarchical bootstrap over three seeds and exact source groups; not confirmatory and no p-values",
        "final_test_access": 0,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
