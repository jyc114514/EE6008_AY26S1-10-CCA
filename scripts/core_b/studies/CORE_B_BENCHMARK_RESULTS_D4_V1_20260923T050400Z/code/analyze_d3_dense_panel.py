"""CPU-only per-query D3 dense-panel scores and factorial contrasts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_d3_d4_dense as common

ROOT = common.ROOT
EXT = ROOT / "d3_extension"
OUT = EXT / "results"
PROTOCOL = ROOT / "protocol/D3_DENSE_PANEL_ANALYSIS_V1.json"
SEEDS = (20261001, 20261002, 20261003)
HORIZONS = (30, 60)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260929
BODIES = {"Hand": slice(0, 14), "Arm": slice(14, 28), "Full28": slice(0, 28)}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    context = common.build_context()
    predictions = common.load_predictions(context)
    data = context["data"]
    n = len(context["dense_ids"])
    targets = np.empty((n, 60, 28), dtype=np.float32)
    episodes = data["episode_index"].astype(np.int64)
    frames = data["anchor_frame"].astype(np.int64)
    for index, (episode, frame) in enumerate(zip(episodes, frames)):
        _, action = context["episode_data"][int(episode)]
        if int(frame) + 60 >= len(action):
            raise ValueError(f"incomplete dense H60 target: {context['dense_ids'][index]}")
        targets[index] = action[int(frame)+1:int(frame)+61].astype(np.float32)
    groups = np.asarray([context["group_by_episode"][int(ep)] for ep in episodes], dtype="U64")
    ordered_groups = sorted(set(groups.tolist()))
    if len(ordered_groups) != 118 or len(set(episodes.tolist())) != 118:
        raise ValueError("dense panel must contain 118 episode-distinct source groups")
    counts = common.bootstrap_counts(len(ordered_groups), BOOTSTRAP_SEED)
    group_scores = {}
    results = []
    for (config, seed, horizon), prediction in sorted(predictions.items()):
        if prediction.shape != (3718, horizon, 28):
            raise ValueError(f"dense prediction shape mismatch: {config}/{seed}/{horizon}")
        truth = targets[:, :horizon]
        for body, sl in BODIES.items():
            metrics = {
                "mae": np.mean(np.abs(prediction[:, :, sl] - truth[:, :, sl]) / common.d4.SIGMA[None, None, sl], axis=(1, 2), dtype=np.float64),
                "fd": np.mean(np.abs(np.diff(prediction[:, :, sl], axis=1) - np.diff(truth[:, :, sl], axis=1)), axis=(1, 2), dtype=np.float64),
            }
            for metric, values in metrics.items():
                grouped = common.group_means(values, groups, ordered_groups)
                estimate, low, high = common.summarize_groups(grouped, counts)
                group_scores[(config, seed, horizon, body, metric)] = grouped
                results.append({
                    "config": config, "seed": seed, "horizon_steps": horizon,
                    "body": body, "metric": metric, "estimate": estimate,
                    "ci95_low": low, "ci95_high": high,
                    "episodes": 118, "source_groups": 118, "queries": 3718,
                    "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_unit": "exact video_sha256_fingerprint source group",
                    "final_test_access": 0,
                })
    write_csv(OUT / "D3_DENSE_PANEL_RESULTS.csv", results)

    contrasts = []
    primary = (
        ("TS11-D", "TS10-D", "TS11_vs_TS10", "spatial_addition_given_temporal_preserved"),
        ("TS11-D", "TS01-D", "TS11_vs_TS01", "temporal_addition_given_spatial_preserved"),
        ("TS11-D", "TS11-S", "TS11D_vs_TS11S", "dynamic_vs_static_video"),
        ("TS11-D", "STATE-CTRL", "TS11_vs_STATE", "full_video_vs_state_only"),
    )
    for horizon in HORIZONS:
        for seed in SEEDS:
            for candidate, control, name, meaning in primary:
                for body in BODIES:
                    for metric in ("mae", "fd"):
                        effect = group_scores[(candidate, seed, horizon, body, metric)] - group_scores[(control, seed, horizon, body, metric)]
                        estimate, low, high = common.summarize_groups(effect, counts)
                        contrasts.append({
                            "contrast": name, "interpretation": meaning, "candidate": candidate,
                            "control": control, "seed": seed, "horizon_steps": horizon,
                            "body": body, "metric": metric, "estimand": "candidate_minus_control; negative favors lower loss",
                            "estimate": estimate, "ci95_low": low, "ci95_high": high,
                            "episodes": 118, "source_groups": 118, "queries": 3718,
                            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
                            "final_test_access": 0,
                        })
            for body in BODIES:
                for metric in ("mae", "fd"):
                    l00 = group_scores[("TS00-D", seed, horizon, body, metric)]
                    l10 = group_scores[("TS10-D", seed, horizon, body, metric)]
                    l01 = group_scores[("TS01-D", seed, horizon, body, metric)]
                    l11 = group_scores[("TS11-D", seed, horizon, body, metric)]
                    derived = (
                        ("factorial_interaction_TS11_TS10_TS01_TS00", (l11-l10)-(l01-l00), "negative favors synergistic loss reduction"),
                        ("spatial_marginal_main_effect", ((l01-l00)+(l11-l10))/2.0, "negative favors spatial-bin preservation"),
                        ("temporal_marginal_main_effect", ((l10-l00)+(l11-l01))/2.0, "negative favors temporal preservation"),
                    )
                    for name, effect, meaning in derived:
                        estimate, low, high = common.summarize_groups(effect, counts)
                        contrasts.append({
                            "contrast": name, "interpretation": meaning, "candidate": "", "control": "",
                            "seed": seed, "horizon_steps": horizon, "body": body, "metric": metric,
                            "estimand": "paired exact-source-group factorial contrast",
                            "estimate": estimate, "ci95_low": low, "ci95_high": high,
                            "episodes": 118, "source_groups": 118, "queries": 3718,
                            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
                            "final_test_access": 0,
                        })
    write_csv(OUT / "D3_DENSE_PANEL_CONTRASTS.csv", contrasts)
    summary = {
        "status": "COMPLETE_POSTHOC_D3_DENSE_PANEL_ANALYSIS",
        "protocol_id": "CORE_B_D3_DENSE_PANEL_ANALYSIS_V1",
        "protocol_sha256": file_hash(PROTOCOL),
        "accepted_cells": len(predictions), "expected_cells": 36,
        "result_rows": len(results), "contrast_rows": len(contrasts),
        "episodes": 118, "source_groups": 118, "queries": 3718,
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "outcome_exposure": "post-hoc development dense-panel extension; no retraining or p-value search",
        "final_test_access": 0,
    }
    (OUT / "D3_DENSE_PANEL_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
