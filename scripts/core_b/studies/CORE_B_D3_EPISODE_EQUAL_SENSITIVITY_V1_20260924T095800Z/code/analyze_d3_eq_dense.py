"""CPU scoring of old D3 and D3-EQ on the fixed 3,718-query dense dev panel."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OLD_PRED_ROOT = RESULT_ROOT / "d3_extension/predictions"
NEW_PRED_ROOT = EQ_ROOT / "d3/dense_predictions"
INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
EXT_CACHE = RESULT_ROOT / "d3_extension/cache/D3_DENSE_EXTRACTION_RUN_MANIFEST.json"
SPLIT_MANIFEST = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
SEEDS = (20261001, 20261002, 20261003)
HORIZONS = (30, 60)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
CONTRASTS = (
    ("TS11-D", "TS10-D", "TS11D_vs_TS10D"),
    ("TS11-D", "TS01-D", "TS11D_vs_TS01D"),
    ("TS11-D", "TS11-S", "TS11D_vs_TS11S"),
    ("TS11-D", "STATE-CTRL", "TS11D_vs_STATE"),
)
BODIES = {"Hand": slice(0, 14), "Arm": slice(14, 28), "Full28": slice(0, 28)}
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260929
OUT = EQ_ROOT / "d3/analysis"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_grid(root: Path, expected_ids: np.ndarray) -> dict[tuple[str, int, int], np.ndarray]:
    selected = {}
    for sidecar in sorted(root.glob("*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            continue
        path = sidecar.with_suffix(".npz")
        if not path.is_file() or sha256_file(path) != record.get("prediction_sha256"):
            raise AssertionError(f"dense prediction hash mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            ids = data["dense_id"].astype("U96")
            pred = data["predictions_native"].astype(np.float32, copy=True)
            config = str(data["config"].item())
            seed = int(data["seed"].item())
            horizon = int(data["horizon_steps"])
        if len(set(ids.tolist())) != 3718 or set(ids.tolist()) != set(expected_ids.tolist()):
            raise AssertionError(f"dense prediction ID set mismatch: {path.name}")
        if pred.shape != (3718, horizon, 28) or not np.isfinite(pred).all():
            raise AssertionError(f"dense prediction shape/finite mismatch: {path.name}")
        id_index = {str(value): i for i, value in enumerate(ids)}
        pred = pred[np.asarray([id_index[str(value)] for value in expected_ids], dtype=np.int64)]
        key = (config, seed, horizon)
        if key in selected:
            raise AssertionError(f"duplicate dense prediction slot: {key}")
        selected[key] = pred
    if len(selected) != 36:
        raise AssertionError(f"dense prediction grid incomplete: {len(selected)}/36 in {root}")
    return selected


def main() -> int:
    protocol = json.loads((EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    cache_audit = json.loads((EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json").read_text(encoding="utf-8"))
    group_audit = json.loads((EQ_ROOT / "qa/D3_EQ_SOURCE_GROUP_AUDIT.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED" or cache_audit.get("status") != "PASS" or group_audit.get("status") != "PASS":
        raise AssertionError("locked dense/source-group prerequisites are not accepted")

    with np.load(INPUTS, allow_pickle=False) as data:
        dense_ids = data["dense_id"].astype("U96", copy=True)
        episodes = data["episode_index"].astype(np.int64, copy=True)
        tasks = data["original_task_index"].astype(np.int64, copy=True)
        frames = data["anchor_frame"].astype(np.int64, copy=True)
    if len(dense_ids) != 3718 or len(set(dense_ids.tolist())) != 3718 or len(np.unique(episodes)) != 118:
        raise AssertionError("fixed dense panel IDs/groups differ from the locked 3718/118 design")

    source_registry = json.loads((V3_ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    raw_root = Path(source_registry["raw_root"]).resolve(strict=True)
    anchor_rows = [row for row in csv.DictReader((V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8")) if row["split"] == "dev" and row["population"] == "primary"]
    meta_by_episode = {}
    for row in anchor_rows:
        meta_by_episode.setdefault(int(row["episode_index"]), row)
    if not set(episodes.tolist()).issubset(meta_by_episode):
        raise AssertionError("dense panel episode not present in primary-dev anchor manifest")
    episode_actions = {}
    for episode in sorted(set(episodes.tolist())):
        row = meta_by_episode[episode]
        raw_path = (raw_root / row["parquet_path"]).resolve(strict=True)
        if not raw_path.is_relative_to(raw_root):
            raise AssertionError("dense dev action path escaped the locked raw root")
        table = pq.read_table(raw_path, columns=["timestamp", "action"])
        timestamps = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        dt = np.diff(timestamps)
        if actions.ndim != 2 or actions.shape[1] != 28 or not np.isfinite(actions).all() or not np.isfinite(timestamps).all():
            raise AssertionError(f"dense dev action schema/finite check failed for episode {episode}")
        if not len(dt) or np.any(dt <= 0) or np.max(np.abs(dt - 1.0 / 30.0)) > 1e-5:
            raise AssertionError(f"dense dev episode is not verified 30Hz: {episode}")
        episode_actions[episode] = (timestamps, actions)
    targets = np.empty((3718, 60, 28), dtype=np.float32)
    anchor_times = np.empty(3718, dtype=np.float64)
    for index, (episode, frame) in enumerate(zip(episodes, frames)):
        timestamps, actions = episode_actions[int(episode)]
        if int(frame) + 60 >= len(actions):
            raise AssertionError(f"dense H60 target out of bounds: {dense_ids[index]}")
        targets[index] = actions[int(frame) + 1:int(frame) + 61].astype(np.float32)
        anchor_times[index] = timestamps[int(frame)]

    normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    split_rows = pq.read_table(SPLIT_MANIFEST, columns=["episode_index", "split", "video_sha256_fingerprint"]).to_pylist()
    source_group = {
        int(row["episode_index"]): str(row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}")
        for row in split_rows if row["split"] in {"train", "validation"}
    }
    query_groups = np.asarray([source_group[int(episode)] for episode in episodes], dtype="U96")
    ordered_groups = np.asarray(sorted(set(query_groups.tolist())), dtype="U96")
    if len(ordered_groups) != 118:
        raise AssertionError("dense panel source fingerprint count is not 118")
    unique_episodes = len(set(episodes.tolist()))
    if unique_episodes != len(ordered_groups):
        raise AssertionError("dense panel episode/source-group mapping is not one-to-one")

    old_predictions = read_prediction_grid(OLD_PRED_ROOT, dense_ids)
    new_predictions = read_prediction_grid(NEW_PRED_ROOT, dense_ids)
    expected_slots = {(config, seed, horizon) for config in CONFIGS for seed in SEEDS for horizon in HORIZONS}
    if set(old_predictions) != expected_slots or set(new_predictions) != expected_slots:
        raise AssertionError("dense old/new prediction grids do not cover all 36 cells")

    rng_by_h = {h: np.random.default_rng(BOOTSTRAP_SEED + h).integers(0, len(ordered_groups), size=(BOOTSTRAPS, len(ordered_groups)), dtype=np.int32) for h in HORIZONS}
    def grouped(values: np.ndarray) -> np.ndarray:
        output = np.empty(len(ordered_groups), dtype=np.float64)
        for i, group in enumerate(ordered_groups):
            local = values[query_groups == group]
            if not len(local):
                raise AssertionError("missing dense source group")
            output[i] = float(local.mean())
        return output
    def score(pred: np.ndarray, truth: np.ndarray, horizon: int) -> dict[tuple[str, str], np.ndarray]:
        out = {}
        for body, sl in BODIES.items():
            out[(body, "standardized_mae")] = np.mean(np.abs(pred[:, :horizon, sl] - truth[:, :horizon, sl]) / sigma[None, None, sl], axis=(1, 2), dtype=np.float64)
            out[(body, "native_future_internal_fd")] = np.mean(np.abs(np.diff(pred[:, :horizon, sl], axis=1) - np.diff(truth[:, :horizon, sl], axis=1)), axis=(1, 2), dtype=np.float64)
        return out
    old_scores, new_scores = {}, {}
    for key, pred in old_predictions.items():
        for (body, metric), values in score(pred, targets, key[2]).items():
            old_scores[(*key, body, metric)] = grouped(values)
    for key, pred in new_predictions.items():
        for (body, metric), values in score(pred, targets, key[2]).items():
            new_scores[(*key, body, metric)] = grouped(values)

    results = []
    for label, scores in (("OLD_D3_ACTUAL", old_scores), ("D3_EQ_V1", new_scores)):
        for (config, seed, horizon, body, metric), group_values in sorted(scores.items()):
            samples = group_values[rng_by_h[horizon]].mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            results.append({
                "study": label,
                "config": config,
                "seed": seed,
                "horizon_steps": horizon,
                "body": body,
                "metric": metric,
                "estimate": float(group_values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "queries": 3718,
                "source_groups": len(ordered_groups),
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_unit": "video_sha256_fingerprint source group; query means within group",
                "final_test_access": 0,
            })
    write_csv(OUT / "D3_EQ_DENSE_RESULTS_BY_SEED.csv", results)

    contrast_rows = []
    objective_rows = []
    for horizon in HORIZONS:
        bidx = rng_by_h[horizon]
        for candidate, control, contrast in CONTRASTS:
            for body in BODIES:
                for metric in ("standardized_mae", "native_future_internal_fd"):
                    per_seed = {}
                    for seed in SEEDS:
                        per_seed[seed] = new_scores[(candidate, seed, horizon, body, metric)] - new_scores[(control, seed, horizon, body, metric)]
                        values = per_seed[seed]
                        samples = values[bidx].mean(axis=1)
                        low, high = np.quantile(samples, [0.025, 0.975])
                        contrast_rows.append({
                            "contrast": contrast, "candidate": candidate, "control": control,
                            "seed": seed, "horizon_steps": horizon, "body": body, "metric": metric,
                            "estimand": "candidate_minus_control; lower loss favors candidate",
                            "estimate": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high),
                            "queries": 3718, "source_groups": len(ordered_groups),
                            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_unit": "video fingerprint source group",
                            "final_test_access": 0,
                        })
                    pooled = np.mean(np.stack([per_seed[s] for s in SEEDS]), axis=0)
                    samples = pooled[bidx].mean(axis=1)
                    low, high = np.quantile(samples, [0.025, 0.975])
                    contrast_rows.append({
                        "contrast": contrast, "candidate": candidate, "control": control,
                        "seed": "THREE_SEED_DESCRIPTIVE", "horizon_steps": horizon, "body": body, "metric": metric,
                        "estimand": "mean of three matched seed-level group contrasts; descriptive",
                        "estimate": float(pooled.mean()), "ci95_low": float(low), "ci95_high": float(high),
                        "seed_mean": float(np.mean([per_seed[s].mean() for s in SEEDS])),
                        "seed_sd": float(np.std([per_seed[s].mean() for s in SEEDS], ddof=1)),
                        "queries": 3718, "source_groups": len(ordered_groups),
                        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_unit": "video fingerprint source group",
                        "final_test_access": 0,
                    })
        for config in CONFIGS:
            for body in BODIES:
                for metric in ("standardized_mae", "native_future_internal_fd"):
                    per_seed = {seed: new_scores[(config, seed, horizon, body, metric)] - old_scores[(config, seed, horizon, body, metric)] for seed in SEEDS}
                    pooled = np.mean(np.stack([per_seed[s] for s in SEEDS]), axis=0)
                    samples = pooled[bidx].mean(axis=1)
                    low, high = np.quantile(samples, [0.025, 0.975])
                    objective_rows.append({
                        "comparison": "D3_EQ_V1_minus_OLD_D3_ACTUAL",
                        "config": config, "horizon_steps": horizon, "body": body, "metric": metric,
                        "estimate_three_seed_mean": float(pooled.mean()), "ci95_low_descriptive": float(low), "ci95_high_descriptive": float(high),
                        "seed_estimates": json.dumps({str(s): float(per_seed[s].mean()) for s in SEEDS}, sort_keys=True),
                        "seed_sd": float(np.std([per_seed[s].mean() for s in SEEDS], ddof=1)),
                        "queries": 3718, "source_groups": len(ordered_groups),
                        "bootstrap_replicates": BOOTSTRAPS,
                        "interpretation": "development-only objective sensitivity",
                        "final_test_access": 0,
                    })
    write_csv(OUT / "D3_EQ_DENSE_CONTRASTS_BY_SEED.csv", contrast_rows)
    write_csv(OUT / "D3_EQ_DENSE_OBJECTIVE_SENSITIVITY.csv", objective_rows)
    qa = {
        "status": "PASS",
        "old_dense_prediction_slots": len(old_predictions),
        "new_dense_prediction_slots": len(new_predictions),
        "queries": len(dense_ids),
        "source_groups": len(ordered_groups),
        "dev_target_source": "118 primary-dev raw Parquet episodes; used post-training for evaluation only",
        "grouping": "video_sha256_fingerprint; one episode per source group in this panel",
        "same_dense_ids_old_and_new": True,
        "final_test_access": 0,
    }
    atomic_json(OUT / "D3_EQ_DENSE_ANALYSIS_QA.json", qa)
    print(json.dumps(qa, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
